from collections.abc import Mapping
from typing import Annotated, Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.llms.vllm.passthrough.transformation import VLLMPassthroughConfig
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    InitPassThroughEndpointHelpers,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CostResponseTypes, ModelResponse

router = APIRouter()


class SystemOneRequestEnvelope(BaseModel):
    model: str = Field(min_length=1)

    model_config = ConfigDict(extra="allow")


class SystemOneUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class SystemOneResponseEnvelope(BaseModel):
    model: str | None = None
    usage: SystemOneUsage | None = None

    model_config = ConfigDict(extra="allow")


class RegisteredPassThroughParams(BaseModel):
    target: str = Field(min_length=1)
    custom_headers: Mapping[str, str] | None = None
    forward_headers: bool | None = False
    merge_query_params: bool | None = False
    default_query_params: dict[str, object] | None = None
    cost_per_request: float | None = None
    guardrails: dict[str, object] | None = None
    timeout: float | None = None

    model_config = ConfigDict(extra="ignore")


class RegisteredPassThroughRoute(BaseModel):
    passthrough_params: RegisteredPassThroughParams

    model_config = ConfigDict(extra="ignore")


class SystemOneModelForwarder(Protocol):
    async def __call__(
        self,
        *,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        request_body: dict[str, object],
        model: str,
    ) -> Response: ...


class LegacySystemOneForwarder(Protocol):
    async def __call__(
        self,
        *,
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        request_body: dict[str, object],
    ) -> Response | None: ...


_request_body_adapter = TypeAdapter(dict[str, object])


def build_systemone_target_url(api_base: str, systemone_path: str = "/v1/systemone") -> str:
    base_url = httpx.URL(api_base)
    configured_path = systemone_path if systemone_path.startswith("/") else f"/{systemone_path}"
    base_path = base_url.path.rstrip("/")
    if base_path.endswith(configured_path.rstrip("/")):
        return str(base_url)
    endpoint_path = (
        "/systemone" if configured_path == "/v1/systemone" and base_path.endswith("/v1") else configured_path
    )
    joined_path = HttpPassThroughEndpointHelpers.join_base_and_endpoint_path(base_url, endpoint_path)
    return str(base_url.copy_with(path=joined_path))


class SystemOnePassthroughConfig(VLLMPassthroughConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key

    def is_streaming_request(self, endpoint: str, request_data: dict[str, object]) -> bool:
        return False

    def should_replace_model_in_request(self) -> bool:
        return False

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict[str, object] | None,
        litellm_params: dict[str, object],
    ) -> tuple[httpx.URL, str]:
        if api_base is None:
            raise ValueError("SystemOne api_base is required")
        configured_path = litellm_params.get("systemone_path")
        systemone_path = configured_path if isinstance(configured_path, str) else "/v1/systemone"
        target = build_systemone_target_url(api_base=api_base, systemone_path=systemone_path)
        return httpx.URL(target, params=request_query_params), api_base

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        configured_headers = litellm_params.get("extra_headers")
        extra_headers = (
            {str(key): str(value) for key, value in configured_headers.items()}
            if isinstance(configured_headers, Mapping)
            else {}
        )
        authorization = {} if api_key is None else {"Authorization": f"Bearer {api_key}"}
        return {"Content-Type": "application/json", **authorization, **extra_headers}

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: httpx.Response,
        request_data: dict[str, object],
        logging_obj: object,
        endpoint: str,
    ) -> CostResponseTypes | None:
        try:
            response = SystemOneResponseEnvelope.model_validate(httpx_response.json())
        except (ValueError, ValidationError):
            return None
        if response.usage is None:
            return None
        return ModelResponse(
            model=response.model or model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )


SYSTEMONE_PASSTHROUGH_CONFIG = SystemOnePassthroughConfig()


async def _forward_registered_systemone_passthrough(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, object],
) -> Response | None:
    registered_route = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
        route="/v1/systemone", method=request.method
    )
    if registered_route is None:
        route_for_other_method = InitPassThroughEndpointHelpers.get_registered_pass_through_route(route="/v1/systemone")
        if route_for_other_method is not None:
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail=f"Method {request.method} is not allowed for pass-through endpoint /v1/systemone.",
            )
        return None

    try:
        params = RegisteredPassThroughRoute.model_validate(registered_route).passthrough_params
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registered SystemOne pass-through endpoint is invalid",
        ) from exc

    return await pass_through_endpoints.pass_through_request(
        request=request,
        target=params.target,
        custom_headers=dict(params.custom_headers or {}),
        user_api_key_dict=user_api_key_dict,
        custom_body=request_body,
        forward_headers=params.forward_headers,
        merge_query_params=params.merge_query_params,
        query_params=dict(request.query_params),
        default_query_params=params.default_query_params,
        cost_per_request=params.cost_per_request,
        guardrails_config=params.guardrails,
        timeout=params.timeout,
    )


async def _forward_model_systemone_request(
    *,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, object],
    model: str,
) -> Response:
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    if llm_router is None:
        raise litellm.BadRequestError(message="No model router is configured", model=model, llm_provider="")

    data = {
        **request_body,
        "model": model,
        "method": request.method,
        "endpoint": "/v1/systemone",
        "json": dict(request_body),
        "provider_config": SYSTEMONE_PASSTHROUGH_CONFIG,
        "use_pass_through_deployments": True,
        "passthrough_on_no_deployment": False,
    }
    processor = ProxyBaseLLMRequestProcessing(data=data)
    return await processor.base_passthrough_process_llm_request(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
        general_settings=general_settings,
        proxy_config=proxy_config,
        select_data_generator=select_data_generator,
        llm_router=llm_router,
        model=model,
        user_model=user_model,
        user_temperature=user_temperature,
        user_request_timeout=user_request_timeout,
        user_max_tokens=user_max_tokens,
        user_api_base=user_api_base,
        version=version,
    )


async def forward_systemone_request(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    model_forwarder: SystemOneModelForwarder = _forward_model_systemone_request,
    legacy_forwarder: LegacySystemOneForwarder = _forward_registered_systemone_passthrough,
) -> Response:
    try:
        request_body = _request_body_adapter.validate_json(await request.body())
        envelope = SystemOneRequestEnvelope.model_validate(request_body)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SystemOne requests require a JSON object with a non-empty model",
        ) from exc

    try:
        return await model_forwarder(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            request_body=request_body,
            model=envelope.model,
        )
    except litellm.BadRequestError:
        legacy_response = await legacy_forwarder(
            request=request, user_api_key_dict=user_api_key_dict, request_body=request_body
        )
        if legacy_response is not None:
            return legacy_response
        raise


@router.post("/v1/systemone", tags=["SystemOne"])
async def systemone_route(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    return await forward_systemone_request(
        request=request, fastapi_response=fastapi_response, user_api_key_dict=user_api_key_dict
    )
