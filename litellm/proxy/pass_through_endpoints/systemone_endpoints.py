from collections.abc import Mapping
from typing import Annotated, Protocol

import httpx
import litellm
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
    InitPassThroughEndpointHelpers,
)

router = APIRouter()


class SystemOneRequestEnvelope(BaseModel):
    model: str = Field(min_length=1)

    model_config = ConfigDict(extra="allow")


class SystemOneDeploymentParams(BaseModel):
    api_base: str = Field(min_length=1)
    api_key: str | None = None
    extra_headers: Mapping[str, str] = Field(default_factory=dict)
    systemone_path: str = "/v1/systemone"

    model_config = ConfigDict(extra="ignore")


class SystemOneDeployment(BaseModel):
    litellm_params: SystemOneDeploymentParams

    model_config = ConfigDict(extra="ignore")


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


class SystemOneDeploymentRouter(Protocol):
    def get_available_deployment_for_pass_through(self, model: str) -> object: ...


class SystemOneForwarder(Protocol):
    async def __call__(
        self,
        *,
        request: Request,
        target: str,
        custom_headers: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
        custom_body: dict[str, object],
        cost_per_request: float,
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


async def _default_forwarder(
    *,
    request: Request,
    target: str,
    custom_headers: dict[str, str],
    user_api_key_dict: UserAPIKeyAuth,
    custom_body: dict[str, object],
    cost_per_request: float,
) -> Response:
    response = await pass_through_endpoints.pass_through_request(
        request=request,
        target=target,
        custom_headers=custom_headers,
        user_api_key_dict=user_api_key_dict,
        custom_body=custom_body,
        cost_per_request=cost_per_request,
    )
    return response


def _deployment_params(deployment: object) -> SystemOneDeploymentParams:
    try:
        raw_deployment = SystemOneDeployment.model_validate(deployment)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Selected SystemOne deployment is missing a valid api_base",
        ) from exc
    return raw_deployment.litellm_params


def build_systemone_target_url(params: SystemOneDeploymentParams) -> str:
    base_url = httpx.URL(params.api_base)
    configured_path = params.systemone_path if params.systemone_path.startswith("/") else f"/{params.systemone_path}"
    if base_url.path.rstrip("/").endswith(configured_path.rstrip("/")):
        return str(base_url)
    return str(
        base_url.copy_with(path=HttpPassThroughEndpointHelpers.join_base_and_endpoint_path(base_url, configured_path))
    )


def _upstream_headers(params: SystemOneDeploymentParams) -> dict[str, str]:
    configured_headers = dict(params.extra_headers)
    if params.api_key is None or any(key.lower() == "authorization" for key in configured_headers):
        return {"Content-Type": "application/json", **configured_headers}
    return {"Content-Type": "application/json", "Authorization": f"Bearer {params.api_key}", **configured_headers}


async def _forward_registered_systemone_passthrough(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    request_body: dict[str, object],
) -> Response | None:
    registered_route = InitPassThroughEndpointHelpers.get_registered_pass_through_route(
        route="/v1/systemone",
        method=request.method,
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

    response = await pass_through_endpoints.pass_through_request(
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
    return response


async def forward_systemone_request(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    deployment_router: SystemOneDeploymentRouter,
    forwarder: SystemOneForwarder = _default_forwarder,
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
        deployment = deployment_router.get_available_deployment_for_pass_through(model=envelope.model)
    except litellm.BadRequestError:
        legacy_response = await legacy_forwarder(
            request=request,
            user_api_key_dict=user_api_key_dict,
            request_body=request_body,
        )
        if legacy_response is not None:
            return legacy_response
        raise
    params = _deployment_params(deployment)
    return await forwarder(
        request=request,
        target=build_systemone_target_url(params),
        custom_headers=_upstream_headers(params),
        user_api_key_dict=user_api_key_dict,
        custom_body=request_body,
        cost_per_request=0.0,
    )


@router.post("/v1/systemone", tags=["SystemOne"])
async def systemone_route(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model router is configured",
        )
    return await forward_systemone_request(
        request=request,
        user_api_key_dict=user_api_key_dict,
        deployment_router=llm_router,
    )
