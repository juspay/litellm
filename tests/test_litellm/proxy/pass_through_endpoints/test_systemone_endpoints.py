import json

import httpx
import litellm
import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.pass_through_endpoints.systemone_endpoints import (
    SystemOnePassthroughConfig,
    build_systemone_target_url,
    forward_systemone_request,
)
from litellm.router import Router


def make_request(body: object) -> Request:
    encoded_body = json.dumps(body).encode()
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": encoded_body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/systemone",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


@pytest.mark.asyncio
async def test_forwards_body_unchanged_to_model_pipeline() -> None:
    request_body = {
        "model": "decision-model",
        "state": "The customer was charged twice.",
        "images": ["data:image/png;base64,AAAA"],
        "questions": {"billing": {"type": "noul", "instructions": "Is this a billing issue?"}},
    }
    forwarded_calls: list[dict[str, object]] = []

    async def model_forwarder(**kwargs: object) -> Response:
        forwarded_calls.append(kwargs)
        return Response(content=b'{"model":"decision-model"}', media_type="application/json")

    response = await forward_systemone_request(
        request=make_request(request_body),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        model_forwarder=model_forwarder,
    )

    assert response.status_code == 200
    assert forwarded_calls[0]["model"] == "decision-model"
    assert forwarded_calls[0]["request_body"] == request_body


@pytest.mark.asyncio
async def test_uses_registered_passthrough_when_model_is_not_configured() -> None:
    request_body: dict[str, object] = {"model": "legacy-model", "state": "hello", "questions": {}}
    fallback_calls: list[dict[str, object]] = []

    async def missing_model(**kwargs: object) -> Response:
        raise litellm.BadRequestError(message="No pass-through deployment", model="legacy-model", llm_provider="")

    async def legacy_forwarder(**kwargs: object) -> Response | None:
        fallback_calls.append(kwargs)
        return Response(content=b'{"model":"legacy-model"}', media_type="application/json")

    response = await forward_systemone_request(
        request=make_request(request_body),
        fastapi_response=Response(),
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        model_forwarder=missing_model,
        legacy_forwarder=legacy_forwarder,
    )

    assert response.status_code == 200
    assert fallback_calls[0]["request_body"] == request_body


@pytest.mark.asyncio
async def test_preserves_model_error_without_registered_passthrough() -> None:
    async def missing_model(**kwargs: object) -> Response:
        raise litellm.BadRequestError(message="No pass-through deployment", model="missing", llm_provider="")

    async def no_legacy_endpoint(**kwargs: object) -> Response | None:
        return None

    with pytest.raises(litellm.BadRequestError):
        await forward_systemone_request(
            request=make_request({"model": "missing", "state": "hello", "questions": {}}),
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            model_forwarder=missing_model,
            legacy_forwarder=no_legacy_endpoint,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"model": ""}, [], "invalid"])
async def test_rejects_request_without_model(body: object) -> None:
    forwarded = False

    async def model_forwarder(**kwargs: object) -> Response:
        nonlocal forwarded
        forwarded = True
        return Response()

    with pytest.raises(HTTPException) as exc_info:
        await forward_systemone_request(
            request=make_request(body),
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            model_forwarder=model_forwarder,
        )

    assert exc_info.value.status_code == 400
    assert forwarded is False


@pytest.mark.parametrize(
    ("api_base", "systemone_path", "expected"),
    [
        ("http://inference.internal:8080", "/v1/systemone", "http://inference.internal:8080/v1/systemone"),
        ("http://inference.internal:8080/v1", "/v1/systemone", "http://inference.internal:8080/v1/systemone"),
        (
            "http://inference.internal:8080/v1/systemone",
            "/v1/systemone",
            "http://inference.internal:8080/v1/systemone",
        ),
        ("https://inference.example/base", "/decision/systemone", "https://inference.example/base/decision/systemone"),
    ],
)
def test_builds_systemone_target_url(api_base: str, systemone_path: str, expected: str) -> None:
    assert build_systemone_target_url(api_base, systemone_path) == expected


def test_systemone_config_sets_headers_and_normalizes_usage() -> None:
    config = SystemOnePassthroughConfig()
    headers = config.validate_environment(
        headers={},
        model="jev-trained",
        messages=[],
        optional_params={},
        litellm_params={"extra_headers": {"X-Deployment": "primary"}},
        api_key="upstream-key",
    )
    request = httpx.Request("POST", "http://inference.internal/v1/systemone")
    response = httpx.Response(
        200,
        request=request,
        json={"model": "jev-trained", "answers": {}, "usage": {"input_tokens": 54, "output_tokens": 1}},
    )

    normalized = config.logging_non_streaming_response(
        model="jev-trained",
        custom_llm_provider="openai",
        httpx_response=response,
        request_data={},
        logging_obj=object(),
        endpoint="/v1/systemone",
    )

    assert headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer upstream-key",
        "X-Deployment": "primary",
    }
    assert isinstance(normalized, litellm.ModelResponse)
    normalized_payload = normalized.model_dump(exclude_none=True)
    assert normalized_payload["usage"] == {
        "prompt_tokens": 54,
        "completion_tokens": 1,
        "total_tokens": 55,
    }


@pytest.mark.asyncio
async def test_router_execution_uses_only_opted_in_passthrough_deployments() -> None:
    deployment_calls: list[str] = []
    deployment_router = Router(
        model_list=[
            {
                "model_name": "decision-model",
                "litellm_params": {
                    "model": "openai/not-opted-in",
                    "api_base": "http://not-opted-in.internal",
                    "api_key": "test-key",
                    "use_in_pass_through": False,
                },
            },
            {
                "model_name": "decision-model",
                "litellm_params": {
                    "model": "openai/opted-in",
                    "api_base": "http://opted-in.internal",
                    "api_key": "test-key",
                    "use_in_pass_through": True,
                },
            },
        ]
    )

    async def generic_call(**kwargs: object) -> str:
        deployment_calls.append(str(kwargs["api_base"]))
        return "ok"

    result = await deployment_router._ageneric_api_call_with_fallbacks_helper(
        model="decision-model",
        original_generic_function=generic_call,
        use_pass_through_deployments=True,
        litellm_metadata={},
    )

    assert result == "ok"
    assert deployment_calls == ["http://opted-in.internal"]


def test_systemone_is_an_llm_api_route() -> None:
    assert "/v1/systemone" in LiteLLMRoutes.litellm_native_routes.value
    assert RouteChecks.is_llm_api_route("/v1/systemone") is True
