import json

import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

from litellm.proxy._types import LiteLLMRoutes, UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.pass_through_endpoints.systemone_endpoints import (
    SystemOneDeploymentParams,
    build_systemone_target_url,
    forward_systemone_request,
)


class StubDeploymentRouter:
    def __init__(self, deployment: object) -> None:
        self.deployment = deployment
        self.selected_models: list[str] = []

    def get_available_deployment_for_pass_through(self, model: str) -> object:
        self.selected_models.append(model)
        return self.deployment


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
async def test_forwards_body_unchanged_to_selected_model_deployment() -> None:
    request_body = {
        "model": "decision-model",
        "state": "The customer was charged twice.",
        "images": ["data:image/png;base64,AAAA"],
        "questions": {
            "billing": {
                "type": "noul",
                "instructions": "Is this a billing issue?",
            }
        },
    }
    deployment_router = StubDeploymentRouter(
        {
            "litellm_params": {
                "api_base": "http://inference.internal:8080",
                "use_in_pass_through": True,
            }
        }
    )
    forwarded_calls: list[dict[str, object]] = []

    async def forwarder(**kwargs: object) -> Response:
        forwarded_calls.append(kwargs)
        return Response(content=b'{"model":"decision-model"}', media_type="application/json")

    response = await forward_systemone_request(
        request=make_request(request_body),
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        deployment_router=deployment_router,
        forwarder=forwarder,
    )

    assert response.status_code == 200
    assert deployment_router.selected_models == ["decision-model"]
    assert len(forwarded_calls) == 1
    assert forwarded_calls[0]["target"] == "http://inference.internal:8080/v1/systemone"
    assert forwarded_calls[0]["custom_body"] == request_body
    assert forwarded_calls[0]["custom_headers"] == {"Content-Type": "application/json"}


@pytest.mark.asyncio
async def test_uses_deployment_path_headers_and_api_key() -> None:
    deployment_router = StubDeploymentRouter(
        {
            "litellm_params": {
                "api_base": "https://inference.example/base",
                "api_key": "upstream-key",
                "extra_headers": {"X-Deployment": "primary"},
                "systemone_path": "/decision/systemone",
                "use_in_pass_through": True,
            }
        }
    )
    forwarded_calls: list[dict[str, object]] = []

    async def forwarder(**kwargs: object) -> Response:
        forwarded_calls.append(kwargs)
        return Response()

    await forward_systemone_request(
        request=make_request({"model": "decision-model", "state": "hello", "questions": {}}),
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        deployment_router=deployment_router,
        forwarder=forwarder,
    )

    assert forwarded_calls[0]["target"] == "https://inference.example/base/decision/systemone"
    assert forwarded_calls[0]["custom_headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer upstream-key",
        "X-Deployment": "primary",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"model": ""}, [], "invalid"])
async def test_rejects_request_without_model(body: object) -> None:
    deployment_router = StubDeploymentRouter({"litellm_params": {"api_base": "http://inference.internal"}})

    async def forwarder(**kwargs: object) -> Response:
        return Response()

    with pytest.raises(HTTPException) as exc_info:
        await forward_systemone_request(
            request=make_request(body),
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            deployment_router=deployment_router,
            forwarder=forwarder,
        )

    assert exc_info.value.status_code == 400
    assert deployment_router.selected_models == []


def test_target_url_does_not_duplicate_systemone_path() -> None:
    params = SystemOneDeploymentParams(api_base="http://inference.internal/v1/systemone")

    assert build_systemone_target_url(params) == "http://inference.internal/v1/systemone"


def test_systemone_is_an_llm_api_route() -> None:
    assert "/v1/systemone" in LiteLLMRoutes.litellm_native_routes.value
    assert RouteChecks.is_llm_api_route("/v1/systemone") is True
