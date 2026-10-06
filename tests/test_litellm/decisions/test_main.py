from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
import respx
import httpx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.decisions import (
    ChoiceAnswer,
    DecisionsResponse,
    DecisionsUsage,
    ExtractionResponse,
    NoulAnswer,
    ScoreAnswer,
)

_QUESTIONS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "is_defect": {"type": "noul", "instructions": "Is this a defect?", "provider_field": "kept"},
        "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
        "severity": {"type": "score", "criteria": ["none", "low", "high"]},
    }
)
_EXTRACTION_RESPONSE: Final[Mapping[str, object]] = {
    "result": {"author": "Example Author"},
    "usage": {"input_tokens": 18, "thinking_tokens": 0, "completion_tokens": 7, "requests": 1, "wall_s": 0.09},
    "thinking": {},
    "confidence": {"author": {"mean_p": 1.0, "min_p": 0.9999}},
}


@pytest.mark.parametrize("audio", (None, {"data": "UklGRg==", "format": "wav"}))
def test_extraction_preserves_wire_contract_and_cost(
    audio: Mapping[str, str] | None, respx_mock: respx.MockRouter
) -> None:
    upstream = respx_mock.post("https://xor.example/v1/systemone").respond(json=_EXTRACTION_RESPONSE)
    response = litellm.decisions(
        model="strands_decider/jev-trained",
        api_base="https://xor.example",
        context="This paper was written by Example Author.",
        questions={"author": {"type": "string", "instructions": "Who is the author?"}},
        audio=audio,
        input_cost_per_token=0.000001,
        output_cost_per_token=0.000002,
    )
    assert isinstance(response, ExtractionResponse)
    assert response.model_dump() == _EXTRACTION_RESPONSE
    payload = json.loads(upstream.calls[0].request.content)
    assert payload["context"] == "This paper was written by Example Author."
    assert "state" not in payload
    assert ("audio" in payload) == (audio is not None)
    if audio is not None:
        assert payload["audio"] == audio
    assert litellm.completion_cost(
        completion_response=response,
        custom_cost_per_token={"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002},
    ) == pytest.approx(0.000032)
    from litellm.cost_calculator import _get_usage_object

    usage = _get_usage_object(response)
    assert usage.prompt_tokens == 18
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 25


@pytest.mark.asyncio
async def test_audio_value_extraction_router(respx_mock: respx.MockRouter) -> None:
    router = litellm.Router(
        model_list=[
            {
                "model_name": "xor",
                "litellm_params": {"model": "strands_decider/jev-trained", "api_base": "https://xor.example"},
            }
        ]
    )
    wire = {
        "answers": {"pickup": {"type": "value", "value": "Airport", "confidence": 0.9}},
        "usage": {"input_tokens": 18, "output_tokens": 7},
    }
    upstream = respx_mock.post("https://xor.example/v1/systemone").respond(json=wire)
    audio = {"data": "UklGRg==", "format": "wav"}
    questions = {"pickup": {"type": "value", "instructions": "Extract pickup", "pattern": '[^"]{1,100}'}}
    response = await router.adecisions(model="xor", state={"current_draft": {}}, questions=questions, audio=audio)
    assert response.answers["pickup"].value == "Airport"
    assert json.loads(upstream.calls[0].request.content) == {
        "model": "jev-trained",
        "state": {"current_draft": {}},
        "questions": questions,
        "audio": audio,
    }


@pytest.mark.parametrize(
    "audio", ({"format": "wav"}, {"data": "", "format": "wav"}, {"data": 3, "format": "wav"}, "bad")
)
def test_invalid_audio_rejected_before_http(audio: object, respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        litellm.decisions(
            model="strands_decider/jev-trained",
            api_base="https://xor.example",
            state="hello",
            questions={"q": {"type": "noul", "instructions": "Is it a greeting?"}},
            audio=audio,
        )
    assert not respx_mock.calls


def test_missing_grounding_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="requires state, context or audio"):
        litellm.decisions(
            model="strands_decider/jev-trained",
            api_base="https://xor.example",
            questions={"q": {"type": "string", "instructions": "Who?"}},
        )
    assert not respx_mock.calls


def test_malformed_extraction_does_not_fall_back_to_classification() -> None:
    from pydantic import TypeAdapter, ValidationError

    from litellm.types.decisions import DecisionsResult

    with pytest.raises(ValidationError):
        TypeAdapter(DecisionsResult).validate_python(
            {
                "result": {"author": "Example Author"},
                "answers": {},
                "usage": {"input_tokens": 18, "completion_tokens": 7},
            }
        )


_INPUT_TOKENS: Final[int] = 367
_OUTPUT_TOKENS: Final[int] = 3
_RESPONSE: Final[Mapping[str, object]] = {
    "model": "jev-1.13",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
        "sentiment": {
            "type": "choice",
            "choice": "positive",
            "confidence": 0.8,
            "probabilities": {"positive": 0.8, "negative": 0.2},
        },
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.7,
            "legend": {"0": "none", "1": "low", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        },
    },
    "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": _OUTPUT_TOKENS},
}
_STRANDS_RESPONSE: Final[Mapping[str, object]] = {
    "model": "strands-decider-2B-hobson-v19",
    "answers": {
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.7,
            "legend": {"0": "none", "1": "low", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        }
    },
    "usage": {"input_tokens": 216, "output_tokens": 3},
    "latency_ms": 3722.17,
}
_PROVIDERS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "perplexity",
        "perplexity/pplx-decider-v1-27b",
        "https://api.perplexity.ai/v1/decisions",
        "pplx-decider-v1-27b",
    ),
    ("typesafe", "typesafe/jev-1.13", "https://api.typesafe.ai/v1/systemone", "jev-1.13"),
    (
        "openrouter",
        "openrouter/typesafe/jev-1.13",
        "https://openrouter.ai/api/alpha/decisions",
        "typesafe/jev-1.13",
    ),
)


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_object: Mapping[str, object] | None = None

    async def async_log_success_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            self.standard_logging_object = standard_logging_object


async def _drain_logging_worker() -> None:
    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "model", "url", "upstream_model"), _PROVIDERS)
async def test_adecisions_sends_the_provider_wire_contract(
    provider: str,
    model: str,
    url: str,
    upstream_model: str,
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(url).respond(json=_RESPONSE)

    response: Final = await litellm.adecisions(
        model=model,
        state={"source": "unit-test"},
        questions=_QUESTIONS,
        api_key="caller-key",
        extra_headers={
            "x-request-tag": "decisions-test",
            "AUTHORIZATION": "attacker-key",
            "Content-Type": "text/plain",
        },
        internal_kwarg="must-not-leak",
    )

    assert route.called
    assert len(respx_mock.calls) == 1
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer caller-key"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["x-request-tag"] == "decisions-test"
    assert json.loads(request.content) == {
        "model": upstream_model,
        "state": {"source": "unit-test"},
        "questions": {
            "is_defect": {
                "type": "noul",
                "instructions": "Is this a defect?",
                "provider_field": "kept",
            },
            "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": "unhappy"}},
            "severity": {"type": "score", "criteria": ["none", "low", "high"]},
        },
    }
    assert isinstance(response.answers["is_defect"], NoulAnswer)
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert isinstance(response.answers["severity"], ScoreAnswer)
    assert response._hidden_params["custom_llm_provider"] == provider


@pytest.mark.asyncio
async def test_router_dispatches_typesafe_decisions_without_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_BASE", raising=False)
    provider_resolution: Final = litellm.get_llm_provider("typesafe/jev-latest")

    assert provider_resolution[:2] == ("jev-latest", "typesafe")

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "jev",
                "litellm_params": {
                    "model": "typesafe/jev-latest",
                    "api_key": "k",
                },
            }
        ]
    )
    upstream: Final = respx_mock.post("https://api.typesafe.ai/v1/systemone").respond(json=_RESPONSE)

    response: Final = await router.adecisions(
        model="jev",
        state="router-test",
        questions={
            "sentiment": {
                "type": "choice",
                "criteria": {"positive": None, "negative": "unhappy"},
            }
        },
    )

    assert upstream.called
    assert len(respx_mock.calls) == 1
    assert json.loads(respx_mock.calls[0].request.content) == {
        "model": "jev-latest",
        "state": "router-test",
        "questions": {
            "sentiment": {
                "type": "choice",
                "criteria": {"positive": None, "negative": "unhappy"},
            }
        },
    }
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer k"
    assert isinstance(response.answers["sentiment"], ChoiceAnswer)
    assert response.answers["sentiment"].choice == "positive"


def test_decisions_uses_the_same_wire_contract_for_sync_calls(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)

    response: Final = litellm.decisions(
        model="perplexity/pplx-decider-v1-27b",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_key="caller-key",
    )

    assert route.called
    assert response.model == "jev-1.13"


@pytest.mark.parametrize("images", (None, (), ("https://example.org/image.png", "data:image/png;base64,aGVsbG8=")))
def test_decisions_forwards_images_without_gateway_kwargs(
    images: tuple[str, ...] | None,
    respx_mock: respx.MockRouter,
) -> None:
    upstream = respx_mock.post("https://xor.example/v1/systemone").respond(json=_RESPONSE)
    litellm.decisions(
        model="strands_decider/jev-trained",
        api_base="https://xor.example/v1",
        state="Look at these images",
        questions={"color": {"type": "choice", "criteria": {"red": "red", "blue": "blue"}}},
        images=images,
        metadata={"tags": ["internal"]},
    )
    payload = json.loads(upstream.calls[0].request.content)
    assert payload["model"] == "jev-trained"
    assert "metadata" not in payload
    assert ("images" in payload) == (images is not None)
    if images is not None:
        assert payload["images"] == list(images)


@pytest.mark.asyncio
async def test_router_forwards_images_to_self_hosted_xor(respx_mock: respx.MockRouter) -> None:
    router = litellm.Router(
        model_list=[
            {
                "model_name": "xor",
                "litellm_params": {"model": "strands_decider/jev-trained", "api_base": "https://xor.example"},
            }
        ]
    )
    upstream = respx_mock.post("https://xor.example/v1/systemone").respond(json=_RESPONSE)
    images = ["data:image/png;base64,aGVsbG8="]
    response = await router.adecisions(
        model="xor",
        state="Look at the attached image",
        questions={"color": {"type": "noul", "instructions": "Is it red?"}},
        images=images,
    )
    assert response.answers["is_defect"].noul == 0.9
    assert json.loads(upstream.calls[0].request.content)["images"] == images


@pytest.mark.asyncio
@pytest.mark.parametrize("images", ("not-a-list", [1], {"url": "https://example.org/image.png"}))
async def test_invalid_images_are_rejected_before_http(images: object, respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.adecisions(
            model="strands_decider/jev-trained",
            api_base="https://xor.example",
            state="image",
            questions={"q": {"type": "noul", "instructions": "Is it red?"}},
            images=images,
        )
    assert not respx_mock.calls


def test_openrouter_response_keeps_provider_fields(respx_mock: respx.MockRouter) -> None:
    payload: Final = {
        **_RESPONSE,
        "id": "decision-1",
        "provider": "typesafe",
        "usage": {**_RESPONSE["usage"], "cost": 0.25},
    }
    respx_mock.post("https://openrouter.ai/api/alpha/decisions").respond(json=payload)

    response: Final = litellm.decisions(
        model="openrouter/typesafe/jev-1.13",
        state="review",
        questions=_QUESTIONS,
        api_key="caller-key",
    )

    assert response.model_extra["id"] == "decision-1"
    assert response.model_extra["provider"] == "typesafe"
    assert response.usage is not None
    assert response.usage.model_extra["cost"] == 0.25


def test_decisions_cost_uses_litellm_token_pricing() -> None:
    response: Final = DecisionsResponse(
        model="pplx-decider-v1-27b",
        answers={},
        usage=DecisionsUsage(input_tokens=_INPUT_TOKENS, output_tokens=_OUTPUT_TOKENS),
    )
    response._hidden_params = {
        "model": "perplexity/pplx-decider-v1-27b",
        "custom_llm_provider": "perplexity",
    }

    cost: Final = litellm.completion_cost(completion_response=response)
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_extraction_standard_logging_keeps_usage_and_original_response(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://xor.example/v1/systemone").respond(json=_EXTRACTION_RESPONSE)
    logger = _RecordingLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [logger]
    router = litellm.Router(
        model_list=[
            {
                "model_name": "xor-extraction-priced",
                "litellm_params": {
                    "model": "strands_decider/jev-trained",
                    "api_base": "https://xor.example",
                    "input_cost_per_token": 0.000001,
                    "output_cost_per_token": 0.000002,
                },
            }
        ]
    )
    try:
        response = await router.adecisions(
            model="xor-extraction-priced",
            context="A paper by Example Author",
            questions={"author": {"type": "string", "instructions": "Who wrote it?"}},
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
        )
        await _drain_logging_worker()
    finally:
        litellm.callbacks = original_callbacks
    assert response.model_dump() == _EXTRACTION_RESPONSE
    assert logger.standard_logging_object is not None
    assert logger.standard_logging_object["prompt_tokens"] == 18
    assert logger.standard_logging_object["completion_tokens"] == 7
    assert logger.standard_logging_object["response_cost"] == pytest.approx(0.000032)


@pytest.mark.asyncio
async def test_decisions_cost_is_in_standard_logging_object(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(json=_RESPONSE)
    recording_logger: Final = _RecordingLogger()
    original_callbacks: Final = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )
        await _drain_logging_worker()
    finally:
        litellm.callbacks = original_callbacks

    assert recording_logger.standard_logging_object is not None
    perplexity_cost: Final = litellm.model_cost["perplexity/pplx-decider-v1-27b"]
    expected_cost: Final = _INPUT_TOKENS * float(perplexity_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        perplexity_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert recording_logger.standard_logging_object["response_cost"] == pytest.approx(expected_cost)
    assert recording_logger.standard_logging_object["prompt_tokens"] == _INPUT_TOKENS
    assert recording_logger.standard_logging_object["completion_tokens"] == _OUTPUT_TOKENS


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Supported providers"):
        await litellm.adecisions(
            model="unknown/jev-1.13",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_empty_custom_provider_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Supported providers"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
            custom_llm_provider="",
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_invalid_question_is_rejected_before_http(respx_mock: respx.MockRouter) -> None:
    with pytest.raises(litellm.BadRequestError, match="Invalid Decisions request"):
        await litellm.adecisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"sentiment": {"type": "choice"}},
            api_key="caller-key",
        )

    assert len(respx_mock.calls) == 0


def test_upstream_bad_request_maps_to_litellm_error(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.perplexity.ai/v1/decisions").respond(
        status_code=400,
        json={"error": {"message": "invalid decision"}},
    )

    with pytest.raises(litellm.BadRequestError):
        litellm.decisions(
            model="perplexity/pplx-decider-v1-27b",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
            api_key="caller-key",
        )


@pytest.mark.parametrize("provider", ("strands_decider", "typesafe", "cloudflare", "perplexity", "openrouter"))
@pytest.mark.parametrize("status", (400, 401, 403, 404, 422, 429, 500, 503))
@pytest.mark.asyncio
async def test_decisions_preserves_upstream_status_and_retry_headers(
    provider: str, status: int, respx_mock: respx.MockRouter
) -> None:
    upstream = respx_mock.route().mock(
        return_value=httpx.Response(status, json={"error": "fixture"}, headers={"retry-after": "3"})
    )
    with pytest.raises(Exception) as caught:
        await litellm.adecisions(
            model=f"{provider}/fixture",
            api_base="https://fixture.example",
            api_key="fixture",
            state="review",
            questions={"q": {"type": "noul", "instructions": "Is this a defect?"}},
        )
    assert caught.value.status_code == status
    assert upstream.call_count == 1
    if status == 429:
        assert isinstance(caught.value, litellm.RateLimitError)
        assert caught.value.response.headers["retry-after"] == "3"


@pytest.mark.parametrize("path", ("/v1/systemone", "/v1/generate"))
def test_strands_decider_accepts_full_endpoint(path: str, respx_mock: respx.MockRouter) -> None:
    upstream = respx_mock.post("https://xor.example" + path).respond(json=_EXTRACTION_RESPONSE)
    response = litellm.decisions(
        model="strands_decider/jev-trained",
        api_base="https://xor.example" + path,
        context="A paper by Example Author",
        questions={"author": {"type": "string", "instructions": "Who wrote it?"}},
    )
    assert response.model_dump() == _EXTRACTION_RESPONSE
    assert upstream.call_count == 1


@pytest.mark.asyncio
async def test_router_does_not_retry_upstream_bad_requests(respx_mock: respx.MockRouter) -> None:
    upstream = respx_mock.post("https://xor.example/v1/systemone").respond(status_code=400, json={"error": "fixture"})
    router = litellm.Router(
        num_retries=2,
        model_list=[
            {
                "model_name": "xor",
                "litellm_params": {"model": "strands_decider/jev-trained", "api_base": "https://xor.example"},
            }
        ],
    )
    with pytest.raises(litellm.BadRequestError):
        await router.adecisions(
            model="xor", state="review", questions={"q": {"type": "noul", "instructions": "Review?"}}
        )
    assert upstream.call_count == 1


def test_server_key_is_sent_to_an_explicit_api_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("PERPLEXITYAI_API_KEY", "server-key")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    route: Final = respx_mock.post("https://egress.example/perplexity/v1/decisions").respond(json=_RESPONSE)

    litellm.decisions(
        model="perplexity/pplx-decider-v1-27b",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        api_base="https://egress.example/perplexity",
    )

    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer server-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("cloudflare/clef", "cloudflare/@cf/cloudflare/clef"))
@pytest.mark.parametrize("wrapped", (False, True))
async def test_cloudflare_clef_resolves_model_and_response_envelope(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    model: str,
    wrapped: bool,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    response_body: Final[Mapping[str, object]] = (
        {"result": _RESPONSE, "success": True, "errors": [], "messages": []} if wrapped else _RESPONSE
    )
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    ).respond(json=response_body)

    response: Final = await litellm.adecisions(
        model=model,
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    request: Final = respx_mock.calls[0].request
    assert request.headers["authorization"] == "Bearer cloudflare-key"
    assert json.loads(request.content) == {
        "model": "clef",
        "state": "review",
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    }
    assert response.answers == DecisionsResponse.model_validate(_RESPONSE).answers
    assert response._hidden_params["model"] == "cloudflare/@cf/cloudflare/clef"


@pytest.mark.asyncio
async def test_cloudflare_clef_flash_uses_flash_endpoint_and_request_model(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    ).respond(json=_RESPONSE)

    await litellm.adecisions(
        model="cloudflare/clef-flash",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called
    assert json.loads(respx_mock.calls[0].request.content)["model"] == "clef-flash"


@pytest.mark.asyncio
async def test_cloudflare_api_base_from_env_uses_workers_ai_run_path(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_BASE", "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    route: Final = respx_mock.post(
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef"
    ).respond(json=_RESPONSE)

    await litellm.adecisions(
        model="cloudflare/clef",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    assert route.called


@pytest.mark.asyncio
async def test_cloudflare_requires_account_id_or_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")

    with pytest.raises(litellm.BadRequestError, match="Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID"):
        await litellm.adecisions(
            model="cloudflare/clef",
            state="review",
            questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_cloudflare_clef_cost_uses_the_model_cost_map(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cloudflare-key")
    monkeypatch.delenv("CLOUDFLARE_API_BASE", raising=False)
    respx_mock.post("https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef").respond(
        json=_RESPONSE
    )

    response: Final = await litellm.adecisions(
        model="cloudflare/clef",
        state="review",
        questions={"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    )

    cost: Final = litellm.completion_cost(completion_response=response)
    clef_cost: Final = litellm.model_cost["cloudflare/@cf/cloudflare/clef"]
    expected_cost: Final = _INPUT_TOKENS * float(clef_cost["input_cost_per_token"]) + _OUTPUT_TOKENS * float(
        clef_cost["output_cost_per_token"]
    )

    assert expected_cost > 0
    assert cost == pytest.approx(expected_cost)


@pytest.mark.asyncio
async def test_strands_decider_requires_api_base_before_http(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)

    with pytest.raises(litellm.BadRequestError, match="api_base is required"):
        await litellm.adecisions(
            model="strands_decider/strands-decider-2B-hobson-v19",
            state="review",
            questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        )

    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
async def test_strands_decider_without_key_preserves_response_extras(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = await litellm.adecisions(
        model="strands_decider/strands-decider-2B-hobson-v19",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        api_base="https://strands.example",
    )

    assert route.called
    assert "authorization" not in respx_mock.calls[0].request.headers
    assert response.model_extra["latency_ms"] == _STRANDS_RESPONSE["latency_ms"]
    severity: Final = response.answers["severity"]
    assert isinstance(severity, ScoreAnswer)
    assert severity.legend == {"0": "none", "1": "low", "2": "high"}


@pytest.mark.asyncio
async def test_strands_decider_uses_key_from_matching_environment_base(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.setenv("STRANDS_DECIDER_API_BASE", "https://strands.example")
    monkeypatch.setenv("STRANDS_DECIDER_API_KEY", "strands-key")
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    await litellm.adecisions(
        model="strands_decider/strands-decider-2B-hobson-v19",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
        api_base="https://strands.example",
    )

    assert route.called
    assert respx_mock.calls[0].request.headers["authorization"] == "Bearer strands-key"


@pytest.mark.asyncio
async def test_strands_decider_provider_resolution_and_router_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
) -> None:
    monkeypatch.delenv("STRANDS_DECIDER_API_BASE", raising=False)
    monkeypatch.delenv("STRANDS_DECIDER_API_KEY", raising=False)
    provider_resolution: Final = litellm.get_llm_provider("strands_decider/strands-decider-2B-hobson-v19")
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "strands",
                "litellm_params": {
                    "model": "strands_decider/strands-decider-2B-hobson-v19",
                    "api_base": "https://strands.example",
                },
            }
        ]
    )
    route: Final = respx_mock.post("https://strands.example/v1/systemone").respond(json=_STRANDS_RESPONSE)

    response: Final = await router.adecisions(
        model="strands",
        state="review",
        questions={"severity": {"type": "score", "criteria": ["none", "low", "high"]}},
    )

    assert provider_resolution[:2] == ("strands-decider-2B-hobson-v19", "strands_decider")
    assert route.called
    assert response.model == _STRANDS_RESPONSE["model"]
