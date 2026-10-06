from typing import Literal

import pytest
from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.exceptions import BadRequestError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms import load_guardrail_translation_mappings
from litellm.llms.strands_decider.decisions.guardrail_translation.handler import DecisionsHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.types.decisions import DecisionsResult
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


class MaskGuard(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        return {**inputs, "texts": [text.replace("secret", "masked") for text in inputs.get("texts", [])]}


class BlockGuard(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise BadRequestError("Blocked by test guardrail", model="xor", llm_provider="")


class EmptyGuard(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        return {"texts": []}


def request() -> dict[str, object]:
    return {
        "model": "xor",
        "state": {"text": "secret state"},
        "context": "secret context",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "secret instruction",
                "criteria": {"yes": "secret criterion", "no": "Other"},
            }
        },
        "metadata": {"api_key": "must-not-be-scanned"},
    }


def test_legacy_message_guardrails_inspect_all_decisions_text() -> None:
    messages = CustomGuardrail().get_guardrails_messages_for_call_type(CallTypes.adecisions, request())
    assert messages is not None
    content = messages[0]["content"]
    assert isinstance(content, str)
    assert all(text in content for text in ("secret state", "secret context", "secret instruction", "secret criterion"))
    assert "must-not-be-scanned" not in content


@pytest.mark.asyncio
async def test_unified_guardrail_masks_decisions_instead_of_skipping() -> None:
    assert load_guardrail_translation_mappings()[CallTypes.adecisions] is DecisionsHandler
    data = {
        **request(),
        "guardrail_to_apply": MaskGuard(
            guardrail_name="mask",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        ),
    }
    guarded = await UnifiedLLMGuardrails().async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="adecisions",
    )
    assert isinstance(guarded, dict)
    assert guarded["state"] == {"text": "masked state"}
    assert guarded["context"] == "masked context"
    assert guarded["questions"]["q"]["instructions"] == "masked instruction"
    assert guarded["questions"]["q"]["criteria"]["yes"] == "masked criterion"
    assert guarded["metadata"]["api_key"] == "must-not-be-scanned"


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", (BlockGuard(), EmptyGuard()))
async def test_decisions_guardrail_failures_do_not_forward_unscanned_text(guard: CustomGuardrail) -> None:
    with pytest.raises(BadRequestError):
        await DecisionsHandler().process_input_messages(request(), guard)


@pytest.mark.asyncio
async def test_guarded_audio_is_rejected_explicitly() -> None:
    data = {**request(), "audio": {"data": "UklGRg==", "format": "wav"}}
    with pytest.raises(BadRequestError, match="audio cannot be inspected"):
        await DecisionsHandler().process_input_messages(data, MaskGuard())
    with pytest.raises(BadRequestError, match="audio cannot be inspected"):
        CustomGuardrail().get_guardrails_messages_for_call_type(CallTypes.adecisions, data)


@pytest.mark.asyncio
@pytest.mark.parametrize("extraction", (False, True))
async def test_output_masking_preserves_usage_and_hidden_billing_data(extraction: bool) -> None:
    wire = (
        {
            "result": {"author": "secret"},
            "usage": {"input_tokens": 18, "completion_tokens": 7},
            "confidence": {},
            "thinking": {},
        }
        if extraction
        else {"answers": {"q": {"type": "value", "value": "secret"}}, "usage": {"input_tokens": 18, "output_tokens": 7}}
    )
    response = TypeAdapter(DecisionsResult).validate_python(wire)
    response._hidden_params.update({"response_cost": 0.000032, "model_id": "priced-deployment"})
    guarded = await UnifiedLLMGuardrails().async_post_call_success_hook(
        data={
            "guardrail_to_apply": MaskGuard(
                guardrail_name="mask",
                event_hook=GuardrailEventHooks.post_call,
                default_on=True,
            )
        },
        user_api_key_dict=UserAPIKeyAuth(),
        response=response,
    )
    assert "secret" not in guarded.model_dump_json()
    assert "masked" in guarded.model_dump_json()
    assert guarded.usage == response.usage
    assert guarded._hidden_params == response._hidden_params
    assert "secret" in response.model_dump_json()
