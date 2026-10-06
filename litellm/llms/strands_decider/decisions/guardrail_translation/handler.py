from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.exceptions import BadRequestError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.decisions import DecisionsRequestBody, DecisionsResult
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import GenericGuardrailAPIInputs

_REQUEST: Final = TypeAdapter(DecisionsRequestBody)
_RESULT: Final[TypeAdapter[DecisionsResult]] = TypeAdapter(DecisionsResult)


def request_text(data: Mapping[str, object]) -> str:
    request: Final = _REQUEST.validate_python(data)
    if request.audio is not None:
        raise BadRequestError(
            "Decisions audio cannot be inspected by text guardrails; use an audio-aware upstream policy",
            model=str(data.get("model", "")),
            llm_provider="",
        )
    return request.model_dump_json(include={"state", "context", "questions"}, exclude_none=True)


def decisions_guardrail_messages(data: Mapping[str, object]) -> list[AllMessageValues]:
    return [{"role": "user", "content": request_text(data)}]


def _guarded_text(inputs: GenericGuardrailAPIInputs, model: str) -> str:
    texts: Final = inputs.get("texts", [])
    if len(texts) != 1:
        raise BadRequestError("Decisions guardrail must return one JSON document", model=model, llm_provider="")
    return texts[0]


class DecisionsHandler(BaseTranslation):
    def get_structured_messages(self, data: dict[str, object]) -> list[AllMessageValues]:
        return decisions_guardrail_messages(data)

    async def process_input_messages(
        self,
        data: dict[str, object],
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: Logging | None = None,
    ) -> dict[str, object]:
        request: Final = _REQUEST.validate_python(data)
        model: Final = str(data.get("model", ""))
        inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(
                texts=[request_text(data)],
                images=list(request.images or ()),
                model=model,
            ),
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        guarded: Final = _REQUEST.validate_json(_guarded_text(inputs, model))
        return {
            **data,
            **guarded.model_dump(mode="json", include={"state", "context", "questions"}),
            **({"images": inputs.get("images", list(request.images))} if request.images is not None else {}),
        }

    async def process_output_response(
        self,
        response: object,
        guardrail_to_apply: CustomGuardrail,
        litellm_logging_obj: Logging | None = None,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        request_data: dict[str, object] | None = None,
    ) -> DecisionsResult:
        result: Final = _RESULT.validate_python(response)
        model: Final = str((request_data or {}).get("model", ""))
        inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=[result.model_dump_json()], model=model),
            request_data=request_data if request_data is not None else {},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        guarded: Final = _RESULT.validate_json(_guarded_text(inputs, model))
        guarded._hidden_params.update(result._hidden_params)
        return guarded.model_copy(update={"usage": result.usage})
