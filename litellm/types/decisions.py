from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Discriminator, Field, JsonValue, PrivateAttr, Tag, model_validator, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.types.llms.base import LiteLLMPydanticObjectBase

DecisionsJSON: TypeAlias = str | Mapping[str, object] | Sequence[object]
NoulCriteria: TypeAlias = Mapping[Literal["true", "false"], DecisionsJSON | None]


class NoulQuestion(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    instructions: DecisionsJSON | None = None
    criteria: NoulCriteria | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="after")
    def require_instructions_or_criteria(self) -> "NoulQuestion":
        if self.instructions is None and self.criteria is None:
            raise ValueError("A noul question requires instructions or criteria")
        return self


class ChoiceQuestion(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    instructions: DecisionsJSON | None = None
    criteria: Annotated[Mapping[str, DecisionsJSON | None], Field(min_length=1, max_length=255)]

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreQuestion(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    instructions: DecisionsJSON | None = None
    criteria: Annotated[Sequence[DecisionsJSON], Field(min_length=1, max_length=10)]

    model_config = ConfigDict(extra="allow", frozen=True)


class ExtractionQuestion(LiteLLMPydanticObjectBase):
    type: Literal["string", "value"]
    instructions: DecisionsJSON
    pattern: str | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsAudio(LiteLLMPydanticObjectBase):
    data: Annotated[str, Field(min_length=1)]
    format: Annotated[str, Field(min_length=1)]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


DecisionQuestion: TypeAlias = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | ExtractionQuestion,
    Field(discriminator="type"),
]

DecisionQuestionMap: TypeAlias = Annotated[
    Mapping[Annotated[str, Field(min_length=1)], DecisionQuestion],
    Field(min_length=1, max_length=128),
]


class DecisionsRequestBody(LiteLLMPydanticObjectBase):
    state: DecisionsJSON | None = None
    context: DecisionsJSON | None = None
    questions: DecisionQuestionMap
    images: Sequence[str] | None = None
    audio: DecisionsAudio | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_validator(mode="after")
    def require_grounding(self) -> "DecisionsRequestBody":
        if self.state is None and self.context is None and self.audio is None:
            raise ValueError("A Decisions request requires state, context or audio")
        return self


class DecisionsRequest(DecisionsRequestBody):
    model: str


@with_config(ConfigDict(extra="allow"))
class DecisionsCallParams(TypedDict, total=False):
    model: Required[ReadOnly[str]]
    state: ReadOnly[DecisionsJSON | None]
    context: ReadOnly[DecisionsJSON | None]
    audio: ReadOnly[DecisionsAudio | Mapping[str, str] | None]
    questions: Required[ReadOnly[DecisionQuestionMap]]
    images: ReadOnly[Sequence[str] | None]
    api_key: ReadOnly[str | None]
    api_base: ReadOnly[str | None]
    timeout: ReadOnly[float | None]
    custom_llm_provider: ReadOnly[str | None]
    extra_headers: ReadOnly[Mapping[str, str] | None]


class NoulAnswer(LiteLLMPydanticObjectBase):
    type: Literal["noul"]
    noul: float

    model_config = ConfigDict(extra="allow", frozen=True)


class ChoiceAnswer(LiteLLMPydanticObjectBase):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


class ScoreAnswer(LiteLLMPydanticObjectBase):
    type: Literal["score"]
    score: float
    confidence: float
    legend: Mapping[str, DecisionsJSON]
    probabilities: Mapping[str, float]

    model_config = ConfigDict(extra="allow", frozen=True)


class ValueAnswer(LiteLLMPydanticObjectBase):
    type: Literal["value"]
    value: JsonValue
    confidence: float | None = None

    model_config = ConfigDict(extra="allow", frozen=True)


DecisionAnswer: TypeAlias = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | ValueAnswer,
    Field(discriminator="type"),
]


class DecisionsUsage(LiteLLMPydanticObjectBase):
    input_tokens: int = 0
    output_tokens: int = 0

    model_config = ConfigDict(extra="allow", frozen=True)


class DecisionsResponse(LiteLLMPydanticObjectBase):
    model: str | None = None
    answers: Mapping[str, DecisionAnswer]
    usage: DecisionsUsage | None = None

    model_config = ConfigDict(extra="allow", frozen=True)

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)


class ExtractionUsage(LiteLLMPydanticObjectBase):
    input_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]
    thinking_tokens: Annotated[int, Field(ge=0)] = 0
    requests: Annotated[int, Field(ge=0)] = 1
    wall_s: Annotated[float, Field(ge=0)] = 0

    model_config = ConfigDict(extra="allow", frozen=True)

    @property
    def output_tokens(self) -> int:
        return self.completion_tokens


class ExtractionConfidence(LiteLLMPydanticObjectBase):
    mean_p: Annotated[float, Field(ge=0, le=1)]
    min_p: Annotated[float, Field(ge=0, le=1)]

    model_config = ConfigDict(extra="allow", frozen=True)


class ExtractionResponse(LiteLLMPydanticObjectBase):
    result: Mapping[str, JsonValue]
    usage: ExtractionUsage
    thinking: Mapping[str, JsonValue]
    confidence: Mapping[str, ExtractionConfidence]

    model_config = ConfigDict(extra="allow", frozen=True)

    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)


def _response_kind(value: object) -> str:
    if isinstance(value, Mapping):
        return "extraction" if "result" in value else "classification"
    return "extraction" if isinstance(value, ExtractionResponse) else "classification"


DecisionsResult: TypeAlias = Annotated[
    Annotated[DecisionsResponse, Tag("classification")] | Annotated[ExtractionResponse, Tag("extraction")],
    Discriminator(_response_kind),
]
