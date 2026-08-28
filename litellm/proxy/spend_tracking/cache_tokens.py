from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError


class _PromptTokensDetails(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    cached_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    created_cache_tokens: Optional[int] = None


class _CacheUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    cache_read_input_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    prompt_tokens_details: Optional[_PromptTokensDetails] = None


@dataclass(frozen=True, slots=True)
class CacheTokenCounts:
    read: Optional[int]
    creation: Optional[int]


UNREPORTED_CACHE_TOKENS = CacheTokenCounts(read=None, creation=None)


def _first_reported(*values: Optional[int]) -> Optional[int]:
    reported = tuple(value for value in values if value is not None)
    if not reported:
        return None
    return next((value for value in reported if value), 0)


def extract_cache_token_counts(usage: object) -> CacheTokenCounts:
    try:
        parsed = _CacheUsage.model_validate(usage)
    except ValidationError:
        return UNREPORTED_CACHE_TOKENS

    details = parsed.prompt_tokens_details
    return CacheTokenCounts(
        read=_first_reported(
            parsed.cache_read_input_tokens,
            details.cached_tokens if details else None,
        ),
        creation=_first_reported(
            parsed.cache_creation_input_tokens,
            details.cache_write_tokens if details else None,
            details.cache_creation_tokens if details else None,
            details.created_cache_tokens if details else None,
        ),
    )
