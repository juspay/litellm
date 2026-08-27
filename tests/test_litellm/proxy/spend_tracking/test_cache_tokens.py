import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.spend_tracking.cache_tokens import (
    UNREPORTED_CACHE_TOKENS,
    CacheTokenCounts,
    extract_cache_token_counts,
)


def test_anthropic_top_level_fields():
    usage = {"cache_read_input_tokens": 6503, "cache_creation_input_tokens": 0}
    result = extract_cache_token_counts(usage)
    assert result == CacheTokenCounts(read=6503, creation=0)


def test_anthropic_genuine_miss_is_zero_not_none():
    """A cold Anthropic call reports 0 explicitly; that must survive as 0, not collapse to unreported."""
    usage = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 6503}
    result = extract_cache_token_counts(usage)
    assert result.read == 0
    assert result.creation == 6503


def test_openai_compatible_prompt_tokens_details():
    usage = {"prompt_tokens": 6431, "prompt_tokens_details": {"cached_tokens": 5632}}
    result = extract_cache_token_counts(usage)
    assert result.read == 5632
    assert result.creation is None


def test_sglang_omits_prompt_tokens_details_entirely():
    """sglang sends no prompt_tokens_details object at all on a cold call; must not raise and must report unreported."""
    usage = {"prompt_tokens": 19, "completion_tokens": 5}
    result = extract_cache_token_counts(usage)
    assert result == UNREPORTED_CACHE_TOKENS


def test_prompt_tokens_details_present_but_null():
    usage = {"prompt_tokens": 19, "prompt_tokens_details": None}
    result = extract_cache_token_counts(usage)
    assert result == UNREPORTED_CACHE_TOKENS


def test_vllm_created_cache_tokens_spelling():
    """vLLM's kimi-k2 deployment spells the write-side field created_cache_tokens; the old extractor missed it."""
    usage = {"prompt_tokens_details": {"created_cache_tokens": 512}}
    result = extract_cache_token_counts(usage)
    assert result.creation == 512


def test_cache_write_tokens_spelling():
    usage = {"prompt_tokens_details": {"cache_write_tokens": 200}}
    result = extract_cache_token_counts(usage)
    assert result.creation == 200


def test_cache_creation_tokens_spelling():
    usage = {"prompt_tokens_details": {"cache_creation_tokens": 300}}
    result = extract_cache_token_counts(usage)
    assert result.creation == 300


def test_no_cache_fields_at_all_is_unreported():
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    result = extract_cache_token_counts(usage)
    assert result == UNREPORTED_CACHE_TOKENS


def test_garbage_input_does_not_raise():
    result = extract_cache_token_counts("not a usage object")
    assert result == UNREPORTED_CACHE_TOKENS


def test_none_input_does_not_raise():
    result = extract_cache_token_counts(None)
    assert result == UNREPORTED_CACHE_TOKENS


def test_accepts_attribute_style_object_not_just_dict():
    """db_spend_update_writer passes a raw usage_obj typed as `object`; pydantic objects use attribute access."""

    class Details:
        cached_tokens = 42
        cache_write_tokens = None
        cache_creation_tokens = None
        created_cache_tokens = None

    class Usage:
        cache_read_input_tokens = None
        cache_creation_input_tokens = None
        prompt_tokens_details = Details()

    result = extract_cache_token_counts(Usage())
    assert result.read == 42
