import hashlib
import hmac

import pytest


@pytest.fixture
def salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-pepper")
    yield "test-pepper"
    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)


def test_hash_token_unsalted_matches_legacy_sha256(monkeypatch) -> None:
    from litellm.proxy.common_utils.token_hash_utils import hash_token

    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    token = "sk-plain-legacy"
    assert hash_token(token) == hashlib.sha256(token.encode()).hexdigest()


def test_hash_token_salted_is_pepered_hmac(salt_key) -> None:
    from litellm.proxy.common_utils.token_hash_utils import hash_token

    token = "sk-pepered"
    expected = hmac.new(salt_key.encode(), token.encode(), hashlib.sha256).hexdigest()
    assert hash_token(token) == expected
    assert hash_token(token) != hashlib.sha256(token.encode()).hexdigest()


def test_hash_token_candidates_unsalted_is_single_legacy_digest(monkeypatch) -> None:
    from litellm.proxy.common_utils.token_hash_utils import hash_token_candidates

    monkeypatch.delenv("LITELLM_SALT_KEY", raising=False)
    token = "sk-plain-legacy"
    assert hash_token_candidates(token) == (hashlib.sha256(token.encode()).hexdigest(),)


def test_hash_token_candidates_salted_includes_both_digests(salt_key) -> None:
    from litellm.proxy.common_utils.token_hash_utils import (
        hash_token,
        hash_token_candidates,
        legacy_hash_token,
    )

    token = "sk-pepered"
    candidates = hash_token_candidates(token)
    assert candidates[0] == hash_token(token)
    assert legacy_hash_token(token) in candidates
    assert len(set(candidates)) == 2


def test_hash_token_stable_across_calls(salt_key) -> None:
    from litellm.proxy.common_utils.token_hash_utils import hash_token

    assert hash_token("sk-deterministic") == hash_token("sk-deterministic")


def test_hash_token_delegates_from_proxy_utils(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-pepper")
    from litellm.proxy.common_utils.token_hash_utils import hash_token as core_hash
    from litellm.proxy.utils import hash_token as proxy_hash

    assert proxy_hash("sk-delegated") == core_hash("sk-delegated")


def test_hash_token_delegates_from_types() -> None:
    import os

    os.environ["LITELLM_SALT_KEY"] = "test-pepper"
    try:
        from litellm.proxy._types import hash_token as types_hash
        from litellm.proxy.common_utils.token_hash_utils import hash_token as core_hash

        assert types_hash("sk-types-delegated") == core_hash("sk-types-delegated")
    finally:
        os.environ.pop("LITELLM_SALT_KEY", None)
