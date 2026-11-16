import copy

import pytest

from litellm.proxy.common_utils.openai_endpoint_utils import (
    remove_sensitive_info_from_deployment,
)


@pytest.mark.parametrize(
    "model_config, expected_config",
    [
        # Test case 1: Empty litellm_params
        (
            {"model_name": "test-model", "litellm_params": {}},
            {"model_name": "test-model", "litellm_params": {}},
        ),
        # Test case 2: Full sensitive data removal, mixed secrets of azure, aws, gcp, and typical api_key
        (
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-sensitive-key-123",
                    "client_secret": "~v8Q4W:Zp9gJ-3sTqX5aB@LkR2mNfYdC",
                    "vertex_credentials": {"type": "service_account"},
                    "aws_access_key_id": "AKIA123456789",
                    "aws_secret_access_key": "secret-access-key",
                    "api_base": "https://api.openai.com/v1",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
        ),
        # Test case 3: Partial sensitive data, api_key
        (
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "anthropic/claude-3",
                    "api_key": "sk-anthropic-key",
                    "temperature": 0.5,
                },
            },
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3", "temperature": 0.5},
            },
        ),
        # Test case 4: No sensitive data
        (
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
        ),
    ],
)
def test_remove_sensitive_info_from_deployment(model_config: dict, expected_config: dict):
    sanitized_config = remove_sensitive_info_from_deployment(model_config)
    assert sanitized_config == expected_config


def test_remove_sensitive_info_from_deployment_with_excluded_keys():
    """
    Test that excluded_keys prevents masking of specific keys (exact match).
    """
    base_config = {
        "model_name": "test-model",
        "litellm_params": {
            "model": "openai/gpt-4",
            "api_key": "sk-sensitive-key-123",
            "litellm_credentials_name": "my-credential-name",
            "access_token": "token-12345",
            "temperature": 0.7,
        },
    }

    # Without excluded_keys, access_token should be masked (contains "token")
    sanitized_config = remove_sensitive_info_from_deployment(copy.deepcopy(base_config))
    assert sanitized_config["litellm_params"]["access_token"] != "token-12345"
    assert "*" in sanitized_config["litellm_params"]["access_token"]

    # With excluded_keys, litellm_credentials_name should NOT be masked.
    # ``remove_sensitive_info_from_deployment`` mutates its input, so feed it
    # a fresh copy rather than the already-sanitized one.
    sanitized_config = remove_sensitive_info_from_deployment(
        copy.deepcopy(base_config), excluded_keys={"litellm_credentials_name"}
    )
    assert sanitized_config["litellm_params"]["litellm_credentials_name"] == "my-credential-name"

    # access_token should still be masked (not in excluded_keys)
    assert sanitized_config["litellm_params"]["access_token"] != "token-12345"
    assert "*" in sanitized_config["litellm_params"]["access_token"]

    # api_key should still be removed (popped) regardless of excluded_keys
    assert "api_key" not in sanitized_config["litellm_params"]


@pytest.mark.parametrize(
    "infra_key, infra_value",
    [
        ("api_base", "https://internal-gw.juspay-prod.internal/v1"),
        ("base_url", "https://internal-gw.juspay-prod.internal/v1"),
        ("aws_role_name", "arn:aws:iam::123456789012:role/llm-proxy-prod"),
        ("aws_bedrock_runtime_endpoint", "https://bedrock-runtime.internal:8443"),
        ("aws_bedrock_project_id", "proj-77b0a6"),
        ("vertex_project", "xyne-prod"),
        ("use_vertex_project_id", "xyne-prod"),
        ("s3_bucket_name", "s3://llm-artifacts-prod"),
        ("s3_encryption_key_id", "arn:aws:kms:us-east-1:123456789012:key/abc"),
        ("gcs_bucket_name", "gs://llm-artifacts-prod"),
        ("azure_ad_token", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"),
        ("configurable_clientside_auth_params", {"Authorization": "Bearer abc"}),
        ("configurable_clients", {"main": {"api_key": "sk-123"}}),
        ("auto_router_config_path", "/etc/litellm/auto_router.yaml"),
        ("auto_router_config", "llm_api_routes: [chat]"),
    ],
)
def test_remove_sensitive_info_drops_infra_metadata(infra_key: str, infra_value):
    """Infra topology keys must be dropped, not masked: masked values would
    round-trip through the dashboard edit form into the DB."""
    deployment = {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "openai/gpt-4",
            infra_key: infra_value,
            "temperature": 0.7,
        },
    }
    sanitized = remove_sensitive_info_from_deployment(deployment)
    assert infra_key not in sanitized["litellm_params"]
    assert sanitized["litellm_params"]["model"] == "openai/gpt-4"
    assert sanitized["litellm_params"]["temperature"] == 0.7


def test_remove_sensitive_info_drops_router_config_paths():
    """Router config blobs are admin-only infra; every *_router_config* variant drops."""
    deployment = {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "openai/gpt-4",
            "auto_router_config": "routes: [a]",
            "complexity_router_config": "routes: [b]",
            "adaptive_router_config": "routes: [c]",
            "quality_router_config": "routes: [d]",
        },
    }
    sanitized = remove_sensitive_info_from_deployment(deployment)
    for key in (
        "auto_router_config",
        "complexity_router_config",
        "adaptive_router_config",
        "quality_router_config",
    ):
        assert key not in sanitized["litellm_params"]
