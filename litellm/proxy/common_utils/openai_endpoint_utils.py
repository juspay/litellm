"""
Contains utils used by OpenAI compatible endpoints
"""

from typing import Final, Optional, Set

from fastapi import Request

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

SENSITIVE_DATA_MASKER = SensitiveDataMasker()

_CREDENTIAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "client_secret",
        "vertex_credentials",
        "vertex_ai_credentials",
        "aws_access_key_id",
        "aws_secret_access_key",
        "azure_ad_token",
        "configurable_clientside_auth_params",
    }
)

_INFRA_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "api_base",
        "base_url",
        "aws_role_name",
        "aws_bedrock_runtime_endpoint",
        "aws_bedrock_project_id",
        "vertex_project",
        "use_vertex_project_id",
        "s3_bucket_name",
        "s3_encryption_key_id",
        "gcs_bucket_name",
        "auto_router_config_path",
        "auto_router_config",
        "complexity_router_config",
        "adaptive_router_config",
        "quality_router_config",
        "configurable_clients",
    }
)


def remove_sensitive_info_from_deployment(
    deployment_dict: dict,
    excluded_keys: Optional[Set[str]] = None,
) -> dict:
    """
    Removes sensitive information from a deployment dictionary.

    Credential keys are dropped entirely; other keys whose *values* look
    sensitive are masked. Infra metadata keys (internal endpoint URLs, cloud
    role/project identifiers, storage bucket names, router config paths) are
    also dropped: they are not credentials, but they reveal the deployment
    topology that a model/info caller has no need for.

    Args:
        deployment_dict (dict): The deployment dictionary to remove sensitive information from.
        excluded_keys (Optional[Set[str]]): Set of keys that should not be masked (exact match).

    Returns:
        dict: The modified deployment dictionary with sensitive information removed.
    """
    for key in _CREDENTIAL_KEYS | _INFRA_METADATA_KEYS:
        deployment_dict["litellm_params"].pop(key, None)

    # Rate-limit config fields must never be masked — they are integers, not credentials.
    # The field names contain "key" which matches the masker's sensitive pattern, so we
    # explicitly exclude them here rather than widening the global non_sensitive_overrides.
    _rate_limit_config_keys = {
        "default_api_key_tpm_limit",
        "default_api_key_rpm_limit",
    }
    _excluded = (excluded_keys or set()) | _rate_limit_config_keys

    deployment_dict["litellm_params"] = SENSITIVE_DATA_MASKER.mask_dict(
        deployment_dict["litellm_params"], excluded_keys=_excluded
    )

    return deployment_dict


async def get_custom_llm_provider_from_request_body(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request body

    Safely reads the request body
    """
    request_body: dict = await _read_request_body(request=request) or {}
    if "custom_llm_provider" in request_body:
        return request_body["custom_llm_provider"]
    return None


def get_custom_llm_provider_from_request_query(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request query parameters

    Safely reads the request query parameters
    """
    if "custom_llm_provider" in request.query_params:
        return request.query_params["custom_llm_provider"]
    return None


def get_custom_llm_provider_from_request_headers(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request header `custom-llm-provider`
    """
    if "custom-llm-provider" in request.headers:
        return request.headers["custom-llm-provider"]
    return None
