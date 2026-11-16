"""
Azure AI Anthropic CountTokens API transformation logic.

Extends the base Anthropic CountTokens transformation with Azure authentication.
"""

from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.types.router import GenericLiteLLMParams


class AzureAIAnthropicCountTokensConfig(AnthropicCountTokensConfig):
    """
    Configuration and transformation logic for Azure AI Anthropic CountTokens API.

    Extends AnthropicCountTokensConfig with Azure authentication.
    Azure AI Anthropic uses the same endpoint format but with Azure auth headers.
    """

    def get_required_headers(
        self,
        api_key: str,
        litellm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Get the required headers for the Azure AI Anthropic CountTokens API.

        Azure AI Anthropic uses Anthropic's native API format, which requires the
        x-api-key header for authentication (in addition to Azure's api-key header).

        Args:
            api_key: The Azure AI API key
            litellm_params: Optional LiteLLM parameters for additional auth config

        Returns:
            Dictionary of required headers with both x-api-key and Azure authentication
        """
        # Start with base headers including x-api-key for Anthropic API compatibility
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
            "x-api-key": api_key,  # Azure AI Anthropic requires this header
        }

        # Also set up Azure auth headers for flexibility
        litellm_params = litellm_params or {}
        if "api_key" not in litellm_params:
            litellm_params["api_key"] = api_key

        litellm_params_obj = GenericLiteLLMParams(**litellm_params)

        # Get Azure auth headers (api-key or Authorization)
        azure_headers = BaseAzureLLM._base_validate_azure_environment(headers={}, litellm_params=litellm_params_obj)

        # Merge Azure auth headers
        headers.update(azure_headers)

        return headers

    def get_count_tokens_endpoint(self, api_base: str) -> str:
        """
        Get the Azure AI Anthropic CountTokens API endpoint.

        Args:
            api_base: The Azure AI API base URL
                      (e.g., https://my-resource.services.ai.azure.com or
                       https://my-resource.services.ai.azure.com/anthropic)

        Returns:
            The endpoint URL for the CountTokens API

        Raises:
            AnthropicError: If api_base is not a parsable http(s) URL
        """
        # Azure AI Anthropic endpoint format:
        # https://<resource>.services.ai.azure.com/anthropic/v1/messages/count_tokens
        parsed = urlparse(api_base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise AnthropicError(
                status_code=400,
                message=f"Invalid api_base for Azure AI Anthropic count_tokens: {api_base}",
            )

        # Rebuild origin + path without query/fragment; ensure the path ends
        # with the /anthropic segment exactly once, appended (not substring-
        # matched, so https://host/x/anthropic/../ or embedded query strings
        # can't smuggle a different path).
        path = parsed.path.rstrip("/")
        if not path.endswith("/anthropic"):
            path = f"{path}/anthropic"

        return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")) + "/v1/messages/count_tokens"
