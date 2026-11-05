"""
Example custom secret manager for LiteLLM Proxy.

This is a simple in-memory secret manager for testing purposes.
In production, replace this with your actual secret management system.
"""

from typing import Optional, Union

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_secret_manager import CustomSecretManager


class InMemorySecretManager(CustomSecretManager):
    def __init__(self):
        super().__init__(secret_manager_name="in_memory_secrets")
        # Store your secrets in memory
        self.secrets = {}

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read secret asynchronously"""
        verbose_proxy_logger.debug("reading secret name=%s", secret_name)
        return self.secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read secret synchronously"""
        verbose_proxy_logger.debug("CUSTOM SECRET MANAGER: looking for secret: %s", secret_name)
        return self.secrets.get(secret_name)

    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None,
    ) -> dict:
        """Write a secret to the in-memory store"""
        self.secrets[secret_name] = secret_value
        return {
            "status": "success",
            "secret_name": secret_name,
            "description": description,
        }

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """Delete a secret from the in-memory store"""
        if secret_name in self.secrets:
            del self.secrets[secret_name]
            return {"status": "deleted", "secret_name": secret_name}
        return {"status": "not_found", "secret_name": secret_name}
