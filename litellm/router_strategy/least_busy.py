#### What this does ####
#   identifies least busy deployment
#   How is this achieved?
#   - Before each call, have the router print the state of requests {"deployment": "requests_in_flight"}
#   - use litellm.input_callbacks to log when a request is just about to be made to a model - {"deployment-id": traffic}
#   - use litellm.success + failure callbacks to log when a request completed
#   - in get_available_deployment, for a given model group name -> pick based on traffic

import random
from typing import Optional

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger


class LeastBusyLoggingHandler(CustomLogger):
    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0

    def __init__(self, router_cache: DualCache):
        self.router_cache = router_cache


    def log_pre_api_call(self, model, messages, kwargs):
        """
        Log when a model is being used.

        Caching based on model group.
        """
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key = f"{model_group}_request_count"
                # update cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_dict[id] = request_count_dict.get(id, 0) + 1

                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_value: Optional[int] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    self.router_cache.get_cache(key=request_count_api_key) or {}
                )
                request_count_value: Optional[int] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                self.router_cache.set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )

                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    await self.router_cache.async_get_cache(key=request_count_api_key)
                    or {}
                )
                request_count_value: Optional[int] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_success += 1
        except Exception:
            pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            if kwargs["litellm_params"].get("metadata") is None:
                pass
            else:
                model_group = kwargs["litellm_params"]["metadata"].get(
                    "model_group", None
                )
                id = kwargs["litellm_params"].get("model_info", {}).get("id", None)
                if model_group is None or id is None:
                    return
                elif isinstance(id, int):
                    id = str(id)

                request_count_api_key = f"{model_group}_request_count"
                # decrement count in cache
                request_count_dict = (
                    await self.router_cache.async_get_cache(key=request_count_api_key)
                    or {}
                )
                request_count_value: Optional[int] = request_count_dict.get(id, 0)
                if request_count_value is None:
                    return
                request_count_dict[id] = request_count_value - 1
                await self.router_cache.async_set_cache(
                    key=request_count_api_key, value=request_count_dict
                )

                ### TESTING ###
                if self.test_flag:
                    self.logged_failure += 1
        except Exception:
            pass

    def _get_available_deployments(
        self,
        healthy_deployments: list,
        all_deployments: dict,
    ):
        """
        Helper to get deployments using least busy strategy
        """
        # Extract healthy deployment IDs for logging
        healthy_ids = [d["model_info"]["id"] for d in healthy_deployments]

        print(f"[Least-Busy DEBUG] Cached all_deployments: {all_deployments}")
        print(f"[Least-Busy DEBUG] Healthy deployment IDs: {healthy_ids}")

        # Pick least busy deployment by iterating only through healthy deployments
        # This ensures we don't consider stale/removed deployments from cache
        min_traffic = float("inf")
        min_deployment = None

        for d in healthy_deployments:
            deployment_id = d["model_info"]["id"]
            # Get traffic count from cache, default to 0 if not yet tracked
            traffic = all_deployments.get(deployment_id, 0)

            if traffic < min_traffic:
                min_traffic = traffic
                min_deployment = d

        # If no deployment found (empty healthy_deployments), return random choice
        if min_deployment is None:
            print("[Least-Busy DEBUG] WARNING: No deployment found, falling back to RANDOM choice")
            min_deployment = random.choice(healthy_deployments)
        else:
            print(f"[Least-Busy DEBUG] Selected deployment ID: {min_deployment['model_info']['id']} with traffic={min_traffic}")

        return min_deployment

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
    ):
        """
        Sync helper to get deployments using least busy strategy
        """
        request_count_api_key = f"{model_group}_request_count"
        all_deployments = self.router_cache.get_cache(key=request_count_api_key) or {}
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )

    async def async_get_available_deployments(
        self, model_group: str, healthy_deployments: list
    ):
        """
        Async helper to get deployments using least busy strategy
        """
        request_count_api_key = f"{model_group}_request_count"
        all_deployments = (
            await self.router_cache.async_get_cache(key=request_count_api_key) or {}
        )
        return self._get_available_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
        )
