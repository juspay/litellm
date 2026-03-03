#### What this does ####
#   identifies lowest tpm deployment
import random
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm import token_counter
from litellm._logging import verbose_logger, verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import RouterErrors
from litellm.types.utils import LiteLLMPydanticObjectBase, StandardLoggingPayload
from litellm.utils import get_utc_datetime, print_verbose

from .base_routing_strategy import BaseRoutingStrategy

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class RoutingArgs(LiteLLMPydanticObjectBase):
    ttl: int = 10 * 60  # 10min (RPM/TPM expire key - handles long-running requests)


class LowestTPMLoggingHandler_v2(BaseRoutingStrategy, CustomLogger):
    """
    Updated version of TPM/RPM Logging.

    Meant to work across instances.

    Caches individual models, not model_groups

    Uses batch get (redis.mget)

    Increments tpm/rpm limit using redis.incr

    NEW: Pre-reserves TPM based on estimated input tokens for fair cross-pod routing.
         Actual tokens are adjusted after request completion.
    """

    test_flag: bool = False
    logged_success: int = 0
    logged_failure: int = 0
    default_cache_time_seconds: int = 1 * 60 * 60  # 1 hour

    # Store estimated tokens per (model_id, minute) for cross-request correlation
    # This is needed because async_log_success_event receives different kwargs
    _estimated_tokens_cache: Dict[str, float] = {}

    def __init__(
        self, router_cache: DualCache, routing_args: dict = {}
    ):
        self.router_cache = router_cache
        self.routing_args = RoutingArgs(**routing_args)
        BaseRoutingStrategy.__init__(
            self,
            dual_cache=router_cache,
            should_batch_redis_writes=True,
            default_sync_interval=0.1,
        )

    def _log_redis_unavailable_warning(self, values: Optional[List]) -> None:
        """Log warning if Redis returned None for all values (Redis may be unavailable)."""
        if values is not None and all(v is None for v in values):
            verbose_router_logger.warning(
                "[Usage-Based-Routing-v2 WARNING] Redis returned None for all deployments - "
                "Redis may be unavailable. Falling back to random routing."
            )

    def pre_call_check(self, deployment: Dict) -> Optional[Dict]:
        """
        Pre-call check + update model rpm

        Returns - deployment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        try:
            # ------------
            # Setup values
            # ------------

            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")
            rpm_key = f"{model_id}:{deployment_name}:rpm:{current_minute}"

            local_result = self.router_cache.get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")

            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}. id={}, model_group={}. Get the model info by calling 'router.get_model_info(id)".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                            model_id,
                            deployment.get("model_name", ""),
                        ),
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks

                result = self.router_cache.increment_cache(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                    )
            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    async def async_pre_call_check(
        self,
        deployment: Dict,
        parent_otel_span: Optional[Span] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Optional[Dict]:
        """
        Pre-call check + update model rpm AND estimated tpm
        - Used inside semaphore
        - raise rate limit error if deployment over limit
        - Increments TPM by estimated input tokens for fair cross-pod routing

        Why? solves concurrency issue - https://github.com/BerriAI/litellm/issues/2994

        Returns - deployment with estimated_input_tokens stored for later adjustment

        Raises - RateLimitError if deployment over defined RPM limit
        """
        # Get messages from deployment dict (fallback if not passed as arg)
        if messages is None and deployment:
            messages = deployment.get("_messages")
        estimated_input_tokens = 0
        try:
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")
            model_id = deployment.get("model_info", {}).get("id")
            deployment_name = deployment.get("litellm_params", {}).get("model")

            rpm_key = f"{model_id}:{deployment_name}:rpm:{current_minute}"
            local_result = await self.router_cache.async_get_cache(
                key=rpm_key, local_only=True
            )  # check local result first

            deployment_rpm = None
            if deployment_rpm is None:
                deployment_rpm = deployment.get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("litellm_params", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = deployment.get("model_info", {}).get("rpm")
            if deployment_rpm is None:
                deployment_rpm = float("inf")
            if local_result is not None and local_result >= deployment_rpm:
                raise litellm.RateLimitError(
                    message="Deployment over defined rpm limit={}. current usage={}".format(
                        deployment_rpm, local_result
                    ),
                    llm_provider="",
                    model=deployment.get("litellm_params", {}).get("model"),
                    response=httpx.Response(
                        status_code=429,
                        content="{} rpm limit={}. current usage={}".format(
                            RouterErrors.user_defined_ratelimit_error.value,
                            deployment_rpm,
                            local_result,
                        ),
                        headers={"retry-after": str(60)},  # type: ignore
                        request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                    ),
                    num_retries=deployment.get("num_retries"),
                )
            else:
                # if local result below limit, check redis ## prevent unnecessary redis checks
                result = await self._increment_value_in_current_window(
                    key=rpm_key, value=1, ttl=self.routing_args.ttl
                )
                if result is not None and result > deployment_rpm:
                    raise litellm.RateLimitError(
                        message="Deployment over defined rpm limit={}. current usage={}".format(
                            deployment_rpm, result
                        ),
                        llm_provider="",
                        model=deployment.get("litellm_params", {}).get("model"),
                        response=httpx.Response(
                            status_code=429,
                            content="{} rpm limit={}. current usage={}".format(
                                RouterErrors.user_defined_ratelimit_error.value,
                                deployment_rpm,
                                result,
                            ),
                            headers={"retry-after": str(60)},  # type: ignore
                            request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                        ),
                        num_retries=deployment.get("num_retries"),
                    )

            # ------------
            # NEW: Estimate and reserve TPM for fair cross-pod routing
            # ------------
            if messages:
                try:
                    # Get model name for token counting
                    model_name = deployment.get("litellm_params", {}).get(
                        "model", "gpt-3.5-turbo"
                    )

                    # Estimate input tokens
                    estimated_input_tokens = token_counter(
                        model=model_name, messages=messages
                    )

                    tpm_key = f"{model_id}:{deployment_name}:tpm:{current_minute}"

                    # Use IMMEDIATE Redis increment (not batched) for cross-pod visibility
                    await self.router_cache.async_increment_cache(
                        key=tpm_key,
                        value=float(estimated_input_tokens),
                        ttl=self.routing_args.ttl,
                        parent_otel_span=parent_otel_span,
                    )

                    print(
                        f"[Usage-Based-Routing-v2] Reserved {estimated_input_tokens} tokens for deployment {model_id}"
                    )

                    # Store estimated tokens keyed by request_id for later adjustment
                    # Using request_id instead of (model_id, minute) to handle cross-minute requests
                    request_id = deployment.get("_request_id")
                    if request_id:
                        cache_key = f"{model_id}:{request_id}"
                        LowestTPMLoggingHandler_v2._estimated_tokens_cache[cache_key] = {
                            "estimated_tokens": float(estimated_input_tokens),
                            "minute": current_minute,
                        }

                except Exception as e:
                    print(
                        f"[Usage-Based-Routing-v2] Token estimation failed: {e}"
                    )
                    pass  # Don't fail if token estimation fails

            return deployment
        except Exception as e:
            if isinstance(e, litellm.RateLimitError):
                raise e
            return deployment  # don't fail calls if eg. redis fails to connect

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM/RPM usage on success
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"].get("litellm_model_name")
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None or model is None:
                return
            elif isinstance(id, int):
                id = str(id)

            total_tokens = standard_logging_object.get("total_tokens")

            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime(
                "%H-%M"
            )  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache

            ## TPM
            self.router_cache.increment_cache(
                key=tpm_key, value=total_tokens, ttl=self.routing_args.ttl
            )
            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            """
            Update TPM usage on success

            NOTE: TPM is adjusted by delta (actual - estimated) because we already
            reserved estimated tokens in async_pre_call_check for fair cross-pod routing.
            """
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object"
            )
            if standard_logging_object is None:
                raise ValueError("standard_logging_object not passed in.")
            model_group = standard_logging_object.get("model_group")
            model = standard_logging_object["hidden_params"]["litellm_model_name"]
            id = standard_logging_object.get("model_id")
            if model_group is None or id is None:
                return
            elif isinstance(id, int):
                id = str(id)
            total_tokens = standard_logging_object.get("total_tokens")
            # ------------
            # Setup values
            # ------------
            dt = get_utc_datetime()
            current_minute = dt.strftime(
                "%H-%M"
            )  # use the same timezone regardless of system clock

            tpm_key = f"{id}:{model}:tpm:{current_minute}"
            # ------------
            # Update usage
            # ------------
            # update cache
            parent_otel_span = _get_parent_otel_span_from_kwargs(kwargs)

            # NEW: Get estimated tokens from class-level cache (set in async_pre_call_check)
            # This is keyed by request_id to handle cross-minute requests correctly
            request_id = kwargs.get("litellm_call_id") or standard_logging_object.get("litellm_call_id")
            estimated_input_tokens = 0
            cache_key = None
            if request_id:
                cache_key = f"{id}:{request_id}"
                cached_data = LowestTPMLoggingHandler_v2._estimated_tokens_cache.get(cache_key)
                if cached_data:
                    estimated_input_tokens = cached_data.get("estimated_tokens", 0)

            # Clean up the cache entry (one-time use per request)
            if cache_key and cache_key in LowestTPMLoggingHandler_v2._estimated_tokens_cache:
                del LowestTPMLoggingHandler_v2._estimated_tokens_cache[cache_key]

            # Calculate delta (actual - estimated)
            # We reserved estimated tokens in pre_call_check, now adjust to actual
            delta = float(total_tokens) - float(estimated_input_tokens)

            ## TPM - adjust by delta instead of adding total_tokens
            await self.router_cache.async_increment_cache(
                key=tpm_key,
                value=delta,
                ttl=self.routing_args.ttl,
                parent_otel_span=parent_otel_span,
            )

            print(
                f"[Usage-Based-Routing-v2] Adjusted TPM by {delta} "
                f"(actual={total_tokens}, estimated={estimated_input_tokens}) "
                f"for deployment {id}"
            )

            ### TESTING ###
            if self.test_flag:
                self.logged_success += 1
        except Exception as e:
            verbose_logger.exception(
                "litellm.proxy.hooks.lowest_tpm_rpm_v2.py::async_log_success_event(): Exception occured - {}".format(
                    str(e)
                )
            )
            pass

    def _return_potential_deployments(
        self,
        healthy_deployments: List[Dict],
        all_deployments: Dict,
        input_tokens: int,
        rpm_dict: Dict,
    ):
        lowest_tpm = float("inf")
        potential_deployments = []  # if multiple deployments have the same low value
        for item, item_tpm in all_deployments.items():
            ## get the item from model list
            _deployment = None
            item = item.split(":")[0]
            for m in healthy_deployments:
                if item == m["model_info"]["id"]:
                    _deployment = m
            if _deployment is None:
                continue  # skip to next one
            elif item_tpm is None:
                continue  # skip if unhealthy deployment

            _deployment_tpm = None
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("litellm_params", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = _deployment.get("model_info", {}).get("tpm")
            if _deployment_tpm is None:
                _deployment_tpm = float("inf")

            _deployment_rpm = None
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("litellm_params", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = _deployment.get("model_info", {}).get("rpm")
            if _deployment_rpm is None:
                _deployment_rpm = float("inf")
            if item_tpm + input_tokens > _deployment_tpm:
                continue
            elif (
                (rpm_dict is not None and item in rpm_dict)
                and rpm_dict[item] is not None
                and (rpm_dict[item] + 1 >= _deployment_rpm)
            ):
                continue
            elif item_tpm == lowest_tpm:
                potential_deployments.append(_deployment)
            elif item_tpm < lowest_tpm:
                lowest_tpm = item_tpm
                potential_deployments = [_deployment]
        return potential_deployments

    def _common_checks_available_deployment(  # noqa: PLR0915
        self,
        model_group: str,
        healthy_deployments: list,
        tpm_keys: list,
        tpm_values: Optional[list],
        rpm_keys: list,
        rpm_values: Optional[list],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ) -> Optional[dict]:
        """
        Common checks for get available deployment, across sync + async implementations
        """

        if tpm_values is None or rpm_values is None:
            return None

        tpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(tpm_keys):
            tpm_dict[tpm_keys[idx].split(":")[0]] = tpm_values[idx]

        rpm_dict = {}  # {model_id: 1, ..}
        for idx, key in enumerate(rpm_keys):
            rpm_dict[rpm_keys[idx].split(":")[0]] = rpm_values[idx]

        try:
            input_tokens = token_counter(messages=messages, text=input)
        except Exception:
            input_tokens = 0
        verbose_router_logger.debug(f"input_tokens={input_tokens}")
        # -----------------------
        # Find lowest used model
        # ----------------------

        if tpm_dict is None:  # base case - none of the deployments have been used
            # initialize a tpm dict with {model_id: 0}
            tpm_dict = {}
            for deployment in healthy_deployments:
                tpm_dict[deployment["model_info"]["id"]] = 0
        else:
            for d in healthy_deployments:
                ## if healthy deployment not yet used
                tpm_key = d["model_info"]["id"]
                if tpm_key not in tpm_dict or tpm_dict[tpm_key] is None:
                    tpm_dict[tpm_key] = 0

        all_deployments = tpm_dict
        potential_deployments = self._return_potential_deployments(
            healthy_deployments=healthy_deployments,
            all_deployments=all_deployments,
            input_tokens=input_tokens,
            rpm_dict=rpm_dict,
        )
        print_verbose("returning picked lowest tpm/rpm deployment.")

        if len(potential_deployments) > 0:
            return random.choice(potential_deployments)
        else:
            return None

    async def async_get_available_deployments(  # noqa: PLR0915
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
    ):
        """
        Async implementation of get deployments.

        Reduces time to retrieve the tpm/rpm values from cache.

        Uses a 10-minute sliding window to track TPM/RPM, which handles:
        - Long-running requests (>60s)
        - Better load distribution across minute boundaries
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()

        # Generate keys for last 10 minutes (sliding window)
        # This ensures we don't lose track of in-flight long requests at minute boundaries
        tpm_keys = []
        rpm_keys = []
        for m in healthy_deployments:
            if isinstance(m, dict):
                id = m.get("model_info", {}).get(
                    "id"
                )  # a deployment should always have an 'id'. this is set in router.py
                deployment_name = m.get("litellm_params", {}).get("model")

                # Query last 10 minutes of keys
                for minute_offset in range(10):
                    minute_dt = dt - timedelta(minutes=minute_offset)
                    minute_str = minute_dt.strftime("%H-%M")
                    tpm_key = "{}:{}:tpm:{}".format(id, deployment_name, minute_str)
                    rpm_key = "{}:{}:rpm:{}".format(id, deployment_name, minute_str)
                    tpm_keys.append(tpm_key)
                    rpm_keys.append(rpm_key)

        combined_tpm_rpm_keys = tpm_keys + rpm_keys

        # Use redis_only=True to get global counts across all pods
        # This is critical for distributed deployments where multiple pods need to see
        # the global TPM/RPM count, not their local stale view
        combined_tpm_rpm_values = await self.router_cache.async_batch_get_cache(
            keys=combined_tpm_rpm_keys,
            redis_only=True,
        )  # [1, 2, None, ..]

        self._log_redis_unavailable_warning(combined_tpm_rpm_values)

        # Sum TPM/RPM values across the 10-minute window for each deployment
        if combined_tpm_rpm_values is not None:
            # Group values by deployment (each deployment has 10 keys)
            num_deployments = len(healthy_deployments)
            tpm_values = []
            rpm_values = []

            for i in range(num_deployments):
                # Sum TPM for this deployment across 10 minutes
                deployment_tpm_values = combined_tpm_rpm_values[
                    i * 10 : (i + 1) * 10
                ]
                deployment_tpm_sum = sum(
                    v for v in deployment_tpm_values if v is not None
                )
                tpm_values.append(deployment_tpm_sum)

                # Sum RPM for this deployment across 10 minutes
                rpm_start_idx = len(tpm_keys) + (i * 10)
                rpm_end_idx = rpm_start_idx + 10
                deployment_rpm_values = combined_tpm_rpm_values[rpm_start_idx:rpm_end_idx]
                deployment_rpm_sum = sum(
                    v for v in deployment_rpm_values if v is not None
                )
                rpm_values.append(deployment_rpm_sum)
        else:
            tpm_values = None
            rpm_values = None

        # Use current minute keys for logging/debugging purposes
        current_minute = dt.strftime("%H-%M")
        current_tpm_keys = [
            "{}:{}:tpm:{}".format(
                m.get("model_info", {}).get("id"),
                m.get("litellm_params", {}).get("model"),
                current_minute,
            )
            for m in healthy_deployments
            if isinstance(m, dict)
        ]
        current_rpm_keys = [
            "{}:{}:rpm:{}".format(
                m.get("model_info", {}).get("id"),
                m.get("litellm_params", {}).get("model"),
                current_minute,
            )
            for m in healthy_deployments
            if isinstance(m, dict)
        ]

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=current_tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=current_rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise litellm.RateLimitError(
                message=f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}",
                llm_provider="",
                model=model_group,
                response=httpx.Response(
                    status_code=429,
                    content="",
                    headers={"retry-after": str(60)},  # type: ignore
                    request=httpx.Request(method="tpm_rpm_limits", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
            )

    def get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: list,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        parent_otel_span: Optional[Span] = None,
    ):
        """
        Returns a deployment with the lowest TPM/RPM usage.

        Uses a 10-minute sliding window to track TPM/RPM, which handles:
        - Long-running requests (>60s)
        - Better load distribution across minute boundaries
        """
        # get list of potential deployments
        verbose_router_logger.debug(
            f"get_available_deployments - Usage Based. model_group: {model_group}, healthy_deployments: {healthy_deployments}"
        )

        dt = get_utc_datetime()

        # Generate keys for last 10 minutes (sliding window)
        # This ensures we don't lose track of in-flight long requests at minute boundaries
        tpm_keys = []
        rpm_keys = []
        for m in healthy_deployments:
            if isinstance(m, dict):
                id = m.get("model_info", {}).get(
                    "id"
                )  # a deployment should always have an 'id'. this is set in router.py
                deployment_name = m.get("litellm_params", {}).get("model")

                # Query last 10 minutes of keys
                for minute_offset in range(10):
                    minute_dt = dt - timedelta(minutes=minute_offset)
                    minute_str = minute_dt.strftime("%H-%M")
                    tpm_key = "{}:{}:tpm:{}".format(id, deployment_name, minute_str)
                    rpm_key = "{}:{}:rpm:{}".format(id, deployment_name, minute_str)
                    tpm_keys.append(tpm_key)
                    rpm_keys.append(rpm_key)

        # Use redis_only=True to get global counts across all pods
        # This is critical for distributed deployments where multiple pods need to see
        # the global TPM/RPM count, not their local stale view
        combined_tpm_values = self.router_cache.batch_get_cache(
            keys=tpm_keys, parent_otel_span=parent_otel_span, redis_only=True
        )  # [1, 2, None, ..]
        combined_rpm_values = self.router_cache.batch_get_cache(
            keys=rpm_keys, parent_otel_span=parent_otel_span, redis_only=True
        )  # [1, 2, None, ..]

        # Sum TPM/RPM values across the 10-minute window for each deployment
        num_deployments = len(healthy_deployments)
        tpm_values = []
        rpm_values = []

        for i in range(num_deployments):
            # Sum TPM for this deployment across 10 minutes
            deployment_tpm_values = combined_tpm_values[i * 10 : (i + 1) * 10]
            deployment_tpm_sum = sum(v for v in deployment_tpm_values if v is not None)
            tpm_values.append(deployment_tpm_sum)

            # Sum RPM for this deployment across 10 minutes
            deployment_rpm_values = combined_rpm_values[i * 10 : (i + 1) * 10]
            deployment_rpm_sum = sum(v for v in deployment_rpm_values if v is not None)
            rpm_values.append(deployment_rpm_sum)

        # Log warning if Redis is unavailable
        if healthy_deployments:
            self._log_redis_unavailable_warning(
                (combined_tpm_values or []) + (combined_rpm_values or []) or None
            )

        # Use current minute keys for logging/debugging purposes
        current_minute = dt.strftime("%H-%M")
        current_tpm_keys = [
            "{}:{}:tpm:{}".format(
                m.get("model_info", {}).get("id"),
                m.get("litellm_params", {}).get("model"),
                current_minute,
            )
            for m in healthy_deployments
            if isinstance(m, dict)
        ]
        current_rpm_keys = [
            "{}:{}:rpm:{}".format(
                m.get("model_info", {}).get("id"),
                m.get("litellm_params", {}).get("model"),
                current_minute,
            )
            for m in healthy_deployments
            if isinstance(m, dict)
        ]

        deployment = self._common_checks_available_deployment(
            model_group=model_group,
            healthy_deployments=healthy_deployments,
            tpm_keys=current_tpm_keys,
            tpm_values=tpm_values,
            rpm_keys=current_rpm_keys,
            rpm_values=rpm_values,
            messages=messages,
            input=input,
        )

        try:
            assert deployment is not None
            return deployment
        except Exception:
            ### GET THE DICT OF TPM / RPM + LIMITS PER DEPLOYMENT ###
            deployment_dict = {}
            for index, _deployment in enumerate(healthy_deployments):
                if isinstance(_deployment, dict):
                    id = _deployment.get("model_info", {}).get("id")
                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_tpm = None
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("tpm", None)
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("litellm_params", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = _deployment.get("model_info", {}).get(
                            "tpm", None
                        )
                    if _deployment_tpm is None:
                        _deployment_tpm = float("inf")

                    ### GET CURRENT TPM ###
                    current_tpm = tpm_values[index] if tpm_values else 0

                    ### GET DEPLOYMENT TPM LIMIT ###
                    _deployment_rpm = None
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("rpm", None)
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("litellm_params", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = _deployment.get("model_info", {}).get(
                            "rpm", None
                        )
                    if _deployment_rpm is None:
                        _deployment_rpm = float("inf")

                    ### GET CURRENT RPM ###
                    current_rpm = rpm_values[index] if rpm_values else 0

                    deployment_dict[id] = {
                        "current_tpm": current_tpm,
                        "tpm_limit": _deployment_tpm,
                        "current_rpm": current_rpm,
                        "rpm_limit": _deployment_rpm,
                    }
            raise ValueError(
                f"{RouterErrors.no_deployments_available.value}. Passed model={model_group}. Deployments={deployment_dict}"
            )
