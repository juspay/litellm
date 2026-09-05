"""
Production Logger with GCS Support for LiteLLM Proxy Server
Logs to separate GCS buckets for success/error events with custom folder structures
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from typing import Awaitable, Callable, Optional, Tuple, TypeVar

import anyio
import anyio.to_process

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.gcs_bucket.gcs_bucket_base import GCSBucketBase
from litellm.integrations.gcs_bucket.redaction import (
    REDACT_ENABLED,
    redact_dict_values,
    redact_messages,
    redact_text,
)

_GCS_CALLBACK_LIMITER = anyio.CapacityLimiter(1)
_GCS_PROCESS_THRESHOLD_BYTES = 64 * 1024
_T = TypeVar("_T")


def _sanitize_for_json(obj, seen=None):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v, seen) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item, seen) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(item, seen) for item in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _sanitize_for_json(obj.model_dump(), seen)
        except Exception:
            return str(obj)
    return str(obj)


async def _redact_messages_async(messages):
    if not REDACT_ENABLED:
        return messages
    return await _run_gcs_cpu_work(redact_messages, messages)


async def _redact_text_async(text):
    if not REDACT_ENABLED:
        return text
    return await _run_gcs_cpu_work(redact_text, text)


def _sanitize_and_redact_dict_values(value):
    return redact_dict_values(_sanitize_for_json(value))


async def _sanitize_and_redact_dict_values_async(value):
    return await _run_gcs_cpu_work(_sanitize_and_redact_dict_values, value)


def _string_content_size(value, seen=None) -> int:
    if isinstance(value, str):
        return len(value.encode(errors="surrogatepass"))
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, dict):
        return sum(_string_content_size(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_string_content_size(item, seen) for item in value)
    return 0


async def _run_gcs_cpu_work(callback, value):
    if _string_content_size(value) >= _GCS_PROCESS_THRESHOLD_BYTES:
        return await anyio.to_process.run_sync(callback, value)
    return await anyio.to_thread.run_sync(callback, value)


async def _sanitize_for_json_async(value):
    if not REDACT_ENABLED:
        return _sanitize_for_json(value)
    return await anyio.to_thread.run_sync(
        _sanitize_for_json,
        value,
    )


def _serialize_for_gcs(data):
    return json.dumps(data, default=str)


async def _serialize_for_gcs_async(data):
    if not REDACT_ENABLED:
        return _serialize_for_gcs(data)
    return await _run_gcs_cpu_work(_serialize_for_gcs, data)


def _serialize_logprobs(logprobs):
    if hasattr(logprobs, "model_dump"):
        return logprobs.model_dump()
    if hasattr(logprobs, "to_dict"):
        return logprobs.to_dict()
    return logprobs


async def _serialize_logprobs_async(logprobs):
    if not REDACT_ENABLED:
        return _serialize_logprobs(logprobs)
    return await anyio.to_thread.run_sync(
        _serialize_logprobs,
        logprobs,
    )


async def _run_with_gcs_callback_slot(
    callback: Callable[[], Awaitable[_T]],
) -> _T:
    borrower = object()

    async def run_callback() -> _T:
        await _GCS_CALLBACK_LIMITER.acquire_on_behalf_of(borrower)
        try:
            return await callback()
        finally:
            _GCS_CALLBACK_LIMITER.release_on_behalf_of(borrower)

    # LoggingWorker can time out an individual callback. The queued CPU work must
    # still finish and upload its audit record after that timeout.
    task = asyncio.create_task(run_callback())
    return await asyncio.shield(task)


class ProductionGCSLogger(CustomLogger):
    """Production logger with async GCS bucket support using custom folder structures"""

    def __init__(self):
        super().__init__()
        self.success_bucket_name = os.getenv("GCS_SUCCESS_BUCKET_NAME")
        self.error_bucket_name = os.getenv("GCS_ERROR_BUCKET_NAME")
        self.service_account_path = os.getenv("GCS_PATH_SERVICE_ACCOUNT")

        # Initialize GCS base for async operations
        self.gcs_base = GCSBucketBase(bucket_name=self.success_bucket_name)

        if not self.success_bucket_name or not self.error_bucket_name:
            verbose_logger.warning(
                "⚠️  GCS bucket names not set. GCS logging disabled."
            )
        else:
            verbose_logger.info(
                f"✅ GCS initialized: {self.success_bucket_name}, {self.error_bucket_name}"
            )

    async def _upload_to_gcs_async(
        self,
        data: dict,
        bucket_name: str,
        log_type: str,
        serialized_data: Optional[str] = None,
    ):
        """Upload log data to GCS bucket using async I/O"""
        if not bucket_name:
            return

        try:
            timestamp = datetime.utcnow().strftime("%H-%M-%S")
            date = datetime.utcnow().strftime("%Y-%m-%d")
            correlation_id = data.get("correlation_id", str(uuid.uuid4()))

            if log_type == "success":
                # Success logs: date={date}/{timestamp}_{correlation_id}.json
                # Using hive-style partitioning for BigQuery cost optimization
                # User/dept/team info is in JSON for querying
                filename = f"{timestamp}_{correlation_id}.json"
                gcs_path = f"success/date={date}/{filename}"
            else:
                # Error logs: date={date}/{timestamp}_{correlation_id}.json
                # Using hive-style partitioning for BigQuery cost optimization
                filename = f"{timestamp}_{correlation_id}.json"
                gcs_path = f"failure/date={date}/{filename}"

            # Use async httpx to upload to GCS
            headers = await self.gcs_base.construct_request_headers(
                service_account_json=self.service_account_path, vertex_instance=None
            )

            # Upload using the GCS REST API
            # Note: No indent - BigQuery requires single-line JSON (NEWLINE_DELIMITED_JSON format)
            json_data = serialized_data
            if json_data is None:
                json_data = await _serialize_for_gcs_async(data)
            await self.gcs_base._log_json_data_on_gcs(
                headers=headers,
                bucket_name=bucket_name,
                object_name=gcs_path,
                logging_payload=json_data,
            )

        except Exception as e:
            verbose_logger.exception(f"❌ GCS upload error: {e}")

    async def _prepare_log_for_upload(
        self,
        build_log: Callable[[], Awaitable[Optional[dict]]],
        bucket_name: Optional[str],
    ) -> Optional[Tuple[dict, Optional[str]]]:
        try:
            data = await build_log()
            if data is None:
                return None
            data = await _sanitize_and_redact_dict_values_async(data)
            serialized_data = (
                await _serialize_for_gcs_async(data) if bucket_name else None
            )
            return data, serialized_data
        except Exception as e:
            verbose_logger.exception(f"Error preparing GCS log: {e}")
            return None

    async def _process_log_event(
        self,
        build_log: Callable[[], Awaitable[Optional[dict]]],
        bucket_name: Optional[str],
        log_type: str,
    ) -> None:
        if REDACT_ENABLED:
            prepared_log = await _run_with_gcs_callback_slot(
                lambda: self._prepare_log_for_upload(build_log, bucket_name)
            )
        else:
            data = await build_log()
            prepared_log = (data, None) if data is not None else None

        if prepared_log is not None and bucket_name:
            data, serialized_data = prepared_log
            await self._upload_to_gcs_async(
                data,
                bucket_name,
                log_type,
                serialized_data=serialized_data,
            )

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def _get_session_id(self, kwargs, litellm_params, metadata) -> Optional[str]:
        """
        Extract session ID from request parameters.
        Priority: litellm_session_id > metadata.session_id
        """
        if litellm_params.get("litellm_session_id"):
            return str(litellm_params.get("litellm_session_id"))
        if metadata.get("session_id"):
            return str(metadata.get("session_id"))
        if kwargs.get("litellm_session_id"):
            return str(kwargs.get("litellm_session_id"))
        return None

    def _should_skip_logging(self, kwargs) -> bool:
        """
        Check if logging should be skipped based on x-litellm-disable-logging header.

        Returns True if header is present and set to "true" (case-insensitive).
        """
        try:
            from litellm.litellm_core_utils.llm_request_utils import (
                get_proxy_server_request_headers,
            )

            litellm_params = kwargs.get("litellm_params", {})
            request_headers = get_proxy_server_request_headers(litellm_params)

            disable_logging_header = request_headers.get(
                "x-litellm-disable-logging", ""
            )

            # Check if header value is "true" (case-insensitive)
            if disable_logging_header.lower().strip() == "true":
                verbose_logger.debug(
                    "GCS Logger: Skipping logging due to x-litellm-disable-logging header"
                )
                return True

            return False
        except Exception as e:
            # Don't fail logging if header check fails
            verbose_logger.debug(
                f"GCS Logger: Error checking disable-logging header: {e}"
            )
            return False

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful requests for LLM training history"""
        # Check if logging should be skipped via header
        if self._should_skip_logging(kwargs):
            print("[GCS Logger] Skipping logging due to x-litellm-disable-logging header", flush=True)
            return

        await self._process_log_event(
            lambda: self._build_success_log(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            ),
            self.success_bucket_name,
            "success",
        )

    async def _build_success_log(
        self, kwargs, response_obj, start_time, end_time
    ):
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get(
                "metadata", litellm_params.get("litellm_metadata", {})
            ) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            success_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "litellm_session_id": session_id,
                "type": "SUCCESS",
                "user": {
                    "email": (
                        metadata.get("user_api_key_user_email")
                        or metadata.get("user_email")
                        or kwargs.get("user_email")
                        or kwargs.get("metadata", {}).get("user_api_key_user_email")
                    ),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": (metadata.get("user_api_key_metadata") or {}).get(
                        "department", "unknown"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "used": getattr(response_obj, "model", None),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "mode": metadata.get("model_info", {}).get("mode"),
                },
                "conversation": {
                    "messages": await _redact_messages_async(
                        kwargs.get("input", kwargs.get("messages", []))
                    ),
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "top_p": kwargs.get("top_p"),
                    "frequency_penalty": kwargs.get("frequency_penalty"),
                    "presence_penalty": kwargs.get("presence_penalty"),
                    "tools": kwargs.get("tools"),
                    "tool_choice": kwargs.get("tool_choice"),
                },
                "response": {},
                "usage": {},
                "cost": 0,
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
                "headers": metadata.get("headers"),
            }
            if hasattr(response_obj, "choices") and response_obj.choices:
                choice = response_obj.choices[0]
                success_log["response"] = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content": None,
                    "tool_calls": None,
                    "function_call": None,
                    "reasoning_content": None,
                    "thinking_blocks": None,
                    "reasoning_items": None,
                }

                if hasattr(choice, "message"):
                    message = choice.message
                    success_log["response"]["content"] = await _redact_text_async(
                        getattr(message, "content", None)
                    )
                    reasoning = getattr(
                        message, "reasoning_content", None
                    ) or getattr(
                        message, "reasoning", None
                    )
                    success_log["response"]["reasoning_content"] = (
                        await _redact_text_async(reasoning)
                    )
                    success_log["response"]["tool_calls"] = (
                        await _sanitize_for_json_async(
                            getattr(message, "tool_calls", None)
                        )
                    )
                    success_log["response"]["function_call"] = (
                        await _sanitize_for_json_async(
                            getattr(message, "function_call", None)
                        )
                    )
                    thinking_blocks = await _sanitize_for_json_async(
                        getattr(message, "thinking_blocks", None)
                    )
                    if isinstance(thinking_blocks, list):
                        for block in thinking_blocks:
                            if isinstance(block, dict) and "thinking" in block:
                                block["thinking"] = await _redact_text_async(
                                    block["thinking"]
                                )
                    success_log["response"]["thinking_blocks"] = thinking_blocks
                    success_log["response"]["reasoning_items"] = (
                        await _sanitize_for_json_async(
                            getattr(message, "reasoning_items", None)
                        )
                    )

                # --- RL training fields: logprobs + token_ids ---
                if hasattr(choice, "logprobs") and choice.logprobs is not None:
                    try:
                        success_log["response"]["logprobs"] = (
                            await _serialize_logprobs_async(choice.logprobs)
                        )
                    except Exception:
                        success_log["response"]["logprobs"] = str(choice.logprobs)

                token_ids = getattr(choice, "token_ids", None)
                if token_ids is None:
                    provider_fields = getattr(choice, "provider_specific_fields", {}) or {}
                    token_ids = provider_fields.get("token_ids")
                if token_ids is not None:
                    success_log["response"]["token_ids"] = token_ids

                provider_fields = getattr(choice, "provider_specific_fields", {}) or {}
                if provider_fields:
                    success_log["response"]["provider_specific_fields"] = provider_fields


            if hasattr(response_obj, "prompt_token_ids"):
                success_log["prompt_token_ids"] = response_obj.prompt_token_ids
            if hasattr(response_obj, "usage"):
                usage = response_obj.usage
                success_log["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }

            try:
                success_log["cost"] = litellm.completion_cost(
                    completion_response=response_obj
                )
            except Exception:
                success_log["cost"] = 0

            return success_log

        except Exception as e:
            verbose_logger.exception(f"Error logging success: {e}")
            return None

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed requests for debugging"""
        # Check if logging should be skipped via header
        if self._should_skip_logging(kwargs):
            return

        await self._process_log_event(
            lambda: self._build_failure_log(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            ),
            self.error_bucket_name,
            "error",
        )

    async def _build_failure_log(
        self, kwargs, response_obj, start_time, end_time
    ):
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get(
                "metadata", litellm_params.get("litellm_metadata", {})
            ) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            error_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat(),
                "litellm_session_id": session_id,
                "type": "ERROR",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                    "team_alias": metadata.get("user_api_key_team_alias"),
                    "department": (metadata.get("user_api_key_metadata") or {}).get(
                        "department"
                    ),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "deployment": metadata.get("deployment"),
                    "model_group": metadata.get("model_group"),
                    "api_base": litellm_params.get("api_base"),
                    "provider": litellm_params.get("custom_llm_provider"),
                },
                "request": {
                    "messages_count": len(kwargs.get("messages", [])),
                    "first_message": await _serialize_for_gcs_async(
                        await _redact_messages_async(kwargs.get("messages", [])),
                    ),
                    "max_tokens": kwargs.get("max_tokens"),
                    "route": metadata.get("user_api_key_request_route"),
                },
                "error": {
                    "type": type(response_obj).__name__,
                    "message": str(response_obj),
                    "exception": str(kwargs.get("exception", "")),
                    "traceback": str(kwargs.get("traceback_exception", "")),
                },
                "timing": {
                    "start_time": str(start_time),
                    "end_time": str(end_time),
                    "duration_seconds": (
                        (end_time - start_time).total_seconds()
                        if start_time and end_time
                        else None
                    ),
                    "llm_api_duration_ms": metadata.get("llm_api_duration_ms"),
                },
            }

            return error_log

        except Exception as e:
            verbose_logger.exception(f"Error logging failure: {e}")
            return None


# Handler instance
logger_instance = ProductionGCSLogger()


if __name__ == "__main__":
    print("=" * 80)
    print("Production Logger with GCS Support")
    print("=" * 80)
    print("\n📝 Logs to:")
    print("   • GCS_SUCCESS_BUCKET_NAME (cloud)")
    print("   • GCS_ERROR_BUCKET_NAME (cloud)")
    print("\n🔧 Environment Variables:")
    print("   GCS_SUCCESS_BUCKET_NAME - Success logs bucket")
    print("   GCS_ERROR_BUCKET_NAME - Error logs bucket")
    print("   GCS_PATH_SERVICE_ACCOUNT - Service account JSON (optional)")
    print("\n📝 Config usage:")
    print("litellm_settings:")
    print("  callbacks: gcs_logger.logger_instance")
    print("=" * 80)
