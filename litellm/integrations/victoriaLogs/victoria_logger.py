import json
import os
import httpx
import uuid
import time
import litellm
from datetime import datetime
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_logger

class VictoriaLogsLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        # VictoriaLogs endpoint (e.g., http://localhost:9428/insert/jsonline)
        self.vl_url = os.getenv("VICTORIA_LOGS_URL")
        
        if not self.vl_url:
            verbose_logger.warning("⚠️ VICTORIA_LOGS_URL not set. VictoriaLogs logging disabled.")
        else:
            verbose_logger.info(f"✅ VictoriaLogs initialized at: {self.vl_url}")

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

    async def _upload_to_victoria_async(self, data: dict):
        """Push log data to VictoriaLogs using async I/O"""
        if not self.vl_url:
            return

        try:
            # Create the message string: everything EXCEPT the temporal field
            msg_payload = {k: v for k, v in data.items() if k != "timestamp_iso"}
            
            # VictoriaLogs specific fields:
            # _time: Required for temporal indexing
            # _msg: The full searchable JSON string
            # _stream: Defines the log source/labels (as JSON object)
            payload = {
                "_time": data.get("timestamp_iso"),
                "_msg": json.dumps(msg_payload, default=str),
                "_stream": {"source": "litellm-proxy", "app": "litellm"},
                **msg_payload  # Unpack fields as individual columns for filtering
            }

            headers = {"Content-Type": "application/stream+json"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.vl_url,
                    content=json.dumps(payload, default=str),
                    headers=headers,
                    timeout=5.0
                )
                response.raise_for_status()
                
        except Exception as e:
            verbose_logger.exception(f"❌ VictoriaLogs upload error: {e}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log successful requests for LLM training history"""
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            success_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat() + "Z",
                "litellm_session_id": session_id,
                "type": "SUCCESS",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
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
                    "messages": kwargs.get("input", kwargs.get("messages", [])),
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
                }

                if hasattr(choice, "message"):
                    message = choice.message
                    success_log["response"]["content"] = getattr(
                        message, "content", None
                    )
                    success_log["response"]["reasoning_content"] = getattr(
                        message, "reasoning_content", None
                    )
                    success_log["response"]["tool_calls"] = getattr(
                        message, "tool_calls", None
                    )
                    success_log["response"]["function_call"] = getattr(
                        message, "function_call", None
                    )

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

            await self._upload_to_victoria_async(success_log)
        except Exception as e:
            verbose_logger.exception(f"Error logging success: {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Log failed requests for debugging"""
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or litellm_params.get("litellm_metadata", {})
            # Extract date and session_id for queryability
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            session_id = self._get_session_id(kwargs, litellm_params, metadata)

            error_log = {
                "correlation_id": correlation_id,
                "timestamp": time.time(),
                "timestamp_iso": datetime.utcnow().isoformat() + "Z",
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
                    "first_message": (
                        kwargs.get("messages", [{}])[0].get("content", "")[:100]
                        if kwargs.get("messages")
                        else None
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

            await self._upload_to_victoria_async(error_log)
        except Exception as e:
            verbose_logger.exception(f"Error logging failure: {e}")

# This is the instance referenced in the config
victoria_logger_instance = VictoriaLogsLogger()