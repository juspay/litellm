import json
import os
import httpx
import uuid
import time
import litellm
from datetime import datetime
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
            # _stream: Defines the log source/labels
            payload = {
                "_time": data.get("timestamp_iso"),
                "_msg": json.dumps(msg_payload, default=str),
                "_stream": "{source=\"litellm-proxy\", app=\"litellm\"}",
                **data # Unpacks all fields as individual columns for filtering
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
        try:
            correlation_id = getattr(response_obj, "id", None) or str(uuid.uuid4())
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})

            success_log = {
                "correlation_id": correlation_id,
                "timestamp_iso": datetime.utcnow().isoformat() + "Z",
                "type": "SUCCESS",
                "user": {
                    "email": metadata.get("user_api_key_user_email"),
                    "user_id": metadata.get("user_api_key_user_id"),
                },
                "model": {
                    "requested": kwargs.get("model"),
                    "used": getattr(response_obj, "model", None),
                },
                "usage": {
                    "prompt_tokens": getattr(response_obj.usage, "prompt_tokens", 0) if hasattr(response_obj, "usage") else 0,
                    "completion_tokens": getattr(response_obj.usage, "completion_tokens", 0) if hasattr(response_obj, "usage") else 0,
                },
            }

            try:
                success_log["cost"] = litellm.completion_cost(completion_response=response_obj) or 0
            except Exception:
                success_log["cost"] = 0

            success_log["timing_ms"] = (end_time - start_time).total_seconds() * 1000 if start_time and end_time else 0
            
            await self._upload_to_victoria_async(success_log)
        except Exception as e:
            verbose_logger.exception(f"Error logging success: {e}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            error_log = {
                "correlation_id": str(uuid.uuid4()),
                "timestamp_iso": datetime.utcnow().isoformat() + "Z",
                "type": "ERROR",
                "model": {"requested": kwargs.get("model")},
                "error": {
                    "type": type(response_obj).__name__,
                    "message": str(response_obj),
                }
            }
            await self._upload_to_victoria_async(error_log)
        except Exception as e:
            verbose_logger.exception(f"Error logging failure: {e}")

# This is the instance referenced in the config
victoria_logger_instance = VictoriaLogsLogger()