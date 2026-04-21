"""
Functions to create audit logs for LiteLLM Proxy
"""

import json
import os
from litellm._uuid import uuid
from datetime import datetime, timezone

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    AUDIT_ACTIONS,
    LiteLLM_AuditLogs,
    LitellmTableNames,
    Optional,
    UserAPIKeyAuth,
)


async def write_audit_log(
    object_id: str,
    action: AUDIT_ACTIONS,
    user_api_key_dict: UserAPIKeyAuth,
    table_name: LitellmTableNames,
    before_value: Optional[str] = None,
    after_value: Optional[str] = None,
    litellm_changed_by: Optional[str] = None,
):
    """
    Lightweight audit log writer — no enterprise/premium gate.
    Enabled by setting LITELLM_STORE_AUDIT_LOGS=true.

    Records who did what (create/update/delete) on which object.
    """
    _enabled = os.environ.get("LITELLM_STORE_AUDIT_LOGS", "").lower() == "true"
    if not _enabled:
        return

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return

    _changed_by = litellm_changed_by or user_api_key_dict.user_email or user_api_key_dict.user_id or ""
    _changed_by_api_key = user_api_key_dict.api_key or ""

    from prisma import Json as PrismaJson

    def _to_prisma_json(val):
        """Wrap value in PrismaJson for Json? Prisma fields."""
        if val is None:
            return None
        if isinstance(val, dict):
            return PrismaJson(val)
        try:
            return PrismaJson(json.loads(val))
        except Exception:
            return PrismaJson({"value": val})

    _before = _to_prisma_json(before_value)
    _after = _to_prisma_json(after_value)

    data: dict = {
        "id": str(uuid.uuid4()),
        "updated_at": datetime.now(timezone.utc),
        "changed_by": _changed_by,
        "changed_by_api_key": _changed_by_api_key,
        "action": action,
        "table_name": table_name.value,
        "object_id": object_id,
    }
    if _before is not None:
        data["before_value"] = _before
    if _after is not None:
        data["updated_values"] = _after

    try:
        await prisma_client.db.litellm_auditlog.create(data=data)
    except Exception as e:
        verbose_proxy_logger.error(f"write_audit_log failed: {e}")


async def create_object_audit_log(
    object_id: str,
    action: AUDIT_ACTIONS,
    litellm_changed_by: Optional[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: Optional[str],
    table_name: LitellmTableNames,
    before_value: Optional[str] = None,
    after_value: Optional[str] = None,
):
    """
    Create an audit log for an internal user.

    Parameters:
    - user_id: str - The id of the user to create the audit log for.
    - action: AUDIT_ACTIONS - The action to create the audit log for.
    - user_row: LiteLLM_UserTable - The user row to create the audit log for.
    - litellm_changed_by: Optional[str] - The user id of the user who is changing the user.
    - user_api_key_dict: UserAPIKeyAuth - The user api key dictionary.
    - litellm_proxy_admin_name: Optional[str] - The name of the proxy admin.
    """
    from litellm.secret_managers.main import get_secret_bool

    store_audit_logs = litellm.store_audit_logs or get_secret_bool(
        "LITELLM_STORE_AUDIT_LOGS"
    )

    if store_audit_logs is not True:
        return

    await create_audit_log_for_update(
        request_data=LiteLLM_AuditLogs(
            id=str(uuid.uuid4()),
            updated_at=datetime.now(timezone.utc),
            changed_by=litellm_changed_by
            or user_api_key_dict.user_id
            or litellm_proxy_admin_name,
            changed_by_api_key=user_api_key_dict.api_key,
            table_name=table_name,
            object_id=object_id,
            action=action,
            updated_values=after_value,
            before_value=before_value,
        )
    )


async def create_audit_log_for_update(request_data: LiteLLM_AuditLogs):
    """
    Create an audit log for an object.
    """
    from litellm.secret_managers.main import get_secret_bool

    store_audit_logs = litellm.store_audit_logs or get_secret_bool(
        "LITELLM_STORE_AUDIT_LOGS"
    )
    if store_audit_logs is not True:
        return

    from litellm.proxy.proxy_server import premium_user, prisma_client

    if premium_user is not True:
        return

    if prisma_client is None:
        raise Exception("prisma_client is None, no DB connected")

    verbose_proxy_logger.debug("creating audit log for %s", request_data)

    if isinstance(request_data.updated_values, dict):
        request_data.updated_values = json.dumps(request_data.updated_values)

    if isinstance(request_data.before_value, dict):
        request_data.before_value = json.dumps(request_data.before_value)

    _request_data = request_data.model_dump(exclude_none=True)

    try:
        await prisma_client.db.litellm_auditlog.create(
            data={
                **_request_data,  # type: ignore
            }
        )
    except Exception as e:
        # [Non-Blocking Exception. Do not allow blocking LLM API call]
        verbose_proxy_logger.error(f"Failed Creating audit log {e}")

    return
