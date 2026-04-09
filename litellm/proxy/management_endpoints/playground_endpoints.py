"""
GPU PLAYGROUND MANAGEMENT

All /playground endpoints.

User-facing:
  GET    /playground/slots
  POST   /playground/bookings
  GET    /playground/bookings/me
  DELETE /playground/bookings/{booking_id}
  GET    /playground/ssh-keys
  POST   /playground/ssh-keys
  DELETE /playground/ssh-keys/{ssh_key_id}

Admin:
  GET    /playground/admin/nodes
  POST   /playground/admin/nodes
  PUT    /playground/admin/nodes/{node_id}
  DELETE /playground/admin/nodes/{node_id}
  GET    /playground/admin/status

Internal (cron, PROXY_ADMIN virtual key):
  GET    /playground/internal/allocations-tonight
  POST   /playground/internal/activation-status
  POST   /playground/internal/teardown-status

Design reference: docs/superpowers/plans/2026-04-09-playground-litellm-integration.md
"""

import base64
import binascii
import hashlib
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    ActivationStatusReportItem,
    ActivationStatusReportRequest,
    ActivationStatusReportResponse,
    AddPlaygroundSSHKeyRequest,
    CreatePlaygroundBookingRequest,
    CreatePlaygroundNodeRequest,
    LitellmUserRoles,
    PlaygroundAdminStatusResponse,
    PlaygroundAllocationBooking,
    PlaygroundAllocationNode,
    PlaygroundAllocationsTonightResponse,
    PlaygroundBookingPhase,
    PlaygroundBookingResponse,
    PlaygroundNodeResponse,
    PlaygroundSlotNode,
    PlaygroundSlotsResponse,
    PlaygroundSSHKeyResponse,
    TeardownStatusResponse,
    UpdatePlaygroundNodeRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

try:
    # pytz is in litellm's dep tree via several other modules; fall back to
    # zoneinfo if not available in some stripped environments
    import pytz

    IST = pytz.timezone("Asia/Kolkata")
except ImportError:  # pragma: no cover
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")


router = APIRouter()


# ---------------------------------------------------------------------------
# Config (from env, with defaults matching grid/backend/config/settings.py)
# ---------------------------------------------------------------------------

PLAYGROUND_MAX_GPUS_PER_USER = int(os.getenv("PLAYGROUND_MAX_GPUS_PER_USER", "8"))
PLAYGROUND_WEEKLY_BOOKING_LIMIT = int(os.getenv("PLAYGROUND_WEEKLY_BOOKING_LIMIT", "1"))
PLAYGROUND_OVERFLOW_START_HOUR = int(os.getenv("PLAYGROUND_OVERFLOW_START_HOUR", "22"))
PLAYGROUND_OVERFLOW_START_MINUTE = int(
    os.getenv("PLAYGROUND_OVERFLOW_START_MINUTE", "0")
)
PLAYGROUND_CUTOFF_HOUR = int(os.getenv("PLAYGROUND_CUTOFF_HOUR", "22"))
PLAYGROUND_CUTOFF_MINUTE = int(os.getenv("PLAYGROUND_CUTOFF_MINUTE", "30"))


# ---------------------------------------------------------------------------
# Helpers — time windows + allocation
# ---------------------------------------------------------------------------


def _tonight_date() -> date:
    """Return today's date in IST (this is the `night_of` booking key)."""
    return datetime.now(IST).date()


def _is_booking_open() -> Tuple[bool, str]:
    now_ist = datetime.now(IST)
    cutoff = now_ist.replace(
        hour=PLAYGROUND_CUTOFF_HOUR,
        minute=PLAYGROUND_CUTOFF_MINUTE,
        second=0,
        microsecond=0,
    )
    if now_ist >= cutoff:
        return False, "Booking cutoff has passed for tonight"
    return True, ""


def _is_overflow_window() -> bool:
    now_ist = datetime.now(IST)
    overflow_start = now_ist.replace(
        hour=PLAYGROUND_OVERFLOW_START_HOUR,
        minute=PLAYGROUND_OVERFLOW_START_MINUTE,
        second=0,
        microsecond=0,
    )
    cutoff = now_ist.replace(
        hour=PLAYGROUND_CUTOFF_HOUR,
        minute=PLAYGROUND_CUTOFF_MINUTE,
        second=0,
        microsecond=0,
    )
    return overflow_start <= now_ist < cutoff


def _booking_phase() -> PlaygroundBookingPhase:
    is_open, _ = _is_booking_open()
    if not is_open:
        return "closed"
    return "overflow" if _is_overflow_window() else "open"


# ---------------------------------------------------------------------------
# SSH public key validation (ported verbatim from
# grid/backend/app/services/playground_ssh_keys.py)
# ---------------------------------------------------------------------------


_SSH_KEY_TYPES = {
    "ssh-rsa",
    "ssh-ed25519",
    "ecdsa-sha2-nistp256",
    "ecdsa-sha2-nistp384",
    "ecdsa-sha2-nistp521",
}


def _validate_ssh_public_key(public_key: str) -> Tuple[bool, str]:
    parts = public_key.strip().split()
    if len(parts) < 2:
        return False, "SSH public key must have at least a type and key data"

    keytype = parts[0]
    if keytype not in _SSH_KEY_TYPES:
        return False, f"Unsupported SSH key type: {keytype}"

    try:
        base64.b64decode(parts[1], validate=True)
    except (binascii.Error, ValueError):
        return False, "SSH public key data is not valid base64"

    return True, ""


def _ssh_key_fingerprint(public_key: str) -> str:
    parts = public_key.strip().split()
    if len(parts) < 2:
        raise ValueError("Invalid SSH public key")
    raw = base64.b64decode(parts[1])
    digest = hashlib.sha256(raw).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


# ---------------------------------------------------------------------------
# Prisma access + auth helpers
# ---------------------------------------------------------------------------


def _get_prisma():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "prisma_client not initialized — connect the proxy to a database "
                "before calling /playground/* endpoints"
            ),
        )
    return prisma_client


def _role_value(user_api_key_dict: UserAPIKeyAuth) -> Optional[str]:
    role = user_api_key_dict.user_role
    if role is None:
        return None
    return role.value if hasattr(role, "value") else role


def _is_admin_read(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """True if the caller can read admin-scoped data. Both PROXY_ADMIN and
    PROXY_ADMIN_VIEW_ONLY satisfy this — view-only admins should be able to
    observe state (list nodes, view bookings) but not mutate it."""
    return _role_value(user_api_key_dict) in (
        LitellmUserRoles.PROXY_ADMIN.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )


def _is_admin_write(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """True if the caller can mutate admin-scoped data. PROXY_ADMIN only —
    PROXY_ADMIN_VIEW_ONLY is explicitly excluded from write paths, matching
    the convention in litellm's key_management_endpoints."""
    return _role_value(user_api_key_dict) == LitellmUserRoles.PROXY_ADMIN.value


# Back-compat alias — retain the short name for `_is_admin_read` callers.
# Prefer the explicit _is_admin_read / _is_admin_write at call sites.
_is_admin = _is_admin_read


def _require_admin_read(user_api_key_dict: UserAPIKeyAuth) -> None:
    if not _is_admin_read(user_api_key_dict):
        raise HTTPException(status_code=403, detail="admin role required")


def _require_admin_write(user_api_key_dict: UserAPIKeyAuth) -> None:
    if not _is_admin_write(user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="admin write role required (PROXY_ADMIN_VIEW_ONLY cannot mutate)",
        )


# Back-compat alias for the read gate — existing call sites stay unchanged.
_require_admin = _require_admin_read


def _effective_user_id_for_read(
    user_api_key_dict: UserAPIKeyAuth,
    target_user_id: Optional[str],
) -> str:
    """Resolve the effective user_id for a playground READ.

    Non-admin callers always act as themselves. Admin callers (including
    view-only) may override the acting user by passing `target_user_id`.
    This is the grid-as-admin forwarding path for listing bookings / keys.
    """
    if target_user_id and _is_admin_read(user_api_key_dict):
        return target_user_id
    return _caller_user_id(user_api_key_dict)


def _effective_user_id_for_write(
    user_api_key_dict: UserAPIKeyAuth,
    target_user_id: Optional[str],
) -> str:
    """Resolve the effective user_id for a playground WRITE.

    Non-admin callers always act as themselves. Only PROXY_ADMIN (not
    view-only) may override the acting user by passing `target_user_id`.
    View-only admins cannot create bookings or ssh keys on behalf of other
    users — they'd fall through to acting as themselves, which this helper
    blocks explicitly so the caller gets a clear 403 instead of silently
    writing under the wrong user_id.
    """
    if target_user_id:
        if not _is_admin_write(user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="target_user_id requires PROXY_ADMIN role",
            )
        return target_user_id
    return _caller_user_id(user_api_key_dict)


def _caller_user_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    caller_id = user_api_key_dict.user_id
    if not caller_id:
        raise HTTPException(
            status_code=400,
            detail="calling key has no user_id — cannot attribute playground operation",
        )
    return caller_id


# Back-compat alias — some call sites still use _effective_user_id. It maps
# to the read variant because every current caller that uses it is either a
# read path (list_my_playground_bookings, list_playground_ssh_keys) OR a
# write path that's being migrated below.
_effective_user_id = _effective_user_id_for_read


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


async def _user_has_ssh_key(user_id: str) -> bool:
    prisma = _get_prisma()
    count = await prisma.db.litellm_usersshkey.count(where={"user_id": user_id})
    return count > 0


async def _user_weekly_booking_count(user_id: str) -> int:
    """Count of bookings this user has made in the last 7 days (excluding
    cancelled). Uses UTC for the created_at comparison to match the rest
    of litellm's management endpoints. Prisma emits `TIMESTAMP(3)` (naive)
    for our created_at columns and stores the wall clock in UTC by
    convention, so comparing against a UTC-aware `now - 7d` is the
    consistent choice."""
    prisma = _get_prisma()
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    count = await prisma.db.litellm_playgroundbooking.count(
        where={
            "user_id": user_id,
            "created_at": {"gte": week_ago},
            "status": {"not": "cancelled"},
        }
    )
    return count


async def _user_has_booking_tonight(user_id: str) -> bool:
    """True if the user already has a non-cancelled booking for tonight —
    used to prevent double-booking even during the overflow window."""
    prisma = _get_prisma()
    existing = await prisma.db.litellm_playgroundbooking.count(
        where={
            "user_id": user_id,
            "night_of": _tonight_date(),
            "status": {"not": "cancelled"},
        }
    )
    return existing > 0


async def _can_user_book(user_id: str) -> Tuple[bool, str]:
    if not await _user_has_ssh_key(user_id):
        return False, "Register an SSH key before booking"

    is_open, msg = _is_booking_open()
    if not is_open:
        return False, msg

    # One booking per user per night, always. Overflow relaxes the weekly
    # limit below but does NOT let a user claim a second slot the same night.
    if await _user_has_booking_tonight(user_id):
        return False, "You already have a booking for tonight"

    # Weekly limit only enforced outside the overflow window. During overflow
    # (22:00–22:30 IST) users who've hit their weekly count can still claim
    # unclaimed slots — that's the whole point of overflow — but the per-
    # night uniqueness check above still applies.
    if not _is_overflow_window():
        count = await _user_weekly_booking_count(user_id)
        if count >= PLAYGROUND_WEEKLY_BOOKING_LIMIT:
            return (
                False,
                f"Weekly booking limit ({PLAYGROUND_WEEKLY_BOOKING_LIMIT}) reached",
            )

    return True, ""


async def _allocate_gpus(
    gpu_count: int, preferred_node: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Find a node with enough free GPUs tonight.

    Returns (node_ip, "0,1,2") or (None, None) when no node has capacity.

    KNOWN LIMITATION — TOCTOU race: the read-then-write sequence
    (`find_many` -> `create`) is not serialized at the DB level, so two
    concurrent booking requests can both see the same free GPU indices
    and create overlapping booking rows for the same `(allocated_node,
    night_of, allocated_gpus)`. The per-user-per-night uniqueness check
    in `_can_user_book` protects the common case (one user spamming),
    but two distinct users racing for the last slot is still possible.
    Mitigations to add in a follow-up: (a) a Postgres advisory lock
    keyed on `(allocated_node, night_of)` around the read-and-write,
    or (b) a `@@unique([allocated_node, allocated_gpus, night_of])`
    constraint and a catch-retry loop on unique violation.
    """
    prisma = _get_prisma()
    tonight = _tonight_date()

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        where={"is_playground_eligible": True, "is_healthy": True},
    )

    # Sort with preferred node first, then by ip_address for determinism
    if preferred_node:
        nodes = sorted(
            nodes,
            key=lambda n: (0 if n.ip_address == preferred_node else 1, n.ip_address),
        )
    else:
        nodes = sorted(nodes, key=lambda n: n.ip_address)

    for node in nodes:
        existing = await prisma.db.litellm_playgroundbooking.find_many(
            where={
                "allocated_node": node.ip_address,
                "night_of": tonight,
                "status": {"in": ["allocated", "active"]},
            }
        )
        used_gpu_indices = set()
        for b in existing:
            used_gpu_indices.update(
                idx for idx in (b.allocated_gpus or "").split(",") if idx
            )
        all_gpus = [str(i) for i in range(node.total_gpus)]
        free = [g for g in all_gpus if g not in used_gpu_indices]
        if len(free) >= gpu_count:
            return node.ip_address, ",".join(free[:gpu_count])

    return None, None


# ---------------------------------------------------------------------------
# Endpoint implementations
# ---------------------------------------------------------------------------


# ---- user-facing: slots ---------------------------------------------------


@router.get(
    "/playground/slots",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundSlotsResponse,
)
@management_endpoint_wrapper
async def get_playground_slots(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundSlotsResponse:
    """Available GPUs per eligible node tonight plus the booking phase."""
    prisma = _get_prisma()
    tonight = _tonight_date()

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        where={"is_playground_eligible": True, "is_healthy": True},
    )

    slot_nodes: List[PlaygroundSlotNode] = []
    for node in nodes:
        existing = await prisma.db.litellm_playgroundbooking.find_many(
            where={
                "allocated_node": node.ip_address,
                "night_of": tonight,
                "status": {"in": ["allocated", "active"]},
            }
        )
        used = 0
        for b in existing:
            used += len([g for g in (b.allocated_gpus or "").split(",") if g])
        slot_nodes.append(
            PlaygroundSlotNode(
                node_id=node.node_id,
                name=node.name,
                ip_address=node.ip_address,
                gpu_type=node.gpu_type,
                total_gpus=node.total_gpus,
                available_gpus=max(0, node.total_gpus - used),
            )
        )

    return PlaygroundSlotsResponse(
        night_of=tonight,
        booking_phase=_booking_phase(),
        nodes=slot_nodes,
    )


# ---- user-facing: bookings ------------------------------------------------


@router.post(
    "/playground/bookings",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundBookingResponse,
)
@management_endpoint_wrapper
async def create_playground_booking(
    request: Request,
    data: CreatePlaygroundBookingRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundBookingResponse:
    """Create a booking for tonight. See _can_user_book / _allocate_gpus."""
    prisma = _get_prisma()
    user_id = _effective_user_id_for_write(user_api_key_dict, data.target_user_id)

    if data.gpu_count > PLAYGROUND_MAX_GPUS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=(
                f"gpu_count must be <= {PLAYGROUND_MAX_GPUS_PER_USER} "
                f"(got {data.gpu_count})"
            ),
        )

    ok, reason = await _can_user_book(user_id)
    if not ok:
        raise HTTPException(status_code=409, detail=reason)

    node_ip, gpu_ids = await _allocate_gpus(data.gpu_count, data.preferred_node)
    if not node_ip or not gpu_ids:
        raise HTTPException(status_code=409, detail="No GPUs available on any node")

    booking = await prisma.db.litellm_playgroundbooking.create(
        data={
            "user_id": user_id,
            "gpu_count": data.gpu_count,
            "preferred_node": data.preferred_node,
            "allocated_node": node_ip,
            "allocated_gpus": gpu_ids,
            "night_of": _tonight_date(),
            "is_overflow": _is_overflow_window(),
        }
    )
    verbose_proxy_logger.info(
        f"playground: booking {booking.booking_id} created for "
        f"user={user_id} gpus={gpu_ids} node={node_ip}"
    )
    return PlaygroundBookingResponse(**booking.model_dump())


@router.get(
    "/playground/bookings/me",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[PlaygroundBookingResponse],
)
@management_endpoint_wrapper
async def list_my_playground_bookings(
    request: Request,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundBookingResponse]:
    """Current user's recent bookings (last 30 days, newest first)."""
    prisma = _get_prisma()
    user_id = _effective_user_id_for_read(user_api_key_dict, target_user_id)
    since = datetime.now(timezone.utc) - timedelta(days=30)

    rows = await prisma.db.litellm_playgroundbooking.find_many(
        where={"user_id": user_id, "created_at": {"gte": since}},
        order={"created_at": "desc"},
    )
    return [PlaygroundBookingResponse(**r.model_dump()) for r in rows]


@router.delete(
    "/playground/bookings/{booking_id}",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundBookingResponse,
)
@management_endpoint_wrapper
async def cancel_playground_booking(
    request: Request,
    booking_id: str,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundBookingResponse:
    """Cancel an allocated (not yet active) booking. Only the owner or a
    PROXY_ADMIN may cancel. target_user_id is accepted for admin-on-behalf-of
    consistency with the create path but is not required — admins can cancel
    any booking by id alone."""
    prisma = _get_prisma()

    booking = await prisma.db.litellm_playgroundbooking.find_unique(
        where={"booking_id": booking_id}
    )
    if booking is None:
        raise HTTPException(status_code=404, detail="booking not found")

    # Admins skip the ownership check entirely. Non-admins can only cancel
    # their own bookings — target_user_id is ignored for non-admin callers
    # (same contract as _effective_user_id_for_write).
    if not _is_admin_write(user_api_key_dict):
        caller_id = _caller_user_id(user_api_key_dict)
        if booking.user_id != caller_id:
            raise HTTPException(status_code=403, detail="not your booking")

    if booking.status != "allocated":
        raise HTTPException(
            status_code=409,
            detail=f"booking status is '{booking.status}' — only 'allocated' can be cancelled",
        )

    updated = await prisma.db.litellm_playgroundbooking.update(
        where={"booking_id": booking_id},
        data={"status": "cancelled"},
    )
    return PlaygroundBookingResponse(**updated.model_dump())


# ---- user-facing: ssh keys ------------------------------------------------


@router.get(
    "/playground/ssh-keys",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[PlaygroundSSHKeyResponse],
)
@management_endpoint_wrapper
async def list_playground_ssh_keys(
    request: Request,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundSSHKeyResponse]:
    prisma = _get_prisma()
    user_id = _effective_user_id_for_read(user_api_key_dict, target_user_id)
    rows = await prisma.db.litellm_usersshkey.find_many(
        where={"user_id": user_id},
        order={"created_at": "desc"},
    )
    return [PlaygroundSSHKeyResponse(**r.model_dump()) for r in rows]


@router.post(
    "/playground/ssh-keys",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundSSHKeyResponse,
)
@management_endpoint_wrapper
async def add_playground_ssh_key(
    request: Request,
    data: AddPlaygroundSSHKeyRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundSSHKeyResponse:
    prisma = _get_prisma()
    user_id = _effective_user_id_for_write(user_api_key_dict, data.target_user_id)

    public_key = data.public_key.strip()
    name = data.name.strip()
    if not public_key:
        raise HTTPException(status_code=400, detail="public_key is required")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    valid, validation_error = _validate_ssh_public_key(public_key)
    if not valid:
        raise HTTPException(status_code=400, detail=validation_error)

    try:
        fingerprint = _ssh_key_fingerprint(public_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    existing = await prisma.db.litellm_usersshkey.find_unique(
        where={"fingerprint": fingerprint}
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="SSH key already registered")

    created = await prisma.db.litellm_usersshkey.create(
        data={
            "user_id": user_id,
            "public_key": public_key,
            "fingerprint": fingerprint,
            "name": name,
        }
    )
    return PlaygroundSSHKeyResponse(**created.model_dump())


@router.delete(
    "/playground/ssh-keys/{ssh_key_id}",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def delete_playground_ssh_key(
    request: Request,
    ssh_key_id: str,
    target_user_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Delete an SSH key. Only the owner or PROXY_ADMIN may delete."""
    prisma = _get_prisma()

    key_row = await prisma.db.litellm_usersshkey.find_unique(
        where={"ssh_key_id": ssh_key_id}
    )
    if key_row is None:
        raise HTTPException(status_code=404, detail="SSH key not found")

    # Admins skip the ownership check; non-admins can only delete their own.
    if not _is_admin_write(user_api_key_dict):
        caller_id = _caller_user_id(user_api_key_dict)
        if key_row.user_id != caller_id:
            raise HTTPException(status_code=403, detail="not your SSH key")

    await prisma.db.litellm_usersshkey.delete(where={"ssh_key_id": ssh_key_id})
    return {"success": True}


# ---- admin: nodes ---------------------------------------------------------


@router.get(
    "/playground/admin/nodes",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[PlaygroundNodeResponse],
)
@management_endpoint_wrapper
async def admin_list_playground_nodes(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> List[PlaygroundNodeResponse]:
    _require_admin(user_api_key_dict)
    prisma = _get_prisma()
    nodes = await prisma.db.litellm_playgroundnode.find_many(
        order={"ip_address": "asc"}
    )
    return [PlaygroundNodeResponse(**n.model_dump()) for n in nodes]


@router.post(
    "/playground/admin/nodes",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundNodeResponse,
)
@management_endpoint_wrapper
async def admin_create_playground_node(
    request: Request,
    data: CreatePlaygroundNodeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundNodeResponse:
    _require_admin_write(user_api_key_dict)
    prisma = _get_prisma()
    # exclude_unset=True so DB defaults apply for fields the caller omitted,
    # matching the update path's semantics.
    created = await prisma.db.litellm_playgroundnode.create(
        data=data.model_dump(exclude_unset=True)
    )
    verbose_proxy_logger.info(
        f"playground: node {created.node_id} registered ({created.ip_address})"
    )
    return PlaygroundNodeResponse(**created.model_dump())


@router.put(
    "/playground/admin/nodes/{node_id}",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundNodeResponse,
)
@management_endpoint_wrapper
async def admin_update_playground_node(
    request: Request,
    node_id: str,
    data: UpdatePlaygroundNodeRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundNodeResponse:
    _require_admin_write(user_api_key_dict)
    prisma = _get_prisma()
    update_fields = data.model_dump(exclude_unset=True, exclude_none=True)
    if not update_fields:
        raise HTTPException(status_code=400, detail="no update fields provided")

    updated = await prisma.db.litellm_playgroundnode.update(
        where={"node_id": node_id}, data=update_fields
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="node not found")
    return PlaygroundNodeResponse(**updated.model_dump())


@router.delete(
    "/playground/admin/nodes/{node_id}",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def admin_delete_playground_node(
    request: Request,
    node_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    _require_admin_write(user_api_key_dict)
    prisma = _get_prisma()
    existing = await prisma.db.litellm_playgroundnode.find_unique(
        where={"node_id": node_id}
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="node not found")
    # ON DELETE RESTRICT at the DB layer will reject this if there are any
    # bookings referencing the node's ip_address — that's the intended guard.
    try:
        await prisma.db.litellm_playgroundnode.delete(where={"node_id": node_id})
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete node with existing bookings: {e}",
        )
    return {"success": True}


@router.get(
    "/playground/admin/status",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundAdminStatusResponse,
)
@management_endpoint_wrapper
async def admin_playground_status(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundAdminStatusResponse:
    _require_admin(user_api_key_dict)
    prisma = _get_prisma()
    tonight = _tonight_date()

    nodes = await prisma.db.litellm_playgroundnode.find_many(
        order={"ip_address": "asc"}
    )
    # Exclude cancelled bookings from dashboard totals — a cancelled row
    # is not a capacity reservation and shouldn't contaminate the metric.
    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={
            "night_of": tonight,
            "status": {"not": "cancelled"},
        }
    )

    by_node: Dict[str, int] = {}
    for b in bookings:
        by_node[b.allocated_node] = by_node.get(b.allocated_node, 0) + 1

    return PlaygroundAdminStatusResponse(
        night_of=tonight,
        booking_phase=_booking_phase(),
        total_bookings_tonight=len(bookings),
        nodes=[PlaygroundNodeResponse(**n.model_dump()) for n in nodes],
        bookings_by_node=by_node,
    )


# ---- internal (cron) ------------------------------------------------------


@router.get(
    "/playground/internal/allocations-tonight",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PlaygroundAllocationsTonightResponse,
)
@management_endpoint_wrapper
async def get_playground_allocations_tonight(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PlaygroundAllocationsTonightResponse:
    """Cron-only: tonight's allocations grouped by node, with each booking's
    first-registered SSH pubkey for the target user."""
    _require_admin(user_api_key_dict)
    prisma = _get_prisma()
    tonight = _tonight_date()

    # Pull allocated bookings for tonight
    bookings = await prisma.db.litellm_playgroundbooking.find_many(
        where={"night_of": tonight, "status": "allocated"},
        order={"created_at": "asc"},
    )

    # Index nodes by ip_address so we can return the ssh_user / model_path etc.
    node_rows = await prisma.db.litellm_playgroundnode.find_many()
    node_by_ip = {n.ip_address: n for n in node_rows}

    # Batch-fetch SSH keys + user rows up-front so the per-booking loop
    # doesn't issue 2×N round-trips (was an N+1 that scaled badly as
    # booking count grew).
    distinct_user_ids = list({b.user_id for b in bookings})
    first_ssh_key_by_user: Dict[str, str] = {}
    user_email_by_id: Dict[str, Optional[str]] = {}

    if distinct_user_ids:
        all_keys = await prisma.db.litellm_usersshkey.find_many(
            where={"user_id": {"in": distinct_user_ids}},
            order={"created_at": "asc"},
        )
        # `order asc` means the first key seen per user is the earliest one.
        for k in all_keys:
            first_ssh_key_by_user.setdefault(k.user_id, k.public_key)

        user_rows = await prisma.db.litellm_usertable.find_many(
            where={"user_id": {"in": distinct_user_ids}},
        )
        for u in user_rows:
            user_email_by_id[u.user_id] = getattr(u, "user_email", None)

    # Group by node
    grouped: Dict[str, PlaygroundAllocationNode] = {}
    for b in bookings:
        node = node_by_ip.get(b.allocated_node)
        if node is None:
            verbose_proxy_logger.warning(
                f"playground: booking {b.booking_id} references unknown node "
                f"{b.allocated_node}; skipping"
            )
            continue

        if b.allocated_node not in grouped:
            grouped[b.allocated_node] = PlaygroundAllocationNode(
                node_ip=node.ip_address,
                ssh_user=node.ssh_user,
                vllm_container_name=node.vllm_container_name,
                model_path=node.model_path,
                bookings=[],
            )

        grouped[b.allocated_node].bookings.append(
            PlaygroundAllocationBooking(
                booking_id=b.booking_id,
                user_id=b.user_id,
                user_email=user_email_by_id.get(b.user_id),
                gpu_devices=b.allocated_gpus,
                ssh_public_key=first_ssh_key_by_user.get(b.user_id, ""),
            )
        )

    return PlaygroundAllocationsTonightResponse(
        night_of=tonight,
        nodes=list(grouped.values()),
    )


@router.post(
    "/playground/internal/activation-status",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ActivationStatusReportResponse,
)
@management_endpoint_wrapper
async def report_playground_activation_status(
    request: Request,
    data: ActivationStatusReportRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ActivationStatusReportResponse:
    """Cron reports activation results — one entry per booking it attempted."""
    _require_admin_write(user_api_key_dict)
    prisma = _get_prisma()

    updated_rows: List[PlaygroundBookingResponse] = []
    for item in data.results:
        update_data: Dict[str, Any] = {"status": item.status}
        if item.container_id is not None:
            update_data["container_id"] = item.container_id

        # Surface the cron's error field in the logs so ops can see why
        # a specific booking failed without trawling the k8s job output.
        if item.status == "activation_failed":
            verbose_proxy_logger.warning(
                f"playground: activation_failed for booking {item.booking_id}"
                f" — error={item.error or 'unspecified'}"
            )

        try:
            row = await prisma.db.litellm_playgroundbooking.update(
                where={"booking_id": item.booking_id},
                data=update_data,
            )
        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.warning(
                f"playground: failed to update booking {item.booking_id}: {e}"
            )
            continue

        if row is not None:
            updated_rows.append(PlaygroundBookingResponse(**row.model_dump()))

    verbose_proxy_logger.info(
        f"playground: activation status updated for {len(updated_rows)} bookings"
    )
    return ActivationStatusReportResponse(
        updated_count=len(updated_rows), bookings=updated_rows
    )


@router.post(
    "/playground/internal/teardown-status",
    tags=["gpu playground"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=TeardownStatusResponse,
)
@management_endpoint_wrapper
async def report_playground_teardown_status(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TeardownStatusResponse:
    """Cron reports teardown completion — every `active` booking is marked
    `terminated`, regardless of `night_of`.

    Why no night_of filter: the teardown cron fires at 09:00 IST on the
    morning *after* the activation night. At that moment `_tonight_date()`
    returns the next calendar day, but the bookings carry `night_of =
    previous day`, so filtering by `night_of = _tonight_date()` would match
    zero rows every single run and active bookings would accumulate
    indefinitely. It is safe to terminate all `active` rows because only
    one night's worth of bookings can ever be `active` at a time (the
    previous day's allocator is the only writer of that status).
    """
    _require_admin_write(user_api_key_dict)
    prisma = _get_prisma()

    active = await prisma.db.litellm_playgroundbooking.find_many(
        where={"status": "active"}
    )

    terminated: List[PlaygroundBookingResponse] = []
    for b in active:
        try:
            row = await prisma.db.litellm_playgroundbooking.update(
                where={"booking_id": b.booking_id},
                data={"status": "terminated"},
            )
        except Exception as e:  # noqa: BLE001
            verbose_proxy_logger.warning(
                f"playground: failed to terminate booking {b.booking_id}: {e}"
            )
            continue

        if row is not None:
            terminated.append(PlaygroundBookingResponse(**row.model_dump()))

    # Report the most recent night_of among terminated rows so the response
    # still carries a meaningful date. Default to today if no rows.
    night_of_reported = (
        max((b.night_of for b in terminated), default=_tonight_date())
    )
    verbose_proxy_logger.info(
        f"playground: teardown complete — {len(terminated)} bookings terminated "
        f"(most recent night_of={night_of_reported})"
    )
    return TeardownStatusResponse(
        night_of=night_of_reported,
        terminated_count=len(terminated),
        bookings=terminated,
    )
