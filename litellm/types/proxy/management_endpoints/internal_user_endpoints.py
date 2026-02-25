from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, EmailStr, field_validator

from litellm.proxy._types import (
    LiteLLM_UserTableWithKeyCount,
    UpdateUserRequest,
    UpdateUserRequestNoUserIDorEmail,
)


class UserListResponse(BaseModel):
    """
    Response model for the user list endpoint
    """

    users: List[LiteLLM_UserTableWithKeyCount]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkUpdateUserRequest(BaseModel):
    """Request for bulk user updates"""

    users: Optional[List[UpdateUserRequest]] = (
        None  # List of specific user update requests
    )
    all_users: Optional[bool] = False  # Flag to update all users
    user_updates: Optional[UpdateUserRequestNoUserIDorEmail] = (
        None  # Updates to apply to all users when all_users=True
    )

    @field_validator("users", "all_users", "user_updates")
    @classmethod
    def validate_request(cls, v, info):
        # Get all field values for validation
        values = info.data if hasattr(info, "data") else {}

        # After all fields are set, validate the combination
        if (
            info.field_name == "user_updates"
        ):  # This is the last field, do validation here
            users = values.get("users")
            all_users = values.get("all_users", False)
            user_updates = v

            # Must specify either users list OR all_users with user_updates
            if not users and not (all_users and user_updates):
                raise ValueError(
                    "Must specify either 'users' for individual updates or 'all_users=True' with 'user_updates' for bulk updates"
                )

            # Cannot specify both users list and all_users
            if users and all_users:
                raise ValueError(
                    "Cannot specify both 'users' and 'all_users=True'. Choose one approach."
                )

        return v


class UserUpdateResult(BaseModel):
    """Result of a single user update operation"""

    user_id: Optional[str] = None
    user_email: Optional[str] = None
    success: bool
    error: Optional[str] = None
    updated_user: Optional[Dict[str, Any]] = None


class BulkUpdateUserResponse(BaseModel):
    """Response for bulk user update operations"""

    results: List[UserUpdateResult]
    total_requested: int
    successful_updates: int
    failed_updates: int


class BatchUpdateUserBudgetRequest(BaseModel):
    """
    Request for batch updating user budgets.
    Supports three modes:
    1. Update all users (target_type: "all")
    2. Update specific users (target_type: "users" with user_emails list)
    3. Update users in teams (target_type: "team" with team_ids list)
    Also supports resetting spend to 0 when reset_spend=True
    """

    target_type: Literal["all", "users", "team"]
    budget_limit: Optional[float] = None
    budget_duration: Optional[str] = None
    user_emails: Optional[List[str]] = None
    team_ids: Optional[List[str]] = None
    reset_spend: bool = False

    @field_validator("budget_limit")
    @classmethod
    def validate_budget_limit(cls, v, info):
        values = info.data if hasattr(info, "data") else {}
        reset_spend = values.get("reset_spend", False)

        # If not resetting spend, budget_limit is required
        if not reset_spend and v is None:
            raise ValueError("budget_limit is required when reset_spend is False")

        # If budget_limit is provided, it must be non-negative
        if v is not None and v < 0:
            raise ValueError("budget_limit must be non-negative")

        return v

    @field_validator("user_emails")
    @classmethod
    def validate_user_emails(cls, v, info):
        values = info.data if hasattr(info, "data") else {}
        target_type = values.get("target_type")

        if target_type == "users" and (not v or len(v) == 0):
            raise ValueError("user_emails is required when target_type is 'users'")

        return v

    @field_validator("team_ids")
    @classmethod
    def validate_team_ids(cls, v, info):
        values = info.data if hasattr(info, "data") else {}
        target_type = values.get("target_type")

        if target_type == "team" and (not v or len(v) == 0):
            raise ValueError("team_ids is required when target_type is 'team'")

        return v


class BatchUpdateUserBudgetResponse(BaseModel):
    """Response for batch update user budget operations"""

    success: bool
    message: str
    affected_rows: int
    budget_limit: Optional[float] = None
    budget_duration: Optional[str] = None
    target_type: str
    reset_spend: bool
