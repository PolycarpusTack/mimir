"""
Users Router for Mimir Enterprise API

Handles user management within organizations.
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.auth_manager import AuthManager
from ..dependencies import RequireUsersRead, get_auth_manager, get_current_user
from ..models import BaseResponse, UserCreate, UserResponse, UserUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user), auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Get current user profile"""
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID not available")

        user_data = auth_manager.get_user_by_id(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(**user_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")
