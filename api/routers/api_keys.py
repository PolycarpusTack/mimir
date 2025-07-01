"""
API Keys Router for Mimir Enterprise API

Handles API key management for programmatic access.
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.auth_manager import AuthManager
from ..dependencies import RequireAdmin, get_auth_manager, get_current_user
from ..models import APIKeyCreate, APIKeyCreateResponse, APIKeyResponse, BaseResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: Dict[str, Any] = RequireAdmin,
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """Create a new API key"""
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user["user_id"]

        api_key, key_info = auth_manager.create_api_key(
            organization_id=organization_id,
            user_id=user_id,
            name=api_key_data.name,
            key_type=api_key_data.type.value,
            expires_in_days=api_key_data.expires_in_days,
            permissions=api_key_data.permissions,
        )

        return APIKeyCreateResponse(api_key=api_key, key_info=APIKeyResponse(**key_info))

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")


@router.get("/", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: Dict[str, Any] = RequireAdmin, auth_manager: AuthManager = Depends(get_auth_manager)
):
    """List organization API keys"""
    try:
        organization_id = current_user["organization_id"]

        keys = auth_manager.api_key_manager.list_api_keys(organization_id)
        return [APIKeyResponse(**key.to_dict()) for key in keys]

    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys")
