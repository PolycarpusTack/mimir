"""
Organizations Router for Mimir Enterprise API

Handles organization management and settings.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from ...auth.auth_manager import AuthManager
from ..dependencies import get_auth_manager, get_current_user
from ..models import OrganizationResponse, OrganizationUsage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/current", response_model=OrganizationResponse)
async def get_current_organization(
    current_user: Dict[str, Any] = Depends(get_current_user), auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Get current organization information"""
    try:
        organization_id = current_user["organization_id"]
        org_data = auth_manager.get_organization(organization_id)

        if not org_data:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Get usage statistics
        usage_stats = auth_manager.get_organization_usage(organization_id)

        response_data = dict(org_data)
        if usage_stats:
            response_data["usage"] = OrganizationUsage(**usage_stats)

        return OrganizationResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting organization: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve organization")
