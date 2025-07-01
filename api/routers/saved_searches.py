"""
Complete Saved Searches Router for Mimir Enterprise API

Comprehensive implementation of saved search management with alerts and analytics.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from ...auth.auth_manager import AuthManager
from ...db_manager_postgres import PostgreSQLManager
from ...services.saved_search_manager import AlertFrequency, SavedSearchManager
from ..dependencies import (
    get_auth_manager,
    get_current_user,
    get_db,
    get_pagination_params,
    require_organization_access,
)
from ..models import (
    BaseResponse,
    PaginatedResponse,
    SavedSearchCreate,
    SavedSearchResponse,
    SavedSearchUpdate,
    SearchResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_saved_search_manager(db: PostgreSQLManager = Depends(get_db)) -> SavedSearchManager:
    """Get saved search manager instance"""
    return SavedSearchManager(db)


@router.get("/", response_model=List[SavedSearchResponse])
async def list_saved_searches(
    include_shared: bool = Query(True, description="Include shared searches"),
    only_alerts: bool = Query(False, description="Only return alert searches"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
):
    """
    List user's saved searches

    - **include_shared**: Include searches shared with the user
    - **only_alerts**: Only return searches configured as alerts
    - **tag**: Filter by tag

    Returns list of saved searches accessible to the user.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required for saved searches")

        # Get saved searches
        saved_searches = saved_search_manager.list_user_saved_searches(
            organization_id=organization_id, user_id=user_id, include_shared=include_shared, only_alerts=only_alerts
        )

        # Filter by tag if specified
        if tag:
            saved_searches = [search for search in saved_searches if tag in search.tags]

        # Convert to response models
        return [
            SavedSearchResponse(
                id=search.id,
                name=search.name,
                description=search.description,
                query=search.query,
                is_alert=search.is_alert,
                alert_frequency=search.alert_frequency,
                last_checked_at=search.last_checked_at,
                next_check_at=search.next_check_at,
                notification_channels=search.notification_channels,
                notification_settings=search.notification_settings,
                is_shared=search.is_shared,
                shared_with=search.shared_with,
                tags=search.tags,
                organization_id=search.organization_id,
                user_id=search.user_id,
                created_at=search.created_at,
                updated_at=search.updated_at,
            )
            for search in saved_searches
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing saved searches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve saved searches"
        )


@router.get("/{search_id}", response_model=SavedSearchResponse)
async def get_saved_search(
    search_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
):
    """
    Get a specific saved search by ID

    Returns saved search details if accessible to the user.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        saved_search = saved_search_manager.get_saved_search(str(search_id), user_id)

        if not saved_search:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Saved search not found or not accessible"
            )

        return SavedSearchResponse(
            id=saved_search.id,
            name=saved_search.name,
            description=saved_search.description,
            query=saved_search.query,
            is_alert=saved_search.is_alert,
            alert_frequency=saved_search.alert_frequency,
            last_checked_at=saved_search.last_checked_at,
            next_check_at=saved_search.next_check_at,
            notification_channels=saved_search.notification_channels,
            notification_settings=saved_search.notification_settings,
            is_shared=saved_search.is_shared,
            shared_with=saved_search.shared_with,
            tags=saved_search.tags,
            organization_id=saved_search.organization_id,
            user_id=saved_search.user_id,
            created_at=saved_search.created_at,
            updated_at=saved_search.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting saved search {search_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve saved search")


@router.post("/", response_model=SavedSearchResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("50/hour")
async def create_saved_search(
    request,
    saved_search: SavedSearchCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Create a new saved search

    - **name**: Search name
    - **description**: Optional description
    - **query**: Search query parameters (filters, sorting, etc.)
    - **is_alert**: Configure as alert for notifications
    - **alert_frequency**: How often to check (in seconds)
    - **notification_channels**: List of channels (email, slack, webhook)
    - **notification_settings**: Channel-specific settings
    - **is_shared**: Share with organization members
    - **shared_with**: Specific user IDs to share with
    - **tags**: Optional tags for organization

    Creates a saved search that can be executed later or configured as an alert.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required to create saved search"
            )

        # Validate alert frequency
        if saved_search.is_alert:
            valid_frequencies = [freq.value for freq in AlertFrequency]
            if saved_search.alert_frequency not in valid_frequencies:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid alert frequency. Must be one of: {valid_frequencies}",
                )

        # Create saved search
        created_search = saved_search_manager.create_saved_search(
            organization_id=organization_id,
            user_id=user_id,
            name=saved_search.name,
            query=saved_search.query,
            description=saved_search.description,
            is_alert=saved_search.is_alert,
            alert_frequency=saved_search.alert_frequency,
            notification_channels=saved_search.notification_channels,
            notification_settings=saved_search.notification_settings,
            is_shared=saved_search.is_shared,
            shared_with=saved_search.shared_with,
            tags=saved_search.tags,
        )

        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=user_id,
            action="create_saved_search",
            resource_type="saved_search",
            resource_id=created_search.id,
            details={"name": saved_search.name, "is_alert": saved_search.is_alert, "tags": saved_search.tags},
        )

        return SavedSearchResponse(
            id=created_search.id,
            name=created_search.name,
            description=created_search.description,
            query=created_search.query,
            is_alert=created_search.is_alert,
            alert_frequency=created_search.alert_frequency,
            last_checked_at=created_search.last_checked_at,
            next_check_at=created_search.next_check_at,
            notification_channels=created_search.notification_channels,
            notification_settings=created_search.notification_settings,
            is_shared=created_search.is_shared,
            shared_with=created_search.shared_with,
            tags=created_search.tags,
            organization_id=created_search.organization_id,
            user_id=created_search.user_id,
            created_at=created_search.created_at,
            updated_at=created_search.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating saved search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create saved search")


@router.put("/{search_id}", response_model=SavedSearchResponse)
async def update_saved_search(
    search_id: UUID,
    search_update: SavedSearchUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Update an existing saved search

    Only the owner can update a saved search.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Build update dict
        updates = {}
        if search_update.name is not None:
            updates["name"] = search_update.name
        if search_update.description is not None:
            updates["description"] = search_update.description
        if search_update.query is not None:
            updates["query"] = search_update.query
        if search_update.is_alert is not None:
            updates["is_alert"] = search_update.is_alert
        if search_update.alert_frequency is not None:
            updates["alert_frequency"] = search_update.alert_frequency
        if search_update.notification_channels is not None:
            updates["notification_channels"] = search_update.notification_channels
        if search_update.notification_settings is not None:
            updates["notification_settings"] = search_update.notification_settings
        if search_update.is_shared is not None:
            updates["is_shared"] = search_update.is_shared
        if search_update.shared_with is not None:
            updates["shared_with"] = search_update.shared_with
        if search_update.tags is not None:
            updates["tags"] = search_update.tags

        if not updates:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

        # Update saved search
        updated_search = saved_search_manager.update_saved_search(str(search_id), user_id, **updates)

        if not updated_search:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Saved search not found or cannot be updated"
            )

        # Log activity
        auth_manager._log_activity(
            organization_id=updated_search.organization_id,
            user_id=user_id,
            action="update_saved_search",
            resource_type="saved_search",
            resource_id=str(search_id),
            details={"updated_fields": list(updates.keys())},
        )

        return SavedSearchResponse(
            id=updated_search.id,
            name=updated_search.name,
            description=updated_search.description,
            query=updated_search.query,
            is_alert=updated_search.is_alert,
            alert_frequency=updated_search.alert_frequency,
            last_checked_at=updated_search.last_checked_at,
            next_check_at=updated_search.next_check_at,
            notification_channels=updated_search.notification_channels,
            notification_settings=updated_search.notification_settings,
            is_shared=updated_search.is_shared,
            shared_with=updated_search.shared_with,
            tags=updated_search.tags,
            organization_id=updated_search.organization_id,
            user_id=updated_search.user_id,
            created_at=updated_search.created_at,
            updated_at=updated_search.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating saved search {search_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update saved search")


@router.delete("/{search_id}", response_model=BaseResponse)
async def delete_saved_search(
    search_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Delete a saved search

    Only the owner can delete a saved search.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Get search for logging
        saved_search = saved_search_manager.get_saved_search(str(search_id), user_id)

        # Delete search
        success = saved_search_manager.delete_saved_search(str(search_id), user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Saved search not found or cannot be deleted"
            )

        # Log activity
        if saved_search:
            auth_manager._log_activity(
                organization_id=saved_search.organization_id,
                user_id=user_id,
                action="delete_saved_search",
                resource_type="saved_search",
                resource_id=str(search_id),
                details={"name": saved_search.name},
            )

        return BaseResponse(success=True, message="Saved search deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting saved search {search_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete saved search")


@router.post("/{search_id}/execute", response_model=Dict[str, Any])
@limiter.limit("100/hour")
async def execute_saved_search(
    request,
    search_id: UUID,
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
):
    """
    Execute a saved search

    Runs the saved search query and returns results.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Execute search
        results = saved_search_manager.execute_saved_search(str(search_id), user_id, limit)

        return results

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing saved search {search_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to execute saved search")


@router.post("/{search_id}/test-alert", response_model=Dict[str, Any])
@limiter.limit("10/hour")
async def test_saved_search_alert(
    request,
    search_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
):
    """
    Test a saved search alert

    Executes the alert immediately and sends test notifications.
    This is useful for testing alert configuration.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Get saved search
        saved_search = saved_search_manager.get_saved_search(str(search_id), user_id)

        if not saved_search:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved search not found")

        if not saved_search.is_alert:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Saved search is not configured as an alert"
            )

        # Execute alert in background
        background_tasks.add_task(saved_search_manager.check_alerts)  # This would execute just this alert

        return {
            "status": "success",
            "message": "Alert test initiated",
            "alert_id": str(search_id),
            "alert_name": saved_search.name,
            "notification_channels": saved_search.notification_channels,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing alert {search_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to test alert")


@router.get("/alerts/statistics", response_model=Dict[str, Any])
async def get_alert_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
    auth: Dict[str, Any] = Depends(require_organization_access),
):
    """
    Get alert statistics for the organization

    Returns aggregated statistics about saved search alerts including
    frequency distribution and recent activity.
    """
    try:
        organization_id = current_user["organization_id"]

        stats = saved_search_manager.get_alert_statistics(organization_id, days)

        return stats

    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve alert statistics"
        )


@router.get("/shared", response_model=List[SavedSearchResponse])
async def get_shared_searches(
    current_user: Dict[str, Any] = Depends(get_current_user),
    saved_search_manager: SavedSearchManager = Depends(get_saved_search_manager),
):
    """
    Get all saved searches shared within the organization

    Returns public searches and searches explicitly shared with the user.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        # Get all saved searches for the organization
        all_searches = saved_search_manager.list_user_saved_searches(
            organization_id=organization_id, user_id=user_id, include_shared=True, only_alerts=False
        )

        # Filter only shared searches (not owned by user)
        shared_searches = [
            search
            for search in all_searches
            if search.user_id != user_id and (search.is_shared or user_id in search.shared_with)
        ]

        # Convert to response models
        return [
            SavedSearchResponse(
                id=search.id,
                name=search.name,
                description=search.description,
                query=search.query,
                is_alert=search.is_alert,
                alert_frequency=search.alert_frequency,
                last_checked_at=search.last_checked_at,
                next_check_at=search.next_check_at,
                notification_channels=search.notification_channels,
                notification_settings=search.notification_settings,
                is_shared=search.is_shared,
                shared_with=search.shared_with,
                tags=search.tags,
                organization_id=search.organization_id,
                user_id=search.user_id,
                created_at=search.created_at,
                updated_at=search.updated_at,
            )
            for search in shared_searches
        ]

    except Exception as e:
        logger.error(f"Error getting shared searches: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve shared searches"
        )
