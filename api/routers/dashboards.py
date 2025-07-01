"""
Dashboard Router for Mimir Enterprise API

Provides endpoints for managing custom dashboards and widgets.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...auth.auth_manager import AuthManager
from ...db_manager_postgres import PostgreSQLManager
from ...services.dashboard_manager import DashboardManager
from ...services.widget_data_service import WidgetDataService
from ..dependencies import get_auth_manager, get_current_user, get_db, require_organization_access
from ..models import BaseResponse
from ..models.dashboard import (
    DashboardCreate,
    DashboardResponse,
    DashboardShareRequest,
    DashboardTemplateResponse,
    DashboardUpdate,
    DashboardWithWidgets,
    WidgetCreate,
    WidgetDataResponse,
    WidgetResponse,
    WidgetUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def get_dashboard_manager(db: PostgreSQLManager = Depends(get_db)) -> DashboardManager:
    """Get dashboard manager instance"""
    return DashboardManager(db)


def get_widget_data_service(db: PostgreSQLManager = Depends(get_db)) -> WidgetDataService:
    """Get widget data service instance"""
    return WidgetDataService(db)


@router.get("/", response_model=List[DashboardResponse])
async def list_dashboards(
    include_shared: bool = Query(True, description="Include shared dashboards"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    List user's dashboards

    Returns all dashboards owned by or shared with the user.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        dashboards = dashboard_manager.list_user_dashboards(
            organization_id=organization_id, user_id=user_id, include_shared=include_shared
        )

        return [
            DashboardResponse(
                id=d.id,
                name=d.name,
                description=d.description,
                layout=d.layout,
                theme=d.theme,
                is_default=d.is_default,
                is_public=d.is_public,
                tags=d.tags,
                settings=d.settings,
                organization_id=d.organization_id,
                user_id=d.user_id,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in dashboards
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing dashboards: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve dashboards")


@router.get("/templates", response_model=List[DashboardTemplateResponse])
async def list_dashboard_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    List available dashboard templates

    Returns pre-configured dashboard templates that can be used to create new dashboards.
    """
    try:
        templates = dashboard_manager.get_dashboard_templates(category)

        return [
            DashboardTemplateResponse(
                id=t["id"],
                name=t["name"],
                description=t["description"],
                preview_image=t.get("preview_image"),
                category=t["category"],
                widgets=t["widgets"],
                tags=t["tags"],
            )
            for t in templates
        ]

    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve templates")


@router.get("/{dashboard_id}", response_model=DashboardWithWidgets)
async def get_dashboard(
    dashboard_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    Get dashboard with widgets

    Returns dashboard configuration and all associated widgets.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        result = dashboard_manager.get_dashboard(str(dashboard_id), user_id, include_widgets=True)

        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found or not accessible")

        dashboard, widgets = result

        return DashboardWithWidgets(
            id=dashboard.id,
            name=dashboard.name,
            description=dashboard.description,
            layout=dashboard.layout,
            theme=dashboard.theme,
            is_default=dashboard.is_default,
            is_public=dashboard.is_public,
            tags=dashboard.tags,
            settings=dashboard.settings,
            organization_id=dashboard.organization_id,
            user_id=dashboard.user_id,
            created_at=dashboard.created_at,
            updated_at=dashboard.updated_at,
            widgets=[
                WidgetResponse(
                    id=w.id,
                    dashboard_id=w.dashboard_id,
                    type=w.type,
                    title=w.title,
                    size=w.size,
                    refresh_interval=w.refresh_interval,
                    data_source=w.data_source,
                    display_options=w.display_options,
                    position=w.position,
                    created_at=w.created_at,
                    updated_at=w.updated_at,
                )
                for w in widgets
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve dashboard")


@router.post("/", response_model=DashboardResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/hour")
async def create_dashboard(
    request,
    dashboard: DashboardCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Create a new dashboard

    Creates a dashboard with optional initial widgets.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        created = dashboard_manager.create_dashboard(
            organization_id=organization_id,
            user_id=user_id,
            name=dashboard.name,
            description=dashboard.description,
            layout=dashboard.layout,
            theme=dashboard.theme,
            is_default=dashboard.is_default,
            is_public=dashboard.is_public,
            tags=dashboard.tags,
            widgets=dashboard.widgets,
            settings=dashboard.settings,
        )

        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=user_id,
            action="create_dashboard",
            resource_type="dashboard",
            resource_id=created.id,
            details={"name": dashboard.name, "widget_count": len(dashboard.widgets)},
        )

        return DashboardResponse(
            id=created.id,
            name=created.name,
            description=created.description,
            layout=created.layout,
            theme=created.theme,
            is_default=created.is_default,
            is_public=created.is_public,
            tags=created.tags,
            settings=created.settings,
            organization_id=created.organization_id,
            user_id=created.user_id,
            created_at=created.created_at,
            updated_at=created.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating dashboard: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create dashboard")


@router.post("/from-template", response_model=DashboardResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
async def create_dashboard_from_template(
    request,
    template_id: str = Query(..., description="Template ID"),
    name: str = Query(..., description="Dashboard name"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    Create dashboard from template

    Creates a new dashboard based on a pre-configured template.
    """
    try:
        organization_id = current_user["organization_id"]
        user_id = current_user.get("user_id")

        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        created = dashboard_manager.create_dashboard_from_template(
            organization_id=organization_id, user_id=user_id, template_id=template_id, name=name
        )

        if not created:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")

        return DashboardResponse(
            id=created.id,
            name=created.name,
            description=created.description,
            layout=created.layout,
            theme=created.theme,
            is_default=created.is_default,
            is_public=created.is_public,
            tags=created.tags,
            settings=created.settings,
            organization_id=created.organization_id,
            user_id=created.user_id,
            created_at=created.created_at,
            updated_at=created.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating from template: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create dashboard from template"
        )


@router.put("/{dashboard_id}", response_model=DashboardResponse)
async def update_dashboard(
    dashboard_id: UUID,
    dashboard_update: DashboardUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Update dashboard configuration

    Updates dashboard properties. Only the owner can update a dashboard.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Build update dict
        updates = {}
        if dashboard_update.name is not None:
            updates["name"] = dashboard_update.name
        if dashboard_update.description is not None:
            updates["description"] = dashboard_update.description
        if dashboard_update.layout is not None:
            updates["layout"] = dashboard_update.layout
        if dashboard_update.theme is not None:
            updates["theme"] = dashboard_update.theme
        if dashboard_update.is_default is not None:
            updates["is_default"] = dashboard_update.is_default
        if dashboard_update.is_public is not None:
            updates["is_public"] = dashboard_update.is_public
        if dashboard_update.tags is not None:
            updates["tags"] = dashboard_update.tags
        if dashboard_update.settings is not None:
            updates["settings"] = dashboard_update.settings

        if not updates:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

        updated = dashboard_manager.update_dashboard(str(dashboard_id), user_id, **updates)

        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found or cannot be updated"
            )

        # Log activity
        auth_manager._log_activity(
            organization_id=updated.organization_id,
            user_id=user_id,
            action="update_dashboard",
            resource_type="dashboard",
            resource_id=str(dashboard_id),
            details={"updated_fields": list(updates.keys())},
        )

        return DashboardResponse(
            id=updated.id,
            name=updated.name,
            description=updated.description,
            layout=updated.layout,
            theme=updated.theme,
            is_default=updated.is_default,
            is_public=updated.is_public,
            tags=updated.tags,
            settings=updated.settings,
            organization_id=updated.organization_id,
            user_id=updated.user_id,
            created_at=updated.created_at,
            updated_at=updated.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update dashboard")


@router.delete("/{dashboard_id}", response_model=BaseResponse)
async def delete_dashboard(
    dashboard_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Delete a dashboard

    Permanently deletes a dashboard and all associated widgets.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        success = dashboard_manager.delete_dashboard(str(dashboard_id), user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found or cannot be deleted"
            )

        # Log activity
        auth_manager._log_activity(
            organization_id=current_user["organization_id"],
            user_id=user_id,
            action="delete_dashboard",
            resource_type="dashboard",
            resource_id=str(dashboard_id),
        )

        return BaseResponse(success=True, message="Dashboard deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete dashboard")


@router.post("/{dashboard_id}/widgets", response_model=WidgetResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("50/hour")
async def add_widget(
    request,
    dashboard_id: UUID,
    widget: WidgetCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    Add widget to dashboard

    Creates a new widget and adds it to the specified dashboard.
    """
    try:
        # Verify dashboard access
        user_id = current_user.get("user_id")
        dashboard_info = dashboard_manager.get_dashboard(str(dashboard_id), user_id, include_widgets=False)

        if not dashboard_info:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard not found or not accessible")

        created = dashboard_manager.add_widget_to_dashboard(
            dashboard_id=str(dashboard_id),
            widget_type=widget.type,
            title=widget.title,
            size=widget.size,
            refresh_interval=widget.refresh_interval,
            data_source=widget.data_source,
            display_options=widget.display_options,
            position=widget.position,
        )

        if not created:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create widget")

        return WidgetResponse(
            id=created.id,
            dashboard_id=created.dashboard_id,
            type=created.type,
            title=created.title,
            size=created.size,
            refresh_interval=created.refresh_interval,
            data_source=created.data_source,
            display_options=created.display_options,
            position=created.position,
            created_at=created.created_at,
            updated_at=created.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding widget: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add widget")


@router.put("/{dashboard_id}/widgets/{widget_id}", response_model=WidgetResponse)
async def update_widget(
    dashboard_id: UUID,
    widget_id: UUID,
    widget_update: WidgetUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    Update widget configuration

    Updates widget properties within a dashboard.
    """
    try:
        # Build update dict
        updates = {}
        if widget_update.title is not None:
            updates["title"] = widget_update.title
        if widget_update.size is not None:
            updates["size"] = widget_update.size
        if widget_update.refresh_interval is not None:
            updates["refresh_interval"] = widget_update.refresh_interval
        if widget_update.data_source is not None:
            updates["data_source"] = widget_update.data_source
        if widget_update.display_options is not None:
            updates["display_options"] = widget_update.display_options
        if widget_update.position is not None:
            updates["position"] = widget_update.position

        if not updates:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

        updated = dashboard_manager.update_widget(str(widget_id), str(dashboard_id), **updates)

        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Widget not found or cannot be updated")

        return WidgetResponse(
            id=updated.id,
            dashboard_id=updated.dashboard_id,
            type=updated.type,
            title=updated.title,
            size=updated.size,
            refresh_interval=updated.refresh_interval,
            data_source=updated.data_source,
            display_options=updated.display_options,
            position=updated.position,
            created_at=updated.created_at,
            updated_at=updated.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating widget {widget_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update widget")


@router.delete("/{dashboard_id}/widgets/{widget_id}", response_model=BaseResponse)
async def delete_widget(
    dashboard_id: UUID,
    widget_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
):
    """
    Remove widget from dashboard

    Deletes a widget from the specified dashboard.
    """
    try:
        success = dashboard_manager.delete_widget(str(widget_id), str(dashboard_id))

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Widget not found or cannot be deleted")

        return BaseResponse(success=True, message="Widget deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting widget {widget_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete widget")


@router.get("/{dashboard_id}/widgets/{widget_id}/data", response_model=WidgetDataResponse)
async def get_widget_data(
    dashboard_id: UUID,
    widget_id: UUID,
    use_cache: bool = Query(True, description="Use cached data if available"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
    widget_data_service: WidgetDataService = Depends(get_widget_data_service),
):
    """
    Get widget data

    Fetches the current data for a widget based on its configuration.
    """
    try:
        # Get widget configuration
        widgets = dashboard_manager.get_dashboard_widgets(str(dashboard_id))
        widget = next((w for w in widgets if w.id == str(widget_id)), None)

        if not widget:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Widget not found")

        # Fetch widget data
        data = await widget_data_service.get_widget_data(
            widget_id=str(widget_id),
            widget_type=widget.type,
            data_source=widget.data_source,
            organization_id=current_user["organization_id"],
            user_id=current_user.get("user_id"),
            use_cache=use_cache,
        )

        return WidgetDataResponse(**data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting widget data: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve widget data")


@router.post("/{dashboard_id}/share", response_model=BaseResponse)
async def share_dashboard(
    dashboard_id: UUID,
    share_request: DashboardShareRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    dashboard_manager: DashboardManager = Depends(get_dashboard_manager),
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Share dashboard with other users

    Shares a dashboard with specified users or makes it public.
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User ID required")

        # Update public status if requested
        if share_request.is_public:
            dashboard_manager.update_dashboard(str(dashboard_id), user_id, is_public=True)

        # Share with specific users
        if share_request.user_ids:
            success = dashboard_manager.share_dashboard(
                dashboard_id=str(dashboard_id),
                owner_id=user_id,
                shared_with_user_ids=share_request.user_ids,
                permissions=share_request.permissions,
            )

            if not success:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot share dashboard - not owner")

        # Log activity
        auth_manager._log_activity(
            organization_id=current_user["organization_id"],
            user_id=user_id,
            action="share_dashboard",
            resource_type="dashboard",
            resource_id=str(dashboard_id),
            details={"shared_with_count": len(share_request.user_ids), "is_public": share_request.is_public},
        )

        return BaseResponse(success=True, message="Dashboard shared successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sharing dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to share dashboard")
