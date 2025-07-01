"""
Notification API Router for Mimir Enterprise

Provides endpoints for managing notification templates, preferences, and sending notifications.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...auth.auth_manager import AuthManager
from ...services.notification_service import (
    NotificationChannel,
    NotificationPriority,
    NotificationService,
    NotificationTemplate,
)
from ..dependencies import get_auth_manager, get_current_user, get_db_manager
from ..models.common import PaginatedResponse

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


# Request/Response Models
class NotificationTemplateCreate(BaseModel):
    """Create notification template request"""

    name: str = Field(..., min_length=1, max_length=200)
    channel: NotificationChannel
    notification_type: str = Field(..., min_length=1, max_length=100)
    subject_template: Optional[str] = None
    body_template: str = Field(..., min_length=1)
    variables: Optional[List[str]] = None
    is_active: bool = True


class NotificationTemplateUpdate(BaseModel):
    """Update notification template request"""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    subject_template: Optional[str] = None
    body_template: Optional[str] = Field(None, min_length=1)
    is_active: Optional[bool] = None


class NotificationTemplateResponse(BaseModel):
    """Notification template response"""

    id: UUID
    organization_id: UUID
    name: str
    channel: NotificationChannel
    notification_type: str
    subject_template: Optional[str]
    body_template: str
    variables: List[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class NotificationPreference(BaseModel):
    """User notification preference"""

    channel: NotificationChannel
    notification_type: str
    enabled: bool = True
    settings: Dict[str, Any] = Field(default_factory=dict)


class NotificationChannelConfig(BaseModel):
    """Channel configuration request"""

    channel: NotificationChannel
    enabled: bool = True
    configuration: Dict[str, Any]
    rate_limit: Optional[int] = Field(None, ge=1, le=10000)
    retry_count: Optional[int] = Field(None, ge=0, le=10)
    retry_delay: Optional[int] = Field(None, ge=1, le=3600)


class SendNotificationRequest(BaseModel):
    """Send notification request"""

    channel: NotificationChannel
    recipient: str = Field(..., min_length=1, max_length=500)
    notification_type: str = Field(..., min_length=1, max_length=100)
    data: Dict[str, Any]
    template_id: Optional[UUID] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM


class NotificationHistoryItem(BaseModel):
    """Notification history item"""

    id: UUID
    channel: NotificationChannel
    recipient: str
    notification_type: str
    subject: Optional[str]
    status: str
    priority: NotificationPriority
    error_message: Optional[str]
    sent_at: Optional[datetime]
    created_at: datetime


# Template Management Endpoints
@router.post("/templates", response_model=NotificationTemplateResponse)
async def create_notification_template(
    template: NotificationTemplateCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Create a new notification template"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:templates:create"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Create template
        query = """
        INSERT INTO notification_templates (
            organization_id, name, channel, notification_type,
            subject_template, body_template, variables, is_active, created_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """

        result = db_manager.fetch_one(
            query,
            (
                current_user["organization_id"],
                template.name,
                template.channel.value,
                template.notification_type,
                template.subject_template,
                template.body_template,
                template.variables or [],
                template.is_active,
                current_user["id"],
            ),
        )

        return NotificationTemplateResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates", response_model=PaginatedResponse[NotificationTemplateResponse])
async def list_notification_templates(
    channel: Optional[NotificationChannel] = None,
    notification_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """List notification templates"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:templates:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Build query
        conditions = ["organization_id = %s"]
        params = [current_user["organization_id"]]

        if channel:
            conditions.append("channel = %s")
            params.append(channel.value)

        if notification_type:
            conditions.append("notification_type = %s")
            params.append(notification_type)

        if is_active is not None:
            conditions.append("is_active = %s")
            params.append(is_active)

        # Count total
        count_query = f"SELECT COUNT(*) as total FROM notification_templates WHERE {' AND '.join(conditions)}"
        total_result = db_manager.fetch_one(count_query, params)
        total = total_result["total"]

        # Get templates
        offset = (page - 1) * page_size
        query = f"""
        SELECT * FROM notification_templates
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """

        params.extend([page_size, offset])
        results = db_manager.fetch_all(query, params)

        templates = [NotificationTemplateResponse(**r) for r in results]

        return PaginatedResponse(
            items=templates, total=total, page=page, page_size=page_size, pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates/{template_id}", response_model=NotificationTemplateResponse)
async def get_notification_template(
    template_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get a notification template by ID"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:templates:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    query = """
    SELECT * FROM notification_templates
    WHERE id = %s AND organization_id = %s
    """

    result = db_manager.fetch_one(query, (str(template_id), current_user["organization_id"]))

    if not result:
        raise HTTPException(status_code=404, detail="Template not found")

    return NotificationTemplateResponse(**result)


@router.put("/templates/{template_id}", response_model=NotificationTemplateResponse)
async def update_notification_template(
    template_id: UUID,
    template_update: NotificationTemplateUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Update a notification template"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:templates:update"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Build update query
        update_fields = []
        params = []

        update_dict = template_update.dict(exclude_unset=True)
        for field, value in update_dict.items():
            update_fields.append(f"{field} = %s")
            params.append(value)

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Add updated_at
        update_fields.append("updated_at = %s")
        params.append(datetime.now(timezone.utc))

        # Add WHERE clause params
        params.extend([str(template_id), current_user["organization_id"]])

        query = f"""
        UPDATE notification_templates
        SET {', '.join(update_fields)}
        WHERE id = %s AND organization_id = %s
        RETURNING *
        """

        result = db_manager.fetch_one(query, params)

        if not result:
            raise HTTPException(status_code=404, detail="Template not found")

        return NotificationTemplateResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/templates/{template_id}")
async def delete_notification_template(
    template_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Delete a notification template"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:templates:delete"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    query = """
    DELETE FROM notification_templates
    WHERE id = %s AND organization_id = %s
    """

    result = db_manager.execute_query(query, (str(template_id), current_user["organization_id"]))

    if not result:
        raise HTTPException(status_code=404, detail="Template not found")

    return {"message": "Template deleted successfully"}


# Preference Management Endpoints
@router.get("/preferences", response_model=List[NotificationPreference])
async def get_notification_preferences(
    current_user: Dict[str, Any] = Depends(get_current_user), db_manager=Depends(get_db_manager)
):
    """Get current user's notification preferences"""
    query = """
    SELECT channel, notification_type, enabled, settings
    FROM notification_preferences
    WHERE user_id = %s
    ORDER BY channel, notification_type
    """

    results = db_manager.fetch_all(query, (current_user["id"],))

    return [NotificationPreference(**r) for r in results]


@router.put("/preferences")
async def update_notification_preferences(
    preferences: List[NotificationPreference],
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager=Depends(get_db_manager),
):
    """Update notification preferences"""
    try:
        # Delete existing preferences
        db_manager.execute_query("DELETE FROM notification_preferences WHERE user_id = %s", (current_user["id"],))

        # Insert new preferences
        for pref in preferences:
            query = """
            INSERT INTO notification_preferences (
                organization_id, user_id, channel, notification_type, enabled, settings
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """

            db_manager.execute_query(
                query,
                (
                    current_user["organization_id"],
                    current_user["id"],
                    pref.channel.value,
                    pref.notification_type,
                    pref.enabled,
                    pref.settings,
                ),
            )

        return {"message": "Preferences updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Channel Configuration Endpoints
@router.post("/channels/configure")
async def configure_notification_channel(
    config: NotificationChannelConfig,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Configure a notification channel for the organization"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:channels:configure"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        query = """
        INSERT INTO notification_channels (
            organization_id, channel, enabled, configuration,
            rate_limit, retry_count, retry_delay
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (organization_id, channel)
        DO UPDATE SET
            enabled = EXCLUDED.enabled,
            configuration = EXCLUDED.configuration,
            rate_limit = EXCLUDED.rate_limit,
            retry_count = EXCLUDED.retry_count,
            retry_delay = EXCLUDED.retry_delay,
            updated_at = CURRENT_TIMESTAMP
        """

        db_manager.execute_query(
            query,
            (
                current_user["organization_id"],
                config.channel.value,
                config.enabled,
                config.configuration,
                config.rate_limit,
                config.retry_count,
                config.retry_delay,
            ),
        )

        return {"message": f"Channel {config.channel.value} configured successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/channels", response_model=List[NotificationChannelConfig])
async def list_notification_channels(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """List configured notification channels"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:channels:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    query = """
    SELECT channel, enabled, configuration, rate_limit, retry_count, retry_delay
    FROM notification_channels
    WHERE organization_id = %s
    ORDER BY channel
    """

    results = db_manager.fetch_all(query, (current_user["organization_id"],))

    return [NotificationChannelConfig(**r) for r in results]


# Send Notification Endpoints
@router.post("/send")
async def send_notification(
    request: SendNotificationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Send a notification"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:send"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Initialize notification service
        notification_service = NotificationService(db_manager)

        # Load channel configuration
        channel_query = """
        SELECT * FROM notification_channels
        WHERE organization_id = %s AND channel = %s AND enabled = true
        """

        channel_config = db_manager.fetch_one(channel_query, (current_user["organization_id"], request.channel.value))

        if not channel_config:
            raise HTTPException(status_code=400, detail=f"Channel {request.channel.value} not configured or disabled")

        # Configure the channel
        notification_service.configure_channel(
            channel=request.channel,
            config=channel_config["configuration"],
            enabled=True,
            rate_limit=channel_config["rate_limit"],
        )

        # Send notification
        success = await notification_service.send_notification(
            channel=request.channel,
            recipient=request.recipient,
            notification_type=request.notification_type,
            data=request.data,
            template_id=str(request.template_id) if request.template_id else None,
            priority=request.priority,
        )

        if success:
            return {"message": "Notification sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send notification")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/send/bulk")
async def send_bulk_notifications(
    requests: List[SendNotificationRequest],
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Send multiple notifications"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:send"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 notifications per request")

    try:
        # Queue notifications
        results = []
        for request in requests:
            query = """
            INSERT INTO notification_queue (
                organization_id, channel, recipient, notification_type,
                template_id, data, priority, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued')
            RETURNING id
            """

            result = db_manager.fetch_one(
                query,
                (
                    current_user["organization_id"],
                    request.channel.value,
                    request.recipient,
                    request.notification_type,
                    str(request.template_id) if request.template_id else None,
                    request.data,
                    request.priority.value,
                ),
            )

            results.append({"queue_id": result["id"], "channel": request.channel.value, "recipient": request.recipient})

        return {"message": f"Queued {len(results)} notifications", "notifications": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# History Endpoints
@router.get("/history", response_model=PaginatedResponse[NotificationHistoryItem])
async def get_notification_history(
    channel: Optional[NotificationChannel] = None,
    recipient: Optional[str] = None,
    notification_type: Optional[str] = None,
    status: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get notification history"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:history:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Build query
        conditions = ["organization_id = %s", "created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'"]
        params = [current_user["organization_id"], days]

        if channel:
            conditions.append("channel = %s")
            params.append(channel.value)

        if recipient:
            conditions.append("recipient = %s")
            params.append(recipient)

        if notification_type:
            conditions.append("notification_type = %s")
            params.append(notification_type)

        if status:
            conditions.append("status = %s")
            params.append(status)

        # Count total
        count_query = f"SELECT COUNT(*) as total FROM notification_log WHERE {' AND '.join(conditions)}"
        total_result = db_manager.fetch_one(count_query, params)
        total = total_result["total"]

        # Get history
        offset = (page - 1) * page_size
        query = f"""
        SELECT * FROM notification_log
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """

        params.extend([page_size, offset])
        results = db_manager.fetch_all(query, params)

        history = [NotificationHistoryItem(**r) for r in results]

        return PaginatedResponse(
            items=history, total=total, page=page, page_size=page_size, pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_notification_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get notification statistics"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "notifications:statistics:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Overall statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_sent,
            COUNT(*) FILTER (WHERE status = 'sent') as successful,
            COUNT(*) FILTER (WHERE status = 'failed') as failed,
            COUNT(DISTINCT recipient) as unique_recipients,
            COUNT(DISTINCT channel) as channels_used
        FROM notification_log
        WHERE organization_id = %s 
          AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
        """

        overall_stats = db_manager.fetch_one(stats_query, (current_user["organization_id"], days))

        # By channel
        channel_query = """
        SELECT 
            channel,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'sent') as successful,
            COUNT(*) FILTER (WHERE status = 'failed') as failed,
            AVG(CASE WHEN retry_count > 0 THEN retry_count ELSE NULL END) as avg_retries
        FROM notification_log
        WHERE organization_id = %s 
          AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY channel
        ORDER BY total DESC
        """

        channel_stats = db_manager.fetch_all(channel_query, (current_user["organization_id"], days))

        # By notification type
        type_query = """
        SELECT 
            notification_type,
            COUNT(*) as total,
            COUNT(DISTINCT recipient) as unique_recipients
        FROM notification_log
        WHERE organization_id = %s 
          AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY notification_type
        ORDER BY total DESC
        LIMIT 10
        """

        type_stats = db_manager.fetch_all(type_query, (current_user["organization_id"], days))

        # Daily trend
        trend_query = """
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE status = 'sent') as successful
        FROM notification_log
        WHERE organization_id = %s 
          AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY DATE(created_at)
        ORDER BY date DESC
        """

        daily_trend = db_manager.fetch_all(trend_query, (current_user["organization_id"], days))

        return {
            "period_days": days,
            "overall": overall_stats,
            "by_channel": channel_stats,
            "by_type": type_stats,
            "daily_trend": daily_trend,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
