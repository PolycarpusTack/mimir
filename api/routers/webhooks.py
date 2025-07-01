"""
Webhooks Router for Mimir Enterprise API

Handles webhook management for external integrations.
"""

import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import get_current_user, get_db
from ..models import BaseResponse, WebhookCreate, WebhookResponse, WebhookUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=List[WebhookResponse])
async def list_webhooks(
    current_user: Dict[str, Any] = Depends(get_current_user), db: PostgreSQLManager = Depends(get_db)
):
    """List organization webhooks"""
    try:
        organization_id = current_user["organization_id"]

        query = """
        SELECT id, name, url, events, headers, timeout_seconds, retry_attempts,
               retry_backoff_seconds, is_active, last_triggered_at, success_count,
               failure_count, last_error, last_error_at, organization_id, user_id,
               created_at, updated_at
        FROM webhooks 
        WHERE organization_id = %s
        ORDER BY created_at DESC
        """

        webhooks = db.fetch_all(query, (organization_id,))
        return [WebhookResponse(**webhook) for webhook in webhooks]

    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve webhooks")
