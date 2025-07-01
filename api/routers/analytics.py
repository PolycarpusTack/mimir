"""
Analytics Router for Mimir Enterprise API

Provides analytics endpoints for data insights and reporting.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import RequireAnalyticsRead, get_current_user, get_db
from ..models import AnalyticsRequest, AnalyticsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/overview", response_model=Dict[str, Any])
async def get_analytics_overview(
    days: int = Query(30, ge=1, le=365, description="Number of days for analytics"),
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    db: PostgreSQLManager = Depends(get_db),
):
    """Get analytics overview for the organization"""
    try:
        organization_id = current_user["organization_id"]

        # Get article counts
        article_stats_query = """
        SELECT 
            COUNT(*) as total_articles,
            COUNT(*) FILTER (WHERE scraped_at >= CURRENT_DATE - INTERVAL '%s days') as recent_articles,
            COUNT(DISTINCT source) as unique_sources,
            COUNT(DISTINCT category) FILTER (WHERE category IS NOT NULL) as unique_categories
        FROM articles 
        WHERE organization_id = %s
        """

        stats = db.fetch_one(article_stats_query, (days, organization_id))

        return {
            "organization_id": organization_id,
            "period_days": days,
            "summary": stats or {},
            "generated_at": "2024-01-01T00:00:00Z",  # Would use actual timestamp
        }

    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")
