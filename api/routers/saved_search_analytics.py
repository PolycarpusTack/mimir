"""
Saved Search Analytics Router for Mimir Enterprise API

Provides analytics endpoints for saved search insights and performance.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...db_manager_postgres import PostgreSQLManager
from ...services.saved_search_analytics import SavedSearchAnalytics
from ..dependencies import RequireAnalyticsRead, get_current_user, get_db

logger = logging.getLogger(__name__)
router = APIRouter()


def get_analytics_service(db: PostgreSQLManager = Depends(get_db)) -> SavedSearchAnalytics:
    """Get saved search analytics service"""
    return SavedSearchAnalytics(db)


@router.get("/saved-searches/usage", response_model=Dict[str, Any])
async def get_saved_search_usage(
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    analytics: SavedSearchAnalytics = Depends(get_analytics_service),
):
    """
    Get usage metrics for saved searches

    Returns comprehensive usage statistics including:
    - Execution frequency
    - User engagement
    - Alert performance
    """
    try:
        organization_id = current_user["organization_id"]

        metrics = analytics.get_search_usage_metrics(
            organization_id=organization_id, start_date=start_date, end_date=end_date
        )

        return metrics

    except Exception as e:
        logger.error(f"Error getting saved search usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve usage metrics"
        )


@router.get("/saved-searches/popular", response_model=List[Dict[str, Any]])
async def get_popular_queries(
    limit: int = Query(10, ge=1, le=50, description="Number of queries to return"),
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    analytics: SavedSearchAnalytics = Depends(get_analytics_service),
):
    """
    Get most popular saved search queries

    Returns the most frequently used search queries with:
    - Query components breakdown
    - Usage statistics
    - User adoption metrics
    """
    try:
        organization_id = current_user["organization_id"]

        popular_queries = analytics.get_popular_search_queries(organization_id=organization_id, limit=limit)

        return popular_queries

    except Exception as e:
        logger.error(f"Error getting popular queries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve popular queries"
        )


@router.get("/saved-searches/alerts/effectiveness", response_model=Dict[str, Any])
async def get_alert_effectiveness(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    analytics: SavedSearchAnalytics = Depends(get_analytics_service),
):
    """
    Analyze alert effectiveness and engagement

    Returns metrics about alert performance including:
    - Trigger frequency and patterns
    - Result relevance
    - Notification channel usage
    - Alert health indicators
    """
    try:
        organization_id = current_user["organization_id"]

        effectiveness = analytics.get_alert_effectiveness(organization_id=organization_id, days=days)

        return effectiveness

    except Exception as e:
        logger.error(f"Error analyzing alert effectiveness: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to analyze alert effectiveness"
        )


@router.get("/saved-searches/performance", response_model=Dict[str, Any])
async def get_search_performance(
    search_ids: Optional[List[str]] = Query(None, description="Specific search IDs to analyze"),
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    analytics: SavedSearchAnalytics = Depends(get_analytics_service),
):
    """
    Get performance metrics for saved searches

    Analyzes search execution performance including:
    - Execution time statistics
    - Query complexity analysis
    - Efficiency scoring
    - Performance optimization recommendations
    """
    try:
        organization_id = current_user["organization_id"]

        performance = analytics.get_search_performance_metrics(organization_id=organization_id, search_ids=search_ids)

        return performance

    except Exception as e:
        logger.error(f"Error getting search performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve performance metrics"
        )


@router.get("/saved-searches/tags", response_model=Dict[str, Any])
async def get_tag_analytics(
    current_user: Dict[str, Any] = RequireAnalyticsRead,
    analytics: SavedSearchAnalytics = Depends(get_analytics_service),
):
    """
    Analyze tag usage across saved searches

    Returns insights about tag usage patterns:
    - Tag frequency and popularity
    - Tag co-occurrence patterns
    - Tag usage by search type
    - Tag effectiveness metrics
    """
    try:
        organization_id = current_user["organization_id"]

        tag_analytics = analytics.get_tag_analytics(organization_id)

        return tag_analytics

    except Exception as e:
        logger.error(f"Error analyzing tags: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to analyze tag usage")
