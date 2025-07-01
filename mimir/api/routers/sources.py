"""Sources API router."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...core.services import SourceService, ArticleService
from ...core.models.source import SourceCategory
from ..dependencies import (
    get_source_service,
    get_article_service,
    get_current_user,
    get_pagination_params,
    PaginationParams,
    RequireReadSources,
    RequireWriteSources
)
from ..schemas import (
    SourceResponse,
    SourceCreate,
    SourceUpdate,
    SourceStatistics,
    SourceHealthReport,
    ArticleResponse,
    PaginatedResponse,
    PaginationMeta,
    BaseResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=List[SourceResponse])
async def list_sources(
    enabled_only: bool = Query(False, description="Only return enabled sources"),
    category: Optional[SourceCategory] = Query(None, description="Filter by category"),
    user: Dict[str, Any] = RequireReadSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    List all sources with optional filtering.
    
    - **enabled_only**: Only return enabled sources
    - **category**: Filter sources by category
    """
    try:
        sources = await source_service.get_all_sources(
            enabled_only=enabled_only,
            category=category
        )
        
        return [SourceResponse.model_validate(source) for source in sources]
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sources"
        )


@router.get("/health", response_model=SourceHealthReport)
async def get_sources_health(
    user: Dict[str, Any] = RequireReadSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Get health report for all sources.
    
    Returns statistics about source status, errors, and performance.
    """
    try:
        health_report = await source_service.get_sources_health_report()
        return SourceHealthReport(**health_report)
        
    except Exception as e:
        logger.error(f"Error getting sources health report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sources health report"
        )


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: UUID,
    user: Dict[str, Any] = RequireReadSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Get a specific source by ID.
    
    Returns detailed source information.
    """
    try:
        source = await source_service.get_by_id(source_id)
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return SourceResponse.model_validate(source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source"
        )


@router.get("/{source_id}/statistics", response_model=SourceStatistics)
async def get_source_statistics(
    source_id: UUID,
    user: Dict[str, Any] = RequireReadSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Get statistics for a specific source.
    
    Returns detailed statistics about the source's performance and content.
    """
    try:
        stats = await source_service.get_source_statistics(source_id)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return SourceStatistics(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source statistics for {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source statistics"
        )


@router.get("/{source_id}/articles", response_model=PaginatedResponse)
async def get_source_articles(
    source_id: UUID,
    pagination: PaginationParams = Depends(get_pagination_params),
    days: int = Query(30, ge=1, le=365, description="Days back to search"),
    user: Dict[str, Any] = RequireReadSources,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Get articles from a specific source.
    
    - **source_id**: ID of the source
    - **days**: How many days back to search for articles
    - **page**: Page number (starting from 1)
    - **limit**: Number of articles per page (max 100)
    """
    try:
        articles = await article_service.get_articles_by_source(
            source_id=source_id,
            limit=pagination.limit + 1,  # Get one extra to check if there's a next page
            days=days
        )
        
        # Check if there are more pages
        has_next = len(articles) > pagination.limit
        if has_next:
            articles = articles[:-1]  # Remove the extra article
        
        # Calculate total pages (approximation for performance)
        total_articles = len(articles) + (pagination.offset if pagination.page > 1 else 0)
        if has_next:
            total_articles += 1
        
        total_pages = max(1, (total_articles + pagination.limit - 1) // pagination.limit)
        
        meta = PaginationMeta(
            page=pagination.page,
            limit=pagination.limit,
            total=total_articles,
            total_pages=total_pages,
            has_next=has_next,
            has_previous=pagination.page > 1
        )
        
        return PaginatedResponse(
            data=[ArticleResponse.model_validate(article) for article in articles],
            meta=meta
        )
        
    except Exception as e:
        logger.error(f"Error getting articles for source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source articles"
        )


@router.post("/", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def create_source(
    source_data: SourceCreate,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Create a new source.
    
    - **name**: Human-readable name for the source
    - **url**: Base URL of the source
    - **source_type**: Type of source (RSS, HTML, or API)
    - **category**: Source category (general, broadcast, technology, etc.)
    - **rss_url**: RSS feed URL (for RSS sources)
    - **scraping_config**: Configuration for scraping behavior
    """
    try:
        source = await source_service.create_source(
            name=source_data.name,
            url=str(source_data.url),
            source_type=source_data.source_type,
            category=source_data.category,
            description=source_data.description,
            rss_url=str(source_data.rss_url) if source_data.rss_url else None,
            scraping_config=source_data.scraping_config,
            max_articles_per_run=source_data.max_articles_per_run,
            request_delay=source_data.request_delay,
        )
        
        return SourceResponse.model_validate(source)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create source"
        )


@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: UUID,
    source_update: SourceUpdate,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Update an existing source.
    
    Updates the specified source with the provided data.
    Only fields that are provided will be updated.
    """
    try:
        # Convert Pydantic model to dict, excluding unset fields
        updates = source_update.model_dump(exclude_unset=True)
        
        # Convert URLs to strings if present
        if 'url' in updates:
            updates['url'] = str(updates['url'])
        if 'rss_url' in updates:
            updates['rss_url'] = str(updates['rss_url'])
        
        source = await source_service.update_source(source_id, **updates)
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return SourceResponse.model_validate(source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update source"
        )


@router.delete("/{source_id}", response_model=BaseResponse)
async def delete_source(
    source_id: UUID,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Delete a source.
    
    Permanently deletes the specified source and all associated data.
    This action cannot be undone.
    """
    try:
        success = await source_service.delete_source(source_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return BaseResponse(
            success=True,
            message="Source deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete source"
        )


@router.post("/{source_id}/enable", response_model=BaseResponse)
async def enable_source(
    source_id: UUID,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Enable a source.
    
    Enables the source for scraping operations.
    """
    try:
        success = await source_service.enable_source(source_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return BaseResponse(
            success=True,
            message="Source enabled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable source"
        )


@router.post("/{source_id}/disable", response_model=BaseResponse)
async def disable_source(
    source_id: UUID,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Disable a source.
    
    Disables the source from scraping operations.
    """
    try:
        success = await source_service.disable_source(source_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return BaseResponse(
            success=True,
            message="Source disabled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable source"
        )


@router.post("/{source_id}/test", response_model=Dict[str, Any])
async def test_source_connection(
    source_id: UUID,
    user: Dict[str, Any] = RequireWriteSources,
    source_service: SourceService = Depends(get_source_service)
):
    """
    Test connection to a source.
    
    Tests if the source is accessible and can be scraped successfully.
    """
    try:
        result = await source_service.test_source_connection(source_id)
        return result
        
    except Exception as e:
        logger.error(f"Error testing source connection {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test source connection"
        )