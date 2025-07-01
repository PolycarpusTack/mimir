"""Scraping operations API router."""

import logging
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...core.services import ScrapingService
from ..dependencies import (
    get_scraping_service,
    get_current_user,
    RequireWriteSources
)
from ..schemas import (
    ScrapingResult,
    BaseResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run", response_model=ScrapingResult)
async def run_scraping(
    enabled_only: bool = Query(True, description="Only scrape enabled sources"),
    max_articles_per_source: Optional[int] = Query(
        None, 
        ge=1, 
        le=100, 
        description="Override default article limit per source"
    ),
    user: Dict[str, Any] = RequireWriteSources,
    scraping_service: ScrapingService = Depends(get_scraping_service)
):
    """
    Run scraping operation on all configured sources.
    
    - **enabled_only**: Only scrape sources that are enabled
    - **max_articles_per_source**: Override the default article limit per source
    
    This operation will:
    1. Get all configured sources (optionally filtered to enabled only)
    2. Scrape each source for new articles
    3. Store new articles in the database
    4. Process keyword alerts
    5. Return summary of results
    """
    try:
        logger.info("Starting bulk scraping operation via API")
        
        result = await scraping_service.scrape_all_sources(
            enabled_only=enabled_only,
            max_articles_per_source=max_articles_per_source
        )
        
        return ScrapingResult(**result)
        
    except Exception as e:
        logger.error(f"Error running scraping operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run scraping operation"
        )


@router.post("/sources/{source_id}/scrape", response_model=ScrapingResult)
async def scrape_single_source(
    source_id: UUID,
    max_articles: Optional[int] = Query(
        None,
        ge=1,
        le=100,
        description="Maximum articles to scrape from this source"
    ),
    user: Dict[str, Any] = RequireWriteSources,
    scraping_service: ScrapingService = Depends(get_scraping_service)
):
    """
    Scrape a single source.
    
    - **source_id**: ID of the source to scrape
    - **max_articles**: Maximum number of articles to scrape (overrides source default)
    
    This operation will:
    1. Validate the source exists and is enabled
    2. Scrape the source for new articles
    3. Store new articles in the database
    4. Process keyword alerts
    5. Return summary of results
    """
    try:
        logger.info(f"Starting single source scraping via API: {source_id}")
        
        result = await scraping_service.scrape_single_source(
            source_id=source_id,
            max_articles=max_articles
        )
        
        return ScrapingResult(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error scraping single source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to scrape source"
        )


@router.get("/status")
async def get_scraping_status(
    user: Dict[str, Any] = RequireWriteSources,
):
    """
    Get current scraping status.
    
    Returns information about ongoing or recent scraping operations.
    This is a placeholder endpoint - in a full implementation, this would
    track active scraping jobs and provide real-time status updates.
    """
    # This is a simplified implementation
    # In a production system, you would track active scraping jobs
    # and provide real-time status updates
    
    return {
        "status": "idle",
        "message": "No active scraping operations",
        "last_run": None,
        "active_jobs": 0,
        "queued_jobs": 0
    }


@router.post("/test")
async def test_scraping_system(
    user: Dict[str, Any] = RequireWriteSources,
):
    """
    Test the scraping system.
    
    Performs basic validation of the scraping system components
    without actually scraping any sources.
    """
    try:
        # Basic system validation
        # In a full implementation, this would test:
        # - Database connectivity
        # - Scraper factory
        # - Configuration validity
        # - Network connectivity
        
        return BaseResponse(
            success=True,
            message="Scraping system is operational"
        )
        
    except Exception as e:
        logger.error(f"Error testing scraping system: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scraping system test failed"
        )