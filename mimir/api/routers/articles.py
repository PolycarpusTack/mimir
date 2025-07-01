"""Articles API router."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...core.services import ArticleService
from ..dependencies import (
    get_article_service,
    get_current_user,
    get_pagination_params,
    PaginationParams,
    RequireReadArticles,
    RequireWriteArticles
)
from ..schemas import (
    ArticleResponse,
    ArticleDetailResponse,
    ArticleCreate,
    ArticleUpdate,
    PaginatedResponse,
    PaginationMeta,
    BaseResponse,
    ArticleStatistics
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=PaginatedResponse)
async def list_articles(
    pagination: PaginationParams = Depends(get_pagination_params),
    hours: Optional[int] = Query(24, ge=1, le=168, description="Hours back to search"),
    broadcast_relevant_only: bool = Query(False, description="Only broadcast relevant articles"),
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    List recent articles with pagination.
    
    - **hours**: How many hours back to look for articles (1-168 hours)
    - **broadcast_relevant_only**: Filter to only broadcast-relevant articles
    - **page**: Page number (starting from 1)
    - **limit**: Number of articles per page (max 100)
    """
    try:
        articles = await article_service.get_recent_articles(
            limit=pagination.limit + 1,  # Get one extra to check if there's a next page
            hours=hours,
            broadcast_relevant_only=broadcast_relevant_only
        )
        
        # Check if there are more pages
        has_next = len(articles) > pagination.limit
        if has_next:
            articles = articles[:-1]  # Remove the extra article
        
        # Calculate total pages (this is an approximation for performance)
        # In a real implementation, you might want to do a separate count query
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
        logger.error(f"Error listing articles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve articles"
        )


@router.get("/featured", response_model=ArticleResponse)
async def get_featured_article(
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Get featured article for hero section.
    
    Returns the most relevant recent article or most recent article as fallback.
    """
    try:
        article = await article_service.get_featured_article()
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No articles found"
            )
        
        return ArticleResponse.model_validate(article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting featured article: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve featured article"
        )


@router.get("/statistics", response_model=ArticleStatistics)
async def get_article_statistics(
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Get article statistics.
    
    Returns various statistics about articles in the system.
    """
    try:
        stats = await article_service.get_statistics()
        return ArticleStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Error getting article statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve article statistics"
        )


@router.get("/search", response_model=List[ArticleResponse])
async def search_articles(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    days: int = Query(30, ge=1, le=365, description="Days back to search"),
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Search articles by text query.
    
    - **q**: Search query (searches title, content, and author)
    - **limit**: Maximum number of results to return
    - **days**: How many days back to search
    """
    try:
        articles = await article_service.search_articles(
            query=q,
            limit=limit,
            days=days
        )
        
        return [ArticleResponse.model_validate(article) for article in articles]
        
    except Exception as e:
        logger.error(f"Error searching articles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search articles"
        )


@router.get("/keywords/{keyword}", response_model=List[ArticleResponse])
async def get_articles_by_keyword(
    keyword: str,
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    days: int = Query(7, ge=1, le=365, description="Days back to search"),
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Get articles matching a specific keyword.
    
    - **keyword**: Keyword to search for
    - **limit**: Maximum number of results to return
    - **days**: How many days back to search
    """
    try:
        articles = await article_service.get_articles_by_keywords(
            keywords=[keyword],
            limit=limit,
            days=days
        )
        
        return [ArticleResponse.model_validate(article) for article in articles]
        
    except Exception as e:
        logger.error(f"Error getting articles by keyword: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve articles by keyword"
        )


@router.get("/{article_id}", response_model=ArticleDetailResponse)
async def get_article(
    article_id: UUID,
    user: Dict[str, Any] = RequireReadArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Get a specific article by ID.
    
    Returns detailed article information including related data.
    """
    try:
        article = await article_service.get_by_id(article_id)
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        return ArticleDetailResponse.model_validate(article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve article"
        )


@router.put("/{article_id}", response_model=ArticleResponse)
async def update_article(
    article_id: UUID,
    article_update: ArticleUpdate,
    user: Dict[str, Any] = RequireWriteArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Update an existing article.
    
    Updates the specified article with the provided data.
    Only fields that are provided will be updated.
    """
    try:
        # Check if article exists
        article = await article_service.get_by_id(article_id)
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        # Update relevance score if provided
        if article_update.relevance_score is not None:
            success = await article_service.update_relevance_score(
                article_id, article_update.relevance_score
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update relevance score"
                )
        
        # Get updated article
        updated_article = await article_service.get_by_id(article_id)
        return ArticleResponse.model_validate(updated_article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update article"
        )


@router.post("/{article_id}/mark-duplicate", response_model=BaseResponse)
async def mark_article_as_duplicate(
    article_id: UUID,
    user: Dict[str, Any] = RequireWriteArticles,
    article_service: ArticleService = Depends(get_article_service)
):
    """
    Mark an article as duplicate.
    
    This will flag the article as a duplicate and may affect its visibility
    in search results and listings.
    """
    try:
        success = await article_service.mark_as_duplicate(article_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        return BaseResponse(
            success=True,
            message="Article marked as duplicate successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking article {article_id} as duplicate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark article as duplicate"
        )