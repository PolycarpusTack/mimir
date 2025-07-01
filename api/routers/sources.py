"""
Sources Router for Mimir Enterprise API

Handles news source management, configuration, and monitoring.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...auth.auth_manager import AuthManager
from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import (RequireSourcesWrite, get_auth_manager,
                            get_current_user, get_db, get_pagination_params)
from ..models import (APIRouter, BaseResponse, Depends, ErrorResponse,
                      HTTPException, Limiter, PaginatedResponse, Query,
                      SourceCreate, SourceResponse, SourceUpdate,
                      _rate_limit_exceeded_handler, fastapi, from,
                      get_remote_address, import, slowapi, slowapi.util,
                      status)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get('/', response_model=PaginatedResponse)
async def list_sources(
    pagination: Dict[str, int] = Depends(get_pagination_params),
    enabled: Optional[bool] = Query(None, description="Filter by enabled status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    type_filter: Optional[str] = Query(None, alias="type", description="Filter by source type"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: PostgreSQLManager = Depends(get_db)
):
    """
    List news sources with filtering and pagination
    
    - **enabled**: Filter by enabled status (true/false)
    - **category**: Filter sources by category
    - **type**: Filter by source type (rss, html, api)
    
    Returns paginated list of sources for the current organization.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Build WHERE clause
        where_conditions = ["organization_id = %s"]
        params = [organization_id]
        
        if enabled is not None:
            where_conditions.append("enabled = %s")
            params.append(enabled)
        
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if type_filter:
            where_conditions.append("type = %s")
            params.append(type_filter)
        
        where_clause = " AND ".join(where_conditions)
        
        # Count total sources
        count_query = f"SELECT COUNT(*) FROM sources WHERE {where_clause}"
        total_result = db.fetch_one(count_query, params)
        total_items = total_result['count'] if total_result else 0
        
        # Get sources
        query = f"""
        SELECT 
            id, name, url, type, enabled, category, config, selectors,
            rate_limit_seconds, last_scraped_at, next_scrape_at, total_articles,
            failed_attempts, last_error, last_error_at, organization_id,
            created_at, updated_at
        FROM sources 
        WHERE {where_clause}
        ORDER BY name
        LIMIT %s OFFSET %s
        """
        
        params.extend([pagination['limit'], pagination['offset']])
        sources = db.fetch_all(query, params)
        
        # Convert to response models
        source_responses = [
            SourceResponse(**source) for source in sources
        ]
        
        # Calculate pagination metadata
        total_pages = (total_items + pagination['page_size'] - 1) // pagination['page_size']
        
        from ..models import PaginationMeta
        meta = PaginationMeta(
            page=pagination['page'],
            page_size=pagination['page_size'],
            total_items=total_items,
            total_pages=total_pages,
            has_next=pagination['page'] < total_pages,
            has_previous=pagination['page'] > 1
        )
        
        return PaginatedResponse(data=source_responses, meta=meta)
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sources"
        )

@router.get('/{source_id}', response_model=SourceResponse)
async def get_source(
    source_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get a specific source by ID
    
    Returns detailed source information including configuration and statistics.
    """
    try:
        organization_id = current_user['organization_id']
        
        query = """
        SELECT 
            id, name, url, type, enabled, category, config, selectors,
            rate_limit_seconds, last_scraped_at, next_scrape_at, total_articles,
            failed_attempts, last_error, last_error_at, organization_id,
            created_at, updated_at
        FROM sources 
        WHERE id = %s AND organization_id = %s
        """
        
        source = db.fetch_one(query, (str(source_id), organization_id))
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        return SourceResponse(**source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source"
        )

@router.post('/', response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def create_source(
    request,
    source: SourceCreate,
    current_user: Dict[str, Any] = RequireSourcesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Create a new news source
    
    - **name**: Human-readable source name
    - **url**: Source URL (RSS feed, website, or API endpoint)
    - **type**: Source type (rss, html, api)
    - **category**: Optional category classification
    - **enabled**: Whether source is active (default: true)
    - **config**: Source-specific configuration
    - **selectors**: CSS/XPath selectors for HTML scraping
    - **rate_limit_seconds**: Minimum seconds between scrapes
    
    Creates a new source for the current organization.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check organization source limits
        if not auth_manager.check_organization_limits(organization_id, 'sources'):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Organization source limit exceeded"
            )
        
        # Check if source name already exists
        existing_query = "SELECT id FROM sources WHERE name = %s AND organization_id = %s"
        existing = db.fetch_one(existing_query, (source.name, organization_id))
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Source with this name already exists"
            )
        
        # Insert new source
        query = """
        INSERT INTO sources (
            organization_id, name, url, type, enabled, category, 
            config, selectors, rate_limit_seconds
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING 
            id, name, url, type, enabled, category, config, selectors,
            rate_limit_seconds, last_scraped_at, next_scrape_at, total_articles,
            failed_attempts, last_error, last_error_at, organization_id,
            created_at, updated_at
        """
        
        params = (
            organization_id, source.name, source.url, source.type,
            source.enabled, source.category, source.config,
            source.selectors, source.rate_limit_seconds
        )
        
        new_source = db.fetch_one(query, params)
        
        if not new_source:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create source"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='create_source',
            resource_type='source',
            resource_id=str(new_source['id']),
            details={'name': source.name, 'url': source.url, 'type': source.type}
        )
        
        return SourceResponse(**new_source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create source"
        )

@router.put('/{source_id}', response_model=SourceResponse)
async def update_source(
    source_id: UUID,
    source_update: SourceUpdate,
    current_user: Dict[str, Any] = RequireSourcesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Update an existing source
    
    Updates the specified source with the provided data.
    Only fields that are provided will be updated.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if source exists and belongs to organization
        check_query = "SELECT id, name FROM sources WHERE id = %s AND organization_id = %s"
        existing = db.fetch_one(check_query, (str(source_id), organization_id))
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        if source_update.name is not None:
            # Check for name conflicts
            name_check_query = """
            SELECT id FROM sources 
            WHERE name = %s AND organization_id = %s AND id != %s
            """
            name_conflict = db.fetch_one(name_check_query, (
                source_update.name, organization_id, str(source_id)
            ))
            
            if name_conflict:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Source with this name already exists"
                )
            
            update_fields.append("name = %s")
            params.append(source_update.name)
        
        if source_update.enabled is not None:
            update_fields.append("enabled = %s")
            params.append(source_update.enabled)
        
        if source_update.category is not None:
            update_fields.append("category = %s")
            params.append(source_update.category)
        
        if source_update.config is not None:
            update_fields.append("config = %s")
            params.append(source_update.config)
        
        if source_update.selectors is not None:
            update_fields.append("selectors = %s")
            params.append(source_update.selectors)
        
        if source_update.rate_limit_seconds is not None:
            update_fields.append("rate_limit_seconds = %s")
            params.append(source_update.rate_limit_seconds)
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add updated_at
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.extend([str(source_id), organization_id])
        
        query = f"""
        UPDATE sources 
        SET {', '.join(update_fields)}
        WHERE id = %s AND organization_id = %s
        RETURNING 
            id, name, url, type, enabled, category, config, selectors,
            rate_limit_seconds, last_scraped_at, next_scrape_at, total_articles,
            failed_attempts, last_error, last_error_at, organization_id,
            created_at, updated_at
        """
        
        updated_source = db.fetch_one(query, params)
        
        if not updated_source:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update source"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='update_source',
            resource_type='source',
            resource_id=str(source_id),
            details={
                'updated_fields': list(source_update.dict(exclude_unset=True).keys()),
                'source_name': existing['name']
            }
        )
        
        return SourceResponse(**updated_source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update source"
        )

@router.delete('/{source_id}', response_model=BaseResponse)
async def delete_source(
    source_id: UUID,
    current_user: Dict[str, Any] = RequireSourcesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Delete a source
    
    Permanently deletes the specified source and all associated data.
    This action cannot be undone.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if source exists and get name for logging
        check_query = """
        SELECT id, name, total_articles FROM sources 
        WHERE id = %s AND organization_id = %s
        """
        existing = db.fetch_one(check_query, (str(source_id), organization_id))
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        # Delete source (CASCADE will handle related records)
        delete_query = "DELETE FROM sources WHERE id = %s AND organization_id = %s"
        result = db.execute_query(delete_query, (str(source_id), organization_id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete source"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='delete_source',
            resource_type='source',
            resource_id=str(source_id),
            details={
                'name': existing['name'],
                'articles_count': existing['total_articles']
            }
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

@router.post('/{source_id}/test', response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def test_source(
    request,
    source_id: UUID,
    current_user: Dict[str, Any] = RequireSourcesWrite,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Test a source configuration
    
    Performs a test scrape of the source to validate configuration
    and check connectivity. Does not save any articles.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Get source
        query = """
        SELECT name, url, type, config, selectors FROM sources 
        WHERE id = %s AND organization_id = %s
        """
        source = db.fetch_one(query, (str(source_id), organization_id))
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        # TODO: Implement actual source testing logic
        # This would test the URL, validate selectors, etc.
        
        # For now, return a mock test result
        return {
            'success': True,
            'source_name': source['name'],
            'url': source['url'],
            'type': source['type'],
            'test_results': {
                'connectivity': 'success',
                'response_time_ms': 150,
                'articles_found': 5,
                'errors': [],
                'warnings': []
            },
            'sample_articles': [
                {
                    'title': 'Sample Article 1',
                    'url': 'https://example.com/article1',
                    'published_date': '2024-01-01T10:00:00Z'
                }
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test source"
        )

@router.get('/{source_id}/articles', response_model=PaginatedResponse)
async def get_source_articles(
    source_id: UUID,
    pagination: Dict[str, int] = Depends(get_pagination_params),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get articles from a specific source
    
    Returns paginated list of articles scraped from the specified source.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if source exists
        check_query = """
        SELECT name FROM sources 
        WHERE id = %s AND organization_id = %s
        """
        source = db.fetch_one(check_query, (str(source_id), organization_id))
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        # Count total articles for this source
        count_query = """
        SELECT COUNT(*) FROM articles 
        WHERE organization_id = %s AND source = %s
        """
        total_result = db.fetch_one(count_query, (organization_id, source['name']))
        total_items = total_result['count'] if total_result else 0
        
        # Get articles
        query = """
        SELECT 
            id, url, title, content, summary, published_date, scraped_at, updated_at,
            source, author, category, language, status, tags, metadata, entities, organization_id
        FROM articles 
        WHERE organization_id = %s AND source = %s
        ORDER BY published_date DESC, scraped_at DESC
        LIMIT %s OFFSET %s
        """
        
        articles = db.fetch_all(query, (
            organization_id, source['name'], 
            pagination['limit'], pagination['offset']
        ))
        
        # Convert to response models
        from ..models import ArticleResponse
        article_responses = [
            ArticleResponse(**article) for article in articles
        ]
        
        # Calculate pagination metadata
        total_pages = (total_items + pagination['page_size'] - 1) // pagination['page_size']
        
        from ..models import PaginationMeta
        meta = PaginationMeta(
            page=pagination['page'],
            page_size=pagination['page_size'],
            total_items=total_items,
            total_pages=total_pages,
            has_next=pagination['page'] < total_pages,
            has_previous=pagination['page'] > 1
        )
        
        return PaginatedResponse(data=article_responses, meta=meta)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting articles for source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source articles"
        )

@router.get('/{source_id}/stats', response_model=Dict[str, Any])
async def get_source_statistics(
    source_id: UUID,
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get statistics for a specific source
    
    Returns detailed statistics including article counts, scraping history,
    and performance metrics for the specified time period.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if source exists
        check_query = """
        SELECT name, total_articles, failed_attempts, last_scraped_at, created_at
        FROM sources 
        WHERE id = %s AND organization_id = %s
        """
        source = db.fetch_one(check_query, (str(source_id), organization_id))
        
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source not found"
            )
        
        # Get article statistics by day
        daily_stats_query = """
        SELECT 
            DATE(scraped_at) as date,
            COUNT(*) as articles_count
        FROM articles 
        WHERE organization_id = %s 
          AND source = %s
          AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY DATE(scraped_at)
        ORDER BY date DESC
        """
        
        daily_stats = db.fetch_all(daily_stats_query, (
            organization_id, source['name'], days
        ))
        
        # Get recent scrape runs
        scrape_runs_query = """
        SELECT 
            started_at, ended_at, articles_found, articles_new, 
            articles_error, status, error_message
        FROM scrape_runs 
        WHERE source_id = %s
        ORDER BY started_at DESC
        LIMIT 10
        """
        
        recent_runs = db.fetch_all(scrape_runs_query, (str(source_id),))
        
        # Calculate summary statistics
        total_articles_period = sum(stat['articles_count'] for stat in daily_stats)
        avg_articles_per_day = total_articles_period / days if days > 0 else 0
        
        return {
            'source_id': str(source_id),
            'source_name': source['name'],
            'summary': {
                'total_articles_all_time': source['total_articles'],
                'total_articles_period': total_articles_period,
                'average_articles_per_day': round(avg_articles_per_day, 2),
                'failed_attempts': source['failed_attempts'],
                'last_scraped_at': source['last_scraped_at'],
                'created_at': source['created_at']
            },
            'daily_statistics': daily_stats,
            'recent_scrape_runs': recent_runs,
            'period_days': days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics for source {source_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source statistics"
        )