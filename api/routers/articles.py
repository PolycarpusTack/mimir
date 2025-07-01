"""
Articles Router for Mimir Enterprise API

Handles article CRUD operations, bulk operations, and article management.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ...auth.auth_manager import AuthManager
from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import (RequireArticlesRead, RequireArticlesWrite,
                            get_auth_manager, get_current_user, get_db,
                            get_pagination_params)
from ..models import (APIRouter, ArticleCreate, ArticleResponse, ArticleUpdate,
                      BaseResponse, Depends, ErrorResponse, HTTPException,
                      Limiter, PaginatedResponse, Query,
                      _rate_limit_exceeded_handler, fastapi, from,
                      get_remote_address, import, slowapi, slowapi.util,
                      status)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get('/', response_model=PaginatedResponse)
@limiter.limit("100/minute")
async def list_articles(
    request,
    pagination: Dict[str, int] = Depends(get_pagination_params),
    source: Optional[str] = Query(None, description="Filter by source"),
    category: Optional[str] = Query(None, description="Filter by category"),
    language: Optional[str] = Query(None, description="Filter by language"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    sort_by: str = Query("published_date", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    List articles with filtering, sorting, and pagination
    
    - **source**: Filter articles by source name
    - **category**: Filter articles by category
    - **language**: Filter articles by language code
    - **status**: Filter articles by status (new, processed, archived, error)
    - **start_date**: Filter articles published after this date
    - **end_date**: Filter articles published before this date
    - **sort_by**: Field to sort by (published_date, scraped_at, title, source)
    - **sort_order**: Sort order (asc or desc)
    
    Returns paginated list of articles for the current organization.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Build WHERE clause
        where_conditions = ["organization_id = %s"]
        params = [organization_id]
        
        if source:
            where_conditions.append("source ILIKE %s")
            params.append(f"%{source}%")
        
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if language:
            where_conditions.append("language = %s")
            params.append(language)
        
        if status:
            where_conditions.append("status = %s")
            params.append(status)
        
        if start_date:
            where_conditions.append("published_date >= %s")
            params.append(start_date)
        
        if end_date:
            where_conditions.append("published_date <= %s")
            params.append(end_date)
        
        where_clause = " AND ".join(where_conditions)
        
        # Validate sort field
        valid_sort_fields = ['published_date', 'scraped_at', 'title', 'source', 'updated_at']
        if sort_by not in valid_sort_fields:
            sort_by = 'published_date'
        
        if sort_order.lower() not in ['asc', 'desc']:
            sort_order = 'desc'
        
        # Count total articles
        count_query = f"SELECT COUNT(*) FROM articles WHERE {where_clause}"
        total_result = db.fetch_one(count_query, params)
        total_items = total_result['count'] if total_result else 0
        
        # Get articles
        query = f"""
        SELECT 
            id, url, title, content, summary, published_date, scraped_at, updated_at,
            source, author, category, language, status, tags, metadata, entities, organization_id
        FROM articles 
        WHERE {where_clause}
        ORDER BY {sort_by} {sort_order.upper()}
        LIMIT %s OFFSET %s
        """
        
        params.extend([pagination['limit'], pagination['offset']])
        articles = db.fetch_all(query, params)
        
        # Convert to response models
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
        
    except Exception as e:
        logger.error(f"Error listing articles: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve articles"
        )

@router.get('/{article_id}', response_model=ArticleResponse)
async def get_article(
    article_id: UUID,
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get a specific article by ID
    
    Returns detailed article information if the article exists and belongs
    to the current user's organization.
    """
    try:
        organization_id = current_user['organization_id']
        
        query = """
        SELECT 
            id, url, title, content, summary, published_date, scraped_at, updated_at,
            source, author, category, language, status, tags, metadata, entities, organization_id
        FROM articles 
        WHERE id = %s AND organization_id = %s
        """
        
        article = db.fetch_one(query, (str(article_id), organization_id))
        
        if not article:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        return ArticleResponse(**article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve article"
        )

@router.post('/', response_model=ArticleResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("50/minute")
async def create_article(
    request,
    article: ArticleCreate,
    current_user: Dict[str, Any] = RequireArticlesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Create a new article
    
    - **url**: Article URL (must be unique)
    - **title**: Article title
    - **content**: Article content (optional)
    - **summary**: Article summary (optional)
    - **published_date**: Publication date (optional)
    - **source**: Source name
    - **author**: Author name (optional)
    - **category**: Article category (optional)
    - **language**: Language code (default: en)
    - **tags**: List of tags
    - **metadata**: Additional metadata
    
    Creates a new article in the current user's organization.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if URL already exists
        existing_query = "SELECT id FROM articles WHERE url = %s AND organization_id = %s"
        existing = db.fetch_one(existing_query, (article.url, organization_id))
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Article with this URL already exists"
            )
        
        # Insert new article
        query = """
        INSERT INTO articles (
            organization_id, url, title, content, summary, published_date,
            source, author, category, language, tags, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id, url, title, content, summary, published_date, scraped_at, 
                   updated_at, source, author, category, language, status, 
                   tags, metadata, entities, organization_id
        """
        
        params = (
            organization_id, article.url, article.title, article.content,
            article.summary, article.published_date, article.source,
            article.author, article.category, article.language,
            article.tags, article.metadata
        )
        
        new_article = db.fetch_one(query, params)
        
        if not new_article:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create article"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='create_article',
            resource_type='article',
            resource_id=str(new_article['id']),
            details={'title': article.title, 'source': article.source}
        )
        
        return ArticleResponse(**new_article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating article: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create article"
        )

@router.put('/{article_id}', response_model=ArticleResponse)
async def update_article(
    article_id: UUID,
    article_update: ArticleUpdate,
    current_user: Dict[str, Any] = RequireArticlesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Update an existing article
    
    Updates the specified article with the provided data.
    Only fields that are provided will be updated.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if article exists and belongs to organization
        check_query = "SELECT id FROM articles WHERE id = %s AND organization_id = %s"
        existing = db.fetch_one(check_query, (str(article_id), organization_id))
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        if article_update.title is not None:
            update_fields.append("title = %s")
            params.append(article_update.title)
        
        if article_update.content is not None:
            update_fields.append("content = %s")
            params.append(article_update.content)
        
        if article_update.summary is not None:
            update_fields.append("summary = %s")
            params.append(article_update.summary)
        
        if article_update.author is not None:
            update_fields.append("author = %s")
            params.append(article_update.author)
        
        if article_update.category is not None:
            update_fields.append("category = %s")
            params.append(article_update.category)
        
        if article_update.tags is not None:
            update_fields.append("tags = %s")
            params.append(article_update.tags)
        
        if article_update.metadata is not None:
            update_fields.append("metadata = %s")
            params.append(article_update.metadata)
        
        if article_update.status is not None:
            update_fields.append("status = %s")
            params.append(article_update.status.value)
        
        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add updated_at
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.extend([str(article_id), organization_id])
        
        query = f"""
        UPDATE articles 
        SET {', '.join(update_fields)}
        WHERE id = %s AND organization_id = %s
        RETURNING id, url, title, content, summary, published_date, scraped_at, 
                  updated_at, source, author, category, language, status, 
                  tags, metadata, entities, organization_id
        """
        
        updated_article = db.fetch_one(query, params)
        
        if not updated_article:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update article"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='update_article',
            resource_type='article',
            resource_id=str(article_id),
            details={'updated_fields': list(article_update.dict(exclude_unset=True).keys())}
        )
        
        return ArticleResponse(**updated_article)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update article"
        )

@router.delete('/{article_id}', response_model=BaseResponse)
async def delete_article(
    article_id: UUID,
    current_user: Dict[str, Any] = RequireArticlesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Delete an article
    
    Permanently deletes the specified article and all associated data.
    This action cannot be undone.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if article exists and get title for logging
        check_query = """
        SELECT id, title FROM articles 
        WHERE id = %s AND organization_id = %s
        """
        existing = db.fetch_one(check_query, (str(article_id), organization_id))
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        # Delete article (CASCADE will handle related records)
        delete_query = "DELETE FROM articles WHERE id = %s AND organization_id = %s"
        result = db.execute_query(delete_query, (str(article_id), organization_id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete article"
            )
        
        # Log activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='delete_article',
            resource_type='article',
            resource_id=str(article_id),
            details={'title': existing['title']}
        )
        
        return BaseResponse(
            success=True,
            message="Article deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting article {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete article"
        )

@router.post('/bulk', response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def bulk_create_articles(
    request,
    articles: List[ArticleCreate],
    current_user: Dict[str, Any] = RequireArticlesWrite,
    db: PostgreSQLManager = Depends(get_db),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Bulk create articles
    
    Creates multiple articles in a single request.
    Returns summary of successful and failed creations.
    """
    try:
        if len(articles) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 articles per bulk request"
            )
        
        organization_id = current_user['organization_id']
        
        created_articles = []
        failed_articles = []
        
        for i, article in enumerate(articles):
            try:
                # Check if URL already exists
                existing_query = "SELECT id FROM articles WHERE url = %s AND organization_id = %s"
                existing = db.fetch_one(existing_query, (article.url, organization_id))
                
                if existing:
                    failed_articles.append({
                        'index': i,
                        'url': article.url,
                        'error': 'URL already exists'
                    })
                    continue
                
                # Insert article
                query = """
                INSERT INTO articles (
                    organization_id, url, title, content, summary, published_date,
                    source, author, category, language, tags, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) RETURNING id, title, source
                """
                
                params = (
                    organization_id, article.url, article.title, article.content,
                    article.summary, article.published_date, article.source,
                    article.author, article.category, article.language,
                    article.tags, article.metadata
                )
                
                new_article = db.fetch_one(query, params)
                
                if new_article:
                    created_articles.append({
                        'id': str(new_article['id']),
                        'title': new_article['title'],
                        'source': new_article['source']
                    })
                else:
                    failed_articles.append({
                        'index': i,
                        'url': article.url,
                        'error': 'Failed to create'
                    })
                    
            except Exception as e:
                failed_articles.append({
                    'index': i,
                    'url': article.url,
                    'error': str(e)
                })
        
        # Log bulk activity
        auth_manager._log_activity(
            organization_id=organization_id,
            user_id=current_user.get('user_id'),
            action='bulk_create_articles',
            details={
                'total_attempted': len(articles),
                'successful': len(created_articles),
                'failed': len(failed_articles)
            }
        )
        
        return {
            'success': True,
            'summary': {
                'total_attempted': len(articles),
                'successful': len(created_articles),
                'failed': len(failed_articles)
            },
            'created_articles': created_articles,
            'failed_articles': failed_articles
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk article creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process bulk article creation"
        )

@router.get('/{article_id}/similar', response_model=List[ArticleResponse])
async def get_similar_articles(
    article_id: UUID,
    limit: int = Query(10, ge=1, le=50, description="Number of similar articles to return"),
    threshold: float = Query(0.3, ge=0.0, le=1.0, description="Similarity threshold"),
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get articles similar to the specified article
    
    Uses semantic similarity search to find related articles.
    Requires the semantic search functionality to be enabled.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Check if article exists
        check_query = """
        SELECT id FROM articles 
        WHERE id = %s AND organization_id = %s
        """
        existing = db.fetch_one(check_query, (str(article_id), organization_id))
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Article not found"
            )
        
        # Use semantic similarity function if available
        try:
            query = """
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, 
                organization_id, similarity_score
            FROM find_similar_articles(%s, %s, %s)
            WHERE organization_id = %s
            """
            
            similar_articles = db.fetch_all(query, (
                str(article_id), threshold, limit, organization_id
            ))
            
            return [ArticleResponse(**article) for article in similar_articles]
            
        except Exception:
            # Fallback to basic similarity using title and source
            fallback_query = """
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, organization_id
            FROM articles 
            WHERE organization_id = %s 
              AND id != %s
              AND (
                source = (SELECT source FROM articles WHERE id = %s)
                OR category = (SELECT category FROM articles WHERE id = %s)
              )
            ORDER BY scraped_at DESC
            LIMIT %s
            """
            
            similar_articles = db.fetch_all(fallback_query, (
                organization_id, str(article_id), str(article_id), 
                str(article_id), limit
            ))
            
            return [ArticleResponse(**article) for article in similar_articles]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar articles for {article_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to find similar articles"
        )