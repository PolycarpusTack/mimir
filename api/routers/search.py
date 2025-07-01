"""
Search Router for Mimir Enterprise API

Handles advanced search functionality including semantic search,
faceted search, and search analytics.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ...db_manager_postgres import PostgreSQLManager
from ..dependencies import RequireArticlesRead, get_current_user, get_db
from ..models import (APIRouter, ArticleResponse, BaseResponse, Depends,
                      HTTPException, Limiter, Query, SearchRequest,
                      SearchResponse, _rate_limit_exceeded_handler, fastapi,
                      from, get_remote_address, import, slowapi, slowapi.util,
                      status)

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.get('/', response_model=SearchResponse)
@limiter.limit("200/minute")
async def search_articles(
    request,
    q: Optional[str] = Query(None, description="Search query"),
    source: Optional[str] = Query(None, description="Filter by source"),
    category: Optional[str] = Query(None, description="Filter by category"),
    language: Optional[str] = Query(None, description="Filter by language"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    sort_by: str = Query("relevance", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    semantic: bool = Query(False, description="Use semantic search"),
    similarity_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Similarity threshold for semantic search"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    facets: bool = Query(True, description="Include search facets"),
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Search articles with advanced filtering and semantic capabilities
    
    - **q**: Search query text (searches title, content, summary)
    - **source**: Filter by source name (partial match)
    - **category**: Filter by exact category
    - **language**: Filter by language code
    - **start_date**: Filter articles published after this date
    - **end_date**: Filter articles published before this date
    - **tags**: Filter by tags (comma-separated)
    - **sort_by**: Sort field (relevance, published_date, scraped_at, title, source)
    - **sort_order**: Sort order (asc or desc)
    - **semantic**: Enable semantic similarity search
    - **similarity_threshold**: Minimum similarity score for semantic search
    - **limit**: Maximum number of results (1-1000)
    - **offset**: Number of results to skip
    - **facets**: Include faceted search results
    
    Returns search results with optional facets and metadata.
    """
    try:
        start_time = time.time()
        organization_id = current_user['organization_id']
        
        # Build base WHERE clause
        where_conditions = ["organization_id = %s"]
        params = [organization_id]
        
        # Add filters
        if source:
            where_conditions.append("source ILIKE %s")
            params.append(f"%{source}%")
        
        if category:
            where_conditions.append("category = %s")
            params.append(category)
        
        if language:
            where_conditions.append("language = %s")
            params.append(language)
        
        if start_date:
            where_conditions.append("published_date >= %s")
            params.append(start_date)
        
        if end_date:
            where_conditions.append("published_date <= %s")
            params.append(end_date)
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',')]
            where_conditions.append("tags && %s")
            params.append(tag_list)
        
        where_clause = " AND ".join(where_conditions)
        
        # Handle different search modes
        if q and semantic:
            # Semantic search
            try:
                # This would use the embedding-based search function
                search_query = """
                SELECT 
                    a.id, a.url, a.title, a.content, a.summary, a.published_date, 
                    a.scraped_at, a.updated_at, a.source, a.author, a.category, 
                    a.language, a.status, a.tags, a.metadata, a.entities, a.organization_id,
                    similarity_score
                FROM search_articles_semantic(
                    get_text_embedding(%s), %s, %s
                ) s
                JOIN articles a ON a.id = s.id
                WHERE a.organization_id = %s
                """
                
                # Add additional filters if needed
                if len(where_conditions) > 1:
                    additional_filters = " AND ".join(where_conditions[1:])
                    search_query += f" AND {additional_filters}"
                
                search_query += f" ORDER BY similarity_score DESC LIMIT %s OFFSET %s"
                
                # Parameters for semantic search
                search_params = [q, similarity_threshold, limit, organization_id]
                if len(params) > 1:
                    search_params.extend(params[1:])
                search_params.extend([limit, offset])
                
                articles = db.fetch_all(search_query, search_params)
                
            except Exception as semantic_error:
                logger.warning(f"Semantic search failed, falling back to text search: {semantic_error}")
                semantic = False
        
        if q and not semantic:
            # Full-text search
            where_conditions.append("search_vector @@ websearch_to_tsquery('english', %s)")
            params.append(q)
            
            search_query = f"""
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, organization_id,
                ts_rank(search_vector, websearch_to_tsquery('english', %s)) as relevance_score
            FROM articles 
            WHERE {" AND ".join(where_conditions)}
            """
            
            # Add sorting
            if sort_by == "relevance":
                search_query += " ORDER BY relevance_score DESC, published_date DESC"
            else:
                valid_sort_fields = ['published_date', 'scraped_at', 'title', 'source', 'updated_at']
                if sort_by in valid_sort_fields:
                    order = sort_order.upper() if sort_order.lower() in ['asc', 'desc'] else 'DESC'
                    search_query += f" ORDER BY {sort_by} {order}"
                else:
                    search_query += " ORDER BY published_date DESC"
            
            search_query += " LIMIT %s OFFSET %s"
            
            # Add query parameter for ranking
            search_params = params + [q, limit, offset]
            articles = db.fetch_all(search_query, search_params)
            
        elif not q:
            # No query, just filtering
            search_query = f"""
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, organization_id
            FROM articles 
            WHERE {where_clause}
            """
            
            # Add sorting
            valid_sort_fields = ['published_date', 'scraped_at', 'title', 'source', 'updated_at']
            if sort_by in valid_sort_fields:
                order = sort_order.upper() if sort_order.lower() in ['asc', 'desc'] else 'DESC'
                search_query += f" ORDER BY {sort_by} {order}"
            else:
                search_query += " ORDER BY published_date DESC"
            
            search_query += " LIMIT %s OFFSET %s"
            search_params = params + [limit, offset]
            articles = db.fetch_all(search_query, search_params)
        
        else:
            articles = []
        
        # Convert to response models
        article_responses = [
            ArticleResponse(**article) for article in articles
        ]
        
        # Get facets if requested
        search_facets = None
        if facets and articles:
            search_facets = await _get_search_facets(db, organization_id, where_clause, params[:-2] if q else params)
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=q,
            total_results=len(articles),
            search_time_ms=round(search_time_ms, 2),
            semantic_search=semantic,
            articles=article_responses,
            facets=search_facets
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search service error"
        )

@router.post('/', response_model=SearchResponse)
@limiter.limit("100/minute")
async def advanced_search(
    request,
    search_request: SearchRequest,
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Advanced search with complex query structure
    
    Supports complex search requests with multiple criteria,
    boolean operators, and advanced filtering options.
    """
    try:
        start_time = time.time()
        organization_id = current_user['organization_id']
        
        # Build query from search request
        where_conditions = ["organization_id = %s"]
        params = [organization_id]
        
        # Add filters from search request
        if search_request.source:
            where_conditions.append("source ILIKE %s")
            params.append(f"%{search_request.source}%")
        
        if search_request.category:
            where_conditions.append("category = %s")
            params.append(search_request.category)
        
        if search_request.language:
            where_conditions.append("language = %s")
            params.append(search_request.language)
        
        if search_request.start_date:
            where_conditions.append("published_date >= %s")
            params.append(search_request.start_date)
        
        if search_request.end_date:
            where_conditions.append("published_date <= %s")
            params.append(search_request.end_date)
        
        if search_request.tags:
            where_conditions.append("tags && %s")
            params.append(search_request.tags)
        
        where_clause = " AND ".join(where_conditions)
        
        # Execute search based on type
        if search_request.query and search_request.semantic_search:
            # Semantic search
            try:
                search_query = """
                SELECT 
                    a.id, a.url, a.title, a.content, a.summary, a.published_date, 
                    a.scraped_at, a.updated_at, a.source, a.author, a.category, 
                    a.language, a.status, a.tags, a.metadata, a.entities, a.organization_id,
                    similarity_score
                FROM search_articles_semantic(
                    get_text_embedding(%s), %s, 100
                ) s
                JOIN articles a ON a.id = s.id
                WHERE a.organization_id = %s
                """
                
                if len(where_conditions) > 1:
                    additional_filters = " AND ".join(where_conditions[1:])
                    search_query += f" AND {additional_filters}"
                
                search_query += " ORDER BY similarity_score DESC LIMIT 100"
                
                search_params = [search_request.query, search_request.similarity_threshold, organization_id]
                if len(params) > 1:
                    search_params.extend(params[1:])
                
                articles = db.fetch_all(search_query, search_params)
                
            except Exception:
                # Fallback to text search
                search_request.semantic_search = False
        
        if search_request.query and not search_request.semantic_search:
            # Full-text search
            where_conditions.append("search_vector @@ websearch_to_tsquery('english', %s)")
            params.append(search_request.query)
            
            search_query = f"""
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, organization_id,
                ts_rank(search_vector, websearch_to_tsquery('english', %s)) as relevance_score
            FROM articles 
            WHERE {" AND ".join(where_conditions)}
            ORDER BY relevance_score DESC, published_date DESC
            LIMIT 100
            """
            
            search_params = params + [search_request.query]
            articles = db.fetch_all(search_query, search_params)
            
        else:
            # Filter-only search
            sort_field = search_request.sort_by
            if sort_field not in ['published_date', 'scraped_at', 'title', 'source', 'updated_at']:
                sort_field = 'published_date'
            
            sort_direction = search_request.sort_order.value.upper()
            
            search_query = f"""
            SELECT 
                id, url, title, content, summary, published_date, scraped_at, updated_at,
                source, author, category, language, status, tags, metadata, entities, organization_id
            FROM articles 
            WHERE {where_clause}
            ORDER BY {sort_field} {sort_direction}
            LIMIT 100
            """
            
            articles = db.fetch_all(search_query, params)
        
        # Convert to response models
        article_responses = [
            ArticleResponse(**article) for article in articles
        ]
        
        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=search_request.query,
            total_results=len(articles),
            search_time_ms=round(search_time_ms, 2),
            semantic_search=search_request.semantic_search,
            articles=article_responses,
            facets=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Advanced search service error"
        )

@router.get('/suggestions', response_model=Dict[str, List[str]])
@limiter.limit("50/minute") 
async def get_search_suggestions(
    request,
    q: str = Query(..., min_length=2, description="Query for suggestions"),
    limit: int = Query(10, ge=1, le=50, description="Maximum suggestions"),
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get search suggestions and autocomplete
    
    Returns suggestions for sources, categories, tags, and authors
    based on the partial query input.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Get source suggestions
        source_query = """
        SELECT DISTINCT source 
        FROM articles 
        WHERE organization_id = %s AND source ILIKE %s
        LIMIT %s
        """
        sources = db.fetch_all(source_query, (organization_id, f"%{q}%", limit))
        source_suggestions = [row['source'] for row in sources]
        
        # Get category suggestions
        category_query = """
        SELECT DISTINCT category 
        FROM articles 
        WHERE organization_id = %s AND category ILIKE %s AND category IS NOT NULL
        LIMIT %s
        """
        categories = db.fetch_all(category_query, (organization_id, f"%{q}%", limit))
        category_suggestions = [row['category'] for row in categories]
        
        # Get tag suggestions
        tag_query = """
        SELECT DISTINCT unnest(tags) as tag
        FROM articles 
        WHERE organization_id = %s AND unnest(tags) ILIKE %s
        LIMIT %s
        """
        tags = db.fetch_all(tag_query, (organization_id, f"%{q}%", limit))
        tag_suggestions = [row['tag'] for row in tags]
        
        # Get author suggestions
        author_query = """
        SELECT DISTINCT author 
        FROM articles 
        WHERE organization_id = %s AND author ILIKE %s AND author IS NOT NULL
        LIMIT %s
        """
        authors = db.fetch_all(author_query, (organization_id, f"%{q}%", limit))
        author_suggestions = [row['author'] for row in authors]
        
        return {
            'sources': source_suggestions,
            'categories': category_suggestions,
            'tags': tag_suggestions,
            'authors': author_suggestions
        }
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get search suggestions"
        )

@router.get('/trending', response_model=Dict[str, Any])
async def get_trending_topics(
    days: int = Query(7, ge=1, le=30, description="Number of days for trending analysis"),
    limit: int = Query(20, ge=1, le=100, description="Maximum trending topics"),
    current_user: Dict[str, Any] = RequireArticlesRead,
    db: PostgreSQLManager = Depends(get_db)
):
    """
    Get trending topics and keywords
    
    Analyzes recent articles to identify trending topics, keywords,
    and emerging themes based on frequency and growth patterns.
    """
    try:
        organization_id = current_user['organization_id']
        
        # Get trending sources
        trending_sources_query = """
        SELECT 
            source,
            COUNT(*) as article_count,
            COUNT(DISTINCT DATE(scraped_at)) as active_days
        FROM articles 
        WHERE organization_id = %s 
          AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY source
        HAVING COUNT(*) > 1
        ORDER BY article_count DESC, active_days DESC
        LIMIT %s
        """
        
        trending_sources = db.fetch_all(trending_sources_query, (organization_id, days, limit))
        
        # Get trending categories
        trending_categories_query = """
        SELECT 
            category,
            COUNT(*) as article_count
        FROM articles 
        WHERE organization_id = %s 
          AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
          AND category IS NOT NULL
        GROUP BY category
        HAVING COUNT(*) > 1
        ORDER BY article_count DESC
        LIMIT %s
        """
        
        trending_categories = db.fetch_all(trending_categories_query, (organization_id, days, limit))
        
        # Get trending tags
        trending_tags_query = """
        SELECT 
            unnest(tags) as tag,
            COUNT(*) as usage_count
        FROM articles 
        WHERE organization_id = %s 
          AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
          AND array_length(tags, 1) > 0
        GROUP BY tag
        HAVING COUNT(*) > 1
        ORDER BY usage_count DESC
        LIMIT %s
        """
        
        trending_tags = db.fetch_all(trending_tags_query, (organization_id, days, limit))
        
        # Get recent high-activity days
        activity_query = """
        SELECT 
            DATE(scraped_at) as date,
            COUNT(*) as articles_count
        FROM articles 
        WHERE organization_id = %s 
          AND scraped_at >= CURRENT_DATE - INTERVAL '%s days'
        GROUP BY DATE(scraped_at)
        ORDER BY date DESC
        """
        
        daily_activity = db.fetch_all(activity_query, (organization_id, days))
        
        return {
            'period_days': days,
            'trending_sources': trending_sources,
            'trending_categories': trending_categories,
            'trending_tags': trending_tags,
            'daily_activity': daily_activity,
            'generated_at': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trending topics"
        )

async def _get_search_facets(
    db: PostgreSQLManager, 
    organization_id: str, 
    where_clause: str, 
    params: List[Any]
) -> Dict[str, Any]:
    """Get search facets for the current search results"""
    try:
        # Source facets
        source_query = f"""
        SELECT source, COUNT(*) as count
        FROM articles 
        WHERE {where_clause}
        GROUP BY source
        ORDER BY count DESC
        LIMIT 20
        """
        
        source_facets = db.fetch_all(source_query, params)
        
        # Category facets
        category_query = f"""
        SELECT category, COUNT(*) as count
        FROM articles 
        WHERE {where_clause} AND category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
        LIMIT 20
        """
        
        category_facets = db.fetch_all(category_query, params)
        
        # Language facets
        language_query = f"""
        SELECT language, COUNT(*) as count
        FROM articles 
        WHERE {where_clause}
        GROUP BY language
        ORDER BY count DESC
        LIMIT 10
        """
        
        language_facets = db.fetch_all(language_query, params)
        
        # Date facets (last 30 days)
        date_query = f"""
        SELECT DATE(published_date) as date, COUNT(*) as count
        FROM articles 
        WHERE {where_clause} AND published_date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY DATE(published_date)
        ORDER BY date DESC
        LIMIT 30
        """
        
        date_facets = db.fetch_all(date_query, params)
        
        return {
            'sources': source_facets,
            'categories': category_facets,
            'languages': language_facets,
            'dates': date_facets
        }
        
    except Exception as e:
        logger.warning(f"Error getting search facets: {e}")
        return {}