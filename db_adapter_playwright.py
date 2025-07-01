"""
Extended database adapter with Playwright support.
This module extends the base db_adapter with functions for Playwright-specific data.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Import base adapter functions
from db_adapter import *

logger = logging.getLogger(__name__)


def add_article_with_playwright_data(
    url: str,
    title: str,
    publication_date: Any,
    source_website: str,
    category: Optional[str] = None,
    content_summary: str = "",
    full_content: str = "",
    author: Optional[str] = None,
    keywords_found: Optional[List[str]] = None,
    rendering_method: str = "html",
    screenshot_path: Optional[str] = None,
    js_errors: Optional[List[Dict[str, Any]]] = None,
    page_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Add article with Playwright-specific data.
    
    Args:
        url: Article URL
        title: Article title
        publication_date: Publication date
        source_website: Source website name
        category: Article category
        content_summary: Brief summary
        full_content: Full article content
        author: Article author
        keywords_found: List of keywords found
        rendering_method: Method used ('html' or 'playwright')
        screenshot_path: Path to screenshot if captured
        js_errors: List of JavaScript errors encountered
        page_metrics: Page performance metrics
        
    Returns:
        Article ID if successful, None otherwise
    """
    if USE_POSTGRES:
        # PostgreSQL implementation with Playwright columns
        from datetime import datetime
        import psycopg2
        
        if isinstance(publication_date, str):
            try:
                from dateutil import parser
                publication_date = parser.parse(publication_date)
            except (ValueError, TypeError):
                publication_date = None
                
        with get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                # Prepare metadata with Playwright info
                metadata = {
                    "original_url": url,
                    "word_count": len(full_content.split()) if full_content else 0,
                    "has_full_content": bool(full_content),
                }
                
                # Add Playwright metrics to metadata if available
                if page_metrics:
                    metadata["playwright_metrics"] = page_metrics
                
                cursor.execute(
                    """
                    INSERT INTO articles (
                        url, title, content, summary, published_date, source,
                        author, category, metadata, tags, rendering_method,
                        screenshot_path, js_errors, page_metrics
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING id
                    """,
                    (
                        url,
                        title,
                        full_content or content_summary,
                        content_summary,
                        publication_date,
                        source_website,
                        author,
                        category,
                        json.dumps(metadata),
                        json.dumps(keywords_found or []),
                        rendering_method,
                        screenshot_path,
                        json.dumps(js_errors or []),
                        json.dumps(page_metrics or {})
                    ),
                )
                
                article_id = cursor.fetchone()[0]
                logger.info(f"Article added with {rendering_method} rendering: {title[:50]}... (ID: {article_id})")
                
                # Add keyword alerts if any keywords were found
                if keywords_found:
                    for keyword in keywords_found:
                        add_keyword_alert(article_id, keyword, conn=conn)
                        
                return article_id
                
            except psycopg2.IntegrityError as e:
                logger.error(f"IntegrityError for URL {url}: {e}")
                return None
                
    else:
        # SQLite fallback - store Playwright data in metadata
        # Extend the metadata to include Playwright-specific fields
        article_id = add_article(
            url=url,
            title=title,
            publication_date=publication_date,
            source_website=source_website,
            category=category,
            content_summary=content_summary,
            full_content=full_content,
            author=author,
            keywords_found=keywords_found,
        )
        
        if article_id and (rendering_method != "html" or screenshot_path or js_errors or page_metrics):
            # Update metadata with Playwright data
            update_article_playwright_metadata(
                article_id,
                rendering_method,
                screenshot_path,
                js_errors,
                page_metrics
            )
            
        return article_id


def update_article_playwright_metadata(
    article_id: Union[str, int],
    rendering_method: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    js_errors: Optional[List[Dict[str, Any]]] = None,
    page_metrics: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update Playwright-specific metadata for an article.
    
    Args:
        article_id: Article ID
        rendering_method: Method used for rendering
        screenshot_path: Path to screenshot
        js_errors: JavaScript errors
        page_metrics: Page performance metrics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get existing metadata
        existing_metadata = get_article_metadata(article_id) or {}
        
        # Update with Playwright data
        playwright_data = {}
        if rendering_method:
            playwright_data["rendering_method"] = rendering_method
        if screenshot_path:
            playwright_data["screenshot_path"] = screenshot_path
        if js_errors is not None:
            playwright_data["js_errors"] = js_errors
        if page_metrics is not None:
            playwright_data["page_metrics"] = page_metrics
            
        if playwright_data:
            existing_metadata["playwright"] = playwright_data
            return update_article_metadata(article_id, existing_metadata)
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to update Playwright metadata for article {article_id}: {e}")
        return False


def get_articles_by_rendering_method(
    method: str,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Get articles filtered by rendering method.
    
    Args:
        method: Rendering method ('html' or 'playwright')
        limit: Maximum number of articles
        offset: Offset for pagination
        
    Returns:
        List of article dictionaries
    """
    if USE_POSTGRES:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(
                """
                SELECT 
                    id, url, title, summary, published_date, source,
                    author, category, rendering_method, screenshot_path,
                    js_errors, page_metrics, created_at
                FROM articles
                WHERE rendering_method = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (method, limit, offset)
            )
            return cursor.fetchall()
    else:
        # For SQLite, search in metadata
        articles = get_recent_articles(limit, offset)
        filtered = []
        
        for article in articles:
            metadata = json.loads(article.get("metadata", "{}"))
            playwright_data = metadata.get("playwright", {})
            
            if playwright_data.get("rendering_method") == method:
                # Add Playwright fields to article
                article["rendering_method"] = method
                article["screenshot_path"] = playwright_data.get("screenshot_path")
                article["js_errors"] = playwright_data.get("js_errors", [])
                article["page_metrics"] = playwright_data.get("page_metrics", {})
                filtered.append(article)
                
        return filtered


def add_playwright_stats(
    browser_type: str,
    pages_loaded: int = 0,
    js_executed: int = 0,
    errors: int = 0,
    fallbacks: int = 0,
    total_time_seconds: float = 0.0,
    session_id: Optional[str] = None
) -> Optional[int]:
    """
    Add Playwright scraping statistics.
    
    Args:
        browser_type: Browser used (chromium, firefox, webkit)
        pages_loaded: Number of pages loaded
        js_executed: Number of JS executions
        errors: Number of errors
        fallbacks: Number of fallbacks to BeautifulSoup
        total_time_seconds: Total time spent
        session_id: Optional session ID
        
    Returns:
        Stats record ID if successful
    """
    if USE_POSTGRES:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                INSERT INTO playwright_stats (
                    browser_type, pages_loaded, js_executed, errors,
                    fallbacks, total_time_seconds
                """
            values = [browser_type, pages_loaded, js_executed, errors, fallbacks, total_time_seconds]
            
            if session_id:
                query += ", session_id"
                values.append(session_id)
                
            query += ") VALUES (" + ", ".join(["%s"] * len(values)) + ") RETURNING id"
            
            cursor.execute(query, values)
            return cursor.fetchone()[0]
    else:
        # For SQLite, log to file or ignore
        logger.info(
            f"Playwright stats - Browser: {browser_type}, Pages: {pages_loaded}, "
            f"Errors: {errors}, Time: {total_time_seconds:.2f}s"
        )
        return None


def get_playwright_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    browser_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get aggregated Playwright statistics.
    
    Args:
        start_date: Start date filter
        end_date: End date filter
        browser_type: Filter by browser type
        
    Returns:
        Dictionary with aggregated statistics
    """
    if USE_POSTGRES:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            conditions = []
            params = []
            
            if start_date:
                conditions.append("created_at >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("created_at <= %s")
                params.append(end_date)
            if browser_type:
                conditions.append("browser_type = %s")
                params.append(browser_type)
                
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor.execute(
                f"""
                SELECT 
                    COUNT(*) as total_sessions,
                    COALESCE(SUM(pages_loaded), 0) as total_pages,
                    COALESCE(SUM(js_executed), 0) as total_js_executed,
                    COALESCE(SUM(errors), 0) as total_errors,
                    COALESCE(SUM(fallbacks), 0) as total_fallbacks,
                    COALESCE(AVG(total_time_seconds / NULLIF(pages_loaded, 0)), 0) as avg_time_per_page,
                    COALESCE(SUM(errors)::float / NULLIF(SUM(pages_loaded), 0), 0) as error_rate,
                    COALESCE(SUM(fallbacks)::float / NULLIF(SUM(pages_loaded), 0), 0) as fallback_rate
                FROM playwright_stats
                WHERE {where_clause}
                """,
                params
            )
            
            result = cursor.fetchone()
            return {
                "total_sessions": result[0],
                "total_pages": result[1],
                "total_js_executed": result[2],
                "total_errors": result[3],
                "total_fallbacks": result[4],
                "avg_time_per_page": round(result[5], 2),
                "error_rate": round(result[6], 4),
                "fallback_rate": round(result[7], 4)
            }
    else:
        # Return empty stats for SQLite
        return {
            "total_sessions": 0,
            "total_pages": 0,
            "total_js_executed": 0,
            "total_errors": 0,
            "total_fallbacks": 0,
            "avg_time_per_page": 0,
            "error_rate": 0,
            "fallback_rate": 0
        }


def update_source_js_requirement(source_name: str, js_required: bool, playwright_settings: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update whether a source requires JavaScript rendering.
    
    Args:
        source_name: Name of the source
        js_required: Whether JS is required
        playwright_settings: Optional Playwright-specific settings
        
    Returns:
        True if successful
    """
    if USE_POSTGRES:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if playwright_settings:
                cursor.execute(
                    """
                    UPDATE sources 
                    SET js_required = %s, playwright_settings = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s
                    """,
                    (js_required, json.dumps(playwright_settings), source_name)
                )
            else:
                cursor.execute(
                    """
                    UPDATE sources 
                    SET js_required = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE name = %s
                    """,
                    (js_required, source_name)
                )
                
            return cursor.rowcount > 0
    else:
        # For SQLite, update would need to be in config file
        logger.info(f"Update sites_to_scrape.json to set js_required={js_required} for {source_name}")
        return False


# Export new functions
__all__ = [
    # All existing exports from db_adapter
    *[name for name in dir() if not name.startswith('_')],
    # New Playwright-specific functions
    'add_article_with_playwright_data',
    'update_article_playwright_metadata',
    'get_articles_by_rendering_method',
    'add_playwright_stats',
    'get_playwright_stats',
    'update_source_js_requirement'
]