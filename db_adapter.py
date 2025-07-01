"""Database adapter for backward compatibility.

This module provides a unified interface that can use either SQLite or PostgreSQL
based on configuration, allowing gradual migration without breaking existing code.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Determine which database to use
USE_POSTGRES = os.getenv("USE_POSTGRES", "false").lower() == "true"

if USE_POSTGRES:
    try:
        import db_manager_postgres as db_impl

        logger.info("Using PostgreSQL database")
    except ImportError as e:
        logger.error(f"Failed to import PostgreSQL module: {e}")
        logger.info("Falling back to SQLite")
        import db_manager as db_impl

        USE_POSTGRES = False
else:
    import db_manager as db_impl

    logger.info("Using SQLite database")

# Export all functions from the selected implementation
get_db_connection = db_impl.get_db_connection
init_db = db_impl.init_db
article_exists = db_impl.article_exists
add_article = db_impl.add_article
add_keyword_alert = db_impl.add_keyword_alert
start_scrape_run = db_impl.start_scrape_run
end_scrape_run = db_impl.end_scrape_run
log_scrape_error = db_impl.log_scrape_error
get_recent_articles = db_impl.get_recent_articles
mark_articles_as_read = db_impl.mark_articles_as_read
get_statistics = db_impl.get_statistics

# Additional functions for EPIC 8 (Summarization)
get_article_by_id = db_impl.get_article_by_id
update_article_metadata = db_impl.update_article_metadata
count_articles = db_impl.count_articles
count_articles_with_metadata_key = db_impl.count_articles_with_metadata_key
get_summary_format_distribution = db_impl.get_summary_format_distribution

# Handle functions that might have different signatures
if USE_POSTGRES:
    # PostgreSQL uses search_articles
    get_articles_by_keyword = db_impl.search_articles

    # Add PostgreSQL-specific functions
    update_source_stats = db_impl.update_source_stats
    get_active_sources = db_impl.get_active_sources
    cleanup_old_data = db_impl.cleanup_old_data
else:
    # SQLite uses get_articles_by_keyword
    get_articles_by_keyword = db_impl.get_articles_by_keyword

    # Provide stub implementations for PostgreSQL-only functions
    def update_source_stats(source_name: str, success: bool = True, error_message: Optional[str] = None) -> None:
        """Stub for SQLite - no source stats tracking."""
        pass

    def get_active_sources() -> List[Dict[str, Any]]:
        """Get sources from config file for SQLite."""
        import json

        try:
            with open("sites_to_scrape.json", "r") as f:
                sites = json.load(f)
                return [s for s in sites if s.get("enabled", True)]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def cleanup_old_data(days: int = 30) -> Dict[str, int]:
        """Stub for SQLite - no cleanup implemented."""
        return {"status": "not implemented for SQLite"}


# Adapter function to handle different signatures
def add_article_compatible(
    url: str,
    title: str,
    publication_date: Any,
    source_website: str,
    category: Optional[str] = None,
    content_summary: str = "",
    full_content: str = "",
    author: Optional[str] = None,
    keywords_found: Optional[List[str]] = None,
) -> Optional[Any]:
    """Add article with compatible signature for both databases."""
    if USE_POSTGRES:
        # PostgreSQL expects datetime object
        from datetime import datetime

        if isinstance(publication_date, str):
            try:
                from dateutil import parser

                publication_date = parser.parse(publication_date)
            except (ValueError, TypeError):
                publication_date = None

        return add_article(
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
    else:
        # SQLite expects string
        if publication_date and not isinstance(publication_date, str):
            publication_date = publication_date.isoformat()

        return add_article(
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


# Feature detection
def get_database_type() -> str:
    """Return the current database type."""
    return "PostgreSQL" if USE_POSTGRES else "SQLite"


def supports_full_text_search() -> bool:
    """Check if full-text search is supported."""
    return USE_POSTGRES


def supports_source_management() -> bool:
    """Check if source management is supported."""
    return USE_POSTGRES


# Migration helper
def check_migration_status() -> Dict[str, Any]:
    """Check if migration to PostgreSQL is needed or completed."""
    status = {
        "current_db": get_database_type(),
        "postgres_available": False,
        "migration_needed": False,
        "sqlite_articles": 0,
        "postgres_articles": 0,
    }

    # Check SQLite
    try:
        import db_manager as sqlite_db

        with sqlite_db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            status["sqlite_articles"] = cursor.fetchone()[0]
    except Exception as e:
        logger.debug(f"Could not check SQLite: {e}")

    # Check PostgreSQL
    try:
        import db_manager_postgres as pg_db

        pg_db.db_pool.initialize()
        stats = pg_db.get_statistics()
        status["postgres_available"] = True
        status["postgres_articles"] = stats.get("total_articles", 0)
    except Exception as e:
        logger.debug(f"Could not check PostgreSQL: {e}")

    # Determine if migration is needed
    if status["sqlite_articles"] > 0 and status["postgres_articles"] == 0:
        status["migration_needed"] = True

    return status


# AI Analysis Extensions
def update_article_metadata(article_id: str, metadata: Dict[str, Any]) -> bool:
    """Update article metadata with AI analysis results."""
    if USE_POSTGRES:
        # PostgreSQL has native JSONB support
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # Use JSONB concatenation to merge metadata
                cursor.execute(
                    """
                    UPDATE articles 
                    SET metadata = COALESCE(metadata, '{}'::jsonb) || %s::jsonb,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """,
                    (json.dumps(metadata), article_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to update article metadata: {e}")
            return False
    else:
        # SQLite doesn't have metadata column yet
        logger.warning("Metadata updates not supported in SQLite mode")
        return False


def count_analyzed_articles() -> int:
    """Count articles that have been analyzed by AI."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM articles 
                    WHERE metadata ? 'ai_analysis'
                """
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count analyzed articles: {e}")
            return 0
    else:
        # SQLite approximation - count articles with content
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM articles 
                    WHERE content IS NOT NULL AND content != ''
                """
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count articles: {e}")
            return 0


def get_unanalyzed_articles(limit: int = 100) -> List[Dict[str, Any]]:
    """Get articles that haven't been analyzed yet."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, url, title, content, source, published_date, category
                    FROM articles 
                    WHERE NOT (metadata ? 'ai_analysis')
                    AND content IS NOT NULL AND content != ''
                    ORDER BY published_date DESC
                    LIMIT %s
                """,
                    (limit,),
                )

                articles = []
                for row in cursor.fetchall():
                    articles.append(
                        {
                            "id": row[0],
                            "url": row[1],
                            "title": row[2],
                            "content": row[3],
                            "source": row[4],
                            "published_date": row[5],
                            "category": row[6],
                        }
                    )
                return articles
        except Exception as e:
            logger.error(f"Failed to get unanalyzed articles: {e}")
            return []
    else:
        # SQLite - just get recent articles
        return get_recent_articles(limit)


def get_article_by_id(article_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific article by ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if USE_POSTGRES:
                cursor.execute(
                    """
                    SELECT id, url, title, content, source, published_date, 
                           category, metadata
                    FROM articles WHERE id = %s
                """,
                    (article_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT url, title, content, source, published_date, category
                    FROM articles WHERE url = ?
                """,
                    (article_id,),
                )  # SQLite uses URL as ID

            row = cursor.fetchone()
            if row:
                if USE_POSTGRES:
                    return {
                        "id": row[0],
                        "url": row[1],
                        "title": row[2],
                        "content": row[3],
                        "source": row[4],
                        "published_date": row[5],
                        "category": row[6],
                        "metadata": row[7],
                    }
                else:
                    return {
                        "id": row[0],  # URL as ID for SQLite
                        "url": row[0],
                        "title": row[1],
                        "content": row[2],
                        "source": row[3],
                        "published_date": row[4],
                        "category": row[5],
                    }
            return None
    except Exception as e:
        logger.error(f"Failed to get article by ID: {e}")
        return None


def get_article_count_by_language() -> Dict[str, int]:
    """Get count of articles by detected language."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT metadata->>'language' as lang, COUNT(*)
                    FROM articles 
                    WHERE metadata ? 'language'
                    GROUP BY lang
                    ORDER BY COUNT(*) DESC
                """
                )
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get language counts: {e}")
            return {}
    else:
        return {"unknown": count_analyzed_articles()}


def get_sentiment_distribution() -> Dict[str, int]:
    """Get distribution of article sentiments."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT metadata->'sentiment'->>'label' as sentiment, COUNT(*)
                    FROM articles 
                    WHERE metadata->'sentiment'->>'label' IS NOT NULL
                    GROUP BY sentiment
                    ORDER BY COUNT(*) DESC
                """
                )
                return {row[0]: row[1] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get sentiment distribution: {e}")
            return {}
    else:
        return {}


def get_top_entities(entity_type: str = "organization", limit: int = 10) -> List[Tuple[str, int]]:
    """Get most mentioned entities of a specific type."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT entity, COUNT(*) as count
                    FROM articles,
                    jsonb_array_elements_text(metadata->'entities'->%s) as entity
                    GROUP BY entity
                    ORDER BY count DESC
                    LIMIT %s
                """,
                    (entity_type, limit),
                )
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get top entities: {e}")
            return []
    else:
        return []


def get_top_keywords(limit: int = 20) -> List[Tuple[str, int]]:
    """Get most common keywords across articles."""
    if USE_POSTGRES:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT keyword, COUNT(*) as count
                    FROM articles,
                    jsonb_array_elements_text(metadata->'keywords') as keyword
                    GROUP BY keyword
                    ORDER BY count DESC
                    LIMIT %s
                """,
                    (limit,),
                )
                return [(row[0], row[1]) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get top keywords: {e}")
            return []
    else:
        return []


# Required import for JSON operations
import json


# Database connection wrapper
def get_connection():
    """Get a database connection object with extended methods."""

    class DBConnection:
        """TODO: Add class docstring."""

        """TODO: Add class docstring."""

        def __init__(self):
            self.db_type = get_database_type()

        def update_article_metadata(self, article_id: str, metadata: Dict[str, Any]) -> bool:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return update_article_metadata(article_id, metadata)

        def count_analyzed_articles(self) -> int:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return count_analyzed_articles()

        def get_unanalyzed_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_unanalyzed_articles(limit)

        def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_article_by_id(article_id)

        def get_article_count_by_language(self) -> Dict[str, int]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_article_count_by_language()

        def get_sentiment_distribution(self) -> Dict[str, int]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_sentiment_distribution()

        def get_top_entities(self, entity_type: str = "organization", limit: int = 10) -> List[Tuple[str, int]]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_top_entities(entity_type, limit)

        def get_top_keywords(self, limit: int = 20) -> List[Tuple[str, int]]:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            return get_top_keywords(limit)

    return DBConnection()


# Print adapter info on import
logger.info(f"Database adapter initialized: {get_database_type()}")
