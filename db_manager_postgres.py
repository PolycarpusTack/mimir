"""PostgreSQL Database Manager for Mimir News Scraper.

This module provides a PostgreSQL implementation with connection pooling,
prepared statements, and enhanced error handling.
"""

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import psycopg2
import psycopg2.errors
from psycopg2 import extras, pool, sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

logger = logging.getLogger(__name__)

# Database configuration from environment variables
DATABASE_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "mimir_news"),
    "user": os.getenv("POSTGRES_USER", "mimir_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "mimir_dev_password"),
}

# Connection pool configuration
POOL_MIN_CONN = int(os.getenv("DB_POOL_MIN", "2"))
POOL_MAX_CONN = int(os.getenv("DB_POOL_MAX", "20"))

# Use pgBouncer if available
USE_PGBOUNCER = os.getenv("USE_PGBOUNCER", "false").lower() == "true"
if USE_PGBOUNCER:
    DATABASE_CONFIG["port"] = int(os.getenv("PGBOUNCER_PORT", "6432"))


class DatabasePool:
    """Singleton database connection pool."""

    _instance = None
    _pool = None

    def __new__(cls) -> "DatabasePool":
        """Create singleton instance of DatabasePool."""
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize the connection pool."""
        if self._pool is None:
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(POOL_MIN_CONN, POOL_MAX_CONN, **DATABASE_CONFIG)
                logger.info(f"Database pool initialized with {POOL_MIN_CONN}-{POOL_MAX_CONN} connections")
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    def get_connection(self):
        """Get a connection from the pool."""
        if self._pool is None:
            self.initialize()
        return self._pool.getconn()

    def return_connection(self, connection):
        """Return a connection to the pool."""
        if self._pool:
            self._pool.putconn(connection)

    def close_all(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None


# Initialize the pool singleton
db_pool = DatabasePool()


@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic pooling."""
    connection = None
    try:
        connection = db_pool.get_connection()
        connection.autocommit = False
        yield connection
        connection.commit()
    except psycopg2.Error as e:
        if connection:
            connection.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if connection:
            db_pool.return_connection(connection)


def init_db():
    """Initialize database schema using migration scripts."""
    # Check if database exists
    conn = None
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            database="postgres",
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE_CONFIG["database"],))

        if not cursor.fetchone():
            # Create database
            cursor.execute(
                sql.SQL("CREATE DATABASE {} ENCODING 'UTF8'").format(sql.Identifier(DATABASE_CONFIG["database"]))
            )
            logger.info(f"Database '{DATABASE_CONFIG['database']}' created")

        cursor.close()
        conn.close()

        # Initialize the connection pool
        db_pool.initialize()

        # Run migrations
        logger.info("Database initialized. Run 'alembic upgrade head' to apply migrations.")

    except psycopg2.Error as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        if conn:
            conn.close()


def generate_url_hash(url: str) -> str:
    """Generate a consistent hash for a URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def article_exists(url: str) -> bool:
    """Check if an article already exists in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS(SELECT 1 FROM articles WHERE url = %s)", (url,))
        return cursor.fetchone()[0]


def add_article(
    url: str,
    title: str,
    publication_date: Optional[datetime],
    source_website: str,
    category: Optional[str] = None,
    content_summary: str = "",
    full_content: str = "",
    author: Optional[str] = None,
    keywords_found: Optional[List[str]] = None,
) -> Optional[str]:
    """Add a new article to the database.

    Returns:
        Article ID (UUID) if successful, None if article already exists or on error.
    """
    if article_exists(url):
        logger.debug(f"Article already exists: {url}")
        return None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            # Prepare metadata
            metadata = {
                "original_url": url,
                "word_count": len(full_content.split()) if full_content else 0,
                "has_full_content": bool(full_content),
            }

            # Extract domain for source if not provided
            if not source_website:
                parsed_url = urlparse(url)
                source_website = parsed_url.netloc

            cursor.execute(
                """
                INSERT INTO articles (
                    url, title, content, summary, published_date, source,
                    author, category, metadata, tags
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                ),
            )

            article_id = cursor.fetchone()[0]
            logger.info(f"Article added: {title[:50]}... (ID: {article_id})")

            # Add keyword alerts if any keywords were found
            if keywords_found:
                for keyword in keywords_found:
                    add_keyword_alert(article_id, keyword, conn=conn)

            return article_id

        except psycopg2.IntegrityError as e:
            logger.error(f"IntegrityError for URL {url}: {e}")
            return None


def add_keyword_alert(article_id: str, keyword: str, context: str = "", conn=None) -> None:
    """Add a keyword alert for an article."""

    def _add_alert(connection) -> None:
        """Add alert using provided connection."""
        cursor = connection.cursor()
        # Find the keyword ID
        cursor.execute("SELECT id FROM keywords WHERE lower(keyword) = lower(%s) AND is_regex = false", (keyword,))
        result = cursor.fetchone()

        if result:
            keyword_id = result[0]
            cursor.execute(
                """
                INSERT INTO keyword_alerts (article_id, keyword_id, matched_text, context)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, keyword_id, position) DO NOTHING
            """,
                (article_id, keyword_id, keyword, context),
            )
            logger.info(f"Keyword alert added: '{keyword}' for article {article_id}")

    if conn:
        _add_alert(conn)
    else:
        with get_db_connection() as new_conn:
            _add_alert(new_conn)


def start_scrape_run(source_id: Optional[str] = None) -> str:
    """Start a new scrape run and return the ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO scrape_runs (source_id, status)
            VALUES (%s, 'running')
            RETURNING id
        """,
            (source_id,),
        )
        return cursor.fetchone()[0]


def end_scrape_run(
    run_id: str, articles_found: int, articles_new: int, articles_updated: int = 0, errors_count: int = 0
) -> None:
    """End a scrape run with statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE scrape_runs 
            SET ended_at = CURRENT_TIMESTAMP,
                articles_found = %s,
                articles_new = %s,
                articles_updated = %s,
                articles_error = %s,
                status = 'completed',
                duration_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at))
            WHERE id = %s
        """,
            (articles_found, articles_new, articles_updated, errors_count, run_id),
        )


def log_scrape_error(
    run_id: str, source_website: str, error_type: str, error_message: str, url: Optional[str] = None
) -> None:
    """Log a scrape error for debugging."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get source_id from source_website
        cursor.execute("SELECT id FROM sources WHERE name = %s OR url = %s LIMIT 1", (source_website, source_website))
        source_result = cursor.fetchone()
        source_id = source_result[0] if source_result else None

        cursor.execute(
            """
            INSERT INTO scrape_errors (
                scrape_run_id, source_id, url, error_type, 
                error_message, error_details
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (
                run_id,
                source_id,
                url,
                error_type,
                error_message[:500],  # Truncate long messages
                json.dumps({"full_message": error_message}),
            ),
        )


def get_recent_articles(limit: int = 10, hours: int = 24) -> List[Dict[str, Any]]:
    """Get recent articles."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        cursor.execute(
            """
            SELECT id, url, title, summary, published_date, source, 
                   author, category, scraped_at, metadata, tags
            FROM articles 
            WHERE scraped_at > CURRENT_TIMESTAMP - INTERVAL '%s hours'
            ORDER BY published_date DESC NULLS LAST, scraped_at DESC 
            LIMIT %s
        """,
            (hours, limit),
        )
        return cursor.fetchall()


def search_articles(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search articles using PostgreSQL full-text search."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        cursor.execute(
            """
            SELECT 
                id, title, url, published_date, source,
                ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank,
                ts_headline('english', title, websearch_to_tsquery('english', %s),
                           'StartSel=<mark>, StopSel=</mark>') AS highlighted_title,
                ts_headline('english', COALESCE(summary, LEFT(content, 500)), 
                           websearch_to_tsquery('english', %s),
                           'StartSel=<mark>, StopSel=</mark>, MaxWords=50') AS snippet
            FROM articles
            WHERE search_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC, published_date DESC
            LIMIT %s
        """,
            (query, query, query, query, limit),
        )
        return cursor.fetchall()


def get_articles_by_keyword(keyword: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get articles that match a specific keyword."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        cursor.execute(
            """
            SELECT DISTINCT a.*
            FROM articles a
            JOIN keyword_alerts ka ON ka.article_id = a.id
            JOIN keywords k ON k.id = ka.keyword_id
            WHERE lower(k.keyword) = lower(%s)
            ORDER BY a.published_date DESC
            LIMIT %s
        """,
            (keyword, limit),
        )
        return cursor.fetchall()


def mark_articles_as_read() -> int:
    """Mark all new articles as processed."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE articles 
            SET status = 'processed',
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'new'
        """
        )
        updated = cursor.rowcount
        logger.info(f"{updated} articles marked as processed")
        return updated


def get_statistics() -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        stats = {}

        # Total articles
        cursor.execute("SELECT COUNT(*) as total FROM articles")
        stats["total_articles"] = cursor.fetchone()["total"]

        # Articles per source
        cursor.execute(
            """
            SELECT source, COUNT(*) as count 
            FROM articles 
            GROUP BY source
            ORDER BY count DESC
        """
        )
        stats["articles_per_source"] = {row["source"]: row["count"] for row in cursor.fetchall()}

        # Recent scrape runs
        cursor.execute(
            """
            SELECT r.*, s.name as source_name
            FROM scrape_runs r
            LEFT JOIN sources s ON s.id = r.source_id
            ORDER BY r.started_at DESC 
            LIMIT 10
        """
        )
        stats["recent_runs"] = cursor.fetchall()

        # Keyword statistics
        cursor.execute(
            """
            SELECT k.keyword, COUNT(ka.id) as alert_count
            FROM keywords k
            LEFT JOIN keyword_alerts ka ON ka.keyword_id = k.id
            GROUP BY k.id, k.keyword
            ORDER BY alert_count DESC
        """
        )
        stats["keyword_counts"] = {row["keyword"]: row["alert_count"] for row in cursor.fetchall()}

        # Articles by status
        cursor.execute(
            """
            SELECT status, COUNT(*) as count
            FROM articles
            GROUP BY status
        """
        )
        stats["articles_by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Storage statistics
        cursor.execute(
            """
            SELECT 
                pg_size_pretty(pg_database_size(current_database())) as database_size,
                pg_size_pretty(pg_total_relation_size('articles')) as articles_table_size,
                pg_size_pretty(pg_total_relation_size('articles_search_idx')) as search_index_size
        """
        )
        stats["storage"] = cursor.fetchone()

        return stats


def update_source_stats(source_name: str, success: bool = True, error_message: Optional[str] = None) -> None:
    """Update source statistics after scraping."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if success:
            cursor.execute(
                """
                UPDATE sources 
                SET last_scraped_at = CURRENT_TIMESTAMP,
                    total_articles = total_articles + 1,
                    next_scrape_at = CURRENT_TIMESTAMP + 
                        (rate_limit_seconds || ' seconds')::INTERVAL + INTERVAL '4 hours'
                WHERE name = %s
            """,
                (source_name,),
            )
        else:
            cursor.execute(
                """
                UPDATE sources 
                SET failed_attempts = failed_attempts + 1,
                    last_error = %s,
                    last_error_at = CURRENT_TIMESTAMP
                WHERE name = %s
            """,
                (error_message, source_name),
            )


def get_active_sources() -> List[Dict[str, Any]]:
    """Get all active sources for scraping."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        cursor.execute(
            """
            SELECT * FROM sources 
            WHERE enabled = true 
                AND (next_scrape_at IS NULL OR next_scrape_at <= CURRENT_TIMESTAMP)
            ORDER BY 
                CASE WHEN last_scraped_at IS NULL THEN 0 ELSE 1 END,
                last_scraped_at ASC
        """
        )
        return cursor.fetchall()


def cleanup_old_data(days: int = 30) -> Dict[str, int]:
    """Clean up old data from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cleaned = {}

        # Archive old articles
        cursor.execute(
            """
            UPDATE articles 
            SET status = 'archived'
            WHERE status != 'archived' 
                AND scraped_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """,
            (days,),
        )
        cleaned["articles_archived"] = cursor.rowcount

        # Delete old scrape errors
        cursor.execute(
            """
            DELETE FROM scrape_errors 
            WHERE occurred_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """,
            (days * 2,),
        )  # Keep errors longer
        cleaned["errors_deleted"] = cursor.rowcount

        # Delete old notifications
        cursor.execute(
            """
            DELETE FROM notifications 
            WHERE sent = true 
                AND created_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
        """,
            (days,),
        )
        cleaned["notifications_deleted"] = cursor.rowcount

        # Refresh materialized view
        cursor.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY article_stats")

        logger.info(f"Cleanup completed: {cleaned}")
        return cleaned


def get_article_by_id(article_id: str) -> Optional[Dict[str, Any]]:
    """Get a single article by its ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=extras.RealDictCursor)
        cursor.execute(
            """
            SELECT 
                id, title, content, url, published_date, source, 
                author, category, metadata, tags, scraped_at
            FROM articles 
            WHERE id = %s
            """,
            (article_id,),
        )
        result = cursor.fetchone()
        return dict(result) if result else None


def update_article_metadata(article_id: str, metadata: Dict[str, Any]) -> bool:
    """Update the metadata field for an article."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles 
                SET metadata = %s
                WHERE id = %s
                """,
                (json.dumps(metadata), article_id),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update article metadata: {e}")
        return False


def count_articles() -> int:
    """Count the total number of articles."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        return cursor.fetchone()[0]


def count_articles_with_metadata_key(key: str) -> int:
    """Count articles that have a specific key in their metadata."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) 
            FROM articles 
            WHERE metadata ? %s
            """,
            (key,),
        )
        return cursor.fetchone()[0]


def get_summary_format_distribution() -> Dict[str, int]:
    """Get the distribution of summary formats in the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                format_type,
                COUNT(*) as count
            FROM (
                SELECT 
                    jsonb_object_keys(metadata->'summaries') as format_type
                FROM articles 
                WHERE metadata ? 'summaries'
            ) summary_formats
            GROUP BY format_type
            ORDER BY count DESC
            """
        )
        return {row[0]: row[1] for row in cursor.fetchall()}


if __name__ == "__main__":
    # Initialize the database if this script is run directly
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("PostgreSQL database initialized!")

    # Show statistics
    try:
        stats = get_statistics()
        print(f"\nDatabase statistics:")
        print(f"Total articles: {stats['total_articles']}")
        print(f"Articles per source: {stats['articles_per_source']}")
        print(f"Storage: {stats.get('storage', {})}")
    except Exception as e:
        print(f"Could not retrieve statistics: {e}")
        print("Make sure to run 'alembic upgrade head' to apply migrations.")
