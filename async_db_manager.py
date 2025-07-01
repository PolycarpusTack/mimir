"""
Async Database Manager for Mimir Scraper
EPIC 4 - Critical Fix: Async database layer to replace sync db_manager calls
"""

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiosqlite
import asyncpg

logger = logging.getLogger(__name__)


class AsyncDatabaseManager:
    """Async database manager supporting both SQLite and PostgreSQL."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_type = self._determine_db_type()
        self.connection_pool = None
        self.sqlite_path = None

        # Database connection settings
        if self.db_type == "postgresql":
            self.pg_config = self._get_postgresql_config()
        else:
            self.sqlite_path = config.get("database_name", "mimir_news.db")

    def _determine_db_type(self) -> str:
        """Determine which database type to use."""
        use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
        if use_postgres or "postgresql" in self.config:
            return "postgresql"
        return "sqlite"

    def _get_postgresql_config(self) -> Dict[str, Any]:
        """Get PostgreSQL configuration."""
        pg_config = self.config.get("postgresql", {})
        return {
            "host": os.getenv("DB_HOST", pg_config.get("host", "localhost")),
            "port": int(os.getenv("DB_PORT", pg_config.get("port", 5432))),
            "database": os.getenv("DB_NAME", pg_config.get("database", "mimir")),
            "user": os.getenv("DB_USER", pg_config.get("user", "mimir")),
            "password": os.getenv("DB_PASSWORD", pg_config.get("password", "")),
            "min_size": pg_config.get("min_pool_size", 5),
            "max_size": pg_config.get("max_pool_size", 20),
        }

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            if self.db_type == "postgresql":
                await self._init_postgresql()
            else:
                await self._init_sqlite()

            logger.info(f"Async database manager initialized ({self.db_type})")

        except Exception as e:
            logger.error(f"Failed to initialize async database: {e}")
            raise

    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.pg_config["host"],
                port=self.pg_config["port"],
                database=self.pg_config["database"],
                user=self.pg_config["user"],
                password=self.pg_config["password"],
                min_size=self.pg_config["min_size"],
                max_size=self.pg_config["max_size"],
                command_timeout=30,
            )

            # Test connection
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise

    async def _init_sqlite(self):
        """Initialize SQLite (create file if needed)."""
        # For SQLite, we'll create connections on demand
        # Test that we can create a connection
        async with aiosqlite.connect(self.sqlite_path) as conn:
            await conn.execute("SELECT 1")

    async def close(self):
        """Close database connections."""
        try:
            if self.db_type == "postgresql" and self.connection_pool:
                await self.connection_pool.close()
            logger.info("Async database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection (context manager)."""
        if self.db_type == "postgresql":
            async with self.connection_pool.acquire() as conn:
                yield conn
        else:
            async with aiosqlite.connect(self.sqlite_path) as conn:
                conn.row_factory = aiosqlite.Row
                yield conn

    async def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a SELECT query and return results."""
        try:
            async with self.get_connection() as conn:
                if self.db_type == "postgresql":
                    if params:
                        result = await conn.fetch(query, *params)
                    else:
                        result = await conn.fetch(query)
                    return [dict(row) for row in result]
                else:
                    if params:
                        cursor = await conn.execute(query, params)
                    else:
                        cursor = await conn.execute(query)
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def execute_command(self, query: str, params: tuple = None) -> Optional[int]:
        """Execute an INSERT/UPDATE/DELETE command."""
        try:
            async with self.get_connection() as conn:
                if self.db_type == "postgresql":
                    if params:
                        result = await conn.execute(query, *params)
                    else:
                        result = await conn.execute(query)
                    # Extract affected rows from result string
                    return int(result.split()[-1]) if result.split()[-1].isdigit() else 1
                else:
                    if params:
                        cursor = await conn.execute(query, params)
                    else:
                        cursor = await conn.execute(query)
                    await conn.commit()
                    return cursor.rowcount

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise

    async def get_last_insert_id(self, conn) -> Optional[int]:
        """Get last inserted row ID."""
        if self.db_type == "postgresql":
            # For PostgreSQL, use RETURNING clause in INSERT statements
            return None
        else:
            return conn.lastrowid

    # Article management methods
    async def article_exists(self, url: str) -> bool:
        """Check if article exists by URL."""
        query = (
            "SELECT 1 FROM articles WHERE url = ? LIMIT 1"
            if self.db_type == "sqlite"
            else "SELECT 1 FROM articles WHERE url = $1 LIMIT 1"
        )

        try:
            result = await self.execute_query(query, (url,))
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking article existence: {e}")
            return False

    async def add_article(
        self,
        url: str,
        title: str,
        publication_date: str,
        source_website: str,
        category: str = "general",
        content_summary: str = "",
        full_content: str = "",
        author: str = "",
        keywords_found: List[str] = None,
    ) -> Optional[int]:
        """Add new article to database."""
        if keywords_found is None:
            keywords_found = []

        try:
            # Check if article already exists
            if await self.article_exists(url):
                logger.debug(f"Article already exists: {url}")
                return None

            if self.db_type == "postgresql":
                query = """
                INSERT INTO articles (url, title, publication_date, source_website, 
                                    category, content_summary, full_content, author, 
                                    keywords_found, scraped_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """
                params = (
                    url,
                    title,
                    publication_date,
                    source_website,
                    category,
                    content_summary,
                    full_content,
                    author,
                    json.dumps(keywords_found),
                    datetime.now().isoformat(),
                )

                async with self.get_connection() as conn:
                    result = await conn.fetchval(query, *params)
                    return result

            else:
                query = """
                INSERT INTO articles (url, title, publication_date, source_website,
                                    category, content_summary, full_content, author,
                                    keywords_found, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    url,
                    title,
                    publication_date,
                    source_website,
                    category,
                    content_summary,
                    full_content,
                    author,
                    json.dumps(keywords_found),
                    datetime.now().isoformat(),
                )

                async with self.get_connection() as conn:
                    cursor = await conn.execute(query, params)
                    await conn.commit()
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error adding article: {e}")
            return None

    async def add_keyword_alert(self, article_id: int, keyword: str, context: str) -> bool:
        """Add keyword alert to database."""
        try:
            if self.db_type == "postgresql":
                query = """
                INSERT INTO keyword_alerts (article_id, keyword, context, created_at)
                VALUES ($1, $2, $3, $4)
                """
            else:
                query = """
                INSERT INTO keyword_alerts (article_id, keyword, context, created_at)
                VALUES (?, ?, ?, ?)
                """

            params = (article_id, keyword, context, datetime.now().isoformat())
            await self.execute_command(query, params)
            return True

        except Exception as e:
            logger.error(f"Error adding keyword alert: {e}")
            return False

    async def get_recent_articles(self, limit: int = 50, hours: int = 24) -> List[Dict]:
        """Get recent articles from the database."""
        try:
            if self.db_type == "postgresql":
                query = (
                    """
                SELECT * FROM articles 
                WHERE scraped_at > NOW() - INTERVAL '%s hours'
                ORDER BY scraped_at DESC 
                LIMIT $1
                """
                    % hours
                )
                params = (limit,)
            else:
                query = """
                SELECT * FROM articles 
                WHERE scraped_at > datetime('now', '-{} hours')
                ORDER BY scraped_at DESC 
                LIMIT ?
                """.format(
                    hours
                )
                params = (limit,)

            return await self.execute_query(query, params)

        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []

    # Scrape run management
    async def start_scrape_run(self) -> int:
        """Start a new scrape run and return run ID."""
        try:
            if self.db_type == "postgresql":
                query = """
                INSERT INTO scrape_runs (started_at, status)
                VALUES ($1, $2)
                RETURNING id
                """
                params = (datetime.now().isoformat(), "running")

                async with self.get_connection() as conn:
                    return await conn.fetchval(query, *params)
            else:
                query = """
                INSERT INTO scrape_runs (started_at, status)
                VALUES (?, ?)
                """
                params = (datetime.now().isoformat(), "running")

                async with self.get_connection() as conn:
                    cursor = await conn.execute(query, params)
                    await conn.commit()
                    return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error starting scrape run: {e}")
            # Return a timestamp-based ID as fallback
            return int(datetime.now().timestamp() * 1000)

    async def end_scrape_run(self, run_id: int, articles_found: int, articles_new: int, errors: int) -> bool:
        """End a scrape run with results."""
        try:
            if self.db_type == "postgresql":
                query = """
                UPDATE scrape_runs 
                SET ended_at = $1, status = $2, articles_found = $3, 
                    articles_new = $4, errors = $5
                WHERE id = $6
                """
            else:
                query = """
                UPDATE scrape_runs 
                SET ended_at = ?, status = ?, articles_found = ?, 
                    articles_new = ?, errors = ?
                WHERE id = ?
                """

            params = (datetime.now().isoformat(), "completed", articles_found, articles_new, errors, run_id)

            await self.execute_command(query, params)
            return True

        except Exception as e:
            logger.error(f"Error ending scrape run: {e}")
            return False

    async def log_scrape_error(
        self, run_id: int, source: str, error_type: str, error_message: str, url: str = ""
    ) -> bool:
        """Log a scrape error."""
        try:
            if self.db_type == "postgresql":
                query = """
                INSERT INTO scrape_errors (run_id, source, error_type, error_message, url, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
            else:
                query = """
                INSERT INTO scrape_errors (run_id, source, error_type, error_message, url, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """

            params = (run_id, source, error_type, error_message, url, datetime.now().isoformat())
            await self.execute_command(query, params)
            return True

        except Exception as e:
            logger.error(f"Error logging scrape error: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}

            # Total articles
            total_query = "SELECT COUNT(*) as count FROM articles"
            result = await self.execute_query(total_query)
            stats["total_articles"] = result[0]["count"] if result else 0

            # Articles per source
            source_query = """
            SELECT source_website, COUNT(*) as count 
            FROM articles 
            GROUP BY source_website 
            ORDER BY count DESC
            LIMIT 10
            """
            result = await self.execute_query(source_query)
            stats["articles_per_source"] = {row["source_website"]: row["count"] for row in result}

            # Recent scrape runs
            if self.db_type == "postgresql":
                runs_query = """
                SELECT * FROM scrape_runs 
                ORDER BY started_at DESC 
                LIMIT 10
                """
            else:
                runs_query = """
                SELECT * FROM scrape_runs 
                ORDER BY started_at DESC 
                LIMIT 10
                """

            result = await self.execute_query(runs_query)
            stats["recent_runs"] = result

            # Keyword counts (if keyword_alerts table exists)
            try:
                keyword_query = """
                SELECT keyword, COUNT(*) as count 
                FROM keyword_alerts 
                GROUP BY keyword 
                ORDER BY count DESC 
                LIMIT 10
                """
                result = await self.execute_query(keyword_query)
                stats["keyword_counts"] = {row["keyword"]: row["count"] for row in result}
            except:
                stats["keyword_counts"] = {}

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"total_articles": 0, "articles_per_source": {}, "recent_runs": [], "keyword_counts": {}}


# Compatibility adapter for existing code
class AsyncDBAdapter:
    """Adapter to provide sync-like interface for async database operations."""

    def __init__(self, async_db_manager: AsyncDatabaseManager):
        self.async_db = async_db_manager
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    def _run_async(self, coro):
        """Run async function in sync context."""
        loop = self._get_loop()
        if loop.is_running():
            # If loop is already running, create a task
            task = asyncio.create_task(coro)
            # Wait for completion (this should be avoided in production)
            return loop.run_until_complete(task)
        else:
            return loop.run_until_complete(coro)

    # Sync wrappers for async methods
    def article_exists(self, url: str) -> bool:
        return self._run_async(self.async_db.article_exists(url))

    def add_article(self, **kwargs) -> Optional[int]:
        return self._run_async(self.async_db.add_article(**kwargs))

    def add_keyword_alert(self, article_id: int, keyword: str, context: str) -> bool:
        return self._run_async(self.async_db.add_keyword_alert(article_id, keyword, context))

    def start_scrape_run(self) -> int:
        return self._run_async(self.async_db.start_scrape_run())

    def end_scrape_run(self, run_id: int, articles_found: int, articles_new: int, errors: int) -> bool:
        return self._run_async(self.async_db.end_scrape_run(run_id, articles_found, articles_new, errors))

    def log_scrape_error(self, run_id: int, source: str, error_type: str, error_message: str, url: str = "") -> bool:
        return self._run_async(self.async_db.log_scrape_error(run_id, source, error_type, error_message, url))

    def get_recent_articles(self, limit: int = 50, hours: int = 24) -> List[Dict]:
        return self._run_async(self.async_db.get_recent_articles(limit, hours))

    def get_statistics(self) -> Dict[str, Any]:
        return self._run_async(self.async_db.get_statistics())


# Global instance management
_async_db_manager = None
_async_db_adapter = None


async def initialize_async_db(config: Dict[str, Any]):
    """Initialize global async database manager."""
    global _async_db_manager, _async_db_adapter

    _async_db_manager = AsyncDatabaseManager(config)
    await _async_db_manager.initialize()
    _async_db_adapter = AsyncDBAdapter(_async_db_manager)


def get_async_db() -> AsyncDatabaseManager:
    """Get async database manager instance."""
    if _async_db_manager is None:
        raise RuntimeError("Async database not initialized. Call initialize_async_db() first.")
    return _async_db_manager


def get_db_adapter() -> AsyncDBAdapter:
    """Get sync-compatible database adapter."""
    if _async_db_adapter is None:
        raise RuntimeError("Async database not initialized. Call initialize_async_db() first.")
    return _async_db_adapter


async def close_async_db():
    """Close async database connections."""
    global _async_db_manager
    if _async_db_manager:
        await _async_db_manager.close()
        _async_db_manager = None
