"""Comprehensive test suite for PostgreSQL database implementation."""

import os
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import db_manager_postgres as db


class TestDatabasePool(unittest.TestCase):
    """Test database connection pooling."""

    def setUp(self):
        """Set up test environment."""
        os.environ["POSTGRES_DB"] = "mimir_test"
        self.pool = db.DatabasePool()

    def test_singleton_pattern(self):
        """Test that DatabasePool is a singleton."""
        pool1 = db.DatabasePool()
        pool2 = db.DatabasePool()
        self.assertIs(pool1, pool2)

    def test_connection_pool_initialization(self):
        """Test pool initialization."""
        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool:
            self.pool._pool = None
            self.pool.initialize()
            mock_pool.assert_called_once()

    def test_connection_lifecycle(self):
        """Test getting and returning connections."""
        with patch("psycopg2.pool.ThreadedConnectionPool") as mock_pool_class:
            mock_pool = MagicMock()
            mock_pool_class.return_value = mock_pool

            self.pool._pool = None
            self.pool.initialize()

            # Get connection
            mock_conn = MagicMock()
            mock_pool.getconn.return_value = mock_conn
            conn = self.pool.get_connection()

            self.assertEqual(conn, mock_conn)
            mock_pool.getconn.assert_called_once()

            # Return connection
            self.pool.return_connection(conn)
            mock_pool.putconn.assert_called_once_with(conn)


class TestDatabaseOperations(unittest.TestCase):
    """Test core database operations."""

    @patch("db_manager_postgres.get_db_connection")
    def test_article_exists(self, mock_db):
        """Test checking if article exists."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Test article exists
        mock_cursor.fetchone.return_value = (True,)
        result = db.article_exists("https://example.com/article")
        self.assertTrue(result)

        # Test article doesn't exist
        mock_cursor.fetchone.return_value = (False,)
        result = db.article_exists("https://example.com/new-article")
        self.assertFalse(result)

    @patch("db_manager_postgres.get_db_connection")
    @patch("db_manager_postgres.article_exists")
    def test_add_article(self, mock_exists, mock_db):
        """Test adding a new article."""
        mock_exists.return_value = False

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock the UUID return
        test_uuid = str(uuid.uuid4())
        mock_cursor.fetchone.return_value = (test_uuid,)

        # Add article
        article_id = db.add_article(
            url="https://example.com/test",
            title="Test Article",
            publication_date=datetime.now(),
            source_website="Test Source",
            category="test",
            content_summary="Summary",
            full_content="Full content",
            author="Test Author",
            keywords_found=["test", "keyword"],
        )

        self.assertEqual(article_id, test_uuid)
        mock_cursor.execute.assert_called()

    @patch("db_manager_postgres.get_db_connection")
    def test_search_articles(self, mock_db):
        """Test full-text search functionality."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock search results
        mock_results = [
            {
                "id": str(uuid.uuid4()),
                "title": "AI Article",
                "url": "https://example.com/ai",
                "rank": 0.75,
                "snippet": "About <mark>AI</mark>...",
            }
        ]
        mock_cursor.fetchall.return_value = mock_results

        results = db.search_articles("artificial intelligence")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "AI Article")

    @patch("db_manager_postgres.get_db_connection")
    def test_get_statistics(self, mock_db):
        """Test statistics gathering."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock various statistics queries
        mock_cursor.fetchone.side_effect = [
            {"total": 100},  # Total articles
            {"database_size": "50 MB", "articles_table_size": "40 MB", "search_index_size": "5 MB"},  # Storage
        ]

        mock_cursor.fetchall.side_effect = [
            [{"source": "Source1", "count": 50}, {"source": "Source2", "count": 50}],  # Articles per source
            [],  # Recent runs
            [{"keyword": "AI", "alert_count": 10}],  # Keyword counts
            [{"status": "processed", "count": 100}],  # Articles by status
        ]

        stats = db.get_statistics()

        self.assertEqual(stats["total_articles"], 100)
        self.assertIn("Source1", stats["articles_per_source"])
        self.assertIn("storage", stats)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    @patch("db_manager_postgres.get_db_connection")
    @patch("db_manager_postgres.article_exists")
    def test_add_duplicate_article(self, mock_exists, mock_db):
        """Test adding duplicate article."""
        mock_exists.return_value = True

        result = db.add_article(
            url="https://example.com/duplicate",
            title="Duplicate Article",
            publication_date=datetime.now(),
            source_website="Test Source",
        )

        self.assertIsNone(result)
        mock_db.assert_not_called()

    @patch("db_manager_postgres.get_db_connection")
    def test_connection_error_handling(self, mock_db):
        """Test handling of connection errors."""
        import psycopg2

        mock_db.side_effect = psycopg2.OperationalError("Connection failed")

        with self.assertRaises(psycopg2.OperationalError):
            db.get_recent_articles()

    def test_generate_url_hash(self):
        """Test URL hash generation."""
        url = "https://example.com/article"
        hash1 = db.generate_url_hash(url)
        hash2 = db.generate_url_hash(url)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length


class TestDataMigration(unittest.TestCase):
    """Test data migration utilities."""

    @patch("db_manager_postgres.get_db_connection")
    def test_cleanup_old_data(self, mock_db):
        """Test old data cleanup."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Mock cleanup operations
        mock_cursor.rowcount = 10

        result = db.cleanup_old_data(days=30)

        self.assertIn("articles_archived", result)
        self.assertEqual(result["articles_archived"], 10)

    @patch("db_manager_postgres.get_db_connection")
    def test_update_source_stats(self, mock_db):
        """Test source statistics update."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Test successful update
        db.update_source_stats("Test Source", success=True)
        mock_cursor.execute.assert_called()

        # Test failed update
        db.update_source_stats("Test Source", success=False, error_message="Network error")
        self.assertEqual(mock_cursor.execute.call_count, 2)


class TestPerformance(unittest.TestCase):
    """Test performance-related features."""

    def test_connection_pool_config(self):
        """Test connection pool configuration."""
        self.assertEqual(db.POOL_MIN_CONN, 2)
        self.assertEqual(db.POOL_MAX_CONN, 20)

    @patch.dict(os.environ, {"USE_PGBOUNCER": "true", "PGBOUNCER_PORT": "6432"})
    def test_pgbouncer_configuration(self):
        """Test pgBouncer configuration when enabled."""
        # Reload module to pick up environment changes
        import importlib

        importlib.reload(db)

        self.assertTrue(db.USE_PGBOUNCER)
        self.assertEqual(db.DATABASE_CONFIG["port"], 6432)


class TestIntegration(unittest.TestCase):
    """Integration tests (requires running PostgreSQL)."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        cls.skip_integration = os.getenv("SKIP_INTEGRATION_TESTS", "true").lower() == "true"
        if cls.skip_integration:
            return

        # Use test database
        os.environ["POSTGRES_DB"] = "mimir_test"
        db.db_pool._pool = None  # Reset pool

        try:
            db.init_db()
        except Exception as e:
            print(f"Skipping integration tests: {e}")
            cls.skip_integration = True

    def setUp(self):
        """Set up each test."""
        if self.skip_integration:
            self.skipTest("Integration tests skipped (no PostgreSQL)")

    def test_full_article_lifecycle(self):
        """Test complete article lifecycle."""
        # Add article
        article_id = db.add_article(
            url=f"https://example.com/integration-test-{uuid.uuid4()}",
            title="Integration Test Article",
            publication_date=datetime.now(),
            source_website="Integration Test",
            content_summary="Test summary",
            keywords_found=["test", "integration"],
        )

        self.assertIsNotNone(article_id)

        # Search for it
        results = db.search_articles("integration test")
        self.assertTrue(any(r["id"] == article_id for r in results))

        # Get recent articles
        recent = db.get_recent_articles(limit=10, hours=1)
        self.assertTrue(any(a["id"] == article_id for a in recent))

        # Clean up
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM articles WHERE id = %s", (article_id,))


if __name__ == "__main__":
    unittest.main()
