import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DATABASE_NAME = "mimir_news.db"
logger = logging.getLogger(__name__)


@contextmanager
def get_db_connection():
    """Context manager voor database connecties."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """Initialiseer de database met alle benodigde tabellen."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Articles tabel
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            publication_date TEXT,
            source_website TEXT NOT NULL,
            category TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content_summary TEXT,
            full_content TEXT,
            author TEXT,
            keywords_found TEXT,
            is_new BOOLEAN DEFAULT 1
        )
        """
        )

        # Keywords monitoring tabel
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS keyword_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            keyword TEXT NOT NULL,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES articles (id)
        )
        """
        )

        # Scrape runs tabel voor statistieken
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            articles_found INTEGER DEFAULT 0,
            articles_new INTEGER DEFAULT 0,
            errors_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running'
        )
        """
        )

        # Errors tabel voor debugging
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS scrape_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            source_website TEXT,
            error_type TEXT,
            error_message TEXT,
            url TEXT,
            occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES scrape_runs (id)
        )
        """
        )

        logger.info(f"Database '{DATABASE_NAME}' geïnitialiseerd met alle tabellen.")


def article_exists(url):
    """Controleer of een artikel al bestaat in de database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
        return cursor.fetchone() is not None


def add_article(
    url,
    title,
    publication_date,
    source_website,
    category=None,
    content_summary="",
    full_content="",
    author=None,
    keywords_found=None,
):
    """Voeg een nieuw artikel toe aan de database."""
    if article_exists(url):
        logger.debug(f"Artikel bestaat al: {url}")
        return None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            keywords_json = json.dumps(keywords_found) if keywords_found else None
            cursor.execute(
                """
            INSERT INTO articles (url, title, publication_date, source_website, 
                                category, content_summary, full_content, author, keywords_found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    url,
                    title,
                    publication_date,
                    source_website,
                    category,
                    content_summary,
                    full_content,
                    author,
                    keywords_json,
                ),
            )

            article_id = cursor.lastrowid
            logger.info(f"Artikel toegevoegd: {title[:50]}... (ID: {article_id})")
            return article_id
        except sqlite3.IntegrityError as e:
            logger.error(f"IntegrityError voor URL {url}: {e}")
            return None


def add_keyword_alert(article_id, keyword, context=""):
    """Voeg een keyword alert toe."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        INSERT INTO keyword_alerts (article_id, keyword, context)
        VALUES (?, ?, ?)
        """,
            (article_id, keyword, context),
        )
        logger.info(f"Keyword alert toegevoegd: '{keyword}' voor artikel {article_id}")


def start_scrape_run():
    """Start een nieuwe scrape run en return het ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        INSERT INTO scrape_runs (started_at) VALUES (?)
        """,
            (datetime.now().isoformat(),),
        )
        return cursor.lastrowid


def end_scrape_run(run_id, articles_found, articles_new, errors_count):
    """Beëindig een scrape run met statistieken."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        UPDATE scrape_runs 
        SET ended_at = ?, articles_found = ?, articles_new = ?, 
            errors_count = ?, status = ?
        WHERE id = ?
        """,
            (datetime.now().isoformat(), articles_found, articles_new, errors_count, "completed", run_id),
        )


def log_scrape_error(run_id, source_website, error_type, error_message, url=None):
    """Log een scrape error voor debugging."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        INSERT INTO scrape_errors (run_id, source_website, error_type, error_message, url)
        VALUES (?, ?, ?, ?, ?)
        """,
            (run_id, source_website, error_type, str(error_message)[:500], url),
        )


def get_recent_articles(limit=10, hours=24):
    """Haal recente artikelen op."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        SELECT * FROM articles 
        WHERE datetime(scraped_at) > datetime('now', '-' || ? || ' hours')
        ORDER BY scraped_at DESC 
        LIMIT ?
        """,
            (hours, limit),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_articles_by_keyword(keyword, limit=50):
    """Zoek artikelen die een specifiek keyword bevatten."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        search_pattern = f"%{keyword}%"
        cursor.execute(
            """
        SELECT * FROM articles 
        WHERE title LIKE ? OR content_summary LIKE ? OR keywords_found LIKE ?
        ORDER BY scraped_at DESC 
        LIMIT ?
        """,
            (search_pattern, search_pattern, search_pattern, limit),
        )
        return [dict(row) for row in cursor.fetchall()]


def mark_articles_as_read():
    """Markeer alle artikelen als gelezen (is_new = 0)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE articles SET is_new = 0 WHERE is_new = 1")
        updated = cursor.rowcount
        logger.info(f"{updated} artikelen gemarkeerd als gelezen")
        return updated


def get_statistics():
    """Haal algemene statistieken op."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        stats = {}

        # Totaal aantal artikelen
        cursor.execute("SELECT COUNT(*) as total FROM articles")
        stats["total_articles"] = cursor.fetchone()["total"]

        # Artikelen per bron
        cursor.execute(
            """
        SELECT source_website, COUNT(*) as count 
        FROM articles 
        GROUP BY source_website
        """
        )
        stats["articles_per_source"] = {row["source_website"]: row["count"] for row in cursor.fetchall()}

        # Recente scrape runs
        cursor.execute(
            """
        SELECT * FROM scrape_runs 
        ORDER BY started_at DESC 
        LIMIT 10
        """
        )
        stats["recent_runs"] = [dict(row) for row in cursor.fetchall()]

        # Keyword alerts
        cursor.execute(
            """
        SELECT keyword, COUNT(*) as count 
        FROM keyword_alerts 
        GROUP BY keyword 
        ORDER BY count DESC
        """
        )
        stats["keyword_counts"] = {row["keyword"]: row["count"] for row in cursor.fetchall()}

        return stats


if __name__ == "__main__":
    # Initialiseer de database als dit script direct wordt uitgevoerd
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database geïnitialiseerd!")

    # Toon statistieken
    stats = get_statistics()
    print(f"\nDatabase statistieken:")
    print(f"Totaal aantal artikelen: {stats['total_articles']}")
    print(f"Artikelen per bron: {stats['articles_per_source']}")
