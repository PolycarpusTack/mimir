#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script for Mimir

This script migrates all data from SQLite to PostgreSQL while maintaining
data integrity and providing progress tracking.
""ff"

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import os
import sqlite3
import sys

sys.path.append(str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=ff'{asctime}'s - {name}"s - {levelname}'s - {message}"s'
)
logger = logging.getLogger('migration')

class MimirMigration:
    """Handle migration from SQLite to PostgreSQL."""
    
    def __init__(self, sqlite_path: str = SQLITE_DB) -> None:
        """Initialize migration with SQLite database path."""
        self.sqlite_path = sqlite_path
        self.stats = {
            'sources': {'total': 0, 'migrated': 0, 'failed': 0},
            'articles': {'total': 0, 'migrated': 0, 'failed': 0},
            'keywords': {'total': 0, 'migrated': 0, 'failed': 0},
            'keyword_alerts': {'total': 0, 'migrated': 0, 'failed': 0},
            'scrape_runs': {'total': 0, 'migrated': 0, 'failed': 0},
            'scrape_errors': {'total': 0, 'migrated': 0, 'failed': 0}
        }
        self.source_mapping = {}  # Map old source names to new IDs
        self.article_mapping = {}  # Map old article IDs to new UUIDs
        self.keyword_mapping = {}  # Map keywords to IDs
        
    def connect_sqlite(self) -> sqlite3.Connection:
        """Connect to SQLite database."""
        if not os.path.exists(self.sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {self.sqlite_path}")
        
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_sites_config(self) -> List[Dict[str, Any]]:
        """Load sites configuration from sites_to_scrape.json."""
        config_path = Path(__file__).parent / 'sites_to_scrape.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def migrate_sources(self) -> None:
        """Migrate sources from configuration file."""
        logger.info("Migrating sources...")
        
        sites_config = self.load_sites_config()
        if not sites_config:
            logger.warning("No sites configuration found")
            return
        
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            for site in sites_config:
                self.stats['sources']['total'] += 1
                
                try:
                    # Prepare data
                    selectors = site.get('selectors', {})
                    config = {
                        'date_format': site.get('date_format'),
                        'pagination': site.get('pagination', {}),
                        'fetch_full_content': site.get('fetch_full_content', True)
                    }
                    
                    # Insert source
                    pg_cursor.execute("""
                        INSERT INTO sources (
                            name, url, type, category, enabled, config, selectors
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (name) DO UPDATE SET
                            url = EXCLUDED.url,
                            type = EXCLUDED.type,
                            category = EXCLUDED.category,
                            config = EXCLUDED.config,
                            selectors = EXCLUDED.selectors,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """, (
                        site['name'],
                        site['url'],
                        site.get('type', 'rss'),
                        site.get('category'),
                        site.get('enabled', True),
                        json.dumps(config),
                        json.dumps(selectors)
                    ))
                    
                    source_id = pg_cursor.fetchone()[0]
                    self.source_mapping[site['name']] = source_id
                    self.stats['sources']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate source {site.get('name')}: {e}")
                    self.stats['sources']['failed'] += 1
        
        logger.info(f"Sources migrated: {self.stats['sources']['migrated']}/{self.stats['sources']['total']}")
    
    def migrate_keywords(self) -> None:
        """Migrate keywords from config.json."""
        logger.info("Migrating keywords...")
        
        # Load config.json
        config_path = Path(__file__).parent / 'config.json'
        if not config_path.exists():
            logger.warning("No config.json found")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        keywords_config = config.get('keywords_monitoring', {})
        keywords = keywords_config.get('keywords', [])
        
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            for keyword in keywords:
                self.stats['keywords']['total'] += 1
                
                try:
                    pg_cursor.execute("""
                        INSERT INTO keywords (keyword, priority, enabled)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (lower(keyword)) WHERE is_regex = false
                        DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """, (keyword, 5, keywords_config.get('enabled', True)))
                    
                    keyword_id = pg_cursor.fetchone()[0]
                    self.keyword_mapping[keyword.lower()] = keyword_id
                    self.stats['keywords']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate keyword {keyword}: {e}")
                    self.stats['keywords']['failed'] += 1
        
        logger.info(f"Keywords migrated: {self.stats['keywords']['migrated']}/{self.stats['keywords']['total']}")
    
    def migrate_articles(self, batch_size: int = 1000) -> None:
        """Migrate articles from SQLite to PostgreSQL in batches."""
        logger.info("Migrating articles...")
        
        sqlite_conn = self.connect_sqlite()
        sqlite_cursor = sqlite_conn.cursor()
        
        # Count total articles
        sqlite_cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = sqlite_cursor.fetchone()[0]
        self.stats['articles']['total'] = total_articles
        
        logger.info(f"Found {total_articles} articles to migrate")
        
        # Migrate in batches
        offset = 0
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            while offset < total_articles:
                # Fetch batch from SQLite
                sqlite_cursor.execute("""
                    SELECT * FROM articles 
                    ORDER BY id 
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))
                
                articles = sqlite_cursor.fetchall()
                
                for article in articles:
                    try:
                        # Parse keywords if stored as JSON
                        keywords = []
                        if article['keywords_found']:
                            try:
                                keywords = json.loads(article['keywords_found'])
                            except (json.JSONDecodeError, TypeError):
                                keywords = [article['keywords_found']]
                        
                        # Prepare metadata
                        metadata = {
                            'sqlite_id': article['id'],
                            'is_new': bool(article['is_new']),
                            'has_full_content': bool(article['full_content'])
                        }
                        
                        # Map source
                        source = article['source_website']
                        if source in self.source_mapping:
                            metadata['source_id'] = str(self.source_mapping[source])
                        
                        # Insert article
                        pg_cursor.execute("""
                            INSERT INTO articles (
                                url, title, content, summary, published_date,
                                source, author, category, metadata, tags,
                                scraped_at, status
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            ) ON CONFLICT (url) DO NOTHING
                            RETURNING id
                        """, (
                            article['url'],
                            article['title'],
                            article['full_content'] or article['content_summary'],
                            article['content_summary'],
                            self._parse_date(article['publication_date']),
                            source,
                            article['author'],
                            article['category'],
                            json.dumps(metadata),
                            json.dumps(keywords),
                            self._parse_date(article['scraped_at']),
                            'new' if article['is_new'] else 'processed'
                        ))
                        
                        result = pg_cursor.fetchone()
                        if result:
                            self.article_mapping[article['id']] = result[0]
                            self.stats['articles']['migrated'] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to migrate article {article['id']}: {e}")
                        self.stats['articles']['failed'] += 1
                
                # Progress update
                offset += batch_size
                progress = min(offset, total_articles)
                logger.info(f"Progress: {progress}/{total_articles} articles ({progress*100/total_articles:.1f}%)")
                
                # Commit batch
                pg_conn.commit()
        
        sqlite_conn.close()
        logger.info(f"Articles migrated: {self.stats['articles']['migrated']}/{self.stats['articles']['total']}")
    
    def migrate_keyword_alerts(self) -> None:
        """Migrate keyword alerts."""
        logger.info("Migrating keyword alerts...")
        
        sqlite_conn = self.connect_sqlite()
        sqlite_cursor = sqlite_conn.cursor()
        
        sqlite_cursor.execute("SELECT COUNT(*) FROM keyword_alerts")
        self.stats['keyword_alerts']['total'] = sqlite_cursor.fetchone()[0]
        
        sqlite_cursor.execute("SELECT * FROM keyword_alerts")
        alerts = sqlite_cursor.fetchall()
        
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            for alert in alerts:
                try:
                    # Skip if article wasn't migrated
                    if alert['article_id'] not in self.article_mapping:
                        continue
                    
                    # Find keyword ID
                    keyword_lower = alert['keyword'].lower()
                    if keyword_lower not in self.keyword_mapping:
                        # Create keyword if it doesn't exist
                        pg_cursor.execute("""
                            INSERT INTO keywords (keyword, priority)
                            VALUES (%s, 5)
                            ON CONFLICT (lower(keyword)) WHERE is_regex = false
                            DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                            RETURNING id
                        """, (alert['keyword'],))
                        self.keyword_mapping[keyword_lower] = pg_cursor.fetchone()[0]
                    
                    # Insert alert
                    pg_cursor.execute("""
                        INSERT INTO keyword_alerts (
                            article_id, keyword_id, matched_text, context, created_at
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        self.article_mapping[alert['article_id']],
                        self.keyword_mapping[keyword_lower],
                        alert['keyword'],
                        alert['context'],
                        self._parse_date(alert['created_at'])
                    ))
                    
                    self.stats['keyword_alerts']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate keyword alert {alert['id']}: {e}")
                    self.stats['keyword_alerts']['failed'] += 1
        
        sqlite_conn.close()
        logger.info(f"Keyword alerts migrated: {self.stats['keyword_alerts']['migrated']}/{self.stats['keyword_alerts']['total']}")
    
    def migrate_scrape_runs(self) -> None:
        """Migrate scrape run history."""
        logger.info("Migrating scrape runs...")
        
        sqlite_conn = self.connect_sqlite()
        sqlite_cursor = sqlite_conn.cursor()
        
        sqlite_cursor.execute("SELECT COUNT(*) FROM scrape_runs")
        self.stats['scrape_runs']['total'] = sqlite_cursor.fetchone()[0]
        
        sqlite_cursor.execute("SELECT * FROM scrape_runs ORDER BY started_at")
        runs = sqlite_cursor.fetchall()
        
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            for run in runs:
                try:
                    # Calculate duration
                    duration = None
                    if run['started_at'] and run['ended_at']:
                        start = self._parse_date(run['started_at'])
                        end = self._parse_date(run['ended_at'])
                        if start and end:
                            duration = (end - start).total_seconds()
                    
                    pg_cursor.execute("""
                        INSERT INTO scrape_runs (
                            started_at, ended_at, articles_found, articles_new,
                            articles_error, status, duration_seconds
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        self._parse_date(run['started_at']),
                        self._parse_date(run['ended_at']),
                        run['articles_found'],
                        run['articles_new'],
                        run['errors_count'],
                        run['status'] or 'completed',
                        duration
                    ))
                    
                    self.stats['scrape_runs']['migrated'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate scrape run {run['id']}: {e}")
                    self.stats['scrape_runs']['failed'] += 1
        
        sqlite_conn.close()
        logger.info(f"Scrape runs migrated: {self.stats['scrape_runs']['migrated']}/{self.stats['scrape_runs']['total']}")
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Try ISO format first
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            try:
                # Try common formats
                from dateutil import parser
                return parser.parse(date_str)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse date: {date_str} - {e}")
                return None
    
    def validate_data(self) -> bool:
        """Validate data integrity before migration."""
        logger.info("Validating source data...")
        
        issues = []
        
        # Check SQLite database
        sqlite_conn = self.connect_sqlite()
        sqlite_cursor = sqlite_conn.cursor()
        
        # Validate articles
        sqlite_cursor.execute("""
            SELECT COUNT(*) FROM articles 
            WHERE url IS NULL OR title IS NULL
        """)
        invalid_articles = sqlite_cursor.fetchone()[0]
        if invalid_articles > 0:
            issues.append(f"{invalid_articles} articles with missing required fields")
        
        # Check for duplicate URLs
        sqlite_cursor.execute("""
            SELECT url, COUNT(*) as cnt 
            FROM articles 
            GROUP BY url 
            HAVING cnt > 1
        """)
        duplicates = sqlite_cursor.fetchall()
        if duplicates:
            issues.append(f"{len(duplicates)} duplicate URLs found")
        
        sqlite_conn.close()
        
        if issues:
            logger.error("Data validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ Data validation passed")
        return True
    
    def create_rollback_point(self) -> str:
        """Create a rollback point before migration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rollback_id = f"migration_{timestamp}"
        
        with pg_db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SAVEPOINT {rollback_id}")
            
        logger.info(f"Created rollback point: {rollback_id}")
        return rollback_id
    
    def rollback(self, rollback_id: str) -> None:
        """Rollback to a previous state."""
        logger.warning(f"Rolling back to {rollback_id}...")
        
        with pg_db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"ROLLBACK TO SAVEPOINT {rollback_id}")
            
        logger.info("Rollback completed")
    
    def verify_migration(self) -> Dict[str, Any]:
        """Verify migration success by comparing counts."""
        logger.info("Verifying migration...")
        
        verification = {}
        
        # Get PostgreSQL counts
        with pg_db.get_db_connection() as pg_conn:
            pg_cursor = pg_conn.cursor()
            
            tables = ['articles', 'sources', 'keywords', 'keyword_alerts', 'scrape_runs']
            for table in tables:
                pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = pg_cursor.fetchone()[0]
                verification[table] = {
                    'postgresql_count': count,
                    'migrated': self.stats.get(table, {}).get('migrated', 0),
                    'failed': self.stats.get(table, {}).get('failed', 0)
                }
        
        return verification
    
    def print_summary(self) -> None:
        """Print migration summary."""
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        
        for table, stats in self.stats.items():
            if stats['total'] > 0:
                success_rate = (stats['migrated'] / stats['total']) * 100
                print(f"\n{table.upper()}:")
                print(f"  Total: {stats['total']}")
                print(f"  Migrated: {stats['migrated']} ({success_rate:.1f}%)")
                print(f"  Failed: {stats['failed']}")
        
        print("\n" + "="*60)
        
        # Verification
        verification = self.verify_migration()
        print("\nVERIFICATION:")
        for table, counts in verification.items():
            print(f"\n{table}: {counts['postgresql_count']} records in PostgreSQL")
    
    def run(self) -> None:
        """Run the complete migration."""
        logger.info("Starting Mimir SQLite to PostgreSQL migration...")
        
        try:
            # Validate data first
            if not self.validate_data():
                logger.error("Data validation failed. Please fix issues before migrating.")
                return
            
            # Initialize PostgreSQL
            pg_db.init_db()
            
            # Create rollback point
            rollback_id = self.create_rollback_point()
            
            try:
                # Run migrations in order
                self.migrate_sources()
                self.migrate_keywords()
                self.migrate_articles()
                self.migrate_keyword_alerts()
                self.migrate_scrape_runs()
                
                # Verify migration
                verification = self.verify_migration()
                
                # Check if verification passed
                all_good = True
                for table, data in verification.items():
                    if data.get('migrated', 0) < data.get('postgresql_count', 0) * 0.95:
                        all_good = False
                        logger.error(f"Migration verification failed for {table}")
                
                if not all_good:
                    self.rollback(rollback_id)
                    raise RuntimeError("Migration verification failed")
                
                # Print summary
                self.print_summary()
                
                logger.info("Migration completed successfully!")
                
            except Exception as e:
                logger.error(f"Migration failed, rolling back: {e}")
                self.rollback(rollback_id)
                raise
                
        except Exception as e:
            logger.error(f"Migration setup failed: {e}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Mimir from SQLite to PostgreSQL")
    parser.add_argument('--sqlite-db', default=SQLITE_DB, help='Path to SQLite database')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for article migration')
    args = parser.parse_args()
    
    migration = MimirMigration(args.sqlite_db)
    migration.run()

if __name__ == '__main__':
    main()