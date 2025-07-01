"""
Scrapy pipelines for Mimir news scraper
Handles validation, deduplication, keyword matching, and database storage
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import scrapy
from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Validates scraped articles before processing"""
    
    def __init__(self):
        self.required_fields = ['url', 'title']
        self.min_title_length = 5
        self.max_title_length = 500
        self.max_content_length = 50000
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # Check required fields
        for field in self.required_fields:
            if not adapter.get(field):
                raise DropItem(f"Missing required field: {field}")
        
        # Validate title
        title = adapter.get('title', '').strip()
        if len(title) < self.min_title_length:
            raise DropItem(f"Title too short: {title}")
        
        if len(title) > self.max_title_length:
            adapter['title'] = title[:self.max_title_length] + "..."
            spider.logger.warning(f"Title truncated for: {adapter.get('url')}")
        
        # Validate URL format
        url = adapter.get('url', '').strip()
        if not self._is_valid_url(url):
            raise DropItem(f"Invalid URL format: {url}")
        
        # Limit content length
        content = adapter.get('full_content', '')
        if content and len(content) > self.max_content_length:
            adapter['full_content'] = content[:self.max_content_length] + "..."
            spider.logger.warning(f"Content truncated for: {url}")
        
        # Ensure metadata is a dict
        if not adapter.get('metadata'):
            adapter['metadata'] = {}
        
        # Set scraped timestamp
        adapter['scraped_at'] = datetime.now().isoformat()
        
        spider.logger.debug(f"Article validated: {title}")
        return item
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None


class DuplicationPipeline:
    """Detects and filters duplicate articles"""
    
    def __init__(self):
        self.seen_urls = set()
        self.seen_content_hashes = set()
        self.duplicate_threshold = 0.8  # 80% similarity
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        url = adapter.get('url')
        
        # Check URL duplicates
        if url in self.seen_urls:
            adapter['duplicate_check'] = 'url_duplicate'
            raise DropItem(f"Duplicate URL: {url}")
        
        # Check content similarity
        content = adapter.get('full_content', '') or adapter.get('content_summary', '')
        if content:
            content_hash = self._calculate_content_hash(content)
            
            # Simple duplicate detection based on content hash
            if content_hash in self.seen_content_hashes:
                adapter['duplicate_check'] = 'content_duplicate'
                raise DropItem(f"Duplicate content detected: {url}")
            
            self.seen_content_hashes.add(content_hash)
        
        self.seen_urls.add(url)
        adapter['duplicate_check'] = 'unique'
        
        spider.logger.debug(f"Article uniqueness verified: {url}")
        return item
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for duplicate detection"""
        # Normalize content - remove extra whitespace, lowercase
        normalized = ' '.join(content.lower().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()


class KeywordPipeline:
    """Checks articles for keyword matches and calculates relevance"""
    
    def __init__(self, keyword_settings: Optional[Dict] = None):
        self.keyword_settings = keyword_settings or {}
        self.keywords = self.keyword_settings.get('keywords_to_monitor', [])
        self.alert_keywords = self.keyword_settings.get('alert_keywords', [])
        self.case_sensitive = self.keyword_settings.get('case_sensitive', False)
    
    @classmethod
    def from_crawler(cls, crawler):
        """Initialize pipeline from crawler settings"""
        keyword_settings = crawler.settings.get('KEYWORD_SETTINGS', {})
        return cls(keyword_settings)
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # Extract text for keyword matching
        title = adapter.get('title', '')
        content = adapter.get('full_content', '') or adapter.get('content_summary', '')
        text_to_search = f"{title} {content}"
        
        if not self.case_sensitive:
            text_to_search = text_to_search.lower()
            keywords = [k.lower() for k in self.keywords]
            alert_keywords = [k.lower() for k in self.alert_keywords]
        else:
            keywords = self.keywords
            alert_keywords = self.alert_keywords
        
        # Find keyword matches
        matches = []
        for keyword in keywords:
            if keyword in text_to_search:
                matches.append(keyword)
        
        # Find alert keyword matches
        alert_matches = []
        for keyword in alert_keywords:
            if keyword in text_to_search:
                alert_matches.append(keyword)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance(matches, len(keywords))
        
        # Store results
        adapter['keyword_matches'] = {
            'regular_keywords': matches,
            'alert_keywords': alert_matches,
            'total_matches': len(matches) + len(alert_matches)
        }
        adapter['relevance_score'] = relevance_score
        
        # Log significant matches
        if matches or alert_matches:
            spider.logger.info(
                f"Keyword matches found in '{title}': "
                f"{len(matches)} regular, {len(alert_matches)} alerts"
            )
        
        return item
    
    def _calculate_relevance(self, matches: List[str], total_keywords: int) -> float:
        """Calculate relevance score based on keyword matches"""
        if total_keywords == 0:
            return 0.5  # Neutral score
        
        match_ratio = len(matches) / total_keywords
        # Scale to give higher scores for more matches
        return min(1.0, match_ratio * 2.0)


class DatabasePipeline:
    """Stores articles in the database with compatibility layer"""
    
    def __init__(self, database_settings: Optional[Dict] = None):
        self.database_settings = database_settings or {}
        self.use_postgres = self.database_settings.get('use_postgres', False)
        self.database_name = self.database_settings.get('database_name', 'mimir_news.db')
        
        # Import appropriate database manager
        self.db_manager = None
        
    @classmethod
    def from_crawler(cls, crawler):
        """Initialize pipeline from crawler settings"""
        database_settings = crawler.settings.get('DATABASE_SETTINGS', {})
        return cls(database_settings)
    
    def open_spider(self, spider):
        """Initialize database connection when spider opens"""
        try:
            if self.use_postgres:
                # Import async database if available
                try:
                    from db_manager_postgres import DatabaseManager
                    self.db_manager = DatabaseManager()
                    spider.logger.info("Using PostgreSQL database")
                except ImportError:
                    spider.logger.warning("PostgreSQL not available, falling back to SQLite")
                    import db_manager
                    self.db_manager = db_manager
            else:
                import db_manager
                self.db_manager = db_manager
                spider.logger.info("Using SQLite database")
                
        except Exception as e:
            spider.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def close_spider(self, spider):
        """Clean up database connections when spider closes"""
        if hasattr(self.db_manager, 'close_connection'):
            try:
                self.db_manager.close_connection()
                spider.logger.info("Database connection closed")
            except Exception as e:
                spider.logger.error(f"Error closing database connection: {e}")
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        try:
            # Prepare article data for database storage
            article_data = {
                'url': adapter.get('url'),
                'title': adapter.get('title'),
                'content_summary': adapter.get('content_summary', ''),
                'full_content': adapter.get('full_content', ''),
                'publication_date': self._parse_date(adapter.get('publication_date')),
                'source_website': adapter.get('source_website', ''),
                'category': adapter.get('category', 'general'),
                'author': adapter.get('author', ''),
                'scraped_at': adapter.get('scraped_at'),
                'metadata': json.dumps(adapter.get('metadata', {})),
                'tags': json.dumps(adapter.get('tags', [])),
                'relevance_score': adapter.get('relevance_score', 0.0),
                'rendering_method': adapter.get('rendering_method', 'scrapy')
            }
            
            # Store in database
            if hasattr(self.db_manager, 'add_article'):
                # New database manager interface
                article_id = self.db_manager.add_article(**article_data)
            else:
                # Legacy database manager interface
                article_id = self.db_manager.insert_article(
                    url=article_data['url'],
                    title=article_data['title'],
                    content=article_data['full_content'],
                    published_date=article_data['publication_date'],
                    source=article_data['source_website'],
                    author=article_data['author'],
                    category=article_data['category']
                )
            
            spider.logger.info(f"Article stored with ID: {article_id}")
            
            # Store keyword matches if any
            keyword_matches = adapter.get('keyword_matches', {})
            if keyword_matches.get('alert_keywords'):
                self._store_keyword_alerts(article_id, keyword_matches, spider)
            
            return item
            
        except Exception as e:
            spider.logger.error(f"Failed to store article in database: {e}")
            raise DropItem(f"Database storage failed: {e}")
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse date string to database format"""
        if not date_str:
            return None
        
        try:
            if isinstance(date_str, datetime):
                return date_str.isoformat()
            
            from dateutil import parser as date_parser
            parsed_date = date_parser.parse(date_str)
            return parsed_date.isoformat()
        except Exception:
            return date_str  # Return as-is if parsing fails
    
    def _store_keyword_alerts(self, article_id: int, keyword_matches: Dict, spider):
        """Store keyword alerts for important matches"""
        try:
            alert_keywords = keyword_matches.get('alert_keywords', [])
            
            for keyword in alert_keywords:
                if hasattr(self.db_manager, 'add_keyword_alert'):
                    self.db_manager.add_keyword_alert(
                        article_id=article_id,
                        keyword=keyword,
                        alert_date=datetime.now().isoformat()
                    )
                
            spider.logger.info(f"Stored {len(alert_keywords)} keyword alerts for article {article_id}")
            
        except Exception as e:
            spider.logger.error(f"Failed to store keyword alerts: {e}")


class StatsPipeline:
    """Collects and reports scraping statistics"""
    
    def __init__(self):
        self.items_processed = 0
        self.items_dropped = 0
        self.start_time = None
        self.keyword_stats = {}
        self.source_stats = {}
    
    def open_spider(self, spider):
        """Initialize stats when spider opens"""
        self.start_time = datetime.now()
        spider.logger.info("Stats collection started")
    
    def close_spider(self, spider):
        """Report final stats when spider closes"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        stats_report = {
            'duration_seconds': duration,
            'items_processed': self.items_processed,
            'items_dropped': self.items_dropped,
            'success_rate': self.items_processed / max(1, self.items_processed + self.items_dropped),
            'processing_rate': self.items_processed / max(1, duration),
            'keyword_stats': self.keyword_stats,
            'source_stats': self.source_stats
        }
        
        spider.logger.info(f"Scraping completed - Stats: {json.dumps(stats_report, indent=2)}")
        
        # Store stats in spider for external access
        spider.crawler.stats.set_value('mimir_pipeline_stats', stats_report)
    
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        self.items_processed += 1
        
        # Track source statistics
        source = adapter.get('source_website', 'unknown')
        self.source_stats[source] = self.source_stats.get(source, 0) + 1
        
        # Track keyword statistics
        keyword_matches = adapter.get('keyword_matches', {})
        total_matches = keyword_matches.get('total_matches', 0)
        if total_matches > 0:
            self.keyword_stats[source] = self.keyword_stats.get(source, 0) + total_matches
        
        return item