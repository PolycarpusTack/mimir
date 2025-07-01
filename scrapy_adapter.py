"""
Scrapy Adapter for Mimir News Scraper
Provides backward compatibility bridge between legacy scraper.py and Scrapy framework
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile

# Scrapy imports
import scrapy
from scrapy.crawler import CrawlerProcess, CrawlerRunner
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from twisted.internet import defer, reactor
from twisted.internet.defer import inlineCallbacks

logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    spider_name: str
    articles_scraped: int
    articles_new: int
    errors: List[str]
    duration_seconds: float
    stats: Dict[str, Any]


@dataclass
class ScrapyMimirConfig:
    """Configuration for Scrapy-Mimir integration"""
    use_scrapy: bool = False
    scrapy_project_path: Optional[str] = None
    concurrent_spiders: int = 1
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}


class SpiderFactory:
    """Factory for creating Scrapy spiders from Mimir site configurations"""
    
    @staticmethod
    def create_spider_class(site_config: Dict[str, Any]) -> type:
        """Create a Scrapy spider class from Mimir site configuration"""
        
        site_type = site_config.get("type", "html")
        site_name = site_config.get("name", "unknown").lower().replace(" ", "_").replace("-", "_")
        
        if site_type == "rss":
            return SpiderFactory._create_rss_spider_class(site_config, site_name)
        elif site_type == "html":
            return SpiderFactory._create_html_spider_class(site_config, site_name)
        elif site_type == "playwright":
            return SpiderFactory._create_playwright_spider_class(site_config, site_name)
        else:
            raise ValueError(f"Unsupported site type: {site_type}")
    
    @staticmethod
    def _create_rss_spider_class(site_config: Dict[str, Any], spider_name: str) -> type:
        """Create RSS spider class"""
        
        class RSSSpider(scrapy.Spider):
            name = f"rss_{spider_name}"
            start_urls = [site_config["url"]]
            site_config_data = site_config
            
            def parse(self, response):
                """Parse RSS feed"""
                # Use feedparser to parse RSS content
                import feedparser
                
                feed = feedparser.parse(response.text)
                
                for entry in feed.entries:
                    item = {
                        'url': entry.get('link', ''),
                        'title': entry.get('title', 'No title'),
                        'content_summary': entry.get('summary', ''),
                        'full_content': '',
                        'publication_date': self._parse_rss_date(entry),
                        'source_website': self.site_config_data.get('name', ''),
                        'category': self.site_config_data.get('category', 'general'),
                        'author': entry.get('author', ''),
                        'scraper_type': 'scrapy',
                        'spider_name': self.name
                    }
                    
                    # Extract full content if available
                    if 'content' in entry and entry.content:
                        if isinstance(entry.content, list) and entry.content:
                            item['full_content'] = entry.content[0].get('value', '')
                        else:
                            item['full_content'] = str(entry.content)
                    
                    yield item
            
            def _parse_rss_date(self, entry):
                """Parse RSS date to ISO format"""
                pub_date_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
                if pub_date_parsed:
                    return datetime(*pub_date_parsed[:6]).isoformat()
                else:
                    pub_date = entry.get('published') or entry.get('updated', '')
                    if pub_date:
                        try:
                            from dateutil import parser as date_parser
                            return date_parser.parse(pub_date).isoformat()
                        except:
                            return pub_date
                    return datetime.now().isoformat()
        
        return RSSSpider
    
    @staticmethod
    def _create_html_spider_class(site_config: Dict[str, Any], spider_name: str) -> type:
        """Create HTML spider class"""
        
        class HTMLSpider(scrapy.Spider):
            name = f"html_{spider_name}"
            start_urls = [site_config["url"]]
            site_config_data = site_config
            
            def parse(self, response):
                """Parse HTML page for article links"""
                selectors = self.site_config_data.get("selectors", {})
                overview_selector = selectors.get("overview_article_link", "")
                
                if not overview_selector:
                    logger.warning(f"No overview selector for {self.name}")
                    return
                
                # Extract article links
                article_links = response.css(overview_selector)
                
                for link in article_links:
                    article_url = link.css("::attr(href)").get()
                    if article_url:
                        # Make absolute URL
                        article_url = response.urljoin(article_url)
                        
                        # Follow link to parse article
                        yield response.follow(
                            article_url, 
                            self.parse_article,
                            meta={'article_url': article_url}
                        )
                
                # Handle pagination
                pagination = self.site_config_data.get("pagination", {})
                if pagination.get("enabled", False):
                    next_page_selector = pagination.get("next_page_selector", "")
                    if next_page_selector:
                        next_page = response.css(next_page_selector + "::attr(href)").get()
                        if next_page:
                            yield response.follow(next_page, self.parse)
            
            def parse_article(self, response):
                """Parse individual article page"""
                selectors = self.site_config_data.get("selectors", {})
                
                # Extract article data
                title_selector = selectors.get("detail_title", "")
                content_selector = selectors.get("detail_content", "")
                author_selector = selectors.get("detail_author", "")
                date_selector = selectors.get("detail_date", "")
                
                title = response.css(title_selector + "::text").get() or "No title"
                full_content = " ".join(response.css(content_selector + "::text").getall()) or ""
                author = response.css(author_selector + "::text").get() or ""
                date_text = response.css(date_selector + "::text").get() or ""
                
                # Parse date
                publication_date = self._parse_html_date(date_text)
                
                # Create content summary
                content_summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
                
                item = {
                    'url': response.meta.get('article_url', response.url),
                    'title': title.strip(),
                    'content_summary': content_summary,
                    'full_content': full_content[:5000],  # Limit size
                    'publication_date': publication_date,
                    'source_website': self.site_config_data.get('name', ''),
                    'category': self.site_config_data.get('category', 'general'),
                    'author': author.strip(),
                    'scraper_type': 'scrapy',
                    'spider_name': self.name
                }
                
                yield item
            
            def _parse_html_date(self, date_text):
                """Parse HTML date to ISO format"""
                if not date_text:
                    return datetime.now().isoformat()
                
                try:
                    from dateutil import parser as date_parser
                    return date_parser.parse(date_text.strip()).isoformat()
                except:
                    return date_text.strip()
        
        return HTMLSpider
    
    @staticmethod
    def _create_playwright_spider_class(site_config: Dict[str, Any], spider_name: str) -> type:
        """Create Playwright spider class"""
        
        class PlaywrightSpider(scrapy.Spider):
            name = f"playwright_{spider_name}"
            start_urls = [site_config["url"]]
            site_config_data = site_config
            
            custom_settings = {
                'DOWNLOAD_HANDLERS': {
                    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                },
                'TWISTED_REACTOR': "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
            }
            
            def start_requests(self):
                for url in self.start_urls:
                    yield scrapy.Request(
                        url,
                        meta={
                            "playwright": True,
                            "playwright_page_methods": [
                                {"method": "wait_for_load_state", "state": "domcontentloaded"},
                            ]
                        }
                    )
            
            def parse(self, response):
                """Parse JavaScript-rendered page"""
                # Similar to HTML spider but with JS-rendered content
                selectors = self.site_config_data.get("selectors", {})
                overview_selector = selectors.get("overview_article_link", "")
                
                if not overview_selector:
                    logger.warning(f"No overview selector for {self.name}")
                    return
                
                # Extract article links from JS-rendered content
                article_links = response.css(overview_selector)
                
                for link in article_links:
                    article_url = link.css("::attr(href)").get()
                    if article_url:
                        article_url = response.urljoin(article_url)
                        
                        yield scrapy.Request(
                            article_url,
                            self.parse_article,
                            meta={
                                'article_url': article_url,
                                "playwright": True,
                                "playwright_page_methods": [
                                    {"method": "wait_for_load_state", "state": "domcontentloaded"},
                                ]
                            }
                        )
            
            def parse_article(self, response):
                """Parse individual article page with JavaScript"""
                selectors = self.site_config_data.get("selectors", {})
                
                title = response.css(selectors.get("detail_title", "") + "::text").get() or "No title"
                full_content = " ".join(response.css(selectors.get("detail_content", "") + "::text").getall()) or ""
                author = response.css(selectors.get("detail_author", "") + "::text").get() or ""
                date_text = response.css(selectors.get("detail_date", "") + "::text").get() or ""
                
                publication_date = datetime.now().isoformat()
                if date_text:
                    try:
                        from dateutil import parser as date_parser
                        publication_date = date_parser.parse(date_text.strip()).isoformat()
                    except:
                        publication_date = date_text.strip()
                
                content_summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
                
                item = {
                    'url': response.meta.get('article_url', response.url),
                    'title': title.strip(),
                    'content_summary': content_summary,
                    'full_content': full_content[:5000],
                    'publication_date': publication_date,
                    'source_website': self.site_config_data.get('name', ''),
                    'category': self.site_config_data.get('category', 'general'),
                    'author': author.strip(),
                    'scraper_type': 'scrapy',
                    'spider_name': self.name,
                    'rendering_method': 'playwright'
                }
                
                yield item
        
        return PlaywrightSpider


class ConfigTranslator:
    """Translates Mimir configurations to Scrapy settings"""
    
    @staticmethod
    def mimir_to_scrapy_settings(mimir_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Mimir config to Scrapy settings"""
        
        scrapy_settings = {
            # Basic settings
            'BOT_NAME': 'mimir_scrapy',
            'USER_AGENT': mimir_config.get('user_agent', 'Mimir-Scrapy/1.0'),
            
            # Request settings
            'DOWNLOAD_DELAY': mimir_config.get('default_request_delay_seconds', 2),
            'RANDOMIZE_DOWNLOAD_DELAY': 0.5,  # Randomize delays
            'CONCURRENT_REQUESTS': 16,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
            
            # Retry settings
            'RETRY_TIMES': mimir_config.get('max_retries', 3),
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
            
            # Autothrottle settings
            'AUTOTHROTTLE_ENABLED': True,
            'AUTOTHROTTLE_START_DELAY': 1,
            'AUTOTHROTTLE_MAX_DELAY': 60,
            'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
            'AUTOTHROTTLE_DEBUG': False,
            
            # Pipelines
            'ITEM_PIPELINES': {
                'mimir_scrapy.pipelines.ValidationPipeline': 100,
                'mimir_scrapy.pipelines.DuplicationPipeline': 200,
                'mimir_scrapy.pipelines.KeywordPipeline': 300,
                'mimir_scrapy.pipelines.DatabasePipeline': 400,
            },
            
            # Middlewares
            'DOWNLOADER_MIDDLEWARES': {
                'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
                'mimir_scrapy.middlewares.UserAgentMiddleware': 400,
                'mimir_scrapy.middlewares.ProxyMiddleware': 500,
            },
            
            # Logging
            'LOG_LEVEL': mimir_config.get('logging', {}).get('level', 'INFO'),
            
            # Database settings (custom)
            'DATABASE_SETTINGS': {
                'database_name': mimir_config.get('database_name', 'mimir_news.db'),
                'use_postgres': os.getenv('USE_POSTGRES', 'false').lower() == 'true'
            },
            
            # Keyword monitoring (custom)
            'KEYWORD_SETTINGS': mimir_config.get('keywords_monitoring', {}),
            
            # Email settings (custom)
            'EMAIL_SETTINGS': mimir_config.get('notification_email', {}),
        }
        
        # Add proxy settings if available
        proxy_settings = mimir_config.get('proxy_settings', {})
        if proxy_settings.get('enabled', False):
            scrapy_settings['PROXY_SETTINGS'] = proxy_settings
            scrapy_settings['DOWNLOADER_MIDDLEWARES']['mimir_scrapy.middlewares.ProxyMiddleware'] = 100
        
        return scrapy_settings


class ScrapyAdapter:
    """
    Main adapter class that provides backward compatibility between 
    legacy Mimir scraper and Scrapy framework
    """
    
    def __init__(self, mimir_config: Dict[str, Any], sites_config: List[Dict[str, Any]]):
        """
        Initialize ScrapyAdapter
        
        Args:
            mimir_config: Main Mimir configuration (config.json)
            sites_config: Sites configuration (sites_to_scrape.json)
        """
        self.mimir_config = mimir_config
        self.sites_config = sites_config
        self.scrapy_config = ScrapyMimirConfig(
            use_scrapy=mimir_config.get('use_scrapy', False),
            log_level=mimir_config.get('logging', {}).get('level', 'INFO')
        )
        
        # Initialize results tracking
        self.scraping_results: List[ScrapingResult] = []
        self.total_articles_new = 0
        self.total_errors = 0
        
        # Set up Scrapy settings
        self.scrapy_settings = ConfigTranslator.mimir_to_scrapy_settings(mimir_config)
        
        logger.info(f"ScrapyAdapter initialized - use_scrapy: {self.scrapy_config.use_scrapy}")
    
    def run_scraper(self) -> Dict[str, Any]:
        """
        Main entry point that maintains compatibility with existing scraper.py
        
        Returns:
            Dictionary with scraping results and statistics
        """
        if not self.scrapy_config.use_scrapy:
            logger.info("Scrapy disabled, falling back to legacy scraper")
            return self._run_legacy_fallback()
        
        logger.info("Running Scrapy-based scraper")
        
        try:
            # Create spiders from site configurations
            spider_classes = self._create_spiders()
            
            if not spider_classes:
                logger.warning("No spiders created, falling back to legacy scraper")
                return self._run_legacy_fallback()
            
            # Run spiders
            results = self._run_spiders(spider_classes)
            
            # Process results and maintain compatibility
            summary = self._process_results(results)
            
            # Send email notifications (maintain compatibility)
            self._send_email_notifications(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Scrapy scraper failed: {e}")
            logger.info("Falling back to legacy scraper")
            return self._run_legacy_fallback()
    
    def _create_spiders(self) -> List[type]:
        """Create Scrapy spider classes from site configurations"""
        spider_classes = []
        
        for site_config in self.sites_config:
            if not site_config.get('enabled', False):
                logger.debug(f"Skipping disabled site: {site_config.get('name', 'unknown')}")
                continue
            
            try:
                spider_class = SpiderFactory.create_spider_class(site_config)
                spider_classes.append(spider_class)
                logger.debug(f"Created spider for: {site_config.get('name', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to create spider for {site_config.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Created {len(spider_classes)} spiders")
        return spider_classes
    
    def _run_spiders(self, spider_classes: List[type]) -> List[ScrapingResult]:
        """Run Scrapy spiders and collect results"""
        results = []
        
        # Configure Scrapy logging
        configure_logging({'LOG_LEVEL': self.scrapy_config.log_level})
        
        # Create crawler process
        process = CrawlerProcess(self.scrapy_settings)
        
        # Add spiders to process
        for spider_class in spider_classes:
            process.crawl(spider_class)
        
        # Run spiders
        logger.info(f"Starting {len(spider_classes)} spiders")
        start_time = datetime.now()
        
        try:
            process.start()  # This blocks until all spiders finish
        except Exception as e:
            logger.error(f"Spider execution failed: {e}")
            raise
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Spider execution completed in {duration:.2f} seconds")
        
        # Note: In a real implementation, we would collect stats from each spider
        # For now, we'll create a summary result
        results.append(ScrapingResult(
            spider_name="scrapy_batch",
            articles_scraped=0,  # This would be populated from spider stats
            articles_new=0,      # This would be populated from spider stats
            errors=[],           # This would be populated from spider stats
            duration_seconds=duration,
            stats={}            # This would contain Scrapy stats
        ))
        
        return results
    
    def _process_results(self, results: List[ScrapingResult]) -> Dict[str, Any]:
        """Process Scrapy results and maintain compatibility with legacy format"""
        
        total_new_articles = sum(r.articles_new for r in results)
        total_errors = sum(len(r.errors) for r in results)
        total_duration = sum(r.duration_seconds for r in results)
        
        summary = {
            'scraper_type': 'scrapy',
            'total_new_articles': total_new_articles,
            'total_errors': total_errors,
            'total_duration_seconds': total_duration,
            'spider_results': [
                {
                    'spider_name': r.spider_name,
                    'articles_scraped': r.articles_scraped,
                    'articles_new': r.articles_new,
                    'errors': r.errors,
                    'duration_seconds': r.duration_seconds
                }
                for r in results
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Scrapy scraping completed: {total_new_articles} new articles, {total_errors} errors")
        
        return summary
    
    def _send_email_notifications(self, summary: Dict[str, Any]):
        """Send email notifications maintaining compatibility with legacy format"""
        email_settings = self.mimir_config.get('notification_email', {})
        
        if not email_settings.get('send_email', False):
            logger.info("Email notifications disabled")
            return
        
        if summary['total_new_articles'] == 0:
            logger.info("No new articles, skipping email notification")
            return
        
        # Import legacy email function to maintain compatibility
        try:
            from scraper import send_email_summary
            
            # Get recent articles for email (would be populated by database pipeline)
            new_articles_details = []  # This would be fetched from database
            keyword_alerts_details = []  # This would be fetched from keyword pipeline
            
            send_email_summary(new_articles_details, keyword_alerts_details)
            logger.info("Email notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    def _run_legacy_fallback(self) -> Dict[str, Any]:
        """Run legacy scraper as fallback"""
        logger.info("Running legacy scraper fallback")
        
        try:
            # Import and run legacy scraper
            from scraper import run_scraper as legacy_run_scraper
            
            # Run legacy scraper (this doesn't return a value in current implementation)
            legacy_run_scraper()
            
            return {
                'scraper_type': 'legacy',
                'total_new_articles': 0,  # Legacy scraper doesn't return this
                'total_errors': 0,
                'fallback_used': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Legacy scraper fallback failed: {e}")
            raise


def create_scrapy_adapter(config_file: str = "config.json", sites_file: str = "sites_to_scrape.json") -> ScrapyAdapter:
    """
    Factory function to create ScrapyAdapter from configuration files
    
    Args:
        config_file: Path to main configuration file
        sites_file: Path to sites configuration file
        
    Returns:
        Configured ScrapyAdapter instance
    """
    
    # Load configurations
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            mimir_config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_file}: {e}")
        raise
    
    try:
        with open(sites_file, 'r', encoding='utf-8') as f:
            sites_data = json.load(f)
            
            # Handle both old format (direct array) and new format (with "sites" key)
            if isinstance(sites_data, list):
                sites_config = sites_data
            elif isinstance(sites_data, dict) and "sites" in sites_data:
                sites_config = sites_data["sites"]
            else:
                raise ValueError("Invalid sites_to_scrape.json format")
                
    except Exception as e:
        logger.error(f"Failed to load sites file {sites_file}: {e}")
        raise
    
    return ScrapyAdapter(mimir_config, sites_config)


# Compatibility function for existing scraper.py integration
def run_scrapy_scraper() -> Dict[str, Any]:
    """
    Drop-in replacement for existing run_scraper() function
    
    Returns:
        Dictionary with scraping results
    """
    adapter = create_scrapy_adapter()
    return adapter.run_scraper()


if __name__ == "__main__":
    # Command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Mimir Scrapy Adapter")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--sites", default="sites_to_scrape.json", help="Sites configuration file")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    if args.test:
        print("Testing Scrapy adapter...")
        
        # Create test configuration
        test_config = {
            'use_scrapy': True,
            'user_agent': 'Mimir-Scrapy-Test/1.0',
            'default_request_delay_seconds': 1,
            'logging': {'level': 'DEBUG'}
        }
        
        test_sites = [
            {
                'name': 'Test RSS Feed',
                'url': 'http://feeds.feedburner.com/TechCrunch',
                'type': 'rss',
                'enabled': True,
                'category': 'technology'
            }
        ]
        
        adapter = ScrapyAdapter(test_config, test_sites)
        results = adapter.run_scraper()
        
        print("Test results:")
        print(json.dumps(results, indent=2))
    else:
        # Run with provided configuration files
        adapter = create_scrapy_adapter(args.config, args.sites)
        results = adapter.run_scraper()
        
        print("Scraping completed:")
        print(json.dumps(results, indent=2))