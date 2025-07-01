"""
Base spider template for Mimir news scraping
Provides common functionality for RSS and HTML spiders
"""

import scrapy
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse

from ..items import MimirArticleItem


class BaseMimirSpider(scrapy.Spider):
    """Base spider with common functionality for news scraping"""
    
    def __init__(self, site_config: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_config = site_config
        self.name = self._generate_spider_name()
        self.start_urls = [site_config['url']]
        
        # Extract configuration
        self.source_name = site_config.get('name', 'Unknown Source')
        self.category = site_config.get('category', 'general')
        self.enabled = site_config.get('enabled', True)
        self.max_articles = site_config.get('max_articles', 20)
        
        # Custom settings from site config
        self.custom_settings = self._build_custom_settings()
        
    def _generate_spider_name(self) -> str:
        """Generate spider name from site configuration"""
        base_name = self.site_config.get('name', 'unknown')
        spider_type = self.site_config.get('type', 'html')
        
        # Clean name for use as spider identifier
        clean_name = base_name.lower().replace(' ', '_').replace('-', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
        
        return f"{spider_type}_{clean_name}"
    
    def _build_custom_settings(self) -> Dict[str, Any]:
        """Build custom settings from site configuration"""
        settings = {}
        
        # Request delay from site config
        if 'request_delay' in self.site_config:
            settings['DOWNLOAD_DELAY'] = self.site_config['request_delay']
        
        # User agent override
        if 'user_agent' in self.site_config:
            settings['USER_AGENT'] = self.site_config['user_agent']
        
        # Proxy settings override
        if self.site_config.get('use_proxy', False):
            settings['PROXY_SETTINGS'] = {'enabled': True}
        
        # JavaScript rendering settings
        if self.site_config.get('type') == 'playwright':
            settings['DOWNLOAD_HANDLERS'] = {
                'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
                'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            }
            settings['TWISTED_REACTOR'] = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
        
        return settings
    
    def create_article_item(self, **kwargs) -> MimirArticleItem:
        """Create standardized article item with common fields"""
        item = MimirArticleItem()
        
        # Set default values
        item['source_website'] = self.source_name
        item['category'] = self.category
        item['scraper_type'] = 'scrapy'
        item['spider_name'] = self.name
        item['scraped_at'] = datetime.now().isoformat()
        item['metadata'] = {}
        item['tags'] = []
        
        # Update with provided values
        for key, value in kwargs.items():
            if key in item.fields:
                item[key] = value
        
        return item
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common HTML entities
        html_entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text.strip()
    
    def parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format"""
        if not date_str:
            return None
        
        try:
            from dateutil import parser as date_parser
            parsed_date = date_parser.parse(date_str)
            return parsed_date.isoformat()
        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_str}': {e}")
            return date_str  # Return original string if parsing fails
    
    def make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URL to absolute URL"""
        if not url:
            return ""
        
        return urljoin(base_url, url)
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except Exception:
            return ""
    
    def should_follow_link(self, url: str) -> bool:
        """Determine if a link should be followed based on configuration"""
        if not url or not self.is_valid_url(url):
            return False
        
        # Check allowed domains
        allowed_domains = self.site_config.get('allowed_domains', [])
        if allowed_domains:
            domain = self.extract_domain(url)
            return any(allowed_domain in domain for allowed_domain in allowed_domains)
        
        # Check blocked patterns
        blocked_patterns = self.site_config.get('blocked_url_patterns', [])
        for pattern in blocked_patterns:
            if pattern in url:
                return False
        
        return True
    
    def handle_error(self, failure, context: str = ""):
        """Standardized error handling"""
        error_msg = f"Error in {self.name}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {failure.value}"
        
        self.logger.error(error_msg)
        
        # Could send error notifications or store in database
        # For now, just log the error
    
    def closed(self, reason):
        """Called when spider is closed"""
        self.logger.info(f"Spider {self.name} closed: {reason}")
        
        # Log final statistics
        stats = self.crawler.stats.get_stats()
        items_scraped = stats.get('item_scraped_count', 0)
        requests_count = stats.get('downloader/request_count', 0)
        
        self.logger.info(
            f"Final stats - Items: {items_scraped}, "
            f"Requests: {requests_count}, "
            f"Reason: {reason}"
        )