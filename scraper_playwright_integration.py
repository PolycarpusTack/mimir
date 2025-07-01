"""
Integration module to add Playwright support to the existing scraper.
This module extends the main scraper with JavaScript rendering capabilities.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from config_loader import config_loader
from playwright_scraper import PlaywrightScraper
from db_adapter_playwright import (
    add_article_with_playwright_data,
    add_playwright_stats,
    update_source_js_requirement
)

logger = logging.getLogger(__name__)


class ScraperWithPlaywright:
    """
    Enhanced scraper that integrates Playwright for JavaScript-heavy sites.
    Falls back to regular scraping when Playwright is not needed or fails.
    """
    
    def __init__(self):
        """Initialize the enhanced scraper."""
        self.config = config_loader.load_config()
        self.playwright_config = self.config.get('playwright_scraping', {})
        self.playwright_enabled = self.playwright_config.get('enabled', False)
        self.playwright_scraper = None
        self.stats = {
            'playwright_used': 0,
            'playwright_success': 0,
            'playwright_failed': 0,
            'fallback_used': 0
        }
        
    async def initialize_playwright(self):
        """Initialize Playwright scraper if enabled."""
        if self.playwright_enabled and not self.playwright_scraper:
            try:
                self.playwright_scraper = PlaywrightScraper(self.config)
                await self.playwright_scraper.initialize()
                logger.info("Playwright scraper initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright: {e}")
                self.playwright_enabled = False
                
    async def cleanup_playwright(self):
        """Clean up Playwright resources."""
        if self.playwright_scraper:
            try:
                # Save statistics
                stats = self.playwright_scraper.get_stats()
                if stats['pages_loaded'] > 0:
                    add_playwright_stats(
                        browser_type='chromium',
                        pages_loaded=stats['pages_loaded'],
                        js_executed=stats['js_executed'],
                        errors=stats['errors'],
                        fallbacks=stats['fallbacks'],
                        total_time_seconds=stats['total_time']
                    )
                    
                await self.playwright_scraper.cleanup()
                logger.info("Playwright scraper cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Playwright: {e}")
                
    def should_use_playwright(self, site_config: Dict, html_content: Optional[str] = None) -> bool:
        """
        Determine if Playwright should be used for a site.
        
        Args:
            site_config: Site configuration
            html_content: Optional HTML content to check
            
        Returns:
            True if Playwright should be used
        """
        if not self.playwright_enabled or not self.playwright_scraper:
            return False
            
        # Check explicit js_required flag
        if site_config.get('js_required', False):
            return True
            
        # Check if Playwright scraper can detect JS requirement
        if self.playwright_scraper and html_content:
            return self.playwright_scraper.is_javascript_required(site_config, html_content)
            
        return False
        
    async def parse_html_site_with_playwright(
        self,
        site_config: Dict,
        run_id: int,
        fetch_url_func,
        extract_keywords_func,
        parse_date_func
    ) -> Tuple[int, List[Dict]]:
        """
        Parse HTML site with Playwright support.
        
        Args:
            site_config: Site configuration
            run_id: Current scrape run ID
            fetch_url_func: Function to fetch URL content
            extract_keywords_func: Function to extract keywords
            parse_date_func: Function to parse dates
            
        Returns:
            Tuple of (new articles count, article details list)
        """
        base_url = site_config["url"]
        new_articles_count = 0
        articles_added = []
        source_name = site_config["name"]
        category = site_config.get("category", "general")
        selectors = site_config.get("selectors", {})
        
        # Keywords configuration
        keywords_config = self.config.get("keywords_monitoring", {})
        monitor_keywords = keywords_config.get("enabled", False)
        keywords_list = keywords_config.get("keywords", [])
        
        # Check if we should use Playwright
        initial_html = fetch_url_func(base_url)
        use_playwright = self.should_use_playwright(site_config, initial_html)
        
        if use_playwright:
            logger.info(f"Using Playwright for {source_name}")
            self.stats['playwright_used'] += 1
            
            try:
                # Get content with Playwright
                content_data, used_playwright = await self.playwright_scraper.scrape_with_fallback(
                    base_url,
                    site_config,
                    initial_html
                )
                
                if used_playwright and content_data:
                    self.stats['playwright_success'] += 1
                    
                    # Extract article links
                    article_links = content_data.get('article_links', [])
                    
                    # Process each article
                    for article_url in article_links:
                        # Get article details with Playwright
                        article_data, _ = await self.playwright_scraper.scrape_with_fallback(
                            article_url,
                            site_config
                        )
                        
                        if article_data:
                            # Extract fields
                            title = article_data.get('title', 'No title')
                            content = article_data.get('content', '')
                            author = article_data.get('author', '')
                            date_str = article_data.get('date', '')
                            
                            # Parse date
                            publication_date = parse_date_func(
                                date_str,
                                site_config.get("date_format")
                            )
                            
                            # Extract keywords
                            keywords_found = []
                            if monitor_keywords and (title or content):
                                combined_text = f"{title} {content}"
                                keyword_matches = extract_keywords_func(combined_text, keywords_list)
                                keywords_found = [kw[0] for kw in keyword_matches]
                                
                            # Add article with Playwright data
                            article_id = add_article_with_playwright_data(
                                url=article_url,
                                title=title,
                                publication_date=publication_date,
                                source_website=source_name,
                                category=category,
                                content_summary=content[:1000],
                                full_content=content[:5000],
                                author=author,
                                keywords_found=keywords_found,
                                rendering_method='playwright',
                                screenshot_path=article_data.get('screenshot_path'),
                                js_errors=article_data.get('js_errors'),
                                page_metrics=article_data.get('metrics')
                            )
                            
                            if article_id:
                                new_articles_count += 1
                                articles_added.append({
                                    'id': article_id,
                                    'url': article_url,
                                    'title': title,
                                    'rendering_method': 'playwright'
                                })
                                
                    # Mark source as requiring JS for future runs
                    if new_articles_count > 0:
                        update_source_js_requirement(
                            source_name,
                            True,
                            site_config.get('playwright_settings', {})
                        )
                        
                else:
                    # Playwright failed or was not used
                    self.stats['playwright_failed'] += 1
                    self.stats['fallback_used'] += 1
                    logger.warning(f"Playwright failed for {source_name}, using fallback")
                    
            except Exception as e:
                logger.error(f"Error using Playwright for {source_name}: {e}")
                self.stats['playwright_failed'] += 1
                self.stats['fallback_used'] += 1
                
        else:
            # Regular scraping without Playwright
            logger.debug(f"Using regular scraping for {source_name}")
            
        return new_articles_count, articles_added
        
    def get_stats(self) -> Dict:
        """Get integration statistics."""
        stats = self.stats.copy()
        
        # Add Playwright scraper stats if available
        if self.playwright_scraper:
            stats['playwright_details'] = self.playwright_scraper.get_stats()
            
        # Calculate success rate
        if stats['playwright_used'] > 0:
            stats['playwright_success_rate'] = (
                stats['playwright_success'] / stats['playwright_used']
            )
        else:
            stats['playwright_success_rate'] = 0
            
        return stats


# Async wrapper for integration with synchronous scraper
def run_with_playwright(scraper_func):
    """
    Decorator to add Playwright support to the main scraper function.
    
    Args:
        scraper_func: The original scraper function
        
    Returns:
        Enhanced scraper function with Playwright support
    """
    def wrapper(*args, **kwargs):
        # Create async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize Playwright scraper
        playwright_integration = ScraperWithPlaywright()
        
        try:
            # Initialize Playwright
            loop.run_until_complete(playwright_integration.initialize_playwright())
            
            # Inject the integration into the scraper context
            # This is a bit hacky but preserves the existing scraper structure
            import builtins
            original_parse_html = builtins.__dict__.get('parse_html_site')
            
            # Create enhanced parse function
            def enhanced_parse_html_site(site_config, run_id):
                """Enhanced HTML parser with Playwright support."""
                if playwright_integration.playwright_enabled:
                    # Run async function in sync context
                    from scraper import fetch_url, extract_keywords, parse_date
                    
                    new_count, articles = loop.run_until_complete(
                        playwright_integration.parse_html_site_with_playwright(
                            site_config,
                            run_id,
                            fetch_url,
                            extract_keywords,
                            parse_date
                        )
                    )
                    
                    if new_count > 0:
                        return new_count
                        
                # Fall back to original parser
                if original_parse_html:
                    return original_parse_html(site_config, run_id)
                else:
                    # Import here to avoid circular dependency
                    from scraper import parse_html_site as original_func
                    return original_func(site_config, run_id)
                    
            # Temporarily replace the parse function
            builtins.parse_html_site = enhanced_parse_html_site
            
            # Run the original scraper
            result = scraper_func(*args, **kwargs)
            
            # Log Playwright statistics
            stats = playwright_integration.get_stats()
            if stats['playwright_used'] > 0:
                logger.info(
                    f"Playwright stats - Used: {stats['playwright_used']}, "
                    f"Success: {stats['playwright_success']}, "
                    f"Failed: {stats['playwright_failed']}, "
                    f"Fallbacks: {stats['fallback_used']}"
                )
                
            return result
            
        finally:
            # Cleanup
            loop.run_until_complete(playwright_integration.cleanup_playwright())
            
            # Restore original function
            if 'original_parse_html' in locals() and original_parse_html:
                builtins.parse_html_site = original_parse_html
                
            loop.close()
            
    return wrapper


# Export decorator
__all__ = ['run_with_playwright', 'ScraperWithPlaywright']