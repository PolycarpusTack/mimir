"""
Playwright integration for scraper.py
Provides parse_playwright_site() function that follows the exact same interface as parse_html_site()
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

import db_manager
from config_loader import config_loader

logger = logging.getLogger(__name__)


def load_playwright_config() -> Dict:
    """Load Playwright configuration following existing patterns."""
    try:
        main_config = config_loader.load_config()
        playwright_config = main_config.get('playwright', {})
        
        # Load detailed config from separate file
        config_file = playwright_config.get('config_file', 'playwright_config.json')
        with open(config_file, 'r') as f:
            detailed_config = json.load(f)
            
        return {**playwright_config, **detailed_config}
    except Exception as e:
        logger.error(f"Failed to load Playwright config: {e}")
        return {}


def is_javascript_required(site_config: Dict, html_content: Optional[str] = None) -> bool:
    """
    Determine if JavaScript rendering is required for a site.
    Follows the same logic as existing scraper functions.
    """
    # Check explicit js_required flag (same as site_config.get pattern)
    if site_config.get('js_required', False):
        return True
        
    # Check if site type is 'playwright'
    if site_config.get('type') == 'playwright':
        return True
        
    # Auto-detect based on HTML content if available
    if html_content:
        config = load_playwright_config()
        js_indicators = config.get('js_indicators', [])
        
        for indicator in js_indicators:
            if indicator in html_content:
                logger.debug(f"Found JS indicator '{indicator}' in HTML")
                return True
                
    return False


async def scrape_page_with_playwright(url: str, site_config: Dict, playwright_config: Dict) -> Dict:
    """
    Scrape a single page using Playwright.
    Returns dict with page content and metadata.
    """
    async with async_playwright() as p:
        # Browser setup following config patterns
        browser_type = playwright_config.get('browser_type', 'chromium')
        browser_settings = playwright_config.get('browsers', {}).get(browser_type, {})
        
        if not browser_settings.get('enabled', True):
            raise Exception(f"Browser {browser_type} is not enabled in config")
            
        # Launch browser
        browser = await getattr(p, browser_type).launch(
            headless=browser_settings.get('headless', True),
            args=browser_settings.get('args', [])
        )
        
        try:
            # Create context with proper settings
            defaults = playwright_config.get('defaults', {})
            context = await browser.new_context(
                viewport=defaults.get('viewport', {'width': 1920, 'height': 1080}),
                user_agent=defaults.get('user_agent'),
                locale=defaults.get('locale', 'en-US'),
                timezone_id=defaults.get('timezone', 'America/New_York'),
                extra_http_headers=defaults.get('extra_http_headers', {})
            )
            
            # Block resources for performance (same pattern as config.get)
            performance_config = playwright_config.get('performance', {})
            if performance_config.get('intercept_requests', False):
                blocked_resources = set(performance_config.get('block_resources', []))
                
                async def block_resources(route, request):
                    if request.resource_type in blocked_resources:
                        await route.abort()
                    else:
                        await route.continue_()
                        
                await context.route('**/*', block_resources)
            
            # Create page
            page = await context.new_page()
            
            # Navigate with timeout (following timeout pattern from config)
            timeouts = playwright_config.get('timeouts', {})
            page_load_timeout = timeouts.get('page_load', 30000)
            
            await page.goto(url, timeout=page_load_timeout)
            
            # Wait strategy based on site config
            playwright_settings = site_config.get('playwright_settings', {})
            wait_selector = playwright_settings.get('wait_for_selector')
            wait_function = playwright_settings.get('wait_for_function')
            wait_state = playwright_settings.get('wait_for_load_state', 
                                               playwright_config.get('default_wait_strategy', 'domcontentloaded'))
            
            # Wait for content to load
            if wait_selector:
                element_timeout = timeouts.get('element_wait', 10000)
                await page.wait_for_selector(wait_selector, timeout=element_timeout)
            elif wait_function:
                script_timeout = timeouts.get('script_execution', 5000)
                await page.wait_for_function(wait_function, timeout=script_timeout)
            else:
                await page.wait_for_load_state(wait_state)
            
            # Handle infinite scroll if configured
            if playwright_settings.get('scroll_to_load', False):
                scroll_times = playwright_settings.get('scroll_times', 3)
                scroll_pause = playwright_settings.get('scroll_pause_time', 2000)
                
                for i in range(scroll_times):
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await page.wait_for_timeout(scroll_pause)
            
            # Execute custom script if provided
            custom_script = playwright_settings.get('execute_script')
            if custom_script:
                await page.evaluate(custom_script)
            
            # Get page content
            html_content = await page.content()
            
            # Capture screenshot if enabled
            screenshot_path = None
            screenshots_config = playwright_config.get('screenshots', {})
            if screenshots_config.get('enabled', False) and screenshots_config.get('on_success', False):
                from pathlib import Path
                screenshot_dir = Path(screenshots_config.get('path', 'logs/playwright_screenshots'))
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                
                # Create safe filename
                safe_filename = urlparse(url).netloc.replace('.', '_')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = screenshot_dir / f"{safe_filename}_{timestamp}.png"
                
                await page.screenshot(path=str(screenshot_path))
            
            # Get basic performance metrics
            performance_metrics = await page.evaluate('''() => {
                const navigation = performance.getEntriesByType('navigation')[0];
                return {
                    load_time: navigation ? navigation.loadEventEnd - navigation.loadEventStart : null,
                    dom_content_loaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : null,
                    resource_count: performance.getEntriesByType('resource').length
                };
            }''')
            
            return {
                'html_content': html_content,
                'screenshot_path': str(screenshot_path) if screenshot_path else None,
                'performance_metrics': performance_metrics,
                'rendering_method': 'playwright',
                'browser_type': browser_type
            }
            
        finally:
            await browser.close()


def parse_playwright_site(site_config: Dict, run_id: int) -> int:
    """
    Parse a site using Playwright with the same interface as parse_html_site().
    
    Args:
        site_config: Site configuration dict (same format as existing sites)
        run_id: Current scrape run ID
        
    Returns:
        int: Number of new articles found
    """
    # Load configuration (following existing pattern)
    playwright_config = load_playwright_config()
    if not playwright_config:
        logger.error("Playwright configuration not available")
        return 0
    
    # Integrate enhanced proxy support if available
    try:
        from scraper_proxy_integration import get_scraper_proxy_manager
        scraper_proxy_manager = get_scraper_proxy_manager()
        
        if scraper_proxy_manager and scraper_proxy_manager.enabled and not scraper_proxy_manager.initialization_error:
            # Get proxy configuration for Playwright
            from proxy_integration import integrate_proxy_with_playwright_scraper
            proxy_manager = scraper_proxy_manager.proxy_manager
            
            if proxy_manager:
                playwright_config = integrate_proxy_with_playwright_scraper(
                    playwright_config, site_config, proxy_manager
                )
                logger.debug("Integrated enhanced proxy support into Playwright config")
            else:
                logger.debug("Proxy manager not available for Playwright")
        else:
            logger.debug("Enhanced proxy system not active for Playwright")
    except ImportError:
        logger.debug("Enhanced proxy integration not available for Playwright")
    
    # Extract same variables as parse_html_site
    base_url = site_config["url"]
    new_articles_count = 0
    source_name = site_config["name"]
    category = site_config.get("category", "general")
    
    # Keywords monitoring (exact same pattern as original)
    from scraper import CONFIG  # Import the global CONFIG for keywords
    keywords_config = CONFIG.get("keywords_monitoring", {})
    monitor_keywords = keywords_config.get("enabled", False)
    keywords_list = keywords_config.get("keywords", [])
    
    # Pagination settings (same pattern)
    pagination = site_config.get("pagination", {})
    max_pages = pagination.get("max_pages", 1) if pagination.get("enabled", False) else 1
    
    current_url = base_url
    pages_processed = 0
    
    # Main scraping loop (same structure as original)
    while current_url and pages_processed < max_pages:
        logger.info(f"Processing page {pages_processed + 1} of {source_name} with Playwright: {current_url}")
        
        # Check robots.txt (same as original)
        from scraper import check_robots_txt  # Import the existing function
        if not check_robots_txt(base_url, urlparse(current_url).path):
            logger.warning(f"Robots.txt forbids access to {current_url}")
            break
        
        try:
            # Scrape page with Playwright
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                page_data = loop.run_until_complete(
                    scrape_page_with_playwright(current_url, site_config, playwright_config)
                )
            finally:
                loop.close()
            
            # Parse the HTML content with BeautifulSoup (same as original)
            soup_overview = BeautifulSoup(page_data['html_content'], "lxml")
            
            # Find article links (exact same pattern as original)
            selectors = site_config.get("selectors", {})
            article_links = soup_overview.select(selectors.get("overview_article_link", ""))
            
            logger.info(f"Found: {len(article_links)} article links on page {pages_processed + 1}")
            
            # Process each article (same loop structure as original)
            for link_element in article_links:
                article_url = link_element.get("href")
                if not article_url:
                    continue
                
                # Make absolute URL (same as original)
                article_url = urljoin(base_url, article_url)
                
                # Skip if exists (same check as original)
                if db_manager.article_exists(article_url):
                    continue
                
                # Scrape individual article
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        article_data = loop.run_until_complete(
                            scrape_page_with_playwright(article_url, site_config, playwright_config)
                        )
                    finally:
                        loop.close()
                    
                    # Parse article content
                    soup_article = BeautifulSoup(article_data['html_content'], "lxml")
                    
                    # Extract fields using selectors (same pattern as original)
                    title_elem = soup_article.select_one(selectors.get("detail_title", ""))
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    date_elem = soup_article.select_one(selectors.get("detail_date", ""))
                    date_str = date_elem.get_text(strip=True) if date_elem else ""
                    
                    content_elem = soup_article.select_one(selectors.get("detail_content", ""))
                    full_content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    author_elem = soup_article.select_one(selectors.get("detail_author", ""))
                    author = author_elem.get_text(strip=True) if author_elem else ""
                    
                    # Parse date (same function as original)
                    from scraper import parse_date  # Import existing function
                    publication_date = parse_date(date_str, site_config.get("date_format"))
                    
                    # Extract keywords (same pattern as original)
                    keywords_found = []
                    if monitor_keywords:
                        from scraper import extract_keywords  # Import existing function
                        combined_text = f"{title} {full_content}"
                        keyword_matches = extract_keywords(combined_text, keywords_list)
                        keywords_found = [kw[0] for kw in keyword_matches]
                    
                    # Add article to database (same call as original, but with Playwright data)
                    if hasattr(db_manager, 'add_article_with_playwright_data'):
                        # Use extended function if available
                        from db_adapter_playwright import add_article_with_playwright_data
                        article_id = add_article_with_playwright_data(
                            url=article_url,
                            title=title,
                            publication_date=publication_date,
                            source_website=source_name,
                            category=category,
                            content_summary=full_content[:1000],  # Same limit as original
                            full_content=full_content[:5000],     # Same limit as original
                            author=author,
                            keywords_found=keywords_found,
                            rendering_method='playwright',
                            screenshot_path=article_data.get('screenshot_path'),
                            page_metrics=article_data.get('performance_metrics')
                        )
                    else:
                        # Fallback to regular function
                        article_id = db_manager.add_article(
                            url=article_url,
                            title=title,
                            publication_date=publication_date,
                            source_website=source_name,
                            category=category,
                            content_summary=full_content[:1000],
                            full_content=full_content[:5000],
                            author=author,
                            keywords_found=keywords_found,
                        )
                    
                    if article_id:
                        new_articles_count += 1
                        
                        # Log keyword alerts (same pattern as original)
                        if monitor_keywords and keywords_found:
                            from scraper import extract_keywords
                            combined_text = f"{title} {full_content}"
                            keyword_matches = extract_keywords(combined_text, keywords_list)
                            for kw, context in keyword_matches:
                                db_manager.add_keyword_alert(article_id, kw, context)
                
                except Exception as e:
                    logger.error(f"Error scraping article {article_url}: {e}")
                    db_manager.log_scrape_error(
                        run_id, source_name, "PLAYWRIGHT_ARTICLE_ERROR", str(e), article_url
                    )
                    continue
            
            # Handle pagination (same logic as original)
            pages_processed += 1
            if pagination.get("enabled", False) and pages_processed < max_pages:
                next_page_elem = soup_overview.select_one(pagination.get("next_page_selector", ""))
                if next_page_elem:
                    next_url = next_page_elem.get("href")
                    if next_url:
                        current_url = urljoin(base_url, next_url)
                        logger.info(f"Moving to next page: {current_url}")
                    else:
                        current_url = None
                else:
                    current_url = None
            else:
                current_url = None
                
        except Exception as e:
            logger.error(f"Error processing page {current_url}: {e}")
            db_manager.log_scrape_error(
                run_id, source_name, "PLAYWRIGHT_PAGE_ERROR", str(e), current_url
            )
            break
    
    return new_articles_count