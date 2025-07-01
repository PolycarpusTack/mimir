"""
Playwright-based scraper for JavaScript-heavy websites.
Provides browser automation capabilities with fallback to BeautifulSoup.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import (
    Browser, BrowserContext, Page, Playwright, Response,
    async_playwright, Error as PlaywrightError
)
from playwright_stealth import stealth_async

from config_loader import config_loader
from playwright_exceptions import (
    AuthenticationError, BrowserContextError, BrowserLaunchError,
    ElementNotFoundError, JavaScriptError, NetworkError, PageLoadError,
    PlaywrightException, PlaywrightTimeoutError, ResourceBlockError,
    ScreenshotError, handle_playwright_error
)


class PlaywrightScraper:
    """
    Browser-based scraper using Playwright for JavaScript-heavy sites.
    Supports multiple browsers, stealth mode, and resource blocking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Playwright scraper.
        
        Args:
            config: Optional configuration dictionary. If not provided,
                   loads from config files.
        """
        # Load configuration
        if config is None:
            main_config = config_loader.load_config()
            self.config = main_config.get('playwright_scraping', {})
        else:
            self.config = config
            
        # Load Playwright-specific config
        playwright_config_file = self.config.get('config_file', 'playwright_config.json')
        self.playwright_config = self._load_playwright_config(playwright_config_file)
        
        # Setup logging
        self.logger = logging.getLogger('mimir_playwright_scraper')
        
        # Core settings
        self.enabled = self.config.get('enabled', False)
        self.fallback_enabled = self.config.get('fallback_to_beautifulsoup', True)
        self.js_site_indicators = self.config.get('js_site_indicators', [])
        
        # Browser settings from playwright_config
        browser_settings = self.playwright_config.get('playwright_settings', {})
        self.browser_configs = browser_settings.get('browsers', {})
        self.timeouts = browser_settings.get('timeouts', {})
        self.viewport = browser_settings.get('viewport', {'width': 1920, 'height': 1080})
        self.user_agent = browser_settings.get('user_agent')
        self.locale = browser_settings.get('locale', 'en-US')
        self.timezone = browser_settings.get('timezone', 'America/New_York')
        self.stealth_mode = browser_settings.get('stealth_mode', True)
        self.block_resources = set(browser_settings.get('block_resources', []))
        self.max_concurrent_contexts = browser_settings.get('max_concurrent_contexts', 3)
        self.retry_attempts = browser_settings.get('retry_attempts', 3)
        self.screenshot_on_error = browser_settings.get('screenshot_on_error', True)
        self.screenshot_path = Path(browser_settings.get('screenshot_path', 'logs/playwright_screenshots'))
        
        # Ensure screenshot directory exists
        self.screenshot_path.mkdir(parents=True, exist_ok=True)
        
        # Runtime state
        self.playwright: Optional[Playwright] = None
        self.browsers: Dict[str, Browser] = {}
        self.active_contexts: Set[BrowserContext] = set()
        self.context_semaphore = asyncio.Semaphore(self.max_concurrent_contexts)
        
        # Performance tracking
        self.stats = {
            'pages_loaded': 0,
            'js_executed': 0,
            'errors': 0,
            'fallbacks': 0,
            'total_time': 0.0
        }
        
    def _load_playwright_config(self, config_file: str) -> Dict[str, Any]:
        """Load Playwright-specific configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load {config_file}: {e}. Using defaults.")
            return {}
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize Playwright and browsers."""
        if not self.enabled:
            self.logger.info("Playwright scraping is disabled")
            return
            
        try:
            self.playwright = await async_playwright().start()
            
            # Launch configured browsers
            for browser_name, browser_config in self.browser_configs.items():
                if browser_config.get('enabled', False):
                    await self._launch_browser(browser_name, browser_config)
                    
            self.logger.info(f"Playwright initialized with browsers: {list(self.browsers.keys())}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Playwright: {e}")
            raise BrowserLaunchError("any", str(e))
            
    async def _launch_browser(self, browser_name: str, config: Dict[str, Any]):
        """Launch a specific browser."""
        try:
            browser_type = getattr(self.playwright, browser_name)
            
            launch_args = {
                'headless': config.get('headless', True),
                'args': config.get('args', [])
            }
            
            # Add proxy if configured
            proxy = config.get('proxy')
            if proxy:
                launch_args['proxy'] = proxy
                
            browser = await browser_type.launch(**launch_args)
            self.browsers[browser_name] = browser
            
            self.logger.info(f"Launched {browser_name} browser")
            
        except Exception as e:
            self.logger.error(f"Failed to launch {browser_name}: {e}")
            raise BrowserLaunchError(browser_name, str(e))
            
    async def cleanup(self):
        """Clean up all browser resources."""
        # Close all active contexts
        for context in list(self.active_contexts):
            try:
                await context.close()
            except Exception as e:
                self.logger.error(f"Error closing context: {e}")
                
        self.active_contexts.clear()
        
        # Close all browsers
        for browser_name, browser in self.browsers.items():
            try:
                await browser.close()
                self.logger.info(f"Closed {browser_name} browser")
            except Exception as e:
                self.logger.error(f"Error closing {browser_name}: {e}")
                
        self.browsers.clear()
        
        # Stop Playwright
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
            
        self.logger.info("Playwright cleanup complete")
        
    def is_javascript_required(self, site_config: Dict[str, Any], html_content: Optional[str] = None) -> bool:
        """
        Determine if JavaScript rendering is required for a site.
        
        Args:
            site_config: Site configuration dictionary
            html_content: Optional HTML content to check for JS indicators
            
        Returns:
            True if JavaScript rendering is required
        """
        # Check explicit js_required flag
        if site_config.get('js_required', False):
            return True
            
        # Check HTML content for JS framework indicators
        if html_content and self.js_site_indicators:
            for indicator in self.js_site_indicators:
                if indicator in html_content:
                    self.logger.debug(f"Found JS indicator '{indicator}' in HTML")
                    return True
                    
        return False
        
    @asynccontextmanager
    async def create_browser_context(self, browser_name: str = 'chromium'):
        """
        Create a new browser context with proper configuration.
        
        Args:
            browser_name: Browser to use (chromium, firefox, webkit)
            
        Yields:
            BrowserContext instance
        """
        async with self.context_semaphore:
            browser = self.browsers.get(browser_name)
            if not browser:
                raise BrowserContextError('create_context', f"Browser {browser_name} not available")
                
            try:
                context_args = {
                    'viewport': self.viewport,
                    'locale': self.locale,
                    'timezone_id': self.timezone,
                    'ignore_https_errors': True
                }
                
                if self.user_agent:
                    context_args['user_agent'] = self.user_agent
                    
                context = await browser.new_context(**context_args)
                self.active_contexts.add(context)
                
                # Setup request interception for resource blocking
                if self.block_resources:
                    await self._setup_resource_blocking(context)
                    
                yield context
                
            finally:
                # Cleanup context
                self.active_contexts.discard(context)
                try:
                    await context.close()
                except Exception as e:
                    self.logger.error(f"Error closing context: {e}")
                    
    async def _setup_resource_blocking(self, context: BrowserContext):
        """Setup resource blocking for performance."""
        async def block_resources(route, request):
            if request.resource_type in self.block_resources:
                await route.abort()
            else:
                await route.continue_()
                
        await context.route('**/*', block_resources)
        
    async def scrape_with_playwright(
        self,
        url: str,
        site_config: Dict[str, Any],
        browser_name: str = 'chromium'
    ) -> Dict[str, Any]:
        """
        Scrape a URL using Playwright.
        
        Args:
            url: URL to scrape
            site_config: Site configuration with selectors
            browser_name: Browser to use
            
        Returns:
            Dictionary with scraped data
        """
        start_time = time.time()
        
        if not self.enabled or browser_name not in self.browsers:
            raise PlaywrightException(f"Playwright not enabled or {browser_name} not available")
            
        async with self.create_browser_context(browser_name) as context:
            page = await context.new_page()
            
            # Apply stealth mode if enabled
            if self.stealth_mode:
                await stealth_async(page)
                
            try:
                # Navigate to URL
                response = await self._navigate_to_page(page, url, site_config)
                
                # Wait for content to load
                await self._wait_for_content(page, site_config)
                
                # Execute any custom JavaScript
                await self._execute_custom_scripts(page, site_config)
                
                # Extract content
                content = await self._extract_content(page, site_config, url)
                
                # Take screenshot if configured
                if site_config.get('capture_screenshot', False):
                    screenshot_path = await self._capture_screenshot(page, url)
                    content['screenshot_path'] = str(screenshot_path)
                    
                # Get page metrics
                content['metrics'] = await self._get_page_metrics(page)
                
                # Update stats
                self.stats['pages_loaded'] += 1
                self.stats['total_time'] += time.time() - start_time
                
                return content
                
            except Exception as e:
                self.stats['errors'] += 1
                
                # Take error screenshot if enabled
                if self.screenshot_on_error:
                    try:
                        error_screenshot = await self._capture_error_screenshot(page, url, e)
                        self.logger.info(f"Error screenshot saved: {error_screenshot}")
                    except Exception as screenshot_error:
                        self.logger.error(f"Failed to capture error screenshot: {screenshot_error}")
                        
                # Handle specific error types
                handle_playwright_error(e, {
                    'url': url,
                    'operation': 'scrape',
                    'browser': browser_name,
                    'selector': site_config.get('selectors', {}).get('detail_content')
                })
                
    async def _navigate_to_page(
        self,
        page: Page,
        url: str,
        site_config: Dict[str, Any]
    ) -> Response:
        """Navigate to a page with proper error handling."""
        playwright_settings = site_config.get('playwright_settings', {})
        timeout = self.timeouts.get('navigation', 30000)
        
        try:
            response = await page.goto(
                url,
                wait_until=playwright_settings.get('wait_for_load_state', 'domcontentloaded'),
                timeout=timeout
            )
            
            if response.status >= 400:
                raise PageLoadError(url, f"HTTP {response.status}", response.status)
                
            return response
            
        except asyncio.TimeoutError:
            raise PlaywrightTimeoutError('navigation', timeout, url)
        except Exception as e:
            if "net::" in str(e):
                raise NetworkError('navigation', url, str(e))
            raise
            
    async def _wait_for_content(self, page: Page, site_config: Dict[str, Any]):
        """Wait for content to be loaded based on configuration."""
        playwright_settings = site_config.get('playwright_settings', {})
        
        # Wait for specific selector if configured
        wait_selector = playwright_settings.get('wait_for_selector')
        if wait_selector:
            timeout = self.timeouts.get('script', 10000)
            try:
                await page.wait_for_selector(wait_selector, timeout=timeout)
            except asyncio.TimeoutError:
                raise ElementNotFoundError(wait_selector, page.url, timeout)
                
        # Wait for custom function if configured
        wait_function = playwright_settings.get('wait_for_function')
        if wait_function:
            timeout = self.timeouts.get('script', 10000)
            try:
                await page.wait_for_function(wait_function, timeout=timeout)
            except asyncio.TimeoutError:
                raise PlaywrightTimeoutError('wait_for_function', timeout, page.url)
                
        # Handle infinite scroll
        if playwright_settings.get('scroll_to_load', False):
            await self._handle_infinite_scroll(page, playwright_settings)
            
    async def _handle_infinite_scroll(self, page: Page, settings: Dict[str, Any]):
        """Handle infinite scroll pages."""
        scroll_pause = settings.get('scroll_pause_time', 2000)
        max_scrolls = settings.get('max_scrolls', 5)
        
        last_height = await page.evaluate('document.body.scrollHeight')
        scrolls = 0
        
        while scrolls < max_scrolls:
            # Scroll to bottom
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            
            # Wait for new content
            await page.wait_for_timeout(scroll_pause)
            
            # Check if new content loaded
            new_height = await page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                break
                
            last_height = new_height
            scrolls += 1
            
        self.logger.debug(f"Performed {scrolls} scrolls on {page.url}")
        
    async def _execute_custom_scripts(self, page: Page, site_config: Dict[str, Any]):
        """Execute custom JavaScript if configured."""
        playwright_settings = site_config.get('playwright_settings', {})
        custom_script = playwright_settings.get('execute_script')
        
        if custom_script:
            try:
                result = await page.evaluate(custom_script)
                self.stats['js_executed'] += 1
                return result
            except Exception as e:
                raise JavaScriptError(custom_script, str(e), page.url)
                
    async def _extract_content(
        self,
        page: Page,
        site_config: Dict[str, Any],
        url: str
    ) -> Dict[str, Any]:
        """Extract content from the page using configured selectors."""
        selectors = site_config.get('selectors', {})
        content = {
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'rendering_method': 'playwright'
        }
        
        # Get full HTML
        html_content = await page.content()
        content['html'] = html_content
        
        # Parse with BeautifulSoup for consistent selector handling
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract fields using selectors
        for field, selector in selectors.items():
            if not selector or not field.startswith('detail_'):
                continue
                
            field_name = field.replace('detail_', '')
            
            try:
                # Try Playwright selector first for dynamic content
                element_text = await page.text_content(selector)
                if element_text:
                    content[field_name] = element_text.strip()
                else:
                    # Fallback to BeautifulSoup
                    element = soup.select_one(selector)
                    if element:
                        content[field_name] = element.get_text(strip=True)
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract {field_name} with selector '{selector}': {e}")
                
        # Extract all links if it's an overview page
        if 'overview_article_link' in selectors:
            links = []
            link_elements = await page.query_selector_all(selectors['overview_article_link'])
            
            for element in link_elements:
                href = await element.get_attribute('href')
                if href:
                    absolute_url = urljoin(url, href)
                    links.append(absolute_url)
                    
            content['article_links'] = links
            
        return content
        
    async def _capture_screenshot(self, page: Page, url: str) -> Path:
        """Capture a screenshot of the page."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = urlparse(url).netloc.replace('.', '_')
        screenshot_file = self.screenshot_path / f"{safe_filename}_{timestamp}.png"
        
        try:
            await page.screenshot(path=str(screenshot_file), full_page=True)
            return screenshot_file
        except Exception as e:
            raise ScreenshotError(str(screenshot_file), str(e))
            
    async def _capture_error_screenshot(self, page: Page, url: str, error: Exception) -> Path:
        """Capture a screenshot when an error occurs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = urlparse(url).netloc.replace('.', '_')
        error_type = type(error).__name__
        screenshot_file = self.screenshot_path / f"error_{safe_filename}_{error_type}_{timestamp}.png"
        
        await page.screenshot(path=str(screenshot_file), full_page=False)
        return screenshot_file
        
    async def _get_page_metrics(self, page: Page) -> Dict[str, Any]:
        """Get page performance metrics."""
        try:
            metrics = await page.evaluate('''() => {
                const navigation = performance.getEntriesByType('navigation')[0];
                const paint = performance.getEntriesByType('paint');
                
                return {
                    domContentLoaded: navigation ? navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart : null,
                    loadComplete: navigation ? navigation.loadEventEnd - navigation.loadEventStart : null,
                    firstPaint: paint.find(p => p.name === 'first-paint')?.startTime,
                    firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime,
                    resources: performance.getEntriesByType('resource').length
                };
            }''')
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to get page metrics: {e}")
            return {}
            
    async def scrape_with_fallback(
        self,
        url: str,
        site_config: Dict[str, Any],
        html_content: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Scrape with Playwright, falling back to BeautifulSoup if needed.
        
        Args:
            url: URL to scrape
            site_config: Site configuration
            html_content: Optional pre-fetched HTML content
            
        Returns:
            Tuple of (content dict, used_playwright bool)
        """
        # Check if Playwright is needed
        if not self.is_javascript_required(site_config, html_content):
            self.logger.debug(f"JavaScript not required for {url}, skipping Playwright")
            return {}, False
            
        # Try Playwright if enabled
        if self.enabled and self.browsers:
            for attempt in range(self.retry_attempts):
                try:
                    content = await self.scrape_with_playwright(url, site_config)
                    return content, True
                    
                except PlaywrightException as e:
                    self.logger.warning(f"Playwright attempt {attempt + 1} failed: {e}")
                    
                    if attempt == self.retry_attempts - 1:
                        # Last attempt failed
                        if self.fallback_enabled:
                            self.logger.info(f"Falling back to BeautifulSoup for {url}")
                            self.stats['fallbacks'] += 1
                            return {}, False
                        else:
                            raise
                            
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)
                    
        return {}, False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get scraper statistics."""
        stats = self.stats.copy()
        
        # Calculate averages
        if stats['pages_loaded'] > 0:
            stats['avg_time_per_page'] = stats['total_time'] / stats['pages_loaded']
            stats['error_rate'] = stats['errors'] / stats['pages_loaded']
            stats['fallback_rate'] = stats['fallbacks'] / stats['pages_loaded']
        else:
            stats['avg_time_per_page'] = 0
            stats['error_rate'] = 0
            stats['fallback_rate'] = 0
            
        # Add browser status
        stats['browsers_active'] = list(self.browsers.keys())
        stats['contexts_active'] = len(self.active_contexts)
        
        return stats
        
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'pages_loaded': 0,
            'js_executed': 0,
            'errors': 0,
            'fallbacks': 0,
            'total_time': 0.0
        }