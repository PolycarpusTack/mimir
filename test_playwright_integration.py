#!/usr/bin/env python
"""
Comprehensive test suite for Playwright integration in Mimir.
Tests all aspects of the Playwright scraper including error handling,
fallback mechanisms, and database integration.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules to test
from playwright_exceptions import (
    BrowserLaunchError, ElementNotFoundError, PageLoadError,
    PlaywrightException, PlaywrightTimeoutError
)
from playwright_scraper import PlaywrightScraper


class TestPlaywrightExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_browser_launch_error(self):
        """Test BrowserLaunchError creation and attributes."""
        error = BrowserLaunchError("chromium", "Failed to launch", {"detail": "test"})
        self.assertEqual(error.browser_type, "chromium")
        self.assertEqual(error.reason, "Failed to launch")
        self.assertIn("Failed to launch chromium browser", str(error))
        
    def test_page_load_error_with_status(self):
        """Test PageLoadError with status code."""
        error = PageLoadError("https://example.com", "Not found", 404)
        self.assertEqual(error.url, "https://example.com")
        self.assertEqual(error.status_code, 404)
        self.assertIn("status: 404", str(error))
        
    def test_element_not_found_error(self):
        """Test ElementNotFoundError with timeout."""
        error = ElementNotFoundError(".article", "https://example.com", 5000)
        self.assertEqual(error.selector, ".article")
        self.assertEqual(error.timeout, 5000)
        self.assertIn("after 5000ms", str(error))
        
    def test_playwright_timeout_error(self):
        """Test PlaywrightTimeoutError."""
        error = PlaywrightTimeoutError("navigation", 30000, "https://example.com")
        self.assertEqual(error.operation, "navigation")
        self.assertEqual(error.timeout_ms, 30000)
        self.assertIn("timed out after 30000ms", str(error))


class TestPlaywrightScraperConfig(unittest.TestCase):
    """Test PlaywrightScraper configuration and initialization."""
    
    def setUp(self):
        """Set up test configuration."""
        self.test_config = {
            'playwright_scraping': {
                'enabled': True,
                'config_file': 'test_playwright_config.json',
                'fallback_to_beautifulsoup': True,
                'js_site_indicators': ['react-root', 'ng-app']
            }
        }
        
        self.playwright_config = {
            'playwright_settings': {
                'browsers': {
                    'chromium': {'enabled': True, 'headless': True}
                },
                'timeouts': {
                    'page_load': 10000,
                    'script': 5000,
                    'navigation': 10000
                },
                'viewport': {'width': 1920, 'height': 1080},
                'stealth_mode': True,
                'block_resources': ['image', 'font'],
                'max_concurrent_contexts': 2,
                'retry_attempts': 2,
                'screenshot_on_error': True,
                'screenshot_path': 'test_screenshots'
            }
        }
        
    def test_scraper_initialization(self):
        """Test PlaywrightScraper initialization with config."""
        with patch('builtins.open', unittest.mock.mock_open(
            read_data=json.dumps(self.playwright_config)
        )):
            scraper = PlaywrightScraper(self.test_config)
            
            self.assertTrue(scraper.enabled)
            self.assertTrue(scraper.fallback_enabled)
            self.assertEqual(scraper.max_concurrent_contexts, 2)
            self.assertEqual(scraper.retry_attempts, 2)
            self.assertIn('image', scraper.block_resources)
            
    def test_scraper_disabled_by_default(self):
        """Test that scraper is disabled when not configured."""
        config = {'playwright_scraping': {'enabled': False}}
        scraper = PlaywrightScraper(config)
        self.assertFalse(scraper.enabled)
        
    def test_javascript_detection(self):
        """Test JavaScript requirement detection."""
        scraper = PlaywrightScraper(self.test_config)
        
        # Test explicit js_required
        site_config = {'js_required': True}
        self.assertTrue(scraper.is_javascript_required(site_config))
        
        # Test JS indicator detection
        site_config = {'js_required': False}
        html_with_react = '<div id="react-root"></div>'
        self.assertTrue(scraper.is_javascript_required(site_config, html_with_react))
        
        # Test no JS required
        html_plain = '<div>Plain HTML</div>'
        self.assertFalse(scraper.is_javascript_required(site_config, html_plain))


class TestPlaywrightScraperAsync(unittest.TestCase):
    """Test async functionality of PlaywrightScraper."""
    
    def setUp(self):
        """Set up async test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.test_config = {
            'playwright_scraping': {
                'enabled': True,
                'config_file': 'test_playwright_config.json'
            }
        }
        
    def tearDown(self):
        """Clean up async environment."""
        self.loop.close()
        
    @patch('playwright.async_api.async_playwright')
    def test_browser_initialization(self, mock_playwright):
        """Test browser initialization."""
        # Mock Playwright components
        mock_browser = AsyncMock()
        mock_browser_type = AsyncMock()
        mock_browser_type.launch = AsyncMock(return_value=mock_browser)
        
        mock_pw = AsyncMock()
        mock_pw.chromium = mock_browser_type
        mock_pw.stop = AsyncMock()
        
        mock_playwright.return_value.start = AsyncMock(return_value=mock_pw)
        
        async def run_test():
            scraper = PlaywrightScraper(self.test_config)
            await scraper.initialize()
            
            self.assertIsNotNone(scraper.playwright)
            self.assertIn('chromium', scraper.browsers)
            
            await scraper.cleanup()
            
        self.loop.run_until_complete(run_test())
        
    @patch('playwright.async_api.async_playwright')
    def test_context_creation(self, mock_playwright):
        """Test browser context creation."""
        # Setup mocks
        mock_context = AsyncMock()
        mock_context.close = AsyncMock()
        
        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        
        async def run_test():
            scraper = PlaywrightScraper(self.test_config)
            scraper.browsers = {'chromium': mock_browser}
            
            async with scraper.create_browser_context() as context:
                self.assertEqual(context, mock_context)
                
            mock_context.close.assert_called_once()
            
        self.loop.run_until_complete(run_test())
        
    def test_resource_blocking_setup(self):
        """Test resource blocking configuration."""
        scraper = PlaywrightScraper(self.test_config)
        
        # Mock context
        mock_context = AsyncMock()
        routes = []
        
        async def mock_route(pattern, handler):
            routes.append((pattern, handler))
            
        mock_context.route = mock_route
        
        async def run_test():
            await scraper._setup_resource_blocking(mock_context)
            self.assertEqual(len(routes), 1)
            self.assertEqual(routes[0][0], '**/*')
            
        self.loop.run_until_complete(run_test())


class TestPlaywrightScraping(unittest.TestCase):
    """Test actual scraping functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.test_config = {
            'playwright_scraping': {
                'enabled': True,
                'fallback_to_beautifulsoup': True
            }
        }
        
        self.site_config = {
            'name': 'Test Site',
            'url': 'https://example.com',
            'type': 'html',
            'selectors': {
                'overview_article_link': 'a.article-link',
                'detail_title': 'h1.title',
                'detail_content': 'div.content',
                'detail_author': 'span.author',
                'detail_date': 'time.date'
            }
        }
        
    def tearDown(self):
        """Clean up."""
        self.loop.close()
        
    @patch('playwright.async_api.Page')
    def test_content_extraction(self, mock_page):
        """Test content extraction from page."""
        # Mock page methods
        mock_page.url = 'https://example.com/article'
        mock_page.content = AsyncMock(return_value='<html><body>Test</body></html>')
        mock_page.text_content = AsyncMock(return_value='Test Article')
        mock_page.query_selector_all = AsyncMock(return_value=[])
        
        async def run_test():
            scraper = PlaywrightScraper(self.test_config)
            content = await scraper._extract_content(
                mock_page,
                self.site_config,
                'https://example.com'
            )
            
            self.assertEqual(content['url'], 'https://example.com')
            self.assertEqual(content['rendering_method'], 'playwright')
            self.assertIn('html', content)
            self.assertIn('scraped_at', content)
            
        self.loop.run_until_complete(run_test())
        
    def test_error_handling(self):
        """Test error handling and screenshot capture."""
        scraper = PlaywrightScraper(self.test_config)
        
        # Mock page with error
        mock_page = AsyncMock()
        mock_page.url = 'https://example.com'
        mock_page.screenshot = AsyncMock()
        
        async def run_test():
            # Test error screenshot
            screenshot_path = await scraper._capture_error_screenshot(
                mock_page,
                'https://example.com',
                Exception('Test error')
            )
            
            self.assertTrue(str(screenshot_path).startswith(str(scraper.screenshot_path)))
            self.assertIn('error', str(screenshot_path))
            mock_page.screenshot.assert_called_once()
            
        self.loop.run_until_complete(run_test())
        
    def test_fallback_mechanism(self):
        """Test fallback to BeautifulSoup."""
        scraper = PlaywrightScraper(self.test_config)
        
        async def run_test():
            # Simulate Playwright failure
            with patch.object(scraper, 'scrape_with_playwright', 
                            side_effect=PlaywrightException('Test failure')):
                
                content, used_playwright = await scraper.scrape_with_fallback(
                    'https://example.com',
                    self.site_config
                )
                
                self.assertFalse(used_playwright)
                self.assertEqual(scraper.stats['fallbacks'], 1)
                
        self.loop.run_until_complete(run_test())


class TestDatabaseIntegration(unittest.TestCase):
    """Test database integration for Playwright data."""
    
    @patch('db_adapter_playwright.get_db_connection')
    @patch('db_adapter_playwright.USE_POSTGRES', True)
    def test_add_article_with_playwright_data(self, mock_get_conn):
        """Test adding article with Playwright-specific data."""
        # Mock database connection
        mock_cursor = MagicMock()
        mock_cursor.fetchone = MagicMock(return_value=['article-id-123'])
        
        mock_conn = MagicMock()
        mock_conn.cursor = MagicMock(return_value=mock_cursor)
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        
        mock_get_conn.return_value = mock_conn
        
        # Import here to use mocked connection
        from db_adapter_playwright import add_article_with_playwright_data
        
        # Test data
        article_id = add_article_with_playwright_data(
            url='https://example.com/article',
            title='Test Article',
            publication_date=datetime.now(),
            source_website='Test Site',
            rendering_method='playwright',
            screenshot_path='/screenshots/test.png',
            js_errors=[{'message': 'Test error'}],
            page_metrics={'loadTime': 1500}
        )
        
        self.assertEqual(article_id, 'article-id-123')
        
        # Verify correct SQL was called
        mock_cursor.execute.assert_called()
        call_args = mock_cursor.execute.call_args[0]
        self.assertIn('rendering_method', call_args[0])
        self.assertIn('screenshot_path', call_args[0])
        self.assertIn('js_errors', call_args[0])
        self.assertIn('page_metrics', call_args[0])


class TestIntegrationWithScraper(unittest.TestCase):
    """Test integration with main scraper."""
    
    def test_enhanced_scraper_import(self):
        """Test that enhanced scraper can be imported."""
        try:
            import scraper_enhanced
            self.assertTrue(hasattr(scraper_enhanced, 'run_scraper'))
            self.assertTrue(hasattr(scraper_enhanced, 'parse_html_site'))
        except ImportError as e:
            self.skipTest(f"Could not import enhanced scraper: {e}")
            
    @patch('scraper_playwright_integration.ScraperWithPlaywright')
    def test_playwright_integration_decorator(self, mock_scraper_class):
        """Test run_with_playwright decorator."""
        from scraper_playwright_integration import run_with_playwright
        
        # Mock scraper instance
        mock_instance = MagicMock()
        mock_instance.playwright_enabled = True
        mock_instance.get_stats = MagicMock(return_value={
            'playwright_used': 5,
            'playwright_success': 4,
            'playwright_failed': 1
        })
        
        mock_scraper_class.return_value = mock_instance
        
        # Test function
        @run_with_playwright
        def test_scraper():
            return "Success"
            
        # Run test
        result = test_scraper()
        self.assertEqual(result, "Success")


class TestPerformanceAndStats(unittest.TestCase):
    """Test performance tracking and statistics."""
    
    def test_stats_tracking(self):
        """Test statistics tracking."""
        scraper = PlaywrightScraper({'playwright_scraping': {'enabled': True}})
        
        # Simulate operations
        scraper.stats['pages_loaded'] = 10
        scraper.stats['errors'] = 2
        scraper.stats['fallbacks'] = 1
        scraper.stats['total_time'] = 45.5
        
        stats = scraper.get_stats()
        
        self.assertEqual(stats['pages_loaded'], 10)
        self.assertEqual(stats['error_rate'], 0.2)
        self.assertEqual(stats['fallback_rate'], 0.1)
        self.assertAlmostEqual(stats['avg_time_per_page'], 4.55, places=2)
        
    def test_stats_reset(self):
        """Test statistics reset."""
        scraper = PlaywrightScraper({'playwright_scraping': {'enabled': True}})
        
        scraper.stats['pages_loaded'] = 5
        scraper.reset_stats()
        
        self.assertEqual(scraper.stats['pages_loaded'], 0)
        self.assertEqual(scraper.stats['errors'], 0)


def run_integration_tests():
    """Run all integration tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPlaywrightExceptions,
        TestPlaywrightScraperConfig,
        TestPlaywrightScraperAsync,
        TestPlaywrightScraping,
        TestDatabaseIntegration,
        TestIntegrationWithScraper,
        TestPerformanceAndStats
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)