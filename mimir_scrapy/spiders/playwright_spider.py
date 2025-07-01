"""
Playwright Spider Template for Mimir news scraping
Handles JavaScript-heavy websites using Playwright for rendering
"""

import scrapy
from typing import Dict, Any, Generator, List

from .base_spider import BaseMimirSpider


class PlaywrightSpiderTemplate(BaseMimirSpider):
    """Template for JavaScript-rendered website spiders using Playwright"""
    
    def __init__(self, site_config: Dict[str, Any], *args, **kwargs):
        super().__init__(site_config, *args, **kwargs)
        
        # Playwright-specific configuration
        self.selectors = site_config.get('selectors', {})
        self.playwright_config = site_config.get('playwright', {})
        
        # Extract selectors
        self.overview_selector = self.selectors.get('overview_article_link', '')
        self.title_selector = self.selectors.get('detail_title', '')
        self.content_selector = self.selectors.get('detail_content', '')
        self.author_selector = self.selectors.get('detail_author', '')
        self.date_selector = self.selectors.get('detail_date', '')
        
        # Playwright settings
        self.wait_for_selector = self.playwright_config.get('wait_for_selector', '')
        self.wait_timeout = self.playwright_config.get('wait_timeout', 10000)
        self.viewport_size = self.playwright_config.get('viewport_size', {'width': 1920, 'height': 1080})
        self.block_resources = self.playwright_config.get('block_resources', ['image', 'font', 'media'])
        self.enable_javascript = self.playwright_config.get('enable_javascript', True)
        self.stealth_mode = self.playwright_config.get('stealth_mode', True)
        
        # Custom settings for Playwright
        self.custom_settings.update({
            'DOWNLOAD_HANDLERS': {
                'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
                'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            },
            'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
            'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
            'PLAYWRIGHT_LAUNCH_OPTIONS': {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=VizDisplayCompositor'
                ]
            }
        })
        
        if self.stealth_mode:
            self.custom_settings['PLAYWRIGHT_LAUNCH_OPTIONS']['args'].extend([
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ])
    
    def start_requests(self):
        """Generate initial requests with Playwright configuration"""
        for url in self.start_urls:
            yield scrapy.Request(
                url,
                callback=self.parse,
                meta=self._get_playwright_meta()
            )
    
    def _get_playwright_meta(self, additional_methods: List[Dict] = None) -> Dict:
        """Get Playwright meta configuration for requests"""
        page_methods = [
            {'method': 'set_viewport_size', **self.viewport_size},
            {'method': 'wait_for_load_state', 'state': 'domcontentloaded'},
        ]
        
        # Add wait for specific selector if configured
        if self.wait_for_selector:
            page_methods.append({
                'method': 'wait_for_selector',
                'selector': self.wait_for_selector,
                'timeout': self.wait_timeout
            })
        
        # Add additional methods if provided
        if additional_methods:
            page_methods.extend(additional_methods)
        
        meta = {
            'playwright': True,
            'playwright_page_methods': page_methods,
            'playwright_context_kwargs': {
                'ignore_https_errors': True,
            }
        }
        
        # Configure resource blocking
        if self.block_resources:
            meta['playwright_page_methods'].append({
                'method': 'route',
                'url': '**/*',
                'handler': self._create_route_handler()
            })
        
        return meta
    
    def _create_route_handler(self):
        """Create route handler for blocking resources"""
        def route_handler(route, request):
            resource_type = request.resource_type
            if resource_type in self.block_resources:
                route.abort()
            else:
                route.continue_()
        return route_handler
    
    def parse(self, response) -> Generator:
        """Parse main page and discover article links"""
        self.logger.info(f"Parsing JavaScript-rendered page: {response.url}")
        
        try:
            # Wait for content to be loaded
            if not self._verify_page_loaded(response):
                self.logger.warning(f"Page may not be fully loaded: {response.url}")
            
            # Extract article links
            article_links = self.extract_article_links(response)
            
            self.logger.info(f"Found {len(article_links)} article links")
            
            # Follow article links
            articles_processed = 0
            for link_info in article_links:
                if articles_processed >= self.max_articles:
                    break
                
                url = link_info['url']
                if self.should_follow_link(url):
                    yield scrapy.Request(
                        url,
                        callback=self.parse_article,
                        meta={
                            **self._get_playwright_meta(),
                            'link_info': link_info
                        }
                    )
                    articles_processed += 1
            
            # Handle infinite scroll or load more buttons
            load_more_button = self.playwright_config.get('load_more_selector')
            if load_more_button and articles_processed < self.max_articles:
                # This would require custom page interaction
                self.logger.info("Load more functionality detected but not implemented in this template")
            
        except Exception as e:
            self.logger.error(f"Failed to parse JavaScript page {response.url}: {e}")
    
    def extract_article_links(self, response) -> List[Dict[str, Any]]:
        """Extract article links from JavaScript-rendered page"""
        links = []
        
        if not self.overview_selector:
            self.logger.warning("No overview selector configured")
            return links
        
        try:
            # Extract links using configured selector
            link_elements = response.css(self.overview_selector)
            
            for element in link_elements:
                # Extract URL
                href = element.css('::attr(href)').get()
                if not href:
                    continue
                
                # Make absolute URL
                absolute_url = self.make_absolute_url(href, response.url)
                
                # Extract preview information
                preview_title = element.css('::text').get() or ''
                preview_title = self.clean_text(preview_title)
                
                link_info = {
                    'url': absolute_url,
                    'preview_title': preview_title,
                    'source_element': element.get()
                }
                
                links.append(link_info)
            
        except Exception as e:
            self.logger.error(f"Failed to extract article links: {e}")
        
        return links
    
    def parse_article(self, response) -> Generator:
        """Parse individual article page rendered with JavaScript"""
        link_info = response.meta.get('link_info', {})
        
        try:
            # Verify article content is loaded
            if not self._verify_article_loaded(response):
                self.logger.warning(f"Article content may not be fully loaded: {response.url}")
            
            # Extract article data
            title = self.extract_title(response)
            content = self.extract_content(response)
            summary = self.extract_summary(response, content)
            author = self.extract_author(response)
            publication_date = self.extract_publication_date(response)
            
            # Validate essential fields
            if not title:
                self.logger.warning(f"No title found for {response.url}")
                return
            
            if not content and not summary:
                self.logger.warning(f"No content found for {response.url}")
                return
            
            # Build metadata
            metadata = {
                'source_type': 'javascript',
                'rendering_engine': 'playwright',
                'page_title': response.css('title::text').get() or '',
                'preview_title': link_info.get('preview_title', ''),
                'language': self.extract_language(response),
                'js_frameworks': self.detect_js_frameworks(response),
            }
            
            # Extract dynamic content metadata
            dynamic_metadata = self.extract_dynamic_metadata(response)
            metadata.update(dynamic_metadata)
            
            # Create article item
            article_item = self.create_article_item(
                url=response.url,
                title=title,
                content_summary=summary,
                full_content=content,
                publication_date=publication_date,
                author=author,
                metadata=metadata,
                rendering_method='playwright'
            )
            
            self.logger.debug(f"Successfully parsed JS article: {title}")
            yield article_item
            
        except Exception as e:
            self.logger.error(f"Failed to parse JS article {response.url}: {e}")
    
    def _verify_page_loaded(self, response) -> bool:
        """Verify that the main page is fully loaded"""
        # Check if overview selector elements are present
        if self.overview_selector:
            elements = response.css(self.overview_selector)
            return len(elements) > 0
        
        # Check for common loading indicators
        loading_indicators = [
            '.loading', '.spinner', '#loading', '[data-loading]',
            '.loading-spinner', '.loader'
        ]
        
        for indicator in loading_indicators:
            if response.css(indicator):
                return False
        
        return True
    
    def _verify_article_loaded(self, response) -> bool:
        """Verify that article content is fully loaded"""
        # Check if content selector has content
        if self.content_selector:
            content_element = response.css(self.content_selector)
            if content_element:
                content_text = ' '.join(content_element.css('::text').getall())
                return len(content_text.strip()) > 50  # Minimum content length
        
        return True
    
    def extract_title(self, response) -> str:
        """Extract article title from JavaScript-rendered content"""
        if self.title_selector:
            title_element = response.css(self.title_selector)
            if title_element:
                title = title_element.css('::text').get()
                if title:
                    return self.clean_text(title)
        
        # Fallback to page title
        page_title = response.css('title::text').get()
        if page_title:
            return self.clean_text(page_title)
        
        return ""
    
    def extract_content(self, response) -> str:
        """Extract article content from JavaScript-rendered page"""
        if not self.content_selector:
            return ""
        
        try:
            content_elements = response.css(self.content_selector)
            if content_elements:
                # Extract all text from content elements
                content_parts = content_elements.css('::text').getall()
                content = ' '.join(content_parts)
                return self.clean_text(content)
        except Exception as e:
            self.logger.error(f"Failed to extract content: {e}")
        
        return ""
    
    def extract_summary(self, response, full_content: str = "") -> str:
        """Extract article summary"""
        # Try meta description first
        meta_desc = response.css('meta[name="description"]::attr(content)').get()
        if meta_desc:
            return self.clean_text(meta_desc)
        
        # Generate summary from content
        if full_content:
            summary_length = 500
            return (
                full_content[:summary_length] + "..." 
                if len(full_content) > summary_length 
                else full_content
            )
        
        return ""
    
    def extract_author(self, response) -> str:
        """Extract article author"""
        if self.author_selector:
            try:
                author_element = response.css(self.author_selector)
                if author_element:
                    author = author_element.css('::text').get()
                    if author:
                        return self.clean_text(author)
            except Exception as e:
                self.logger.error(f"Failed to extract author: {e}")
        
        return ""
    
    def extract_publication_date(self, response) -> str:
        """Extract publication date"""
        if self.date_selector:
            try:
                date_element = response.css(self.date_selector)
                if date_element:
                    date_text = date_element.css('::text').get()
                    if date_text:
                        parsed_date = self.parse_date(date_text)
                        if parsed_date:
                            return parsed_date
            except Exception as e:
                self.logger.error(f"Failed to extract date: {e}")
        
        return ""
    
    def extract_language(self, response) -> str:
        """Extract page language"""
        lang = response.css('html::attr(lang)').get()
        if lang:
            return lang[:2]
        return 'en'
    
    def detect_js_frameworks(self, response) -> List[str]:
        """Detect JavaScript frameworks used on the page"""
        frameworks = []
        
        # Check for common framework signatures
        page_content = response.text.lower()
        
        framework_signatures = {
            'react': ['react', 'reactdom', '__react'],
            'vue': ['vue.js', 'vue.min.js', '__vue__'],
            'angular': ['angular', 'ng-app', '@angular'],
            'jquery': ['jquery', '$.', 'jquery.min.js'],
            'bootstrap': ['bootstrap', 'bootstrap.min.js'],
            'next.js': ['next.js', '__next', '_next'],
            'nuxt': ['nuxt', '__nuxt'],
        }
        
        for framework, signatures in framework_signatures.items():
            if any(sig in page_content for sig in signatures):
                frameworks.append(framework)
        
        return frameworks
    
    def extract_dynamic_metadata(self, response) -> Dict[str, Any]:
        """Extract metadata specific to JavaScript-rendered content"""
        metadata = {}
        
        # Check for Single Page Application indicators
        if response.css('[data-reactroot]') or '__REACT__' in response.text:
            metadata['spa_type'] = 'react'
        elif '__vue__' in response.text or response.css('[data-v-]'):
            metadata['spa_type'] = 'vue'
        elif 'ng-version' in response.text or response.css('[ng-app]'):
            metadata['spa_type'] = 'angular'
        
        # Extract JSON-LD structured data
        json_ld_scripts = response.css('script[type="application/ld+json"]::text').getall()
        if json_ld_scripts:
            metadata['json_ld_count'] = len(json_ld_scripts)
        
        # Check for lazy loading
        if response.css('[data-src]') or 'loading="lazy"' in response.text:
            metadata['has_lazy_loading'] = True
        
        return metadata


def create_playwright_spider_class(site_config: Dict[str, Any]) -> type:
    """Factory function to create Playwright spider class from configuration"""
    
    class DynamicPlaywrightSpider(PlaywrightSpiderTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(site_config, *args, **kwargs)
    
    # Set dynamic class name
    spider_name = site_config.get('name', 'Unknown').replace(' ', '').replace('-', '')
    DynamicPlaywrightSpider.__name__ = f"Playwright{spider_name}Spider"
    
    return DynamicPlaywrightSpider