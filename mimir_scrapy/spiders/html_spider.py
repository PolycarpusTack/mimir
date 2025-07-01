"""
HTML Spider Template for Mimir news scraping
Handles HTML websites with article discovery and content extraction
"""

from typing import Dict, Any, Generator, List
from urllib.parse import urljoin

from .base_spider import BaseMimirSpider


class HTMLSpiderTemplate(BaseMimirSpider):
    """Template for HTML website spiders"""
    
    def __init__(self, site_config: Dict[str, Any], *args, **kwargs):
        super().__init__(site_config, *args, **kwargs)
        
        # HTML-specific configuration
        self.selectors = site_config.get('selectors', {})
        self.pagination = site_config.get('pagination', {})
        self.article_discovery = site_config.get('article_discovery', {})
        
        # Extract selectors
        self.overview_selector = self.selectors.get('overview_article_link', '')
        self.title_selector = self.selectors.get('detail_title', '')
        self.content_selector = self.selectors.get('detail_content', '')
        self.author_selector = self.selectors.get('detail_author', '')
        self.date_selector = self.selectors.get('detail_date', '')
        self.summary_selector = self.selectors.get('detail_summary', '')
        
        # Pagination settings
        self.pagination_enabled = self.pagination.get('enabled', False)
        self.next_page_selector = self.pagination.get('next_page_selector', '')
        self.max_pages = self.pagination.get('max_pages', 5)
        self.current_page = 0
        
        # Article discovery settings
        self.follow_internal_links = self.article_discovery.get('follow_internal_links', True)
        self.link_patterns = self.article_discovery.get('link_patterns', [])
        self.exclude_patterns = self.article_discovery.get('exclude_patterns', [])
    
    def parse(self, response) -> Generator:
        """Parse main page and discover article links"""
        self.logger.info(f"Parsing HTML page: {response.url}")
        self.current_page += 1
        
        try:
            # Extract article links from overview page
            article_links = self.extract_article_links(response)
            
            self.logger.info(f"Found {len(article_links)} article links on page {self.current_page}")
            
            # Follow article links
            articles_processed = 0
            for link_info in article_links:
                if articles_processed >= self.max_articles:
                    break
                
                url = link_info['url']
                if self.should_follow_link(url):
                    yield response.follow(
                        url,
                        callback=self.parse_article,
                        meta={'link_info': link_info}
                    )
                    articles_processed += 1
                else:
                    self.logger.debug(f"Skipping URL: {url}")
            
            # Handle pagination
            if (self.pagination_enabled and 
                self.current_page < self.max_pages and 
                self.next_page_selector):
                
                next_page_url = self.extract_next_page_url(response)
                if next_page_url:
                    self.logger.info(f"Following pagination to page {self.current_page + 1}")
                    yield response.follow(
                        next_page_url,
                        callback=self.parse
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to parse HTML page {response.url}: {e}")
    
    def extract_article_links(self, response) -> List[Dict[str, Any]]:
        """Extract article links from overview page"""
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
                
                # Extract preview information if available
                preview_title = element.css('::text').get() or ''
                preview_title = self.clean_text(preview_title)
                
                # Check link patterns if configured
                if self.link_patterns and not self._matches_patterns(absolute_url, self.link_patterns):
                    continue
                
                # Check exclude patterns
                if self.exclude_patterns and self._matches_patterns(absolute_url, self.exclude_patterns):
                    continue
                
                link_info = {
                    'url': absolute_url,
                    'preview_title': preview_title,
                    'source_element': element.get()
                }
                
                links.append(link_info)
            
        except Exception as e:
            self.logger.error(f"Failed to extract article links: {e}")
        
        return links
    
    def extract_next_page_url(self, response) -> str:
        """Extract next page URL for pagination"""
        if not self.next_page_selector:
            return ""
        
        try:
            next_page_element = response.css(self.next_page_selector)
            if next_page_element:
                href = next_page_element.css('::attr(href)').get()
                if href:
                    return self.make_absolute_url(href, response.url)
        except Exception as e:
            self.logger.error(f"Failed to extract next page URL: {e}")
        
        return ""
    
    def parse_article(self, response) -> Generator:
        """Parse individual article page"""
        link_info = response.meta.get('link_info', {})
        
        try:
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
                'source_type': 'html',
                'page_title': response.css('title::text').get() or '',
                'meta_description': response.css('meta[name="description"]::attr(content)').get() or '',
                'canonical_url': response.css('link[rel="canonical"]::attr(href)').get() or '',
                'preview_title': link_info.get('preview_title', ''),
                'language': self.extract_language(response),
            }
            
            # Extract additional metadata
            keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords:
                metadata['meta_keywords'] = [k.strip() for k in keywords.split(',')]
            
            # Extract Open Graph data
            og_data = self.extract_open_graph_data(response)
            if og_data:
                metadata['open_graph'] = og_data
            
            # Create article item
            article_item = self.create_article_item(
                url=response.url,
                title=title,
                content_summary=summary,
                full_content=content,
                publication_date=publication_date,
                author=author,
                metadata=metadata,
                rendering_method='html'
            )
            
            self.logger.debug(f"Successfully parsed article: {title}")
            yield article_item
            
        except Exception as e:
            self.logger.error(f"Failed to parse article {response.url}: {e}")
    
    def extract_title(self, response) -> str:
        """Extract article title"""
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
        """Extract article content"""
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
        # Try configured summary selector first
        if self.summary_selector:
            try:
                summary_element = response.css(self.summary_selector)
                if summary_element:
                    summary = summary_element.css('::text').get()
                    if summary:
                        return self.clean_text(summary)
            except Exception as e:
                self.logger.error(f"Failed to extract summary: {e}")
        
        # Try meta description
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
        
        # Try meta author
        meta_author = response.css('meta[name="author"]::attr(content)').get()
        if meta_author:
            return self.clean_text(meta_author)
        
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
        
        # Try various meta date fields
        meta_selectors = [
            'meta[property="article:published_time"]::attr(content)',
            'meta[name="publishdate"]::attr(content)',
            'meta[name="date"]::attr(content)',
            'time[datetime]::attr(datetime)',
            'time::attr(datetime)'
        ]
        
        for selector in meta_selectors:
            try:
                date_value = response.css(selector).get()
                if date_value:
                    parsed_date = self.parse_date(date_value)
                    if parsed_date:
                        return parsed_date
            except Exception:
                continue
        
        return ""
    
    def extract_language(self, response) -> str:
        """Extract page language"""
        # Try HTML lang attribute
        lang = response.css('html::attr(lang)').get()
        if lang:
            return lang[:2]  # Return just language code
        
        # Try meta content-language
        meta_lang = response.css('meta[http-equiv="content-language"]::attr(content)').get()
        if meta_lang:
            return meta_lang[:2]
        
        return 'en'  # Default to English
    
    def extract_open_graph_data(self, response) -> Dict[str, str]:
        """Extract Open Graph metadata"""
        og_data = {}
        
        og_selectors = {
            'title': 'meta[property="og:title"]::attr(content)',
            'description': 'meta[property="og:description"]::attr(content)',
            'image': 'meta[property="og:image"]::attr(content)',
            'url': 'meta[property="og:url"]::attr(content)',
            'type': 'meta[property="og:type"]::attr(content)',
            'site_name': 'meta[property="og:site_name"]::attr(content)',
        }
        
        for key, selector in og_selectors.items():
            try:
                value = response.css(selector).get()
                if value:
                    og_data[key] = value.strip()
            except Exception:
                continue
        
        return og_data
    
    def _matches_patterns(self, url: str, patterns: List[str]) -> bool:
        """Check if URL matches any of the given patterns"""
        for pattern in patterns:
            if pattern in url:
                return True
        return False


def create_html_spider_class(site_config: Dict[str, Any]) -> type:
    """Factory function to create HTML spider class from configuration"""
    
    class DynamicHTMLSpider(HTMLSpiderTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(site_config, *args, **kwargs)
    
    # Set dynamic class name
    spider_name = site_config.get('name', 'Unknown').replace(' ', '').replace('-', '')
    DynamicHTMLSpider.__name__ = f"HTML{spider_name}Spider"
    
    return DynamicHTMLSpider