"""
RSS Spider Template for Mimir news scraping
Handles RSS and Atom feeds with full content extraction
"""

import feedparser
from datetime import datetime
from typing import Dict, Any, Generator

from .base_spider import BaseMimirSpider


class RSSSpiderTemplate(BaseMimirSpider):
    """Template for RSS feed spiders"""
    
    def __init__(self, site_config: Dict[str, Any], *args, **kwargs):
        super().__init__(site_config, *args, **kwargs)
        
        # RSS-specific configuration
        self.feed_format = site_config.get('feed_format', 'rss')  # rss, atom, or auto
        self.extract_full_content = site_config.get('extract_full_content', False)
        self.follow_article_links = site_config.get('follow_article_links', True)
        
    def parse(self, response) -> Generator:
        """Parse RSS feed and extract articles"""
        self.logger.info(f"Parsing RSS feed: {response.url}")
        
        try:
            # Parse RSS content using feedparser
            feed = feedparser.parse(response.text)
            
            # Check for parsing errors
            if feed.bozo and feed.bozo_exception:
                self.logger.warning(
                    f"RSS feed has issues: {feed.bozo_exception}"
                )
            
            # Extract feed metadata
            feed_info = {
                'title': getattr(feed.feed, 'title', ''),
                'description': getattr(feed.feed, 'description', ''),
                'language': getattr(feed.feed, 'language', 'en'),
                'last_updated': getattr(feed.feed, 'updated', ''),
                'generator': getattr(feed.feed, 'generator', ''),
                'link': getattr(feed.feed, 'link', ''),
            }
            
            self.logger.info(
                f"Feed parsed: {feed_info['title']} - "
                f"{len(feed.entries)} entries found"
            )
            
            # Process entries
            processed_count = 0
            for entry in feed.entries:
                if processed_count >= self.max_articles:
                    break
                
                try:
                    # Extract article from RSS entry
                    article_item = self.parse_rss_entry(entry, feed_info)
                    
                    if article_item:
                        # If extract_full_content is enabled and we have article URL,
                        # follow the link to get full content
                        if (self.extract_full_content and 
                            self.follow_article_links and 
                            article_item.get('url')):
                            
                            yield response.follow(
                                article_item['url'],
                                callback=self.parse_article_page,
                                meta={'article_item': article_item}
                            )
                        else:
                            yield article_item
                        
                        processed_count += 1
                
                except Exception as e:
                    self.logger.error(
                        f"Failed to parse RSS entry: {e}"
                    )
                    continue
            
            self.logger.info(f"Processed {processed_count} articles from RSS feed")
            
        except Exception as e:
            self.logger.error(f"Failed to parse RSS feed {response.url}: {e}")
    
    def parse_rss_entry(self, entry, feed_info: Dict[str, Any]):
        """Parse individual RSS entry into article item"""
        try:
            # Extract basic information
            title = self.clean_text(getattr(entry, 'title', ''))
            link = getattr(entry, 'link', '')
            
            if not title or not link:
                self.logger.debug("Skipping entry with missing title or link")
                return None
            
            # Extract content
            content_summary = self._extract_summary(entry)
            full_content = self._extract_full_content(entry)
            
            # Extract publication date
            publication_date = self._extract_publication_date(entry)
            
            # Extract author
            author = self._extract_author(entry)
            
            # Extract tags/categories
            tags = self._extract_tags(entry)
            
            # Build metadata
            metadata = {
                'entry_id': getattr(entry, 'id', ''),
                'feed_title': feed_info.get('title', ''),
                'feed_language': feed_info.get('language', 'en'),
                'source_type': 'rss',
                'guid': getattr(entry, 'guid', ''),
                'comments_url': getattr(entry, 'comments', ''),
            }
            
            # Add enclosures (media attachments)
            if hasattr(entry, 'enclosures') and entry.enclosures:
                metadata['enclosures'] = [
                    {
                        'url': enc.get('href', ''),
                        'type': enc.get('type', ''),
                        'length': enc.get('length', '')
                    }
                    for enc in entry.enclosures
                ]
            
            # Create article item
            article_item = self.create_article_item(
                url=link,
                title=title,
                content_summary=content_summary,
                full_content=full_content,
                publication_date=publication_date,
                author=author,
                metadata=metadata,
                tags=tags,
                rendering_method='rss'
            )
            
            return article_item
            
        except Exception as e:
            self.logger.error(
                f"Error parsing RSS entry '{getattr(entry, 'title', 'Unknown')}': {e}"
            )
            return None
    
    def parse_article_page(self, response):
        """Parse full article page to extract complete content"""
        article_item = response.meta.get('article_item')
        
        if not article_item:
            self.logger.error("No article item in meta data")
            return
        
        try:
            # Get content selectors from configuration
            selectors = self.site_config.get('selectors', {})
            content_selector = selectors.get('detail_content', 'article, .content, .post-content, .entry-content')
            
            # Extract full content from article page
            content_elements = response.css(content_selector)
            if content_elements:
                full_content = ' '.join(content_elements.css('::text').getall())
                full_content = self.clean_text(full_content)
                
                # Update article item with full content
                article_item['full_content'] = full_content
                
                # Update content summary if it was empty
                if not article_item.get('content_summary') and full_content:
                    summary_length = 500
                    article_item['content_summary'] = (
                        full_content[:summary_length] + "..." 
                        if len(full_content) > summary_length 
                        else full_content
                    )
                
                self.logger.debug(f"Extracted full content for: {article_item['url']}")
            else:
                self.logger.warning(
                    f"No content found with selector '{content_selector}' "
                    f"for {response.url}"
                )
            
            yield article_item
            
        except Exception as e:
            self.logger.error(f"Failed to extract full content from {response.url}: {e}")
            # Yield the original item even if full content extraction failed
            yield article_item
    
    def _extract_summary(self, entry) -> str:
        """Extract content summary from RSS entry"""
        # Try different summary fields
        for field in ['summary', 'description', 'subtitle']:
            if hasattr(entry, field):
                summary = getattr(entry, field, '')
                if summary:
                    return self.clean_text(summary)
        return ""
    
    def _extract_full_content(self, entry) -> str:
        """Extract full content from RSS entry"""
        # Try content field
        if hasattr(entry, 'content') and entry.content:
            if isinstance(entry.content, list) and entry.content:
                # Take the first content item
                content_item = entry.content[0]
                content = content_item.get('value', '')
            else:
                content = str(entry.content)
            
            return self.clean_text(content)
        
        # Fallback to summary if no content field
        return self._extract_summary(entry)
    
    def _extract_publication_date(self, entry) -> str:
        """Extract publication date from RSS entry"""
        # Try different date fields
        for field in ['published', 'updated', 'created']:
            if hasattr(entry, field):
                date_str = getattr(entry, field, '')
                if date_str:
                    parsed_date = self.parse_date(date_str)
                    if parsed_date:
                        return parsed_date
        
        # Try parsed date fields
        for field in ['published_parsed', 'updated_parsed']:
            if hasattr(entry, field):
                date_parsed = getattr(entry, field)
                if date_parsed:
                    try:
                        dt = datetime(*date_parsed[:6])
                        return dt.isoformat()
                    except Exception:
                        continue
        
        return datetime.now().isoformat()
    
    def _extract_author(self, entry) -> str:
        """Extract author from RSS entry"""
        # Try author field
        if hasattr(entry, 'author'):
            return self.clean_text(entry.author)
        
        # Try author_detail
        if hasattr(entry, 'author_detail') and hasattr(entry.author_detail, 'name'):
            return self.clean_text(entry.author_detail.name)
        
        # Try contributors
        if hasattr(entry, 'contributors') and entry.contributors:
            contributor = entry.contributors[0]
            if hasattr(contributor, 'name'):
                return self.clean_text(contributor.name)
        
        return ""
    
    def _extract_tags(self, entry) -> list:
        """Extract tags/categories from RSS entry"""
        tags = []
        
        # Extract from tags field
        if hasattr(entry, 'tags') and entry.tags:
            for tag in entry.tags:
                if hasattr(tag, 'term'):
                    tags.append(tag.term)
                elif hasattr(tag, 'label'):
                    tags.append(tag.label)
        
        # Extract from category field
        if hasattr(entry, 'category') and entry.category:
            tags.append(entry.category)
        
        return tags


def create_rss_spider_class(site_config: Dict[str, Any]) -> type:
    """Factory function to create RSS spider class from configuration"""
    
    class DynamicRSSSpider(RSSSpiderTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(site_config, *args, **kwargs)
    
    # Set dynamic class name
    spider_name = site_config.get('name', 'Unknown').replace(' ', '').replace('-', '')
    DynamicRSSSpider.__name__ = f"RSS{spider_name}Spider"
    
    return DynamicRSSSpider