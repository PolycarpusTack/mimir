"""RSS feed scraper implementation."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser

from .base import BaseScraper, ScrapedArticle, ScrapingResult


class RSScraper(BaseScraper):
    """Scraper for RSS feeds."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.user_agent = self.config.get(
            "user_agent",
            "Mimir Broadcast Intelligence Bot 1.0 (+https://mimir.ai/bot)"
        )
    
    def validate_config(self) -> bool:
        """Validate RSS scraper configuration."""
        required_fields = ["url"]
        
        for field in required_fields:
            if field not in self.config:
                self.logger.error("Missing required configuration field", field=field)
                return False
        
        # Validate URL format
        url = self.config["url"]
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                self.logger.error("Invalid URL format", url=url)
                return False
        except Exception as e:
            self.logger.error("URL validation failed", url=url, error=str(e))
            return False
        
        return True
    
    async def scrape(self, max_articles: int = 20) -> ScrapingResult:
        """Scrape articles from RSS feed."""
        if not self.validate_config():
            return ScrapingResult(
                articles=[],
                total_found=0,
                errors=["Invalid configuration"]
            )
        
        url = self.config["url"]
        errors = []
        articles = []
        
        try:
            # Fetch RSS content
            content = await self._fetch_feed_content(url)
            if not content:
                return ScrapingResult(
                    articles=[],
                    total_found=0,
                    errors=["Failed to fetch RSS content"]
                )
            
            # Parse RSS feed
            feed = feedparser.parse(content)
            
            if feed.bozo and feed.bozo_exception:
                self.logger.warning(
                    "RSS feed has issues",
                    url=url,
                    error=str(feed.bozo_exception)
                )
                errors.append(f"RSS parsing warning: {feed.bozo_exception}")
            
            # Extract articles
            entries = feed.entries[:max_articles]
            total_found = len(feed.entries)
            
            for entry in entries:
                try:
                    article = await self._parse_rss_entry(entry)
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.error(
                        "Failed to parse RSS entry",
                        entry_title=getattr(entry, "title", "Unknown"),
                        error=str(e)
                    )
                    errors.append(f"Entry parsing error: {str(e)}")
                
                # Rate limiting
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
            
            self.logger.info(
                "RSS scraping completed",
                url=url,
                total_found=total_found,
                articles_parsed=len(articles),
                errors=len(errors)
            )
            
            return ScrapingResult(
                articles=articles,
                total_found=total_found,
                errors=errors,
                metadata={
                    "feed_title": getattr(feed.feed, "title", ""),
                    "feed_description": getattr(feed.feed, "description", ""),
                    "feed_language": getattr(feed.feed, "language", "en"),
                    "last_updated": getattr(feed.feed, "updated", ""),
                }
            )
            
        except Exception as e:
            self.logger.error("RSS scraping failed", url=url, error=str(e))
            return ScrapingResult(
                articles=[],
                total_found=0,
                errors=[f"Scraping failed: {str(e)}"]
            )
    
    async def _fetch_feed_content(self, url: str) -> Optional[str]:
        """Fetch RSS feed content with proper error handling."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                ) as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            content = await response.text()
                            self.logger.debug(
                                "RSS feed fetched successfully",
                                url=url,
                                content_length=len(content),
                                attempt=attempt + 1
                            )
                            return content
                        else:
                            self.logger.warning(
                                "HTTP error fetching RSS feed",
                                url=url,
                                status=response.status,
                                attempt=attempt + 1
                            )
                            
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Timeout fetching RSS feed",
                    url=url,
                    attempt=attempt + 1
                )
            except Exception as e:
                self.logger.warning(
                    "Error fetching RSS feed",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1
                )
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.request_delay * (attempt + 1))
        
        return None
    
    async def _parse_rss_entry(self, entry: Any) -> Optional[ScrapedArticle]:
        """Parse a single RSS entry into a ScrapedArticle."""
        try:
            # Extract basic fields
            title = self._clean_text(getattr(entry, "title", ""))
            link = getattr(entry, "link", "")
            
            if not title or not link:
                self.logger.debug("Skipping entry with missing title or link")
                return None
            
            # Extract content
            content_summary = ""
            full_content = ""
            
            # Try different content fields
            if hasattr(entry, "summary"):
                content_summary = self._clean_text(entry.summary)
            elif hasattr(entry, "description"):
                content_summary = self._clean_text(entry.description)
            
            if hasattr(entry, "content"):
                if isinstance(entry.content, list) and entry.content:
                    full_content = self._clean_text(entry.content[0].value)
                else:
                    full_content = self._clean_text(str(entry.content))
            
            # Parse publication date
            publication_date = None
            for date_field in ["published", "updated", "created"]:
                if hasattr(entry, date_field):
                    date_str = getattr(entry, date_field)
                    publication_date = self._parse_date(date_str)
                    if publication_date:
                        break
            
            # Extract author
            author = ""
            if hasattr(entry, "author"):
                author = self._clean_text(entry.author)
            elif hasattr(entry, "author_detail") and hasattr(entry.author_detail, "name"):
                author = self._clean_text(entry.author_detail.name)
            
            # Extract tags/categories for metadata
            tags = []
            if hasattr(entry, "tags"):
                tags = [tag.term for tag in entry.tags if hasattr(tag, "term")]
            
            metadata = {
                "tags": tags,
                "entry_id": getattr(entry, "id", ""),
                "source_type": "rss",
            }
            
            return ScrapedArticle(
                url=link,
                title=title,
                content_summary=content_summary,
                full_content=full_content,
                publication_date=publication_date,
                author=author,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to parse RSS entry",
                error=str(e),
                entry_title=getattr(entry, "title", "Unknown")
            )
            return None