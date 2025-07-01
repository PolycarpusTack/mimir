"""HTML scraper implementation for websites without RSS feeds."""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from bs4 import BeautifulSoup

from .base import BaseScraper, ScrapedArticle, ScrapingResult


class HTMLScraper(BaseScraper):
    """Scraper for HTML websites."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.user_agent = self.config.get(
            "user_agent",
            "Mimir Broadcast Intelligence Bot 1.0 (+https://mimir.ai/bot)"
        )
        self.respect_robots_txt = self.config.get("respect_robots_txt", True)
    
    def validate_config(self) -> bool:
        """Validate HTML scraper configuration."""
        required_fields = ["url", "selectors"]
        
        for field in required_fields:
            if field not in self.config:
                self.logger.error("Missing required configuration field", field=field)
                return False
        
        # Validate selectors
        selectors = self.config["selectors"]
        required_selectors = ["article_list", "article_link", "title"]
        
        for selector in required_selectors:
            if selector not in selectors:
                self.logger.error("Missing required selector", selector=selector)
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
        """Scrape articles from HTML website."""
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
            # Check robots.txt if enabled
            if self.respect_robots_txt:
                if not await self._check_robots_txt(url):
                    return ScrapingResult(
                        articles=[],
                        total_found=0,
                        errors=["Blocked by robots.txt"]
                    )
            
            # Fetch main page
            content = await self._fetch_page_content(url)
            if not content:
                return ScrapingResult(
                    articles=[],
                    total_found=0,
                    errors=["Failed to fetch main page content"]
                )
            
            # Parse article links
            article_links = await self._extract_article_links(content, url)
            total_found = len(article_links)
            
            if not article_links:
                return ScrapingResult(
                    articles=[],
                    total_found=0,
                    errors=["No article links found on main page"]
                )
            
            # Limit to max_articles
            article_links = article_links[:max_articles]
            
            # Scrape individual articles
            for link in article_links:
                try:
                    article = await self._scrape_article(link)
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.error(
                        "Failed to scrape article",
                        url=link,
                        error=str(e)
                    )
                    errors.append(f"Article scraping error: {str(e)}")
                
                # Rate limiting
                if self.request_delay > 0:
                    await asyncio.sleep(self.request_delay)
            
            self.logger.info(
                "HTML scraping completed",
                base_url=url,
                total_found=total_found,
                articles_scraped=len(articles),
                errors=len(errors)
            )
            
            return ScrapingResult(
                articles=articles,
                total_found=total_found,
                errors=errors,
                metadata={
                    "base_url": url,
                    "scraping_method": "html",
                    "selectors_used": self.config["selectors"],
                }
            )
            
        except Exception as e:
            self.logger.error("HTML scraping failed", url=url, error=str(e))
            return ScrapingResult(
                articles=[],
                total_found=0,
                errors=[f"Scraping failed: {str(e)}"]
            )
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if scraping is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(robots_url) as response:
                    if response.status == 404:
                        # No robots.txt means allowed
                        return True
                    
                    if response.status == 200:
                        robots_content = await response.text()
                        
                        # Use urllib's RobotFileParser
                        rp = RobotFileParser()
                        rp.set_url(robots_url)
                        rp.feed(robots_content)
                        
                        return rp.can_fetch(self.user_agent, url)
            
            return True  # Default to allowed if we can't check
            
        except Exception as e:
            self.logger.warning(
                "Failed to check robots.txt",
                url=url,
                error=str(e)
            )
            return True  # Default to allowed on error
    
    async def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch HTML page content."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
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
                                "Page fetched successfully",
                                url=url,
                                content_length=len(content),
                                attempt=attempt + 1
                            )
                            return content
                        else:
                            self.logger.warning(
                                "HTTP error fetching page",
                                url=url,
                                status=response.status,
                                attempt=attempt + 1
                            )
                            
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Timeout fetching page",
                    url=url,
                    attempt=attempt + 1
                )
            except Exception as e:
                self.logger.warning(
                    "Error fetching page",
                    url=url,
                    error=str(e),
                    attempt=attempt + 1
                )
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.request_delay * (attempt + 1))
        
        return None
    
    async def _extract_article_links(self, content: str, base_url: str) -> List[str]:
        """Extract article links from main page."""
        try:
            soup = BeautifulSoup(content, "html.parser")
            selectors = self.config["selectors"]
            
            article_links = []
            
            # Find article containers
            containers = soup.select(selectors["article_list"])
            
            for container in containers:
                # Find link within container
                link_element = container.select_one(selectors["article_link"])
                
                if link_element:
                    href = link_element.get("href")
                    if href:
                        # Convert relative URLs to absolute
                        full_url = urljoin(base_url, href)
                        article_links.append(full_url)
            
            self.logger.debug(
                "Extracted article links",
                base_url=base_url,
                link_count=len(article_links)
            )
            
            return article_links
            
        except Exception as e:
            self.logger.error(
                "Failed to extract article links",
                base_url=base_url,
                error=str(e)
            )
            return []
    
    async def _scrape_article(self, url: str) -> Optional[ScrapedArticle]:
        """Scrape individual article content."""
        content = await self._fetch_page_content(url)
        if not content:
            return None
        
        try:
            soup = BeautifulSoup(content, "html.parser")
            selectors = self.config["selectors"]
            
            # Extract title
            title = ""
            title_element = soup.select_one(selectors["title"])
            if title_element:
                title = self._clean_text(title_element.get_text())
            
            if not title:
                self.logger.debug("No title found", url=url)
                return None
            
            # Extract content
            content_text = ""
            if "content" in selectors:
                content_element = soup.select_one(selectors["content"])
                if content_element:
                    content_text = self._clean_text(content_element.get_text())
            
            # Extract date
            publication_date = None
            if "date" in selectors:
                date_element = soup.select_one(selectors["date"])
                if date_element:
                    date_str = date_element.get_text().strip()
                    publication_date = self._parse_date(date_str)
            
            # Extract author
            author = ""
            if "author" in selectors:
                author_element = soup.select_one(selectors["author"])
                if author_element:
                    author = self._clean_text(author_element.get_text())
            
            # Try to extract structured data as fallback
            if not content_text or not publication_date:
                structured_data = await self._extract_structured_data(soup)
                if structured_data:
                    if not content_text and "content" in structured_data:
                        content_text = structured_data["content"]
                    if not publication_date and "date" in structured_data:
                        publication_date = self._parse_date(structured_data["date"])
                    if not author and "author" in structured_data:
                        author = structured_data["author"]
            
            metadata = {
                "source_type": "html",
                "extraction_method": "css_selectors",
                "selectors_used": selectors,
            }
            
            return ScrapedArticle(
                url=url,
                title=title,
                content_summary=content_text[:500] if content_text else "",
                full_content=content_text,
                publication_date=publication_date,
                author=author,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to parse article",
                url=url,
                error=str(e)
            )
            return None
    
    async def _extract_structured_data(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract structured data (JSON-LD, meta tags) as fallback."""
        structured_data = {}
        
        try:
            # Try JSON-LD structured data
            json_ld_scripts = soup.find_all("script", type="application/ld+json")
            for script in json_ld_scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    
                    if isinstance(data, dict):
                        if data.get("@type") in ["NewsArticle", "Article"]:
                            if "headline" in data:
                                structured_data["title"] = data["headline"]
                            if "articleBody" in data:
                                structured_data["content"] = data["articleBody"]
                            if "datePublished" in data:
                                structured_data["date"] = data["datePublished"]
                            if "author" in data:
                                author_data = data["author"]
                                if isinstance(author_data, dict) and "name" in author_data:
                                    structured_data["author"] = author_data["name"]
                                elif isinstance(author_data, str):
                                    structured_data["author"] = author_data
                except:
                    continue
            
            # Try Open Graph meta tags
            if not structured_data.get("title"):
                og_title = soup.find("meta", property="og:title")
                if og_title:
                    structured_data["title"] = og_title.get("content", "")
            
            if not structured_data.get("content"):
                og_description = soup.find("meta", property="og:description")
                if og_description:
                    structured_data["content"] = og_description.get("content", "")
            
        except Exception as e:
            self.logger.debug(
                "Failed to extract structured data",
                error=str(e)
            )
        
        return structured_data