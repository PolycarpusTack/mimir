"""
Mimir Async News Scraper - EPIC 4 Implementation
High-Performance Async Architecture with aiohttp, connection pooling, and rate limiting
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import feedparser
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

import db_manager
from async_db_manager import AsyncDatabaseManager, close_async_db, get_async_db, initialize_async_db
from async_monitor import AsyncResourceMonitor, ConcurrencyController, MonitoredAsyncSession
from event_streaming import EventPublisher, EventType
from redis_cache import CacheConfig, RedisCache, async_cache_result
from resilience import (
    CircuitBreakerConfig,
    ResilienceManager,
    ResilientHttpClient,
    RetryConfig,
    circuit_breaker,
    retry,
    timeout,
)


class AsyncNewsScraper:
    """High-performance async news scraper with connection pooling and rate limiting."""

    def __init__(self, config: dict, sites_config: list):
        self.config = config
        self.sites_config = sites_config
        self.logger = self._setup_logging()

        # Async session configuration
        self.user_agent = config.get("user_agent", "Mimir-AsyncScraper/1.0")
        self.default_delay = config.get("default_request_delay_seconds", 2)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 5)
        self.concurrent_limit = config.get("concurrent_requests", 10)
        self.timeout = config.get("request_timeout_seconds", 30)

        # Connection pooling settings
        self.connector_limit = config.get("connector_limit", 100)
        self.connector_limit_per_host = config.get("connector_limit_per_host", 30)

        # Rate limiting - domain-specific delays
        self.domain_delays: Dict[str, float] = {}
        self.last_request_times: Dict[str, float] = {}

        # Session will be created in async context
        self.session: Optional[ClientSession] = None
        self.semaphore: Optional[asyncio.Semaphore] = None

        # Monitoring and concurrency control
        self.monitor = AsyncResourceMonitor()
        self.concurrency_controller = ConcurrencyController(
            initial_limit=self.concurrent_limit,
            min_limit=max(2, self.concurrent_limit // 5),
            max_limit=min(100, self.concurrent_limit * 3),
        )

        # Redis cache setup
        redis_config = config.get("redis", {})
        redis_host = redis_config.get("host", "localhost")
        redis_port = redis_config.get("port", 6379)
        redis_db = redis_config.get("db", 0)
        redis_password = redis_config.get("password", None)

        if redis_password:
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
        else:
            redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        cache_config = CacheConfig(default_ttl=config.get("cache_ttl_seconds", 3600), max_connections=20)

        self.cache = RedisCache(redis_url, cache_config)
        self.cache_enabled = config.get("enable_cache", True)

        # Event streaming setup
        self.event_publisher = EventPublisher(redis_url)
        self.events_enabled = config.get("enable_events", True)

        # Async database setup
        self.async_db: Optional[AsyncDatabaseManager] = None
        self.use_async_db = config.get("use_async_db", True)

        # Resilience setup
        self.resilience_manager = ResilienceManager()
        self.resilient_client: Optional[ResilientHttpClient] = None
        self.use_resilience = config.get("use_resilience", True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for async scraper."""
        log_config = self.config.get("logging", {})

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_config.get("file", "logs/mimir_async_scraper.log"))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configure logger
        logger = logging.getLogger("mimir_async_scraper")
        logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler with rotation
            file_handler = RotatingFileHandler(
                log_config.get("file", "logs/mimir_async_scraper.log"),
                maxBytes=log_config.get("max_bytes", 10485760),  # 10MB
                backupCount=log_config.get("backup_count", 5),
            )
            file_handler.setFormatter(
                logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            )
            logger.addHandler(file_handler)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(console_handler)

        return logger

    async def __aenter__(self):
        """Async context manager entry."""
        await self.create_session()
        await self.monitor.start_monitoring()

        # Initialize cache connection
        if self.cache_enabled:
            try:
                await self.cache.async_connect()
                self.logger.info("Cache system initialized")
            except Exception as e:
                self.logger.warning(f"Cache initialization failed: {e}")
                self.cache_enabled = False

        # Initialize event streaming
        if self.events_enabled:
            try:
                await self.event_publisher.async_connect()
                self.logger.info("Event streaming initialized")
            except Exception as e:
                self.logger.warning(f"Event streaming initialization failed: {e}")
                self.events_enabled = False

        # Initialize async database
        if self.use_async_db:
            try:
                await initialize_async_db(self.config)
                self.async_db = get_async_db()
                self.logger.info("Async database initialized")
            except Exception as e:
                self.logger.warning(f"Async database initialization failed: {e}")
                self.use_async_db = False
                # Fall back to sync database
                db_manager.init_db()

        # Initialize resilient HTTP client
        if self.use_resilience:
            try:
                circuit_config = CircuitBreakerConfig(
                    failure_threshold=config.get("circuit_breaker_threshold", 5),
                    timeout=config.get("circuit_breaker_timeout", 60.0),
                )
                retry_config = RetryConfig(
                    max_attempts=config.get("max_retries", 3),
                    base_delay=config.get("retry_delay_seconds", 1.0),
                    max_delay=config.get("max_retry_delay", 30.0),
                )

                self.resilient_client = ResilientHttpClient(
                    circuit_breaker_config=circuit_config,
                    retry_config=retry_config,
                    timeout=self.timeout,
                    max_concurrent=self.concurrent_limit,
                )
                await self.resilient_client.__aenter__()
                self.logger.info("Resilient HTTP client initialized")
            except Exception as e:
                self.logger.warning(f"Resilient client initialization failed: {e}")
                self.use_resilience = False

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.monitor.stop_monitoring()

        # Close cache connection
        if self.cache_enabled:
            await self.cache.async_disconnect()

        # Close event streaming
        if self.events_enabled:
            await self.event_publisher.async_disconnect()

        # Close async database
        if self.use_async_db:
            await close_async_db()

        # Close resilient client
        if self.use_resilience and self.resilient_client:
            await self.resilient_client.__aexit__(exc_type, exc_val, exc_tb)

        await self.close_session()

        # Print final performance report
        self.monitor.print_status_report()

        # Print cache statistics
        if self.cache_enabled:
            cache_stats = self.cache.get_stats()
            print(f"\nCache Statistics:")
            print(f"Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
            print(f"Total Operations: {cache_stats.get('total_operations', 0)}")
            print(f"Errors: {cache_stats.get('errors', 0)}")

        # Print resilience statistics
        if self.use_resilience and self.resilient_client:
            health_status = self.resilient_client.get_health_status()
            print(f"\nResilience Health:")
            circuit_state = health_status["circuit_breaker"]["state"]
            failure_count = health_status["circuit_breaker"]["failure_count"]
            print(f"Circuit Breaker: {circuit_state} (failures: {failure_count})")
            bulkhead_stats = health_status["bulkhead"]
            print(f"Bulkhead: {bulkhead_stats['active_requests']}/{bulkhead_stats['max_concurrent']} active")

    async def create_session(self):
        """Create aiohttp session with connection pooling."""
        # Configure TCP connector with connection pooling
        connector = TCPConnector(
            limit=self.connector_limit,
            limit_per_host=self.connector_limit_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            use_dns_cache=True,
            ttl_dns_cache=300,
        )

        # Configure timeout
        timeout = ClientTimeout(total=self.timeout)

        # Create session with headers
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.session = ClientSession(connector=connector, timeout=timeout, headers=headers)

        # Create semaphore for concurrent request limiting
        self.semaphore = asyncio.Semaphore(self.concurrent_limit)

        self.logger.info(f"Async session created with {self.concurrent_limit} concurrent connections")

    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.logger.info("Async session closed")

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting."""
        return urlparse(url).netloc

    async def _rate_limit(self, url: str):
        """Implement domain-specific rate limiting."""
        domain = self._get_domain(url)

        # Use domain-specific delay or default
        delay = self.domain_delays.get(domain, self.default_delay)

        # Check if we need to wait
        last_request = self.last_request_times.get(domain, 0)
        time_since_last = time.time() - last_request

        if time_since_last < delay:
            wait_time = delay - time_since_last
            self.logger.debug(f"Rate limiting {domain}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)

        # Update last request time
        self.last_request_times[domain] = time.time()

    async def check_robots_txt(self, base_url: str, path: str = "/") -> bool:
        """Check if we can scrape a path according to robots.txt."""
        try:
            # This is still synchronous as RobotFileParser doesn't have async support
            # In production, consider caching robots.txt results
            rp = RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            rp.read()
            can_fetch = rp.can_fetch(self.user_agent, urljoin(base_url, path))
            if not can_fetch:
                self.logger.warning(f"Robots.txt forbids access to {path} on {base_url}")
            return can_fetch
        except Exception as e:
            self.logger.warning(f"Could not read robots.txt for {base_url}: {e}")
            return True  # When in doubt, proceed (but be careful)

    async def fetch_url(self, url: str, retries: int = 0, use_cache: bool = True) -> Optional[str]:
        """Fetch URL content with retry mechanism, rate limiting, and caching."""
        # Check cache first
        if use_cache and self.cache_enabled:
            cached_content = await self.cache.async_get("html_page", url)
            if cached_content is not None:
                self.logger.debug(f"Cache hit for: {url}")
                return cached_content

        # Record request start for monitoring
        self.monitor.record_request_start()
        start_time = time.time()

        try:
            # Apply rate limiting
            await self._rate_limit(url)

            self.logger.debug(f"Fetching: {url}")

            # Use resilient client if available, otherwise fall back to regular client
            if self.use_resilience and self.resilient_client:
                content = await self.resilient_client.get(url)
            else:
                # Use advanced concurrency controller for non-resilient requests
                await self.concurrency_controller.acquire()
                try:
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        content = await response.text()
                finally:
                    self.concurrency_controller.release()

            content_length = len(content.encode("utf-8"))

            # Cache the content if caching is enabled
            if use_cache and self.cache_enabled:
                await self.cache.async_set("html_page", url, content)

            # Record successful request
            response_time = time.time() - start_time
            self.monitor.record_request_end(True, response_time, content_length)

            self.logger.debug(f"Successfully fetched: {url}")
            return content

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            self.monitor.record_request_end(False, response_time)

            error_msg = f"Timeout fetching {url}"
            self.logger.warning(error_msg)
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self.fetch_url(url, retries + 1)
            self.logger.error(f"Permanently failed (timeout): {url}")
            return None

        except aiohttp.ClientError as e:
            response_time = time.time() - start_time
            self.monitor.record_request_end(False, response_time)

            error_msg = f"Client error fetching {url}: {e}"
            self.logger.warning(error_msg)
            if retries < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self.fetch_url(url, retries + 1)
            self.logger.error(f"Permanently failed (client error): {url}")
            return None

        except Exception as e:
            response_time = time.time() - start_time
            self.monitor.record_request_end(False, response_time)

            error_msg = f"Error fetching {url}: {e}"
            self.logger.error(error_msg)

            # For non-resilient requests, apply simple retry logic
            if not self.use_resilience and retries < self.max_retries:
                await asyncio.sleep(self.retry_delay)
                return await self.fetch_url(url, retries + 1)

            return None

    def extract_keywords(self, text: str, keywords_list: List[str]) -> List[Tuple[str, str]]:
        """Extract keywords from text and return (keyword, context) tuples."""
        found_keywords = []
        text_lower = text.lower()

        for keyword in keywords_list:
            keyword_lower = keyword.lower()
            if keyword_lower in text_lower:
                # Find context around keyword (50 characters before and after)
                import re

                pattern = re.compile(f".{{0,50}}{re.escape(keyword_lower)}.{{0,50}}", re.IGNORECASE)
                matches = pattern.findall(text)
                for match in matches[:3]:  # Max 3 contexts per keyword
                    found_keywords.append((keyword, match.strip()))

        return found_keywords

    def parse_date(self, date_string: str, date_format: Optional[str] = None) -> str:
        """Parse a date string to ISO format."""
        if not date_string:
            return datetime.now().isoformat()

        try:
            if date_format:
                # Try the specified format first
                dt = datetime.strptime(date_string, date_format)
            else:
                # Use dateutil parser for flexible parsing
                dt = date_parser.parse(date_string)
            return dt.isoformat()
        except Exception as e:
            self.logger.warning(f"Could not parse date '{date_string}': {e}")
            return date_string  # Return original string as fallback

    async def parse_rss_feed(self, feed_content: str, site_config: Dict) -> int:
        """Parse RSS feed and add articles to database."""
        feed = feedparser.parse(feed_content)
        new_articles_count = 0
        source_name = site_config["name"]
        category = site_config.get("category", "general")

        # Keywords for monitoring
        keywords_config = self.config.get("keywords_monitoring", {})
        monitor_keywords = keywords_config.get("enabled", False)
        keywords_list = keywords_config.get("keywords", [])

        for entry in feed.entries:
            title = entry.get("title", "No title")
            link = entry.get("link", None)

            if not link:
                continue

            # Check if article exists (async or sync)
            if self.use_async_db:
                if await self.async_db.article_exists(link):
                    continue
            else:
                if db_manager.article_exists(link):
                    continue

            # Parse publication date
            pub_date_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub_date_parsed:
                publication_date = datetime(*pub_date_parsed[:6]).isoformat()
            else:
                publication_date = self.parse_date(entry.get("published") or entry.get("updated", ""))

            # Get content
            content_summary = entry.get("summary", "")
            full_content = entry.get("content", [{}])[0].get("value", "") if "content" in entry else ""
            author = entry.get("author", "")

            # Search for keywords
            keywords_found = []
            if monitor_keywords:
                combined_text = f"{title} {content_summary} {full_content}"
                keyword_matches = self.extract_keywords(combined_text, keywords_list)
                keywords_found = [kw[0] for kw in keyword_matches]

            # Add article (async or sync)
            if self.use_async_db:
                article_id = await self.async_db.add_article(
                    url=link,
                    title=title,
                    publication_date=publication_date,
                    source_website=source_name,
                    category=category,
                    content_summary=content_summary[:1000],  # Limit summary
                    full_content=full_content[:5000],  # Limit content
                    author=author,
                    keywords_found=keywords_found,
                )
            else:
                article_id = db_manager.add_article(
                    url=link,
                    title=title,
                    publication_date=publication_date,
                    source_website=source_name,
                    category=category,
                    content_summary=content_summary[:1000],  # Limit summary
                    full_content=full_content[:5000],  # Limit content
                    author=author,
                    keywords_found=keywords_found,
                )

            if article_id:
                new_articles_count += 1

                # Log keyword alerts (async or sync)
                if monitor_keywords and keywords_found:
                    for kw, context in keyword_matches:
                        if self.use_async_db:
                            await self.async_db.add_keyword_alert(article_id, kw, context)
                        else:
                            db_manager.add_keyword_alert(article_id, kw, context)

        return new_articles_count

    async def parse_html_site(self, site_config: Dict, run_id: int) -> int:
        """Parse HTML site with async pagination support."""
        base_url = site_config["url"]
        new_articles_count = 0
        source_name = site_config["name"]
        category = site_config.get("category", "general")

        # Keywords for monitoring
        keywords_config = self.config.get("keywords_monitoring", {})
        monitor_keywords = keywords_config.get("enabled", False)
        keywords_list = keywords_config.get("keywords", [])

        # Pagination settings
        pagination = site_config.get("pagination", {})
        max_pages = pagination.get("max_pages", 1) if pagination.get("enabled", False) else 1

        current_url = base_url
        pages_processed = 0

        while current_url and pages_processed < max_pages:
            self.logger.info(f"Processing page {pages_processed + 1} of {source_name}: {current_url}")

            # Check robots.txt
            if not await self.check_robots_txt(base_url, urlparse(current_url).path):
                self.logger.warning(f"Robots.txt forbids access to {current_url}")
                break

            overview_html = await self.fetch_url(current_url)
            if not overview_html:
                # Log error (async or sync)
                if self.use_async_db:
                    await self.async_db.log_scrape_error(
                        run_id, source_name, "FETCH_ERROR", f"Could not fetch overview page", current_url
                    )
                else:
                    db_manager.log_scrape_error(
                        run_id, source_name, "FETCH_ERROR", f"Could not fetch overview page", current_url
                    )
                break

            soup_overview = BeautifulSoup(overview_html, "lxml")

            # Find article links
            selectors = site_config.get("selectors", {})
            article_links = soup_overview.select(selectors.get("overview_article_link", ""))

            self.logger.info(f"Found: {len(article_links)} article links on page {pages_processed + 1}")

            # Process articles concurrently
            article_tasks = []
            for link_element in article_links:
                article_url = link_element.get("href")
                if not article_url:
                    continue

                # Make absolute URL
                article_url = urljoin(base_url, article_url)

                # Check if article exists (async or sync)
                if self.use_async_db:
                    if await self.async_db.article_exists(article_url):
                        continue
                else:
                    if db_manager.article_exists(article_url):
                        continue

                # Create task for processing this article
                task = self.process_article(article_url, site_config, run_id, monitor_keywords, keywords_list)
                article_tasks.append(task)

            # Process all articles on this page concurrently
            if article_tasks:
                results = await asyncio.gather(*article_tasks, return_exceptions=True)

                # Count successful articles
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Article processing failed: {result}")
                    elif result:
                        new_articles_count += 1

            # Find next page
            pages_processed += 1
            if pagination.get("enabled", False) and pages_processed < max_pages:
                next_page_elem = soup_overview.select_one(pagination.get("next_page_selector", ""))
                if next_page_elem:
                    next_url = next_page_elem.get("href")
                    if next_url:
                        current_url = urljoin(base_url, next_url)
                        self.logger.info(f"Going to next page: {current_url}")
                    else:
                        current_url = None
                else:
                    current_url = None
            else:
                current_url = None

        return new_articles_count

    async def process_article(
        self, article_url: str, site_config: Dict, run_id: int, monitor_keywords: bool, keywords_list: List[str]
    ) -> Optional[int]:
        """Process a single article asynchronously."""
        source_name = site_config["name"]
        category = site_config.get("category", "general")

        # Fetch article details
        self.logger.debug(f"Fetching article details: {article_url}")
        detail_html = await self.fetch_url(article_url)

        if not detail_html:
            # Log error (async or sync)
            if self.use_async_db:
                await self.async_db.log_scrape_error(
                    run_id, source_name, "FETCH_ERROR", f"Could not fetch article", article_url
                )
            else:
                db_manager.log_scrape_error(run_id, source_name, "FETCH_ERROR", f"Could not fetch article", article_url)
            return None

        soup_detail = BeautifulSoup(detail_html, "lxml")
        selectors = site_config.get("selectors", {})

        # Extract article information
        try:
            title_elem = soup_detail.select_one(selectors.get("detail_title", ""))
            title = title_elem.get_text(strip=True) if title_elem else "No title"
        except Exception as e:
            self.logger.error(f"Error extracting title: {e}")
            title = "No title"

        # Date
        publication_date = datetime.now().isoformat()
        try:
            date_elem = soup_detail.select_one(selectors.get("detail_date", ""))
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                publication_date = self.parse_date(date_text, site_config.get("date_format"))
        except Exception as e:
            self.logger.warning(f"Error extracting date: {e}")

        # Content
        content_summary = ""
        full_content = ""
        try:
            content_elem = soup_detail.select_one(selectors.get("detail_content", ""))
            if content_elem:
                full_content = content_elem.get_text(strip=True, separator="\n")
                content_summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
        except Exception as e:
            self.logger.warning(f"Error extracting content: {e}")

        # Author
        author = ""
        try:
            author_elem = soup_detail.select_one(selectors.get("detail_author", ""))
            if author_elem:
                author = author_elem.get_text(strip=True)
        except Exception:
            pass

        # Keywords monitoring
        keywords_found = []
        keyword_matches = []
        if monitor_keywords:
            combined_text = f"{title} {full_content}"
            keyword_matches = self.extract_keywords(combined_text, keywords_list)
            keywords_found = [kw[0] for kw in keyword_matches]

        # Add article (async or sync)
        if self.use_async_db:
            article_id = await self.async_db.add_article(
                url=article_url,
                title=title,
                publication_date=publication_date,
                source_website=source_name,
                category=category,
                content_summary=content_summary,
                full_content=full_content[:5000],
                author=author,
                keywords_found=keywords_found,
            )
        else:
            article_id = db_manager.add_article(
                url=article_url,
                title=title,
                publication_date=publication_date,
                source_website=source_name,
                category=category,
                content_summary=content_summary,
                full_content=full_content[:5000],
                author=author,
                keywords_found=keywords_found,
            )

        if article_id:
            # Record article processing for monitoring
            self.monitor.record_article_processed()

            # Publish article processed event
            if self.events_enabled:
                try:
                    await self.event_publisher.async_publish(
                        self.event_publisher.create_event(
                            EventType.ARTICLE_PROCESSED,
                            {
                                "article_id": article_id,
                                "url": article_url,
                                "title": title,
                                "source": source_name,
                                "keywords_found": keywords_found,
                                "processed_at": datetime.now().isoformat(),
                            },
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to publish article processed event: {e}")

            # Log keyword alerts and publish keyword alert events
            if keyword_matches:
                for kw, context in keyword_matches:
                    # Add keyword alert (async or sync)
                    if self.use_async_db:
                        await self.async_db.add_keyword_alert(article_id, kw, context)
                    else:
                        db_manager.add_keyword_alert(article_id, kw, context)

                    # Publish keyword alert event
                    if self.events_enabled:
                        try:
                            await self.event_publisher.async_publish(
                                self.event_publisher.create_event(
                                    EventType.KEYWORD_ALERT,
                                    {
                                        "keyword": kw,
                                        "article_id": article_id,
                                        "article_title": title,
                                        "article_url": article_url,
                                        "context": context,
                                        "source": source_name,
                                        "alert_time": datetime.now().isoformat(),
                                    },
                                )
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to publish keyword alert event: {e}")

            return article_id

        return None

    async def run_scraper(self):
        """Main async scraper function."""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Async scraper started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize database (if not using async)
        if not self.use_async_db:
            db_manager.init_db()

        # Start new scrape run (async or sync)
        if self.use_async_db:
            run_id = await self.async_db.start_scrape_run()
        else:
            run_id = db_manager.start_scrape_run()

        # Count enabled sites
        enabled_sites = [site for site in self.sites_config if site.get("enabled", False)]

        # Publish scrape started event
        if self.events_enabled:
            try:
                await self.event_publisher.async_publish(
                    self.event_publisher.create_event(
                        EventType.SCRAPE_STARTED,
                        {"run_id": run_id, "sites_count": len(enabled_sites), "started_at": datetime.now().isoformat()},
                        correlation_id=str(run_id),
                    )
                )
            except Exception as e:
                self.logger.warning(f"Failed to publish scrape started event: {e}")

        # Statistics
        total_new_articles = 0
        total_errors = 0

        # Process each site
        site_tasks = []
        for site_config in self.sites_config:
            if not site_config.get("enabled", False):
                self.logger.info(f"Site '{site_config['name']}' is disabled, skipping")
                continue

            # Create task for each site
            task = self.process_site(site_config, run_id)
            site_tasks.append(task)

        # Process all sites concurrently
        if site_tasks:
            results = await asyncio.gather(*site_tasks, return_exceptions=True)

            # Collect results
            for i, result in enumerate(results):
                site_config = [s for s in self.sites_config if s.get("enabled", False)][i]
                if isinstance(result, Exception):
                    self.logger.error(f"Site processing failed for {site_config['name']}: {result}")
                    # Log error (async or sync)
                    if self.use_async_db:
                        await self.async_db.log_scrape_error(
                            run_id, site_config["name"], type(result).__name__, str(result), site_config["url"]
                        )
                    else:
                        db_manager.log_scrape_error(
                            run_id, site_config["name"], type(result).__name__, str(result), site_config["url"]
                        )
                    total_errors += 1
                else:
                    total_new_articles += result
                    self.logger.info(f"✓ {site_config['name']}: {result} new articles")

        # Update scrape run statistics (async or sync)
        if self.use_async_db:
            await self.async_db.end_scrape_run(run_id, 0, total_new_articles, total_errors)
        else:
            db_manager.end_scrape_run(run_id, 0, total_new_articles, total_errors)

        # Publish scrape completed event
        if self.events_enabled:
            try:
                scrape_duration = time.time() - (run_id / 1000.0)  # Approximate duration
                await self.event_publisher.async_publish(
                    self.event_publisher.create_event(
                        EventType.SCRAPE_COMPLETED,
                        {
                            "run_id": run_id,
                            "new_articles": total_new_articles,
                            "errors": total_errors,
                            "duration": scrape_duration,
                            "completed_at": datetime.now().isoformat(),
                        },
                        correlation_id=str(run_id),
                    )
                )
            except Exception as e:
                self.logger.warning(f"Failed to publish scrape completed event: {e}")

        # Log summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Async scraper run completed")
        self.logger.info(f"New articles: {total_new_articles}")
        self.logger.info(f"Errors: {total_errors}")
        self.logger.info(f"{'='*50}\n")

        return total_new_articles, total_errors

    async def process_site(self, site_config: Dict, run_id: int) -> int:
        """Process a single site asynchronously."""
        self.logger.info(f"\nProcessing: {site_config['name']} ({site_config['type']})")

        try:
            site_new_articles = 0

            if site_config["type"] == "rss":
                # Use shorter cache TTL for RSS feeds
                feed_content = await self.fetch_url(site_config["url"])
                if feed_content:
                    # Cache RSS content separately with shorter TTL
                    if self.cache_enabled:
                        await self.cache.async_set("rss_feed", site_config["url"], feed_content, ttl=1800)  # 30 minutes
                    site_new_articles = await self.parse_rss_feed(feed_content, site_config)
                else:
                    raise Exception("Could not fetch RSS feed")

            elif site_config["type"] == "html":
                site_new_articles = await self.parse_html_site(site_config, run_id)

            else:
                self.logger.warning(f"Unknown site type: {site_config['type']}")
                return 0

            return site_new_articles

        except Exception as e:
            self.logger.error(f"Error processing {site_config['name']}: {e}")
            raise e


async def main():
    """Main async function."""
    # Load configuration
    CONFIG_FILE = "config.json"
    SITES_FILE = "sites_to_scrape.json"

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(SITES_FILE, "r", encoding="utf-8") as f:
            sites_to_scrape = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Make sure these files exist.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    # Set database name
    db_manager.DATABASE_NAME = config.get("database_name", "mimir_news.db")

    # Run async scraper
    async with AsyncNewsScraper(config, sites_to_scrape) as scraper:
        await scraper.run_scraper()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mimir Async News Scraper")
    parser.add_argument("--run", action="store_true", help="Run async scraper")

    args = parser.parse_args()

    if args.run:
        asyncio.run(main())
    else:
        print("Use --run to start the async scraper")
        print("Example: python async_scraper.py --run")
