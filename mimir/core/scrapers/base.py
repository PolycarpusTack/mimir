"""Base scraper interface and result classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ScrapedArticle:
    """Represents a scraped article before database storage."""
    
    url: str
    title: str
    content_summary: Optional[str] = None
    full_content: Optional[str] = None
    publication_date: Optional[datetime] = None
    author: Optional[str] = None
    language: str = "en"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ScrapingResult:
    """Result of a scraping operation."""
    
    articles: List[ScrapedArticle]
    total_found: int
    errors: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_found == 0:
            return 1.0
        return len(self.articles) / self.total_found
    
    @property
    def has_errors(self) -> bool:
        """Check if there were any errors."""
        return len(self.errors) > 0


class BaseScraper(ABC):
    """Abstract base class for all scrapers."""
    
    def __init__(
        self,
        source_id: UUID,
        config: Dict[str, Any],
        request_timeout: int = 30,
        max_retries: int = 3,
        request_delay: float = 2.0,
    ) -> None:
        """
        Initialize scraper.
        
        Args:
            source_id: UUID of the source being scraped
            config: Scraper configuration
            request_timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts
            request_delay: Delay between requests in seconds
        """
        self.source_id = source_id
        self.config = config
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.logger = logger.bind(
            scraper=self.__class__.__name__,
            source_id=str(source_id)
        )
    
    @abstractmethod
    async def scrape(self, max_articles: int = 20) -> ScrapingResult:
        """
        Scrape articles from the source.
        
        Args:
            max_articles: Maximum number of articles to scrape
            
        Returns:
            ScrapingResult containing scraped articles and metadata
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate scraper configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    async def test_connection(self) -> bool:
        """
        Test connection to the source.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = await self.scrape(max_articles=1)
            return not result.has_errors
        except Exception as e:
            self.logger.error("Connection test failed", error=str(e))
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common HTML entities if they slipped through
        html_entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&#39;": "'",
            "&nbsp;": " ",
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text.strip()
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            from dateutil import parser as date_parser
            return date_parser.parse(date_str)
        except (ValueError, TypeError) as e:
            self.logger.warning(
                "Failed to parse date",
                date_string=date_str,
                error=str(e)
            )
            return None
    
    def _calculate_relevance_score(self, article: ScrapedArticle, keywords: List[str]) -> float:
        """
        Calculate relevance score for broadcast industry.
        
        Args:
            article: Scraped article
            keywords: List of relevant keywords
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not keywords:
            return 0.5  # Neutral score if no keywords provided
        
        text = f"{article.title} {article.content_summary or ''} {article.full_content or ''}".lower()
        
        # Count keyword matches
        matches = 0
        for keyword in keywords:
            if keyword.lower() in text:
                matches += 1
        
        # Simple relevance calculation
        return min(1.0, matches / len(keywords) * 2)  # Scale up to give higher scores