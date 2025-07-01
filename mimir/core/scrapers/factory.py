"""Factory for creating scrapers based on source type."""

from typing import Dict, Type
from uuid import UUID

from ..models.source import SourceType
from .base import BaseScraper
from .html_scraper import HTMLScraper
from .rss_scraper import RSScraper


class ScraperFactory:
    """Factory for creating appropriate scrapers."""
    
    _scrapers: Dict[SourceType, Type[BaseScraper]] = {
        SourceType.RSS: RSScraper,
        SourceType.HTML: HTMLScraper,
    }
    
    @classmethod
    def create_scraper(
        self,
        source_type: SourceType,
        source_id: UUID,
        config: Dict,
        **kwargs
    ) -> BaseScraper:
        """
        Create a scraper instance for the given source type.
        
        Args:
            source_type: Type of source to scrape
            source_id: UUID of the source
            config: Scraper configuration
            **kwargs: Additional arguments for scraper initialization
            
        Returns:
            Configured scraper instance
            
        Raises:
            ValueError: If source type is not supported
        """
        if source_type not in self._scrapers:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        scraper_class = self._scrapers[source_type]
        return scraper_class(source_id=source_id, config=config, **kwargs)
    
    @classmethod
    def get_supported_types(cls) -> list[SourceType]:
        """Get list of supported source types."""
        return list(cls._scrapers.keys())
    
    @classmethod
    def register_scraper(cls, source_type: SourceType, scraper_class: Type[BaseScraper]) -> None:
        """
        Register a new scraper type.
        
        Args:
            source_type: Source type to register
            scraper_class: Scraper class to use for this type
        """
        cls._scrapers[source_type] = scraper_class