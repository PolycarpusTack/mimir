"""Service for managing news sources."""

from typing import Dict, List, Optional
from uuid import UUID

import logging
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Source, Article
from ..models.source import SourceType, SourceCategory, SourceStatus

logger = logging.getLogger(__name__)


class SourceService:
    """Service for managing news sources."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize source service."""
        self.session = session
    
    async def get_by_id(self, source_id: UUID) -> Optional[Source]:
        """Get source by ID."""
        return await self.session.get(
            Source,
            source_id,
            options=[selectinload(Source.articles)]
        )
    
    async def get_by_name(self, name: str) -> Optional[Source]:
        """Get source by name."""
        result = await self.session.execute(
            select(Source).where(Source.name == name)
        )
        return result.scalar_one_or_none()
    
    async def get_all_sources(
        self,
        enabled_only: bool = False,
        category: Optional[SourceCategory] = None
    ) -> List[Source]:
        """
        Get all sources with optional filtering.
        
        Args:
            enabled_only: Only return enabled sources
            category: Filter by source category
            
        Returns:
            List of sources
        """
        query = select(Source)
        
        conditions = []
        if enabled_only:
            conditions.append(Source.is_enabled == True)
        
        if category:
            conditions.append(Source.category == category)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Source.name)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def create_source(
        self,
        name: str,
        url: str,
        source_type: SourceType,
        category: SourceCategory = SourceCategory.GENERAL,
        description: Optional[str] = None,
        rss_url: Optional[str] = None,
        scraping_config: Optional[Dict] = None,
        **kwargs
    ) -> Source:
        """
        Create a new source.
        
        Args:
            name: Human-readable name
            url: Base URL of the source
            source_type: Type of source (RSS, HTML, API)
            category: Source category
            description: Optional description
            rss_url: RSS feed URL (for RSS sources)
            scraping_config: Configuration for scraping
            **kwargs: Additional source attributes
            
        Returns:
            Created source
            
        Raises:
            ValueError: If source with same name already exists
        """
        # Check if source with same name exists
        existing = await self.get_by_name(name)
        if existing:
            raise ValueError(f"Source with name '{name}' already exists")
        
        source = Source(
            name=name,
            url=url,
            source_type=source_type,
            category=category,
            description=description,
            rss_url=rss_url,
            scraping_config=scraping_config or {},
            **kwargs
        )
        
        self.session.add(source)
        await self.session.commit()
        
        logger.info(
            "Source created",
            source_id=str(source.id),
            name=name,
            type=source_type,
            category=category
        )
        
        return source
    
    async def update_source(
        self,
        source_id: UUID,
        **updates
    ) -> Optional[Source]:
        """
        Update a source.
        
        Args:
            source_id: ID of source to update
            **updates: Fields to update
            
        Returns:
            Updated source or None if not found
        """
        source = await self.get_by_id(source_id)
        if not source:
            return None
        
        for field, value in updates.items():
            if hasattr(source, field):
                setattr(source, field, value)
        
        await self.session.commit()
        
        logger.info(
            "Source updated",
            source_id=str(source_id),
            updates=list(updates.keys())
        )
        
        return source
    
    async def delete_source(self, source_id: UUID) -> bool:
        """
        Delete a source.
        
        Args:
            source_id: ID of source to delete
            
        Returns:
            True if deleted, False if not found
        """
        source = await self.get_by_id(source_id)
        if not source:
            return False
        
        source_name = source.name
        await self.session.delete(source)
        await self.session.commit()
        
        logger.info(
            "Source deleted",
            source_id=str(source_id),
            name=source_name
        )
        
        return True
    
    async def enable_source(self, source_id: UUID) -> bool:
        """Enable a source."""
        source = await self.get_by_id(source_id)
        if not source:
            return False
        
        source.is_enabled = True
        source.status = SourceStatus.ACTIVE
        await self.session.commit()
        
        logger.info("Source enabled", source_id=str(source_id), name=source.name)
        return True
    
    async def disable_source(self, source_id: UUID) -> bool:
        """Disable a source."""
        source = await self.get_by_id(source_id)
        if not source:
            return False
        
        source.is_enabled = False
        source.status = SourceStatus.INACTIVE
        await self.session.commit()
        
        logger.info("Source disabled", source_id=str(source_id), name=source.name)
        return True
    
    async def get_source_statistics(self, source_id: UUID) -> Optional[Dict]:
        """
        Get statistics for a specific source.
        
        Args:
            source_id: ID of the source
            
        Returns:
            Dictionary with source statistics or None if not found
        """
        source = await self.get_by_id(source_id)
        if not source:
            return None
        
        # Count articles
        article_count_result = await self.session.execute(
            select(func.count(Article.id))
            .where(Article.source_id == source_id)
        )
        total_articles = article_count_result.scalar()
        
        # Count broadcast relevant articles
        relevant_count_result = await self.session.execute(
            select(func.count(Article.id))
            .where(
                and_(
                    Article.source_id == source_id,
                    Article.is_broadcast_relevant == True
                )
            )
        )
        relevant_articles = relevant_count_result.scalar()
        
        # Average relevance score
        avg_relevance_result = await self.session.execute(
            select(func.avg(Article.relevance_score))
            .where(Article.source_id == source_id)
        )
        avg_relevance = avg_relevance_result.scalar() or 0.0
        
        # Most recent article
        recent_article_result = await self.session.execute(
            select(Article.publication_date)
            .where(Article.source_id == source_id)
            .order_by(desc(Article.publication_date))
            .limit(1)
        )
        last_article_date = recent_article_result.scalar()
        
        return {
            "source_id": str(source_id),
            "name": source.name,
            "type": source.source_type,
            "category": source.category,
            "status": source.status,
            "is_enabled": source.is_enabled,
            "health_score": source.health_score,
            "total_articles": total_articles,
            "relevant_articles": relevant_articles,
            "relevance_percentage": (relevant_articles / total_articles * 100) if total_articles > 0 else 0,
            "average_relevance_score": float(avg_relevance),
            "last_successful_scrape": source.last_successful_scrape,
            "last_article_date": last_article_date,
            "consecutive_errors": source.consecutive_errors,
            "last_error": source.last_error,
        }
    
    async def get_sources_health_report(self) -> Dict:
        """
        Get health report for all sources.
        
        Returns:
            Dictionary with health statistics
        """
        # Count by status
        status_counts = {}
        for status in SourceStatus:
            count_result = await self.session.execute(
                select(func.count(Source.id))
                .where(Source.status == status)
            )
            status_counts[status.value] = count_result.scalar()
        
        # Count enabled/disabled
        enabled_result = await self.session.execute(
            select(func.count(Source.id))
            .where(Source.is_enabled == True)
        )
        enabled_count = enabled_result.scalar()
        
        total_result = await self.session.execute(select(func.count(Source.id)))
        total_count = total_result.scalar()
        
        # Sources with errors
        error_sources_result = await self.session.execute(
            select(Source.name, Source.consecutive_errors, Source.last_error)
            .where(Source.consecutive_errors > 0)
            .order_by(desc(Source.consecutive_errors))
        )
        error_sources = [
            {
                "name": row.name,
                "consecutive_errors": row.consecutive_errors,
                "last_error": row.last_error
            }
            for row in error_sources_result
        ]
        
        # Top performing sources
        top_sources_result = await self.session.execute(
            select(
                Source.name,
                Source.total_articles_scraped,
                Source.average_relevance_score
            )
            .where(Source.is_enabled == True)
            .order_by(desc(Source.total_articles_scraped))
            .limit(5)
        )
        top_sources = [
            {
                "name": row.name,
                "articles_scraped": row.total_articles_scraped,
                "avg_relevance": float(row.average_relevance_score or 0)
            }
            for row in top_sources_result
        ]
        
        return {
            "total_sources": total_count,
            "enabled_sources": enabled_count,
            "status_counts": status_counts,
            "sources_with_errors": error_sources,
            "top_performing_sources": top_sources,
            "overall_health": (enabled_count / total_count * 100) if total_count > 0 else 0
        }
    
    async def test_source_connection(self, source_id: UUID) -> Dict:
        """
        Test connection to a source.
        
        Args:
            source_id: ID of source to test
            
        Returns:
            Dictionary with test results
        """
        source = await self.get_by_id(source_id)
        if not source:
            return {"success": False, "error": "Source not found"}
        
        try:
            from ..scrapers import ScraperFactory
            
            # Create scraper
            scraper_config = source.scraping_config or {}
            scraper_config.update({
                "url": source.rss_url or source.url,
            })
            
            scraper = ScraperFactory.create_scraper(
                source_type=source.source_type,
                source_id=source.id,
                config=scraper_config,
                request_timeout=10,  # Short timeout for testing
                max_retries=1,
            )
            
            # Test connection
            success = await scraper.test_connection()
            
            if success:
                return {
                    "success": True,
                    "message": "Connection successful",
                    "source_name": source.name
                }
            else:
                return {
                    "success": False,
                    "error": "Connection test failed",
                    "source_name": source.name
                }
                
        except Exception as e:
            logger.error(
                "Source connection test failed",
                source_id=str(source_id),
                error=str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "source_name": source.name
            }