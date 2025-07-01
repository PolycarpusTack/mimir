"""Service for orchestrating news scraping operations."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select

from ..models import Source, ScrapeRun, Article, KeywordAlert
from ..models.source import SourceStatus
from ..scrapers import ScraperFactory
from .article_service import ArticleService
from mimir.config import get_settings

logger = structlog.get_logger(__name__)


class ScrapingService:
    """Service for managing news scraping operations."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize scraping service."""
        self.session = session
        self.settings = get_settings()
        self.article_service = ArticleService(session)
    
    async def scrape_all_sources(
        self,
        enabled_only: bool = True,
        max_articles_per_source: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Scrape all configured sources.
        
        Args:
            enabled_only: Only scrape enabled sources
            max_articles_per_source: Override default article limit per source
            
        Returns:
            Dictionary with scraping results and statistics
        """
        # Create overall scrape run
        scrape_run = ScrapeRun()
        scrape_run.start()
        self.session.add(scrape_run)
        await self.session.commit()
        
        logger.info(
            "Starting bulk scraping operation",
            scrape_run_id=str(scrape_run.id),
            enabled_only=enabled_only
        )
        
        try:
            # Get sources to scrape
            sources = await self._get_sources_to_scrape(enabled_only)
            
            if not sources:
                logger.warning("No sources found to scrape")
                scrape_run.complete()
                await self.session.commit()
                return {
                    "scrape_run_id": str(scrape_run.id),
                    "sources_processed": 0,
                    "total_articles": 0,
                    "new_articles": 0,
                    "errors": ["No sources found to scrape"]
                }
            
            # Determine article limit
            article_limit = (
                max_articles_per_source or 
                self.settings.scraping.articles_per_source
            )
            
            # Process sources
            results = await self._process_sources(sources, article_limit, scrape_run)
            
            # Update scrape run with results
            scrape_run.total_articles_found = results["total_articles"]
            scrape_run.new_articles_count = results["new_articles"]
            scrape_run.error_count = len(results["errors"])
            scrape_run.complete()
            
            await self.session.commit()
            
            logger.info(
                "Bulk scraping operation completed",
                scrape_run_id=str(scrape_run.id),
                sources_processed=results["sources_processed"],
                total_articles=results["total_articles"],
                new_articles=results["new_articles"],
                errors=len(results["errors"])
            )
            
            return {
                "scrape_run_id": str(scrape_run.id),
                **results
            }
            
        except Exception as e:
            logger.error(
                "Bulk scraping operation failed",
                scrape_run_id=str(scrape_run.id),
                error=str(e)
            )
            scrape_run.fail(str(e))
            await self.session.commit()
            raise
    
    async def scrape_single_source(
        self,
        source_id: UUID,
        max_articles: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Scrape a single source.
        
        Args:
            source_id: ID of source to scrape
            max_articles: Maximum articles to scrape
            
        Returns:
            Dictionary with scraping results
        """
        # Get source
        source = await self.session.get(Source, source_id)
        if not source:
            raise ValueError(f"Source {source_id} not found")
        
        if not source.is_enabled:
            raise ValueError(f"Source {source.name} is disabled")
        
        # Create source-specific scrape run
        scrape_run = ScrapeRun(source_id=source_id)
        scrape_run.start()
        self.session.add(scrape_run)
        await self.session.commit()
        
        logger.info(
            "Starting single source scraping",
            source_id=str(source_id),
            source_name=source.name,
            scrape_run_id=str(scrape_run.id)
        )
        
        try:
            # Determine article limit
            article_limit = max_articles or source.max_articles_per_run
            
            # Scrape source
            result = await self._scrape_source(source, article_limit)
            
            # Update scrape run
            scrape_run.total_articles_found = result["total_found"]
            scrape_run.new_articles_count = result["new_articles"]
            scrape_run.error_count = len(result["errors"])
            scrape_run.complete()
            
            # Update source statistics
            if result["errors"]:
                source.mark_error("; ".join(result["errors"]))
            else:
                source.mark_successful_scrape(result["new_articles"])
            
            await self.session.commit()
            
            logger.info(
                "Single source scraping completed",
                source_id=str(source_id),
                scrape_run_id=str(scrape_run.id),
                total_found=result["total_found"],
                new_articles=result["new_articles"],
                errors=len(result["errors"])
            )
            
            return {
                "scrape_run_id": str(scrape_run.id),
                "source_id": str(source_id),
                "source_name": source.name,
                **result
            }
            
        except Exception as e:
            logger.error(
                "Single source scraping failed",
                source_id=str(source_id),
                scrape_run_id=str(scrape_run.id),
                error=str(e)
            )
            scrape_run.fail(str(e))
            source.mark_error(str(e))
            await self.session.commit()
            raise
    
    async def _get_sources_to_scrape(self, enabled_only: bool) -> List[Source]:
        """Get list of sources to scrape."""
        query = select(Source)
        
        if enabled_only:
            query = query.where(Source.is_enabled == True)
        
        # Order by last successful scrape (oldest first)
        query = query.order_by(Source.last_successful_scrape.asc().nullsfirst())
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def _process_sources(
        self,
        sources: List[Source],
        article_limit: int,
        scrape_run: ScrapeRun
    ) -> Dict[str, any]:
        """Process multiple sources with proper error handling."""
        total_articles = 0
        new_articles = 0
        errors = []
        sources_processed = 0
        
        for source in sources:
            try:
                logger.info(
                    "Processing source",
                    source_id=str(source.id),
                    source_name=source.name,
                    source_type=source.source_type
                )
                
                result = await self._scrape_source(source, article_limit)
                
                total_articles += result["total_found"]
                new_articles += result["new_articles"]
                errors.extend(result["errors"])
                sources_processed += 1
                
                # Update source statistics
                if result["errors"]:
                    source.mark_error("; ".join(result["errors"]))
                else:
                    source.mark_successful_scrape(result["new_articles"])
                
                # Rate limiting between sources
                if self.settings.scraping.request_delay > 0:
                    await asyncio.sleep(self.settings.scraping.request_delay)
                
            except Exception as e:
                error_msg = f"Source {source.name}: {str(e)}"
                errors.append(error_msg)
                source.mark_error(str(e))
                
                logger.error(
                    "Failed to process source",
                    source_id=str(source.id),
                    source_name=source.name,
                    error=str(e)
                )
        
        return {
            "sources_processed": sources_processed,
            "total_articles": total_articles,
            "new_articles": new_articles,
            "errors": errors
        }
    
    async def _scrape_source(self, source: Source, max_articles: int) -> Dict[str, any]:
        """Scrape a single source and process results."""
        try:
            # Create scraper
            scraper_config = source.scraping_config or {}
            scraper_config.update({
                "url": source.rss_url or source.url,
                "user_agent": self.settings.scraping.user_agent,
                "respect_robots_txt": self.settings.scraping.respect_robots_txt,
            })
            
            scraper = ScraperFactory.create_scraper(
                source_type=source.source_type,
                source_id=source.id,
                config=scraper_config,
                request_timeout=self.settings.scraping.request_timeout,
                max_retries=self.settings.scraping.max_retries,
                request_delay=source.request_delay or self.settings.scraping.request_delay,
            )
            
            # Perform scraping
            scraping_result = await scraper.scrape(max_articles=max_articles)
            
            # Process articles
            new_articles_count = 0
            
            for scraped_article in scraping_result.articles:
                try:
                    # Check if article already exists
                    existing = await self.article_service.get_by_url(scraped_article.url)
                    if existing:
                        continue
                    
                    # Create article
                    article = Article(
                        url=scraped_article.url,
                        title=scraped_article.title,
                        content_summary=scraped_article.content_summary,
                        full_content=scraped_article.full_content,
                        publication_date=scraped_article.publication_date,
                        author=scraped_article.author,
                        language=scraped_article.language,
                        source_id=source.id,
                        category=source.category.value,
                    )
                    
                    # Calculate relevance score
                    relevance_score = scraper._calculate_relevance_score(
                        scraped_article,
                        self.settings.monitoring.keywords
                    )
                    article.relevance_score = relevance_score
                    article.is_broadcast_relevant = relevance_score >= self.settings.monitoring.min_relevance_score
                    
                    self.session.add(article)
                    await self.session.flush()  # Get article ID
                    
                    # Check for keyword alerts
                    await self._process_keyword_alerts(article)
                    
                    new_articles_count += 1
                    
                except Exception as e:
                    logger.error(
                        "Failed to process scraped article",
                        url=scraped_article.url,
                        source_id=str(source.id),
                        error=str(e)
                    )
                    scraping_result.errors.append(f"Article processing error: {str(e)}")
            
            await self.session.commit()
            
            return {
                "total_found": scraping_result.total_found,
                "new_articles": new_articles_count,
                "duplicates": scraping_result.total_found - len(scraping_result.articles),
                "errors": scraping_result.errors,
                "metadata": scraping_result.metadata
            }
            
        except Exception as e:
            await self.session.rollback()
            raise
    
    async def _process_keyword_alerts(self, article: Article) -> None:
        """Process keyword alerts for an article."""
        text = f"{article.title} {article.content_summary or ''} {article.full_content or ''}".lower()
        
        for keyword in self.settings.monitoring.keywords:
            if keyword.lower() in text:
                # Find context around keyword
                keyword_pos = text.find(keyword.lower())
                context_start = max(0, keyword_pos - 50)
                context_end = min(len(text), keyword_pos + len(keyword) + 50)
                context = text[context_start:context_end].strip()
                
                # Create alert
                alert = KeywordAlert(
                    keyword=keyword,
                    context=context,
                    match_position=keyword_pos,
                    confidence_score=1.0,  # Simple exact match
                    article_id=article.id
                )
                
                self.session.add(alert)
                
                logger.info(
                    "Keyword alert created",
                    keyword=keyword,
                    article_id=str(article.id),
                    article_title=article.title[:50]
                )