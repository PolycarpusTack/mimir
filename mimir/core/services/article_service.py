"""Service for managing articles."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import logging
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Article, KeywordAlert, Source

logger = logging.getLogger(__name__)


class ArticleService:
    """Service for managing articles."""
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize article service."""
        self.session = session
    
    async def get_by_id(self, article_id: UUID) -> Optional[Article]:
        """Get article by ID."""
        return await self.session.get(
            Article, 
            article_id,
            options=[selectinload(Article.source), selectinload(Article.keyword_alerts)]
        )
    
    async def get_by_url(self, url: str) -> Optional[Article]:
        """Get article by URL."""
        result = await self.session.execute(
            select(Article)
            .where(Article.url == url)
            .options([selectinload(Article.source)])
        )
        return result.scalar_one_or_none()
    
    async def get_recent_articles(
        self,
        limit: int = 50,
        hours: int = 24,
        broadcast_relevant_only: bool = False
    ) -> List[Article]:
        """
        Get recent articles.
        
        Args:
            limit: Maximum number of articles to return
            hours: How many hours back to look
            broadcast_relevant_only: Only return broadcast-relevant articles
            
        Returns:
            List of recent articles
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = (
            select(Article)
            .where(Article.created_at >= cutoff_time)
            .options([selectinload(Article.source)])
            .order_by(desc(Article.publication_date))
            .limit(limit)
        )
        
        if broadcast_relevant_only:
            query = query.where(Article.is_broadcast_relevant == True)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_articles_by_keywords(
        self,
        keywords: List[str],
        limit: int = 50,
        days: int = 7
    ) -> List[Article]:
        """
        Get articles matching specific keywords.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of articles to return
            days: How many days back to search
            
        Returns:
            List of matching articles
        """
        if not keywords:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Build keyword search conditions
        conditions = []
        for keyword in keywords:
            keyword_like = f"%{keyword.lower()}%"
            conditions.append(
                func.lower(Article.title).like(keyword_like) |
                func.lower(Article.content_summary).like(keyword_like)
            )
        
        query = (
            select(Article)
            .where(
                and_(
                    Article.created_at >= cutoff_time,
                    *conditions
                )
            )
            .options([selectinload(Article.source)])
            .order_by(desc(Article.relevance_score), desc(Article.publication_date))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_featured_article(self) -> Optional[Article]:
        """
        Get a featured article for the hero section.
        
        Returns:
            Most relevant recent article or None
        """
        # Try to get a recent broadcast-relevant article with high relevance
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        query = (
            select(Article)
            .where(
                and_(
                    Article.publication_date >= cutoff_time,
                    Article.is_broadcast_relevant == True,
                    Article.relevance_score >= 0.7
                )
            )
            .options([selectinload(Article.source), selectinload(Article.keyword_alerts)])
            .order_by(desc(Article.relevance_score), desc(Article.publication_date))
            .limit(1)
        )
        
        result = await self.session.execute(query)
        featured = result.scalar_one_or_none()
        
        if featured:
            return featured
        
        # Fallback to most recent article
        query = (
            select(Article)
            .options([selectinload(Article.source)])
            .order_by(desc(Article.publication_date))
            .limit(1)
        )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_articles_by_source(
        self,
        source_id: UUID,
        limit: int = 50,
        days: int = 30
    ) -> List[Article]:
        """
        Get articles from a specific source.
        
        Args:
            source_id: ID of the source
            limit: Maximum number of articles to return
            days: How many days back to search
            
        Returns:
            List of articles from the source
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        query = (
            select(Article)
            .where(
                and_(
                    Article.source_id == source_id,
                    Article.created_at >= cutoff_time
                )
            )
            .options([selectinload(Article.source)])
            .order_by(desc(Article.publication_date))
            .limit(limit)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def search_articles(
        self,
        query: str,
        limit: int = 50,
        days: int = 30
    ) -> List[Article]:
        """
        Search articles by text query.
        
        Args:
            query: Search query
            limit: Maximum number of articles to return
            days: How many days back to search
            
        Returns:
            List of matching articles
        """
        if not query.strip():
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        search_term = f"%{query.lower()}%"
        
        sql_query = (
            select(Article)
            .where(
                and_(
                    Article.created_at >= cutoff_time,
                    func.lower(Article.title).like(search_term) |
                    func.lower(Article.content_summary).like(search_term) |
                    func.lower(Article.author).like(search_term)
                )
            )
            .options([selectinload(Article.source)])
            .order_by(desc(Article.relevance_score), desc(Article.publication_date))
            .limit(limit)
        )
        
        result = await self.session.execute(sql_query)
        return result.scalars().all()
    
    async def get_statistics(self) -> Dict[str, any]:
        """
        Get article statistics.
        
        Returns:
            Dictionary with various statistics
        """
        # Total articles
        total_result = await self.session.execute(select(func.count(Article.id)))
        total_articles = total_result.scalar()
        
        # Articles today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_result = await self.session.execute(
            select(func.count(Article.id))
            .where(Article.created_at >= today_start)
        )
        todays_articles = today_result.scalar()
        
        # Broadcast relevant articles
        relevant_result = await self.session.execute(
            select(func.count(Article.id))
            .where(Article.is_broadcast_relevant == True)
        )
        relevant_articles = relevant_result.scalar()
        
        # Articles by source (top 10)
        source_result = await self.session.execute(
            select(Source.name, func.count(Article.id).label('count'))
            .join(Article)
            .group_by(Source.id, Source.name)
            .order_by(desc('count'))
            .limit(10)
        )
        articles_by_source = [
            {"source": row.name, "count": row.count}
            for row in source_result
        ]
        
        # Recent keyword alerts
        recent_alerts_result = await self.session.execute(
            select(func.count(KeywordAlert.id))
            .where(KeywordAlert.created_at >= today_start)
        )
        recent_alerts = recent_alerts_result.scalar()
        
        return {
            "total_articles": total_articles,
            "todays_articles": todays_articles,
            "broadcast_relevant_articles": relevant_articles,
            "articles_by_source": articles_by_source,
            "recent_keyword_alerts": recent_alerts,
            "relevance_percentage": (relevant_articles / total_articles * 100) if total_articles > 0 else 0
        }
    
    async def mark_as_duplicate(self, article_id: UUID) -> bool:
        """
        Mark an article as duplicate.
        
        Args:
            article_id: ID of the article to mark
            
        Returns:
            True if successful, False otherwise
        """
        article = await self.get_by_id(article_id)
        if not article:
            return False
        
        article.is_duplicate = True
        await self.session.commit()
        
        logger.info(
            "Article marked as duplicate",
            article_id=str(article_id),
            title=article.title[:50]
        )
        
        return True
    
    async def update_relevance_score(self, article_id: UUID, score: float) -> bool:
        """
        Update an article's relevance score.
        
        Args:
            article_id: ID of the article
            score: New relevance score (0.0-1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError("Relevance score must be between 0.0 and 1.0")
        
        article = await self.get_by_id(article_id)
        if not article:
            return False
        
        old_score = article.relevance_score
        article.relevance_score = score
        article.is_broadcast_relevant = score >= 0.3  # Configurable threshold
        
        await self.session.commit()
        
        logger.info(
            "Article relevance score updated",
            article_id=str(article_id),
            old_score=old_score,
            new_score=score
        )
        
        return True