"""Article model for storing scraped news articles."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Index,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class Article(Base, TimestampMixin):
    """Model for news articles."""
    
    __tablename__ = "articles"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        doc="Unique identifier for the article"
    )
    
    # Core fields
    url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        unique=True,
        doc="Original URL of the article"
    )
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        doc="Article title"
    )
    content_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Brief summary or excerpt of the article"
    )
    full_content: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Full article content (when available)"
    )
    
    # Metadata
    publication_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        doc="When the article was originally published"
    )
    author: Mapped[Optional[str]] = mapped_column(
        String(200),
        doc="Article author"
    )
    language: Mapped[str] = mapped_column(
        String(10),
        default="en",
        doc="Article language (ISO 639-1 code)"
    )
    
    # Source information
    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to the source that provided this article"
    )
    
    # Classification and relevance
    category: Mapped[str] = mapped_column(
        String(100),
        default="general",
        doc="Article category (technology, business, etc.)"
    )
    relevance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        doc="Computed relevance score for broadcast industry (0.0-1.0)"
    )
    is_broadcast_relevant: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        doc="Whether this article is relevant to broadcast industry"
    )
    
    # Processing status
    is_processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        doc="Whether this article has been processed by AI/NLP"
    )
    is_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        doc="Whether this article is flagged as a duplicate"
    )
    
    # Relationships
    source: Mapped["Source"] = relationship(
        "Source",
        back_populates="articles",
        doc="Source that provided this article"
    )
    keyword_alerts: Mapped[List["KeywordAlert"]] = relationship(
        "KeywordAlert",
        back_populates="article",
        cascade="all, delete-orphan",
        doc="Keyword alerts triggered by this article"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_articles_publication_date", "publication_date"),
        Index("ix_articles_created_at", "created_at"),
        Index("ix_articles_source_id", "source_id"),
        Index("ix_articles_category", "category"),
        Index("ix_articles_relevance", "relevance_score"),
        Index("ix_articles_broadcast_relevant", "is_broadcast_relevant"),
        Index(
            "ix_articles_recent_relevant",
            "publication_date",
            "is_broadcast_relevant",
            postgresql_where="is_broadcast_relevant = true"
        ),
        UniqueConstraint("url", name="uq_articles_url"),
    )
    
    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:50]}...', source={self.source.name if self.source else 'Unknown'})>"
    
    @property
    def word_count(self) -> int:
        """Calculate approximate word count."""
        content = self.full_content or self.content_summary or ""
        return len(content.split())
    
    @property
    def reading_time_minutes(self) -> int:
        """Estimate reading time in minutes (assuming 200 words per minute)."""
        return max(1, self.word_count // 200)