"""Keyword alert model for monitoring specific terms."""

from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Index, String, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class KeywordAlert(Base, TimestampMixin):
    """Model for keyword-based alerts."""
    
    __tablename__ = "keyword_alerts"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        doc="Unique identifier for the alert"
    )
    
    # Alert details
    keyword: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="The keyword that triggered this alert"
    )
    context: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Text context around the keyword match"
    )
    match_position: Mapped[Optional[int]] = mapped_column(
        doc="Position of the match in the text"
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(
        doc="Confidence score of the match (0.0-1.0)"
    )
    
    # Relationships
    article_id: Mapped[UUID] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to the article that triggered this alert"
    )
    article: Mapped["Article"] = relationship(
        "Article",
        back_populates="keyword_alerts",
        doc="Article that triggered this alert"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_keyword_alerts_keyword", "keyword"),
        Index("ix_keyword_alerts_article_id", "article_id"),
        Index("ix_keyword_alerts_created_at", "created_at"),
        Index("ix_keyword_alerts_confidence", "confidence_score"),
    )
    
    def __repr__(self) -> str:
        return f"<KeywordAlert(id={self.id}, keyword='{self.keyword}', article_id={self.article_id})>"