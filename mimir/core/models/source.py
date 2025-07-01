"""Source model for managing news sources."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Index,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class SourceType(str, Enum):
    """Types of news sources."""
    RSS = "rss"
    HTML = "html"
    API = "api"


class SourceCategory(str, Enum):
    """Categories of news sources."""
    BROADCASTERS = "broadcasters"
    STANDARDS = "standards"
    TRADE_PUBLICATIONS = "trade_publications"
    TECHNOLOGY_VENDORS = "technology_vendors"
    GENERAL = "general"


class SourceStatus(str, Enum):
    """Status of news sources."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class Source(Base, TimestampMixin):
    """Model for news sources."""
    
    __tablename__ = "sources"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        doc="Unique identifier for the source"
    )
    
    # Basic information
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        unique=True,
        doc="Human-readable name of the source"
    )
    url: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
        doc="Base URL of the source"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Description of the source"
    )
    
    # Source configuration
    source_type: Mapped[SourceType] = mapped_column(
        SQLEnum(SourceType),
        nullable=False,
        doc="Type of source (RSS, HTML, API)"
    )
    category: Mapped[SourceCategory] = mapped_column(
        SQLEnum(SourceCategory),
        default=SourceCategory.GENERAL,
        doc="Category of the source"
    )
    
    # Status and activity
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        doc="Whether this source is currently enabled"
    )
    status: Mapped[SourceStatus] = mapped_column(
        SQLEnum(SourceStatus),
        default=SourceStatus.ACTIVE,
        doc="Current status of the source"
    )
    
    # Scraping configuration
    scraping_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        doc="JSON configuration for scraping (selectors, etc.)"
    )
    
    # RSS specific fields
    rss_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        doc="RSS feed URL (for RSS sources)"
    )
    
    # Rate limiting and politeness
    request_delay: Mapped[float] = mapped_column(
        default=2.0,
        doc="Delay between requests in seconds"
    )
    max_articles_per_run: Mapped[int] = mapped_column(
        Integer,
        default=20,
        doc="Maximum articles to fetch per scraping run"
    )
    
    # Health monitoring
    last_successful_scrape: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        doc="Timestamp of last successful scrape"
    )
    last_error: Mapped[Optional[str]] = mapped_column(
        Text,
        doc="Last error message encountered"
    )
    consecutive_errors: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Number of consecutive errors"
    )
    total_articles_scraped: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Total number of articles scraped from this source"
    )
    
    # Quality metrics
    average_relevance_score: Mapped[Optional[float]] = mapped_column(
        doc="Average relevance score of articles from this source"
    )
    robots_txt_compliant: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        doc="Whether this source respects robots.txt"
    )
    
    # Relationships
    articles: Mapped[List["Article"]] = relationship(
        "Article",
        back_populates="source",
        cascade="all, delete-orphan",
        doc="Articles from this source"
    )
    scrape_runs: Mapped[List["ScrapeRun"]] = relationship(
        "ScrapeRun",
        back_populates="source",
        doc="Scraping runs for this source"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_sources_name", "name"),
        Index("ix_sources_type", "source_type"),
        Index("ix_sources_category", "category"),
        Index("ix_sources_enabled", "is_enabled"),
        Index("ix_sources_status", "status"),
        Index("ix_sources_last_scrape", "last_successful_scrape"),
    )
    
    def __repr__(self) -> str:
        return f"<Source(id={self.id}, name='{self.name}', type={self.source_type}, enabled={self.is_enabled})>"
    
    @property
    def health_score(self) -> float:
        """Calculate source health score (0.0-1.0)."""
        if self.consecutive_errors == 0:
            return 1.0
        elif self.consecutive_errors >= 5:
            return 0.0
        else:
            return 1.0 - (self.consecutive_errors * 0.2)
    
    @property
    def is_healthy(self) -> bool:
        """Check if source is considered healthy."""
        return self.consecutive_errors < 3 and self.status == SourceStatus.ACTIVE
    
    def mark_successful_scrape(self, article_count: int) -> None:
        """Mark a successful scrape and update statistics."""
        self.last_successful_scrape = datetime.utcnow()
        self.consecutive_errors = 0
        self.status = SourceStatus.ACTIVE
        self.total_articles_scraped += article_count
    
    def mark_error(self, error_message: str) -> None:
        """Mark an error and update error tracking."""
        self.last_error = error_message
        self.consecutive_errors += 1
        
        if self.consecutive_errors >= 5:
            self.status = SourceStatus.ERROR
            self.is_enabled = False