"""Scrape run model for tracking scraping sessions."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Index,
    String,
    Text,
    Integer,
    Float,
    DateTime,
    JSON,
    ForeignKey,
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin


class ScrapeRunStatus(str, Enum):
    """Status of scraping runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScrapeRun(Base, TimestampMixin):
    """Model for tracking scraping runs."""
    
    __tablename__ = "scrape_runs"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        doc="Unique identifier for the scrape run"
    )
    
    # Run metadata
    status: Mapped[ScrapeRunStatus] = mapped_column(
        SQLEnum(ScrapeRunStatus),
        default=ScrapeRunStatus.PENDING,
        doc="Current status of the scrape run"
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        doc="When the scrape run started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        doc="When the scrape run completed"
    )
    
    # Source information (optional - for source-specific runs)
    source_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("sources.id", ondelete="SET NULL"),
        doc="Source this run was specific to (if any)"
    )
    
    # Results
    total_articles_found: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Total number of articles found"
    )
    new_articles_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Number of new articles added"
    )
    duplicate_articles_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Number of duplicate articles skipped"
    )
    error_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Number of errors encountered"
    )
    
    # Performance metrics
    duration_seconds: Mapped[Optional[float]] = mapped_column(
        Float,
        doc="Duration of the run in seconds"
    )
    articles_per_second: Mapped[Optional[float]] = mapped_column(
        Float,
        doc="Processing rate (articles per second)"
    )
    
    # Error tracking
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        doc="Detailed error information"
    )
    
    # Configuration used
    run_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        doc="Configuration used for this run"
    )
    
    # Statistics
    keyword_matches: Mapped[int] = mapped_column(
        Integer,
        default=0,
        doc="Number of keyword matches found"
    )
    average_relevance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        doc="Average relevance score of articles in this run"
    )
    
    # Relationships
    source: Mapped[Optional["Source"]] = relationship(
        "Source",
        back_populates="scrape_runs",
        doc="Source this run was specific to"
    )
    
    # Indexes for performance
    __table_args__ = (
        Index("ix_scrape_runs_status", "status"),
        Index("ix_scrape_runs_started_at", "started_at"),
        Index("ix_scrape_runs_completed_at", "completed_at"),
        Index("ix_scrape_runs_source_id", "source_id"),
        Index("ix_scrape_runs_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<ScrapeRun(id={self.id}, status={self.status}, "
            f"new_articles={self.new_articles_count}, source={self.source.name if self.source else 'All'})>"
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (articles processed vs errors)."""
        total_processed = self.total_articles_found + self.error_count
        if total_processed == 0:
            return 1.0
        return self.total_articles_found / total_processed
    
    def start(self) -> None:
        """Mark the run as started."""
        self.status = ScrapeRunStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark the run as completed and calculate metrics."""
        self.status = ScrapeRunStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            self.duration_seconds = duration
            
            if duration > 0:
                self.articles_per_second = self.total_articles_found / duration
    
    def fail(self, error_message: str) -> None:
        """Mark the run as failed with error details."""
        self.status = ScrapeRunStatus.FAILED
        self.completed_at = datetime.utcnow()
        
        if not self.error_details:
            self.error_details = {}
        
        self.error_details["final_error"] = error_message
        
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            self.duration_seconds = duration