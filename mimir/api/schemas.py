"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, ConfigDict

from ..core.models.source import SourceType, SourceCategory, SourceStatus


# Base schemas
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    page: int
    limit: int
    total: int
    total_pages: int
    has_next: bool
    has_previous: bool


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    data: List[Any]
    meta: PaginationMeta


# Article schemas
class ArticleBase(BaseModel):
    """Base article schema."""
    url: HttpUrl
    title: str = Field(..., min_length=1, max_length=500)
    content_summary: Optional[str] = None
    full_content: Optional[str] = None
    author: Optional[str] = None
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    category: Optional[str] = None


class ArticleCreate(ArticleBase):
    """Schema for creating articles."""
    publication_date: Optional[datetime] = None


class ArticleUpdate(BaseModel):
    """Schema for updating articles."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content_summary: Optional[str] = None
    full_content: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_broadcast_relevant: Optional[bool] = None


class ArticleResponse(ArticleBase):
    """Schema for article responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    publication_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    source_id: UUID
    relevance_score: Optional[float] = None
    is_broadcast_relevant: bool = False
    is_duplicate: bool = False
    
    # Relationship data
    source: Optional["SourceResponse"] = None


class ArticleDetailResponse(ArticleResponse):
    """Detailed article response with related data."""
    keyword_alerts: List["KeywordAlertResponse"] = []


# Source schemas
class SourceBase(BaseModel):
    """Base source schema."""
    name: str = Field(..., min_length=1, max_length=255)
    url: HttpUrl
    description: Optional[str] = None
    source_type: SourceType
    category: SourceCategory = SourceCategory.GENERAL
    is_enabled: bool = True


class SourceCreate(SourceBase):
    """Schema for creating sources."""
    rss_url: Optional[HttpUrl] = None
    scraping_config: Optional[Dict[str, Any]] = None
    max_articles_per_run: int = Field(default=20, ge=1, le=100)
    request_delay: Optional[float] = Field(None, ge=0.0, le=60.0)


class SourceUpdate(BaseModel):
    """Schema for updating sources."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    url: Optional[HttpUrl] = None
    description: Optional[str] = None
    rss_url: Optional[HttpUrl] = None
    is_enabled: Optional[bool] = None
    scraping_config: Optional[Dict[str, Any]] = None
    max_articles_per_run: Optional[int] = Field(None, ge=1, le=100)
    request_delay: Optional[float] = Field(None, ge=0.0, le=60.0)


class SourceResponse(SourceBase):
    """Schema for source responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    created_at: datetime
    updated_at: datetime
    status: SourceStatus
    rss_url: Optional[str] = None
    scraping_config: Dict[str, Any] = {}
    max_articles_per_run: int
    request_delay: Optional[float] = None
    last_successful_scrape: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    total_articles_scraped: int = 0
    average_relevance_score: float = 0.0
    health_score: float = 1.0


class SourceStatistics(BaseModel):
    """Source statistics schema."""
    source_id: str
    name: str
    type: SourceType
    category: SourceCategory
    status: SourceStatus
    is_enabled: bool
    health_score: float
    total_articles: int
    relevant_articles: int
    relevance_percentage: float
    average_relevance_score: float
    last_successful_scrape: Optional[datetime]
    last_article_date: Optional[datetime]
    consecutive_errors: int
    last_error: Optional[str]


class SourceHealthReport(BaseModel):
    """Source health report schema."""
    total_sources: int
    enabled_sources: int
    status_counts: Dict[str, int]
    sources_with_errors: List[Dict[str, Any]]
    top_performing_sources: List[Dict[str, Any]]
    overall_health: float


# Keyword Alert schemas
class KeywordAlertResponse(BaseModel):
    """Schema for keyword alert responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    keyword: str
    context: str
    match_position: int
    confidence_score: float
    created_at: datetime
    article_id: UUID


# Scraping schemas
class ScrapeRunResponse(BaseModel):
    """Schema for scrape run responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    total_articles_found: int = 0
    new_articles_count: int = 0
    error_count: int = 0
    error_message: Optional[str] = None
    source_id: Optional[UUID] = None


class ScrapingResult(BaseModel):
    """Schema for scraping operation results."""
    scrape_run_id: str
    sources_processed: Optional[int] = None
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    total_articles: int
    new_articles: int
    duplicates: Optional[int] = None
    errors: List[str]
    metadata: Optional[Dict[str, Any]] = None


# Health check schemas
class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    version: str
    timestamp: datetime
    database: str = "connected"
    services: Dict[str, str] = {}


# Statistics schemas
class ArticleStatistics(BaseModel):
    """Article statistics schema."""
    total_articles: int
    todays_articles: int
    broadcast_relevant_articles: int
    articles_by_source: List[Dict[str, Any]]
    recent_keyword_alerts: int
    relevance_percentage: float


# Update forward references
ArticleResponse.model_rebuild()
ArticleDetailResponse.model_rebuild()