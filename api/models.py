"""
Pydantic Models for Mimir Enterprise API

Defines request and response models for all API endpoints with validation,
serialization, and OpenAPI documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator


# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    ANALYST = "analyst"
    VIEWER = "viewer"


class APIKeyType(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"


class ArticleStatus(str, Enum):
    NEW = "new"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    ERROR = "error"


class OrganizationTier(str, Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


# Base Models
class BaseResponse(BaseModel):
    """Base response model"""

    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class PaginationMeta(BaseModel):
    """Pagination metadata"""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_previous: bool


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""

    data: List[Any]
    meta: PaginationMeta


# Authentication Models
class LoginRequest(BaseModel):
    """User login request"""

    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginResponse(BaseModel):
    """User login response"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class RefreshTokenRequest(BaseModel):
    """Token refresh request"""

    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request"""

    current_password: str
    new_password: str = Field(..., min_length=8)


# User Models
class UserCreate(BaseModel):
    """Create user request"""

    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.VIEWER


class UserUpdate(BaseModel):
    """Update user request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model"""

    id: UUID
    email: EmailStr
    name: str
    role: UserRole
    organization_id: UUID
    organization_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Organization Models
class OrganizationCreate(BaseModel):
    """Create organization request"""

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, regex=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
    tier: OrganizationTier = OrganizationTier.STARTER


class OrganizationUpdate(BaseModel):
    """Update organization request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    tier: Optional[OrganizationTier] = None
    settings: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None


class OrganizationUsage(BaseModel):
    """Organization usage statistics"""

    sources: Dict[str, Union[int, float]]
    keywords: Dict[str, Union[int, float]]
    users: Dict[str, Union[int, float]]
    storage: Dict[str, Union[int, float]]
    api_calls_limit: int


class OrganizationResponse(BaseModel):
    """Organization response model"""

    id: UUID
    name: str
    slug: str
    tier: OrganizationTier
    subscription_status: SubscriptionStatus
    max_sources: int
    max_keywords: int
    max_users: int
    current_sources: int
    current_keywords: int
    current_users: int
    settings: Dict[str, Any]
    features: Dict[str, Any]
    created_at: datetime
    usage: Optional[OrganizationUsage] = None

    class Config:
        from_attributes = True


# API Key Models
class APIKeyCreate(BaseModel):
    """Create API key request"""

    name: str = Field(..., min_length=1, max_length=255)
    type: APIKeyType = APIKeyType.READ_ONLY
    expires_in_days: Optional[int] = Field(None, gt=0, le=365)
    permissions: Optional[List[str]] = None
    allowed_ips: Optional[List[str]] = None


class APIKeyResponse(BaseModel):
    """API key response model"""

    id: UUID
    name: str
    key_prefix: str
    type: APIKeyType
    permissions: List[str]
    rate_limit_per_hour: Optional[int]
    allowed_ips: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    created_at: datetime
    created_by: Optional[UUID]

    class Config:
        from_attributes = True


class APIKeyCreateResponse(BaseModel):
    """API key creation response with actual key"""

    api_key: str
    key_info: APIKeyResponse


# Article Models
class ArticleCreate(BaseModel):
    """Create article request"""

    url: str = Field(..., regex=r"^https?://")
    title: str = Field(..., min_length=1)
    content: Optional[str] = None
    summary: Optional[str] = None
    published_date: Optional[datetime] = None
    source: str = Field(..., min_length=1)
    author: Optional[str] = None
    category: Optional[str] = None
    language: str = "en"
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class ArticleUpdate(BaseModel):
    """Update article request"""

    title: Optional[str] = Field(None, min_length=1)
    content: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[ArticleStatus] = None


class ArticleResponse(BaseModel):
    """Article response model"""

    id: UUID
    url: str
    title: str
    content: Optional[str] = None
    summary: Optional[str] = None
    published_date: Optional[datetime]
    scraped_at: datetime
    updated_at: datetime
    source: str
    author: Optional[str]
    category: Optional[str]
    language: str
    status: ArticleStatus
    tags: List[str]
    metadata: Dict[str, Any]
    entities: Dict[str, Any]
    organization_id: UUID

    class Config:
        from_attributes = True


# Source Models
class SourceCreate(BaseModel):
    """Create source request"""

    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., regex=r"^https?://")
    type: str = Field(..., regex=r"^(rss|html|api)$")
    category: Optional[str] = None
    enabled: bool = True
    config: Dict[str, Any] = {}
    selectors: Dict[str, Any] = {}
    rate_limit_seconds: int = Field(2, ge=1, le=3600)


class SourceUpdate(BaseModel):
    """Update source request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    enabled: Optional[bool] = None
    category: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    selectors: Optional[Dict[str, Any]] = None
    rate_limit_seconds: Optional[int] = Field(None, ge=1, le=3600)


class SourceResponse(BaseModel):
    """Source response model"""

    id: UUID
    name: str
    url: str
    type: str
    enabled: bool
    category: Optional[str]
    config: Dict[str, Any]
    selectors: Dict[str, Any]
    rate_limit_seconds: int
    last_scraped_at: Optional[datetime]
    next_scrape_at: Optional[datetime]
    total_articles: int
    failed_attempts: int
    last_error: Optional[str]
    last_error_at: Optional[datetime]
    organization_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Keyword Models
class KeywordCreate(BaseModel):
    """Create keyword request"""

    keyword: str = Field(..., min_length=1, max_length=255)
    is_regex: bool = False
    case_sensitive: bool = False
    enabled: bool = True
    alert_email: bool = True
    priority: int = Field(5, ge=1, le=10)
    metadata: Dict[str, Any] = {}


class KeywordUpdate(BaseModel):
    """Update keyword request"""

    keyword: Optional[str] = Field(None, min_length=1, max_length=255)
    is_regex: Optional[bool] = None
    case_sensitive: Optional[bool] = None
    enabled: Optional[bool] = None
    alert_email: Optional[bool] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None


class KeywordResponse(BaseModel):
    """Keyword response model"""

    id: UUID
    keyword: str
    is_regex: bool
    case_sensitive: bool
    enabled: bool
    alert_email: bool
    priority: int
    metadata: Dict[str, Any]
    organization_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Search Models
class SearchRequest(BaseModel):
    """Search request model"""

    query: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    sort_by: str = "published_date"
    sort_order: SortOrder = SortOrder.DESC
    semantic_search: bool = False
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Search response model"""

    query: Optional[str]
    total_results: int
    search_time_ms: float
    semantic_search: bool
    articles: List[ArticleResponse]
    facets: Optional[Dict[str, Any]] = None


# Saved Search Models
class SavedSearchCreate(BaseModel):
    """Create saved search request"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    query: Dict[str, Any]
    is_alert: bool = False
    alert_frequency: int = Field(3600, ge=300, le=86400)
    notification_channels: List[str] = []
    notification_settings: Dict[str, Any] = {}
    is_shared: bool = False
    shared_with: List[str] = []
    tags: List[str] = []


class SavedSearchUpdate(BaseModel):
    """Update saved search request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    query: Optional[Dict[str, Any]] = None
    is_alert: Optional[bool] = None
    alert_frequency: Optional[int] = Field(None, ge=300, le=86400)
    notification_channels: Optional[List[str]] = None
    notification_settings: Optional[Dict[str, Any]] = None
    is_shared: Optional[bool] = None
    shared_with: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class SavedSearchResponse(BaseModel):
    """Saved search response model"""

    id: UUID
    name: str
    description: Optional[str]
    query: Dict[str, Any]
    is_alert: bool
    alert_frequency: int
    last_checked_at: Optional[datetime]
    next_check_at: Optional[datetime]
    notification_channels: List[str]
    notification_settings: Dict[str, Any]
    is_shared: bool
    shared_with: List[str]
    tags: List[str]
    organization_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Webhook Models
class WebhookCreate(BaseModel):
    """Create webhook request"""

    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., regex=r"^https?://")
    events: List[str] = Field(..., min_items=1)
    headers: Dict[str, str] = {}
    secret: Optional[str] = None
    timeout_seconds: int = Field(30, ge=5, le=300)
    retry_attempts: int = Field(3, ge=0, le=10)
    retry_backoff_seconds: int = Field(60, ge=1, le=3600)


class WebhookUpdate(BaseModel):
    """Update webhook request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    url: Optional[str] = Field(None, regex=r"^https?://")
    events: Optional[List[str]] = Field(None, min_items=1)
    headers: Optional[Dict[str, str]] = None
    secret: Optional[str] = None
    timeout_seconds: Optional[int] = Field(None, ge=5, le=300)
    retry_attempts: Optional[int] = Field(None, ge=0, le=10)
    retry_backoff_seconds: Optional[int] = Field(None, ge=1, le=3600)
    is_active: Optional[bool] = None


class WebhookResponse(BaseModel):
    """Webhook response model"""

    id: UUID
    name: str
    url: str
    events: List[str]
    headers: Dict[str, str]
    timeout_seconds: int
    retry_attempts: int
    retry_backoff_seconds: int
    is_active: bool
    last_triggered_at: Optional[datetime]
    success_count: int
    failure_count: int
    last_error: Optional[str]
    last_error_at: Optional[datetime]
    organization_id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Analytics Models
class AnalyticsRequest(BaseModel):
    """Analytics request model"""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: str = Field("day", regex=r"^(hour|day|week|month)$")
    metrics: List[str] = ["article_count", "source_count"]
    filters: Dict[str, Any] = {}


class AnalyticsResponse(BaseModel):
    """Analytics response model"""

    period: Dict[str, datetime]
    granularity: str
    metrics: Dict[str, List[Dict[str, Any]]]
    summary: Dict[str, Any]


# System Models
class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    timestamp: datetime
    database: str
    services: Dict[str, str]


class APIInfoResponse(BaseModel):
    """API information response"""

    name: str
    version: str
    description: str
    documentation: str
    openapi: str


# Update forward references
LoginResponse.update_forward_refs()
APIKeyCreateResponse.update_forward_refs()
