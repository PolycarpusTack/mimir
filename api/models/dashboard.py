"""
Dashboard Models for Mimir Enterprise API

Models for custom dashboards and widget system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class WidgetType(str, Enum):
    """Available widget types"""

    ARTICLE_COUNT = "article_count"
    ARTICLE_LIST = "article_list"
    SEARCH_RESULTS = "search_results"
    CHART_LINE = "chart_line"
    CHART_BAR = "chart_bar"
    CHART_PIE = "chart_pie"
    METRIC_CARD = "metric_card"
    SOURCE_STATS = "source_stats"
    KEYWORD_CLOUD = "keyword_cloud"
    ALERT_FEED = "alert_feed"
    ACTIVITY_TIMELINE = "activity_timeline"
    CUSTOM_HTML = "custom_html"


class WidgetSize(str, Enum):
    """Widget size options"""

    SMALL = "small"  # 1x1 grid units
    MEDIUM = "medium"  # 2x1 grid units
    LARGE = "large"  # 2x2 grid units
    WIDE = "wide"  # 3x1 grid units
    TALL = "tall"  # 1x2 grid units
    EXTRA_LARGE = "extra_large"  # 3x2 grid units


class RefreshInterval(int, Enum):
    """Widget refresh intervals in seconds"""

    REALTIME = 0  # WebSocket updates
    MINUTE_1 = 60
    MINUTE_5 = 300
    MINUTE_15 = 900
    MINUTE_30 = 1800
    HOUR_1 = 3600
    HOUR_6 = 21600
    HOUR_12 = 43200
    DAY_1 = 86400


class WidgetConfig(BaseModel):
    """Configuration for a dashboard widget"""

    type: WidgetType = Field(..., description="Type of widget")
    title: str = Field(..., min_length=1, max_length=200, description="Widget title")
    size: WidgetSize = Field(WidgetSize.MEDIUM, description="Widget size")
    refresh_interval: RefreshInterval = Field(RefreshInterval.MINUTE_5, description="Auto-refresh interval")

    # Data source configuration
    data_source: Dict[str, Any] = Field(..., description="Configuration for data source")

    # Display configuration
    display_options: Dict[str, Any] = Field(default_factory=dict, description="Display customization options")

    # Position on dashboard grid
    position: Dict[str, int] = Field(
        default_factory=lambda: {"x": 0, "y": 0}, description="Grid position (x, y coordinates)"
    )

    @validator("data_source")
    def validate_data_source(cls, v, values) -> Any:
        """Validate data source based on widget type"""
        widget_type = values.get("type")

        if widget_type == WidgetType.ARTICLE_LIST:
            required_fields = ["query", "limit"]
        elif widget_type in [WidgetType.CHART_LINE, WidgetType.CHART_BAR]:
            required_fields = ["metric", "time_range", "group_by"]
        elif widget_type == WidgetType.SEARCH_RESULTS:
            required_fields = ["saved_search_id"]
        elif widget_type == WidgetType.METRIC_CARD:
            required_fields = ["metric", "aggregation"]
        else:
            required_fields = []

        for field in required_fields:
            if field not in v:
                raise ValueError(f"data_source must include '{field}' for {widget_type} widgets")

        return v


class DashboardLayout(str, Enum):
    """Dashboard layout types"""

    GRID = "grid"  # Flexible grid layout
    SINGLE_COLUMN = "single_column"  # Mobile-friendly single column
    TWO_COLUMN = "two_column"  # Two equal columns
    SIDEBAR_LEFT = "sidebar_left"  # Main content with left sidebar
    SIDEBAR_RIGHT = "sidebar_right"  # Main content with right sidebar


class DashboardTheme(str, Enum):
    """Dashboard color themes"""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"  # Follow system preference
    CUSTOM = "custom"  # User-defined colors


class DashboardCreate(BaseModel):
    """Create dashboard request"""

    name: str = Field(..., min_length=1, max_length=200, description="Dashboard name")
    description: Optional[str] = Field(None, max_length=1000, description="Dashboard description")
    layout: DashboardLayout = Field(DashboardLayout.GRID, description="Layout type")
    theme: DashboardTheme = Field(DashboardTheme.LIGHT, description="Color theme")
    is_default: bool = Field(False, description="Set as default dashboard")
    is_public: bool = Field(False, description="Make publicly accessible")
    tags: List[str] = Field(default_factory=list, description="Dashboard tags")
    widgets: List[WidgetConfig] = Field(default_factory=list, description="Initial widgets")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Dashboard settings")


class DashboardUpdate(BaseModel):
    """Update dashboard request"""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    layout: Optional[DashboardLayout] = None
    theme: Optional[DashboardTheme] = None
    is_default: Optional[bool] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    settings: Optional[Dict[str, Any]] = None


class WidgetCreate(BaseModel):
    """Add widget to dashboard request"""

    type: WidgetType = Field(..., description="Widget type")
    title: str = Field(..., min_length=1, max_length=200, description="Widget title")
    size: WidgetSize = Field(WidgetSize.MEDIUM, description="Widget size")
    refresh_interval: RefreshInterval = Field(RefreshInterval.MINUTE_5)
    data_source: Dict[str, Any] = Field(..., description="Data source configuration")
    display_options: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[Dict[str, int]] = None


class WidgetUpdate(BaseModel):
    """Update widget request"""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    size: Optional[WidgetSize] = None
    refresh_interval: Optional[RefreshInterval] = None
    data_source: Optional[Dict[str, Any]] = None
    display_options: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, int]] = None


class DashboardResponse(BaseModel):
    """Dashboard response model"""

    id: UUID
    name: str
    description: Optional[str]
    layout: DashboardLayout
    theme: DashboardTheme
    is_default: bool
    is_public: bool
    tags: List[str]
    settings: Dict[str, Any]
    organization_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WidgetResponse(BaseModel):
    """Widget response model"""

    id: UUID
    dashboard_id: UUID
    type: WidgetType
    title: str
    size: WidgetSize
    refresh_interval: RefreshInterval
    data_source: Dict[str, Any]
    display_options: Dict[str, Any]
    position: Dict[str, int]
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DashboardWithWidgets(DashboardResponse):
    """Dashboard with widgets response"""

    widgets: List[WidgetResponse]


class WidgetDataResponse(BaseModel):
    """Response containing widget data"""

    widget_id: UUID
    data: Any  # Varies by widget type
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime
    next_update: Optional[datetime] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class DashboardShareRequest(BaseModel):
    """Share dashboard request"""

    user_ids: List[str] = Field(default_factory=list, description="User IDs to share with")
    is_public: bool = Field(False, description="Make dashboard public")
    permissions: List[str] = Field(
        default_factory=lambda: ["view"], description="Permissions for shared users (view, edit)"
    )


class DashboardTemplateResponse(BaseModel):
    """Dashboard template response"""

    id: str
    name: str
    description: str
    preview_image: Optional[str]
    category: str
    widgets: List[WidgetConfig]
    tags: List[str]
