"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    url: str = Field(
        default="sqlite:///./mimir.db",
        description="Database URL (SQLite or PostgreSQL)"
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries (for debugging)"
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size"
    )
    max_overflow: int = Field(
        default=10,
        description="Maximum connection overflow"
    )

    model_config = ConfigDict(env_prefix="MIMIR_DB_")


class ScrapingSettings(BaseSettings):
    """Scraping configuration."""
    
    user_agent: str = Field(
        default="Mimir Broadcast Intelligence Bot 1.0 (+https://mimir.ai/bot)",
        description="User agent for HTTP requests"
    )
    request_delay: float = Field(
        default=2.0,
        description="Delay between requests in seconds"
    )
    request_timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=5.0,
        description="Delay between retries in seconds"
    )
    articles_per_run: int = Field(
        default=100,
        description="Maximum articles to scrape per run"
    )
    articles_per_source: int = Field(
        default=20,
        description="Maximum articles per source per run"
    )
    daily_article_limit: int = Field(
        default=500,
        description="Maximum articles to scrape per day"
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to respect robots.txt"
    )

    model_config = ConfigDict(env_prefix="MIMIR_SCRAPING_")


class MonitoringSettings(BaseSettings):
    """Monitoring and alerting configuration."""
    
    keywords: List[str] = Field(
        default=[
            "streaming", "OTT", "broadcast", "television", "TV",
            "ATSC 3.0", "NextGen TV", "DVB", "IPTV", "5G broadcast",
            "video", "media", "content delivery", "CDN", "playout"
        ],
        description="Keywords to monitor for alerts"
    )
    exclude_keywords: List[str] = Field(
        default=[
            "sports scores", "celebrity gossip", "reality TV drama",
            "fashion", "lifestyle", "cooking shows"
        ],
        description="Keywords to exclude from alerts"
    )
    alert_on_match: bool = Field(
        default=True,
        description="Whether to send alerts on keyword matches"
    )
    min_relevance_score: float = Field(
        default=0.3,
        description="Minimum relevance score for articles"
    )

    model_config = ConfigDict(env_prefix="MIMIR_MONITORING_")


class EmailSettings(BaseSettings):
    """Email notification configuration."""
    
    enabled: bool = Field(
        default=False,
        description="Whether email notifications are enabled"
    )
    smtp_server: Optional[str] = Field(
        default=None,
        description="SMTP server hostname"
    )
    smtp_port: int = Field(
        default=587,
        description="SMTP server port"
    )
    smtp_user: Optional[str] = Field(
        default=None,
        description="SMTP username"
    )
    smtp_password: Optional[str] = Field(
        default=None,
        description="SMTP password"
    )
    sender_email: Optional[str] = Field(
        default=None,
        description="Sender email address"
    )
    recipient_emails: List[str] = Field(
        default=[],
        description="List of recipient email addresses"
    )
    use_tls: bool = Field(
        default=True,
        description="Whether to use TLS for SMTP"
    )

    model_config = ConfigDict(env_prefix="MIMIR_EMAIL_")


class APISettings(BaseSettings):
    """External API configuration."""
    
    newsapi_enabled: bool = Field(
        default=False,
        description="Enable NewsAPI integration"
    )
    newsapi_key: Optional[str] = Field(
        default=None,
        description="NewsAPI.org API key"
    )
    reddit_enabled: bool = Field(
        default=False,
        description="Enable Reddit API integration"
    )
    reddit_client_id: Optional[str] = Field(
        default=None,
        description="Reddit API client ID"
    )
    reddit_client_secret: Optional[str] = Field(
        default=None,
        description="Reddit API client secret"
    )
    reddit_user_agent: str = Field(
        default="Mimir News Aggregator 1.0",
        description="Reddit API user agent"
    )

    model_config = ConfigDict(env_prefix="MIMIR_API_")


class WebSettings(BaseSettings):
    """Web interface configuration."""
    
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the web server to"
    )
    port: int = Field(
        default=5000,
        description="Port to bind the web server to"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload in development"
    )
    cors_origins: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )

    model_config = ConfigDict(env_prefix="MIMIR_WEB_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file: Optional[str] = Field(
        default="logs/mimir.log",
        description="Log file path"
    )
    max_bytes: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        description="Number of log file backups to keep"
    )

    model_config = ConfigDict(env_prefix="MIMIR_LOG_")


class Settings(BaseSettings):
    """Main application settings."""
    
    # App metadata
    app_name: str = Field(
        default="Mimir",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, production, testing)"
    )
    
    # Component settings
    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings,
        description="Database configuration"
    )
    scraping: ScrapingSettings = Field(
        default_factory=ScrapingSettings,
        description="Scraping configuration"
    )
    monitoring: MonitoringSettings = Field(
        default_factory=MonitoringSettings,
        description="Monitoring configuration"
    )
    email: EmailSettings = Field(
        default_factory=EmailSettings,
        description="Email configuration"
    )
    api: APISettings = Field(
        default_factory=APISettings,
        description="External API configuration"
    )
    web: WebSettings = Field(
        default_factory=WebSettings,
        description="Web interface configuration"
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging configuration"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = ["development", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()