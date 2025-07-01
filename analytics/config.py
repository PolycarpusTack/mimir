"""Configuration management for Mimir Analytics.

This module provides centralized configuration management with environment
variable support and validation.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5433
    database: str = "mimir_analytics"
    user: str = "mimir_user"
    password: str = "mimir_dev_password"
    pool_min_conn: int = 2
    pool_max_conn: int = 10
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv("TIMESCALE_HOST", "localhost"),
            port=int(os.getenv("TIMESCALE_PORT", "5433")),
            database=os.getenv("TIMESCALE_DB", "mimir_analytics"),
            user=os.getenv("TIMESCALE_USER", "mimir_user"),
            password=os.getenv("TIMESCALE_PASSWORD", "mimir_dev_password"),
            pool_min_conn=int(os.getenv("TIMESCALE_POOL_MIN", "2")),
            pool_max_conn=int(os.getenv("TIMESCALE_POOL_MAX", "10"))
        )


@dataclass
class CacheConfig:
    """Caching configuration settings."""
    enabled: bool = True
    query_cache_size: int = 1000
    query_cache_ttl: int = 300  # 5 minutes
    result_cache_size: int = 500
    result_cache_ttl: int = 600  # 10 minutes
    visualization_cache_size: int = 200
    visualization_cache_ttl: int = 180  # 3 minutes
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            query_cache_size=int(os.getenv("QUERY_CACHE_SIZE", "1000")),
            query_cache_ttl=int(os.getenv("QUERY_CACHE_TTL", "300")),
            result_cache_size=int(os.getenv("RESULT_CACHE_SIZE", "500")),
            result_cache_ttl=int(os.getenv("RESULT_CACHE_TTL", "600")),
            visualization_cache_size=int(os.getenv("VIZ_CACHE_SIZE", "200")),
            visualization_cache_ttl=int(os.getenv("VIZ_CACHE_TTL", "180"))
        )


@dataclass
class MLConfig:
    """Machine Learning configuration settings."""
    model_registry_path: str = "analytics/model_registry"
    feature_store_path: str = "analytics/feature_store"
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: str = "mimir_analytics"
    enable_model_validation: bool = True
    model_integrity_checks: bool = True
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Create config from environment variables."""
        return cls(
            model_registry_path=os.getenv("MODEL_REGISTRY_PATH", "analytics/model_registry"),
            feature_store_path=os.getenv("FEATURE_STORE_PATH", "analytics/feature_store"),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "mimir_analytics"),
            enable_model_validation=os.getenv("ENABLE_MODEL_VALIDATION", "true").lower() == "true",
            model_integrity_checks=os.getenv("MODEL_INTEGRITY_CHECKS", "true").lower() == "true"
        )


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_input_validation: bool = True
    max_query_time: float = 30.0  # seconds
    max_file_size: int = 16 * 1024 * 1024  # 16MB
    allowed_file_types: List[str] = None
    rate_limit_requests_per_minute: int = 100
    enable_csrf_protection: bool = True
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.csv', '.json', '.xlsx', '.parquet']
    
    @classmethod
    def from_env(cls) -> 'SecurityConfig':
        """Create config from environment variables."""
        return cls(
            enable_input_validation=os.getenv("ENABLE_INPUT_VALIDATION", "true").lower() == "true",
            max_query_time=float(os.getenv("MAX_QUERY_TIME", "30.0")),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(16 * 1024 * 1024))),
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            enable_csrf_protection=os.getenv("ENABLE_CSRF_PROTECTION", "true").lower() == "true"
        )


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    memory_threshold: float = 0.8  # 80%
    cpu_threshold: float = 0.9  # 90%
    enable_query_optimization: bool = True
    enable_precomputation: bool = True
    batch_size: int = 1000
    max_workers: int = 4
    enable_async_processing: bool = False
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create config from environment variables."""
        return cls(
            memory_threshold=float(os.getenv("MEMORY_THRESHOLD", "0.8")),
            cpu_threshold=float(os.getenv("CPU_THRESHOLD", "0.9")),
            enable_query_optimization=os.getenv("ENABLE_QUERY_OPT", "true").lower() == "true",
            enable_precomputation=os.getenv("ENABLE_PRECOMPUTATION", "true").lower() == "true",
            batch_size=int(os.getenv("BATCH_SIZE", "1000")),
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            enable_async_processing=os.getenv("ENABLE_ASYNC", "false").lower() == "true"
        )


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_structured_logging: bool = False
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create config from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            enable_structured_logging=os.getenv("ENABLE_STRUCTURED_LOGGING", "false").lower() == "true"
        )


@dataclass
class AnalyticsConfig:
    """Main analytics configuration."""
    database: DatabaseConfig
    cache: CacheConfig
    ml: MLConfig
    security: SecurityConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    
    # Analytics-specific settings
    default_time_range_days: int = 30
    default_trend_window: int = 7
    min_data_points_for_analysis: int = 10
    enable_real_time_updates: bool = True
    data_retention_days: int = 365
    
    @classmethod
    def from_env(cls) -> 'AnalyticsConfig':
        """Create complete config from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            cache=CacheConfig.from_env(),
            ml=MLConfig.from_env(),
            security=SecurityConfig.from_env(),
            performance=PerformanceConfig.from_env(),
            logging=LoggingConfig.from_env(),
            default_time_range_days=int(os.getenv("DEFAULT_TIME_RANGE_DAYS", "30")),
            default_trend_window=int(os.getenv("DEFAULT_TREND_WINDOW", "7")),
            min_data_points_for_analysis=int(os.getenv("MIN_DATA_POINTS", "10")),
            enable_real_time_updates=os.getenv("ENABLE_REAL_TIME", "true").lower() == "true",
            data_retention_days=int(os.getenv("DATA_RETENTION_DAYS", "365"))
        )
    
    def validate(self) -> List[str]:
        """Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Database validation
        if self.database.port < 1 or self.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")
        
        if self.database.pool_min_conn < 1:
            errors.append("Minimum database connections must be at least 1")
        
        if self.database.pool_max_conn < self.database.pool_min_conn:
            errors.append("Maximum database connections must be >= minimum connections")
        
        # Cache validation
        if self.cache.query_cache_ttl < 1:
            errors.append("Query cache TTL must be at least 1 second")
        
        # Performance validation
        if not 0.0 < self.performance.memory_threshold < 1.0:
            errors.append("Memory threshold must be between 0.0 and 1.0")
        
        if not 0.0 < self.performance.cpu_threshold < 1.0:
            errors.append("CPU threshold must be between 0.0 and 1.0")
        
        if self.performance.batch_size < 1:
            errors.append("Batch size must be at least 1")
        
        # Security validation
        if self.security.max_query_time < 1.0:
            errors.append("Maximum query time must be at least 1.0 second")
        
        if self.security.rate_limit_requests_per_minute < 1:
            errors.append("Rate limit must be at least 1 request per minute")
        
        # Analytics validation
        if self.default_time_range_days < 1:
            errors.append("Default time range must be at least 1 day")
        
        if self.min_data_points_for_analysis < 2:
            errors.append("Minimum data points for analysis must be at least 2")
        
        return errors


# Global configuration instance
config = AnalyticsConfig.from_env()


def get_config() -> AnalyticsConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AnalyticsConfig:
    """Reload configuration from environment variables."""
    global config
    config = AnalyticsConfig.from_env()
    return config


def validate_config() -> List[str]:
    """Validate the current configuration.
    
    Returns:
        List of validation errors (empty if valid)
    """
    return config.validate()