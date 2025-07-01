"""Custom exceptions for Mimir Analytics system.

This module defines a hierarchy of custom exceptions to provide better error
handling and debugging capabilities across the analytics codebase.
"""

from typing import Any, Dict, Optional


class AnalyticsBaseException(Exception):
    """Base exception for all analytics-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Additional error details for debugging
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class DataWarehouseException(AnalyticsBaseException):
    """Exception raised for data warehouse operations."""
    pass


class DatabaseConnectionException(DataWarehouseException):
    """Exception raised for database connection failures."""
    pass


class QueryExecutionException(DataWarehouseException):
    """Exception raised for query execution failures."""
    pass


class DataValidationException(AnalyticsBaseException):
    """Exception raised for data validation failures."""
    pass


class MetricsCollectionException(AnalyticsBaseException):
    """Exception raised for metrics collection failures."""
    pass


class TrendAnalysisException(AnalyticsBaseException):
    """Exception raised for trend analysis failures."""
    pass


class PredictionException(AnalyticsBaseException):
    """Exception raised for prediction/forecasting failures."""
    pass


class ModelRegistryException(AnalyticsBaseException):
    """Exception raised for ML model registry operations."""
    pass


class ModelIntegrityException(ModelRegistryException):
    """Exception raised when model integrity checks fail."""
    pass


class ModelNotFoundException(ModelRegistryException):
    """Exception raised when a requested model is not found."""
    pass


class FeatureEngineeringException(AnalyticsBaseException):
    """Exception raised for feature engineering operations."""
    pass


class ReportGenerationException(AnalyticsBaseException):
    """Exception raised for report generation failures."""
    pass


class TemplateNotFoundException(ReportGenerationException):
    """Exception raised when a report template is not found."""
    pass


class ClusteringException(AnalyticsBaseException):
    """Exception raised for clustering operations."""
    pass


class RecommendationException(AnalyticsBaseException):
    """Exception raised for recommendation engine operations."""
    pass


class PerformanceException(AnalyticsBaseException):
    """Exception raised for performance optimization operations."""
    pass


class ConfigurationException(AnalyticsBaseException):
    """Exception raised for configuration-related issues."""
    pass


class SecurityException(AnalyticsBaseException):
    """Exception raised for security-related issues."""
    pass


class InputValidationException(SecurityException):
    """Exception raised for input validation failures."""
    pass


class AuthenticationException(SecurityException):
    """Exception raised for authentication failures."""
    pass


class AuthorizationException(SecurityException):
    """Exception raised for authorization failures."""
    pass