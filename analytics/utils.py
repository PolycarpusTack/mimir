"""Utility functions for Mimir Analytics.

This module provides common utility functions to reduce code duplication
across the analytics codebase.
"""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .exceptions import DataValidationException, AnalyticsBaseException

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, 
                      required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Validate DataFrame meets minimum requirements.
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
        required_columns: List of required column names
        
    Returns:
        Validated DataFrame
        
    Raises:
        DataValidationException: If validation fails
    """
    if df is None or df.empty:
        raise DataValidationException("DataFrame is empty or None")
    
    if len(df) < min_rows:
        raise DataValidationException(
            f"DataFrame has {len(df)} rows, minimum {min_rows} required"
        )
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationException(
                f"Missing required columns: {missing_cols}"
            )
    
    return df


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], 
                default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def normalize_values(values: List[Union[int, float]], 
                    method: str = 'standard') -> List[float]:
    """Normalize a list of values using specified method.
    
    Args:
        values: List of numeric values to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        
    Returns:
        List of normalized values
        
    Raises:
        DataValidationException: If normalization fails
    """
    if not values:
        return []
    
    try:
        arr = np.array(values).reshape(-1, 1)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise DataValidationException(f"Unknown normalization method: {method}")
        
        normalized = scaler.fit_transform(arr).flatten()
        return normalized.tolist()
        
    except Exception as e:
        raise DataValidationException(f"Normalization failed: {e}")


def calculate_moving_average(values: List[Union[int, float]], 
                           window: int = 7) -> List[float]:
    """Calculate moving average for a list of values.
    
    Args:
        values: List of numeric values
        window: Window size for moving average
        
    Returns:
        List of moving average values
    """
    if len(values) < window:
        return values
    
    df = pd.DataFrame({'values': values})
    ma = df['values'].rolling(window=window, min_periods=1).mean()
    return ma.tolist()


def calculate_growth_rate(current: Union[int, float], 
                         previous: Union[int, float]) -> float:
    """Calculate growth rate between two values.
    
    Args:
        current: Current value
        previous: Previous value
        
    Returns:
        Growth rate as percentage
    """
    if previous == 0:
        return 100.0 if current > 0 else 0.0
    
    return ((current - previous) / previous) * 100.0


def time_series_to_dataframe(data: List[Dict[str, Any]], 
                           time_column: str = 'time') -> pd.DataFrame:
    """Convert time series data to DataFrame with proper datetime handling.
    
    Args:
        data: List of dictionaries containing time series data
        time_column: Name of the time column
        
    Returns:
        DataFrame with datetime index
        
    Raises:
        DataValidationException: If conversion fails
    """
    if not data:
        raise DataValidationException("No data provided for conversion")
    
    try:
        df = pd.DataFrame(data)
        
        if time_column not in df.columns:
            raise DataValidationException(f"Time column '{time_column}' not found")
        
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        raise DataValidationException(f"Time series conversion failed: {e}")


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)) -> Callable:
    """Decorator to retry function execution on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, retrying: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def performance_timer(func: Callable) -> Callable:
    """Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Function {func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper


def generate_file_hash(file_path: Path) -> str:
    """Generate SHA-256 hash for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def clean_numeric_data(data: List[Any], default_value: float = 0.0) -> List[float]:
    """Clean and convert data to numeric values.
    
    Args:
        data: List of values to clean
        default_value: Default value for non-numeric data
        
    Returns:
        List of cleaned numeric values
    """
    cleaned = []
    
    for value in data:
        try:
            if pd.isna(value) or value is None:
                cleaned.append(default_value)
            elif isinstance(value, (int, float)):
                if np.isfinite(value):
                    cleaned.append(float(value))
                else:
                    cleaned.append(default_value)
            else:
                # Try to convert string to float
                cleaned.append(float(value))
        except (ValueError, TypeError):
            cleaned.append(default_value)
    
    return cleaned


def batch_process(items: List[Any], batch_size: int = 100) -> List[List[Any]]:
    """Split items into batches for processing.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    if not items:
        return []
    
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


def format_bytes(size_bytes: int) -> str:
    """Format byte size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def calculate_percentiles(values: List[Union[int, float]], 
                         percentiles: List[float] = [25, 50, 75, 90, 95]) -> Dict[float, float]:
    """Calculate percentiles for a list of values.
    
    Args:
        values: List of numeric values
        percentiles: List of percentile values to calculate
        
    Returns:
        Dictionary mapping percentile to value
    """
    if not values:
        return {}
    
    clean_values = clean_numeric_data(values)
    
    result = {}
    for p in percentiles:
        result[p] = np.percentile(clean_values, p)
    
    return result


def detect_outliers(values: List[Union[int, float]], 
                   method: str = 'iqr', threshold: float = 1.5) -> List[bool]:
    """Detect outliers in a list of values.
    
    Args:
        values: List of numeric values
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        List of boolean values indicating outliers
    """
    if not values:
        return []
    
    clean_values = np.array(clean_numeric_data(values))
    outliers = np.zeros(len(clean_values), dtype=bool)
    
    if method == 'iqr':
        q1 = np.percentile(clean_values, 25)
        q3 = np.percentile(clean_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = (clean_values < lower_bound) | (clean_values > upper_bound)
        
    elif method == 'zscore':
        mean_val = np.mean(clean_values)
        std_val = np.std(clean_values)
        
        if std_val > 0:
            z_scores = np.abs((clean_values - mean_val) / std_val)
            outliers = z_scores > threshold
    
    return outliers.tolist()


def aggregate_time_series(df: pd.DataFrame, time_column: str = 'time',
                         agg_interval: str = '1H', 
                         agg_functions: Dict[str, str] = None) -> pd.DataFrame:
    """Aggregate time series data by specified interval.
    
    Args:
        df: DataFrame with time series data
        time_column: Name of the time column
        agg_interval: Aggregation interval (e.g., '1H', '1D')
        agg_functions: Dictionary mapping column names to aggregation functions
        
    Returns:
        Aggregated DataFrame
        
    Raises:
        DataValidationException: If aggregation fails
    """
    if agg_functions is None:
        agg_functions = {}
    
    try:
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
        
        # Default aggregation functions
        default_agg = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            default_agg[col] = agg_functions.get(col, 'mean')
        
        # Perform aggregation
        aggregated = df.resample(agg_interval).agg(default_agg)
        aggregated = aggregated.reset_index()
        
        return aggregated
        
    except Exception as e:
        raise DataValidationException(f"Time series aggregation failed: {e}")