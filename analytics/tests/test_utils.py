"""Unit tests for analytics utilities.

This module tests all utility functions in the analytics.utils module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from analytics.utils import (
    validate_dataframe, safe_divide, normalize_values, calculate_moving_average,
    calculate_growth_rate, time_series_to_dataframe, clean_numeric_data,
    batch_process, format_bytes, calculate_percentiles, detect_outliers,
    aggregate_time_series, retry_on_exception, performance_timer
)
from analytics.exceptions import DataValidationException


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_dataframe_valid(self, sample_dataframe):
        """Test validation of valid DataFrame."""
        result = validate_dataframe(sample_dataframe, min_rows=3, required_columns=['id', 'name'])
        assert result is sample_dataframe
    
    def test_validate_dataframe_empty(self):
        """Test validation of empty DataFrame."""
        with pytest.raises(DataValidationException, match="DataFrame is empty"):
            validate_dataframe(pd.DataFrame())
    
    def test_validate_dataframe_insufficient_rows(self, sample_dataframe):
        """Test validation with insufficient rows."""
        with pytest.raises(DataValidationException, match="minimum 10 required"):
            validate_dataframe(sample_dataframe, min_rows=10)
    
    def test_validate_dataframe_missing_columns(self, sample_dataframe):
        """Test validation with missing required columns."""
        with pytest.raises(DataValidationException, match="Missing required columns"):
            validate_dataframe(sample_dataframe, required_columns=['id', 'missing_column'])
    
    def test_validate_dataframe_none(self):
        """Test validation of None DataFrame."""
        with pytest.raises(DataValidationException, match="DataFrame is empty or None"):
            validate_dataframe(None)


class TestMathematicalOperations:
    """Test mathematical utility functions."""
    
    def test_safe_divide_normal(self):
        """Test normal division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 3) == pytest.approx(2.333, abs=0.01)
    
    def test_safe_divide_by_zero(self):
        """Test division by zero."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=5.0) == 5.0
    
    def test_safe_divide_invalid_types(self):
        """Test division with invalid types."""
        assert safe_divide("invalid", 2) == 0.0
        assert safe_divide(10, "invalid") == 0.0
    
    def test_calculate_growth_rate(self):
        """Test growth rate calculation."""
        assert calculate_growth_rate(110, 100) == 10.0
        assert calculate_growth_rate(90, 100) == -10.0
        assert calculate_growth_rate(150, 100) == 50.0
    
    def test_calculate_growth_rate_zero_previous(self):
        """Test growth rate with zero previous value."""
        assert calculate_growth_rate(10, 0) == 100.0
        assert calculate_growth_rate(0, 0) == 0.0


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_normalize_values_standard(self):
        """Test standard normalization."""
        values = [1, 2, 3, 4, 5]
        normalized = normalize_values(values, method='standard')
        
        assert len(normalized) == len(values)
        assert abs(np.mean(normalized)) < 0.001  # Mean should be ~0
        assert abs(np.std(normalized) - 1.0) < 0.001  # Std should be ~1
    
    def test_normalize_values_empty(self):
        """Test normalization of empty list."""
        assert normalize_values([]) == []
    
    def test_normalize_values_invalid_method(self):
        """Test normalization with invalid method."""
        with pytest.raises(DataValidationException, match="Unknown normalization method"):
            normalize_values([1, 2, 3], method='invalid')
    
    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ma = calculate_moving_average(values, window=3)
        
        assert len(ma) == len(values)
        assert ma[-1] == 9.0  # Last 3 values: (8+9+10)/3 = 9
    
    def test_calculate_moving_average_small_window(self):
        """Test moving average with small dataset."""
        values = [1, 2]
        ma = calculate_moving_average(values, window=5)
        assert ma == values  # Should return original values
    
    def test_clean_numeric_data(self):
        """Test numeric data cleaning."""
        data = [1, 2.5, "3", None, np.nan, np.inf, "invalid", 4]
        cleaned = clean_numeric_data(data, default_value=-1)
        
        expected = [1.0, 2.5, 3.0, -1.0, -1.0, -1.0, -1.0, 4.0]
        assert cleaned == expected
    
    def test_batch_process(self):
        """Test batch processing."""
        items = list(range(10))
        batches = batch_process(items, batch_size=3)
        
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]
    
    def test_batch_process_empty(self):
        """Test batch processing with empty list."""
        assert batch_process([]) == []


class TestTimeSeriesProcessing:
    """Test time series processing utilities."""
    
    def test_time_series_to_dataframe(self):
        """Test time series conversion to DataFrame."""
        data = [
            {'time': '2023-01-01', 'value': 100},
            {'time': '2023-01-02', 'value': 110},
            {'time': '2023-01-03', 'value': 120}
        ]
        
        df = time_series_to_dataframe(data)
        
        assert len(df) == 3
        assert 'time' in df.columns
        assert df['time'].dtype.kind == 'M'  # datetime type
        assert df['value'].tolist() == [100, 110, 120]
    
    def test_time_series_to_dataframe_empty(self):
        """Test time series conversion with empty data."""
        with pytest.raises(DataValidationException, match="No data provided"):
            time_series_to_dataframe([])
    
    def test_time_series_to_dataframe_missing_time_column(self):
        """Test time series conversion with missing time column."""
        data = [{'value': 100}, {'value': 110}]
        
        with pytest.raises(DataValidationException, match="Time column 'time' not found"):
            time_series_to_dataframe(data)
    
    def test_aggregate_time_series(self, sample_time_series_data):
        """Test time series aggregation."""
        aggregated = aggregate_time_series(
            sample_time_series_data, 
            time_column='time',
            agg_interval='W',  # Weekly
            agg_functions={'value': 'mean', 'count': 'sum'}
        )
        
        assert len(aggregated) <= len(sample_time_series_data)
        assert 'time' in aggregated.columns
        assert 'value' in aggregated.columns


class TestStatisticalFunctions:
    """Test statistical utility functions."""
    
    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        values = list(range(100))
        percentiles = calculate_percentiles(values, [25, 50, 75, 90])
        
        assert percentiles[25] == pytest.approx(24.75, abs=0.5)
        assert percentiles[50] == pytest.approx(49.5, abs=0.5)
        assert percentiles[75] == pytest.approx(74.25, abs=0.5)
        assert percentiles[90] == pytest.approx(89.1, abs=0.5)
    
    def test_calculate_percentiles_empty(self):
        """Test percentile calculation with empty data."""
        assert calculate_percentiles([]) == {}
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Create data with known outliers
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # 100 is outlier
        outliers = detect_outliers(values, method='iqr', threshold=1.5)
        
        assert len(outliers) == len(values)
        assert outliers[-1] == True  # Last value (100) should be outlier
        assert not any(outliers[:-1])  # Other values should not be outliers
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 100)) + [10]  # Add clear outlier
        outliers = detect_outliers(values, method='zscore', threshold=3)
        
        assert len(outliers) == len(values)
        assert outliers[-1] == True  # Last value should be outlier
    
    def test_detect_outliers_empty(self):
        """Test outlier detection with empty data."""
        assert detect_outliers([]) == []


class TestUtilityFunctions:
    """Test general utility functions."""
    
    def test_format_bytes(self):
        """Test byte formatting."""
        assert format_bytes(0) == "0 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1536) == "1.5 KB"  # 1.5 KB
    
    def test_format_bytes_large(self):
        """Test formatting of large byte values."""
        tb_size = 1024 ** 4
        assert format_bytes(tb_size) == "1.0 TB"


class TestDecorators:
    """Test utility decorators."""
    
    def test_retry_on_exception_success(self):
        """Test retry decorator with successful function."""
        call_count = 0
        
        @retry_on_exception(max_retries=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_exception_eventual_success(self):
        """Test retry decorator with eventual success."""
        call_count = 0
        
        @retry_on_exception(max_retries=3, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = test_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_exception_max_retries_exceeded(self):
        """Test retry decorator with max retries exceeded."""
        call_count = 0
        
        @retry_on_exception(max_retries=2, delay=0.01)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            test_function()
        
        assert call_count == 3  # Initial call + 2 retries
    
    def test_performance_timer(self):
        """Test performance timer decorator."""
        import time
        
        @performance_timer
        def test_function():
            time.sleep(0.01)  # Sleep for 10ms
            return "done"
        
        result = test_function()
        assert result == "done"
        # Note: We can't easily test the timing without mocking the logger


class TestIntegration:
    """Integration tests for utility functions."""
    
    def test_data_processing_pipeline(self):
        """Test a complete data processing pipeline using utilities."""
        # Generate test data
        raw_data = [
            {'time': f'2023-01-{i:02d}', 'value': i + np.random.normal(0, 0.1)}
            for i in range(1, 31)
        ]
        
        # Convert to DataFrame
        df = time_series_to_dataframe(raw_data)
        
        # Validate
        validated_df = validate_dataframe(df, min_rows=20, required_columns=['time', 'value'])
        
        # Clean values
        clean_values = clean_numeric_data(validated_df['value'].tolist())
        
        # Calculate moving average
        ma_values = calculate_moving_average(clean_values, window=7)
        
        # Detect outliers
        outliers = detect_outliers(clean_values, method='iqr')
        
        # Verify results
        assert len(df) == 30
        assert len(clean_values) == 30
        assert len(ma_values) == 30
        assert len(outliers) == 30
        assert isinstance(validated_df, pd.DataFrame)
    
    def test_statistical_analysis_pipeline(self):
        """Test statistical analysis pipeline."""
        # Generate test data with known characteristics
        np.random.seed(42)
        values = np.random.normal(100, 15, 1000).tolist()
        
        # Add some outliers
        values.extend([200, 250, -50])
        
        # Clean data
        clean_values = clean_numeric_data(values)
        
        # Calculate statistics
        percentiles = calculate_percentiles(clean_values, [25, 50, 75, 95])
        outliers = detect_outliers(clean_values, method='zscore', threshold=2.5)
        
        # Verify results
        assert len(clean_values) == len(values)
        assert 50 in percentiles  # Median
        assert any(outliers)  # Should detect some outliers
        assert percentiles[50] == pytest.approx(100, abs=10)  # Median near 100


if __name__ == '__main__':
    pytest.main([__file__])