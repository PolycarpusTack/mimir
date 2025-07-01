"""Unit tests for analytics validator.

This module tests the AnalyticsValidator class and its methods.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from analytics.analytics_validator import AnalyticsValidator
from analytics.exceptions import (
    DatabaseConnectionException, QueryExecutionException,
    DataValidationException
)


class TestAnalyticsValidator:
    """Test AnalyticsValidator class."""
    
    @pytest.fixture
    def validator(self, mock_db_adapter, mock_analytics_warehouse):
        """Create validator instance for testing."""
        return AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
    
    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor for database operations."""
        cursor = Mock()
        cursor.fetchone.return_value = (100,)  # Default row count
        cursor.fetchall.return_value = []
        return cursor
    
    def test_init(self, mock_db_adapter, mock_analytics_warehouse):
        """Test validator initialization."""
        validator = AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
        
        assert validator.db_adapter == mock_db_adapter
        assert validator.warehouse == mock_analytics_warehouse
        assert validator.metrics_collector is not None
        assert validator.trend_analyzer is not None


class TestTableValidationMethods:
    """Test individual table validation methods."""
    
    @pytest.fixture
    def validator(self, mock_db_adapter, mock_analytics_warehouse):
        """Create validator instance for testing."""
        return AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
    
    def test_get_table_row_count(self, validator):
        """Test table row count retrieval."""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (150,)
        
        count = validator._get_table_row_count(mock_cursor, "test_table")
        
        assert count == 150
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) FROM test_table")
    
    def test_check_null_values_no_issues(self, validator):
        """Test null value checking with no issues."""
        mock_cursor = Mock()
        
        # Mock schema query
        mock_cursor.fetchall.side_effect = [
            [('id', 'integer', 'NO'), ('name', 'text', 'YES')],  # Schema query
            (0,)  # Null count query
        ]
        mock_cursor.fetchone.return_value = (0,)
        
        issues = validator._check_null_values(mock_cursor, "analytics.test_table")
        
        assert issues == []
    
    def test_check_null_values_with_issues(self, validator):
        """Test null value checking with issues found."""
        mock_cursor = Mock()
        
        # Mock schema query - one non-nullable column
        mock_cursor.fetchall.return_value = [('id', 'integer', 'NO')]
        mock_cursor.fetchone.return_value = (5,)  # 5 null values found
        
        issues = validator._check_null_values(mock_cursor, "analytics.test_table")
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'null_values'
        assert issues[0]['column'] == 'id'
        assert issues[0]['count'] == 5
        assert issues[0]['severity'] == 'high'
    
    def test_check_duplicate_keys_no_duplicates(self, validator):
        """Test duplicate key checking with no duplicates."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []  # No duplicates found
        
        issues = validator._check_duplicate_keys(mock_cursor, "analytics.article_metrics")
        
        assert issues == []
    
    def test_check_duplicate_keys_with_duplicates(self, validator):
        """Test duplicate key checking with duplicates found."""
        mock_cursor = Mock()
        # Mock duplicate records
        mock_cursor.fetchall.return_value = [
            (1, datetime.utcnow(), 2),
            (2, datetime.utcnow(), 3)
        ]
        
        issues = validator._check_duplicate_keys(mock_cursor, "analytics.article_metrics")
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'duplicate_keys'
        assert issues[0]['count'] == 2
        assert issues[0]['severity'] == 'high'
    
    def test_check_duplicate_keys_non_article_table(self, validator):
        """Test duplicate key checking for non-article table."""
        mock_cursor = Mock()
        
        issues = validator._check_duplicate_keys(mock_cursor, "analytics.source_metrics")
        
        assert issues == []
        # Should not execute any queries for non-article tables
        mock_cursor.execute.assert_not_called()
    
    def test_check_data_freshness_fresh_data(self, validator):
        """Test data freshness checking with fresh data."""
        mock_cursor = Mock()
        # Fresh data (1 hour old)
        fresh_time = datetime.utcnow() - timedelta(hours=1)
        mock_cursor.fetchone.return_value = (fresh_time,)
        
        issues = validator._check_data_freshness(mock_cursor, "analytics.test_table")
        
        assert issues == []
    
    def test_check_data_freshness_stale_data_medium(self, validator):
        """Test data freshness checking with medium staleness."""
        mock_cursor = Mock()
        # Stale data (30 hours old)
        stale_time = datetime.utcnow() - timedelta(hours=30)
        mock_cursor.fetchone.return_value = (stale_time,)
        
        issues = validator._check_data_freshness(mock_cursor, "analytics.test_table")
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'stale_data'
        assert issues[0]['severity'] == 'medium'
        assert issues[0]['age_hours'] == pytest.approx(30, abs=1)
    
    def test_check_data_freshness_stale_data_high(self, validator):
        """Test data freshness checking with high staleness."""
        mock_cursor = Mock()
        # Very stale data (60 hours old)
        stale_time = datetime.utcnow() - timedelta(hours=60)
        mock_cursor.fetchone.return_value = (stale_time,)
        
        issues = validator._check_data_freshness(mock_cursor, "analytics.test_table")
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'stale_data'
        assert issues[0]['severity'] == 'high'
        assert issues[0]['age_hours'] == pytest.approx(60, abs=1)
    
    def test_check_data_freshness_no_data(self, validator):
        """Test data freshness checking with no time data."""
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (None,)
        
        issues = validator._check_data_freshness(mock_cursor, "analytics.test_table")
        
        assert issues == []
    
    def test_compile_check_results_all_passed(self, validator):
        """Test check results compilation with all checks passed."""
        issues = []  # No issues
        
        results = validator._compile_check_results(issues)
        
        expected = {
            'null_check': 'passed',
            'duplicate_check': 'passed',
            'freshness_check': 'passed'
        }
        assert results == expected
    
    def test_compile_check_results_with_failures(self, validator):
        """Test check results compilation with some failures."""
        issues = [
            {'type': 'null_values', 'column': 'id', 'count': 5},
            {'type': 'stale_data', 'age_hours': 30}
        ]
        
        results = validator._compile_check_results(issues)
        
        expected = {
            'null_check': 'failed',
            'duplicate_check': 'passed',
            'freshness_check': 'failed'
        }
        assert results == expected


class TestTableQualityValidation:
    """Test the main table quality validation method."""
    
    @pytest.fixture
    def validator(self, mock_db_adapter, mock_analytics_warehouse):
        """Create validator instance for testing."""
        return AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
    
    @patch('analytics.analytics_validator.AnalyticsValidator._get_table_row_count')
    @patch('analytics.analytics_validator.AnalyticsValidator._check_null_values')
    @patch('analytics.analytics_validator.AnalyticsValidator._check_duplicate_keys')
    @patch('analytics.analytics_validator.AnalyticsValidator._check_data_freshness')
    @patch('analytics.analytics_validator.AnalyticsValidator._compile_check_results')
    def test_validate_table_quality_success(self, mock_compile, mock_freshness, 
                                           mock_duplicates, mock_nulls, mock_count,
                                           validator, mock_analytics_warehouse):
        """Test successful table quality validation."""
        # Mock the connection context manager
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_analytics_warehouse.get_connection.return_value.__enter__.return_value = mock_conn
        
        # Mock method returns
        mock_count.return_value = 100
        mock_nulls.return_value = []
        mock_duplicates.return_value = []
        mock_freshness.return_value = []
        mock_compile.return_value = {'null_check': 'passed'}
        
        result = validator._validate_table_quality("analytics.test_table")
        
        assert result['row_count'] == 100
        assert result['issues'] == []
        assert result['checks'] == {'null_check': 'passed'}
        
        # Verify all methods were called
        mock_count.assert_called_once_with(mock_cursor, "analytics.test_table")
        mock_nulls.assert_called_once_with(mock_cursor, "analytics.test_table")
        mock_duplicates.assert_called_once_with(mock_cursor, "analytics.test_table")
        mock_freshness.assert_called_once_with(mock_cursor, "analytics.test_table")
        mock_compile.assert_called_once_with([])
    
    def test_validate_table_quality_database_error(self, validator, mock_analytics_warehouse):
        """Test table quality validation with database connection error."""
        # Mock connection to raise DatabaseConnectionException
        mock_analytics_warehouse.get_connection.side_effect = DatabaseConnectionException("Connection failed")
        
        result = validator._validate_table_quality("analytics.test_table")
        
        assert result['row_count'] == 0
        assert len(result['issues']) == 1
        assert result['issues'][0]['type'] == 'database_connection_error'
        assert result['issues'][0]['severity'] == 'critical'
    
    def test_validate_table_quality_query_error(self, validator, mock_analytics_warehouse):
        """Test table quality validation with query execution error."""
        # Mock connection but cursor raises QueryExecutionException
        mock_conn = Mock()
        mock_analytics_warehouse.get_connection.return_value.__enter__.return_value = mock_conn
        
        with patch('analytics.analytics_validator.AnalyticsValidator._get_table_row_count') as mock_count:
            mock_count.side_effect = QueryExecutionException("Query failed")
            
            result = validator._validate_table_quality("analytics.test_table")
            
            assert result['row_count'] == 0
            assert len(result['issues']) == 1
            assert result['issues'][0]['type'] == 'query_execution_error'
            assert result['issues'][0]['severity'] == 'high'
    
    def test_validate_table_quality_unexpected_error(self, validator, mock_analytics_warehouse):
        """Test table quality validation with unexpected error."""
        # Mock connection but raise unexpected exception
        mock_conn = Mock()
        mock_analytics_warehouse.get_connection.return_value.__enter__.return_value = mock_conn
        
        with patch('analytics.analytics_validator.AnalyticsValidator._get_table_row_count') as mock_count:
            mock_count.side_effect = RuntimeError("Unexpected error")
            
            result = validator._validate_table_quality("analytics.test_table")
            
            assert result['row_count'] == 0
            assert len(result['issues']) == 1
            assert result['issues'][0]['type'] == 'validation_error'
            assert result['issues'][0]['severity'] == 'critical'


class TestComprehensiveValidation:
    """Test the comprehensive validation method."""
    
    @pytest.fixture
    def validator(self, mock_db_adapter, mock_analytics_warehouse):
        """Create validator instance for testing."""
        return AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
    
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_data_quality')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_metrics')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_reports')
    @patch('analytics.analytics_validator.AnalyticsValidator.run_performance_benchmarks')
    def test_run_comprehensive_validation_excellent(self, mock_benchmarks, mock_reports,
                                                   mock_metrics, mock_data_quality, validator):
        """Test comprehensive validation with excellent health."""
        # Mock all validation methods to return no issues
        mock_data_quality.return_value = {'issues': []}
        mock_metrics.return_value = {'issues': []}
        mock_reports.return_value = {'issues': []}
        mock_benchmarks.return_value = {'summary': {'performance_rating': 'excellent'}}
        
        result = validator.run_comprehensive_validation()
        
        assert 'timestamp' in result
        assert result['overall_health'] == 'excellent'
        assert 'data_quality' in result
        assert 'metric_accuracy' in result
        assert 'report_accuracy' in result
        assert 'performance_benchmarks' in result
    
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_data_quality')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_metrics')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_reports')
    @patch('analytics.analytics_validator.AnalyticsValidator.run_performance_benchmarks')
    def test_run_comprehensive_validation_needs_attention(self, mock_benchmarks, mock_reports,
                                                         mock_metrics, mock_data_quality, validator):
        """Test comprehensive validation with issues needing attention."""
        # Mock validation methods to return many issues
        mock_data_quality.return_value = {'issues': [{'type': 'issue1'} for _ in range(5)]}
        mock_metrics.return_value = {'issues': [{'type': 'issue2'} for _ in range(6)]}
        mock_reports.return_value = {'issues': [{'type': 'issue3'} for _ in range(4)]}
        mock_benchmarks.return_value = {'summary': {'performance_rating': 'poor'}}
        
        result = validator.run_comprehensive_validation()
        
        assert result['overall_health'] == 'needs_attention'
        # Total issues: 5 + 6 + 4 = 15, which is > 10, so needs_attention
    
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_data_quality')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_metrics')
    @patch('analytics.analytics_validator.AnalyticsValidator.validate_reports')
    @patch('analytics.analytics_validator.AnalyticsValidator.run_performance_benchmarks')
    def test_run_comprehensive_validation_good_health(self, mock_benchmarks, mock_reports,
                                                     mock_metrics, mock_data_quality, validator):
        """Test comprehensive validation with good health."""
        # Mock validation methods to return few issues
        mock_data_quality.return_value = {'issues': [{'type': 'issue1'}]}
        mock_metrics.return_value = {'issues': [{'type': 'issue2'}]}
        mock_reports.return_value = {'issues': []}
        mock_benchmarks.return_value = {'summary': {'performance_rating': 'good'}}
        
        result = validator.run_comprehensive_validation()
        
        assert result['overall_health'] == 'good'
        # Total issues: 1 + 1 + 0 = 2, which is < 5, so good


class TestValidationResultSaving:
    """Test validation result saving functionality."""
    
    @pytest.fixture
    def validator(self, mock_db_adapter, mock_analytics_warehouse, temp_dir):
        """Create validator instance with temporary directory."""
        validator = AnalyticsValidator(mock_db_adapter, mock_analytics_warehouse)
        # Override results directory to use temp directory
        validator.results_dir = temp_dir / "validation_results"
        validator.results_dir.mkdir(parents=True, exist_ok=True)
        return validator
    
    def test_save_validation_results(self, validator):
        """Test saving validation results to file."""
        import json
        
        test_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_health': 'good',
            'data_quality': {'issues': []},
            'metric_accuracy': {'issues': []}
        }
        
        validator._save_validation_results(test_results)
        
        # Check that file was created
        result_files = list(validator.results_dir.glob("validation_*.json"))
        assert len(result_files) == 1
        
        # Check file contents
        with open(result_files[0], 'r') as f:
            saved_results = json.load(f)
        
        assert saved_results['overall_health'] == 'good'
        assert saved_results['timestamp'] == test_results['timestamp']


if __name__ == '__main__':
    pytest.main([__file__])