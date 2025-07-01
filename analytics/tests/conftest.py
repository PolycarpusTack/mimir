"""Pytest configuration and shared fixtures for analytics tests.

This module provides common fixtures and configuration for all analytics tests.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

import pandas as pd
import pytest
import numpy as np

# Set test environment
os.environ['TESTING'] = 'true'


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_article_data():
    """Sample article data for testing."""
    return [
        {
            'id': 1,
            'title': 'Test Article 1',
            'content': 'This is test content for article 1',
            'source': 'test_source_1',
            'published_date': datetime.utcnow() - timedelta(hours=1),
            'scraped_at': datetime.utcnow(),
            'sentiment_score': 0.5,
            'word_count': 100,
            'url': 'https://example.com/article1'
        },
        {
            'id': 2,
            'title': 'Test Article 2',
            'content': 'This is test content for article 2',
            'source': 'test_source_2',
            'published_date': datetime.utcnow() - timedelta(hours=2),
            'scraped_at': datetime.utcnow(),
            'sentiment_score': -0.3,
            'word_count': 150,
            'url': 'https://example.com/article2'
        },
        {
            'id': 3,
            'title': 'Test Article 3',
            'content': 'This is test content for article 3',
            'source': 'test_source_1',
            'published_date': datetime.utcnow() - timedelta(hours=3),
            'scraped_at': datetime.utcnow(),
            'sentiment_score': 0.8,
            'word_count': 200,
            'url': 'https://example.com/article3'
        }
    ]


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing."""
    base_time = datetime.utcnow() - timedelta(days=1)
    return [
        {
            'time': base_time + timedelta(hours=i),
            'article_count': 10 + i,
            'avg_sentiment': 0.1 * i,
            'source': 'test_source_1',
            'total_views': 100 + i * 10
        }
        for i in range(24)
    ]


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing trend analysis."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)  # For reproducible tests
    
    return pd.DataFrame({
        'time': dates,
        'value': np.random.randn(30).cumsum() + 100,
        'category': np.random.choice(['A', 'B', 'C'], 30),
        'count': np.random.poisson(10, 30)
    })


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    mock_conn = Mock()
    mock_cursor = Mock()
    
    # Set up cursor mock
    mock_cursor.fetchone.return_value = (100,)  # Default row count
    mock_cursor.fetchall.return_value = []
    mock_cursor.execute.return_value = None
    
    # Set up connection mock
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.rollback.return_value = None
    
    return mock_conn


@pytest.fixture
def mock_analytics_warehouse(mock_database_connection):
    """Mock analytics data warehouse."""
    from unittest.mock import patch
    
    with patch('analytics.data_warehouse.AnalyticsDataWarehouse') as mock_warehouse:
        instance = mock_warehouse.return_value
        instance.get_connection.return_value.__enter__.return_value = mock_database_connection
        instance.get_connection.return_value.__exit__.return_value = None
        yield instance


@pytest.fixture
def mock_db_adapter():
    """Mock database adapter."""
    from unittest.mock import patch
    
    with patch('db_adapter.DatabaseAdapter') as mock_adapter:
        instance = mock_adapter.return_value
        instance.get_all_articles.return_value = []
        instance.get_articles_by_source.return_value = []
        yield instance


@pytest.fixture
def sample_export_config():
    """Sample export configuration for testing."""
    from analytics.data_exporter import ExportConfig, ExportFormat
    
    return ExportConfig(
        format=ExportFormat.CSV,
        compression=False,
        include_metadata=True,
        filters={},
        columns=None
    )


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing exports."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'score': [85.5, 92.0, 78.5, 89.0, 91.5],
        'active': [True, False, True, True, False],
        'created_at': pd.date_range('2023-01-01', periods=5, freq='D')
    })


@pytest.fixture
def sample_clustering_data():
    """Sample data for clustering tests."""
    np.random.seed(42)
    
    # Create clusters with known structure
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    return pd.DataFrame({
        'feature1': data[:, 0],
        'feature2': data[:, 1],
        'article_id': range(150),
        'title': [f'Article {i}' for i in range(150)],
        'content': [f'Content for article {i}' * 10 for i in range(150)]
    })


@pytest.fixture
def sample_ml_model():
    """Sample ML model for testing."""
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    
    # Create simple model
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    
    return model, X, y


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    from unittest.mock import patch, MagicMock
    
    with patch('analytics.ml_infrastructure.mlflow') as mock_mlflow:
        mock_mlflow.start_run.return_value = MagicMock()
        mock_mlflow.log_metric.return_value = None
        mock_mlflow.log_param.return_value = None
        mock_mlflow.log_artifact.return_value = None
        mock_mlflow.end_run.return_value = None
        
        # Mock client
        mock_client = MagicMock()
        mock_client.create_experiment.return_value = "test_experiment_id"
        mock_client.list_experiments.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        
        yield mock_mlflow


@pytest.fixture
def sample_feature_data():
    """Sample feature data for testing feature engineering."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    return pd.DataFrame({
        'time': dates,
        'article_count': np.random.poisson(5, 100),
        'avg_sentiment': np.random.normal(0, 0.3, 100),
        'source': np.random.choice(['source1', 'source2', 'source3'], 100),
        'category': np.random.choice(['tech', 'politics', 'sports'], 100),
        'word_count': np.random.lognormal(5, 0.5, 100).astype(int)
    })


@pytest.fixture
def sample_recommendation_data():
    """Sample data for recommendation testing."""
    # User-item interaction matrix
    np.random.seed(42)
    
    users = [f'user_{i}' for i in range(20)]
    items = [f'article_{i}' for i in range(50)]
    
    interactions = []
    for user in users:
        # Each user interacts with 5-15 random articles
        n_interactions = np.random.randint(5, 16)
        user_items = np.random.choice(items, n_interactions, replace=False)
        
        for item in user_items:
            interactions.append({
                'user_id': user,
                'article_id': item,
                'interaction_type': 'view',
                'timestamp': datetime.utcnow() - timedelta(days=np.random.randint(0, 30)),
                'duration_seconds': np.random.randint(30, 300)
            })
    
    return pd.DataFrame(interactions)


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing optimization."""
    return {
        'cpu_usage': [45.2, 67.8, 23.1, 89.4, 34.6],
        'memory_usage': [62.1, 78.9, 41.3, 95.2, 58.7],
        'query_times': [0.123, 2.456, 0.089, 5.678, 1.234],
        'cache_hit_rates': [0.85, 0.92, 0.78, 0.67, 0.89],
        'timestamps': [
            datetime.utcnow() - timedelta(minutes=i*5) 
            for i in range(5)
        ]
    }


@pytest.fixture(autouse=True)
def clean_test_environment():
    """Clean up test environment before and after each test."""
    # Setup
    yield
    
    # Cleanup
    # Remove any test files, reset global state, etc.
    pass


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.ml = pytest.mark.ml


# Test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow tests")
    config.addinivalue_line("markers", "ml: marks tests as ML-related tests")


# Test helpers
class TestDataHelper:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_time_series(start_date: str, periods: int, freq: str = 'D', 
                          trend: float = 0.0, noise: float = 1.0) -> pd.DataFrame:
        """Create time series data with trend and noise."""
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        np.random.seed(42)
        
        trend_values = np.arange(periods) * trend
        noise_values = np.random.normal(0, noise, periods)
        values = trend_values + noise_values + 100
        
        return pd.DataFrame({
            'time': dates,
            'value': values,
            'trend': trend_values,
            'noise': noise_values
        })
    
    @staticmethod
    def create_article_metrics(n_articles: int = 100) -> List[Dict[str, Any]]:
        """Create sample article metrics data."""
        np.random.seed(42)
        
        metrics = []
        for i in range(n_articles):
            metrics.append({
                'time': datetime.utcnow() - timedelta(hours=i),
                'article_id': i + 1,
                'source': f'source_{np.random.randint(1, 6)}',
                'title_length': np.random.randint(20, 100),
                'content_length': np.random.randint(500, 5000),
                'word_count': np.random.randint(100, 1000),
                'sentiment_score': np.random.uniform(-1, 1),
                'readability_score': np.random.uniform(0, 100),
                'views': np.random.poisson(50),
                'shares': np.random.poisson(5),
                'comments': np.random.poisson(3)
            })
        
        return metrics


@pytest.fixture
def test_data_helper():
    """Test data helper fixture."""
    return TestDataHelper