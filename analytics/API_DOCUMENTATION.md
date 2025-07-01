# Mimir Analytics API Documentation

## Overview

The Mimir Analytics system provides comprehensive data analysis capabilities for news scraping and content analytics. This document details all available APIs, classes, and methods across the analytics modules.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Data Warehouse](#data-warehouse)
3. [ML Infrastructure](#ml-infrastructure)
4. [Analytics Validator](#analytics-validator)
5. [Data Exporter](#data-exporter)
6. [Article Clustering](#article-clustering)
7. [Async Operations](#async-operations)
8. [Configuration](#configuration)
9. [Utilities](#utilities)
10. [Exceptions](#exceptions)

---

## Core Modules

### AnalyticsDataWarehouse

Primary data warehouse interface for analytics operations.

#### Constructor

```python
AnalyticsDataWarehouse(connection_params: Dict[str, Any])
```

**Parameters:**
- `connection_params`: Database connection configuration

#### Methods

##### `get_connection() -> psycopg2.connection`
Returns a database connection from the pool.

**Returns:** Database connection object

**Example:**
```python
warehouse = AnalyticsDataWarehouse(config)
with warehouse.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM articles")
```

##### `execute_query(query: str, params: List[Any] = None) -> List[Tuple]`
Execute a parameterized query and return results.

**Parameters:**
- `query`: SQL query string
- `params`: Query parameters (optional)

**Returns:** List of result tuples

**Raises:**
- `QueryExecutionException`: When query execution fails
- `DatabaseConnectionException`: When connection fails

##### `get_article_metrics(start_time: datetime, end_time: datetime) -> pd.DataFrame`
Retrieve article metrics for specified time range.

**Parameters:**
- `start_time`: Start of time range
- `end_time`: End of time range

**Returns:** DataFrame with article metrics

##### `get_source_performance(days_back: int = 30) -> Dict[str, Any]`
Get performance metrics for all sources.

**Parameters:**
- `days_back`: Number of days to analyze (default: 30)

**Returns:** Dictionary with source performance data

---

## ML Infrastructure

### MLModelManager

Manages machine learning models with MLflow integration.

#### Constructor

```python
MLModelManager(tracking_uri: str = None, experiment_name: str = "mimir_analytics")
```

**Parameters:**
- `tracking_uri`: MLflow tracking server URI (optional)
- `experiment_name`: Name of MLflow experiment

#### Methods

##### `train_sentiment_model(articles_df: pd.DataFrame) -> Dict[str, Any]`
Train a sentiment analysis model.

**Parameters:**
- `articles_df`: DataFrame with article data

**Returns:** Training results with model metrics

**Example:**
```python
manager = MLModelManager()
results = manager.train_sentiment_model(articles_df)
print(f"Model accuracy: {results['metrics']['accuracy']}")
```

##### `load_model(model_name: str, version: str = "latest") -> Any`
Load a trained model from storage.

**Parameters:**
- `model_name`: Name of the model to load
- `version`: Model version (default: "latest")

**Returns:** Loaded model object

**Raises:**
- `ModelLoadException`: When model loading fails

##### `predict_batch(model_name: str, data: pd.DataFrame) -> np.ndarray`
Make batch predictions using a trained model.

**Parameters:**
- `model_name`: Name of the model to use
- `data`: Input data for prediction

**Returns:** Array of predictions

---

## Analytics Validator

### AnalyticsValidator

Validates data quality and system performance.

#### Constructor

```python
AnalyticsValidator(db_adapter: DatabaseAdapter, analytics_warehouse: AnalyticsDataWarehouse)
```

#### Methods

##### `validate_data_quality() -> Dict[str, Any]`
Comprehensive data quality validation.

**Returns:** Validation results with issues and metrics

**Example:**
```python
validator = AnalyticsValidator(db_adapter, warehouse)
results = validator.validate_data_quality()
if results['overall_health'] == 'excellent':
    print("Data quality is excellent!")
```

##### `run_comprehensive_validation() -> Dict[str, Any]`
Run complete system validation including performance benchmarks.

**Returns:** Full validation report

##### `validate_metrics() -> Dict[str, Any]`
Validate metric calculation accuracy.

**Returns:** Metric validation results

##### `run_performance_benchmarks() -> Dict[str, Any]`
Execute performance benchmarks on key operations.

**Returns:** Performance benchmark results

---

## Data Exporter

### DataExporter

Exports analytics data in various formats.

#### Constructor

```python
DataExporter(analytics_warehouse: AnalyticsDataWarehouse, output_dir: Path = None)
```

#### Methods

##### `export_articles(config: ExportConfig) -> Dict[str, Any]`
Export article data with specified configuration.

**Parameters:**
- `config`: Export configuration object

**Returns:** Export results with file paths and metrics

**Example:**
```python
config = ExportConfig(
    format=ExportFormat.CSV,
    compression=True,
    filters={'source': 'tech_news'}
)
results = exporter.export_articles(config)
```

##### `export_metrics(metric_type: str, format: ExportFormat) -> str`
Export specific metrics in requested format.

**Parameters:**
- `metric_type`: Type of metrics to export
- `format`: Output format

**Returns:** Path to exported file

##### `create_scheduled_export(config: Dict[str, Any]) -> str`
Create a scheduled export job.

**Parameters:**
- `config`: Export job configuration

**Returns:** Job ID for tracking

---

## Article Clustering

### ArticleClusterer

Provides advanced clustering capabilities for articles.

#### Constructor

```python
ArticleClusterer(db_adapter: DatabaseAdapter, analytics_warehouse: AnalyticsDataWarehouse)
```

#### Methods

##### `cluster_articles_by_topic(start_date: datetime = None, end_date: datetime = None, n_clusters: int = None, method: str = 'kmeans', min_cluster_size: int = 5) -> Dict[str, Any]`
Cluster articles by topic using various algorithms.

**Parameters:**
- `start_date`: Start date for articles (optional)
- `end_date`: End date for articles (optional)  
- `n_clusters`: Number of clusters (auto-detected if None)
- `method`: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan', 'hierarchical')
- `min_cluster_size`: Minimum size for clusters

**Returns:** Clustering results with assignments and metrics

**Example:**
```python
clusterer = ArticleClusterer(db_adapter, warehouse)
results = clusterer.cluster_articles_by_topic(
    method='dbscan',
    min_cluster_size=10
)
print(f"Found {results['n_clusters']} topic clusters")
```

##### `cluster_by_source_behavior(days_back: int = 30, features: List[str] = None) -> Dict[str, Any]`
Cluster sources based on publishing behavior patterns.

**Parameters:**
- `days_back`: Number of days to analyze
- `features`: List of behavioral features to use

**Returns:** Source clustering results

##### `create_hierarchical_topic_tree(start_date: datetime = None, end_date: datetime = None, max_depth: int = 4) -> Dict[str, Any]`
Create hierarchical topic structure using agglomerative clustering.

**Parameters:**
- `start_date`: Start date for articles (optional)
- `end_date`: End date for articles (optional)
- `max_depth`: Maximum tree depth

**Returns:** Hierarchical topic tree structure

---

## Async Operations

### AsyncAnalyticsProcessor

Asynchronous processor for high-performance analytics operations.

#### Constructor

```python
AsyncAnalyticsProcessor(analytics_warehouse: AnalyticsDataWarehouse, max_workers: int = 4)
```

#### Methods

##### `async collect_metrics_async(sources: List[str], time_range: Dict[str, datetime]) -> Dict[str, Any]`
Collect metrics from multiple sources asynchronously.

**Parameters:**
- `sources`: List of source names
- `time_range`: Dictionary with 'start' and 'end' datetime

**Returns:** Collected metrics for all sources

**Example:**
```python
async with AsyncAnalyticsContext(warehouse) as processor:
    results = await processor.collect_metrics_async(
        sources=['source1', 'source2'],
        time_range={'start': yesterday, 'end': today}
    )
```

##### `async analyze_trends_async(metric_type: str, days_back: int = 30) -> Dict[str, Any]`
Analyze trends asynchronously with caching.

**Parameters:**
- `metric_type`: Type of metric to analyze
- `days_back`: Number of days to analyze

**Returns:** Trend analysis results

##### `async generate_reports_async(report_configs: List[Dict[str, Any]]) -> Dict[str, Any]`
Generate multiple reports asynchronously.

**Parameters:**
- `report_configs`: List of report configurations

**Returns:** Report generation results

---

## Configuration

### AnalyticsConfig

Centralized configuration management using dataclasses.

#### Structure

```python
@dataclass
class AnalyticsConfig:
    database: DatabaseConfig
    cache: CacheConfig
    ml: MLConfig
    security: SecurityConfig
    performance: PerformanceConfig
    logging: LoggingConfig
```

#### Loading Configuration

```python
# From environment variables
config = AnalyticsConfig.from_env()

# From file
config = AnalyticsConfig.from_file('config.yaml')

# Validate configuration
errors = config.validate()
if errors:
    raise ConfigurationError(f"Invalid config: {errors}")
```

---

## Utilities

### Data Validation

##### `validate_dataframe(df: pd.DataFrame, min_rows: int = 1, required_columns: List[str] = None) -> pd.DataFrame`
Validate DataFrame structure and content.

**Parameters:**
- `df`: DataFrame to validate
- `min_rows`: Minimum required rows
- `required_columns`: List of required column names

**Returns:** Validated DataFrame

**Raises:**
- `DataValidationException`: When validation fails

### Mathematical Operations

##### `safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float`
Perform safe division with zero-handling.

##### `calculate_growth_rate(current: float, previous: float) -> float`
Calculate percentage growth rate between two values.

##### `normalize_values(values: List[float], method: str = 'standard') -> List[float]`
Normalize values using specified method.

### Data Processing

##### `clean_numeric_data(data: List[Any], default_value: float = 0.0) -> List[float]`
Clean and convert data to numeric format.

##### `batch_process(items: List[Any], batch_size: int = 10) -> List[List[Any]]`
Split items into batches for processing.

##### `detect_outliers(values: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[bool]`
Detect outliers using statistical methods.

### Decorators

##### `@retry_on_exception(max_retries: int = 3, delay: float = 1.0)`
Decorator for automatic retry on exceptions.

##### `@performance_timer`
Decorator for timing function execution.

##### `@async_cached(ttl: int = 300)`
Decorator for async function caching with TTL.

---

## Exceptions

### Custom Exception Hierarchy

#### AnalyticsBaseException
Base exception for all analytics-related errors.

```python
class AnalyticsBaseException(Exception):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
```

#### Database Exceptions
- `DatabaseConnectionException`: Database connection failures
- `QueryExecutionException`: SQL query execution errors
- `TransactionException`: Database transaction errors

#### Data Exceptions
- `DataValidationException`: Data validation failures
- `DataProcessingException`: Data processing errors
- `DataExportException`: Export operation failures

#### ML Exceptions
- `ModelTrainingException`: Model training failures
- `ModelLoadException`: Model loading errors
- `PredictionException`: Prediction failures

#### Performance Exceptions
- `PerformanceException`: Performance-related issues
- `TimeoutException`: Operation timeout errors

---

## Usage Examples

### Basic Analytics Workflow

```python
from analytics import AnalyticsDataWarehouse, AnalyticsValidator, DataExporter
from analytics.config import AnalyticsConfig

# Load configuration
config = AnalyticsConfig.from_env()

# Initialize components
warehouse = AnalyticsDataWarehouse(config.database)
validator = AnalyticsValidator(db_adapter, warehouse)
exporter = DataExporter(warehouse)

# Validate data quality
validation_results = validator.run_comprehensive_validation()
if validation_results['overall_health'] != 'excellent':
    print("Data quality issues detected!")

# Export data
export_config = ExportConfig(format=ExportFormat.JSON, compression=True)
export_results = exporter.export_articles(export_config)
```

### Async Analytics Pipeline

```python
import asyncio
from analytics.async_operations import run_analytics_pipeline

async def main():
    pipeline_config = {
        'metrics_collection': {
            'sources': ['source1', 'source2'],
            'time_range': {'start': yesterday, 'end': today}
        },
        'trend_analysis': {
            'metric_type': 'article_volume',
            'days_back': 30
        },
        'report_generation': {
            'reports': [
                {'type': 'summary', 'time_period_days': 7},
                {'type': 'detailed', 'time_period_days': 30}
            ]
        }
    }
    
    results = await run_analytics_pipeline(warehouse, pipeline_config)
    print(f"Pipeline completed in {results['total_execution_time']:.2f} seconds")

asyncio.run(main())
```

### ML Model Training

```python
from analytics.ml_infrastructure import MLModelManager

# Initialize ML manager
ml_manager = MLModelManager(experiment_name="sentiment_analysis")

# Train model
training_results = ml_manager.train_sentiment_model(articles_df)

# Load and use model
model = ml_manager.load_model("sentiment_classifier")
predictions = ml_manager.predict_batch("sentiment_classifier", new_articles_df)
```

---

## Performance Guidelines

### Best Practices

1. **Connection Management**: Always use context managers for database connections
2. **Batch Operations**: Use batch processing for large datasets
3. **Async Operations**: Use async methods for I/O intensive operations
4. **Caching**: Leverage built-in caching for repeated operations
5. **Memory Management**: Monitor memory usage with large datasets

### Performance Tuning

1. **Database Queries**: Use parameterized queries and appropriate indexes
2. **Feature Generation**: Cache embeddings and TF-IDF matrices
3. **Clustering**: Limit dataset size for hierarchical clustering
4. **Exports**: Use compression for large exports
5. **Monitoring**: Use performance decorators to identify bottlenecks

---

## Error Handling

All API methods include comprehensive error handling with specific exceptions:

```python
try:
    results = clusterer.cluster_articles_by_topic()
except DataValidationException as e:
    print(f"Data validation failed: {e.message}")
    print(f"Details: {e.details}")
except PerformanceException as e:
    print(f"Performance issue: {e.message}")
except AnalyticsBaseException as e:
    print(f"Analytics error: {e.message}")
```

---

## Version Information

- **Version**: 2.0.0
- **Last Updated**: 2024-12-19
- **Python Compatibility**: 3.8+
- **Dependencies**: See requirements.txt

---

## Support

For issues and questions:
- Check the troubleshooting section in the main documentation
- Review exception details and logs
- Consult the performance guidelines for optimization tips