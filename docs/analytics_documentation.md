# Mimir Analytics & Business Intelligence Documentation

## Overview

The Mimir Analytics & Business Intelligence platform provides comprehensive data analytics, machine learning, and reporting capabilities for the Mimir News Scraper. This document serves as the complete reference for all analytics features, metrics definitions, and usage instructions.

## Table of Contents

1. [Architecture](#architecture)
2. [Data Warehouse](#data-warehouse)
3. [Metrics Definitions](#metrics-definitions)
4. [Trend Analysis](#trend-analysis)
5. [Predictive Analytics](#predictive-analytics)
6. [Machine Learning Infrastructure](#machine-learning-infrastructure)
7. [Reporting System](#reporting-system)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Architecture

### System Components

The analytics platform consists of the following key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main DB       │────│   Analytics     │────│   Dashboard     │
│   (Articles)    │    │   Warehouse     │    │   Server        │
│                 │    │   (TimescaleDB) │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                                 │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  Metrics    │  │   Trend     │  │ Predictive  │  │    ML     │ │
│  │ Collector   │  │  Analyzer   │  │ Analytics   │  │Framework  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   Report    │  │    Data     │  │ Performance │  │   Cache   │ │
│  │ Generator   │  │  Exporter   │  │ Optimizer   │  │  System   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Collection**: Articles are scraped and stored in the main database
2. **Metrics Extraction**: The metrics collector processes articles and generates analytics metrics
3. **Storage**: Metrics are stored in TimescaleDB with automatic aggregation
4. **Analysis**: Trend analysis and predictive models process the time-series data
5. **Visualization**: Dashboard and reports present insights to users

## Data Warehouse

### TimescaleDB Schema

The analytics warehouse uses TimescaleDB (PostgreSQL extension) for time-series data:

#### Core Tables

**analytics.article_metrics**
- Stores article-level metrics aggregated by time windows
- Hypertable partitioned by time (1 day chunks)
- Includes: article counts, sentiment scores, source performance

**analytics.source_metrics**
- Source-specific performance metrics
- Tracks success rates, publication patterns, content quality

**analytics.keyword_metrics**
- Keyword/topic tracking and trending analysis
- Mention counts, growth rates, sentiment associations

**analytics.user_activity**
- User interaction tracking for recommendations
- View patterns, engagement metrics, feedback

**analytics.system_metrics**
- System performance and health monitoring
- Resource usage, processing rates, error counts

#### Continuous Aggregates

Pre-computed aggregations for common queries:

```sql
-- Hourly article volume aggregate
CREATE MATERIALIZED VIEW analytics.hourly_article_summary AS
SELECT 
    time_bucket('1 hour', time) as bucket,
    SUM(article_count) as total_articles,
    AVG(avg_sentiment) as avg_sentiment,
    COUNT(DISTINCT source) as unique_sources
FROM analytics.article_metrics
GROUP BY bucket;

-- Daily source performance aggregate
CREATE MATERIALIZED VIEW analytics.daily_source_performance AS
SELECT 
    time_bucket('1 day', time) as day,
    source,
    AVG(success_rate) as avg_success_rate,
    SUM(article_count) as total_articles
FROM analytics.source_metrics
GROUP BY day, source;
```

## Metrics Definitions

### Article Metrics

| Metric | Definition | Calculation | Range |
|--------|------------|-------------|-------|
| `article_count` | Number of articles processed | COUNT(*) per time bucket | 0-∞ |
| `avg_sentiment` | Average sentiment score | AVG(ai_analysis.sentiment.compound) | -1.0 to 1.0 |
| `unique_sources` | Distinct sources | COUNT(DISTINCT source) | 0-∞ |
| `avg_article_length` | Average content length | AVG(LENGTH(content)) | 0-∞ |
| `keywords_diversity` | Unique keywords count | COUNT(DISTINCT keyword) | 0-∞ |

### Source Metrics

| Metric | Definition | Calculation | Range |
|--------|------------|-------------|-------|
| `success_rate` | Scraping success percentage | successful_scrapes / total_attempts | 0.0-1.0 |
| `avg_response_time` | Average response time | AVG(response_time_ms) | 0-∞ |
| `article_frequency` | Articles per hour | articles / hours_active | 0-∞ |
| `content_quality_score` | Quality assessment | Weighted score based on length, structure | 0.0-1.0 |

### Keyword Metrics

| Metric | Definition | Calculation | Range |
|--------|------------|-------------|-------|
| `mention_count` | Keyword mentions | COUNT(keyword_occurrences) | 0-∞ |
| `growth_rate` | Mention growth rate | (current - previous) / previous | -∞ to ∞ |
| `sentiment_association` | Keyword sentiment | AVG(sentiment WHERE keyword present) | -1.0 to 1.0 |
| `source_diversity` | Sources mentioning keyword | COUNT(DISTINCT source) | 0-∞ |

### System Metrics

| Metric | Definition | Calculation | Range |
|--------|------------|-------------|-------|
| `cpu_usage_percent` | CPU utilization | psutil.cpu_percent() | 0.0-100.0 |
| `memory_usage_mb` | Memory consumption | psutil.memory_info().rss / 1024² | 0-∞ |
| `articles_per_minute` | Processing rate | articles_processed / elapsed_minutes | 0-∞ |
| `error_rate` | Error percentage | errors / total_operations | 0.0-1.0 |

## Trend Analysis

### Statistical Methods

#### Moving Averages
- **Simple Moving Average (SMA)**: Unweighted mean of previous n periods
- **Exponential Moving Average (EMA)**: Weighted average giving more importance to recent values
- **Usage**: Smoothing noisy data, identifying trends

#### Anomaly Detection
- **Z-Score Method**: Identifies outliers based on standard deviations
- **Isolation Forest**: ML-based anomaly detection for complex patterns
- **Prophet**: Facebook's time series forecasting with anomaly detection

#### Seasonality Analysis
- **Fourier Transform**: Frequency domain analysis for periodic patterns
- **Prophet Decomposition**: Trend, seasonal, and holiday components
- **Usage**: Understanding publication patterns, user activity cycles

#### Breakout Detection
- **Change Point Detection**: Identifies significant shifts in time series
- **CUSUM**: Cumulative sum algorithm for detecting mean shifts
- **Ruptures Library**: Multiple change point detection algorithms

### Trend Classification

| Trend Type | Definition | Indicators |
|------------|------------|------------|
| **Increasing** | Consistent upward movement | Positive slope, R² > 0.7 |
| **Decreasing** | Consistent downward movement | Negative slope, R² > 0.7 |
| **Stable** | Little to no change | Slope ≈ 0, low variance |
| **Volatile** | High variation, no clear direction | High coefficient of variation |
| **Cyclical** | Repeating patterns | Significant seasonal component |

## Predictive Analytics

### Forecasting Models

#### Prophet Model
- **Purpose**: Time series forecasting with trend and seasonality
- **Features**: Holiday effects, trend changepoints, uncertainty intervals
- **Best For**: Data with clear trends and seasonal patterns

```python
# Example Prophet forecast
forecast = predictive_analytics.forecast_article_volume(
    forecast_days=7,
    model_type='prophet'
)
```

#### Ensemble Models
- **Components**: Random Forest, Gradient Boosting, Linear Regression
- **Voting**: Weighted average based on historical performance
- **Best For**: Robust predictions across different data patterns

#### Trend-Based Forecasting
- **Method**: Linear regression on recent trends
- **Features**: Simple, fast, interpretable
- **Best For**: Short-term predictions with clear trends

### Prediction Types

#### Article Volume Forecasting
- **Metrics**: Daily/hourly article counts
- **Horizon**: 1-30 days ahead
- **Accuracy**: MAPE typically 15-25%

#### Topic Trend Prediction
- **Metrics**: Keyword mention growth
- **Features**: Historical patterns, external events
- **Output**: Trending keywords, emerging topics

#### Sentiment Forecasting
- **Metrics**: Aggregate sentiment scores
- **Uses**: Market sentiment analysis, content planning
- **Accuracy**: R² typically 0.6-0.8

### Alert Thresholds

#### Adaptive Learning
- **Method**: Historical percentiles with trend adjustment
- **Sensitivity Levels**:
  - **Low**: 95th percentile threshold
  - **Medium**: 90th percentile threshold
  - **High**: 80th percentile threshold

#### Alert Types
- **Volume Alerts**: Unusual article publication rates
- **Sentiment Alerts**: Significant sentiment shifts
- **Source Alerts**: Source performance degradation
- **System Alerts**: Resource usage or error rate spikes

## Machine Learning Infrastructure

### Model Registry

The ML infrastructure provides versioned model storage and management:

#### Model Types Supported
- **scikit-learn**: Traditional ML models
- **PyTorch**: Deep learning models
- **TensorFlow**: Neural networks
- **Custom**: Pickle-serializable models

#### Model Lifecycle
1. **Development**: Train and validate models
2. **Registration**: Store in registry with metadata
3. **Staging**: Test in production-like environment
4. **Production**: Deploy for real-time predictions
5. **Monitoring**: Track performance and drift

### Feature Store

Centralized feature management for ML models:

#### Feature Categories
- **Time-series Features**: Rolling statistics, lags, differences
- **Text Features**: TF-IDF, embeddings, sentiment scores
- **Aggregate Features**: Source statistics, keyword metrics
- **Behavioral Features**: User interaction patterns

#### Feature Engineering Pipeline
```python
# Example feature computation
feature_eng = FeatureEngineering()
features = feature_eng.compute_article_volume_features(
    df, 
    window_sizes=[1, 6, 12, 24]  # hours
)
```

### A/B Testing Framework

Production model comparison and optimization:

#### Test Configuration
- **Traffic Split**: Percentage routing to each model
- **Duration**: Test runtime and sample size requirements
- **Metrics**: Success criteria and statistical significance

#### Evaluation Methods
- **Statistical Tests**: t-tests, chi-square tests
- **Business Metrics**: Accuracy, precision, recall, user engagement
- **Performance Metrics**: Latency, resource usage

### Recommendation Engine

#### Collaborative Filtering
- **Method**: Matrix factorization (ALS, BPR)
- **Data**: User-article interaction matrix
- **Output**: Personalized article recommendations

#### Content-Based Filtering
- **Method**: Cosine similarity on article features
- **Features**: TF-IDF vectors, embeddings, metadata
- **Output**: Similar articles, topic-based recommendations

#### Hybrid Approach
- **Combination**: Weighted ensemble of collaborative and content methods
- **Weights**: Learned from user feedback and A/B tests
- **Fallback**: Content-based for new users, trending for cold start

## Reporting System

### Report Templates

#### Executive Summary
- **Purpose**: High-level overview for management
- **Metrics**: Key performance indicators, trends, alerts
- **Frequency**: Daily, weekly, monthly
- **Format**: PDF, HTML with executive-friendly visualizations

#### Source Performance Report
- **Purpose**: Detailed source analysis
- **Metrics**: Success rates, article volumes, quality scores
- **Comparison**: Period-over-period analysis
- **Actionable Insights**: Underperforming sources, optimization opportunities

#### Keyword Trends Report
- **Purpose**: Content and topic analysis
- **Metrics**: Trending keywords, sentiment analysis, topic evolution
- **Visualizations**: Word clouds, trend charts, correlation matrices
- **Use Cases**: Content strategy, market research

#### Technical Performance Report
- **Purpose**: System health and performance monitoring
- **Metrics**: Resource usage, processing rates, error rates
- **Alerts**: Threshold breaches, anomalies
- **Recommendations**: Optimization suggestions, capacity planning

### Report Generation

#### Template Engine
- **Technology**: Jinja2 templating with custom filters
- **Data Binding**: Automatic metric population from analytics warehouse
- **Styling**: CSS styling for consistent branding

#### Output Formats
- **PDF**: Professional reports with charts and tables
- **HTML**: Interactive reports with embedded visualizations
- **JSON**: Structured data for API consumption
- **Excel**: Tabular data with multiple sheets

#### Scheduling System
- **Triggers**: Time-based (cron-like), event-based
- **Delivery**: Email, file system, webhooks, cloud storage
- **Monitoring**: Success/failure tracking, retry logic

### Custom Report Builder

#### Interactive Interface
- **Drag-and-Drop**: Visual report composition
- **Widget Library**: Charts, tables, metrics cards, text blocks
- **Data Sources**: Pre-defined queries and custom SQL support

#### Parameterization
- **Date Ranges**: Flexible time period selection
- **Filters**: Source, keyword, user segment filters
- **Aggregation Levels**: Hourly, daily, weekly, monthly

## Performance Optimization

### Query Optimization

#### Indexing Strategy
```sql
-- Time-based queries
CREATE INDEX idx_article_metrics_time ON analytics.article_metrics (time);

-- Source filtering
CREATE INDEX idx_article_metrics_source_time ON analytics.article_metrics (source, time);

-- Keyword lookups
CREATE INDEX idx_keyword_metrics_keyword_time ON analytics.keyword_metrics (keyword, time);
```

#### Query Patterns
- **Time Range Queries**: Always include time bounds
- **Aggregations**: Use continuous aggregates when possible
- **Joins**: Minimize cross-table joins, use appropriate indexes

### Caching Strategy

#### Multi-Level Caching
1. **Query Cache**: SQL query results with TTL
2. **Result Cache**: Processed analytics results
3. **Visualization Cache**: Chart data and configurations

#### Cache Configuration
```python
# Cache settings
query_cache = QueryCache(max_size=1000, default_ttl=300)  # 5 minutes
result_cache = QueryCache(max_size=500, default_ttl=600)  # 10 minutes
viz_cache = QueryCache(max_size=200, default_ttl=180)     # 3 minutes
```

#### Cache Invalidation
- **Time-based**: TTL expiration
- **Event-based**: Data updates trigger invalidation
- **Manual**: Administrative cache clearing

### Resource Monitoring

#### System Metrics
- **CPU Usage**: Process and system-level monitoring
- **Memory Usage**: Heap, cache, and buffer monitoring
- **Disk I/O**: Database and file system performance
- **Network**: Database connection and API response times

#### Performance Thresholds
```python
optimization_settings = {
    'max_query_time': 30.0,      # seconds
    'memory_threshold': 0.8,      # 80% usage
    'cpu_threshold': 0.9,         # 90% usage
    'cache_hit_rate_min': 0.7     # 70% minimum
}
```

### Optimization Recommendations

#### Automated Suggestions
- **Slow Query Detection**: Identify queries exceeding time thresholds
- **Index Recommendations**: Suggest indexes based on query patterns
- **Cache Tuning**: Adjust cache sizes and TTLs based on hit rates
- **Resource Scaling**: Recommend capacity increases based on usage patterns

## API Reference

### Analytics Manager CLI

#### Basic Commands
```bash
# System status and health
python analytics_manager.py status

# Data collection
python analytics_manager.py collect

# Trend analysis
python analytics_manager.py analyze --days 30

# Forecasting
python analytics_manager.py forecast --type volume --days 7
```

#### ML Commands
```bash
# Model management
python analytics_manager.py ml models list
python analytics_manager.py ml models promote --model-name forecast_model --version 1.0.0

# Feature store
python analytics_manager.py ml features compute
python analytics_manager.py ml features list

# A/B testing
python analytics_manager.py ml abtest list
```

#### Export Commands
```bash
# Data export
python analytics_manager.py export article_metrics --format parquet
python analytics_manager.py export trending_keywords --format json --compress
```

### Dashboard API

#### Metrics Endpoints
```
GET /api/metrics/overview
GET /api/charts/volume_trend?hours=24
GET /api/charts/source_performance?days=7
GET /api/forecast/volume?days=7
```

#### Response Format
```json
{
  "status": "success",
  "data": {
    "metrics": {...},
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "cache_info": {
    "cached": true,
    "cache_age": 120
  }
}
```

### Python API

#### Direct Usage
```python
from analytics import AnalyticsDataWarehouse, TrendAnalyzer

# Initialize components
warehouse = AnalyticsDataWarehouse()
analyzer = TrendAnalyzer(warehouse)

# Analyze trends
trends = analyzer.analyze_article_volume_trends(days_back=30)

# Generate forecasts
forecast = analyzer.forecast_article_volume(forecast_days=7)
```

## Troubleshooting

### Common Issues

#### Data Quality Problems

**Symptom**: Missing or inconsistent metrics
**Causes**: 
- Network issues during scraping
- Article parsing failures
- Database connection problems

**Solutions**:
1. Check scraper logs for errors
2. Validate data pipeline integrity
3. Run data quality validation
```bash
python analytics_manager.py validate
```

#### Performance Issues

**Symptom**: Slow dashboard loading, query timeouts
**Causes**:
- Missing database indexes
- Inefficient queries
- Resource constraints

**Solutions**:
1. Run performance analysis
```bash
python analytics_manager.py analyze-performance
```
2. Apply optimizations
```bash
python analytics_manager.py optimize --apply-indexes
```
3. Monitor resource usage
```bash
python analytics_manager.py monitor-resources
```

#### Prediction Accuracy Issues

**Symptom**: Poor forecast accuracy, unexpected predictions
**Causes**:
- Insufficient training data
- Data distribution changes
- Model drift

**Solutions**:
1. Validate model performance
2. Retrain with recent data
3. Check for data distribution shifts
4. Consider ensemble methods

### Monitoring and Alerts

#### Health Checks
- **Database Connectivity**: Connection pool status
- **Data Freshness**: Recent metric availability
- **System Resources**: CPU, memory, disk usage
- **Cache Performance**: Hit rates and response times

#### Alert Configuration
```python
# Example alert thresholds
alerts = {
    'slow_query_threshold': 30.0,     # seconds
    'memory_usage_threshold': 85.0,   # percent
    'cache_hit_rate_threshold': 60.0, # percent
    'data_staleness_threshold': 2.0   # hours
}
```

### Log Analysis

#### Log Locations
- **Analytics Logs**: `analytics.log`
- **Dashboard Logs**: `dashboard.log`
- **Performance Logs**: `performance.log`
- **Error Logs**: `errors.log`

#### Common Log Patterns
```bash
# Find slow queries
grep "Slow query" analytics.log

# Check cache performance
grep "Cache hit rate" performance.log

# Monitor errors
tail -f errors.log | grep ERROR
```

### Database Maintenance

#### Regular Maintenance Tasks
```sql
-- Update table statistics
ANALYZE analytics.article_metrics;

-- Vacuum to reclaim space
VACUUM analytics.article_metrics;

-- Refresh continuous aggregates
CALL refresh_continuous_aggregate('hourly_article_summary', NULL, NULL);
```

#### Backup and Recovery
```bash
# Backup analytics database
pg_dump -h localhost -p 5433 analytics > analytics_backup.sql

# Restore from backup
psql -h localhost -p 5433 analytics < analytics_backup.sql
```

---

## Support and Resources

### Documentation Links
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Contact Information
- **Technical Issues**: See GitHub issues
- **Feature Requests**: Submit enhancement requests
- **Documentation**: Contribute to docs/

### Version History
- **v1.0.0**: Initial analytics platform
- **v1.1.0**: Added ML infrastructure
- **v1.2.0**: Performance optimizations
- **v1.3.0**: Enhanced reporting system

---

*This documentation is automatically updated with each release. Last updated: 2024-01-15*