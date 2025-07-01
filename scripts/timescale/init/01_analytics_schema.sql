-- TimescaleDB Analytics Schema for Mimir News Scraper
-- Creates time-series optimized tables for analytics and business intelligence

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID extension for primary keys
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create analytics schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- Article metrics hypertable for time-series data
CREATE TABLE analytics.article_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    article_id UUID NOT NULL,
    source VARCHAR(255) NOT NULL,
    
    -- Content metrics
    title_length INTEGER,
    content_length INTEGER,
    word_count INTEGER,
    paragraph_count INTEGER,
    
    -- Engagement metrics (to be filled by external systems)
    views BIGINT DEFAULT 0,
    shares BIGINT DEFAULT 0,  
    comments BIGINT DEFAULT 0,
    
    -- AI-derived metrics
    sentiment_score DECIMAL(3,2), -- -1.0 to 1.0
    sentiment_confidence DECIMAL(3,2), -- 0.0 to 1.0
    readability_score DECIMAL(5,2),
    topic_confidence DECIMAL(3,2),
    
    -- Named entities count
    person_entities INTEGER DEFAULT 0,
    organization_entities INTEGER DEFAULT 0,
    location_entities INTEGER DEFAULT 0,
    misc_entities INTEGER DEFAULT 0,
    
    -- Keywords and tags
    keyword_count INTEGER DEFAULT 0,
    tag_count INTEGER DEFAULT 0,
    
    -- Metadata
    language VARCHAR(10),
    category VARCHAR(100),
    priority INTEGER DEFAULT 0,
    
    PRIMARY KEY (time, article_id)
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable('analytics.article_metrics', 'time');

-- Source metrics hypertable
CREATE TABLE analytics.source_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(255) NOT NULL,
    
    -- Volume metrics
    articles_scraped INTEGER DEFAULT 0,
    articles_published INTEGER DEFAULT 0,
    articles_failed INTEGER DEFAULT 0,
    
    -- Quality metrics
    avg_content_length DECIMAL(10,2),
    avg_sentiment DECIMAL(3,2),
    avg_readability DECIMAL(5,2),
    
    -- Performance metrics
    scrape_duration_ms INTEGER,
    success_rate DECIMAL(5,4), -- 0.0000 to 1.0000
    error_count INTEGER DEFAULT 0,
    
    -- Freshness metrics
    latest_article_age_hours DECIMAL(10,2),
    avg_article_age_hours DECIMAL(10,2),
    
    PRIMARY KEY (time, source)
);

SELECT create_hypertable('analytics.source_metrics', 'time');

-- Keyword trend metrics
CREATE TABLE analytics.keyword_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    keyword VARCHAR(255) NOT NULL,
    
    -- Occurrence metrics
    mention_count INTEGER DEFAULT 0,
    article_count INTEGER DEFAULT 0,
    
    -- Sentiment metrics for keyword
    avg_sentiment DECIMAL(3,2),
    sentiment_variance DECIMAL(5,4),
    
    -- Trend metrics
    growth_rate DECIMAL(8,4), -- percentage change
    momentum_score DECIMAL(5,2),
    
    -- Context metrics
    primary_sources TEXT[], -- JSON array of top sources
    related_keywords TEXT[], -- JSON array of related terms
    
    PRIMARY KEY (time, keyword)
);

SELECT create_hypertable('analytics.keyword_metrics', 'time');

-- User activity metrics (for multi-tenant analytics)
CREATE TABLE analytics.user_activity (
    id UUID DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID NOT NULL,
    organization_id UUID NOT NULL,
    
    -- Activity types
    action_type VARCHAR(50) NOT NULL, -- search, view, export, etc.
    resource_type VARCHAR(50), -- article, dashboard, report
    resource_id UUID,
    
    -- Session metrics
    session_id VARCHAR(255),
    duration_seconds INTEGER,
    
    -- Request metrics
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    response_time_ms INTEGER,
    
    -- Context
    user_agent TEXT,
    ip_address INET,
    referrer TEXT,
    
    PRIMARY KEY (time, user_id, action_type)
);

SELECT create_hypertable('analytics.user_activity', 'time');

-- System performance metrics
CREATE TABLE analytics.system_metrics (
    id UUID DEFAULT uuid_generate_v4(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- System resources
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb BIGINT,
    disk_usage_mb BIGINT,
    
    -- Database metrics
    active_connections INTEGER,
    queries_per_second DECIMAL(10,2),
    avg_query_time_ms DECIMAL(10,2),
    
    -- Scraper metrics
    active_scrapers INTEGER,
    articles_per_minute DECIMAL(10,2),
    error_rate_percent DECIMAL(5,2),
    
    -- AI processing metrics
    ai_queue_size INTEGER,
    ai_processing_time_ms DECIMAL(10,2),
    ai_success_rate DECIMAL(5,4),
    
    PRIMARY KEY (time)
);

SELECT create_hypertable('analytics.system_metrics', 'time');

-- Trend analysis results table
CREATE TABLE analytics.trend_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Analysis metadata
    analysis_type VARCHAR(50) NOT NULL, -- keyword_trend, source_trend, sentiment_trend
    subject VARCHAR(255) NOT NULL, -- the keyword, source, or topic being analyzed
    time_period_start TIMESTAMPTZ NOT NULL,
    time_period_end TIMESTAMPTZ NOT NULL,
    
    -- Trend detection results
    trend_direction VARCHAR(20), -- up, down, stable, volatile
    trend_strength DECIMAL(5,2), -- 0-100 scale
    trend_confidence DECIMAL(3,2), -- 0.0-1.0
    
    -- Statistical metrics
    slope DECIMAL(10,6),
    r_squared DECIMAL(5,4),
    p_value DECIMAL(10,8),
    
    -- Anomaly detection
    anomalies_detected INTEGER DEFAULT 0,
    anomaly_dates TIMESTAMPTZ[],
    anomaly_scores DECIMAL(5,2)[],
    
    -- Seasonality analysis
    has_seasonality BOOLEAN DEFAULT FALSE,
    seasonal_period_days INTEGER,
    seasonal_strength DECIMAL(3,2),
    
    -- Forecasting results
    next_period_forecast DECIMAL(10,2),
    forecast_confidence_interval DECIMAL(10,2)[2], -- [lower, upper]
    forecast_accuracy_score DECIMAL(3,2),
    
    -- Metadata
    analysis_duration_seconds INTEGER,
    model_used VARCHAR(100),
    parameters JSONB DEFAULT '{}'::jsonb
);

-- Indexes for optimal query performance
CREATE INDEX idx_article_metrics_source_time ON analytics.article_metrics (source, time DESC);
CREATE INDEX idx_article_metrics_sentiment ON analytics.article_metrics (time DESC) WHERE sentiment_score IS NOT NULL;
CREATE INDEX idx_source_metrics_success_rate ON analytics.source_metrics (time DESC, success_rate);
CREATE INDEX idx_keyword_metrics_growth ON analytics.keyword_metrics (time DESC, growth_rate DESC);
CREATE INDEX idx_user_activity_action ON analytics.user_activity (time DESC, action_type);
CREATE INDEX idx_user_activity_org ON analytics.user_activity (organization_id, time DESC);
CREATE INDEX idx_trend_analysis_type_subject ON analytics.trend_analysis (analysis_type, subject);
CREATE INDEX idx_trend_analysis_period ON analytics.trend_analysis (analysis_type, time_period_start, time_period_end);

-- Continuous aggregates for common time-based queries

-- Hourly article metrics rollup
CREATE MATERIALIZED VIEW analytics.article_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    source,
    COUNT(*) as article_count,
    AVG(sentiment_score) as avg_sentiment,
    AVG(content_length) as avg_content_length,
    AVG(word_count) as avg_word_count,
    SUM(views) as total_views,
    SUM(shares) as total_shares
FROM analytics.article_metrics 
WHERE time > NOW() - INTERVAL '1 month'
GROUP BY bucket, source
WITH NO DATA;

-- Daily source performance rollup
CREATE MATERIALIZED VIEW analytics.source_metrics_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    source,
    SUM(articles_scraped) as total_scraped,
    AVG(success_rate) as avg_success_rate,
    AVG(scrape_duration_ms) as avg_scrape_time,
    SUM(error_count) as total_errors
FROM analytics.source_metrics 
WHERE time > NOW() - INTERVAL '3 months'
GROUP BY bucket, source
WITH NO DATA;

-- Hourly keyword trends rollup
CREATE MATERIALIZED VIEW analytics.keyword_trends_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    keyword,
    SUM(mention_count) as total_mentions,
    COUNT(DISTINCT article_count) as unique_articles,
    AVG(avg_sentiment) as avg_sentiment,
    MAX(growth_rate) as max_growth_rate
FROM analytics.keyword_metrics 
WHERE time > NOW() - INTERVAL '7 days'
GROUP BY bucket, keyword
WITH NO DATA;

-- Daily user activity summary
CREATE MATERIALIZED VIEW analytics.user_activity_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS bucket,
    organization_id,
    action_type,
    COUNT(*) as action_count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(duration_seconds) as avg_duration,
    AVG(response_time_ms) as avg_response_time
FROM analytics.user_activity 
WHERE time > NOW() - INTERVAL '6 months'
GROUP BY bucket, organization_id, action_type
WITH NO DATA;

-- Enable continuous aggregate refresh policies
SELECT add_continuous_aggregate_policy('analytics.article_metrics_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes');

SELECT add_continuous_aggregate_policy('analytics.source_metrics_daily',
    start_offset => INTERVAL '7 days', 
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('analytics.keyword_trends_hourly',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour', 
    schedule_interval => INTERVAL '15 minutes');

SELECT add_continuous_aggregate_policy('analytics.user_activity_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '2 hours');

-- Data retention policies
SELECT add_retention_policy('analytics.article_metrics', INTERVAL '2 years');
SELECT add_retention_policy('analytics.source_metrics', INTERVAL '1 year');
SELECT add_retention_policy('analytics.keyword_metrics', INTERVAL '6 months');
SELECT add_retention_policy('analytics.user_activity', INTERVAL '2 years');
SELECT add_retention_policy('analytics.system_metrics', INTERVAL '3 months');

-- Grant permissions
GRANT USAGE ON SCHEMA analytics TO mimir_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO mimir_user;
GRANT SELECT ON ALL MATERIALIZED VIEWS IN SCHEMA analytics TO mimir_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA analytics TO mimir_user;

-- Insert initial system information
INSERT INTO analytics.system_metrics (
    cpu_usage_percent, memory_usage_mb, disk_usage_mb,
    active_connections, queries_per_second, avg_query_time_ms,
    active_scrapers, articles_per_minute, error_rate_percent,
    ai_queue_size, ai_processing_time_ms, ai_success_rate
) VALUES (
    0.0, 0, 0,
    0, 0.0, 0.0,
    0, 0.0, 0.0,
    0, 0.0, 1.0000
);