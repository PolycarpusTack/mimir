"""TimescaleDB Data Warehouse for Analytics.

This module provides the core data warehouse functionality for storing and
retrieving time-series analytics data using TimescaleDB.
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from psycopg2 import pool, sql
from psycopg2 import OperationalError, DatabaseError, IntegrityError

from .exceptions import (
    DatabaseConnectionException, QueryExecutionException,
    DataWarehouseException
)

logger = logging.getLogger(__name__)

# TimescaleDB configuration
TIMESCALE_CONFIG = {
    "host": os.getenv("TIMESCALE_HOST", "localhost"),
    "port": int(os.getenv("TIMESCALE_PORT", "5433")),
    "database": os.getenv("TIMESCALE_DB", "mimir_analytics"),
    "user": os.getenv("TIMESCALE_USER", "mimir_user"),
    "password": os.getenv("TIMESCALE_PASSWORD", "mimir_dev_password"),
}

POOL_MIN_CONN = int(os.getenv("TIMESCALE_POOL_MIN", "2"))
POOL_MAX_CONN = int(os.getenv("TIMESCALE_POOL_MAX", "10"))


class AnalyticsDataWarehouse:
    """TimescaleDB-based data warehouse for analytics data."""

    def __init__(self):
        """Initialize the analytics data warehouse."""
        self._pool = None
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the database connection pool."""
        try:
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                POOL_MIN_CONN,
                POOL_MAX_CONN,
                **TIMESCALE_CONFIG
            )
            logger.info("TimescaleDB connection pool initialized successfully")
        except OperationalError as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise DatabaseConnectionException(f"Database connection failed: {e}", {'config': TIMESCALE_CONFIG})
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB pool: {e}")
            raise DataWarehouseException(f"Pool initialization failed: {e}")

    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except (OperationalError, DatabaseError) as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise QueryExecutionException(f"Database operation failed: {e}")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Unexpected database error: {e}")
            raise DataWarehouseException(f"Unexpected database error: {e}")
        finally:
            if conn:
                self._pool.putconn(conn)

    def store_article_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store article metrics in batch.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        if not metrics:
            return True

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Prepare insert statement
                    insert_query = sql.SQL("""
                        INSERT INTO analytics.article_metrics (
                            time, article_id, source, title_length, content_length,
                            word_count, paragraph_count, views, shares, comments,
                            sentiment_score, sentiment_confidence, readability_score,
                            topic_confidence, person_entities, organization_entities,
                            location_entities, misc_entities, keyword_count, tag_count,
                            language, category, priority
                        ) VALUES %s
                        ON CONFLICT (time, article_id) DO UPDATE SET
                            views = EXCLUDED.views,
                            shares = EXCLUDED.shares,
                            comments = EXCLUDED.comments,
                            sentiment_score = COALESCE(EXCLUDED.sentiment_score, analytics.article_metrics.sentiment_score),
                            sentiment_confidence = COALESCE(EXCLUDED.sentiment_confidence, analytics.article_metrics.sentiment_confidence)
                    """)

                    # Prepare data tuples
                    data = []
                    for metric in metrics:
                        data.append((
                            metric.get('time', datetime.utcnow()),
                            metric['article_id'],
                            metric['source'],
                            metric.get('title_length'),
                            metric.get('content_length'),
                            metric.get('word_count'),
                            metric.get('paragraph_count'),
                            metric.get('views', 0),
                            metric.get('shares', 0),
                            metric.get('comments', 0),
                            metric.get('sentiment_score'),
                            metric.get('sentiment_confidence'),
                            metric.get('readability_score'),
                            metric.get('topic_confidence'),
                            metric.get('person_entities', 0),
                            metric.get('organization_entities', 0),
                            metric.get('location_entities', 0),
                            metric.get('misc_entities', 0),
                            metric.get('keyword_count', 0),
                            metric.get('tag_count', 0),
                            metric.get('language'),
                            metric.get('category'),
                            metric.get('priority', 0)
                        ))

                    # Execute batch insert
                    psycopg2.extras.execute_values(
                        cursor, insert_query, data, page_size=1000
                    )
                    conn.commit()
                    
                    logger.info(f"Stored {len(metrics)} article metrics")
                    return True

        except (DatabaseError, IntegrityError) as e:
            logger.error(f"Failed to store article metrics: {e}")
            raise QueryExecutionException(f"Failed to store article metrics: {e}")
        except Exception as e:
            logger.error(f"Unexpected error storing article metrics: {e}")
            raise DataWarehouseException(f"Failed to store article metrics: {e}")

    def store_source_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store source performance metrics."""
        if not metrics:
            return True

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = sql.SQL("""
                        INSERT INTO analytics.source_metrics (
                            time, source, articles_scraped, articles_published,
                            articles_failed, avg_content_length, avg_sentiment,
                            avg_readability, scrape_duration_ms, success_rate,
                            error_count, latest_article_age_hours, avg_article_age_hours
                        ) VALUES %s
                        ON CONFLICT (time, source) DO UPDATE SET
                            articles_scraped = EXCLUDED.articles_scraped,
                            articles_published = EXCLUDED.articles_published,
                            articles_failed = EXCLUDED.articles_failed,
                            success_rate = EXCLUDED.success_rate,
                            error_count = EXCLUDED.error_count
                    """)

                    data = []
                    for metric in metrics:
                        data.append((
                            metric.get('time', datetime.utcnow()),
                            metric['source'],
                            metric.get('articles_scraped', 0),
                            metric.get('articles_published', 0),
                            metric.get('articles_failed', 0),
                            metric.get('avg_content_length'),
                            metric.get('avg_sentiment'),
                            metric.get('avg_readability'),
                            metric.get('scrape_duration_ms'),
                            metric.get('success_rate', 1.0),
                            metric.get('error_count', 0),
                            metric.get('latest_article_age_hours'),
                            metric.get('avg_article_age_hours')
                        ))

                    psycopg2.extras.execute_values(
                        cursor, insert_query, data, page_size=1000
                    )
                    conn.commit()
                    
                    logger.info(f"Stored {len(metrics)} source metrics")
                    return True

        except Exception as e:
            logger.error(f"Failed to store source metrics: {e}")
            return False

    def store_keyword_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """Store keyword trend metrics."""
        if not metrics:
            return True

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = sql.SQL("""
                        INSERT INTO analytics.keyword_metrics (
                            time, keyword, mention_count, article_count,
                            avg_sentiment, sentiment_variance, growth_rate,
                            momentum_score, primary_sources, related_keywords
                        ) VALUES %s
                        ON CONFLICT (time, keyword) DO UPDATE SET
                            mention_count = EXCLUDED.mention_count,
                            article_count = EXCLUDED.article_count,
                            avg_sentiment = EXCLUDED.avg_sentiment,
                            growth_rate = EXCLUDED.growth_rate,
                            momentum_score = EXCLUDED.momentum_score
                    """)

                    data = []
                    for metric in metrics:
                        data.append((
                            metric.get('time', datetime.utcnow()),
                            metric['keyword'],
                            metric.get('mention_count', 0),
                            metric.get('article_count', 0),
                            metric.get('avg_sentiment'),
                            metric.get('sentiment_variance'),
                            metric.get('growth_rate', 0.0),
                            metric.get('momentum_score', 0.0),
                            metric.get('primary_sources', []),
                            metric.get('related_keywords', [])
                        ))

                    psycopg2.extras.execute_values(
                        cursor, insert_query, data, page_size=1000
                    )
                    conn.commit()
                    
                    logger.info(f"Stored {len(metrics)} keyword metrics")
                    return True

        except Exception as e:
            logger.error(f"Failed to store keyword metrics: {e}")
            return False

    def store_user_activity(self, activities: List[Dict[str, Any]]) -> bool:
        """Store user activity metrics."""
        if not activities:
            return True

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    insert_query = sql.SQL("""
                        INSERT INTO analytics.user_activity (
                            time, user_id, organization_id, action_type,
                            resource_type, resource_id, session_id,
                            duration_seconds, request_size_bytes,
                            response_size_bytes, response_time_ms,
                            user_agent, ip_address, referrer
                        ) VALUES %s
                    """)

                    data = []
                    for activity in activities:
                        data.append((
                            activity.get('time', datetime.utcnow()),
                            activity['user_id'],
                            activity['organization_id'],
                            activity['action_type'],
                            activity.get('resource_type'),
                            activity.get('resource_id'),
                            activity.get('session_id'),
                            activity.get('duration_seconds'),
                            activity.get('request_size_bytes'),
                            activity.get('response_size_bytes'),
                            activity.get('response_time_ms'),
                            activity.get('user_agent'),
                            activity.get('ip_address'),
                            activity.get('referrer')
                        ))

                    psycopg2.extras.execute_values(
                        cursor, insert_query, data, page_size=1000
                    )
                    conn.commit()
                    
                    logger.info(f"Stored {len(activities)} user activities")
                    return True

        except Exception as e:
            logger.error(f"Failed to store user activities: {e}")
            return False

    def get_article_trends(self, 
                          sources: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          interval: str = '1 hour') -> List[Dict[str, Any]]:
        """Get article volume trends over time.
        
        Args:
            sources: Optional list of sources to filter by
            start_time: Start of time range
            end_time: End of time range  
            interval: Time bucket interval (e.g., '1 hour', '1 day')
            
        Returns:
            List of trend data points
        """
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    query = sql.SQL("""
                        SELECT 
                            time_bucket({interval}, time) AS bucket,
                            source,
                            COUNT(*) as article_count,
                            AVG(sentiment_score) as avg_sentiment,
                            AVG(content_length) as avg_content_length,
                            SUM(views) as total_views
                        FROM analytics.article_metrics
                        WHERE time >= %s AND time <= %s
                        {source_filter}
                        GROUP BY bucket, source
                        ORDER BY bucket, source
                    """).format(
                        interval=sql.Literal(interval),
                        source_filter=sql.SQL("AND source = ANY(%s)") if sources else sql.SQL("")
                    )

                    params = [start_time, end_time]
                    if sources:
                        params.append(sources)

                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get article trends: {e}")
            return []

    def get_keyword_trends(self,
                          keywords: Optional[List[str]] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get keyword trend data."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=30)
        if not end_time:
            end_time = datetime.utcnow()

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    query = sql.SQL("""
                        SELECT 
                            time_bucket('1 day', time) AS bucket,
                            keyword,
                            SUM(mention_count) as total_mentions,
                            SUM(article_count) as total_articles,
                            AVG(avg_sentiment) as avg_sentiment,
                            MAX(growth_rate) as max_growth_rate,
                            AVG(momentum_score) as avg_momentum
                        FROM analytics.keyword_metrics
                        WHERE time >= %s AND time <= %s
                        {keyword_filter}
                        GROUP BY bucket, keyword
                        HAVING SUM(mention_count) > 0
                        ORDER BY bucket DESC, total_mentions DESC
                        LIMIT %s
                    """).format(
                        keyword_filter=sql.SQL("AND keyword = ANY(%s)") if keywords else sql.SQL("")
                    )

                    params = [start_time, end_time]
                    if keywords:
                        params.append(keywords)
                    params.append(limit)

                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get keyword trends: {e}")
            return []

    def get_source_performance(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get source performance metrics."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            source,
                            SUM(articles_scraped) as total_scraped,
                            SUM(articles_published) as total_published,
                            SUM(articles_failed) as total_failed,
                            AVG(success_rate) as avg_success_rate,
                            AVG(scrape_duration_ms) as avg_scrape_time,
                            SUM(error_count) as total_errors,
                            AVG(avg_content_length) as avg_content_length,
                            AVG(avg_sentiment) as avg_sentiment
                        FROM analytics.source_metrics
                        WHERE time >= %s AND time <= %s
                        GROUP BY source
                        ORDER BY total_published DESC
                    """, [start_time, end_time])
                    
                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get source performance: {e}")
            return []

    def store_trend_analysis(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Store trend analysis results.
        
        Args:
            analysis: Trend analysis results dictionary
            
        Returns:
            Analysis ID if successful, None otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO analytics.trend_analysis (
                            analysis_type, subject, time_period_start, time_period_end,
                            trend_direction, trend_strength, trend_confidence,
                            slope, r_squared, p_value, anomalies_detected,
                            anomaly_dates, anomaly_scores, has_seasonality,
                            seasonal_period_days, seasonal_strength,
                            next_period_forecast, forecast_confidence_interval,
                            forecast_accuracy_score, analysis_duration_seconds,
                            model_used, parameters
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) RETURNING id
                    """, [
                        analysis['analysis_type'],
                        analysis['subject'],
                        analysis['time_period_start'],
                        analysis['time_period_end'],
                        analysis.get('trend_direction'),
                        analysis.get('trend_strength'),
                        analysis.get('trend_confidence'),
                        analysis.get('slope'),
                        analysis.get('r_squared'),
                        analysis.get('p_value'),
                        analysis.get('anomalies_detected', 0),
                        analysis.get('anomaly_dates', []),
                        analysis.get('anomaly_scores', []),
                        analysis.get('has_seasonality', False),
                        analysis.get('seasonal_period_days'),
                        analysis.get('seasonal_strength'),
                        analysis.get('next_period_forecast'),
                        analysis.get('forecast_confidence_interval', []),
                        analysis.get('forecast_accuracy_score'),
                        analysis.get('analysis_duration_seconds'),
                        analysis.get('model_used'),
                        analysis.get('parameters', {})
                    ])
                    
                    analysis_id = cursor.fetchone()[0]
                    conn.commit()
                    
                    logger.info(f"Stored trend analysis: {analysis_id}")
                    return str(analysis_id)

        except Exception as e:
            logger.error(f"Failed to store trend analysis: {e}")
            return None

    def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("TimescaleDB connection pool closed")