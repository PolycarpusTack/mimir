"""Metrics Collector for Analytics Pipeline.

This module collects metrics from various parts of the Mimir system and
aggregates them for storage in the TimescaleDB data warehouse.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil

from db_adapter import DatabaseAdapter
from .data_warehouse import AnalyticsDataWarehouse

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics from the Mimir system."""

    def __init__(self, db_adapter: DatabaseAdapter, analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the metrics collector.
        
        Args:
            db_adapter: Main database adapter for reading data
            analytics_warehouse: Analytics data warehouse for storing metrics
        """
        self.db_adapter = db_adapter
        self.analytics_warehouse = analytics_warehouse
        self.logger = logging.getLogger(__name__)

    def collect_article_metrics(self, 
                               start_time: Optional[datetime] = None,
                               batch_size: int = 1000) -> bool:
        """Collect article-level metrics and store in analytics warehouse.
        
        Args:
            start_time: Only collect metrics for articles after this time
            batch_size: Number of articles to process in each batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=1)

            # Get articles from main database
            with self.db_adapter.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id, url, title, content, published_date, source,
                        author, category, scraped_at, metadata
                    FROM articles 
                    WHERE scraped_at >= %s
                    ORDER BY scraped_at
                """, [start_time])

                articles = cursor.fetchall()
                
                if not articles:
                    self.logger.info("No new articles to process for metrics")
                    return True

                # Process articles in batches
                metrics_batch = []
                for article in articles:
                    article_id, url, title, content, published_date, source, author, category, scraped_at, metadata = article
                    
                    # Calculate content metrics
                    title_length = len(title) if title else 0
                    content_length = len(content) if content else 0
                    word_count = len(content.split()) if content else 0
                    paragraph_count = content.count('\n\n') + 1 if content else 0

                    # Parse metadata if available
                    meta_dict = {}
                    if metadata:
                        try:
                            meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Extract AI-derived metrics from metadata
                    sentiment_score = meta_dict.get('sentiment_score')
                    sentiment_confidence = meta_dict.get('sentiment_confidence')
                    readability_score = meta_dict.get('readability_score')
                    topic_confidence = meta_dict.get('topic_confidence')
                    
                    # Extract entity counts
                    entities = meta_dict.get('entities', {})
                    person_entities = len(entities.get('persons', []))
                    organization_entities = len(entities.get('organizations', []))
                    location_entities = len(entities.get('locations', []))
                    misc_entities = len(entities.get('misc', []))
                    
                    # Extract keyword and tag counts
                    keywords = meta_dict.get('keywords', [])
                    tags = meta_dict.get('tags', [])
                    keyword_count = len(keywords) if isinstance(keywords, list) else 0
                    tag_count = len(tags) if isinstance(tags, list) else 0

                    # Determine language from metadata or content
                    language = meta_dict.get('language', 'unknown')
                    
                    # Create metrics record
                    metric = {
                        'time': scraped_at or datetime.utcnow(),
                        'article_id': article_id,
                        'source': source or 'unknown',
                        'title_length': title_length,
                        'content_length': content_length,
                        'word_count': word_count,
                        'paragraph_count': paragraph_count,
                        'views': 0,  # To be updated by external systems
                        'shares': 0,  # To be updated by external systems
                        'comments': 0,  # To be updated by external systems
                        'sentiment_score': sentiment_score,
                        'sentiment_confidence': sentiment_confidence,
                        'readability_score': readability_score,
                        'topic_confidence': topic_confidence,
                        'person_entities': person_entities,
                        'organization_entities': organization_entities,
                        'location_entities': location_entities,
                        'misc_entities': misc_entities,
                        'keyword_count': keyword_count,
                        'tag_count': tag_count,
                        'language': language,
                        'category': category,
                        'priority': meta_dict.get('priority', 0)
                    }
                    
                    metrics_batch.append(metric)
                    
                    # Store batch when it reaches the batch size
                    if len(metrics_batch) >= batch_size:
                        self.analytics_warehouse.store_article_metrics(metrics_batch)
                        metrics_batch = []

                # Store remaining metrics
                if metrics_batch:
                    self.analytics_warehouse.store_article_metrics(metrics_batch)

                self.logger.info(f"Collected metrics for {len(articles)} articles")
                return True

        except Exception as e:
            self.logger.error(f"Failed to collect article metrics: {e}")
            return False

    def collect_source_metrics(self, time_window_hours: int = 1) -> bool:
        """Collect source performance metrics.
        
        Args:
            time_window_hours: Time window to analyze for source performance
            
        Returns:
            True if successful, False otherwise  
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)

            with self.db_adapter.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get source performance data
                cursor.execute("""
                    SELECT 
                        source,
                        COUNT(*) as articles_scraped,
                        COUNT(CASE WHEN content IS NOT NULL AND content != '' THEN 1 END) as articles_published,
                        COUNT(CASE WHEN content IS NULL OR content = '' THEN 1 END) as articles_failed,
                        AVG(LENGTH(content)) as avg_content_length,
                        AVG(CASE WHEN metadata->>'sentiment_score' IS NOT NULL 
                             THEN CAST(metadata->>'sentiment_score' AS DECIMAL) END) as avg_sentiment,
                        AVG(CASE WHEN metadata->>'readability_score' IS NOT NULL 
                             THEN CAST(metadata->>'readability_score' AS DECIMAL) END) as avg_readability,
                        EXTRACT(EPOCH FROM (MAX(scraped_at) - MIN(scraped_at))) * 1000 as scrape_duration_ms,
                        MIN(EXTRACT(EPOCH FROM (NOW() - published_date)) / 3600) as latest_article_age_hours,
                        AVG(EXTRACT(EPOCH FROM (NOW() - published_date)) / 3600) as avg_article_age_hours
                    FROM articles 
                    WHERE scraped_at >= %s AND scraped_at <= %s
                      AND source IS NOT NULL
                    GROUP BY source
                """, [start_time, end_time])

                source_stats = cursor.fetchall()
                
                if not source_stats:
                    self.logger.info("No source data to process for metrics")
                    return True

                # Process source metrics
                metrics = []
                for stats in source_stats:
                    (source, articles_scraped, articles_published, articles_failed,
                     avg_content_length, avg_sentiment, avg_readability, scrape_duration_ms,
                     latest_article_age_hours, avg_article_age_hours) = stats
                    
                    # Calculate success rate
                    success_rate = articles_published / articles_scraped if articles_scraped > 0 else 0.0
                    
                    metric = {
                        'time': end_time,
                        'source': source,
                        'articles_scraped': articles_scraped or 0,
                        'articles_published': articles_published or 0,
                        'articles_failed': articles_failed or 0,
                        'avg_content_length': float(avg_content_length) if avg_content_length else None,
                        'avg_sentiment': float(avg_sentiment) if avg_sentiment else None,
                        'avg_readability': float(avg_readability) if avg_readability else None,
                        'scrape_duration_ms': int(scrape_duration_ms) if scrape_duration_ms else None,
                        'success_rate': success_rate,
                        'error_count': articles_failed or 0,
                        'latest_article_age_hours': float(latest_article_age_hours) if latest_article_age_hours else None,
                        'avg_article_age_hours': float(avg_article_age_hours) if avg_article_age_hours else None
                    }
                    
                    metrics.append(metric)

                # Store source metrics
                self.analytics_warehouse.store_source_metrics(metrics)
                self.logger.info(f"Collected metrics for {len(metrics)} sources")
                return True

        except Exception as e:
            self.logger.error(f"Failed to collect source metrics: {e}")
            return False

    def collect_keyword_metrics(self, time_window_hours: int = 24) -> bool:
        """Collect keyword trend metrics.
        
        Args:
            time_window_hours: Time window to analyze for keyword trends
            
        Returns:
            True if successful, False otherwise
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=time_window_hours)

            with self.db_adapter.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get keyword occurrences from articles
                cursor.execute("""
                    WITH keyword_extraction AS (
                        SELECT 
                            scraped_at,
                            source,
                            CASE WHEN metadata->>'sentiment_score' IS NOT NULL 
                                 THEN CAST(metadata->>'sentiment_score' AS DECIMAL) END as sentiment_score,
                            UNNEST(
                                CASE WHEN metadata->>'keywords' IS NOT NULL 
                                     THEN ARRAY(SELECT json_array_elements_text(metadata->'keywords'))
                                     ELSE ARRAY[]::text[] END
                            ) as keyword
                        FROM articles 
                        WHERE scraped_at >= %s AND scraped_at <= %s
                          AND metadata IS NOT NULL
                          AND metadata->>'keywords' IS NOT NULL
                    )
                    SELECT 
                        keyword,
                        COUNT(*) as mention_count,
                        COUNT(DISTINCT source) as article_count,
                        AVG(sentiment_score) as avg_sentiment,
                        VAR_POP(sentiment_score) as sentiment_variance,
                        ARRAY_AGG(DISTINCT source ORDER BY source) as primary_sources
                    FROM keyword_extraction
                    WHERE keyword IS NOT NULL AND keyword != ''
                    GROUP BY keyword
                    HAVING COUNT(*) >= 2  -- Only keywords with multiple mentions
                    ORDER BY mention_count DESC
                    LIMIT 1000
                """, [start_time, end_time])

                keyword_stats = cursor.fetchall()
                
                if not keyword_stats:
                    self.logger.info("No keyword data to process for metrics")
                    return True

                # Calculate growth rates (requires historical data)
                keyword_growth = self._calculate_keyword_growth_rates(
                    keywords=[stat[0] for stat in keyword_stats],
                    current_period_start=start_time,
                    current_period_end=end_time
                )

                # Process keyword metrics
                metrics = []
                for stats in keyword_stats:
                    (keyword, mention_count, article_count, avg_sentiment, 
                     sentiment_variance, primary_sources) = stats
                    
                    growth_rate = keyword_growth.get(keyword, 0.0)
                    momentum_score = self._calculate_momentum_score(mention_count, growth_rate)
                    
                    metric = {
                        'time': end_time,
                        'keyword': keyword,
                        'mention_count': mention_count or 0,
                        'article_count': article_count or 0,
                        'avg_sentiment': float(avg_sentiment) if avg_sentiment else None,
                        'sentiment_variance': float(sentiment_variance) if sentiment_variance else None,
                        'growth_rate': growth_rate,
                        'momentum_score': momentum_score,
                        'primary_sources': primary_sources[:10] if primary_sources else [],  # Top 10 sources
                        'related_keywords': []  # To be filled by keyword similarity analysis
                    }
                    
                    metrics.append(metric)

                # Store keyword metrics
                self.analytics_warehouse.store_keyword_metrics(metrics)
                self.logger.info(f"Collected metrics for {len(metrics)} keywords")
                return True

        except Exception as e:
            self.logger.error(f"Failed to collect keyword metrics: {e}")
            return False

    def collect_system_metrics(self) -> bool:
        """Collect system performance metrics.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get system resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get database connection metrics
            db_connections = self._get_database_connections()
            
            # Get scraper metrics (if available)
            scraper_metrics = self._get_scraper_metrics()
            
            # Get AI processing metrics (if available)
            ai_metrics = self._get_ai_processing_metrics()

            # Create system metrics record
            metric = {
                'time': datetime.utcnow(),
                'cpu_usage_percent': cpu_percent,
                'memory_usage_mb': memory.used // 1024 // 1024,
                'disk_usage_mb': disk.used // 1024 // 1024,
                'active_connections': db_connections.get('active', 0),
                'queries_per_second': db_connections.get('qps', 0.0),
                'avg_query_time_ms': db_connections.get('avg_time', 0.0),
                'active_scrapers': scraper_metrics.get('active', 0),
                'articles_per_minute': scraper_metrics.get('rate', 0.0),
                'error_rate_percent': scraper_metrics.get('error_rate', 0.0),
                'ai_queue_size': ai_metrics.get('queue_size', 0),
                'ai_processing_time_ms': ai_metrics.get('avg_time', 0.0),
                'ai_success_rate': ai_metrics.get('success_rate', 1.0)
            }

            # Store system metrics
            with self.analytics_warehouse.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO analytics.system_metrics (
                            time, cpu_usage_percent, memory_usage_mb, disk_usage_mb,
                            active_connections, queries_per_second, avg_query_time_ms,
                            active_scrapers, articles_per_minute, error_rate_percent,
                            ai_queue_size, ai_processing_time_ms, ai_success_rate
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, [
                        metric['time'], metric['cpu_usage_percent'], metric['memory_usage_mb'],
                        metric['disk_usage_mb'], metric['active_connections'], 
                        metric['queries_per_second'], metric['avg_query_time_ms'],
                        metric['active_scrapers'], metric['articles_per_minute'],
                        metric['error_rate_percent'], metric['ai_queue_size'],
                        metric['ai_processing_time_ms'], metric['ai_success_rate']
                    ])
                    conn.commit()

            self.logger.info("Collected system metrics")
            return True

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return False

    def _calculate_keyword_growth_rates(self, 
                                       keywords: List[str],
                                       current_period_start: datetime,
                                       current_period_end: datetime) -> Dict[str, float]:
        """Calculate growth rates for keywords compared to previous period."""
        growth_rates = {}
        
        try:
            # Calculate previous period
            period_duration = current_period_end - current_period_start
            previous_period_end = current_period_start
            previous_period_start = previous_period_end - period_duration

            # Get historical keyword data from analytics warehouse
            current_data = self.analytics_warehouse.get_keyword_trends(
                keywords=keywords,
                start_time=current_period_start,
                end_time=current_period_end
            )
            
            previous_data = self.analytics_warehouse.get_keyword_trends(
                keywords=keywords,
                start_time=previous_period_start,
                end_time=previous_period_end
            )

            # Create lookup for previous period data
            previous_lookup = {}
            for data in previous_data:
                previous_lookup[data['keyword']] = data['total_mentions']

            # Calculate growth rates
            for data in current_data:
                keyword = data['keyword']
                current_mentions = data['total_mentions']
                previous_mentions = previous_lookup.get(keyword, 0)
                
                if previous_mentions > 0:
                    growth_rate = ((current_mentions - previous_mentions) / previous_mentions) * 100
                else:
                    growth_rate = 100.0 if current_mentions > 0 else 0.0
                
                growth_rates[keyword] = growth_rate

        except Exception as e:
            self.logger.error(f"Failed to calculate keyword growth rates: {e}")

        return growth_rates

    def _calculate_momentum_score(self, mention_count: int, growth_rate: float) -> float:
        """Calculate momentum score based on mention count and growth rate."""
        # Simple momentum calculation: combines volume and growth
        volume_score = min(mention_count / 100.0, 1.0)  # Normalize to 0-1
        growth_score = max(min(growth_rate / 100.0, 1.0), -1.0)  # Normalize to -1 to 1
        
        momentum = (volume_score * 0.3) + (growth_score * 0.7)  # Weight growth more heavily
        return round(momentum * 100, 2)  # Return as 0-100 scale

    def _get_database_connections(self) -> Dict[str, Any]:
        """Get database connection metrics."""
        try:
            with self.db_adapter.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
                active_connections = cursor.fetchone()[0]
                
                return {
                    'active': active_connections,
                    'qps': 0.0,  # Would need more sophisticated monitoring
                    'avg_time': 0.0  # Would need query performance monitoring
                }
        except Exception:
            return {'active': 0, 'qps': 0.0, 'avg_time': 0.0}

    def _get_scraper_metrics(self) -> Dict[str, Any]:
        """Get scraper performance metrics."""
        # This would integrate with scraper monitoring
        # For now, return default values
        return {
            'active': 0,
            'rate': 0.0,
            'error_rate': 0.0
        }

    def _get_ai_processing_metrics(self) -> Dict[str, Any]:
        """Get AI processing metrics."""
        # This would integrate with AI pipeline monitoring
        # For now, return default values  
        return {
            'queue_size': 0,
            'avg_time': 0.0,
            'success_rate': 1.0
        }

    def run_full_collection(self) -> bool:
        """Run full metrics collection cycle.
        
        Returns:
            True if all collections successful, False otherwise
        """
        success = True
        
        self.logger.info("Starting full metrics collection cycle")
        
        # Collect article metrics (last hour)
        if not self.collect_article_metrics():
            success = False
        
        # Collect source metrics (last hour)
        if not self.collect_source_metrics():
            success = False
        
        # Collect keyword metrics (last 24 hours)
        if not self.collect_keyword_metrics():
            success = False
        
        # Collect system metrics
        if not self.collect_system_metrics():
            success = False
        
        self.logger.info(f"Metrics collection cycle completed: {'SUCCESS' if success else 'PARTIAL_FAILURE'}")
        return success