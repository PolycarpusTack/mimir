"""Async operations for Mimir Analytics.

This module provides asynchronous versions of expensive operations to improve
performance and responsiveness of the analytics system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import numpy as np
from functools import wraps

from .exceptions import AnalyticsBaseException, PerformanceException
from .utils import performance_timer, batch_process

logger = logging.getLogger(__name__)


class AsyncOperationManager:
    """Manager for asynchronous analytics operations."""
    
    def __init__(self, max_workers: int = 4, timeout: float = 300.0):
        """Initialize async operation manager.
        
        Args:
            max_workers: Maximum number of worker threads
            timeout: Default timeout for operations in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
    
    async def run_async(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous function asynchronously.
        
        Args:
            func: Function to run asynchronously
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            PerformanceException: If operation times out or fails
        """
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, func, *args, **kwargs),
                timeout=self.timeout
            )
            return result
        except asyncio.TimeoutError:
            raise PerformanceException(f"Operation timed out after {self.timeout} seconds")
        except Exception as e:
            raise PerformanceException(f"Async operation failed: {e}")
    
    async def run_batch_async(self, func: Callable, items: List[Any], 
                             batch_size: int = 10) -> List[Any]:
        """Run function on batches of items asynchronously.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            batch_size: Size of each batch
            
        Returns:
            List of results
        """
        batches = batch_process(items, batch_size)
        tasks = []
        
        for batch in batches:
            task = self.run_async(self._process_batch, func, batch)
            tasks.append(task)
        
        try:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results and handle exceptions
            results = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self.logger.error(f"Batch processing failed: {batch_result}")
                    # Add None for failed items
                    results.extend([None] * batch_size)
                else:
                    results.extend(batch_result)
            
            return results
            
        except Exception as e:
            raise PerformanceException(f"Batch async operation failed: {e}")
    
    @staticmethod
    def _process_batch(func: Callable, batch: List[Any]) -> List[Any]:
        """Process a batch of items with the given function."""
        return [func(item) for item in batch]
    
    async def parallel_data_processing(self, data_sources: List[Dict[str, Any]],
                                     processing_func: Callable) -> Dict[str, Any]:
        """Process multiple data sources in parallel.
        
        Args:
            data_sources: List of data source configurations
            processing_func: Function to process each data source
            
        Returns:
            Dictionary mapping source names to results
        """
        tasks = {}
        
        for source in data_sources:
            source_name = source.get('name', f'source_{id(source)}')
            task = self.run_async(processing_func, source)
            tasks[source_name] = task
        
        results = {}
        completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for source_name, result in zip(tasks.keys(), completed_tasks):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process source {source_name}: {result}")
                results[source_name] = {'error': str(result)}
            else:
                results[source_name] = result
        
        return results
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def async_cached(ttl: int = 300):
    """Decorator for async caching with TTL.
    
    Args:
        ttl: Cache time-to-live in seconds
    """
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check cache
            if key in cache:
                cached_result, cached_time = cache[key]
                if current_time - cached_time < ttl:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[key] = (result, current_time)
            
            # Clean old entries (simple cleanup)
            if len(cache) > 100:  # Limit cache size
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator


class AsyncAnalyticsProcessor:
    """Async processor for analytics operations."""
    
    def __init__(self, analytics_warehouse, max_workers: int = 4):
        """Initialize async analytics processor.
        
        Args:
            analytics_warehouse: Analytics data warehouse instance
            max_workers: Maximum number of worker threads
        """
        self.warehouse = analytics_warehouse
        self.operation_manager = AsyncOperationManager(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
    
    async def collect_metrics_async(self, sources: List[str], 
                                  time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Collect metrics from multiple sources asynchronously.
        
        Args:
            sources: List of source names
            time_range: Dictionary with 'start' and 'end' datetime
            
        Returns:
            Dictionary containing collected metrics
        """
        start_time = time.time()
        
        # Create data source configurations
        data_sources = [
            {
                'name': source,
                'source': source,
                'start_time': time_range['start'],
                'end_time': time_range['end']
            }
            for source in sources
        ]
        
        # Process sources in parallel
        results = await self.operation_manager.parallel_data_processing(
            data_sources, self._collect_source_metrics
        )
        
        # Aggregate results
        total_articles = 0
        total_errors = 0
        source_results = {}
        
        for source, result in results.items():
            if 'error' in result:
                total_errors += 1
                source_results[source] = {'status': 'error', 'error': result['error']}
            else:
                total_articles += result.get('article_count', 0)
                source_results[source] = {
                    'status': 'success',
                    'article_count': result.get('article_count', 0),
                    'avg_sentiment': result.get('avg_sentiment', 0),
                    'processing_time': result.get('processing_time', 0)
                }
        
        execution_time = time.time() - start_time
        
        return {
            'sources_processed': len(sources),
            'total_articles': total_articles,
            'total_errors': total_errors,
            'source_results': source_results,
            'execution_time': execution_time,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _collect_source_metrics(self, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics for a single source (sync function for async execution)."""
        start_time = time.time()
        
        try:
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get article metrics for the source
                cursor.execute("""
                    SELECT 
                        COUNT(*) as article_count,
                        AVG(sentiment_score) as avg_sentiment,
                        MAX(published_date) as latest_article,
                        MIN(published_date) as earliest_article
                    FROM articles 
                    WHERE source = %s 
                    AND published_date >= %s 
                    AND published_date <= %s
                """, [
                    source_config['source'],
                    source_config['start_time'],
                    source_config['end_time']
                ])
                
                result = cursor.fetchone()
                
                processing_time = time.time() - start_time
                
                return {
                    'article_count': result[0] or 0,
                    'avg_sentiment': float(result[1]) if result[1] else 0.0,
                    'latest_article': result[2].isoformat() if result[2] else None,
                    'earliest_article': result[3].isoformat() if result[3] else None,
                    'processing_time': processing_time
                }
                
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for source {source_config['source']}: {e}")
            return {'error': str(e)}
    
    @async_cached(ttl=300)  # Cache for 5 minutes
    async def analyze_trends_async(self, metric_type: str, 
                                 days_back: int = 30) -> Dict[str, Any]:
        """Analyze trends asynchronously with caching.
        
        Args:
            metric_type: Type of metric to analyze
            days_back: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        return await self.operation_manager.run_async(
            self._analyze_trends_sync, metric_type, days_back
        )
    
    def _analyze_trends_sync(self, metric_type: str, days_back: int) -> Dict[str, Any]:
        """Synchronous trend analysis (for async execution)."""
        start_time = time.time()
        
        try:
            end_time_dt = datetime.utcnow()
            start_time_dt = end_time_dt - timedelta(days=days_back)
            
            with self.warehouse.get_connection() as conn:
                if metric_type == 'article_volume':
                    query = """
                        SELECT 
                            time_bucket('1 hour', published_date) as bucket,
                            COUNT(*) as count
                        FROM articles
                        WHERE published_date >= %s AND published_date <= %s
                        GROUP BY bucket
                        ORDER BY bucket
                    """
                    params = [start_time_dt, end_time_dt]
                elif metric_type == 'sentiment':
                    query = """
                        SELECT 
                            time_bucket('1 hour', published_date) as bucket,
                            AVG(sentiment_score) as avg_sentiment
                        FROM articles
                        WHERE published_date >= %s AND published_date <= %s
                        AND sentiment_score IS NOT NULL
                        GROUP BY bucket
                        ORDER BY bucket
                    """
                    params = [start_time_dt, end_time_dt]
                else:
                    raise ValueError(f"Unknown metric type: {metric_type}")
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return {'error': 'No data available for analysis'}
                
                # Perform trend analysis
                values = df.iloc[:, 1].values  # Second column contains the values
                
                # Calculate basic statistics
                mean_value = np.mean(values)
                std_value = np.std(values)
                min_value = np.min(values)
                max_value = np.max(values)
                
                # Calculate trend (simple linear regression slope)
                x = np.arange(len(values))
                if len(values) > 1:
                    slope = np.polyfit(x, values, 1)[0]
                    trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                else:
                    slope = 0
                    trend_direction = 'insufficient_data'
                
                processing_time = time.time() - start_time
                
                return {
                    'metric_type': metric_type,
                    'data_points': len(values),
                    'time_range': {
                        'start': start_time_dt.isoformat(),
                        'end': end_time_dt.isoformat()
                    },
                    'statistics': {
                        'mean': float(mean_value),
                        'std': float(std_value),
                        'min': float(min_value),
                        'max': float(max_value),
                        'slope': float(slope),
                        'trend_direction': trend_direction
                    },
                    'processing_time': processing_time,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Trend analysis failed for {metric_type}: {e}")
            return {'error': str(e)}
    
    async def generate_reports_async(self, report_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate multiple reports asynchronously.
        
        Args:
            report_configs: List of report configuration dictionaries
            
        Returns:
            Dictionary mapping report names to generation results
        """
        start_time = time.time()
        
        # Generate reports in parallel
        results = await self.operation_manager.parallel_data_processing(
            report_configs, self._generate_single_report
        )
        
        # Summarize results
        successful_reports = sum(1 for result in results.values() if 'error' not in result)
        failed_reports = len(results) - successful_reports
        total_time = time.time() - start_time
        
        return {
            'total_reports': len(report_configs),
            'successful_reports': successful_reports,
            'failed_reports': failed_reports,
            'results': results,
            'total_time': total_time,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_single_report(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single report (sync function for async execution)."""
        start_time = time.time()
        
        try:
            # Simulate report generation
            report_type = report_config.get('type', 'unknown')
            time_period = report_config.get('time_period_days', 7)
            
            # This would be replaced with actual report generation logic
            time.sleep(0.1)  # Simulate processing time
            
            processing_time = time.time() - start_time
            
            return {
                'report_type': report_type,
                'time_period_days': time_period,
                'processing_time': processing_time,
                'status': 'completed',
                'file_size': 1024 * 50,  # Mock file size
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    async def bulk_data_export_async(self, export_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export multiple datasets asynchronously.
        
        Args:
            export_configs: List of export configuration dictionaries
            
        Returns:
            Dictionary containing export results
        """
        start_time = time.time()
        
        # Process exports with smaller batch size due to I/O intensity
        results = await self.operation_manager.run_batch_async(
            self._export_single_dataset, export_configs, batch_size=2
        )
        
        # Summarize results
        successful_exports = sum(1 for result in results if result and 'error' not in result)
        failed_exports = len(results) - successful_exports
        total_size = sum(
            result.get('file_size', 0) for result in results 
            if result and 'error' not in result
        )
        
        total_time = time.time() - start_time
        
        return {
            'total_exports': len(export_configs),
            'successful_exports': successful_exports,
            'failed_exports': failed_exports,
            'total_size_bytes': total_size,
            'total_time': total_time,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _export_single_dataset(self, export_config: Dict[str, Any]) -> Dict[str, Any]:
        """Export a single dataset (sync function for async execution)."""
        start_time = time.time()
        
        try:
            dataset_name = export_config.get('dataset', 'unknown')
            format_type = export_config.get('format', 'csv')
            
            # Simulate data export
            time.sleep(0.2)  # Simulate I/O time
            
            processing_time = time.time() - start_time
            mock_file_size = 1024 * 100  # Mock file size
            
            return {
                'dataset': dataset_name,
                'format': format_type,
                'file_size': mock_file_size,
                'processing_time': processing_time,
                'status': 'completed',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Export failed for dataset: {e}")
            return {'error': str(e)}


# Async context manager for analytics operations
class AsyncAnalyticsContext:
    """Async context manager for analytics operations."""
    
    def __init__(self, analytics_warehouse, max_workers: int = 4):
        """Initialize async analytics context.
        
        Args:
            analytics_warehouse: Analytics data warehouse instance
            max_workers: Maximum number of worker threads
        """
        self.processor = AsyncAnalyticsProcessor(analytics_warehouse, max_workers)
    
    async def __aenter__(self):
        """Enter async context."""
        return self.processor
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        # Cleanup if needed
        if hasattr(self.processor.operation_manager, 'executor'):
            self.processor.operation_manager.executor.shutdown(wait=True)


# Utility functions for async operations
async def run_analytics_pipeline(analytics_warehouse, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a complete analytics pipeline asynchronously.
    
    Args:
        analytics_warehouse: Analytics data warehouse instance
        pipeline_config: Pipeline configuration
        
    Returns:
        Pipeline execution results
    """
    start_time = time.time()
    
    async with AsyncAnalyticsContext(analytics_warehouse) as processor:
        results = {}
        
        # Step 1: Collect metrics if configured
        if 'metrics_collection' in pipeline_config:
            metrics_config = pipeline_config['metrics_collection']
            results['metrics'] = await processor.collect_metrics_async(
                sources=metrics_config.get('sources', []),
                time_range=metrics_config.get('time_range', {
                    'start': datetime.utcnow() - timedelta(days=1),
                    'end': datetime.utcnow()
                })
            )
        
        # Step 2: Analyze trends if configured
        if 'trend_analysis' in pipeline_config:
            trend_config = pipeline_config['trend_analysis']
            results['trends'] = await processor.analyze_trends_async(
                metric_type=trend_config.get('metric_type', 'article_volume'),
                days_back=trend_config.get('days_back', 30)
            )
        
        # Step 3: Generate reports if configured
        if 'report_generation' in pipeline_config:
            report_configs = pipeline_config['report_generation'].get('reports', [])
            results['reports'] = await processor.generate_reports_async(report_configs)
        
        # Step 4: Export data if configured
        if 'data_export' in pipeline_config:
            export_configs = pipeline_config['data_export'].get('exports', [])
            results['exports'] = await processor.bulk_data_export_async(export_configs)
    
    total_time = time.time() - start_time
    
    return {
        'pipeline_results': results,
        'total_execution_time': total_time,
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'completed'
    }