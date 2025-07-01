"""Analytics Validation Module for Mimir Analytics.

This module provides data quality checks, metric validation, report accuracy testing,
and performance benchmarking for the analytics system.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .data_warehouse import AnalyticsDataWarehouse
from .exceptions import (
    DataValidationException, DatabaseConnectionException,
    QueryExecutionException, AnalyticsBaseException
)
from .metrics_collector import MetricsCollector
from .trend_analyzer import TrendAnalyzer
from .predictive_analytics import PredictiveAnalytics
from .report_generator import ReportGenerator
from db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


class AnalyticsValidator:
    """Validates analytics data quality and accuracy."""
    
    def __init__(self, db_adapter: DatabaseAdapter,
                 analytics_warehouse: AnalyticsDataWarehouse) -> None:
        """Initialize the analytics validator.
        
        Args:
            db_adapter: Main database adapter
            analytics_warehouse: Analytics data warehouse
            
        Raises:
            DatabaseConnectionException: If connection to warehouse fails
        """
        self.db_adapter = db_adapter
        self.warehouse = analytics_warehouse
        self.metrics_collector = MetricsCollector(db_adapter, analytics_warehouse)
        self.trend_analyzer = TrendAnalyzer(analytics_warehouse)
        self.predictive_analytics = PredictiveAnalytics(analytics_warehouse)
        self.report_generator = ReportGenerator(analytics_warehouse)
        
        # Validation results storage
        self.results_dir = Path("analytics/validation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the analytics system.
        
        Returns:
            Validation results summary
        """
        self.logger.info("Starting comprehensive analytics validation")
        
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_quality': self.validate_data_quality(),
            'metric_accuracy': self.validate_metrics(),
            'report_accuracy': self.validate_reports(),
            'performance_benchmarks': self.run_performance_benchmarks(),
            'overall_health': 'healthy'  # Will be updated based on results
        }
        
        # Determine overall health
        issues_found = 0
        for category, results in validation_results.items():
            if isinstance(results, dict) and 'issues' in results:
                issues_found += len(results['issues'])
        
        if issues_found == 0:
            validation_results['overall_health'] = 'excellent'
        elif issues_found < 5:
            validation_results['overall_health'] = 'good'
        elif issues_found < 10:
            validation_results['overall_health'] = 'fair'
        else:
            validation_results['overall_health'] = 'needs_attention'
        
        # Save results
        self._save_validation_results(validation_results)
        
        return validation_results
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality across all analytics tables.
        
        Returns:
            Data quality validation results
        """
        self.logger.info("Validating data quality")
        
        quality_results = {
            'checked_at': datetime.utcnow().isoformat(),
            'tables': {},
            'issues': []
        }
        
        tables_to_check = [
            'analytics.article_metrics',
            'analytics.source_metrics',
            'analytics.keyword_metrics',
            'analytics.user_activity',
            'analytics.system_metrics'
        ]
        
        for table in tables_to_check:
            table_results = self._validate_table_quality(table)
            quality_results['tables'][table] = table_results
            
            # Collect issues
            if table_results['issues']:
                for issue in table_results['issues']:
                    quality_results['issues'].append({
                        'table': table,
                        **issue
                    })
        
        # Cross-table consistency checks
        consistency_issues = self._check_cross_table_consistency()
        quality_results['consistency_checks'] = consistency_issues
        
        if consistency_issues:
            quality_results['issues'].extend(consistency_issues)
        
        return quality_results
    
    def validate_metrics(self) -> Dict[str, Any]:
        """Validate metric calculations and accuracy.
        
        Returns:
            Metric validation results
        """
        self.logger.info("Validating metrics accuracy")
        
        metric_results = {
            'checked_at': datetime.utcnow().isoformat(),
            'metrics': {},
            'issues': []
        }
        
        # Validate article metrics
        article_metrics_valid = self._validate_article_metrics()
        metric_results['metrics']['article_metrics'] = article_metrics_valid
        
        # Validate source metrics
        source_metrics_valid = self._validate_source_metrics()
        metric_results['metrics']['source_metrics'] = source_metrics_valid
        
        # Validate keyword metrics
        keyword_metrics_valid = self._validate_keyword_metrics()
        metric_results['metrics']['keyword_metrics'] = keyword_metrics_valid
        
        # Validate trend calculations
        trend_metrics_valid = self._validate_trend_calculations()
        metric_results['metrics']['trend_calculations'] = trend_metrics_valid
        
        # Collect all issues
        for metric_type, validation in metric_results['metrics'].items():
            if 'issues' in validation:
                for issue in validation['issues']:
                    metric_results['issues'].append({
                        'metric_type': metric_type,
                        **issue
                    })
        
        return metric_results
    
    def validate_reports(self) -> Dict[str, Any]:
        """Validate report generation accuracy.
        
        Returns:
            Report validation results
        """
        self.logger.info("Validating report accuracy")
        
        report_results = {
            'checked_at': datetime.utcnow().isoformat(),
            'templates': {},
            'issues': []
        }
        
        # Get available templates
        templates = self.report_generator.get_available_templates()
        
        for template in templates:
            template_id = template['template_id']
            
            # Generate test report
            test_result = self._validate_report_template(template_id)
            report_results['templates'][template_id] = test_result
            
            if test_result['issues']:
                for issue in test_result['issues']:
                    report_results['issues'].append({
                        'template_id': template_id,
                        **issue
                    })
        
        return report_results
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks on analytics operations.
        
        Returns:
            Performance benchmark results
        """
        self.logger.info("Running performance benchmarks")
        
        benchmark_results = {
            'run_at': datetime.utcnow().isoformat(),
            'benchmarks': {},
            'summary': {}
        }
        
        # Benchmark data collection
        collection_benchmark = self._benchmark_data_collection()
        benchmark_results['benchmarks']['data_collection'] = collection_benchmark
        
        # Benchmark query performance
        query_benchmark = self._benchmark_queries()
        benchmark_results['benchmarks']['queries'] = query_benchmark
        
        # Benchmark trend analysis
        trend_benchmark = self._benchmark_trend_analysis()
        benchmark_results['benchmarks']['trend_analysis'] = trend_benchmark
        
        # Benchmark predictions
        prediction_benchmark = self._benchmark_predictions()
        benchmark_results['benchmarks']['predictions'] = prediction_benchmark
        
        # Benchmark report generation
        report_benchmark = self._benchmark_report_generation()
        benchmark_results['benchmarks']['report_generation'] = report_benchmark
        
        # Calculate summary statistics
        total_time = sum(
            b.get('total_time', 0) 
            for b in benchmark_results['benchmarks'].values()
        )
        
        benchmark_results['summary'] = {
            'total_benchmark_time': total_time,
            'average_operation_time': total_time / len(benchmark_results['benchmarks']) if benchmark_results['benchmarks'] else 0,
            'performance_rating': self._calculate_performance_rating(benchmark_results['benchmarks'])
        }
        
        return benchmark_results
    
    # Helper validation methods
    
    def _validate_table_quality(self, table_name: str) -> Dict[str, Any]:
        """Validate quality of a specific table.
        
        Args:
            table_name: Name of the table to validate
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'row_count': 0,
            'issues': [],
            'checks': {}
        }
        
        try:
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic table info
                results['row_count'] = self._get_table_row_count(cursor, table_name)
                
                # Run individual validation checks
                null_issues = self._check_null_values(cursor, table_name)
                duplicate_issues = self._check_duplicate_keys(cursor, table_name)
                freshness_issues = self._check_data_freshness(cursor, table_name)
                
                # Combine all issues
                results['issues'].extend(null_issues)
                results['issues'].extend(duplicate_issues)
                results['issues'].extend(freshness_issues)
                
                # Set check status
                results['checks'] = self._compile_check_results(results['issues'])
                
        except DatabaseConnectionException as e:
            self.logger.error(f"Database connection error validating table {table_name}: {e}")
            results['issues'].append({
                'type': 'database_connection_error',
                'error': str(e),
                'severity': 'critical'
            })
        except QueryExecutionException as e:
            self.logger.error(f"Query execution error validating table {table_name}: {e}")
            results['issues'].append({
                'type': 'query_execution_error',
                'error': str(e),
                'severity': 'high'
            })
        except Exception as e:
            self.logger.error(f"Unexpected error validating table {table_name}: {e}")
            results['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        return results
    
    def _get_table_row_count(self, cursor, table_name: str) -> int:
        """Get row count for a table.
        
        Args:
            cursor: Database cursor
            table_name: Name of the table
            
        Returns:
            Number of rows in the table
        """
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    
    def _check_null_values(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Check for null values in non-nullable columns.
        
        Args:
            cursor: Database cursor
            table_name: Name of the table
            
        Returns:
            List of null value issues
        """
        issues = []
        
        # Get column information
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """, table_name.split('.'))
        
        columns = cursor.fetchall()
        
        for col_name, data_type, is_nullable in columns:
            if is_nullable == 'NO':
                # Check for nulls in non-nullable columns
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name}
                    WHERE {col_name} IS NULL
                """)
                null_count = cursor.fetchone()[0]
                
                if null_count > 0:
                    issues.append({
                        'type': 'null_values',
                        'column': col_name,
                        'count': null_count,
                        'severity': 'high'
                    })
        
        return issues
    
    def _check_duplicate_keys(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Check for duplicate keys in the table.
        
        Args:
            cursor: Database cursor
            table_name: Name of the table
            
        Returns:
            List of duplicate key issues
        """
        issues = []
        
        # Check for duplicate keys (currently only for article_metrics)
        if 'article_metrics' in table_name:
            cursor.execute(f"""
                SELECT article_id, time, COUNT(*)
                FROM {table_name}
                GROUP BY article_id, time
                HAVING COUNT(*) > 1
                LIMIT 10
            """)
            
            duplicates = cursor.fetchall()
            if duplicates:
                issues.append({
                    'type': 'duplicate_keys',
                    'count': len(duplicates),
                    'severity': 'high',
                    'sample': str(duplicates[:3])
                })
        
        return issues
    
    def _check_data_freshness(self, cursor, table_name: str) -> List[Dict[str, Any]]:
        """Check data freshness for time-based tables.
        
        Args:
            cursor: Database cursor
            table_name: Name of the table
            
        Returns:
            List of data freshness issues
        """
        issues = []
        
        cursor.execute(f"""
            SELECT MAX(time) as latest_time
            FROM {table_name}
            WHERE time IS NOT NULL
        """)
        
        latest_row = cursor.fetchone()
        if latest_row and latest_row[0]:
            age_hours = (datetime.utcnow() - latest_row[0]).total_seconds() / 3600
            
            if age_hours > 24:
                issues.append({
                    'type': 'stale_data',
                    'age_hours': age_hours,
                    'severity': 'medium' if age_hours < 48 else 'high'
                })
        
        return issues
    
    def _compile_check_results(self, issues: List[Dict[str, Any]]) -> Dict[str, str]:
        """Compile check results from issues list.
        
        Args:
            issues: List of validation issues
            
        Returns:
            Dictionary of check names to status
        """
        return {
            'null_check': 'passed' if not any(
                i['type'] == 'null_values' for i in issues
            ) else 'failed',
            'duplicate_check': 'passed' if not any(
                i['type'] == 'duplicate_keys' for i in issues
            ) else 'failed',
            'freshness_check': 'passed' if not any(
                i['type'] == 'stale_data' for i in issues
            ) else 'failed'
        }
    
    def _check_cross_table_consistency(self) -> List[Dict[str, Any]]:
        """Check consistency across tables."""
        issues = []
        
        try:
            with self.warehouse.get_connection() as conn:
                # Check article_id consistency
                cursor = conn.cursor()
                
                # Articles in metrics but not in main table
                cursor.execute("""
                    SELECT COUNT(DISTINCT am.article_id)
                    FROM analytics.article_metrics am
                    LEFT JOIN articles a ON am.article_id = a.id
                    WHERE a.id IS NULL
                """)
                
                orphaned_metrics = cursor.fetchone()[0]
                if orphaned_metrics > 0:
                    issues.append({
                        'type': 'orphaned_metrics',
                        'description': 'Article metrics exist for non-existent articles',
                        'count': orphaned_metrics,
                        'severity': 'high'
                    })
                
                # Check time consistency
                cursor.execute("""
                    SELECT 
                        MIN(time) as min_time,
                        MAX(time) as max_time
                    FROM analytics.article_metrics
                """)
                
                metrics_time_range = cursor.fetchone()
                
                cursor.execute("""
                    SELECT 
                        MIN(published_date) as min_date,
                        MAX(published_date) as max_date
                    FROM articles
                    WHERE published_date IS NOT NULL
                """)
                
                articles_time_range = cursor.fetchone()
                
                if metrics_time_range[0] and articles_time_range[0]:
                    if metrics_time_range[0] < articles_time_range[0]:
                        issues.append({
                            'type': 'time_inconsistency',
                            'description': 'Metrics exist before earliest article',
                            'severity': 'medium'
                        })
        
        except Exception as e:
            self.logger.error(f"Error checking cross-table consistency: {e}")
            issues.append({
                'type': 'consistency_check_error',
                'error': str(e),
                'severity': 'high'
            })
        
        return issues
    
    def _validate_article_metrics(self) -> Dict[str, Any]:
        """Validate article metrics calculations."""
        validation = {
            'status': 'passed',
            'issues': [],
            'checks_performed': []
        }
        
        try:
            # Sample recent metrics
            with self.warehouse.get_connection() as conn:
                query = """
                    SELECT 
                        am.*,
                        a.published_date,
                        a.source
                    FROM analytics.article_metrics am
                    JOIN articles a ON am.article_id = a.id
                    WHERE am.time >= NOW() - INTERVAL '7 days'
                    LIMIT 100
                """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    validation['issues'].append({
                        'type': 'no_recent_data',
                        'severity': 'high'
                    })
                    return validation
                
                # Check sentiment values
                invalid_sentiments = df[
                    (df['avg_sentiment'] < -1) | (df['avg_sentiment'] > 1)
                ]
                
                if not invalid_sentiments.empty:
                    validation['issues'].append({
                        'type': 'invalid_sentiment_values',
                        'count': len(invalid_sentiments),
                        'severity': 'high'
                    })
                
                # Check for reasonable article counts
                extreme_counts = df[df['article_count'] > 1000]
                if not extreme_counts.empty:
                    validation['issues'].append({
                        'type': 'extreme_article_counts',
                        'count': len(extreme_counts),
                        'max_value': extreme_counts['article_count'].max(),
                        'severity': 'medium'
                    })
                
                validation['checks_performed'] = [
                    'sentiment_range_check',
                    'article_count_reasonability',
                    'data_recency_check'
                ]
        
        except Exception as e:
            self.logger.error(f"Error validating article metrics: {e}")
            validation['status'] = 'failed'
            validation['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        if validation['issues']:
            validation['status'] = 'failed'
        
        return validation
    
    def _validate_source_metrics(self) -> Dict[str, Any]:
        """Validate source metrics calculations."""
        validation = {
            'status': 'passed',
            'issues': [],
            'checks_performed': []
        }
        
        try:
            with self.warehouse.get_connection() as conn:
                # Check source success rates
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT source, AVG(success_rate) as avg_success_rate
                    FROM analytics.source_metrics
                    WHERE time >= NOW() - INTERVAL '7 days'
                    GROUP BY source
                    HAVING AVG(success_rate) < 0 OR AVG(success_rate) > 1
                """)
                
                invalid_rates = cursor.fetchall()
                if invalid_rates:
                    validation['issues'].append({
                        'type': 'invalid_success_rates',
                        'sources': [row[0] for row in invalid_rates],
                        'severity': 'high'
                    })
                
                # Check for missing sources
                cursor.execute("""
                    SELECT DISTINCT source FROM articles
                    WHERE source NOT IN (
                        SELECT DISTINCT source FROM analytics.source_metrics
                    )
                    AND scraped_at >= NOW() - INTERVAL '7 days'
                """)
                
                missing_sources = cursor.fetchall()
                if missing_sources:
                    validation['issues'].append({
                        'type': 'missing_source_metrics',
                        'sources': [row[0] for row in missing_sources][:10],
                        'count': len(missing_sources),
                        'severity': 'medium'
                    })
                
                validation['checks_performed'] = [
                    'success_rate_validation',
                    'source_coverage_check'
                ]
        
        except Exception as e:
            self.logger.error(f"Error validating source metrics: {e}")
            validation['status'] = 'failed'
            validation['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        if validation['issues']:
            validation['status'] = 'failed'
        
        return validation
    
    def _validate_keyword_metrics(self) -> Dict[str, Any]:
        """Validate keyword metrics calculations."""
        validation = {
            'status': 'passed',
            'issues': [],
            'checks_performed': []
        }
        
        try:
            with self.warehouse.get_connection() as conn:
                # Check for reasonable growth rates
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT keyword, growth_rate
                    FROM analytics.keyword_metrics
                    WHERE ABS(growth_rate) > 10
                    LIMIT 10
                """)
                
                extreme_growth = cursor.fetchall()
                if extreme_growth:
                    validation['issues'].append({
                        'type': 'extreme_growth_rates',
                        'keywords': [row[0] for row in extreme_growth],
                        'severity': 'low'
                    })
                
                # Check for negative mention counts
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM analytics.keyword_metrics
                    WHERE mention_count < 0
                """)
                
                negative_counts = cursor.fetchone()[0]
                if negative_counts > 0:
                    validation['issues'].append({
                        'type': 'negative_mention_counts',
                        'count': negative_counts,
                        'severity': 'high'
                    })
                
                validation['checks_performed'] = [
                    'growth_rate_reasonability',
                    'mention_count_validation'
                ]
        
        except Exception as e:
            self.logger.error(f"Error validating keyword metrics: {e}")
            validation['status'] = 'failed'
            validation['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        if validation['issues']:
            validation['status'] = 'failed'
        
        return validation
    
    def _validate_trend_calculations(self) -> Dict[str, Any]:
        """Validate trend calculation accuracy."""
        validation = {
            'status': 'passed',
            'issues': [],
            'checks_performed': []
        }
        
        try:
            # Run a sample trend analysis
            sample_trends = self.trend_analyzer.analyze_article_volume_trends(
                days_back=7
            )
            
            if 'error' in sample_trends:
                validation['issues'].append({
                    'type': 'trend_calculation_error',
                    'error': sample_trends['error'],
                    'severity': 'high'
                })
            else:
                # Validate trend results
                if 'statistics' in sample_trends:
                    stats = sample_trends['statistics']
                    
                    # Check for NaN or infinite values
                    for key, value in stats.items():
                        if isinstance(value, (int, float)):
                            if np.isnan(value) or np.isinf(value):
                                validation['issues'].append({
                                    'type': 'invalid_statistic',
                                    'statistic': key,
                                    'value': str(value),
                                    'severity': 'high'
                                })
                
                validation['checks_performed'] = [
                    'trend_calculation_test',
                    'statistic_validation'
                ]
        
        except Exception as e:
            self.logger.error(f"Error validating trend calculations: {e}")
            validation['status'] = 'failed'
            validation['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        if validation['issues']:
            validation['status'] = 'failed'
        
        return validation
    
    def _validate_report_template(self, template_id: str) -> Dict[str, Any]:
        """Validate a specific report template."""
        validation = {
            'status': 'passed',
            'issues': [],
            'generation_time': 0
        }
        
        try:
            start_time = time.time()
            
            # Generate test report
            result = self.report_generator.generate_report(
                template_id=template_id,
                parameters={'time_period_days': 7},
                output_format='json'  # JSON for easy validation
            )
            
            generation_time = time.time() - start_time
            validation['generation_time'] = generation_time
            
            if 'error' in result:
                validation['issues'].append({
                    'type': 'generation_error',
                    'error': result['error'],
                    'severity': 'high'
                })
            else:
                # Validate report content
                if 'content' in result:
                    content = result['content']
                    
                    # Check for required sections
                    if isinstance(content, dict):
                        required_sections = ['summary', 'data']
                        
                        for section in required_sections:
                            if section not in content:
                                validation['issues'].append({
                                    'type': 'missing_section',
                                    'section': section,
                                    'severity': 'medium'
                                })
                        
                        # Check data completeness
                        if 'data' in content and isinstance(content['data'], dict):
                            for key, value in content['data'].items():
                                if value is None:
                                    validation['issues'].append({
                                        'type': 'null_data',
                                        'field': key,
                                        'severity': 'low'
                                    })
                
                # Check generation time
                if generation_time > 30:  # 30 seconds threshold
                    validation['issues'].append({
                        'type': 'slow_generation',
                        'time_seconds': generation_time,
                        'severity': 'medium'
                    })
        
        except Exception as e:
            self.logger.error(f"Error validating report template {template_id}: {e}")
            validation['status'] = 'failed'
            validation['issues'].append({
                'type': 'validation_error',
                'error': str(e),
                'severity': 'critical'
            })
        
        if validation['issues']:
            validation['status'] = 'failed'
        
        return validation
    
    # Performance benchmarking methods
    
    def _benchmark_data_collection(self) -> Dict[str, Any]:
        """Benchmark data collection performance."""
        benchmark = {
            'operations': {},
            'total_time': 0
        }
        
        try:
            # Benchmark metrics collection
            start_time = time.time()
            
            # Run a limited collection
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                # Collect article metrics for last hour
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM articles 
                    WHERE scraped_at >= NOW() - INTERVAL '1 hour'
                """)
                
                article_count = cursor.fetchone()[0]
            
            collection_time = time.time() - start_time
            
            benchmark['operations']['recent_article_count'] = {
                'time': collection_time,
                'records': article_count,
                'rate': article_count / collection_time if collection_time > 0 else 0
            }
            
            # Benchmark aggregation
            start_time = time.time()
            
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        source,
                        COUNT(*) as count,
                        AVG(CHAR_LENGTH(content)) as avg_length
                    FROM articles
                    WHERE scraped_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY source
                """)
                
                aggregation_results = cursor.fetchall()
            
            aggregation_time = time.time() - start_time
            
            benchmark['operations']['source_aggregation'] = {
                'time': aggregation_time,
                'groups': len(aggregation_results),
                'rate': len(aggregation_results) / aggregation_time if aggregation_time > 0 else 0
            }
            
            benchmark['total_time'] = sum(
                op['time'] for op in benchmark['operations'].values()
            )
        
        except Exception as e:
            self.logger.error(f"Benchmark error: {e}")
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_queries(self) -> Dict[str, Any]:
        """Benchmark common query performance."""
        benchmark = {
            'queries': {},
            'total_time': 0
        }
        
        queries_to_test = [
            {
                'name': 'article_trends',
                'query': """
                    SELECT 
                        time_bucket('1 hour', time) as hour,
                        SUM(article_count) as total
                    FROM analytics.article_metrics
                    WHERE time >= NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour
                """
            },
            {
                'name': 'top_keywords',
                'query': """
                    SELECT 
                        keyword,
                        SUM(mention_count) as total_mentions
                    FROM analytics.keyword_metrics
                    WHERE time >= NOW() - INTERVAL '7 days'
                    GROUP BY keyword
                    ORDER BY total_mentions DESC
                    LIMIT 100
                """
            },
            {
                'name': 'source_performance',
                'query': """
                    SELECT 
                        source,
                        AVG(success_rate) as avg_success,
                        SUM(article_count) as total_articles
                    FROM analytics.source_metrics
                    WHERE time >= NOW() - INTERVAL '7 days'
                    GROUP BY source
                    ORDER BY total_articles DESC
                    LIMIT 50
                """
            }
        ]
        
        for query_info in queries_to_test:
            try:
                start_time = time.time()
                
                with self.warehouse.get_connection() as conn:
                    df = pd.read_sql_query(query_info['query'], conn)
                
                query_time = time.time() - start_time
                
                benchmark['queries'][query_info['name']] = {
                    'time': query_time,
                    'rows': len(df),
                    'performance': 'good' if query_time < 1 else 'acceptable' if query_time < 5 else 'slow'
                }
            
            except Exception as e:
                benchmark['queries'][query_info['name']] = {
                    'error': str(e),
                    'performance': 'failed'
                }
        
        benchmark['total_time'] = sum(
            q.get('time', 0) for q in benchmark['queries'].values()
        )
        
        return benchmark
    
    def _benchmark_trend_analysis(self) -> Dict[str, Any]:
        """Benchmark trend analysis performance."""
        benchmark = {
            'analyses': {},
            'total_time': 0
        }
        
        try:
            # Benchmark volume trend analysis
            start_time = time.time()
            volume_trends = self.trend_analyzer.analyze_article_volume_trends(days_back=7)
            volume_time = time.time() - start_time
            
            benchmark['analyses']['volume_trends'] = {
                'time': volume_time,
                'success': 'error' not in volume_trends,
                'performance': 'good' if volume_time < 2 else 'acceptable' if volume_time < 10 else 'slow'
            }
            
            # Benchmark keyword trend analysis
            start_time = time.time()
            keyword_trends = self.trend_analyzer.analyze_keyword_trends(
                days_back=7,
                top_k=10
            )
            keyword_time = time.time() - start_time
            
            benchmark['analyses']['keyword_trends'] = {
                'time': keyword_time,
                'keywords_analyzed': len(keyword_trends) if isinstance(keyword_trends, list) else 0,
                'performance': 'good' if keyword_time < 5 else 'acceptable' if keyword_time < 15 else 'slow'
            }
            
            benchmark['total_time'] = sum(
                a['time'] for a in benchmark['analyses'].values()
            )
        
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_predictions(self) -> Dict[str, Any]:
        """Benchmark prediction performance."""
        benchmark = {
            'predictions': {},
            'total_time': 0
        }
        
        try:
            # Benchmark volume forecasting
            start_time = time.time()
            volume_forecast = self.predictive_analytics.forecast_article_volume(
                forecast_days=7,
                model_type='prophet'
            )
            forecast_time = time.time() - start_time
            
            benchmark['predictions']['volume_forecast'] = {
                'time': forecast_time,
                'success': 'error' not in volume_forecast,
                'performance': 'good' if forecast_time < 5 else 'acceptable' if forecast_time < 20 else 'slow'
            }
            
            # Benchmark topic prediction
            start_time = time.time()
            topic_predictions = self.predictive_analytics.predict_topic_trends(
                prediction_days=3
            )
            topic_time = time.time() - start_time
            
            benchmark['predictions']['topic_trends'] = {
                'time': topic_time,
                'topics_predicted': len(topic_predictions.get('top_trending_keywords', [])),
                'performance': 'good' if topic_time < 10 else 'acceptable' if topic_time < 30 else 'slow'
            }
            
            benchmark['total_time'] = sum(
                p['time'] for p in benchmark['predictions'].values()
            )
        
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_report_generation(self) -> Dict[str, Any]:
        """Benchmark report generation performance."""
        benchmark = {
            'reports': {},
            'total_time': 0
        }
        
        templates_to_test = ['executive_summary', 'source_performance']
        
        for template_id in templates_to_test:
            try:
                start_time = time.time()
                
                result = self.report_generator.generate_report(
                    template_id=template_id,
                    parameters={'time_period_days': 7},
                    output_format='html'
                )
                
                generation_time = time.time() - start_time
                
                benchmark['reports'][template_id] = {
                    'time': generation_time,
                    'success': 'error' not in result,
                    'file_size': result.get('file_size', 0),
                    'performance': 'good' if generation_time < 5 else 'acceptable' if generation_time < 15 else 'slow'
                }
            
            except Exception as e:
                benchmark['reports'][template_id] = {
                    'error': str(e),
                    'performance': 'failed'
                }
        
        benchmark['total_time'] = sum(
            r.get('time', 0) for r in benchmark['reports'].values()
        )
        
        return benchmark
    
    def _calculate_performance_rating(self, benchmarks: Dict[str, Any]) -> str:
        """Calculate overall performance rating."""
        total_operations = 0
        good_operations = 0
        acceptable_operations = 0
        slow_operations = 0
        failed_operations = 0
        
        for category, results in benchmarks.items():
            for operation, metrics in results.items():
                if operation in ['total_time', 'error']:
                    continue
                
                if isinstance(metrics, dict):
                    total_operations += 1
                    performance = metrics.get('performance', 'unknown')
                    
                    if performance == 'good':
                        good_operations += 1
                    elif performance == 'acceptable':
                        acceptable_operations += 1
                    elif performance == 'slow':
                        slow_operations += 1
                    elif performance == 'failed':
                        failed_operations += 1
        
        if total_operations == 0:
            return 'unknown'
        
        good_ratio = good_operations / total_operations
        failed_ratio = failed_operations / total_operations
        
        if failed_ratio > 0.1:
            return 'poor'
        elif good_ratio > 0.8:
            return 'excellent'
        elif good_ratio > 0.6:
            return 'good'
        elif good_ratio > 0.4:
            return 'acceptable'
        else:
            return 'needs_improvement'
    
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results to file."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Validation results saved to {filename}")