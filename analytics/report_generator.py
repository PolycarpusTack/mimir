"""Custom Report Generator for Mimir Analytics.

This module provides comprehensive report generation capabilities including
template-based reports, PDF generation, scheduled reporting, and custom
report builder functionality.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, Flowable
)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('seaborn-v0_8')
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .data_warehouse import AnalyticsDataWarehouse
from .trend_analyzer import TrendAnalyzer
from .predictive_analytics import PredictiveAnalytics

logger = logging.getLogger(__name__)


class ReportTemplate:
    """Represents a report template with metadata and configuration."""
    
    def __init__(self, template_id: str, name: str, description: str, 
                 template_file: str, parameters: Dict[str, Any]):
        """Initialize a report template.
        
        Args:
            template_id: Unique template identifier
            name: Human-readable template name
            description: Template description
            template_file: Path to Jinja2 template file
            parameters: Template parameters with defaults and validation
        """
        self.template_id = template_id
        self.name = name
        self.description = description
        self.template_file = template_file
        self.parameters = parameters
        self.created_at = datetime.utcnow()


class ReportGenerator:
    """Advanced report generation engine with template support."""
    
    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the report generator.
        
        Args:
            analytics_warehouse: Analytics data warehouse for data access
        """
        self.analytics_warehouse = analytics_warehouse
        self.trend_analyzer = TrendAnalyzer(analytics_warehouse)
        self.predictive_analytics = PredictiveAnalytics(analytics_warehouse)
        
        # Setup directories
        self.reports_dir = Path("analytics/reports")
        self.templates_dir = Path("analytics/templates")
        self.exports_dir = Path("analytics/exports")
        
        for directory in [self.reports_dir, self.templates_dir, self.exports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Initialize report templates
        self.templates = {}
        self._initialize_default_templates()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_default_templates(self):
        """Initialize default report templates."""
        
        # Executive Summary Template
        self.templates['executive_summary'] = ReportTemplate(
            template_id='executive_summary',
            name='Executive Summary Report',
            description='High-level overview of key metrics and trends',
            template_file='executive_summary.html',
            parameters={
                'time_period_days': {'default': 30, 'type': 'int', 'min': 1, 'max': 365},
                'include_forecasts': {'default': True, 'type': 'bool'},
                'include_sources': {'default': True, 'type': 'bool'},
                'include_keywords': {'default': True, 'type': 'bool'},
                'keyword_limit': {'default': 20, 'type': 'int', 'min': 5, 'max': 100}
            }
        )
        
        # Source Performance Template
        self.templates['source_performance'] = ReportTemplate(
            template_id='source_performance',
            name='Source Performance Analysis',
            description='Detailed analysis of news source performance and reliability',
            template_file='source_performance.html',
            parameters={
                'time_period_days': {'default': 14, 'type': 'int', 'min': 1, 'max': 90},
                'sources': {'default': None, 'type': 'list', 'required': False},
                'include_quality_metrics': {'default': True, 'type': 'bool'},
                'include_trends': {'default': True, 'type': 'bool'},
                'min_articles': {'default': 10, 'type': 'int', 'min': 1}
            }
        )
        
        # Keyword Trends Template
        self.templates['keyword_trends'] = ReportTemplate(
            template_id='keyword_trends',
            name='Keyword Trends Analysis',
            description='Comprehensive keyword and topic trend analysis',
            template_file='keyword_trends.html',
            parameters={
                'time_period_days': {'default': 30, 'type': 'int', 'min': 7, 'max': 180},
                'keywords': {'default': None, 'type': 'list', 'required': False},
                'min_mentions': {'default': 5, 'type': 'int', 'min': 1},
                'include_sentiment': {'default': True, 'type': 'bool'},
                'include_predictions': {'default': True, 'type': 'bool'},
                'trend_threshold': {'default': 10.0, 'type': 'float', 'min': 0.0}
            }
        )
        
        # System Performance Template
        self.templates['system_performance'] = ReportTemplate(
            template_id='system_performance',
            name='System Performance Report',
            description='Technical performance metrics and system health analysis',
            template_file='system_performance.html',
            parameters={
                'time_period_days': {'default': 7, 'type': 'int', 'min': 1, 'max': 30},
                'include_alerts': {'default': True, 'type': 'bool'},
                'include_recommendations': {'default': True, 'type': 'bool'},
                'detailed_metrics': {'default': False, 'type': 'bool'}
            }
        )
        
        # Custom Analysis Template
        self.templates['custom_analysis'] = ReportTemplate(
            template_id='custom_analysis',
            name='Custom Analytics Report',
            description='Flexible report template for custom analysis requirements',
            template_file='custom_analysis.html',
            parameters={
                'time_period_days': {'default': 30, 'type': 'int', 'min': 1, 'max': 365},
                'analysis_types': {'default': ['trends', 'forecasts'], 'type': 'list'},
                'custom_title': {'default': 'Custom Analytics Report', 'type': 'str'},
                'custom_description': {'default': '', 'type': 'str', 'required': False},
                'include_charts': {'default': True, 'type': 'bool'},
                'chart_types': {'default': ['line', 'bar'], 'type': 'list'}
            }
        )
    
    def generate_report(self, template_id: str, parameters: Dict[str, Any] = None,
                       output_format: str = 'html', output_file: str = None) -> Dict[str, Any]:
        """Generate a report using specified template and parameters.
        
        Args:
            template_id: Template identifier
            parameters: Report parameters (overrides defaults)
            output_format: Output format ('html', 'pdf', 'json')
            output_file: Optional output file path
            
        Returns:
            Report generation results with file path and metadata
        """
        try:
            if template_id not in self.templates:
                raise ValueError(f"Unknown template: {template_id}")
            
            template = self.templates[template_id]
            
            # Merge parameters with defaults
            final_params = self._merge_parameters(template.parameters, parameters or {})
            
            # Validate parameters
            validation_errors = self._validate_parameters(template.parameters, final_params)
            if validation_errors:
                raise ValueError(f"Parameter validation failed: {validation_errors}")
            
            # Generate report data
            report_data = self._generate_report_data(template_id, final_params)
            
            # Generate output based on format
            if output_format.lower() == 'html':
                result = self._generate_html_report(template, report_data, final_params, output_file)
            elif output_format.lower() == 'pdf':
                result = self._generate_pdf_report(template, report_data, final_params, output_file)
            elif output_format.lower() == 'json':
                result = self._generate_json_report(template, report_data, final_params, output_file)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Add metadata
            result.update({
                'template_id': template_id,
                'template_name': template.name,
                'parameters': final_params,
                'generated_at': datetime.utcnow().isoformat(),
                'format': output_format
            })
            
            self.logger.info(f"Generated {output_format} report: {template.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate report {template_id}: {e}")
            return {'error': str(e)}
    
    def _generate_report_data(self, template_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data for the report based on template and parameters."""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=parameters['time_period_days'])
        
        report_data = {
            'report_metadata': {
                'template_id': template_id,
                'generated_at': end_time,
                'period_start': start_time,
                'period_end': end_time,
                'period_days': parameters['time_period_days']
            }
        }
        
        if template_id == 'executive_summary':
            report_data.update(self._generate_executive_summary_data(parameters, start_time, end_time))
        elif template_id == 'source_performance':
            report_data.update(self._generate_source_performance_data(parameters, start_time, end_time))
        elif template_id == 'keyword_trends':
            report_data.update(self._generate_keyword_trends_data(parameters, start_time, end_time))
        elif template_id == 'system_performance':
            report_data.update(self._generate_system_performance_data(parameters, start_time, end_time))
        elif template_id == 'custom_analysis':
            report_data.update(self._generate_custom_analysis_data(parameters, start_time, end_time))
        
        return report_data
    
    def _generate_executive_summary_data(self, params: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate data for executive summary report."""
        data = {}
        
        # Overall metrics
        with self.analytics_warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # Article volume metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(DISTINCT source) as unique_sources,
                    AVG(article_count) as avg_hourly_volume,
                    MAX(article_count) as peak_hourly_volume,
                    AVG(CASE WHEN avg_sentiment IS NOT NULL THEN avg_sentiment END) as overall_sentiment
                FROM analytics.article_metrics 
                WHERE time >= %s AND time <= %s
            """, [start_time, end_time])
            
            metrics = cursor.fetchone()
            data['overview_metrics'] = {
                'total_articles': metrics[0] or 0,
                'unique_sources': metrics[1] or 0,
                'avg_hourly_volume': round(metrics[2] or 0, 2),
                'peak_hourly_volume': metrics[3] or 0,
                'overall_sentiment': round(metrics[4] or 0, 3)
            }
            
            # Growth comparison (current vs previous period)
            prev_start = start_time - timedelta(days=params['time_period_days'])
            cursor.execute("""
                SELECT COUNT(*) FROM analytics.article_metrics 
                WHERE time >= %s AND time < %s
            """, [prev_start, start_time])
            
            previous_articles = cursor.fetchone()[0] or 0
            current_articles = data['overview_metrics']['total_articles']
            
            if previous_articles > 0:
                growth_rate = ((current_articles - previous_articles) / previous_articles) * 100
            else:
                growth_rate = 100.0 if current_articles > 0 else 0.0
            
            data['overview_metrics']['growth_rate'] = round(growth_rate, 2)
        
        # Top sources performance
        if params.get('include_sources', True):
            source_data = self.analytics_warehouse.get_source_performance(start_time, end_time)
            data['top_sources'] = sorted(source_data, key=lambda x: x['total_published'] or 0, reverse=True)[:10]
        
        # Trending keywords
        if params.get('include_keywords', True):
            keyword_data = self.analytics_warehouse.get_keyword_trends(
                start_time=start_time,
                end_time=end_time,
                limit=params.get('keyword_limit', 20)
            )
            
            # Group and rank keywords
            keyword_summary = {}
            for row in keyword_data:
                keyword = row['keyword']
                if keyword not in keyword_summary:
                    keyword_summary[keyword] = {
                        'total_mentions': 0,
                        'total_articles': 0,
                        'avg_sentiment': [],
                        'max_growth': 0
                    }
                
                keyword_summary[keyword]['total_mentions'] += row['total_mentions'] or 0
                keyword_summary[keyword]['total_articles'] += row['total_articles'] or 0
                if row['avg_sentiment'] is not None:
                    keyword_summary[keyword]['avg_sentiment'].append(row['avg_sentiment'])
                if row['max_growth_rate'] and row['max_growth_rate'] > keyword_summary[keyword]['max_growth']:
                    keyword_summary[keyword]['max_growth'] = row['max_growth_rate']
            
            # Calculate averages and sort
            trending_keywords = []
            for keyword, stats in keyword_summary.items():
                avg_sentiment = sum(stats['avg_sentiment']) / len(stats['avg_sentiment']) if stats['avg_sentiment'] else None
                trending_keywords.append({
                    'keyword': keyword,
                    'total_mentions': stats['total_mentions'],
                    'total_articles': stats['total_articles'],
                    'avg_sentiment': round(avg_sentiment, 3) if avg_sentiment else None,
                    'max_growth': round(stats['max_growth'], 2)
                })
            
            data['trending_keywords'] = sorted(trending_keywords, key=lambda x: x['total_mentions'], reverse=True)
        
        # Forecasts and predictions
        if params.get('include_forecasts', True):
            try:
                volume_forecast = self.predictive_analytics.forecast_article_volume(forecast_days=7)
                sentiment_forecast = self.predictive_analytics.forecast_sentiment_trends(forecast_days=7)
                
                data['forecasts'] = {
                    'volume_forecast': volume_forecast,
                    'sentiment_forecast': sentiment_forecast
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate forecasts for executive summary: {e}")
                data['forecasts'] = {'error': str(e)}
        
        # Trend analysis summary
        try:
            trend_analysis = self.trend_analyzer.analyze_article_volume_trends(
                days_back=params['time_period_days']
            )
            data['trend_summary'] = {
                'trend_direction': trend_analysis.get('trend_direction', 'unknown'),
                'trend_strength': trend_analysis.get('trend_strength', 0),
                'anomalies_detected': trend_analysis.get('anomalies_detected', 0),
                'has_seasonality': trend_analysis.get('has_seasonality', False)
            }
        except Exception as e:
            self.logger.warning(f"Failed to generate trend analysis: {e}")
            data['trend_summary'] = {'error': str(e)}
        
        return data
    
    def _generate_source_performance_data(self, params: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate data for source performance report."""
        data = {}
        
        # Get source performance data
        sources_filter = params.get('sources')
        source_data = self.analytics_warehouse.get_source_performance(start_time, end_time)
        
        # Filter sources if specified
        if sources_filter:
            source_data = [s for s in source_data if s['source'] in sources_filter]
        
        # Filter by minimum articles
        min_articles = params.get('min_articles', 10)
        source_data = [s for s in source_data if (s['total_published'] or 0) >= min_articles]
        
        data['source_performance'] = source_data
        
        # Performance rankings
        data['performance_rankings'] = {
            'top_volume': sorted(source_data, key=lambda x: x['total_published'] or 0, reverse=True)[:10],
            'top_success_rate': sorted(source_data, key=lambda x: x['avg_success_rate'] or 0, reverse=True)[:10],
            'fastest_sources': sorted([s for s in source_data if s['avg_scrape_time']], 
                                    key=lambda x: x['avg_scrape_time'], reverse=False)[:10]
        }
        
        # Quality metrics
        if params.get('include_quality_metrics', True):
            quality_metrics = []
            for source in source_data:
                quality_score = 0
                factors = []
                
                # Success rate factor (40%)
                success_rate = source['avg_success_rate'] or 0
                quality_score += success_rate * 0.4
                factors.append(f"Success Rate: {success_rate:.1%}")
                
                # Content length factor (20%)
                avg_length = source['avg_content_length'] or 0
                length_score = min(avg_length / 2000, 1.0)  # Normalize to 2000 chars
                quality_score += length_score * 0.2
                factors.append(f"Avg Length: {avg_length:.0f} chars")
                
                # Sentiment consistency factor (20%)
                avg_sentiment = source['avg_sentiment']
                if avg_sentiment is not None:
                    sentiment_score = 1.0 - abs(avg_sentiment)  # Closer to neutral = more consistent
                    quality_score += sentiment_score * 0.2
                    factors.append(f"Sentiment: {avg_sentiment:.3f}")
                
                # Error rate factor (20%)
                total_scraped = source['total_scraped'] or 1
                error_rate = (source['total_errors'] or 0) / total_scraped
                error_score = max(0, 1.0 - error_rate * 5)  # Penalize high error rates
                quality_score += error_score * 0.2
                factors.append(f"Error Rate: {error_rate:.1%}")
                
                quality_metrics.append({
                    'source': source['source'],
                    'quality_score': round(quality_score, 3),
                    'factors': factors
                })
            
            data['quality_metrics'] = sorted(quality_metrics, key=lambda x: x['quality_score'], reverse=True)
        
        # Trend analysis per source
        if params.get('include_trends', True):
            source_trends = []
            for source in source_data[:20]:  # Limit to top 20 sources
                try:
                    trend_data = self.analytics_warehouse.get_article_trends(
                        sources=[source['source']],
                        start_time=start_time,
                        end_time=end_time,
                        interval='1 day'
                    )
                    
                    if len(trend_data) >= 3:
                        # Simple trend calculation
                        volumes = [row['article_count'] for row in trend_data]
                        if len(volumes) >= 2:
                            recent_avg = sum(volumes[-3:]) / min(3, len(volumes))
                            early_avg = sum(volumes[:3]) / min(3, len(volumes))
                            trend_direction = 'up' if recent_avg > early_avg else 'down' if recent_avg < early_avg else 'stable'
                            trend_change = ((recent_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
                        else:
                            trend_direction = 'stable'
                            trend_change = 0
                        
                        source_trends.append({
                            'source': source['source'],
                            'trend_direction': trend_direction,
                            'trend_change_percent': round(trend_change, 2),
                            'data_points': len(trend_data)
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to analyze trend for source {source['source']}: {e}")
            
            data['source_trends'] = source_trends
        
        return data
    
    def _generate_keyword_trends_data(self, params: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate data for keyword trends report."""
        data = {}
        
        # Get keyword data
        keywords_filter = params.get('keywords')
        keyword_data = self.analytics_warehouse.get_keyword_trends(
            keywords=keywords_filter,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # Process and aggregate keyword data
        keyword_summary = {}
        for row in keyword_data:
            keyword = row['keyword']
            if keyword not in keyword_summary:
                keyword_summary[keyword] = {
                    'keyword': keyword,
                    'total_mentions': 0,
                    'total_articles': 0,
                    'sentiment_values': [],
                    'growth_rates': [],
                    'momentum_scores': []
                }
            
            summary = keyword_summary[keyword]
            summary['total_mentions'] += row['total_mentions'] or 0
            summary['total_articles'] += row['total_articles'] or 0
            
            if row['avg_sentiment'] is not None:
                summary['sentiment_values'].append(row['avg_sentiment'])
            if row['max_growth_rate'] is not None:
                summary['growth_rates'].append(row['max_growth_rate'])
            if row['avg_momentum'] is not None:
                summary['momentum_scores'].append(row['avg_momentum'])
        
        # Calculate final metrics and filter
        min_mentions = params.get('min_mentions', 5)
        processed_keywords = []
        
        for keyword, summary in keyword_summary.items():
            if summary['total_mentions'] < min_mentions:
                continue
            
            # Calculate averages
            avg_sentiment = sum(summary['sentiment_values']) / len(summary['sentiment_values']) if summary['sentiment_values'] else None
            avg_growth = sum(summary['growth_rates']) / len(summary['growth_rates']) if summary['growth_rates'] else 0
            avg_momentum = sum(summary['momentum_scores']) / len(summary['momentum_scores']) if summary['momentum_scores'] else 0
            
            processed_keywords.append({
                'keyword': keyword,
                'total_mentions': summary['total_mentions'],
                'total_articles': summary['total_articles'],
                'avg_sentiment': round(avg_sentiment, 3) if avg_sentiment else None,
                'avg_growth_rate': round(avg_growth, 2),
                'avg_momentum': round(avg_momentum, 2),
                'sentiment_consistency': round(1.0 - (np.std(summary['sentiment_values']) if len(summary['sentiment_values']) > 1 else 0), 3)
            })
        
        # Sort and categorize
        data['all_keywords'] = sorted(processed_keywords, key=lambda x: x['total_mentions'], reverse=True)
        
        # Trending analysis
        trend_threshold = params.get('trend_threshold', 10.0)
        data['trending_up'] = [k for k in processed_keywords if k['avg_growth_rate'] > trend_threshold]
        data['trending_down'] = [k for k in processed_keywords if k['avg_growth_rate'] < -trend_threshold]
        data['stable_keywords'] = [k for k in processed_keywords if abs(k['avg_growth_rate']) <= trend_threshold]
        
        # Sentiment analysis
        if params.get('include_sentiment', True):
            sentiment_keywords = [k for k in processed_keywords if k['avg_sentiment'] is not None]
            data['sentiment_analysis'] = {
                'most_positive': sorted(sentiment_keywords, key=lambda x: x['avg_sentiment'], reverse=True)[:10],
                'most_negative': sorted(sentiment_keywords, key=lambda x: x['avg_sentiment'])[:10],
                'most_neutral': sorted(sentiment_keywords, key=lambda x: abs(x['avg_sentiment'] or 0))[:10]
            }
        
        # Predictions
        if params.get('include_predictions', True):
            try:
                predictions = self.predictive_analytics.predict_topic_trends(
                    keywords=[k['keyword'] for k in data['all_keywords'][:50]],  # Top 50 keywords
                    prediction_days=7
                )
                data['predictions'] = predictions
            except Exception as e:
                self.logger.warning(f"Failed to generate keyword predictions: {e}")
                data['predictions'] = {'error': str(e)}
        
        return data
    
    def _generate_system_performance_data(self, params: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate data for system performance report."""
        data = {}
        
        with self.analytics_warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # System metrics overview
            cursor.execute("""
                SELECT 
                    AVG(cpu_usage_percent) as avg_cpu,
                    MAX(cpu_usage_percent) as max_cpu,
                    AVG(memory_usage_mb) as avg_memory,
                    MAX(memory_usage_mb) as max_memory,
                    AVG(active_connections) as avg_connections,
                    MAX(active_connections) as max_connections,
                    AVG(articles_per_minute) as avg_article_rate,
                    MAX(articles_per_minute) as max_article_rate,
                    AVG(error_rate_percent) as avg_error_rate,
                    COUNT(*) as data_points
                FROM analytics.system_metrics 
                WHERE time >= %s AND time <= %s
            """, [start_time, end_time])
            
            metrics = cursor.fetchone()
            data['system_overview'] = {
                'avg_cpu_usage': round(metrics[0] or 0, 2),
                'max_cpu_usage': round(metrics[1] or 0, 2),
                'avg_memory_mb': round(metrics[2] or 0),
                'max_memory_mb': round(metrics[3] or 0),
                'avg_connections': round(metrics[4] or 0, 1),
                'max_connections': metrics[5] or 0,
                'avg_article_rate': round(metrics[6] or 0, 2),
                'max_article_rate': round(metrics[7] or 0, 2),
                'avg_error_rate': round(metrics[8] or 0, 2),
                'monitoring_coverage': metrics[9] or 0
            }
            
            # Performance trends
            cursor.execute("""
                SELECT 
                    time_bucket('1 hour', time) as hour,
                    AVG(cpu_usage_percent) as avg_cpu,
                    AVG(memory_usage_mb) as avg_memory,
                    AVG(articles_per_minute) as avg_rate
                FROM analytics.system_metrics 
                WHERE time >= %s AND time <= %s
                GROUP BY hour
                ORDER BY hour
            """, [start_time, end_time])
            
            hourly_data = cursor.fetchall()
            data['performance_trends'] = [
                {
                    'hour': row[0].isoformat(),
                    'cpu_usage': round(row[1] or 0, 2),
                    'memory_usage': round(row[2] or 0),
                    'article_rate': round(row[3] or 0, 2)
                }
                for row in hourly_data
            ]
            
            # Source-specific performance
            cursor.execute("""
                SELECT 
                    source,
                    AVG(success_rate) as avg_success_rate,
                    AVG(scrape_duration_ms) as avg_duration,
                    SUM(error_count) as total_errors,
                    COUNT(*) as measurements
                FROM analytics.source_metrics 
                WHERE time >= %s AND time <= %s
                GROUP BY source
                HAVING COUNT(*) >= 3
                ORDER BY avg_success_rate DESC
            """, [start_time, end_time])
            
            source_performance = cursor.fetchall()
            data['source_performance'] = [
                {
                    'source': row[0],
                    'success_rate': round(row[1] or 0, 4),
                    'avg_duration_ms': round(row[2] or 0),
                    'total_errors': row[3] or 0,
                    'measurements': row[4]
                }
                for row in source_performance
            ]
        
        # Alert recommendations
        if params.get('include_alerts', True):
            try:
                alert_thresholds = self.predictive_analytics.learn_alert_thresholds(
                    metric_type='article_volume',
                    learning_days=params['time_period_days']
                )
                data['alert_recommendations'] = alert_thresholds
            except Exception as e:
                self.logger.warning(f"Failed to generate alert recommendations: {e}")
                data['alert_recommendations'] = {'error': str(e)}
        
        # Performance recommendations
        if params.get('include_recommendations', True):
            recommendations = []
            
            overview = data['system_overview']
            
            # CPU recommendations
            if overview['avg_cpu_usage'] > 80:
                recommendations.append({
                    'type': 'warning',
                    'metric': 'CPU Usage',
                    'current_value': f"{overview['avg_cpu_usage']:.1f}%",
                    'recommendation': 'Consider scaling up CPU resources or optimizing scraping processes'
                })
            elif overview['avg_cpu_usage'] > 60:
                recommendations.append({
                    'type': 'info',
                    'metric': 'CPU Usage',
                    'current_value': f"{overview['avg_cpu_usage']:.1f}%",
                    'recommendation': 'Monitor CPU usage trends for potential scaling needs'
                })
            
            # Memory recommendations
            memory_gb = overview['avg_memory_mb'] / 1024
            if memory_gb > 8:
                recommendations.append({
                    'type': 'warning',
                    'metric': 'Memory Usage',
                    'current_value': f"{memory_gb:.1f}GB",
                    'recommendation': 'High memory usage detected. Consider memory optimization or scaling'
                })
            
            # Error rate recommendations
            if overview['avg_error_rate'] > 5:
                recommendations.append({
                    'type': 'error',
                    'metric': 'Error Rate',
                    'current_value': f"{overview['avg_error_rate']:.1f}%",
                    'recommendation': 'High error rate requires immediate attention. Check source configurations'
                })
            
            # Article rate recommendations
            if overview['avg_article_rate'] < 1:
                recommendations.append({
                    'type': 'warning',
                    'metric': 'Article Processing Rate',
                    'current_value': f"{overview['avg_article_rate']:.2f}/min",
                    'recommendation': 'Low processing rate. Check scraper performance and source availability'
                })
            
            data['recommendations'] = recommendations
        
        return data
    
    def _generate_custom_analysis_data(self, params: Dict, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate data for custom analysis report."""
        data = {
            'custom_title': params.get('custom_title', 'Custom Analytics Report'),
            'custom_description': params.get('custom_description', ''),
            'analysis_types': params.get('analysis_types', ['trends'])
        }
        
        analysis_types = params.get('analysis_types', [])
        
        # Trends analysis
        if 'trends' in analysis_types:
            try:
                trend_results = self.trend_analyzer.run_comprehensive_analysis(
                    days_back=params['time_period_days'],
                    include_keywords=True,
                    include_sources=True
                )
                data['trend_analysis'] = trend_results
            except Exception as e:
                self.logger.warning(f"Failed to generate trend analysis: {e}")
                data['trend_analysis'] = {'error': str(e)}
        
        # Forecasts
        if 'forecasts' in analysis_types:
            try:
                volume_forecast = self.predictive_analytics.forecast_article_volume(forecast_days=7)
                sentiment_forecast = self.predictive_analytics.forecast_sentiment_trends(forecast_days=7)
                topic_predictions = self.predictive_analytics.predict_topic_trends(prediction_days=7)
                
                data['forecasts'] = {
                    'volume': volume_forecast,
                    'sentiment': sentiment_forecast,
                    'topics': topic_predictions
                }
            except Exception as e:
                self.logger.warning(f"Failed to generate forecasts: {e}")
                data['forecasts'] = {'error': str(e)}
        
        # Performance analysis
        if 'performance' in analysis_types:
            source_data = self.analytics_warehouse.get_source_performance(start_time, end_time)
            data['performance_analysis'] = {
                'source_performance': source_data,
                'top_performers': sorted(source_data, key=lambda x: x['avg_success_rate'] or 0, reverse=True)[:10]
            }
        
        # Custom metrics
        if 'metrics' in analysis_types:
            with self.analytics_warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                # Custom aggregations
                cursor.execute("""
                    SELECT 
                        DATE(time) as date,
                        COUNT(*) as daily_articles,
                        AVG(article_count) as avg_hourly,
                        COUNT(DISTINCT source) as active_sources
                    FROM analytics.article_metrics 
                    WHERE time >= %s AND time <= %s
                    GROUP BY DATE(time)
                    ORDER BY date
                """, [start_time, end_time])
                
                daily_metrics = cursor.fetchall()
                data['custom_metrics'] = [
                    {
                        'date': row[0].isoformat(),
                        'daily_articles': row[1],
                        'avg_hourly': round(row[2] or 0, 2),
                        'active_sources': row[3]
                    }
                    for row in daily_metrics
                ]
        
        return data
    
    def _generate_html_report(self, template: ReportTemplate, report_data: Dict, 
                             parameters: Dict, output_file: str = None) -> Dict[str, Any]:
        """Generate HTML report."""
        try:
            # Create HTML template if it doesn't exist
            template_path = self.templates_dir / template.template_file
            if not template_path.exists():
                self._create_default_html_template(template.template_id, template_path)
            
            # Load and render template
            jinja_template = self.jinja_env.get_template(template.template_file)
            html_content = jinja_template.render(
                report_data=report_data,
                parameters=parameters,
                generated_at=datetime.utcnow()
            )
            
            # Save to file
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = self.exports_dir / f"{template.template_id}_{timestamp}.html"
            else:
                output_file = Path(output_file)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'content_preview': html_content[:500] + '...' if len(html_content) > 500 else html_content
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return {'error': str(e)}
    
    def _generate_pdf_report(self, template: ReportTemplate, report_data: Dict,
                            parameters: Dict, output_file: str = None) -> Dict[str, Any]:
        """Generate PDF report using ReportLab."""
        try:
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = self.exports_dir / f"{template.template_id}_{timestamp}.pdf"
            else:
                output_file = Path(output_file)
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_file), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title page
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30
            )
            
            story.append(Paragraph(template.name, title_style))
            story.append(Paragraph(template.description, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata = report_data.get('report_metadata', {})
            story.append(Paragraph("Report Information", styles['Heading2']))
            
            metadata_data = [
                ['Generated At', metadata.get('generated_at', 'N/A')],
                ['Period Start', metadata.get('period_start', 'N/A')],
                ['Period End', metadata.get('period_end', 'N/A')],
                ['Analysis Period', f"{metadata.get('period_days', 'N/A')} days"]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(PageBreak())
            
            # Generate content based on template type
            if template.template_id == 'executive_summary':
                story.extend(self._generate_pdf_executive_summary(report_data, styles))
            elif template.template_id == 'source_performance':
                story.extend(self._generate_pdf_source_performance(report_data, styles))
            elif template.template_id == 'keyword_trends':
                story.extend(self._generate_pdf_keyword_trends(report_data, styles))
            elif template.template_id == 'system_performance':
                story.extend(self._generate_pdf_system_performance(report_data, styles))
            else:
                story.extend(self._generate_pdf_custom_analysis(report_data, styles))
            
            # Build PDF
            doc.build(story)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'pages': len(story)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate PDF report: {e}")
            return {'error': str(e)}
    
    def _generate_json_report(self, template: ReportTemplate, report_data: Dict,
                             parameters: Dict, output_file: str = None) -> Dict[str, Any]:
        """Generate JSON report."""
        try:
            if not output_file:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_file = self.exports_dir / f"{template.template_id}_{timestamp}.json"
            else:
                output_file = Path(output_file)
            
            # Prepare JSON data
            json_data = {
                'template': {
                    'id': template.template_id,
                    'name': template.name,
                    'description': template.description
                },
                'parameters': parameters,
                'data': report_data,
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            return {
                'success': True,
                'file_path': str(output_file),
                'file_size': output_file.stat().st_size,
                'record_count': len(str(json_data))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            return {'error': str(e)}
    
    def _merge_parameters(self, template_params: Dict, user_params: Dict) -> Dict[str, Any]:
        """Merge user parameters with template defaults."""
        merged = {}
        
        for param_name, param_config in template_params.items():
            if param_name in user_params:
                merged[param_name] = user_params[param_name]
            else:
                merged[param_name] = param_config.get('default')
        
        return merged
    
    def _validate_parameters(self, template_params: Dict, user_params: Dict) -> List[str]:
        """Validate parameters against template configuration."""
        errors = []
        
        for param_name, param_config in template_params.items():
            value = user_params.get(param_name)
            
            # Check required parameters
            if param_config.get('required', True) and value is None:
                errors.append(f"Required parameter '{param_name}' is missing")
                continue
            
            if value is None:
                continue
            
            # Type validation
            expected_type = param_config.get('type', 'str')
            if expected_type == 'int' and not isinstance(value, int):
                errors.append(f"Parameter '{param_name}' must be an integer")
            elif expected_type == 'float' and not isinstance(value, (int, float)):
                errors.append(f"Parameter '{param_name}' must be a number")
            elif expected_type == 'bool' and not isinstance(value, bool):
                errors.append(f"Parameter '{param_name}' must be a boolean")
            elif expected_type == 'list' and not isinstance(value, list):
                errors.append(f"Parameter '{param_name}' must be a list")
            elif expected_type == 'str' and not isinstance(value, str):
                errors.append(f"Parameter '{param_name}' must be a string")
            
            # Range validation
            if 'min' in param_config and value < param_config['min']:
                errors.append(f"Parameter '{param_name}' must be >= {param_config['min']}")
            if 'max' in param_config and value > param_config['max']:
                errors.append(f"Parameter '{param_name}' must be <= {param_config['max']}")
        
        return errors
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available report templates."""
        return [
            {
                'template_id': template.template_id,
                'name': template.name,
                'description': template.description,
                'parameters': template.parameters,
                'created_at': template.created_at.isoformat()
            }
            for template in self.templates.values()
        ]
    
    def _create_default_html_template(self, template_id: str, template_path: Path):
        """Create a default HTML template if it doesn't exist."""
        
        if template_id == 'executive_summary':
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report_data.custom_title or "Executive Summary Report" }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .metric { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #27ae60; }
        .section { margin: 30px 0; }
        .keyword { background: #e8f6f3; padding: 5px 10px; margin: 3px; border-radius: 15px; display: inline-block; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Executive Summary Report</h1>
        <p>Generated on {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p>Analysis Period: {{ report_data.report_metadata.period_days }} days 
           ({{ report_data.report_metadata.period_start.strftime('%Y-%m-%d') }} to 
           {{ report_data.report_metadata.period_end.strftime('%Y-%m-%d') }})</p>
    </div>
    
    <div class="section">
        <h2>Key Metrics Overview</h2>
        {% if report_data.overview_metrics %}
        <div class="metric">
            <div class="metric-value">{{ "{:,}".format(report_data.overview_metrics.total_articles) }}</div>
            <div>Total Articles</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ report_data.overview_metrics.unique_sources }}</div>
            <div>Active Sources</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.1f"|format(report_data.overview_metrics.avg_hourly_volume) }}</div>
            <div>Average Hourly Volume</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ "%.3f"|format(report_data.overview_metrics.overall_sentiment) }}</div>
            <div>Overall Sentiment</div>
        </div>
        {% endif %}
    </div>
    
    {% if report_data.trending_keywords %}
    <div class="section">
        <h2>Trending Keywords</h2>
        {% for keyword in report_data.trending_keywords[:20] %}
        <span class="keyword">{{ keyword.keyword }} ({{ keyword.total_mentions }} mentions)</span>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if report_data.top_sources %}
    <div class="section">
        <h2>Top Performing Sources</h2>
        <table>
            <tr><th>Source</th><th>Articles</th><th>Success Rate</th><th>Avg Sentiment</th></tr>
            {% for source in report_data.top_sources[:10] %}
            <tr>
                <td>{{ source.source }}</td>
                <td>{{ source.total_published or 0 }}</td>
                <td>{{ "%.1%"|format(source.avg_success_rate or 0) }}</td>
                <td>{{ "%.3f"|format(source.avg_sentiment or 0) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
</body>
</html>
            """
        else:
            # Generic template for other types
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report_data.custom_title or "Analytics Report" }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .section { margin: 20px 0; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report_data.custom_title or "Analytics Report" }}</h1>
        <p>Generated on {{ generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>
    
    <div class="section">
        <h2>Report Data</h2>
        <pre>{{ report_data | tojson(indent=2) }}</pre>
    </div>
</body>
</html>
            """
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content.strip())
    
    def _generate_pdf_executive_summary(self, report_data: Dict, styles) -> List:
        """Generate PDF content for executive summary."""
        content = []
        
        # Overview metrics
        content.append(Paragraph("Executive Overview", styles['Heading2']))
        
        if 'overview_metrics' in report_data:
            metrics = report_data['overview_metrics']
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Articles', f"{metrics.get('total_articles', 0):,}"],
                ['Active Sources', str(metrics.get('unique_sources', 0))],
                ['Avg Hourly Volume', f"{metrics.get('avg_hourly_volume', 0):.1f}"],
                ['Overall Sentiment', f"{metrics.get('overall_sentiment', 0):.3f}"],
                ['Growth Rate', f"{metrics.get('growth_rate', 0):.1f}%"]
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(metrics_table)
            content.append(Spacer(1, 20))
        
        # Top sources
        if 'top_sources' in report_data and report_data['top_sources']:
            content.append(Paragraph("Top Performing Sources", styles['Heading3']))
            
            sources_data = [['Source', 'Articles', 'Success Rate']]
            for source in report_data['top_sources'][:10]:
                sources_data.append([
                    source['source'][:30],  # Truncate long source names
                    str(source['total_published'] or 0),
                    f"{(source['avg_success_rate'] or 0):.1%}"
                ])
            
            sources_table = Table(sources_data, colWidths=[3*inch, 1*inch, 1*inch])
            sources_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(sources_table)
            content.append(Spacer(1, 20))
        
        return content
    
    def _generate_pdf_source_performance(self, report_data: Dict, styles) -> List:
        """Generate PDF content for source performance report."""
        content = []
        
        content.append(Paragraph("Source Performance Analysis", styles['Heading2']))
        
        if 'source_performance' in report_data:
            perf_data = [['Source', 'Articles', 'Success Rate', 'Avg Duration (ms)', 'Errors']]
            
            for source in report_data['source_performance'][:20]:
                perf_data.append([
                    source['source'][:25],
                    str(source['total_published'] or 0),
                    f"{(source['avg_success_rate'] or 0):.2%}",
                    str(int(source['avg_scrape_time'] or 0)),
                    str(source['total_errors'] or 0)
                ])
            
            perf_table = Table(perf_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1*inch, 0.8*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(perf_table)
        
        return content
    
    def _generate_pdf_keyword_trends(self, report_data: Dict, styles) -> List:
        """Generate PDF content for keyword trends report."""
        content = []
        
        content.append(Paragraph("Keyword Trends Analysis", styles['Heading2']))
        
        # Trending up keywords
        if 'trending_up' in report_data and report_data['trending_up']:
            content.append(Paragraph("Trending Up Keywords", styles['Heading3']))
            
            trending_data = [['Keyword', 'Mentions', 'Growth Rate', 'Sentiment']]
            for keyword in report_data['trending_up'][:15]:
                trending_data.append([
                    keyword['keyword'],
                    str(keyword['total_mentions']),
                    f"{keyword['avg_growth_rate']:.1f}%",
                    f"{keyword['avg_sentiment']:.3f}" if keyword['avg_sentiment'] else 'N/A'
                ])
            
            trending_table = Table(trending_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch])
            trending_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(trending_table)
        
        return content
    
    def _generate_pdf_system_performance(self, report_data: Dict, styles) -> List:
        """Generate PDF content for system performance report."""
        content = []
        
        content.append(Paragraph("System Performance Analysis", styles['Heading2']))
        
        if 'system_overview' in report_data:
            overview = report_data['system_overview']
            
            sys_data = [
                ['Metric', 'Average', 'Maximum'],
                ['CPU Usage', f"{overview['avg_cpu_usage']:.1f}%", f"{overview['max_cpu_usage']:.1f}%"],
                ['Memory Usage', f"{overview['avg_memory_mb']:.0f}MB", f"{overview['max_memory_mb']:.0f}MB"],
                ['Active Connections', f"{overview['avg_connections']:.1f}", str(overview['max_connections'])],
                ['Article Rate', f"{overview['avg_article_rate']:.2f}/min", f"{overview['max_article_rate']:.2f}/min"],
                ['Error Rate', f"{overview['avg_error_rate']:.2f}%", "N/A"]
            ]
            
            sys_table = Table(sys_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            sys_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(sys_table)
        
        return content
    
    def _generate_pdf_custom_analysis(self, report_data: Dict, styles) -> List:
        """Generate PDF content for custom analysis report."""
        content = []
        
        title = report_data.get('custom_title', 'Custom Analysis Report')
        content.append(Paragraph(title, styles['Heading2']))
        
        description = report_data.get('custom_description', '')
        if description:
            content.append(Paragraph(description, styles['Normal']))
            content.append(Spacer(1, 20))
        
        # Add summary of included analysis types
        analysis_types = report_data.get('analysis_types', [])
        if analysis_types:
            content.append(Paragraph("Analysis Types Included:", styles['Heading3']))
            for analysis_type in analysis_types:
                content.append(Paragraph(f"• {analysis_type.title()}", styles['Normal']))
        
        return content


# Import numpy for statistical calculations
try:
    import numpy as np
except ImportError:
    # Fallback implementations for basic statistical functions
    class np:
        @staticmethod
        def std(arr):
            if not arr:
                return 0
            mean = sum(arr) / len(arr)
            return (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5