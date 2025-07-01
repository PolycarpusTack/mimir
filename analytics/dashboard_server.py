"""Analytics Dashboard Server for Mimir News Intelligence.

This module provides a Flask-based web dashboard for visualizing analytics data
with real-time updates, interactive charts, and comprehensive metrics display.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import bleach

from .data_warehouse import AnalyticsDataWarehouse
from .metrics_collector import MetricsCollector
from .trend_analyzer import TrendAnalyzer
from .predictive_analytics import PredictiveAnalytics
from .report_generator import ReportGenerator
from .report_scheduler import ReportScheduler
from db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


# Input validation functions
def validate_integer_param(value: str, min_val: int = 1, max_val: int = 1000, default: int = None) -> int:
    """Validate integer parameters with bounds checking."""
    try:
        int_val = int(value)
        if min_val <= int_val <= max_val:
            return int_val
        else:
            return default if default is not None else min_val
    except (ValueError, TypeError):
        return default if default is not None else min_val


def validate_time_interval(interval: str) -> str:
    """Validate time interval parameter."""
    allowed_intervals = ['1 minute', '5 minutes', '15 minutes', '30 minutes', 
                        '1 hour', '2 hours', '6 hours', '12 hours', '1 day']
    return interval if interval in allowed_intervals else '1 hour'


def sanitize_string_param(value: str, max_length: int = 100) -> str:
    """Sanitize string parameters."""
    if not value:
        return ''
    # Remove potentially dangerous characters
    sanitized = bleach.clean(value, tags=[], strip=True)
    return sanitized[:max_length]

# Initialize Flask app with security settings
app = Flask(__name__, 
           template_folder='../templates/dashboard',
           static_folder='../static')

# Security configuration
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Enable CORS with security restrictions
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'])

# Initialize analytics components
try:
    db_adapter = DatabaseAdapter()
    analytics_warehouse = AnalyticsDataWarehouse()
    metrics_collector = MetricsCollector(db_adapter, analytics_warehouse)
    trend_analyzer = TrendAnalyzer(analytics_warehouse)
    predictive_analytics = PredictiveAnalytics(analytics_warehouse)
    report_generator = ReportGenerator(analytics_warehouse)
    report_scheduler = ReportScheduler(report_generator)
    
    logger.info("Analytics dashboard components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize analytics components: {e}")
    db_adapter = None
    analytics_warehouse = None
    report_generator = None
    report_scheduler = None


@app.route('/')
def dashboard_home():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """Get system status and health information."""
    try:
        if not analytics_warehouse:
            return jsonify({'error': 'Analytics warehouse not available'}), 500
            
        with analytics_warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get data counts
            tables = [
                'analytics.article_metrics',
                'analytics.source_metrics',
                'analytics.keyword_metrics',
                'analytics.user_activity',
                'analytics.system_metrics'
            ]
            
            data_counts = {}
            for table in tables:
                # Use parameterized queries for security
                table_name = table.split('.')[-1]
                if table_name in ['article_metrics', 'source_metrics', 'keyword_metrics', 'user_activity', 'system_metrics']:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    data_counts[table_name] = cursor.fetchone()[0]
            
            # Get latest activity
            cursor.execute("""
                SELECT 
                    MAX(time) as latest_metric,
                    COUNT(*) as recent_count
                FROM analytics.article_metrics 
                WHERE time > NOW() - INTERVAL '24 hours'
            """)
            latest_time, recent_count = cursor.fetchone()
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'data_counts': data_counts,
                'latest_activity': {
                    'latest_metric': latest_time.isoformat() if latest_time else None,
                    'recent_articles': recent_count or 0
                }
            })
            
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/overview')
def api_metrics_overview():
    """Get overview metrics for dashboard."""
    try:
        if not analytics_warehouse:
            return jsonify({'error': 'Analytics warehouse not available'}), 500
            
        # Get time ranges
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        last_30d = now - timedelta(days=30)
        
        with analytics_warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # Article volume metrics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_articles_24h,
                    AVG(article_count) as avg_hourly_volume,
                    MAX(article_count) as peak_hourly_volume
                FROM analytics.article_metrics 
                WHERE time >= %s
            """, [last_24h])
            
            article_stats = cursor.fetchone()
            
            # Source performance
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT source) as active_sources,
                    AVG(avg_sentiment) as overall_sentiment
                FROM analytics.article_metrics 
                WHERE time >= %s AND avg_sentiment IS NOT NULL
            """, [last_24h])
            
            source_stats = cursor.fetchone()
            
            # Trending keywords (top 10)
            cursor.execute("""
                SELECT 
                    keyword,
                    SUM(mention_count) as total_mentions,
                    AVG(growth_rate) as avg_growth
                FROM analytics.keyword_metrics 
                WHERE time >= %s
                GROUP BY keyword
                ORDER BY total_mentions DESC
                LIMIT 10
            """, [last_7d])
            
            trending_keywords = [
                {
                    'keyword': row[0],
                    'mentions': row[1],
                    'growth': round(row[2] or 0, 2)
                }
                for row in cursor.fetchall()
            ]
            
            # System performance
            cursor.execute("""
                SELECT 
                    AVG(cpu_usage_percent) as avg_cpu,
                    AVG(memory_usage_mb) as avg_memory,
                    AVG(articles_per_minute) as avg_article_rate
                FROM analytics.system_metrics 
                WHERE time >= %s
            """, [last_24h])
            
            system_stats = cursor.fetchone()
            
            return jsonify({
                'overview': {
                    'articles_24h': article_stats[0] or 0,
                    'avg_hourly_volume': round(article_stats[1] or 0, 1),
                    'peak_hourly_volume': article_stats[2] or 0,
                    'active_sources': source_stats[0] or 0,
                    'overall_sentiment': round(source_stats[1] or 0, 3),
                    'avg_cpu_usage': round(system_stats[0] or 0, 1),
                    'avg_memory_mb': round(system_stats[1] or 0),
                    'avg_article_rate': round(system_stats[2] or 0, 2)
                },
                'trending_keywords': trending_keywords,
                'generated_at': now.isoformat()
            })
            
    except Exception as e:
        logger.error(f"Overview API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/volume_trend')
def api_volume_trend():
    """Get article volume trend data for charts."""
    try:
        # Validate input parameters
        hours = validate_integer_param(request.args.get('hours', '24'), min_val=1, max_val=168, default=24)
        interval = validate_time_interval(request.args.get('interval', '1 hour'))
        
        start_time = datetime.utcnow() - timedelta(hours=hours)
        end_time = datetime.utcnow()
        
        trend_data = analytics_warehouse.get_article_trends(
            start_time=start_time,
            end_time=end_time,
            interval=interval
        )
        
        # Format for chart display
        chart_data = {
            'labels': [row['bucket'].isoformat() for row in trend_data],
            'datasets': [
                {
                    'label': 'Article Count',
                    'data': [row['article_count'] for row in trend_data],
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.1
                }
            ]
        }
        
        # Add sentiment data if available
        sentiment_data = [row.get('avg_sentiment') for row in trend_data]
        if any(s is not None for s in sentiment_data):
            chart_data['datasets'].append({
                'label': 'Average Sentiment',
                'data': sentiment_data,
                'borderColor': 'rgb(255, 99, 132)',
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'yAxisID': 'y1'
            })
        
        return jsonify(chart_data)
        
    except Exception as e:
        logger.error(f"Volume trend API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/source_performance')
def api_source_performance():
    """Get source performance data for charts."""
    try:
        # Validate input parameters
        days = validate_integer_param(request.args.get('days', '7'), min_val=1, max_val=365, default=7)
        
        start_time = datetime.utcnow() - timedelta(days=days)
        end_time = datetime.utcnow()
        
        source_data = analytics_warehouse.get_source_performance(
            start_time=start_time,
            end_time=end_time
        )
        
        # Sort by article count and take top 15
        source_data = sorted(source_data, key=lambda x: x['total_published'] or 0, reverse=True)[:15]
        
        chart_data = {
            'labels': [row['source'] for row in source_data],
            'datasets': [
                {
                    'label': 'Articles Published',
                    'data': [row['total_published'] or 0 for row in source_data],
                    'backgroundColor': 'rgba(54, 162, 235, 0.5)',
                    'borderColor': 'rgba(54, 162, 235, 1)',
                    'borderWidth': 1
                },
                {
                    'label': 'Success Rate (%)',
                    'data': [round((row['avg_success_rate'] or 0) * 100, 1) for row in source_data],
                    'backgroundColor': 'rgba(75, 192, 192, 0.5)',
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'borderWidth': 1,
                    'yAxisID': 'y1'
                }
            ]
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        logger.error(f"Source performance API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/charts/keyword_trends')
def api_keyword_trends():
    """Get keyword trend data for charts."""
    try:
        # Validate input parameters
        days = validate_integer_param(request.args.get('days', '7'), min_val=1, max_val=365, default=7)
        limit = validate_integer_param(request.args.get('limit', '10'), min_val=1, max_val=100, default=10)
        
        start_time = datetime.utcnow() - timedelta(days=days)
        end_time = datetime.utcnow()
        
        keyword_data = analytics_warehouse.get_keyword_trends(
            start_time=start_time,
            end_time=end_time,
            limit=limit * 10  # Get more data for filtering
        )
        
        # Group by keyword and aggregate
        keyword_aggregates = {}
        for row in keyword_data:
            keyword = row['keyword']
            if keyword not in keyword_aggregates:
                keyword_aggregates[keyword] = {
                    'total_mentions': 0,
                    'total_articles': 0,
                    'avg_sentiment': [],
                    'growth_rates': []
                }
            
            keyword_aggregates[keyword]['total_mentions'] += row['total_mentions'] or 0
            keyword_aggregates[keyword]['total_articles'] += row['total_articles'] or 0
            
            if row['avg_sentiment'] is not None:
                keyword_aggregates[keyword]['avg_sentiment'].append(row['avg_sentiment'])
            if row['max_growth_rate'] is not None:
                keyword_aggregates[keyword]['growth_rates'].append(row['max_growth_rate'])
        
        # Sort by total mentions and take top keywords
        sorted_keywords = sorted(
            keyword_aggregates.items(),
            key=lambda x: x[1]['total_mentions'],
            reverse=True
        )[:limit]
        
        chart_data = {
            'labels': [item[0] for item in sorted_keywords],
            'datasets': [
                {
                    'label': 'Total Mentions',
                    'data': [item[1]['total_mentions'] for item in sorted_keywords],
                    'backgroundColor': 'rgba(255, 206, 86, 0.5)',
                    'borderColor': 'rgba(255, 206, 86, 1)',
                    'borderWidth': 1
                }
            ]
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        logger.error(f"Keyword trends API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<forecast_type>')
def api_forecast(forecast_type):
    """Get forecast data for specified type."""
    try:
        days_ahead = request.args.get('days', 7, type=int)
        
        if forecast_type == 'volume':
            result = predictive_analytics.forecast_article_volume(
                forecast_days=days_ahead,
                model_type='ensemble'
            )
        elif forecast_type == 'sentiment':
            result = predictive_analytics.forecast_sentiment_trends(
                forecast_days=days_ahead
            )
        elif forecast_type == 'topics':
            result = predictive_analytics.predict_topic_trends(
                prediction_days=days_ahead
            )
        else:
            return jsonify({'error': f'Unsupported forecast type: {forecast_type}'}), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Forecast API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/trends')
def api_trend_analysis():
    """Get comprehensive trend analysis."""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        # Run comprehensive trend analysis
        results = trend_analyzer.run_comprehensive_analysis(
            days_back=days_back,
            include_keywords=True,
            include_sources=True,
            keyword_limit=20
        )
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Trend analysis API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/thresholds')
def api_alert_thresholds():
    """Get adaptive alert thresholds."""
    try:
        sensitivity = request.args.get('sensitivity', 'medium')
        
        # Get thresholds for different metrics
        volume_thresholds = predictive_analytics.learn_alert_thresholds(
            metric_type='article_volume',
            sensitivity=sensitivity
        )
        
        sentiment_thresholds = predictive_analytics.learn_alert_thresholds(
            metric_type='sentiment',
            sensitivity=sensitivity
        )
        
        return jsonify({
            'volume_thresholds': volume_thresholds,
            'sentiment_thresholds': sentiment_thresholds,
            'sensitivity': sensitivity
        })
        
    except Exception as e:
        logger.error(f"Alert thresholds API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/collect_metrics', methods=['POST'])
def api_collect_metrics():
    """Trigger manual metrics collection."""
    try:
        if not metrics_collector:
            return jsonify({'error': 'Metrics collector not available'}), 500
            
        success = metrics_collector.run_full_collection()
        
        return jsonify({
            'success': success,
            'message': 'Metrics collection completed' if success else 'Metrics collection had issues',
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Collect metrics API error: {e}")
        return jsonify({'error': str(e)}), 500


# Report Management API Endpoints

@app.route('/api/reports/templates')
def api_report_templates():
    """Get available report templates."""
    try:
        if not report_generator:
            return jsonify({'error': 'Report generator not available'}), 500
        
        templates = report_generator.get_available_templates()
        return jsonify(templates)
        
    except Exception as e:
        logger.error(f"Report templates API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/generate', methods=['POST'])
def api_generate_report():
    """Generate a report on demand."""
    try:
        if not report_generator:
            return jsonify({'error': 'Report generator not available'}), 500
        
        data = request.get_json()
        
        template_id = data.get('template_id')
        parameters = data.get('parameters', {})
        output_format = data.get('output_format', 'html')
        
        if not template_id:
            return jsonify({'error': 'template_id is required'}), 400
        
        result = report_generator.generate_report(
            template_id=template_id,
            parameters=parameters,
            output_format=output_format
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Generate report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled')
def api_scheduled_reports():
    """Get all scheduled reports."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        reports = report_scheduler.get_scheduled_reports()
        return jsonify(reports)
        
    except Exception as e:
        logger.error(f"Scheduled reports API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled', methods=['POST'])
def api_create_scheduled_report():
    """Create a new scheduled report."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        data = request.get_json()
        
        name = data.get('name')
        template_id = data.get('template_id')
        schedule_config = data.get('schedule_config')
        parameters = data.get('parameters', {})
        delivery_config = data.get('delivery_config', {})
        
        if not all([name, template_id, schedule_config]):
            return jsonify({'error': 'name, template_id, and schedule_config are required'}), 400
        
        report_id = report_scheduler.create_scheduled_report(
            name=name,
            template_id=template_id,
            schedule_config=schedule_config,
            parameters=parameters,
            delivery_config=delivery_config
        )
        
        return jsonify({'report_id': report_id, 'message': 'Scheduled report created successfully'})
        
    except Exception as e:
        logger.error(f"Create scheduled report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled/<report_id>')
def api_get_scheduled_report(report_id):
    """Get details of a specific scheduled report."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        report = report_scheduler.get_scheduled_report(report_id)
        
        if not report:
            return jsonify({'error': 'Scheduled report not found'}), 404
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Get scheduled report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled/<report_id>', methods=['PUT'])
def api_update_scheduled_report(report_id):
    """Update a scheduled report."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        data = request.get_json()
        
        success = report_scheduler.update_scheduled_report(report_id, **data)
        
        if not success:
            return jsonify({'error': 'Failed to update scheduled report'}), 400
        
        return jsonify({'message': 'Scheduled report updated successfully'})
        
    except Exception as e:
        logger.error(f"Update scheduled report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled/<report_id>', methods=['DELETE'])
def api_delete_scheduled_report(report_id):
    """Delete a scheduled report."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        success = report_scheduler.delete_scheduled_report(report_id)
        
        if not success:
            return jsonify({'error': 'Failed to delete scheduled report'}), 400
        
        return jsonify({'message': 'Scheduled report deleted successfully'})
        
    except Exception as e:
        logger.error(f"Delete scheduled report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduled/<report_id>/run', methods=['POST'])
def api_run_scheduled_report(report_id):
    """Run a scheduled report immediately."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        force = request.args.get('force', 'false').lower() == 'true'
        
        result = report_scheduler.run_scheduled_report(report_id, force=force)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Run scheduled report API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduler/start', methods=['POST'])
def api_start_scheduler():
    """Start the report scheduler."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        report_scheduler.start_scheduler()
        
        return jsonify({'message': 'Report scheduler started successfully'})
        
    except Exception as e:
        logger.error(f"Start scheduler API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduler/stop', methods=['POST'])
def api_stop_scheduler():
    """Stop the report scheduler."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        report_scheduler.stop_scheduler()
        
        return jsonify({'message': 'Report scheduler stopped successfully'})
        
    except Exception as e:
        logger.error(f"Stop scheduler API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/reports/scheduler/status')
def api_scheduler_status():
    """Get scheduler status."""
    try:
        if not report_scheduler:
            return jsonify({'error': 'Report scheduler not available'}), 500
        
        return jsonify({
            'running': report_scheduler.running,
            'scheduled_reports_count': len(report_scheduler.scheduled_reports),
            'enabled_reports_count': len([r for r in report_scheduler.scheduled_reports.values() if r.enabled])
        })
        
    except Exception as e:
        logger.error(f"Scheduler status API error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the dashboard server
    app.run(host='0.0.0.0', port=5001, debug=True)