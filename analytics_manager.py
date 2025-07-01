#!/usr/bin/env python3
"""Analytics Manager CLI for Mimir News Scraper.

This script provides command-line interface for managing the analytics and
business intelligence features including TimescaleDB setup, metrics collection,
and trend analysis.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from analytics.data_warehouse import AnalyticsDataWarehouse
from analytics.metrics_collector import MetricsCollector
from analytics.trend_analyzer import TrendAnalyzer
from analytics.predictive_analytics import PredictiveAnalytics
from analytics.report_generator import ReportGenerator
from analytics.report_scheduler import ReportScheduler
from analytics.data_exporter import DataExporter, ExportConfig, ExportFormat
from analytics.ml_infrastructure import (
    MLInfrastructureManager,
    ModelRegistry,
    ExperimentTracker,
    ABTestingFramework,
    FeatureStore
)
from analytics.feature_engineering import FeatureEngineering
from analytics.article_clustering import ArticleClusterer
from analytics.recommendation_engine import RecommendationEngine
from analytics.analytics_validator import AnalyticsValidator
from analytics.performance_optimizer import PerformanceOptimizer
from db_adapter import DatabaseAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analytics.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_analytics_infrastructure():
    """Set up the analytics infrastructure including TimescaleDB."""
    print("🚀 Setting up Analytics Infrastructure...")
    
    try:
        # Check if TimescaleDB is available
        warehouse = AnalyticsDataWarehouse()
        
        with warehouse.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to database: {version}")
            
            # Check if TimescaleDB extension is available
            cursor.execute("SELECT * FROM pg_available_extensions WHERE name = 'timescaledb'")
            if cursor.fetchone():
                print("✅ TimescaleDB extension is available")
            else:
                print("❌ TimescaleDB extension not found. Please install TimescaleDB.")
                return False
                
            # Check if analytics schema exists
            cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'analytics'")
            if cursor.fetchone():
                print("✅ Analytics schema found")
            else:
                print("❌ Analytics schema not found. Please run TimescaleDB initialization script.")
                return False
                
        print("✅ Analytics infrastructure is ready!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup analytics infrastructure: {e}")
        return False


def collect_metrics():
    """Collect metrics from the main database and store in analytics warehouse."""
    print("📊 Collecting metrics...")
    
    try:
        # Initialize components
        db_adapter = DatabaseAdapter()
        warehouse = AnalyticsDataWarehouse()
        collector = MetricsCollector(db_adapter, warehouse)
        
        # Run full metrics collection
        success = collector.run_full_collection()
        
        if success:
            print("✅ Metrics collection completed successfully")
            return True
        else:
            print("⚠️ Metrics collection completed with some failures")
            return False
            
    except Exception as e:
        print(f"❌ Failed to collect metrics: {e}")
        return False


def analyze_trends(days_back=30, keywords_only=False, sources_only=False):
    """Run trend analysis on collected metrics."""
    print(f"📈 Analyzing trends for the last {days_back} days...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        analyzer = TrendAnalyzer(warehouse)
        
        if keywords_only:
            # Analyze only keyword trends
            print("🔍 Analyzing keyword trends...")
            results = analyzer.analyze_keyword_trends(days_back=days_back)
            print(f"✅ Analyzed {len(results)} keywords")
            
            # Show top trending keywords
            trending_up = [r for r in results if r.get('trend_direction') == 'up']
            trending_down = [r for r in results if r.get('trend_direction') == 'down']
            
            print(f"\n📈 Trending UP keywords ({len(trending_up)}):")
            for keyword in trending_up[:10]:
                print(f"  • {keyword['subject']}: {keyword.get('total_mentions', 0)} mentions")
                
            print(f"\n📉 Trending DOWN keywords ({len(trending_down)}):")
            for keyword in trending_down[:10]:
                print(f"  • {keyword['subject']}: {keyword.get('total_mentions', 0)} mentions")
                
        elif sources_only:
            # Analyze only source performance
            print("📰 Analyzing source performance trends...")
            results = analyzer.analyze_source_performance_trends(days_back=days_back)
            print(f"✅ Analyzed {len(results)} sources")
            
            # Show source performance summary
            improving = [r for r in results if r.get('trend_direction') == 'up']
            declining = [r for r in results if r.get('trend_direction') == 'down']
            
            print(f"\n📈 Improving sources ({len(improving)}):")
            for source in improving[:10]:
                print(f"  • {source['subject']}: {source.get('total_articles', 0)} articles")
                
            print(f"\n📉 Declining sources ({len(declining)}):")
            for source in declining[:10]:
                print(f"  • {source['subject']}: {source.get('total_articles', 0)} articles")
                
        else:
            # Run comprehensive analysis
            print("🔍 Running comprehensive trend analysis...")
            results = analyzer.run_comprehensive_analysis(days_back=days_back)
            
            # Display summary
            volume_analysis = results.get('article_volume_analysis', {})
            keyword_analyses = results.get('keyword_analyses', [])
            source_analyses = results.get('source_analyses', [])
            
            print(f"\n📊 Analysis Summary:")
            print(f"  • Article Volume Trend: {volume_analysis.get('trend_direction', 'unknown')}")
            print(f"  • Keywords Analyzed: {len(keyword_analyses)}")
            print(f"  • Sources Analyzed: {len(source_analyses)}")
            print(f"  • Analysis Duration: {results.get('analysis_duration_seconds', 0)} seconds")
            
            # Show anomalies if detected
            anomalies = volume_analysis.get('anomalies_detected', 0)
            if anomalies > 0:
                print(f"  ⚠️ {anomalies} anomalies detected in article volume")
                
            # Show seasonality
            if volume_analysis.get('has_seasonality'):
                period = volume_analysis.get('seasonal_period_days', 'unknown')
                print(f"  📅 Seasonality detected: {period} day period")
                
        print("✅ Trend analysis completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to analyze trends: {e}")
        return False


def show_analytics_status():
    """Show the current status of the analytics system."""
    print("📊 Analytics System Status:")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        
        with warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check data availability
            tables = [
                'analytics.article_metrics',
                'analytics.source_metrics', 
                'analytics.keyword_metrics',
                'analytics.user_activity',
                'analytics.system_metrics'
            ]
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  • {table.split('.')[-1]}: {count:,} records")
                
            # Check recent data
            cursor.execute("""
                SELECT 
                    MAX(time) as latest_article,
                    COUNT(*) as total_articles
                FROM analytics.article_metrics 
                WHERE time > NOW() - INTERVAL '24 hours'
            """)
            result = cursor.fetchone()
            latest_time, recent_count = result
            
            if latest_time:
                print(f"\n📈 Recent Activity (24h):")
                print(f"  • Latest article metric: {latest_time}")
                print(f"  • Articles processed: {recent_count:,}")
            else:
                print(f"\n⚠️ No recent activity detected")
                
            # Check trend analyses
            cursor.execute("SELECT COUNT(*) FROM analytics.trend_analysis")
            trend_count = cursor.fetchone()[0]
            print(f"  • Trend analyses stored: {trend_count:,}")
            
        print("✅ Analytics system is operational")
        return True
        
    except Exception as e:
        print(f"❌ Failed to get analytics status: {e}")
        return False


def cleanup_old_data(days_to_keep=90):
    """Clean up old analytics data beyond retention period."""
    print(f"🧹 Cleaning up data older than {days_to_keep} days...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with warehouse.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count records to be deleted
            tables_to_clean = [
                'analytics.user_activity',
                'analytics.system_metrics'
            ]
            
            total_deleted = 0
            for table in tables_to_clean:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE time < %s", [cutoff_date])
                count = cursor.fetchone()[0]
                
                if count > 0:
                    cursor.execute(f"DELETE FROM {table} WHERE time < %s", [cutoff_date])
                    deleted = cursor.rowcount
                    total_deleted += deleted
                    print(f"  • Deleted {deleted:,} records from {table.split('.')[-1]}")
                    
            conn.commit()
            
            if total_deleted > 0:
                print(f"✅ Cleaned up {total_deleted:,} old records")
            else:
                print("✅ No old records to clean up")
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to cleanup old data: {e}")
        return False


def run_continuous_collection(interval_minutes=60):
    """Run continuous metrics collection at specified interval."""
    print(f"🔄 Starting continuous metrics collection (every {interval_minutes} minutes)...")
    print("Press Ctrl+C to stop")
    
    try:
        db_adapter = DatabaseAdapter()
        warehouse = AnalyticsDataWarehouse()
        collector = MetricsCollector(db_adapter, warehouse)
        
        while True:
            print(f"\n📊 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Collecting metrics...")
            
            success = collector.run_full_collection()
            
            if success:
                print("✅ Collection completed successfully")
            else:
                print("⚠️ Collection completed with some issues")
                
            print(f"😴 Sleeping for {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping continuous collection...")
        return True
    except Exception as e:
        print(f"❌ Continuous collection failed: {e}")
        return False


def run_dashboard_server(port=5001, debug=False):
    """Run the analytics dashboard server."""
    print(f"🌐 Starting analytics dashboard server on port {port}...")
    
    try:
        from analytics.dashboard_server import app
        
        print(f"📊 Dashboard will be available at: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        
        app.run(host='0.0.0.0', port=port, debug=debug)
        
    except ImportError as e:
        print(f"❌ Failed to import dashboard server: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to start dashboard server: {e}")
        return False


def generate_forecasts(forecast_type='all', days_ahead=7):
    """Generate predictive forecasts for various metrics."""
    print(f"🔮 Generating {forecast_type} forecasts for {days_ahead} days ahead...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        predictor = PredictiveAnalytics(warehouse)
        
        results = {}
        
        if forecast_type in ['all', 'volume']:
            print("📊 Generating article volume forecast...")
            volume_forecast = predictor.forecast_article_volume(
                forecast_days=days_ahead,
                model_type='ensemble'
            )
            results['volume_forecast'] = volume_forecast
            
            if 'error' not in volume_forecast:
                avg_forecast = np.mean(volume_forecast.get('forecast_values', [0]))
                print(f"  • Average daily volume forecast: {avg_forecast:.1f} articles")
        
        if forecast_type in ['all', 'topics']:
            print("🏷️ Generating topic trend predictions...")
            topic_predictions = predictor.predict_topic_trends(
                prediction_days=days_ahead
            )
            results['topic_predictions'] = topic_predictions
            
            if 'error' not in topic_predictions:
                trending_count = len(topic_predictions.get('top_trending_keywords', []))
                print(f"  • {trending_count} trending topics identified")
        
        if forecast_type in ['all', 'sentiment']:
            print("😊 Generating sentiment forecasts...")
            sentiment_forecast = predictor.forecast_sentiment_trends(
                forecast_days=days_ahead
            )
            results['sentiment_forecast'] = sentiment_forecast
            
            if 'error' not in sentiment_forecast:
                current_sentiment = sentiment_forecast.get('current_sentiment', 0)
                forecasted_sentiment = sentiment_forecast.get('ensemble_forecast', {}).get('forecast_value', 0)
                print(f"  • Current sentiment: {current_sentiment:.3f}")
                print(f"  • Forecasted sentiment: {forecasted_sentiment:.3f}")
        
        if forecast_type in ['all', 'alerts']:
            print("🚨 Learning adaptive alert thresholds...")
            volume_thresholds = predictor.learn_alert_thresholds(
                metric_type='article_volume',
                sensitivity='medium'
            )
            sentiment_thresholds = predictor.learn_alert_thresholds(
                metric_type='sentiment',
                sensitivity='medium'
            )
            
            results['alert_thresholds'] = {
                'volume': volume_thresholds,
                'sentiment': sentiment_thresholds
            }
            
            if 'error' not in volume_thresholds:
                upper_threshold = volume_thresholds.get('recommended_upper', 0)
                print(f"  • Volume alert threshold: {upper_threshold:.1f} articles/hour")
        
        print("✅ Forecast generation completed successfully")
        
        # Display summary
        if results:
            print(f"\n📋 Forecast Summary:")
            for forecast_name, forecast_data in results.items():
                if 'error' in forecast_data:
                    print(f"  • {forecast_name}: ❌ {forecast_data['error']}")
                else:
                    print(f"  • {forecast_name}: ✅ Generated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate forecasts: {e}")
        return False


def generate_report(template_id, output_format='pdf', **parameters):
    """Generate a report using specified template."""
    print(f"📋 Generating {template_id} report in {output_format} format...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        generator = ReportGenerator(warehouse)
        
        # Get available templates
        templates = generator.get_available_templates()
        template_names = [t['template_id'] for t in templates]
        
        if template_id not in template_names:
            print(f"❌ Unknown template: {template_id}")
            print(f"Available templates: {', '.join(template_names)}")
            return False
        
        # Generate the report
        result = generator.generate_report(
            template_id=template_id,
            parameters=parameters,
            output_format=output_format
        )
        
        if 'error' in result:
            print(f"❌ Report generation failed: {result['error']}")
            return False
        
        print(f"✅ Report generated successfully:")
        print(f"  • Template: {result.get('template_name', template_id)}")
        print(f"  • Format: {result.get('format', output_format)}")
        print(f"  • File: {result.get('file_path', 'N/A')}")
        
        if result.get('file_size'):
            size_mb = result['file_size'] / 1024 / 1024
            print(f"  • Size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate report: {e}")
        return False


def list_report_templates():
    """List available report templates."""
    print("📋 Available Report Templates:")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        generator = ReportGenerator(warehouse)
        
        templates = generator.get_available_templates()
        
        if not templates:
            print("No templates available")
            return True
        
        for template in templates:
            print(f"\n🔹 {template['template_id']}")
            print(f"   Name: {template['name']}")
            print(f"   Description: {template['description']}")
            
            if template['parameters']:
                print("   Parameters:")
                for param_name, param_config in template['parameters'].items():
                    param_type = param_config.get('type', 'str')
                    default = param_config.get('default', 'None')
                    required = param_config.get('required', True)
                    print(f"     • {param_name} ({param_type}): default={default}, required={required}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to list templates: {e}")
        return False


def export_data(data_type, format='csv', **kwargs):
    """Export analytics data in various formats."""
    print(f"📤 Exporting {data_type} data in {format} format...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        db_adapter = DatabaseAdapter()
        exporter = DataExporter(warehouse, db_adapter)
        
        # Configure export
        config = ExportConfig(
            format=format,
            compression=kwargs.get('compress', False),
            filters=kwargs.get('filters', {}),
            columns=kwargs.get('columns')
        )
        
        # Perform export
        result = exporter.export_data(
            data_type=data_type,
            config=config,
            output_file=kwargs.get('output'),
            stream=kwargs.get('stream', False)
        )
        
        if 'error' in result:
            print(f"❌ Export failed: {result['error']}")
            return False
        
        print(f"✅ Export completed successfully:")
        print(f"  • File: {result['file_path']}")
        print(f"  • Records: {result['record_count']:,}")
        print(f"  • Size: {result['file_size'] / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to export data: {e}")
        return False


def manage_ml_models(action, **kwargs):
    """Manage ML models and experiments."""
    print(f"🤖 ML Model Management: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        ml_manager = MLInfrastructureManager(warehouse)
        
        if action == 'list':
            model_name = kwargs.get('model_name')
            models = ml_manager.model_registry.list_models(model_name)
            
            if not models:
                print("No models found in registry")
                return True
            
            print(f"\n📋 Registered Models ({len(models)} total):")
            for model in models:
                status = model.get('stage', 'development')
                status_icon = "🟢" if status == 'production' else "🟡" if status == 'staging' else "⚪"
                
                print(f"\n{status_icon} {model['model_name']} v{model['version']}")
                print(f"   ID: {model['model_id']}")
                print(f"   Type: {model['model_type']}")
                print(f"   Registered: {model['registered_at']}")
                
                if model.get('metrics'):
                    print("   Metrics:")
                    for metric, value in model['metrics'].items():
                        print(f"     • {metric}: {value:.4f}")
        
        elif action == 'register':
            # This would be called programmatically with a trained model
            print("Model registration should be done through the training pipeline")
            return False
        
        elif action == 'promote':
            model_name = kwargs.get('model_name')
            version = kwargs.get('version')
            stage = kwargs.get('stage', 'production')
            
            if not all([model_name, version]):
                print("❌ model_name and version are required")
                return False
            
            ml_manager.model_registry.promote_model(model_name, version, stage)
            print(f"✅ Promoted {model_name} v{version} to {stage}")
        
        elif action == 'experiment':
            # List MLflow experiments
            if ml_manager.experiment_tracker.enabled:
                experiments = ml_manager.experiment_tracker.client.list_experiments()
                print(f"\n📊 MLflow Experiments ({len(experiments)} total):")
                
                for exp in experiments:
                    print(f"\n• {exp.name}")
                    print(f"  ID: {exp.experiment_id}")
                    print(f"  Artifact Location: {exp.artifact_location}")
            else:
                print("⚠️ MLflow not available")
        
        else:
            print(f"❌ Unknown action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to manage ML models: {e}")
        return False


def manage_feature_store(action, **kwargs):
    """Manage ML feature store."""
    print(f"🎯 Feature Store Management: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        ml_manager = MLInfrastructureManager(warehouse)
        feature_store = ml_manager.feature_store
        
        if action == 'list':
            if not feature_store.metadata:
                print("No features registered in store")
                return True
            
            print(f"\n📋 Registered Features:")
            for feature_name, feature_list in feature_store.metadata.items():
                latest = feature_list[-1] if feature_list else None
                if latest:
                    print(f"\n• {feature_name}")
                    print(f"  Type: {latest['feature_type']}")
                    print(f"  Description: {latest.get('description', 'N/A')}")
                    print(f"  Versions: {len(latest.get('versions', []))}")
                    print(f"  Tags: {', '.join(latest.get('tags', []))}")
        
        elif action == 'register':
            feature_name = kwargs.get('feature_name')
            feature_type = kwargs.get('feature_type', 'numeric')
            description = kwargs.get('description')
            
            if not feature_name:
                print("❌ feature_name is required")
                return False
            
            feature_id = feature_store.register_feature(
                feature_name=feature_name,
                feature_type=feature_type,
                description=description,
                tags=kwargs.get('tags', [])
            )
            
            print(f"✅ Registered feature: {feature_id}")
        
        elif action == 'compute':
            # Example: Compute article volume features
            feature_eng = FeatureEngineering()
            
            print("📊 Computing article volume features...")
            
            # Get recent article metrics
            with warehouse.get_connection() as conn:
                query = """
                    SELECT * FROM analytics.article_metrics
                    WHERE time >= NOW() - INTERVAL '7 days'
                    ORDER BY time
                """
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                print("⚠️ No article metrics found")
                return False
            
            # Compute features
            features = feature_eng.compute_article_volume_features(df)
            
            # Store in feature store
            feature_store.compute_features(
                feature_name='article_volume_features',
                start_time=datetime.utcnow() - timedelta(days=7),
                end_time=datetime.utcnow(),
                computation_function=lambda x: features
            )
            
            print(f"✅ Computed and stored {len(features)} feature records")
        
        else:
            print(f"❌ Unknown action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to manage feature store: {e}")
        return False


def run_clustering_analysis(action, **kwargs):
    """Run article clustering analysis."""
    print(f"🔬 Article Clustering: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        db_adapter = DatabaseAdapter()
        clusterer = ArticleClusterer(db_adapter, warehouse)
        
        if action == 'topic':
            method = kwargs.get('method', 'kmeans')
            n_clusters = kwargs.get('n_clusters')
            days_back = kwargs.get('days_back', 30)
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            result = clusterer.cluster_articles_by_topic(
                start_date=start_date,
                end_date=end_date,
                n_clusters=n_clusters,
                method=method
            )
            
            if 'error' in result:
                print(f"❌ Topic clustering failed: {result['error']}")
                return False
            
            print(f"✅ Topic clustering completed:")
            print(f"  • Articles analyzed: {result['n_articles']:,}")
            print(f"  • Clusters found: {result['n_clusters']}")
            print(f"  • Method: {method}")
            print(f"  • Silhouette score: {result['metrics'].get('silhouette_score', 'N/A')}")
            
        elif action == 'sources':
            days_back = kwargs.get('days_back', 30)
            
            result = clusterer.cluster_by_source_behavior(days_back=days_back)
            
            if 'error' in result:
                print(f"❌ Source clustering failed: {result['error']}")
                return False
            
            print(f"✅ Source clustering completed:")
            print(f"  • Sources analyzed: {result['n_sources']}")
            print(f"  • Clusters found: {result['n_clusters']}")
            
            # Show cluster profiles
            if 'cluster_profiles' in result:
                print(f"\n📊 Cluster Profiles:")
                for cluster_id, profile in result['cluster_profiles'].items():
                    print(f"  • {profile['label']}: {profile['n_sources']} sources")
        
        elif action == 'temporal':
            time_window = kwargs.get('time_window', '1H')
            days_back = kwargs.get('days_back', 30)
            
            result = clusterer.cluster_temporal_patterns(
                time_window=time_window,
                days_back=days_back
            )
            
            if 'error' in result:
                print(f"❌ Temporal clustering failed: {result['error']}")
                return False
            
            print(f"✅ Temporal clustering completed:")
            print(f"  • Time periods analyzed: {result['n_time_periods']}")
            print(f"  • Patterns found: {result['n_clusters']}")
            print(f"  • Time window: {time_window}")
        
        else:
            print(f"❌ Unknown clustering action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run clustering analysis: {e}")
        return False


def run_recommendations(action, **kwargs):
    """Manage recommendation engine."""
    print(f"🎯 Recommendation Engine: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        db_adapter = DatabaseAdapter()
        recommender = RecommendationEngine(db_adapter, warehouse)
        
        if action == 'train_collaborative':
            days_back = kwargs.get('days_back', 90)
            model_type = kwargs.get('model_type', 'als')
            
            result = recommender.train_collaborative_filter(
                interaction_type='view',
                days_back=days_back,
                model_type=model_type
            )
            
            if 'error' in result:
                print(f"❌ Collaborative filter training failed: {result['error']}")
                return False
            
            print(f"✅ Collaborative filter trained:")
            print(f"  • Model type: {model_type}")
            print(f"  • Users: {result['n_users']:,}")
            print(f"  • Items: {result['n_items']:,}")
            print(f"  • Interactions: {result['n_interactions']:,}")
        
        elif action == 'train_content':
            days_back = kwargs.get('days_back', 90)
            feature_type = kwargs.get('feature_type', 'embeddings')
            
            result = recommender.train_content_model(
                days_back=days_back,
                feature_type=feature_type
            )
            
            if 'error' in result:
                print(f"❌ Content model training failed: {result['error']}")
                return False
            
            print(f"✅ Content model trained:")
            print(f"  • Feature type: {feature_type}")
            print(f"  • Articles: {result['n_articles']:,}")
            print(f"  • Features: {result['n_features']:,}")
        
        elif action == 'get_recommendations':
            user_id = kwargs.get('user_id', 'test_user')
            method = kwargs.get('method', 'hybrid')
            n_recs = kwargs.get('n_recommendations', 10)
            
            recommendations = recommender.get_user_recommendations(
                user_id=user_id,
                n_recommendations=n_recs,
                method=method
            )
            
            print(f"✅ Generated {len(recommendations)} recommendations for {user_id}:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec['title'][:50]}... (score: {rec['score']:.3f})")
        
        elif action == 'trending':
            n_recs = kwargs.get('n_recommendations', 10)
            
            trending = recommender.get_trending_recommendations(
                n_recommendations=n_recs
            )
            
            print(f"✅ Top {len(trending)} trending articles:")
            for i, article in enumerate(trending, 1):
                metrics = article.get('metrics', {})
                print(f"  {i}. {article['title'][:50]}... (viewers: {metrics.get('viewers', 0)})")
        
        else:
            print(f"❌ Unknown recommendation action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run recommendations: {e}")
        return False


def validate_analytics_system():
    """Run comprehensive analytics validation."""
    print("🔍 Running comprehensive analytics validation...")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        db_adapter = DatabaseAdapter()
        validator = AnalyticsValidator(db_adapter, warehouse)
        
        # Run full validation
        results = validator.run_comprehensive_validation()
        
        print(f"✅ Validation completed - Overall health: {results['overall_health']}")
        
        # Data quality results
        if 'data_quality' in results:
            dq = results['data_quality']
            print(f"\n📊 Data Quality:")
            print(f"  • Tables checked: {len(dq['tables'])}")
            print(f"  • Issues found: {len(dq['issues'])}")
            
            for issue in dq['issues'][:3]:  # Top 3 issues
                print(f"    ⚠️ {issue['type']}: {issue.get('description', 'See details')}")
        
        # Metric accuracy results
        if 'metric_accuracy' in results:
            ma = results['metric_accuracy']
            print(f"\n🎯 Metric Accuracy:")
            print(f"  • Metrics validated: {len(ma['metrics'])}")
            print(f"  • Issues found: {len(ma['issues'])}")
        
        # Performance results
        if 'performance_benchmarks' in results:
            pb = results['performance_benchmarks']
            print(f"\n⚡ Performance Benchmarks:")
            print(f"  • Total benchmark time: {pb['summary']['total_benchmark_time']:.2f}s")
            print(f"  • Performance rating: {pb['summary']['performance_rating']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to validate analytics system: {e}")
        return False


def optimize_performance(action, **kwargs):
    """Run performance optimizations."""
    print(f"⚡ Performance Optimization: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        db_adapter = DatabaseAdapter()
        optimizer = PerformanceOptimizer(db_adapter, warehouse)
        
        if action == 'analyze':
            # Monitor resource usage
            resource_stats = optimizer.monitor_resource_usage()
            
            print(f"📊 Current Resource Usage:")
            print(f"  • CPU: {resource_stats['cpu']['percent']:.1f}%")
            print(f"  • Memory: {resource_stats['memory']['percent']:.1f}%")
            print(f"  • Disk: {resource_stats['disk']['percent']:.1f}%")
            
            # Get recommendations
            recommendations = optimizer.get_performance_recommendations()
            
            if recommendations:
                print(f"\n💡 Optimization Recommendations ({len(recommendations)}):")
                for rec in recommendations[:3]:
                    print(f"  • {rec['title']} ({rec['priority']} priority)")
                    print(f"    {rec['description']}")
            else:
                print("\n✅ No optimization recommendations at this time")
        
        elif action == 'precompute':
            # Precompute common aggregations
            result = optimizer.precompute_common_aggregations()
            
            print(f"✅ Precomputation completed:")
            for agg_name, agg_result in result['aggregations'].items():
                if agg_result['status'] == 'success':
                    print(f"  • {agg_name}: {agg_result['rows']:,} rows in {agg_result['computation_time']:.2f}s")
                else:
                    print(f"  ❌ {agg_name}: {agg_result.get('error', 'Failed')}")
        
        elif action == 'optimize_dashboard':
            # Optimize dashboard queries
            result = optimizer.optimize_dashboard_queries()
            
            print(f"✅ Dashboard optimization completed:")
            for query_name, query_result in result['queries'].items():
                if query_result['status'] == 'success':
                    print(f"  • {query_name}: {query_result['execution_time']:.3f}s ({query_result['rows']} rows)")
                else:
                    print(f"  ❌ {query_name}: {query_result.get('error', 'Failed')}")
        
        elif action == 'cache_stats':
            # Show cache statistics
            stats = optimizer.query_cache.get_stats()
            print(f"📈 Query Cache Statistics:")
            print(f"  • Size: {stats['size']}/{stats['max_size']}")
            print(f"  • Hit rate: {stats['hit_rate']:.1%}")
            print(f"  • Hits: {stats['hit_count']:,}")
            print(f"  • Misses: {stats['miss_count']:,}")
        
        else:
            print(f"❌ Unknown optimization action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to run performance optimization: {e}")
        return False


def run_ab_test(action, **kwargs):
    """Manage A/B tests for models."""
    print(f"🔬 A/B Test Management: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        ml_manager = MLInfrastructureManager(warehouse)
        ab_testing = ml_manager.ab_testing
        
        if action == 'create':
            print("A/B test creation should be done through the ML pipeline")
            print("Example usage:")
            print("  1. Load two models from registry")
            print("  2. Create test with traffic split configuration")
            print("  3. Route predictions through get_model_for_request()")
            return True
        
        elif action == 'list':
            # List active tests
            if not ab_testing.active_tests:
                print("No active A/B tests")
            else:
                print(f"\n📋 Active A/B Tests ({len(ab_testing.active_tests)} total):")
                for test_id, test_data in ab_testing.active_tests.items():
                    data = test_data['data']
                    print(f"\n• {data['test_name']}")
                    print(f"  ID: {test_id}")
                    print(f"  Status: {data['status']}")
                    print(f"  Traffic Split: {data['config']['traffic_split']}% to model B")
                    print(f"  Model A predictions: {data['model_a_stats']['predictions']}")
                    print(f"  Model B predictions: {data['model_b_stats']['predictions']}")
        
        elif action == 'evaluate':
            test_id = kwargs.get('test_id')
            if not test_id:
                print("❌ test_id is required")
                return False
            
            results = ab_testing.evaluate_test(test_id)
            
            if 'error' in results:
                print(f"❌ Evaluation failed: {results['error']}")
                return False
            
            print(f"\n📊 A/B Test Evaluation: {results['test_name']}")
            print(f"Total predictions: {results['total_predictions']}")
            
            print(f"\nModel A:")
            print(f"  • Predictions: {results['model_a']['predictions']} ({results['model_a']['percentage']:.1f}%)")
            if 'metrics' in results['model_a']:
                for metric, value in results['model_a']['metrics'].items():
                    print(f"  • {metric}: {value:.4f}")
            
            print(f"\nModel B:")
            print(f"  • Predictions: {results['model_b']['predictions']} ({results['model_b']['percentage']:.1f}%)")
            if 'metrics' in results['model_b']:
                for metric, value in results['model_b']['metrics'].items():
                    print(f"  • {metric}: {value:.4f}")
            
            if 'winner' in results:
                print(f"\n🏆 Winner: {results['winner']}")
                print(f"Improvement: {results['improvement']:.4f}")
            
            if 'statistical_test' in results:
                sig = results['statistical_test']
                print(f"\nStatistical Significance:")
                print(f"  • p-value: {sig['p_value']:.4f}")
                print(f"  • Significant: {'Yes' if sig['significant'] else 'No'}")
        
        else:
            print(f"❌ Unknown action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to manage A/B test: {e}")
        return False


def manage_scheduled_reports(action, **kwargs):
    """Manage scheduled reports."""
    print(f"📅 Managing scheduled reports: {action}")
    
    try:
        warehouse = AnalyticsDataWarehouse()
        generator = ReportGenerator(warehouse)
        scheduler = ReportScheduler(generator)
        
        if action == 'list':
            reports = scheduler.get_scheduled_reports()
            
            if not reports:
                print("No scheduled reports found")
                return True
            
            print(f"\n📋 Scheduled Reports ({len(reports)} total):")
            for report in reports:
                status = "🟢 Enabled" if report['enabled'] else "🔴 Disabled"
                print(f"\n{status} {report['name']}")
                print(f"   ID: {report['report_id']}")
                print(f"   Template: {report['template_id']}")
                print(f"   Schedule: {report['schedule_config']}")
                print(f"   Last Run: {report['last_run'] or 'Never'}")
                print(f"   Next Run: {report['next_run'] or 'Not scheduled'}")
                print(f"   Run Count: {report['run_count']}")
                if report['error_count'] > 0:
                    print(f"   ⚠️ Errors: {report['error_count']}")
        
        elif action == 'create':
            name = kwargs.get('name')
            template_id = kwargs.get('template_id')
            schedule_type = kwargs.get('schedule_type', 'daily')
            schedule_time = kwargs.get('schedule_time', '09:00')
            
            if not all([name, template_id]):
                print("❌ Name and template_id are required for creating scheduled reports")
                return False
            
            schedule_config = {
                'type': schedule_type,
                'time': schedule_time
            }
            
            if schedule_type == 'weekly':
                schedule_config['day_of_week'] = kwargs.get('day_of_week', 'monday')
            elif schedule_type == 'monthly':
                schedule_config['day_of_month'] = kwargs.get('day_of_month', 1)
            elif schedule_type == 'interval':
                schedule_config['interval_hours'] = kwargs.get('interval_hours', 24)
            
            report_id = scheduler.create_scheduled_report(
                name=name,
                template_id=template_id,
                schedule_config=schedule_config,
                parameters=kwargs.get('parameters', {}),
                delivery_config=kwargs.get('delivery_config', {})
            )
            
            print(f"✅ Created scheduled report: {name} ({report_id})")
        
        elif action == 'run':
            report_id = kwargs.get('report_id')
            if not report_id:
                print("❌ report_id is required for running scheduled reports")
                return False
            
            result = scheduler.run_scheduled_report(report_id, force=True)
            
            if 'error' in result:
                print(f"❌ Failed to run scheduled report: {result['error']}")
                return False
            
            print(f"✅ Scheduled report executed successfully")
            if result.get('file_path'):
                print(f"   • File: {result['file_path']}")
        
        elif action == 'delete':
            report_id = kwargs.get('report_id')
            if not report_id:
                print("❌ report_id is required for deleting scheduled reports")
                return False
            
            success = scheduler.delete_scheduled_report(report_id)
            
            if not success:
                print(f"❌ Failed to delete scheduled report: {report_id}")
                return False
            
            print(f"✅ Deleted scheduled report: {report_id}")
        
        elif action == 'start_scheduler':
            scheduler.start_scheduler()
            print("✅ Report scheduler started")
        
        elif action == 'stop_scheduler':
            scheduler.stop_scheduler()
            print("✅ Report scheduler stopped")
        
        else:
            print(f"❌ Unknown action: {action}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to manage scheduled reports: {e}")
        return False


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Mimir Analytics Manager - Business Intelligence and Analytics CLI"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup analytics infrastructure')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show analytics system status')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect metrics from main database')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run trend analysis')
    analyze_parser.add_argument('--days', type=int, default=30, help='Days to analyze (default: 30)')
    analyze_parser.add_argument('--keywords-only', action='store_true', help='Analyze only keyword trends')
    analyze_parser.add_argument('--sources-only', action='store_true', help='Analyze only source performance')
    
    # Continuous command
    continuous_parser = subparsers.add_parser('continuous', help='Run continuous metrics collection')
    continuous_parser.add_argument('--interval', type=int, default=60, help='Collection interval in minutes (default: 60)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old analytics data')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Days to keep (default: 90)')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate predictive forecasts')
    forecast_parser.add_argument('--type', choices=['all', 'volume', 'topics', 'sentiment', 'alerts'], 
                                default='all', help='Type of forecast to generate (default: all)')
    forecast_parser.add_argument('--days', type=int, default=7, help='Days ahead to forecast (default: 7)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run analytics dashboard server')
    dashboard_parser.add_argument('--port', type=int, default=5001, help='Server port (default: 5001)')
    dashboard_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Report commands
    report_parser = subparsers.add_parser('report', help='Report generation and management')
    report_subparsers = report_parser.add_subparsers(dest='report_action', help='Report actions')
    
    # List templates
    list_templates_parser = report_subparsers.add_parser('templates', help='List available report templates')
    
    # Generate report
    generate_parser = report_subparsers.add_parser('generate', help='Generate a report')
    generate_parser.add_argument('template_id', help='Template ID to use')
    generate_parser.add_argument('--format', choices=['html', 'pdf', 'json'], default='pdf', help='Output format')
    generate_parser.add_argument('--time-period-days', type=int, default=30, help='Analysis time period in days')
    generate_parser.add_argument('--output', help='Output file path')
    
    # Scheduled reports
    scheduled_parser = report_subparsers.add_parser('scheduled', help='Manage scheduled reports')
    scheduled_subparsers = scheduled_parser.add_subparsers(dest='scheduled_action', help='Scheduled report actions')
    
    # List scheduled reports
    scheduled_list_parser = scheduled_subparsers.add_parser('list', help='List scheduled reports')
    
    # Create scheduled report
    scheduled_create_parser = scheduled_subparsers.add_parser('create', help='Create scheduled report')
    scheduled_create_parser.add_argument('name', help='Report name')
    scheduled_create_parser.add_argument('template_id', help='Template ID')
    scheduled_create_parser.add_argument('--schedule-type', choices=['daily', 'weekly', 'monthly', 'hourly', 'interval'], 
                                       default='daily', help='Schedule type')
    scheduled_create_parser.add_argument('--schedule-time', default='09:00', help='Schedule time (HH:MM)')
    scheduled_create_parser.add_argument('--day-of-week', help='Day of week for weekly schedules')
    scheduled_create_parser.add_argument('--day-of-month', type=int, help='Day of month for monthly schedules')
    scheduled_create_parser.add_argument('--interval-hours', type=int, help='Interval hours for interval schedules')
    
    # Run scheduled report
    scheduled_run_parser = scheduled_subparsers.add_parser('run', help='Run scheduled report')
    scheduled_run_parser.add_argument('report_id', help='Report ID to run')
    
    # Delete scheduled report
    scheduled_delete_parser = scheduled_subparsers.add_parser('delete', help='Delete scheduled report')
    scheduled_delete_parser.add_argument('report_id', help='Report ID to delete')
    
    # Scheduler control
    scheduler_start_parser = scheduled_subparsers.add_parser('start-scheduler', help='Start report scheduler')
    scheduler_stop_parser = scheduled_subparsers.add_parser('stop-scheduler', help='Stop report scheduler')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export analytics data')
    export_parser.add_argument('data_type', choices=['articles', 'article_metrics', 'source_metrics', 
                                                     'keyword_metrics', 'trend_analysis', 'user_activity'],
                              help='Type of data to export')
    export_parser.add_argument('--format', choices=['csv', 'json', 'jsonl', 'excel', 'parquet', 
                                                   'avro', 'tsv', 'xml', 'sql'],
                              default='csv', help='Export format')
    export_parser.add_argument('--output', help='Output file path')
    export_parser.add_argument('--compress', action='store_true', help='Compress output')
    export_parser.add_argument('--stream', action='store_true', help='Use streaming for large datasets')
    
    # ML commands
    ml_parser = subparsers.add_parser('ml', help='ML infrastructure management')
    ml_subparsers = ml_parser.add_subparsers(dest='ml_action', help='ML actions')
    
    # ML models
    ml_models_parser = ml_subparsers.add_parser('models', help='Manage ML models')
    ml_models_parser.add_argument('action', choices=['list', 'promote', 'experiment'],
                                 help='Model management action')
    ml_models_parser.add_argument('--model-name', help='Model name')
    ml_models_parser.add_argument('--version', help='Model version')
    ml_models_parser.add_argument('--stage', choices=['staging', 'production', 'archived'],
                                 help='Promotion stage')
    
    # Feature store
    feature_parser = ml_subparsers.add_parser('features', help='Manage feature store')
    feature_parser.add_argument('action', choices=['list', 'register', 'compute'],
                               help='Feature store action')
    feature_parser.add_argument('--feature-name', help='Feature name')
    feature_parser.add_argument('--feature-type', choices=['numeric', 'categorical', 'embedding'],
                               default='numeric', help='Feature type')
    feature_parser.add_argument('--description', help='Feature description')
    
    # A/B testing
    ab_parser = ml_subparsers.add_parser('abtest', help='Manage A/B tests')
    ab_parser.add_argument('action', choices=['list', 'create', 'evaluate'],
                          help='A/B test action')
    ab_parser.add_argument('--test-id', help='Test ID')
    
    # Clustering commands
    clustering_parser = subparsers.add_parser('cluster', help='Article clustering analysis')
    clustering_parser.add_argument('action', choices=['topic', 'sources', 'temporal'],
                                  help='Clustering analysis type')
    clustering_parser.add_argument('--method', choices=['kmeans', 'dbscan', 'hdbscan', 'hierarchical'],
                                  default='kmeans', help='Clustering method')
    clustering_parser.add_argument('--n-clusters', type=int, help='Number of clusters')
    clustering_parser.add_argument('--days-back', type=int, default=30, help='Days of data to analyze')
    clustering_parser.add_argument('--time-window', default='1H', help='Time window for temporal clustering')
    
    # Recommendation commands
    recommendation_parser = subparsers.add_parser('recommend', help='Recommendation engine')
    recommendation_parser.add_argument('action', choices=['train_collaborative', 'train_content', 
                                                         'get_recommendations', 'trending'],
                                     help='Recommendation action')
    recommendation_parser.add_argument('--days-back', type=int, default=90, help='Training data period')
    recommendation_parser.add_argument('--model-type', choices=['als', 'bpr', 'lmf'], 
                                      default='als', help='Collaborative filtering model type')
    recommendation_parser.add_argument('--feature-type', choices=['tfidf', 'embeddings'],
                                      default='embeddings', help='Content feature type')
    recommendation_parser.add_argument('--user-id', default='test_user', help='User ID for recommendations')
    recommendation_parser.add_argument('--method', choices=['collaborative', 'content', 'hybrid'],
                                      default='hybrid', help='Recommendation method')
    recommendation_parser.add_argument('--n-recommendations', type=int, default=10,
                                      help='Number of recommendations')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate analytics system')
    
    # Optimization commands
    optimize_parser = subparsers.add_parser('optimize', help='Performance optimization')
    optimize_parser.add_argument('action', choices=['analyze', 'precompute', 'optimize_dashboard', 'cache_stats'],
                                help='Optimization action')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set environment variables for TimescaleDB if not set
    if 'TIMESCALE_HOST' not in os.environ:
        os.environ['TIMESCALE_HOST'] = 'localhost'
    if 'TIMESCALE_PORT' not in os.environ:
        os.environ['TIMESCALE_PORT'] = '5433'
        
    success = False
    
    if args.command == 'setup':
        success = setup_analytics_infrastructure()
    elif args.command == 'status':
        success = show_analytics_status()
    elif args.command == 'collect':
        success = collect_metrics()
    elif args.command == 'analyze':
        success = analyze_trends(
            days_back=args.days,
            keywords_only=args.keywords_only,
            sources_only=args.sources_only
        )
    elif args.command == 'continuous':
        success = run_continuous_collection(interval_minutes=args.interval)
    elif args.command == 'cleanup':
        success = cleanup_old_data(days_to_keep=args.days)
    elif args.command == 'forecast':
        success = generate_forecasts(
            forecast_type=args.type,
            days_ahead=args.days
        )
    elif args.command == 'dashboard':
        success = run_dashboard_server(
            port=args.port,
            debug=args.debug
        )
    elif args.command == 'report':
        if args.report_action == 'templates':
            success = list_report_templates()
        elif args.report_action == 'generate':
            success = generate_report(
                template_id=args.template_id,
                output_format=args.format,
                time_period_days=args.time_period_days,
                output_file=args.output
            )
        elif args.report_action == 'scheduled':
            if args.scheduled_action == 'list':
                success = manage_scheduled_reports('list')
            elif args.scheduled_action == 'create':
                success = manage_scheduled_reports(
                    'create',
                    name=args.name,
                    template_id=args.template_id,
                    schedule_type=args.schedule_type,
                    schedule_time=args.schedule_time,
                    day_of_week=args.day_of_week,
                    day_of_month=args.day_of_month,
                    interval_hours=args.interval_hours
                )
            elif args.scheduled_action == 'run':
                success = manage_scheduled_reports('run', report_id=args.report_id)
            elif args.scheduled_action == 'delete':
                success = manage_scheduled_reports('delete', report_id=args.report_id)
            elif args.scheduled_action == 'start-scheduler':
                success = manage_scheduled_reports('start_scheduler')
            elif args.scheduled_action == 'stop-scheduler':
                success = manage_scheduled_reports('stop_scheduler')
            else:
                print("No scheduled report action specified")
                success = False
        else:
            print("No report action specified")
            success = False
    elif args.command == 'export':
        success = export_data(
            data_type=args.data_type,
            format=args.format,
            output=args.output,
            compress=args.compress,
            stream=args.stream
        )
    elif args.command == 'ml':
        if args.ml_action == 'models':
            success = manage_ml_models(
                action=args.action,
                model_name=args.model_name,
                version=args.version,
                stage=args.stage
            )
        elif args.ml_action == 'features':
            success = manage_feature_store(
                action=args.action,
                feature_name=args.feature_name,
                feature_type=args.feature_type,
                description=args.description
            )
        elif args.ml_action == 'abtest':
            success = run_ab_test(
                action=args.action,
                test_id=args.test_id
            )
        else:
            print("No ML action specified")
            success = False
    elif args.command == 'cluster':
        success = run_clustering_analysis(
            action=args.action,
            method=args.method,
            n_clusters=args.n_clusters,
            days_back=args.days_back,
            time_window=args.time_window
        )
    elif args.command == 'recommend':
        success = run_recommendations(
            action=args.action,
            days_back=args.days_back,
            model_type=args.model_type,
            feature_type=args.feature_type,
            user_id=args.user_id,
            method=args.method,
            n_recommendations=args.n_recommendations
        )
    elif args.command == 'validate':
        success = validate_analytics_system()
    elif args.command == 'optimize':
        success = optimize_performance(action=args.action)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()