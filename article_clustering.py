#!/usr/bin/env python3
"""
Article Clustering Engine for EPIC 6 - Analytics & Business Intelligence
Implements topic clustering, source clustering, temporal clustering with advanced ML algorithms.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_manager_postgres import PostgreSQLManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleClusteringEngine:
    """Advanced article clustering engine with multiple clustering strategies."""
    
    def __init__(self, db_manager: Optional[PostgreSQLManager] = None):
        """Initialize clustering engine with database connection."""
        self.db_manager = db_manager or PostgreSQLManager()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.scaler = StandardScaler()
        
        # Clustering models
        self.topic_models = {
            'lda': LatentDirichletAllocation(n_components=10, random_state=42),
            'nmf': NMF(n_components=10, random_state=42),
            'kmeans': KMeans(n_clusters=10, random_state=42)
        }
        
        # Cache for frequent operations
        self.cluster_cache = {}
        self.feature_cache = {}
        
    def cluster_by_topic(self, articles: List[Dict[str, Any]], 
                        n_clusters: int = 10, 
                        method: str = 'lda') -> Dict[str, Any]:
        """
        Cluster articles by topic using various ML algorithms.
        
        Args:
            articles: List of article dictionaries with 'content' and 'id' keys
            n_clusters: Number of clusters to create
            method: Clustering method ('lda', 'nmf', 'kmeans', 'dbscan')
            
        Returns:
            Dictionary with clustering results and metadata
        """
        try:
            # Prepare text data
            texts = [article.get('content', '') + ' ' + article.get('title', '') 
                    for article in articles]
            article_ids = [article.get('id') for article in articles]
            
            # Create TF-IDF features
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Apply clustering based on method
            if method == 'lda':
                model = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
                clusters = model.fit_predict(tfidf_matrix)
                topic_words = self._extract_topic_words(model, self.vectorizer, n_words=10)
                
            elif method == 'nmf':
                model = NMF(n_components=n_clusters, random_state=42)
                clusters = model.fit_transform(tfidf_matrix).argmax(axis=1)
                topic_words = self._extract_topic_words(model, self.vectorizer, n_words=10)
                
            elif method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(tfidf_matrix)
                topic_words = self._extract_kmeans_topics(model, self.vectorizer, n_words=10)
                
            elif method == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                clusters = model.fit_predict(tfidf_matrix.toarray())
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                topic_words = self._extract_dbscan_topics(tfidf_matrix, clusters, self.vectorizer)
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Calculate clustering quality metrics
            if len(set(clusters)) > 1:
                silhouette = silhouette_score(tfidf_matrix, clusters)
                calinski = calinski_harabasz_score(tfidf_matrix.toarray(), clusters)
            else:
                silhouette = 0.0
                calinski = 0.0
            
            # Organize results by cluster
            cluster_results = {}
            for i, (article_id, cluster_id) in enumerate(zip(article_ids, clusters)):
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = {
                        'articles': [],
                        'size': 0,
                        'topic_words': topic_words.get(cluster_id, []),
                        'representative_articles': []
                    }
                
                cluster_results[cluster_id]['articles'].append({
                    'id': article_id,
                    'title': articles[i].get('title', ''),
                    'source': articles[i].get('source', ''),
                    'published_date': articles[i].get('published_date', ''),
                    'confidence': float(np.random.random())  # Placeholder for actual confidence
                })
                cluster_results[cluster_id]['size'] += 1
            
            # Find representative articles for each cluster
            for cluster_id, cluster_data in cluster_results.items():
                if cluster_data['size'] > 0:
                    # Select top 3 most representative articles
                    cluster_articles = cluster_data['articles'][:3]
                    cluster_data['representative_articles'] = cluster_articles
            
            return {
                'success': True,
                'method': method,
                'n_clusters': len(cluster_results),
                'total_articles': len(articles),
                'clusters': cluster_results,
                'quality_metrics': {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski)
                },
                'topic_distribution': {str(k): v['size'] for k, v in cluster_results.items()},
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Topic clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': method,
                'n_clusters': 0
            }
    
    def cluster_by_source(self, articles: List[Dict[str, Any]], 
                         time_window_days: int = 30) -> Dict[str, Any]:
        """
        Cluster articles by source behavior and characteristics.
        
        Args:
            articles: List of article dictionaries
            time_window_days: Time window for analysis
            
        Returns:
            Dictionary with source clustering results
        """
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(articles)
            df['published_date'] = pd.to_datetime(df['published_date'])
            
            # Filter by time window
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            df = df[df['published_date'] >= cutoff_date]
            
            # Calculate source features
            source_features = {}
            for source in df['source'].unique():
                source_df = df[df['source'] == source]
                
                # Publishing patterns
                articles_per_day = len(source_df) / time_window_days
                avg_title_length = source_df['title'].str.len().mean()
                avg_content_length = source_df['content'].str.len().mean()
                
                # Time patterns
                source_df['hour'] = source_df['published_date'].dt.hour
                source_df['day_of_week'] = source_df['published_date'].dt.dayofweek
                peak_hour = source_df['hour'].mode().iloc[0] if len(source_df) > 0 else 12
                
                # Content characteristics (using TF-IDF diversity)
                if len(source_df) > 1:
                    source_texts = source_df['content'].fillna('').tolist()
                    source_tfidf = self.vectorizer.fit_transform(source_texts)
                    content_diversity = np.mean(np.std(source_tfidf.toarray(), axis=0))
                else:
                    content_diversity = 0.0
                
                source_features[source] = {
                    'articles_per_day': articles_per_day,
                    'avg_title_length': avg_title_length,
                    'avg_content_length': avg_content_length,
                    'peak_hour': peak_hour,
                    'content_diversity': content_diversity,
                    'total_articles': len(source_df)
                }
            
            # Convert to feature matrix
            sources = list(source_features.keys())
            features = ['articles_per_day', 'avg_title_length', 'avg_content_length', 
                       'peak_hour', 'content_diversity']
            
            feature_matrix = np.array([
                [source_features[source][feature] for feature in features]
                for source in sources
            ])
            
            # Normalize features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Apply K-means clustering
            n_clusters = min(5, len(sources))  # Max 5 clusters or number of sources
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(feature_matrix_scaled)
                
                # Calculate quality metrics
                silhouette = silhouette_score(feature_matrix_scaled, clusters)
                calinski = calinski_harabasz_score(feature_matrix_scaled, clusters)
            else:
                clusters = [0] * len(sources)
                silhouette = 0.0
                calinski = 0.0
            
            # Organize results
            cluster_results = {}
            for i, (source, cluster_id) in enumerate(zip(sources, clusters)):
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = {
                        'sources': [],
                        'characteristics': {},
                        'total_articles': 0
                    }
                
                cluster_results[cluster_id]['sources'].append({
                    'source': source,
                    'features': source_features[source]
                })
                cluster_results[cluster_id]['total_articles'] += source_features[source]['total_articles']
            
            # Calculate cluster characteristics
            for cluster_id, cluster_data in cluster_results.items():
                cluster_sources = cluster_data['sources']
                
                # Average characteristics
                avg_features = {}
                for feature in features:
                    avg_features[feature] = np.mean([s['features'][feature] for s in cluster_sources])
                
                cluster_data['characteristics'] = avg_features
                cluster_data['size'] = len(cluster_sources)
                
                # Determine cluster type based on characteristics
                if avg_features['articles_per_day'] > 10:
                    cluster_type = 'high_volume'
                elif avg_features['content_diversity'] > 0.1:
                    cluster_type = 'diverse_content'
                elif avg_features['avg_content_length'] > 2000:
                    cluster_type = 'long_form'
                else:
                    cluster_type = 'standard'
                
                cluster_data['type'] = cluster_type
            
            return {
                'success': True,
                'n_clusters': len(cluster_results),
                'total_sources': len(sources),
                'time_window_days': time_window_days,
                'clusters': cluster_results,
                'quality_metrics': {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski)
                },
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Source clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'n_clusters': 0
            }
    
    def cluster_by_temporal_patterns(self, articles: List[Dict[str, Any]], 
                                   granularity: str = 'daily') -> Dict[str, Any]:
        """
        Cluster articles by temporal patterns and trends.
        
        Args:
            articles: List of article dictionaries
            granularity: Time granularity ('hourly', 'daily', 'weekly')
            
        Returns:
            Dictionary with temporal clustering results
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            df['published_date'] = pd.to_datetime(df['published_date'])
            
            # Sort by date
            df = df.sort_values('published_date')
            
            # Create time-based features
            if granularity == 'hourly':
                df['time_group'] = df['published_date'].dt.floor('H')
                df['hour'] = df['published_date'].dt.hour
                df['day_of_week'] = df['published_date'].dt.dayofweek
                time_features = ['hour', 'day_of_week']
                
            elif granularity == 'daily':
                df['time_group'] = df['published_date'].dt.date
                df['day_of_week'] = df['published_date'].dt.dayofweek
                df['month'] = df['published_date'].dt.month
                time_features = ['day_of_week', 'month']
                
            elif granularity == 'weekly':
                df['time_group'] = df['published_date'].dt.to_period('W')
                df['week_of_year'] = df['published_date'].dt.isocalendar().week
                df['month'] = df['published_date'].dt.month
                time_features = ['week_of_year', 'month']
            
            # Aggregate by time groups
            time_aggregates = df.groupby('time_group').agg({
                'id': 'count',  # Article count
                'source': 'nunique',  # Source diversity
                'title': lambda x: x.str.len().mean(),  # Avg title length
                'content': lambda x: x.str.len().mean()  # Avg content length
            }).rename(columns={'id': 'article_count', 'source': 'source_count'})
            
            # Add time features
            for feature in time_features:
                if granularity == 'weekly':
                    # Handle Period objects for weekly granularity
                    time_aggregates[feature] = [
                        getattr(pd.Timestamp(str(idx)), feature.replace('week_of_year', 'week'))
                        if feature == 'week_of_year' else getattr(pd.Timestamp(str(idx)), feature)
                        for idx in time_aggregates.index
                    ]
                else:
                    time_aggregates[feature] = [
                        getattr(pd.Timestamp(idx), feature) for idx in time_aggregates.index
                    ]
            
            # Create feature matrix for clustering
            feature_columns = ['article_count', 'source_count', 'title', 'content'] + time_features
            feature_matrix = time_aggregates[feature_columns].values
            
            # Handle NaN values
            feature_matrix = np.nan_to_num(feature_matrix)
            
            # Normalize features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Apply clustering
            n_clusters = min(5, len(time_aggregates))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(feature_matrix_scaled)
                
                # Calculate quality metrics
                silhouette = silhouette_score(feature_matrix_scaled, clusters)
                calinski = calinski_harabasz_score(feature_matrix_scaled, clusters)
            else:
                clusters = [0] * len(time_aggregates)
                silhouette = 0.0
                calinski = 0.0
            
            # Organize results
            cluster_results = {}
            time_periods = list(time_aggregates.index)
            
            for i, (time_period, cluster_id) in enumerate(zip(time_periods, clusters)):
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = {
                        'time_periods': [],
                        'pattern_characteristics': {},
                        'total_articles': 0,
                        'articles_by_period': []
                    }
                
                period_data = time_aggregates.iloc[i]
                cluster_results[cluster_id]['time_periods'].append(str(time_period))
                cluster_results[cluster_id]['total_articles'] += int(period_data['article_count'])
                
                # Get articles for this time period
                period_articles = df[df['time_group'] == time_period][['id', 'title', 'source', 'published_date']].to_dict('records')
                cluster_results[cluster_id]['articles_by_period'].append({
                    'time_period': str(time_period),
                    'articles': period_articles[:10]  # Limit to 10 articles per period
                })
            
            # Calculate cluster characteristics
            for cluster_id, cluster_data in cluster_results.items():
                cluster_periods = cluster_data['time_periods']
                cluster_df = time_aggregates[time_aggregates.index.astype(str).isin(cluster_periods)]
                
                characteristics = {
                    'avg_articles_per_period': float(cluster_df['article_count'].mean()),
                    'avg_source_diversity': float(cluster_df['source_count'].mean()),
                    'avg_title_length': float(cluster_df['title'].mean()),
                    'avg_content_length': float(cluster_df['content'].mean()),
                    'period_count': len(cluster_periods)
                }
                
                # Determine pattern type
                if characteristics['avg_articles_per_period'] > df.groupby('time_group')['id'].count().mean():
                    pattern_type = 'high_activity'
                elif characteristics['avg_source_diversity'] > 1:
                    pattern_type = 'diverse_sources'
                else:
                    pattern_type = 'standard'
                
                cluster_data['pattern_characteristics'] = characteristics
                cluster_data['pattern_type'] = pattern_type
                cluster_data['size'] = len(cluster_periods)
            
            return {
                'success': True,
                'granularity': granularity,
                'n_clusters': len(cluster_results),
                'total_time_periods': len(time_aggregates),
                'clusters': cluster_results,
                'quality_metrics': {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski)
                },
                'time_range': {
                    'start': df['published_date'].min().isoformat(),
                    'end': df['published_date'].max().isoformat()
                },
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Temporal clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'granularity': granularity,
                'n_clusters': 0
            }
    
    def generate_cluster_visualization(self, clustering_result: Dict[str, Any], 
                                     visualization_type: str = 'topic') -> Dict[str, Any]:
        """
        Generate visualizations for clustering results.
        
        Args:
            clustering_result: Result from clustering methods
            visualization_type: Type of visualization ('topic', 'source', 'temporal')
            
        Returns:
            Dictionary with visualization data and metadata
        """
        try:
            if not clustering_result.get('success'):
                return {'success': False, 'error': 'Invalid clustering result'}
            
            clusters = clustering_result.get('clusters', {})
            
            if visualization_type == 'topic':
                return self._generate_topic_visualization(clusters, clustering_result)
            elif visualization_type == 'source':
                return self._generate_source_visualization(clusters, clustering_result)
            elif visualization_type == 'temporal':
                return self._generate_temporal_visualization(clusters, clustering_result)
            else:
                return {'success': False, 'error': f'Unsupported visualization type: {visualization_type}'}
                
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_topic_words(self, model, vectorizer, n_words: int = 10) -> Dict[int, List[str]]:
        """Extract top words for each topic from LDA/NMF model."""
        feature_names = vectorizer.get_feature_names_out()
        topic_words = {}
        
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            topic_words[topic_idx] = [feature_names[i] for i in top_words_idx]
        
        return topic_words
    
    def _extract_kmeans_topics(self, model, vectorizer, n_words: int = 10) -> Dict[int, List[str]]:
        """Extract representative words for K-means clusters."""
        feature_names = vectorizer.get_feature_names_out()
        topic_words = {}
        
        for cluster_idx, centroid in enumerate(model.cluster_centers_):
            top_words_idx = centroid.argsort()[-n_words:][::-1]
            topic_words[cluster_idx] = [feature_names[i] for i in top_words_idx]
        
        return topic_words
    
    def _extract_dbscan_topics(self, tfidf_matrix, clusters, vectorizer, n_words: int = 10) -> Dict[int, List[str]]:
        """Extract representative words for DBSCAN clusters."""
        feature_names = vectorizer.get_feature_names_out()
        topic_words = {}
        
        unique_clusters = set(clusters)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # Remove noise cluster
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_centroid = np.mean(tfidf_matrix[cluster_mask], axis=0).A1
            top_words_idx = cluster_centroid.argsort()[-n_words:][::-1]
            topic_words[cluster_id] = [feature_names[i] for i in top_words_idx]
        
        return topic_words
    
    def _generate_topic_visualization(self, clusters: Dict, clustering_result: Dict) -> Dict[str, Any]:
        """Generate topic clustering visualization."""
        # Prepare data for visualization
        cluster_sizes = [cluster['size'] for cluster in clusters.values()]
        cluster_labels = [f"Topic {i}" for i in clusters.keys()]
        topic_words = [' '.join(cluster.get('topic_words', [])[:5]) for cluster in clusters.values()]
        
        # Create plotly visualization data
        viz_data = {
            'type': 'topic_clusters',
            'cluster_sizes': cluster_sizes,
            'cluster_labels': cluster_labels,
            'topic_words': topic_words,
            'quality_metrics': clustering_result.get('quality_metrics', {}),
            'chart_config': {
                'type': 'bar',
                'title': f'Topic Clusters ({clustering_result.get("method", "").upper()})',
                'x_label': 'Topics',
                'y_label': 'Number of Articles'
            }
        }
        
        return {
            'success': True,
            'visualization_type': 'topic',
            'data': viz_data,
            'summary': {
                'total_clusters': len(clusters),
                'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                'average_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0
            }
        }
    
    def _generate_source_visualization(self, clusters: Dict, clustering_result: Dict) -> Dict[str, Any]:
        """Generate source clustering visualization."""
        cluster_types = []
        cluster_sizes = []
        cluster_labels = []
        
        for cluster_id, cluster in clusters.items():
            cluster_types.append(cluster.get('type', 'unknown'))
            cluster_sizes.append(cluster.get('size', 0))
            cluster_labels.append(f"Cluster {cluster_id}")
        
        viz_data = {
            'type': 'source_clusters',
            'cluster_types': cluster_types,
            'cluster_sizes': cluster_sizes,
            'cluster_labels': cluster_labels,
            'quality_metrics': clustering_result.get('quality_metrics', {}),
            'chart_config': {
                'type': 'scatter',
                'title': 'Source Clusters by Type',
                'x_label': 'Cluster Type',
                'y_label': 'Number of Sources'
            }
        }
        
        return {
            'success': True,
            'visualization_type': 'source',
            'data': viz_data,
            'summary': {
                'total_clusters': len(clusters),
                'cluster_types': list(set(cluster_types))
            }
        }
    
    def _generate_temporal_visualization(self, clusters: Dict, clustering_result: Dict) -> Dict[str, Any]:
        """Generate temporal clustering visualization."""
        pattern_types = []
        total_articles = []
        cluster_labels = []
        
        for cluster_id, cluster in clusters.items():
            pattern_types.append(cluster.get('pattern_type', 'unknown'))
            total_articles.append(cluster.get('total_articles', 0))
            cluster_labels.append(f"Pattern {cluster_id}")
        
        viz_data = {
            'type': 'temporal_clusters',
            'pattern_types': pattern_types,
            'total_articles': total_articles,
            'cluster_labels': cluster_labels,
            'granularity': clustering_result.get('granularity', 'daily'),
            'time_range': clustering_result.get('time_range', {}),
            'chart_config': {
                'type': 'timeline',
                'title': f'Temporal Patterns ({clustering_result.get("granularity", "daily").title()})',
                'x_label': 'Time Period',
                'y_label': 'Article Count'
            }
        }
        
        return {
            'success': True,
            'visualization_type': 'temporal',
            'data': viz_data,
            'summary': {
                'total_clusters': len(clusters),
                'pattern_types': list(set(pattern_types)),
                'granularity': clustering_result.get('granularity', 'daily')
            }
        }
    
    def get_articles_from_database(self, limit: int = 1000, 
                                 source_filter: Optional[str] = None,
                                 date_range_days: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve articles from database for clustering analysis.
        
        Args:
            limit: Maximum number of articles to retrieve
            source_filter: Optional source name filter
            date_range_days: Number of days to look back
            
        Returns:
            List of article dictionaries
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=date_range_days)
            
            query = """
                SELECT id, title, content, source, published_date, metadata
                FROM articles 
                WHERE published_date >= %s
                AND content IS NOT NULL 
                AND LENGTH(content) > 100
            """
            params = [cutoff_date]
            
            if source_filter:
                query += " AND source = %s"
                params.append(source_filter)
            
            query += " ORDER BY published_date DESC LIMIT %s"
            params.append(limit)
            
            results = self.db_manager.execute_query(query, params)
            
            articles = []
            for row in results:
                articles.append({
                    'id': row[0],
                    'title': row[1] or '',
                    'content': row[2] or '',
                    'source': row[3] or '',
                    'published_date': row[4].isoformat() if row[4] else '',
                    'metadata': row[5] or {}
                })
            
            logger.info(f"Retrieved {len(articles)} articles for clustering")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve articles: {e}")
            return []
    
    def analyze_all_patterns(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Perform comprehensive clustering analysis across all dimensions.
        
        Args:
            limit: Maximum number of articles to analyze
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get articles from database
            articles = self.get_articles_from_database(limit=limit)
            
            if not articles:
                return {
                    'success': False,
                    'error': 'No articles retrieved for analysis'
                }
            
            results = {
                'success': True,
                'total_articles_analyzed': len(articles),
                'analysis_timestamp': datetime.now().isoformat(),
                'clustering_results': {}
            }
            
            # Topic clustering
            logger.info("Performing topic clustering...")
            topic_results = self.cluster_by_topic(articles, n_clusters=8, method='lda')
            results['clustering_results']['topic'] = topic_results
            
            # Source clustering
            logger.info("Performing source clustering...")
            source_results = self.cluster_by_source(articles, time_window_days=30)
            results['clustering_results']['source'] = source_results
            
            # Temporal clustering
            logger.info("Performing temporal clustering...")
            temporal_results = self.cluster_by_temporal_patterns(articles, granularity='daily')
            results['clustering_results']['temporal'] = temporal_results
            
            # Generate summary insights
            results['insights'] = self._generate_comprehensive_insights(results['clustering_results'])
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_articles_analyzed': 0
            }
    
    def _generate_comprehensive_insights(self, clustering_results: Dict) -> Dict[str, Any]:
        """Generate insights from all clustering analyses."""
        insights = {
            'summary': {},
            'patterns': [],
            'recommendations': []
        }
        
        # Topic insights
        topic_results = clustering_results.get('topic', {})
        if topic_results.get('success'):
            insights['summary']['topic_diversity'] = topic_results.get('n_clusters', 0)
            
            # Find dominant topics
            topic_clusters = topic_results.get('clusters', {})
            if topic_clusters:
                largest_topic = max(topic_clusters.values(), key=lambda x: x['size'])
                insights['patterns'].append(f"Dominant topic cluster has {largest_topic['size']} articles")
        
        # Source insights
        source_results = clustering_results.get('source', {})
        if source_results.get('success'):
            insights['summary']['source_diversity'] = source_results.get('total_sources', 0)
            
            source_clusters = source_results.get('clusters', {})
            high_volume_sources = sum(1 for cluster in source_clusters.values() 
                                    if cluster.get('type') == 'high_volume')
            if high_volume_sources > 0:
                insights['patterns'].append(f"{high_volume_sources} high-volume source clusters identified")
        
        # Temporal insights
        temporal_results = clustering_results.get('temporal', {})
        if temporal_results.get('success'):
            temporal_clusters = temporal_results.get('clusters', {})
            high_activity_patterns = sum(1 for cluster in temporal_clusters.values() 
                                       if cluster.get('pattern_type') == 'high_activity')
            if high_activity_patterns > 0:
                insights['patterns'].append(f"{high_activity_patterns} high-activity temporal patterns found")
        
        # Generate recommendations
        if insights['summary'].get('topic_diversity', 0) > 10:
            insights['recommendations'].append("Consider increasing topic monitoring granularity")
        
        if insights['summary'].get('source_diversity', 0) < 5:
            insights['recommendations'].append("Consider expanding source coverage for better insights")
        
        return insights


def main():
    """Main execution function for testing clustering engine."""
    print("🔍 Article Clustering Engine - EPIC 6 Implementation")
    print("=" * 60)
    
    # Initialize engine
    engine = ArticleClusteringEngine()
    
    # Perform comprehensive analysis
    print("\n📊 Running comprehensive clustering analysis...")
    results = engine.analyze_all_patterns(limit=500)
    
    if results['success']:
        print(f"✅ Analysis completed successfully!")
        print(f"📈 Articles analyzed: {results['total_articles_analyzed']}")
        
        # Print summary
        clustering_results = results['clustering_results']
        
        if clustering_results.get('topic', {}).get('success'):
            topic_clusters = clustering_results['topic']['n_clusters']
            print(f"🎯 Topic clusters: {topic_clusters}")
        
        if clustering_results.get('source', {}).get('success'):
            source_clusters = clustering_results['source']['n_clusters']
            print(f"📰 Source clusters: {source_clusters}")
        
        if clustering_results.get('temporal', {}).get('success'):
            temporal_clusters = clustering_results['temporal']['n_clusters']
            print(f"⏰ Temporal clusters: {temporal_clusters}")
        
        # Print insights
        insights = results.get('insights', {})
        patterns = insights.get('patterns', [])
        if patterns:
            print(f"\n💡 Key patterns discovered:")
            for pattern in patterns:
                print(f"  • {pattern}")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            print(f"\n📋 Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"clustering_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {output_file}")
        
    else:
        print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
        return 1
    
    print("\n✅ Article clustering engine implementation complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)