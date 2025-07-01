"""Article Clustering Module for Mimir Analytics.

This module provides clustering capabilities for articles including topic clustering,
source clustering, temporal clustering, and visualization support.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .data_warehouse import AnalyticsDataWarehouse
from .feature_engineering import FeatureEngineering
from db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


class ArticleClusterer:
    """Main class for article clustering operations."""
    
    def __init__(self, db_adapter: DatabaseAdapter, 
                 analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the article clusterer.
        
        Args:
            db_adapter: Database adapter for article data
            analytics_warehouse: Analytics warehouse for storing results
        """
        self.db_adapter = db_adapter
        self.warehouse = analytics_warehouse
        self.feature_eng = FeatureEngineering()
        
        # Initialize sentence transformer if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer model")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Storage for clustering results
        self.results_dir = Path("analytics/clustering_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def cluster_articles_by_topic(self, start_date: datetime = None,
                                 end_date: datetime = None,
                                 n_clusters: int = None,
                                 method: str = 'kmeans',
                                 min_cluster_size: int = 5) -> Dict[str, Any]:
        """Cluster articles by topic using text content.
        
        Args:
            start_date: Start date for articles
            end_date: End date for articles
            n_clusters: Number of clusters (None for automatic)
            method: Clustering method (kmeans, dbscan, hdbscan, hierarchical)
            min_cluster_size: Minimum cluster size
            
        Returns:
            Clustering results with assignments and metrics
        """
        try:
            self.logger.info(f"Starting topic clustering with method: {method}")
            
            # Get and validate articles
            articles_df = self._get_and_validate_articles(start_date, end_date)
            if 'error' in articles_df:
                return articles_df
            
            # Generate features for clustering
            features = self._generate_clustering_features(articles_df, method)
            
            # Perform clustering based on method
            cluster_result = self._perform_clustering(features, method, n_clusters, min_cluster_size)
            if 'error' in cluster_result:
                return cluster_result
            
            cluster_labels, metrics = cluster_result['labels'], cluster_result['metrics']
            
            # Post-process clustering results
            return self._process_clustering_results(
                articles_df, features, cluster_labels, metrics, method,
                n_clusters, min_cluster_size, start_date, end_date
            )
            
        except Exception as e:
            self.logger.error(f"Topic clustering failed: {e}")
            return {'error': str(e)}
    
    def cluster_by_source_behavior(self, days_back: int = 30,
                                  features: List[str] = None) -> Dict[str, Any]:
        """Cluster sources based on their publishing behavior.
        
        Args:
            days_back: Number of days to analyze
            features: List of features to use for clustering
            
        Returns:
            Source clustering results
        """
        try:
            self.logger.info("Starting source behavior clustering")
            
            # Get source metrics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            source_metrics = self._get_source_metrics(start_time, end_time)
            
            if source_metrics.empty:
                return {'error': 'No source metrics found'}
            
            # Prepare features
            if not features:
                features = [
                    'avg_articles_per_day',
                    'avg_article_length',
                    'avg_sentiment',
                    'sentiment_variance',
                    'publish_hour_entropy',
                    'weekend_ratio',
                    'keyword_diversity',
                    'success_rate'
                ]
            
            # Calculate behavioral features
            behavior_features = self._calculate_source_behavior_features(
                source_metrics
            )
            
            # Select and scale features
            feature_matrix = behavior_features[features].fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
            
            # Determine optimal clusters
            n_clusters = self._find_optimal_clusters(
                scaled_features, 
                min_k=2, 
                max_k=min(10, len(behavior_features) - 1)
            )
            
            # Perform clustering
            cluster_labels, metrics = self._kmeans_clustering(
                scaled_features, n_clusters
            )
            
            # Analyze source clusters
            behavior_features['cluster'] = cluster_labels
            cluster_profiles = self._analyze_source_clusters(
                behavior_features, features
            )
            
            # Store results
            result_id = self._store_clustering_results(
                clustering_type='source_behavior',
                method='kmeans',
                cluster_labels=cluster_labels,
                cluster_analysis=cluster_profiles,
                metrics=metrics,
                parameters={
                    'days_back': days_back,
                    'features': features,
                    'n_sources': len(behavior_features)
                }
            )
            
            return {
                'result_id': result_id,
                'n_sources': len(behavior_features),
                'n_clusters': n_clusters,
                'metrics': metrics,
                'cluster_profiles': cluster_profiles,
                'source_assignments': behavior_features[['cluster']].to_dict('index')
            }
            
        except Exception as e:
            self.logger.error(f"Source clustering failed: {e}")
            return {'error': str(e)}
    
    def cluster_temporal_patterns(self, time_window: str = '1H',
                                 days_back: int = 30) -> Dict[str, Any]:
        """Cluster temporal patterns in article publishing.
        
        Args:
            time_window: Time window for aggregation
            days_back: Number of days to analyze
            
        Returns:
            Temporal clustering results
        """
        try:
            self.logger.info("Starting temporal pattern clustering")
            
            # Get temporal data
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            temporal_data = self._get_temporal_patterns(
                start_time, end_time, time_window
            )
            
            if temporal_data.empty:
                return {'error': 'No temporal data found'}
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(temporal_data)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(temporal_features)
            
            # Perform clustering
            n_clusters = min(24, len(temporal_features) // 10)  # Reasonable default
            cluster_labels, metrics = self._kmeans_clustering(
                scaled_features, n_clusters
            )
            
            # Analyze temporal clusters
            temporal_data['cluster'] = cluster_labels
            cluster_patterns = self._analyze_temporal_clusters(
                temporal_data, time_window
            )
            
            # Identify recurring patterns
            recurring_patterns = self._identify_recurring_patterns(
                temporal_data, cluster_labels
            )
            
            # Store results
            result_id = self._store_clustering_results(
                clustering_type='temporal_patterns',
                method='kmeans',
                cluster_labels=cluster_labels,
                cluster_analysis=cluster_patterns,
                metrics=metrics,
                parameters={
                    'time_window': time_window,
                    'days_back': days_back,
                    'n_time_periods': len(temporal_features)
                }
            )
            
            return {
                'result_id': result_id,
                'n_time_periods': len(temporal_features),
                'n_clusters': n_clusters,
                'metrics': metrics,
                'cluster_patterns': cluster_patterns,
                'recurring_patterns': recurring_patterns,
                'time_window': time_window
            }
            
        except Exception as e:
            self.logger.error(f"Temporal clustering failed: {e}")
            return {'error': str(e)}
    
    def create_hierarchical_topic_tree(self, start_date: datetime = None,
                                     end_date: datetime = None,
                                     max_depth: int = 4) -> Dict[str, Any]:
        """Create hierarchical topic tree using agglomerative clustering.
        
        Args:
            start_date: Start date for articles
            end_date: End date for articles
            max_depth: Maximum tree depth
            
        Returns:
            Hierarchical topic structure
        """
        try:
            self.logger.info("Creating hierarchical topic tree")
            
            # Get articles
            articles_df = self._get_articles(start_date, end_date)
            
            if articles_df.empty:
                return {'error': 'No articles found'}
            
            # Generate features
            if self.sentence_model:
                features = self._generate_embeddings(articles_df)
            else:
                features = self._generate_tfidf_features(articles_df)
            
            # Limit size for hierarchical clustering
            if len(features) > 5000:
                # Sample articles
                sample_idx = np.random.choice(
                    len(features), 5000, replace=False
                )
                features = features[sample_idx]
                articles_df = articles_df.iloc[sample_idx]
            
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0,
                linkage='ward'
            )
            clustering.fit(features)
            
            # Build hierarchy
            hierarchy = self._build_hierarchy(
                clustering.children_,
                articles_df,
                max_depth
            )
            
            # Generate topic labels for each level
            topic_tree = self._generate_topic_tree(
                hierarchy, articles_df, features
            )
            
            # Store results
            result_id = self._store_clustering_results(
                clustering_type='hierarchical_topics',
                method='agglomerative',
                cluster_labels=np.zeros(len(articles_df)),  # Placeholder
                cluster_analysis=topic_tree,
                metrics={'max_depth': max_depth},
                parameters={
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None,
                    'n_articles': len(articles_df)
                }
            )
            
            return {
                'result_id': result_id,
                'n_articles': len(articles_df),
                'max_depth': max_depth,
                'topic_tree': topic_tree,
                'n_topics': self._count_tree_nodes(topic_tree)
            }
            
        except Exception as e:
            self.logger.error(f"Hierarchical clustering failed: {e}")
            return {'error': str(e)}
    
    def get_cluster_evolution(self, clustering_type: str = 'topic',
                            time_windows: List[Tuple[datetime, datetime]] = None,
                            method: str = 'kmeans') -> Dict[str, Any]:
        """Track cluster evolution over time.
        
        Args:
            clustering_type: Type of clustering to track
            time_windows: List of time windows to analyze
            method: Clustering method
            
        Returns:
            Cluster evolution analysis
        """
        try:
            self.logger.info(f"Analyzing cluster evolution for {clustering_type}")
            
            # Default time windows if not provided
            if not time_windows:
                end_date = datetime.utcnow()
                time_windows = [
                    (end_date - timedelta(days=30), end_date - timedelta(days=21)),
                    (end_date - timedelta(days=21), end_date - timedelta(days=14)),
                    (end_date - timedelta(days=14), end_date - timedelta(days=7)),
                    (end_date - timedelta(days=7), end_date)
                ]
            
            evolution_data = []
            
            for i, (start, end) in enumerate(time_windows):
                # Perform clustering for this window
                if clustering_type == 'topic':
                    result = self.cluster_articles_by_topic(
                        start_date=start,
                        end_date=end,
                        method=method
                    )
                elif clustering_type == 'source':
                    days = (end - start).days
                    result = self.cluster_by_source_behavior(days_back=days)
                else:
                    continue
                
                if 'error' not in result:
                    evolution_data.append({
                        'window_id': i,
                        'start_date': start.isoformat(),
                        'end_date': end.isoformat(),
                        'n_clusters': result['n_clusters'],
                        'cluster_summaries': result.get('cluster_summaries', {}),
                        'metrics': result['metrics']
                    })
            
            # Analyze evolution
            if len(evolution_data) > 1:
                evolution_analysis = self._analyze_cluster_evolution(
                    evolution_data
                )
            else:
                evolution_analysis = {'message': 'Insufficient data for evolution analysis'}
            
            return {
                'clustering_type': clustering_type,
                'n_time_windows': len(time_windows),
                'evolution_data': evolution_data,
                'evolution_analysis': evolution_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Cluster evolution analysis failed: {e}")
            return {'error': str(e)}
    
    def _get_and_validate_articles(self, start_date: datetime = None,
                                  end_date: datetime = None) -> Union[pd.DataFrame, Dict[str, str]]:
        """Get articles and validate for clustering."""
        articles_df = self._get_articles(start_date, end_date)
        
        if articles_df.empty:
            return {'error': 'No articles found for clustering'}
        
        self.logger.info(f"Found {len(articles_df)} articles for clustering")
        return articles_df
    
    def _generate_clustering_features(self, articles_df: pd.DataFrame, method: str) -> np.ndarray:
        """Generate appropriate features based on clustering method."""
        if self.sentence_model and method != 'kmeans':
            # Use sentence embeddings for better semantic clustering
            self.logger.info("Generating sentence embeddings for clustering")
            return self._generate_embeddings(articles_df)
        else:
            # Use TF-IDF for traditional clustering
            self.logger.info("Generating TF-IDF features for clustering")
            return self._generate_tfidf_features(articles_df)
    
    def _perform_clustering(self, features: np.ndarray, method: str,
                           n_clusters: int = None, min_cluster_size: int = 5) -> Dict[str, Any]:
        """Perform clustering using the specified method."""
        if method == 'kmeans':
            cluster_labels, metrics = self._kmeans_clustering(
                features, n_clusters or 10
            )
        elif method == 'dbscan':
            cluster_labels, metrics = self._dbscan_clustering(
                features, min_samples=min_cluster_size
            )
        elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
            cluster_labels, metrics = self._hdbscan_clustering(
                features, min_cluster_size=min_cluster_size
            )
        elif method == 'hierarchical':
            cluster_labels, metrics = self._hierarchical_clustering(
                features, n_clusters or 10
            )
        else:
            return {'error': f'Unsupported clustering method: {method}'}
        
        return {'labels': cluster_labels, 'metrics': metrics}
    
    def _process_clustering_results(self, articles_df: pd.DataFrame, features: np.ndarray,
                                  cluster_labels: np.ndarray, metrics: Dict[str, float],
                                  method: str, n_clusters: int = None,
                                  min_cluster_size: int = 5,
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> Dict[str, Any]:
        """Process and package clustering results."""
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(
            articles_df, cluster_labels, 'topic'
        )
        
        # Generate cluster summaries
        cluster_summaries = self._generate_cluster_summaries(
            articles_df, cluster_labels
        )
        
        # Store results
        result_id = self._store_clustering_results(
            clustering_type='topic',
            method=method,
            cluster_labels=cluster_labels,
            cluster_analysis=cluster_analysis,
            metrics=metrics,
            parameters={
                'n_clusters': n_clusters,
                'min_cluster_size': min_cluster_size,
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            }
        )
        
        # Prepare visualizations if dataset is manageable
        viz_data = self._prepare_visualization_if_feasible(
            features, cluster_labels, articles_df
        )
        
        return {
            'result_id': result_id,
            'n_articles': len(articles_df),
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': sum(cluster_labels == -1),
            'metrics': metrics,
            'cluster_summaries': cluster_summaries,
            'cluster_analysis': cluster_analysis,
            'visualization_data': viz_data
        }
    
    def _prepare_visualization_if_feasible(self, features: np.ndarray,
                                         cluster_labels: np.ndarray,
                                         articles_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Prepare visualization data if dataset size is manageable."""
        if len(features) <= 10000:  # Limit visualization to manageable size
            self.logger.info("Preparing visualization data")
            return self._prepare_visualization_data(features, cluster_labels, articles_df)
        else:
            self.logger.info("Dataset too large for visualization - skipping")
            return None
    
    # Helper methods
    
    def _get_articles(self, start_date: datetime = None, 
                     end_date: datetime = None) -> pd.DataFrame:
        """Get articles from database."""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        query = """
            SELECT id, url, title, content, published_date, source, 
                   metadata, tags
            FROM articles
            WHERE published_date BETWEEN %s AND %s
            ORDER BY published_date DESC
        """
        
        with self.db_adapter.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=[start_date, end_date]
            )
        
        # Parse JSON fields
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return df
    
    def _generate_embeddings(self, articles_df: pd.DataFrame) -> np.ndarray:
        """Generate sentence embeddings for articles."""
        # Combine title and content
        texts = (articles_df['title'].fillna('') + ' ' + 
                articles_df['content'].fillna('')).tolist()
        
        # Truncate very long texts
        texts = [text[:10000] for text in texts]
        
        # Generate embeddings in batches
        embeddings = self.sentence_model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True
        )
        
        return embeddings
    
    def _generate_tfidf_features(self, articles_df: pd.DataFrame,
                               max_features: int = 1000) -> np.ndarray:
        """Generate TF-IDF features for articles."""
        # Combine title and content
        texts = (articles_df['title'].fillna('') + ' ' + 
                articles_df['content'].fillna('')).tolist()
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        features = vectorizer.fit_transform(texts).toarray()
        
        # Store vocabulary for later analysis
        self.tfidf_vocabulary = vectorizer.get_feature_names_out()
        
        return features
    
    def _kmeans_clustering(self, features: np.ndarray, 
                          n_clusters: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform K-means clustering."""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(features)
        
        # Calculate metrics
        metrics = {
            'inertia': float(kmeans.inertia_),
            'silhouette_score': float(silhouette_score(features, cluster_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(features, cluster_labels))
        }
        
        return cluster_labels, metrics
    
    def _dbscan_clustering(self, features: np.ndarray,
                          eps: float = None,
                          min_samples: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform DBSCAN clustering."""
        if eps is None:
            # Estimate eps using k-nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(features)
            distances, indices = neighbors_fit.kneighbors(features)
            
            # Sort and find elbow
            distances = np.sort(distances[:, -1], axis=0)
            eps = np.percentile(distances, 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': min_samples
        }
        
        if n_clusters > 1:
            # Only calculate if we have valid clusters
            valid_mask = cluster_labels != -1
            if sum(valid_mask) > 0:
                metrics['silhouette_score'] = float(
                    silhouette_score(features[valid_mask], cluster_labels[valid_mask])
                )
        
        return cluster_labels, metrics
    
    def _hdbscan_clustering(self, features: np.ndarray,
                           min_cluster_size: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform HDBSCAN clustering."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(features)
        
        # Calculate metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'min_cluster_size': min_cluster_size
        }
        
        if n_clusters > 1:
            valid_mask = cluster_labels != -1
            if sum(valid_mask) > 0:
                metrics['silhouette_score'] = float(
                    silhouette_score(features[valid_mask], cluster_labels[valid_mask])
                )
        
        return cluster_labels, metrics
    
    def _hierarchical_clustering(self, features: np.ndarray,
                               n_clusters: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform hierarchical clustering."""
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(features)
        
        # Calculate metrics
        metrics = {
            'silhouette_score': float(silhouette_score(features, cluster_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(features, cluster_labels))
        }
        
        return cluster_labels, metrics
    
    def _find_optimal_clusters(self, features: np.ndarray,
                             min_k: int = 2,
                             max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        scores = []
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            scores.append(kmeans.inertia_)
        
        # Find elbow point
        deltas = np.diff(scores)
        deltas2 = np.diff(deltas)
        
        if len(deltas2) > 0:
            elbow = np.argmax(deltas2) + min_k + 1
        else:
            elbow = min_k + 1
        
        return min(max(elbow, min_k), max_k)
    
    def _analyze_clusters(self, articles_df: pd.DataFrame,
                         cluster_labels: np.ndarray,
                         cluster_type: str) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        articles_df['cluster'] = cluster_labels
        cluster_analysis = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_articles = articles_df[articles_df['cluster'] == cluster_id]
            
            analysis = {
                'size': len(cluster_articles),
                'sources': cluster_articles['source'].value_counts().head(5).to_dict(),
                'date_range': {
                    'min': cluster_articles['published_date'].min().isoformat(),
                    'max': cluster_articles['published_date'].max().isoformat()
                }
            }
            
            # Add metadata analysis if available
            if 'metadata' in cluster_articles.columns:
                sentiments = []
                entities = []
                
                for _, row in cluster_articles.iterrows():
                    if isinstance(row['metadata'], dict):
                        if 'ai_analysis' in row['metadata']:
                            ai_data = row['metadata']['ai_analysis']
                            if 'sentiment' in ai_data:
                                sentiments.append(ai_data['sentiment']['compound'])
                            if 'entities' in ai_data:
                                entities.extend(ai_data['entities'])
                
                if sentiments:
                    analysis['avg_sentiment'] = float(np.mean(sentiments))
                
                if entities:
                    # Top entities
                    entity_counts = defaultdict(int)
                    for entity in entities:
                        entity_counts[entity['text']] += 1
                    
                    analysis['top_entities'] = dict(
                        sorted(entity_counts.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:5]
                    )
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def _generate_cluster_summaries(self, articles_df: pd.DataFrame,
                                  cluster_labels: np.ndarray) -> Dict[str, str]:
        """Generate text summaries for each cluster."""
        summaries = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
            
            cluster_articles = articles_df[cluster_labels == cluster_id]
            
            # Get top keywords for cluster
            cluster_text = ' '.join(
                cluster_articles['title'].fillna('').tolist() +
                cluster_articles['content'].fillna('').head(10).tolist()
            )
            
            # Extract keywords using TF-IDF
            try:
                vectorizer = TfidfVectorizer(
                    max_features=10,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                vectorizer.fit([cluster_text])
                keywords = vectorizer.get_feature_names_out()
                
                summary = f"Cluster {cluster_id}: {', '.join(keywords[:5])}"
                summaries[f'cluster_{cluster_id}'] = summary
            except:
                summaries[f'cluster_{cluster_id}'] = f"Cluster {cluster_id}"
        
        return summaries
    
    def _get_source_metrics(self, start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Get source metrics from warehouse."""
        query = """
            SELECT 
                source,
                AVG(article_count) as avg_articles,
                AVG(avg_sentiment) as avg_sentiment,
                AVG(success_rate) as avg_success_rate,
                COUNT(DISTINCT DATE(time)) as active_days
            FROM analytics.source_metrics
            WHERE time BETWEEN %s AND %s
            GROUP BY source
        """
        
        with self.warehouse.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=[start_time, end_time]
            )
        
        return df
    
    def _calculate_source_behavior_features(self, 
                                          source_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral features for sources."""
        features = source_metrics.copy()
        
        # Get additional metrics
        with self.warehouse.get_connection() as conn:
            # Publishing patterns
            query = """
                SELECT 
                    source,
                    EXTRACT(HOUR FROM time) as hour,
                    SUM(article_count) as articles
                FROM analytics.source_metrics
                WHERE time >= NOW() - INTERVAL '30 days'
                GROUP BY source, hour
            """
            hourly_df = pd.read_sql_query(query, conn)
            
            # Calculate entropy of publishing hours
            for source in features['source'].unique():
                source_hourly = hourly_df[hourly_df['source'] == source]
                if not source_hourly.empty:
                    # Calculate entropy
                    probs = source_hourly['articles'] / source_hourly['articles'].sum()
                    entropy = -sum(probs * np.log2(probs + 1e-10))
                    features.loc[features['source'] == source, 'publish_hour_entropy'] = entropy
        
        # Additional features
        features['avg_articles_per_day'] = features['avg_articles']
        features['sentiment_variance'] = 0.1  # Placeholder
        features['weekend_ratio'] = 0.3  # Placeholder
        features['keyword_diversity'] = 50  # Placeholder
        features['avg_article_length'] = 500  # Placeholder
        features['success_rate'] = features['avg_success_rate']
        
        return features
    
    def _analyze_source_clusters(self, behavior_features: pd.DataFrame,
                               feature_names: List[str]) -> Dict[str, Any]:
        """Analyze source cluster profiles."""
        cluster_profiles = {}
        
        for cluster_id in behavior_features['cluster'].unique():
            cluster_sources = behavior_features[
                behavior_features['cluster'] == cluster_id
            ]
            
            profile = {
                'n_sources': len(cluster_sources),
                'sources': cluster_sources['source'].tolist(),
                'feature_means': {}
            }
            
            # Calculate feature statistics
            for feature in feature_names:
                if feature in cluster_sources.columns:
                    profile['feature_means'][feature] = float(
                        cluster_sources[feature].mean()
                    )
            
            # Characterize cluster
            if profile['feature_means'].get('avg_articles_per_day', 0) > 50:
                profile['label'] = 'High Volume Publishers'
            elif profile['feature_means'].get('avg_sentiment', 0) > 0.1:
                profile['label'] = 'Positive Content Sources'
            elif profile['feature_means'].get('weekend_ratio', 0) > 0.4:
                profile['label'] = 'Weekend Active Sources'
            else:
                profile['label'] = f'Source Group {cluster_id}'
            
            cluster_profiles[f'cluster_{cluster_id}'] = profile
        
        return cluster_profiles
    
    def _get_temporal_patterns(self, start_time: datetime,
                             end_time: datetime,
                             time_window: str) -> pd.DataFrame:
        """Get temporal patterns from warehouse."""
        query = f"""
            SELECT 
                time_bucket('{time_window}', time) as time_bucket,
                SUM(article_count) as total_articles,
                AVG(avg_sentiment) as avg_sentiment,
                COUNT(DISTINCT source) as unique_sources
            FROM analytics.article_metrics
            WHERE time BETWEEN %s AND %s
            GROUP BY time_bucket
            ORDER BY time_bucket
        """
        
        with self.warehouse.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=[start_time, end_time]
            )
        
        return df
    
    def _extract_temporal_features(self, temporal_data: pd.DataFrame) -> np.ndarray:
        """Extract features from temporal data."""
        features = []
        
        for _, row in temporal_data.iterrows():
            time_features = [
                row['total_articles'],
                row['avg_sentiment'] if pd.notna(row['avg_sentiment']) else 0,
                row['unique_sources'],
                row['time_bucket'].hour,
                row['time_bucket'].dayofweek,
                row['time_bucket'].day,
                1 if row['time_bucket'].dayofweek in [5, 6] else 0  # Weekend
            ]
            features.append(time_features)
        
        return np.array(features)
    
    def _analyze_temporal_clusters(self, temporal_data: pd.DataFrame,
                                 time_window: str) -> Dict[str, Any]:
        """Analyze temporal cluster patterns."""
        cluster_patterns = {}
        
        for cluster_id in temporal_data['cluster'].unique():
            cluster_periods = temporal_data[temporal_data['cluster'] == cluster_id]
            
            pattern = {
                'n_periods': len(cluster_periods),
                'avg_articles': float(cluster_periods['total_articles'].mean()),
                'avg_sources': float(cluster_periods['unique_sources'].mean()),
                'time_characteristics': {}
            }
            
            # Analyze time characteristics
            hours = pd.to_datetime(cluster_periods['time_bucket']).dt.hour
            days = pd.to_datetime(cluster_periods['time_bucket']).dt.dayofweek
            
            pattern['time_characteristics'] = {
                'peak_hour': int(hours.mode()[0]) if not hours.empty else 0,
                'peak_day': int(days.mode()[0]) if not days.empty else 0,
                'is_weekend_pattern': float(days.isin([5, 6]).mean()) > 0.5
            }
            
            # Label pattern
            if pattern['avg_articles'] > temporal_data['total_articles'].mean() * 1.5:
                pattern['label'] = 'High Activity Period'
            elif pattern['time_characteristics']['is_weekend_pattern']:
                pattern['label'] = 'Weekend Pattern'
            elif pattern['time_characteristics']['peak_hour'] < 6:
                pattern['label'] = 'Early Morning Pattern'
            elif pattern['time_characteristics']['peak_hour'] > 20:
                pattern['label'] = 'Late Night Pattern'
            else:
                pattern['label'] = f'Pattern {cluster_id}'
            
            cluster_patterns[f'cluster_{cluster_id}'] = pattern
        
        return cluster_patterns
    
    def _identify_recurring_patterns(self, temporal_data: pd.DataFrame,
                                   cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Identify recurring temporal patterns."""
        recurring_patterns = []
        
        # Group by cluster and analyze recurrence
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
            
            cluster_times = temporal_data[cluster_labels == cluster_id]['time_bucket']
            
            if len(cluster_times) < 2:
                continue
            
            # Calculate time differences
            time_diffs = pd.Series(cluster_times).diff().dropna()
            
            # Look for regular intervals
            if not time_diffs.empty:
                common_interval = time_diffs.mode()[0]
                interval_count = (time_diffs == common_interval).sum()
                
                if interval_count >= len(time_diffs) * 0.3:  # 30% threshold
                    recurring_patterns.append({
                        'cluster_id': int(cluster_id),
                        'interval': str(common_interval),
                        'frequency': int(interval_count),
                        'confidence': float(interval_count / len(time_diffs))
                    })
        
        return recurring_patterns
    
    def _build_hierarchy(self, children: np.ndarray,
                        articles_df: pd.DataFrame,
                        max_depth: int) -> Dict[str, Any]:
        """Build hierarchical structure from clustering."""
        n_samples = len(articles_df)
        n_nodes = len(children) + n_samples
        
        # Build tree structure
        tree = {}
        for i, (left, right) in enumerate(children):
            node_id = n_samples + i
            tree[node_id] = {
                'left': int(left),
                'right': int(right),
                'size': 0  # Will be calculated
            }
        
        # Calculate sizes
        def get_size(node_id):
            if node_id < n_samples:
                return 1
            if tree[node_id]['size'] > 0:
                return tree[node_id]['size']
            
            size = get_size(tree[node_id]['left']) + get_size(tree[node_id]['right'])
            tree[node_id]['size'] = size
            return size
        
        for node_id in tree:
            get_size(node_id)
        
        return tree
    
    def _generate_topic_tree(self, hierarchy: Dict[str, Any],
                           articles_df: pd.DataFrame,
                           features: np.ndarray) -> Dict[str, Any]:
        """Generate topic labels for hierarchical tree."""
        # Simplified version - just return cluster counts
        topic_tree = {
            'total_articles': len(articles_df),
            'n_nodes': len(hierarchy),
            'structure': 'hierarchical'
        }
        
        return topic_tree
    
    def _count_tree_nodes(self, tree: Dict[str, Any]) -> int:
        """Count nodes in tree structure."""
        if isinstance(tree, dict):
            return len(tree)
        return 1
    
    def _prepare_visualization_data(self, features: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  articles_df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for visualization."""
        # Reduce dimensions for visualization
        if features.shape[1] > 50:
            # Use TruncatedSVD for sparse matrices
            reducer = TruncatedSVD(n_components=50, random_state=42)
            features_reduced = reducer.fit_transform(features)
        else:
            features_reduced = features
        
        # Further reduce to 2D
        if UMAP_AVAILABLE and len(features) < 5000:
            reducer_2d = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer_2d.fit_transform(features_reduced)
        else:
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(features_reduced)
        
        # Prepare visualization data
        viz_data = {
            'points': [
                {
                    'x': float(coords_2d[i, 0]),
                    'y': float(coords_2d[i, 1]),
                    'cluster': int(cluster_labels[i]),
                    'title': articles_df.iloc[i]['title'][:100],
                    'source': articles_df.iloc[i]['source']
                }
                for i in range(min(len(coords_2d), 1000))  # Limit points
            ]
        }
        
        return viz_data
    
    def _analyze_cluster_evolution(self, evolution_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how clusters evolve over time."""
        analysis = {
            'cluster_stability': [],
            'cluster_growth': []
        }
        
        # Compare consecutive windows
        for i in range(1, len(evolution_data)):
            prev = evolution_data[i-1]
            curr = evolution_data[i]
            
            # Cluster count change
            cluster_change = curr['n_clusters'] - prev['n_clusters']
            
            analysis['cluster_stability'].append({
                'window_transition': f"{i-1}_to_{i}",
                'cluster_count_change': cluster_change,
                'stability_score': 1.0 - abs(cluster_change) / max(prev['n_clusters'], 1)
            })
        
        return analysis
    
    def _store_clustering_results(self, clustering_type: str,
                                method: str,
                                cluster_labels: np.ndarray,
                                cluster_analysis: Dict[str, Any],
                                metrics: Dict[str, float],
                                parameters: Dict[str, Any]) -> str:
        """Store clustering results in warehouse."""
        result_id = f"{clustering_type}_{method}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to file
        result_file = self.results_dir / f"{result_id}.json"
        
        result_data = {
            'result_id': result_id,
            'clustering_type': clustering_type,
            'method': method,
            'timestamp': datetime.utcnow().isoformat(),
            'parameters': parameters,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis,
            'cluster_labels': cluster_labels.tolist() if isinstance(cluster_labels, np.ndarray) else cluster_labels
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Store summary in database
        try:
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analytics.ml_results 
                    (result_id, model_type, parameters, metrics, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, [
                    result_id,
                    f"{clustering_type}_{method}",
                    json.dumps(parameters),
                    json.dumps(metrics),
                    datetime.utcnow()
                ])
                conn.commit()
        except Exception as e:
            self.logger.warning(f"Could not store result in database: {e}")
        
        return result_id