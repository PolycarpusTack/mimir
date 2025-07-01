"""Recommendation Engine for Mimir Analytics.

This module provides article recommendation capabilities using collaborative filtering,
content-based filtering, and hybrid approaches with feedback loop integration.
"""

import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

try:
    from implicit import als, bpr, lmf
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .data_warehouse import AnalyticsDataWarehouse
from .feature_engineering import FeatureEngineering
from .ml_infrastructure import FeatureStore, ModelRegistry
from db_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Main recommendation engine for article recommendations."""
    
    def __init__(self, db_adapter: DatabaseAdapter,
                 analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the recommendation engine.
        
        Args:
            db_adapter: Database adapter for article data
            analytics_warehouse: Analytics warehouse
        """
        self.db_adapter = db_adapter
        self.warehouse = analytics_warehouse
        self.feature_eng = FeatureEngineering()
        self.model_registry = ModelRegistry()
        
        # Initialize models
        self.collaborative_model = None
        self.content_model = None
        self.hybrid_weights = {'collaborative': 0.5, 'content': 0.5}
        
        # Initialize sentence transformer for content similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Initialized sentence transformer for content recommendations")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Cache for recommendations
        self.recommendation_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Storage paths
        self.models_dir = Path("analytics/recommendation_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def train_collaborative_filter(self, interaction_type: str = 'view',
                                 days_back: int = 90,
                                 model_type: str = 'als',
                                 factors: int = 50) -> Dict[str, Any]:
        """Train collaborative filtering model on user interactions.
        
        Args:
            interaction_type: Type of interaction to use (view, share, save)
            days_back: Number of days of history to use
            model_type: Model type (als, bpr, lmf)
            factors: Number of latent factors
            
        Returns:
            Training results
        """
        try:
            self.logger.info(f"Training collaborative filter with {model_type}")
            
            # Get user interactions
            interactions_df = self._get_user_interactions(
                interaction_type, days_back
            )
            
            if interactions_df.empty:
                return {'error': 'No interactions found for training'}
            
            # Create user-item matrix
            user_item_matrix, user_mapping, item_mapping = self._create_interaction_matrix(
                interactions_df
            )
            
            self.logger.info(f"Created matrix with {len(user_mapping)} users and {len(item_mapping)} items")
            
            # Train model based on type
            if not IMPLICIT_AVAILABLE:
                return {'error': 'Implicit library not available for collaborative filtering'}
            
            if model_type == 'als':
                model = als.AlternatingLeastSquares(
                    factors=factors,
                    regularization=0.1,
                    iterations=50,
                    calculate_training_loss=True
                )
            elif model_type == 'bpr':
                model = bpr.BayesianPersonalizedRanking(
                    factors=factors,
                    learning_rate=0.01,
                    regularization=0.01,
                    iterations=100
                )
            elif model_type == 'lmf':
                model = lmf.LogisticMatrixFactorization(
                    factors=factors,
                    learning_rate=1.0,
                    regularization=0.01,
                    iterations=50
                )
            else:
                return {'error': f'Unsupported model type: {model_type}'}
            
            # Train model
            model.fit(user_item_matrix)
            
            # Save model and mappings
            model_data = {
                'model': model,
                'user_mapping': user_mapping,
                'item_mapping': item_mapping,
                'reverse_user_mapping': {v: k for k, v in user_mapping.items()},
                'reverse_item_mapping': {v: k for k, v in item_mapping.items()},
                'metadata': {
                    'model_type': model_type,
                    'factors': factors,
                    'n_users': len(user_mapping),
                    'n_items': len(item_mapping),
                    'trained_at': datetime.utcnow().isoformat()
                }
            }
            
            # Save to disk
            model_path = self.models_dir / f"collaborative_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Register model
            model_id = self.model_registry.register_model(
                model=model_data,
                model_name=f"collaborative_filter_{interaction_type}",
                model_type='collaborative_filtering',
                metadata=model_data['metadata'],
                metrics={'n_interactions': len(interactions_df)}
            )
            
            self.collaborative_model = model_data
            
            return {
                'model_id': model_id,
                'model_type': model_type,
                'n_users': len(user_mapping),
                'n_items': len(item_mapping),
                'n_interactions': len(interactions_df),
                'factors': factors
            }
            
        except Exception as e:
            self.logger.error(f"Collaborative filter training failed: {e}")
            return {'error': str(e)}
    
    def train_content_model(self, days_back: int = 90,
                          feature_type: str = 'embeddings',
                          max_features: int = 1000) -> Dict[str, Any]:
        """Train content-based recommendation model.
        
        Args:
            days_back: Number of days of articles to use
            feature_type: Type of features (tfidf, embeddings)
            max_features: Maximum features for TF-IDF
            
        Returns:
            Training results
        """
        try:
            self.logger.info(f"Training content model with {feature_type}")
            
            # Get articles
            articles_df = self._get_recent_articles(days_back)
            
            if articles_df.empty:
                return {'error': 'No articles found for training'}
            
            # Generate content features
            if feature_type == 'embeddings' and self.sentence_model:
                features = self._generate_article_embeddings(articles_df)
                feature_names = [f'embedding_{i}' for i in range(features.shape[1])]
            else:
                features, feature_names = self._generate_tfidf_features(
                    articles_df, max_features
                )
            
            self.logger.info(f"Generated {features.shape[1]} features for {len(articles_df)} articles")
            
            # Build content model
            content_model = {
                'features': features,
                'feature_names': feature_names,
                'article_ids': articles_df['id'].tolist(),
                'article_mapping': {row['id']: idx for idx, row in articles_df.iterrows()},
                'metadata': {
                    'feature_type': feature_type,
                    'n_articles': len(articles_df),
                    'n_features': features.shape[1],
                    'trained_at': datetime.utcnow().isoformat()
                }
            }
            
            # Calculate and store similarity matrix for faster recommendations
            if len(articles_df) < 10000:  # Only for manageable sizes
                similarity_matrix = cosine_similarity(features)
                content_model['similarity_matrix'] = similarity_matrix
            
            # Save model
            model_path = self.models_dir / f"content_{feature_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(content_model, f)
            
            # Register model
            model_id = self.model_registry.register_model(
                model=content_model,
                model_name='content_recommender',
                model_type='content_based',
                metadata=content_model['metadata']
            )
            
            self.content_model = content_model
            
            return {
                'model_id': model_id,
                'feature_type': feature_type,
                'n_articles': len(articles_df),
                'n_features': features.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Content model training failed: {e}")
            return {'error': str(e)}
    
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 10,
                               method: str = 'hybrid',
                               filter_viewed: bool = True) -> List[Dict[str, Any]]:
        """Get article recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            method: Recommendation method (collaborative, content, hybrid)
            filter_viewed: Filter out already viewed articles
            
        Returns:
            List of recommended articles
        """
        try:
            # Check cache
            cache_key = f"{user_id}_{method}_{n_recommendations}"
            if cache_key in self.recommendation_cache:
                cached_time, cached_recs = self.recommendation_cache[cache_key]
                if (datetime.utcnow() - cached_time).seconds < self.cache_ttl:
                    return cached_recs
            
            recommendations = []
            
            if method in ['collaborative', 'hybrid']:
                collab_recs = self._get_collaborative_recommendations(
                    user_id, n_recommendations * 2
                )
                recommendations.extend(collab_recs)
            
            if method in ['content', 'hybrid']:
                # Get user's recent interactions for content-based
                recent_articles = self._get_user_recent_articles(user_id, limit=5)
                if recent_articles:
                    content_recs = self._get_content_recommendations(
                        recent_articles, n_recommendations * 2
                    )
                    recommendations.extend(content_recs)
            
            # Combine and rank for hybrid
            if method == 'hybrid':
                recommendations = self._combine_recommendations(
                    recommendations, self.hybrid_weights
                )
            
            # Filter viewed if requested
            if filter_viewed:
                viewed_articles = self._get_user_viewed_articles(user_id)
                recommendations = [
                    rec for rec in recommendations 
                    if rec['article_id'] not in viewed_articles
                ]
            
            # Limit to requested number
            recommendations = recommendations[:n_recommendations]
            
            # Enrich with article details
            recommendations = self._enrich_recommendations(recommendations)
            
            # Cache results
            self.recommendation_cache[cache_key] = (
                datetime.utcnow(), recommendations
            )
            
            # Log recommendation event
            self._log_recommendations(user_id, recommendations, method)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations for user {user_id}: {e}")
            return []
    
    def get_similar_articles(self, article_id: str, n_similar: int = 10,
                           method: str = 'content') -> List[Dict[str, Any]]:
        """Get articles similar to a given article.
        
        Args:
            article_id: Article ID
            n_similar: Number of similar articles
            method: Similarity method
            
        Returns:
            List of similar articles
        """
        try:
            if method == 'content' and self.content_model:
                # Get article index
                if article_id not in self.content_model['article_mapping']:
                    # Article not in model, compute on the fly
                    article = self._get_article_by_id(article_id)
                    if not article:
                        return []
                    
                    return self._get_content_recommendations(
                        [article_id], n_similar
                    )
                
                article_idx = self.content_model['article_mapping'][article_id]
                
                # Use precomputed similarity if available
                if 'similarity_matrix' in self.content_model:
                    similarities = self.content_model['similarity_matrix'][article_idx]
                    
                    # Get top similar articles
                    similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
                    
                    recommendations = []
                    for idx in similar_indices:
                        similar_article_id = self.content_model['article_ids'][idx]
                        recommendations.append({
                            'article_id': similar_article_id,
                            'score': float(similarities[idx]),
                            'method': 'content_similarity'
                        })
                else:
                    # Compute similarities on the fly
                    article_features = self.content_model['features'][article_idx]
                    similarities = cosine_similarity(
                        [article_features],
                        self.content_model['features']
                    )[0]
                    
                    similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
                    
                    recommendations = []
                    for idx in similar_indices:
                        similar_article_id = self.content_model['article_ids'][idx]
                        recommendations.append({
                            'article_id': similar_article_id,
                            'score': float(similarities[idx]),
                            'method': 'content_similarity'
                        })
                
                # Enrich recommendations
                recommendations = self._enrich_recommendations(recommendations)
                
                return recommendations
            
            else:
                # Fallback to metadata-based similarity
                return self._get_metadata_similar_articles(article_id, n_similar)
                
        except Exception as e:
            self.logger.error(f"Failed to get similar articles for {article_id}: {e}")
            return []
    
    def update_user_feedback(self, user_id: str, article_id: str,
                           feedback_type: str, feedback_value: float = 1.0) -> bool:
        """Update user feedback for recommendation improvement.
        
        Args:
            user_id: User ID
            article_id: Article ID
            feedback_type: Type of feedback (click, share, save, dismiss, etc.)
            feedback_value: Feedback value/weight
            
        Returns:
            Success status
        """
        try:
            # Store feedback in analytics warehouse
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert into user activity
                cursor.execute("""
                    INSERT INTO analytics.user_activity 
                    (time, user_id, action_type, item_id, item_type, value, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, [
                    datetime.utcnow(),
                    user_id,
                    f"recommendation_{feedback_type}",
                    article_id,
                    'article',
                    feedback_value,
                    json.dumps({
                        'recommendation_method': self.recommendation_cache.get(
                            f"{user_id}_last_method", 'unknown'
                        )
                    })
                ])
                
                conn.commit()
            
            # Update recommendation weights based on feedback
            self._update_recommendation_weights(feedback_type, feedback_value)
            
            # Clear user's recommendation cache
            self._clear_user_cache(user_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update feedback: {e}")
            return False
    
    def get_trending_recommendations(self, time_window: str = '24 hours',
                                   n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get trending article recommendations based on recent activity.
        
        Args:
            time_window: Time window for trending calculation
            n_recommendations: Number of recommendations
            
        Returns:
            List of trending articles
        """
        try:
            query = f"""
                SELECT 
                    am.article_id,
                    am.source,
                    am.article_count,
                    am.avg_sentiment,
                    COUNT(DISTINCT ua.user_id) as unique_viewers,
                    SUM(CASE WHEN ua.action_type = 'share' THEN 1 ELSE 0 END) as shares,
                    SUM(CASE WHEN ua.action_type = 'save' THEN 1 ELSE 0 END) as saves
                FROM analytics.article_metrics am
                LEFT JOIN analytics.user_activity ua 
                    ON am.article_id = ua.item_id 
                    AND ua.item_type = 'article'
                WHERE am.time >= NOW() - INTERVAL '{time_window}'
                GROUP BY am.article_id, am.source, am.article_count, am.avg_sentiment
                ORDER BY 
                    unique_viewers DESC,
                    shares DESC,
                    saves DESC
                LIMIT %s
            """
            
            with self.warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[n_recommendations * 2])
            
            if df.empty:
                return []
            
            # Calculate trending score
            df['trending_score'] = (
                df['unique_viewers'] * 1.0 +
                df['shares'] * 3.0 +
                df['saves'] * 2.0
            )
            
            # Sort by trending score
            df = df.nlargest(n_recommendations, 'trending_score')
            
            # Convert to recommendations
            recommendations = []
            for _, row in df.iterrows():
                recommendations.append({
                    'article_id': row['article_id'],
                    'score': float(row['trending_score']),
                    'method': 'trending',
                    'metrics': {
                        'viewers': int(row['unique_viewers']),
                        'shares': int(row['shares']),
                        'saves': int(row['saves'])
                    }
                })
            
            # Enrich with article details
            recommendations = self._enrich_recommendations(recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get trending recommendations: {e}")
            return []
    
    def get_personalized_trending(self, user_id: str, 
                                n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get personalized trending recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            
        Returns:
            Personalized trending articles
        """
        try:
            # Get user preferences
            user_preferences = self._get_user_preferences(user_id)
            
            # Get trending articles
            trending = self.get_trending_recommendations(
                time_window='24 hours',
                n_recommendations=n_recommendations * 3
            )
            
            if not trending or not user_preferences:
                return trending[:n_recommendations]
            
            # Score trending based on user preferences
            for article in trending:
                preference_score = 0.0
                
                # Check source preference
                if article.get('source') in user_preferences.get('preferred_sources', []):
                    preference_score += 0.3
                
                # Check keyword matches
                article_keywords = set(article.get('keywords', []))
                user_keywords = set(user_preferences.get('preferred_keywords', []))
                if article_keywords & user_keywords:
                    preference_score += 0.5
                
                # Adjust score
                article['score'] = article['score'] * (1 + preference_score)
            
            # Re-sort by adjusted score
            trending.sort(key=lambda x: x['score'], reverse=True)
            
            return trending[:n_recommendations]
            
        except Exception as e:
            self.logger.error(f"Failed to get personalized trending: {e}")
            return []
    
    # Helper methods
    
    def _get_user_interactions(self, interaction_type: str,
                             days_back: int) -> pd.DataFrame:
        """Get user interaction data."""
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        query = """
            SELECT 
                user_id,
                item_id as article_id,
                COUNT(*) as interaction_count,
                MAX(time) as last_interaction
            FROM analytics.user_activity
            WHERE item_type = 'article'
                AND action_type = %s
                AND time >= %s
            GROUP BY user_id, item_id
        """
        
        with self.warehouse.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn, params=[interaction_type, start_date]
            )
        
        return df
    
    def _create_interaction_matrix(self, interactions_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """Create sparse user-item interaction matrix."""
        # Create mappings
        users = interactions_df['user_id'].unique()
        items = interactions_df['article_id'].unique()
        
        user_mapping = {user: idx for idx, user in enumerate(users)}
        item_mapping = {item: idx for idx, item in enumerate(items)}
        
        # Create sparse matrix
        row_indices = [user_mapping[user] for user in interactions_df['user_id']]
        col_indices = [item_mapping[item] for item in interactions_df['article_id']]
        values = interactions_df['interaction_count'].values
        
        matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(users), len(items))
        )
        
        return matrix, user_mapping, item_mapping
    
    def _get_recent_articles(self, days_back: int) -> pd.DataFrame:
        """Get recent articles for content model."""
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        query = """
            SELECT 
                a.id, a.url, a.title, a.content, a.published_date,
                a.source, a.metadata
            FROM articles a
            WHERE a.published_date >= %s
            ORDER BY a.published_date DESC
        """
        
        with self.db_adapter.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[start_date])
        
        # Parse metadata
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
        
        return df
    
    def _generate_article_embeddings(self, articles_df: pd.DataFrame) -> np.ndarray:
        """Generate embeddings for articles."""
        # Combine title and content
        texts = (articles_df['title'].fillna('') + ' ' + 
                articles_df['content'].fillna('')).tolist()
        
        # Truncate long texts
        texts = [text[:5000] for text in texts]
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True
        )
        
        return embeddings
    
    def _generate_tfidf_features(self, articles_df: pd.DataFrame,
                               max_features: int) -> Tuple[np.ndarray, List[str]]:
        """Generate TF-IDF features for articles."""
        # Combine title and content
        texts = (articles_df['title'].fillna('') + ' ' + 
                articles_df['content'].fillna('')).tolist()
        
        # Create vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        features = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        return features, feature_names
    
    def _get_collaborative_recommendations(self, user_id: str,
                                         n_recommendations: int) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations."""
        if not self.collaborative_model or not IMPLICIT_AVAILABLE:
            return []
        
        # Check if user exists in model
        if user_id not in self.collaborative_model['user_mapping']:
            return []
        
        user_idx = self.collaborative_model['user_mapping'][user_id]
        model = self.collaborative_model['model']
        
        # Get recommendations
        recommendations, scores = model.recommend(
            user_idx,
            self.collaborative_model.get('user_item_matrix'),
            N=n_recommendations,
            filter_already_liked_items=True
        )
        
        # Convert to article IDs
        reverse_mapping = self.collaborative_model['reverse_item_mapping']
        
        rec_list = []
        for item_idx, score in zip(recommendations, scores):
            article_id = reverse_mapping.get(item_idx)
            if article_id:
                rec_list.append({
                    'article_id': article_id,
                    'score': float(score),
                    'method': 'collaborative'
                })
        
        return rec_list
    
    def _get_content_recommendations(self, article_ids: List[str],
                                   n_recommendations: int) -> List[Dict[str, Any]]:
        """Get content-based recommendations."""
        if not self.content_model:
            return []
        
        recommendations = []
        seen_articles = set(article_ids)
        
        for article_id in article_ids:
            similar = self.get_similar_articles(
                article_id,
                n_similar=n_recommendations // len(article_ids) + 5
            )
            
            for sim_article in similar:
                if sim_article['article_id'] not in seen_articles:
                    recommendations.append({
                        'article_id': sim_article['article_id'],
                        'score': sim_article['score'],
                        'method': 'content'
                    })
                    seen_articles.add(sim_article['article_id'])
        
        # Sort by score and limit
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def _combine_recommendations(self, recommendations: List[Dict[str, Any]],
                               weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Combine recommendations from different methods."""
        # Group by article ID
        article_scores = defaultdict(lambda: {'scores': {}, 'final_score': 0.0})
        
        for rec in recommendations:
            article_id = rec['article_id']
            method = rec['method']
            score = rec['score']
            
            article_scores[article_id]['scores'][method] = score
        
        # Calculate weighted scores
        for article_id, data in article_scores.items():
            final_score = 0.0
            total_weight = 0.0
            
            for method, score in data['scores'].items():
                weight = weights.get(method, 1.0)
                final_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                data['final_score'] = final_score / total_weight
        
        # Convert to list and sort
        combined = [
            {
                'article_id': article_id,
                'score': data['final_score'],
                'method': 'hybrid',
                'component_scores': data['scores']
            }
            for article_id, data in article_scores.items()
        ]
        
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        return combined
    
    def _get_user_recent_articles(self, user_id: str, limit: int = 5) -> List[str]:
        """Get user's recently viewed articles."""
        query = """
            SELECT DISTINCT item_id
            FROM analytics.user_activity
            WHERE user_id = %s
                AND item_type = 'article'
                AND action_type IN ('view', 'read')
            ORDER BY time DESC
            LIMIT %s
        """
        
        with self.warehouse.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, [user_id, limit])
            return [row[0] for row in cursor.fetchall()]
    
    def _get_user_viewed_articles(self, user_id: str, days: int = 30) -> Set[str]:
        """Get articles viewed by user in recent days."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = """
            SELECT DISTINCT item_id
            FROM analytics.user_activity
            WHERE user_id = %s
                AND item_type = 'article'
                AND action_type IN ('view', 'read')
                AND time >= %s
        """
        
        with self.warehouse.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, [user_id, start_date])
            return {row[0] for row in cursor.fetchall()}
    
    def _enrich_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich recommendations with article details."""
        if not recommendations:
            return []
        
        article_ids = [rec['article_id'] for rec in recommendations]
        
        # Get article details
        placeholders = ','.join(['%s'] * len(article_ids))
        query = f"""
            SELECT 
                id, url, title, content, published_date,
                source, author, category, metadata
            FROM articles
            WHERE id IN ({placeholders})
        """
        
        with self.db_adapter.get_connection() as conn:
            articles_df = pd.read_sql_query(query, conn, params=article_ids)
        
        # Create lookup
        articles_dict = articles_df.set_index('id').to_dict('index')
        
        # Enrich recommendations
        enriched = []
        for rec in recommendations:
            article_id = rec['article_id']
            if article_id in articles_dict:
                article_data = articles_dict[article_id]
                
                # Parse metadata
                metadata = article_data.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                enriched_rec = {
                    **rec,
                    'title': article_data['title'],
                    'url': article_data['url'],
                    'source': article_data['source'],
                    'published_date': article_data['published_date'].isoformat() if article_data['published_date'] else None,
                    'summary': article_data['content'][:200] + '...' if article_data['content'] else '',
                    'keywords': metadata.get('ai_analysis', {}).get('keywords', [])
                }
                
                enriched.append(enriched_rec)
        
        return enriched
    
    def _get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article details by ID."""
        query = """
            SELECT * FROM articles WHERE id = %s
        """
        
        with self.db_adapter.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, [article_id])
            
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            
            if row:
                return dict(zip(columns, row))
        
        return None
    
    def _get_metadata_similar_articles(self, article_id: str,
                                     n_similar: int) -> List[Dict[str, Any]]:
        """Get similar articles based on metadata."""
        # Get source article
        article = self._get_article_by_id(article_id)
        if not article:
            return []
        
        # Find similar by source and time
        query = """
            SELECT 
                id, url, title, published_date, source,
                CASE
                    WHEN source = %s THEN 0.5
                    ELSE 0.0
                END +
                CASE
                    WHEN ABS(EXTRACT(EPOCH FROM (published_date - %s))) < 86400 THEN 0.3
                    WHEN ABS(EXTRACT(EPOCH FROM (published_date - %s))) < 604800 THEN 0.2
                    ELSE 0.0
                END as similarity_score
            FROM articles
            WHERE id != %s
                AND published_date IS NOT NULL
            ORDER BY similarity_score DESC, published_date DESC
            LIMIT %s
        """
        
        with self.db_adapter.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn,
                params=[
                    article['source'],
                    article['published_date'],
                    article['published_date'],
                    article_id,
                    n_similar
                ]
            )
        
        recommendations = []
        for _, row in df.iterrows():
            recommendations.append({
                'article_id': row['id'],
                'score': float(row['similarity_score']),
                'method': 'metadata_similarity',
                'title': row['title'],
                'url': row['url'],
                'source': row['source'],
                'published_date': row['published_date'].isoformat() if row['published_date'] else None
            })
        
        return recommendations
    
    def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences based on activity."""
        preferences = {
            'preferred_sources': [],
            'preferred_keywords': [],
            'preferred_categories': []
        }
        
        try:
            # Get user's interaction history
            query = """
                SELECT 
                    a.source,
                    a.category,
                    a.metadata,
                    COUNT(*) as interaction_count
                FROM analytics.user_activity ua
                JOIN articles a ON ua.item_id = a.id
                WHERE ua.user_id = %s
                    AND ua.item_type = 'article'
                    AND ua.action_type IN ('view', 'read', 'share', 'save')
                    AND ua.time >= NOW() - INTERVAL '30 days'
                GROUP BY a.source, a.category, a.metadata
                ORDER BY interaction_count DESC
                LIMIT 100
            """
            
            with self.warehouse.get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=[user_id])
            
            if not df.empty:
                # Top sources
                source_counts = df.groupby('source')['interaction_count'].sum()
                preferences['preferred_sources'] = source_counts.nlargest(5).index.tolist()
                
                # Extract keywords from metadata
                all_keywords = []
                for metadata_str in df['metadata'].dropna():
                    try:
                        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                        if 'ai_analysis' in metadata and 'keywords' in metadata['ai_analysis']:
                            all_keywords.extend(metadata['ai_analysis']['keywords'])
                    except:
                        pass
                
                if all_keywords:
                    keyword_counts = pd.Series(all_keywords).value_counts()
                    preferences['preferred_keywords'] = keyword_counts.head(10).index.tolist()
            
        except Exception as e:
            self.logger.warning(f"Could not get user preferences: {e}")
        
        return preferences
    
    def _update_recommendation_weights(self, feedback_type: str, feedback_value: float):
        """Update recommendation weights based on feedback."""
        # Positive feedback increases weight for the method
        if feedback_type in ['click', 'share', 'save'] and feedback_value > 0:
            # Slightly increase hybrid weights for successful recommendations
            adjustment = 0.01 * feedback_value
            
            # This is a simplified version - in production, you'd want to
            # track which method led to the successful recommendation
            self.hybrid_weights = {
                k: min(0.8, v + adjustment) 
                for k, v in self.hybrid_weights.items()
            }
            
            # Normalize weights
            total = sum(self.hybrid_weights.values())
            self.hybrid_weights = {
                k: v / total 
                for k, v in self.hybrid_weights.items()
            }
    
    def _clear_user_cache(self, user_id: str):
        """Clear recommendation cache for a user."""
        keys_to_remove = [
            key for key in self.recommendation_cache.keys() 
            if key.startswith(f"{user_id}_")
        ]
        
        for key in keys_to_remove:
            del self.recommendation_cache[key]
    
    def _log_recommendations(self, user_id: str, recommendations: List[Dict[str, Any]],
                           method: str):
        """Log recommendation event."""
        # Store last method used for user
        self.recommendation_cache[f"{user_id}_last_method"] = method
        
        # Log to analytics warehouse
        try:
            with self.warehouse.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO analytics.user_activity 
                    (time, user_id, action_type, item_type, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, [
                    datetime.utcnow(),
                    user_id,
                    'recommendations_generated',
                    'recommendation_set',
                    json.dumps({
                        'method': method,
                        'count': len(recommendations),
                        'article_ids': [r['article_id'] for r in recommendations[:10]]
                    })
                ])
                
                conn.commit()
        except Exception as e:
            self.logger.warning(f"Could not log recommendations: {e}")