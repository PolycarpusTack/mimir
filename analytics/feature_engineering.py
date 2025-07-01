"""Feature Engineering Functions for Mimir Analytics.

This module provides feature computation functions for the ML feature store,
including time-series features, text features, and aggregate statistics.
"""

import hashlib
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class FeatureEngineering:
    """Feature engineering utilities for ML models."""
    
    def __init__(self):
        """Initialize feature engineering utilities."""
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        
        # Load spaCy model if available
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
        else:
            self.nlp = None
    
    # Time-Series Features
    
    def compute_article_volume_features(self, df: pd.DataFrame, 
                                      window_sizes: List[int] = [1, 6, 12, 24]) -> pd.DataFrame:
        """Compute article volume features with various time windows.
        
        Args:
            df: DataFrame with article metrics
            window_sizes: List of window sizes in hours
            
        Returns:
            DataFrame with volume features
        """
        features = df.copy()
        
        # Ensure datetime index
        if 'time' in features.columns:
            features['time'] = pd.to_datetime(features['time'])
            features = features.set_index('time').sort_index()
        
        # Rolling statistics
        for window in window_sizes:
            features[f'volume_mean_{window}h'] = features['article_count'].rolling(
                f'{window}H', min_periods=1
            ).mean()
            
            features[f'volume_std_{window}h'] = features['article_count'].rolling(
                f'{window}H', min_periods=1
            ).std()
            
            features[f'volume_max_{window}h'] = features['article_count'].rolling(
                f'{window}H', min_periods=1
            ).max()
            
            features[f'volume_min_{window}h'] = features['article_count'].rolling(
                f'{window}H', min_periods=1
            ).min()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            features[f'volume_lag_{lag}h'] = features['article_count'].shift(lag)
        
        # Change features
        features['volume_change_1h'] = features['article_count'].diff(1)
        features['volume_change_6h'] = features['article_count'].diff(6)
        features['volume_change_24h'] = features['article_count'].diff(24)
        
        # Percentage changes
        features['volume_pct_change_1h'] = features['article_count'].pct_change(1)
        features['volume_pct_change_24h'] = features['article_count'].pct_change(24)
        
        # Time-based features
        features['hour_of_day'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        features['day_of_month'] = features.index.day
        features['week_of_year'] = features.index.isocalendar().week
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Reset index for output
        features = features.reset_index()
        
        # Add entity ID (can be source, keyword, etc.)
        if 'source' in df.columns:
            features['entity_id'] = df['source']
        else:
            features['entity_id'] = 'global'
        
        features['timestamp'] = features['time']
        
        return features
    
    def compute_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute sentiment-based features.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            DataFrame with sentiment features
        """
        features = df.copy()
        
        if 'avg_sentiment' not in features.columns:
            return features
        
        # Basic sentiment features
        features['sentiment_positive'] = (features['avg_sentiment'] > 0.1).astype(int)
        features['sentiment_negative'] = (features['avg_sentiment'] < -0.1).astype(int)
        features['sentiment_neutral'] = (
            (features['avg_sentiment'] >= -0.1) & 
            (features['avg_sentiment'] <= 0.1)
        ).astype(int)
        
        # Sentiment volatility
        features['sentiment_std_24h'] = features['avg_sentiment'].rolling(
            '24H', min_periods=1
        ).std()
        
        features['sentiment_range_24h'] = (
            features['avg_sentiment'].rolling('24H', min_periods=1).max() -
            features['avg_sentiment'].rolling('24H', min_periods=1).min()
        )
        
        # Sentiment momentum
        features['sentiment_ma_6h'] = features['avg_sentiment'].rolling(
            '6H', min_periods=1
        ).mean()
        
        features['sentiment_ma_24h'] = features['avg_sentiment'].rolling(
            '24H', min_periods=1
        ).mean()
        
        features['sentiment_momentum'] = (
            features['sentiment_ma_6h'] - features['sentiment_ma_24h']
        )
        
        # Sentiment change detection
        features['sentiment_change_1h'] = features['avg_sentiment'].diff(1)
        features['sentiment_change_24h'] = features['avg_sentiment'].diff(24)
        
        return features
    
    def compute_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute keyword-based features.
        
        Args:
            df: DataFrame with keyword metrics
            
        Returns:
            DataFrame with keyword features
        """
        features = df.copy()
        
        if 'keyword' not in features.columns:
            return features
        
        # Keyword statistics
        keyword_stats = features.groupby('keyword').agg({
            'mention_count': ['sum', 'mean', 'std', 'max'],
            'growth_rate': ['mean', 'std', 'max', 'min']
        }).reset_index()
        
        keyword_stats.columns = ['_'.join(col).strip() for col in keyword_stats.columns]
        keyword_stats.rename(columns={'keyword_': 'keyword'}, inplace=True)
        
        # Merge back
        features = features.merge(keyword_stats, on='keyword', how='left')
        
        # Keyword popularity rank
        features['keyword_rank'] = features.groupby('time')['mention_count'].rank(
            ascending=False, method='dense'
        )
        
        # Keyword momentum
        features['keyword_momentum_24h'] = features.groupby('keyword')['mention_count'].transform(
            lambda x: x.rolling('24H', min_periods=1).mean()
        )
        
        # Keyword seasonality
        features['keyword_hour_avg'] = features.groupby(
            ['keyword', features['time'].dt.hour]
        )['mention_count'].transform('mean')
        
        features['keyword_dow_avg'] = features.groupby(
            ['keyword', features['time'].dt.dayofweek]
        )['mention_count'].transform('mean')
        
        # Relative frequency
        total_mentions = features.groupby('time')['mention_count'].transform('sum')
        features['keyword_relative_freq'] = features['mention_count'] / total_mentions
        
        return features
    
    def compute_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute source-based features.
        
        Args:
            df: DataFrame with source metrics
            
        Returns:
            DataFrame with source features
        """
        features = df.copy()
        
        if 'source' not in features.columns:
            return features
        
        # Source reliability features
        features['source_success_rate_ma'] = features.groupby('source')['success_rate'].transform(
            lambda x: x.rolling('7D', min_periods=1).mean()
        )
        
        # Source activity patterns
        features['source_articles_per_hour'] = features.groupby('source')['article_count'].transform(
            lambda x: x.rolling('1H', min_periods=1).sum()
        )
        
        features['source_peak_hour'] = features.groupby(
            ['source', features['time'].dt.hour]
        )['article_count'].transform('mean')
        
        # Source diversity (unique topics/keywords)
        if 'unique_keywords' in features.columns:
            features['source_topic_diversity'] = features.groupby('source')['unique_keywords'].transform(
                lambda x: x.rolling('24H', min_periods=1).mean()
            )
        
        # Source consistency
        features['source_volume_cv'] = features.groupby('source')['article_count'].transform(
            lambda x: x.rolling('7D', min_periods=1).std() / x.rolling('7D', min_periods=1).mean()
        )
        
        return features
    
    # Text Features
    
    def compute_text_features(self, texts: List[str], 
                            method: str = 'tfidf',
                            max_features: int = 100) -> pd.DataFrame:
        """Compute text-based features from article content.
        
        Args:
            texts: List of text documents
            method: Feature extraction method (tfidf, bow, embeddings)
            max_features: Maximum number of features
            
        Returns:
            DataFrame with text features
        """
        if not texts:
            return pd.DataFrame()
        
        if method == 'tfidf':
            # TF-IDF features
            if 'tfidf' not in self.vectorizers:
                self.vectorizers['tfidf'] = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            vectorizer = self.vectorizers['tfidf']
            features = vectorizer.fit_transform(texts)
            
            feature_names = [f'tfidf_{name}' for name in vectorizer.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)
            
        elif method == 'bow':
            # Bag of words
            if 'bow' not in self.vectorizers:
                from sklearn.feature_extraction.text import CountVectorizer
                self.vectorizers['bow'] = CountVectorizer(
                    max_features=max_features,
                    stop_words='english'
                )
            
            vectorizer = self.vectorizers['bow']
            features = vectorizer.fit_transform(texts)
            
            feature_names = [f'bow_{name}' for name in vectorizer.get_feature_names_out()]
            return pd.DataFrame(features.toarray(), columns=feature_names)
            
        elif method == 'embeddings' and self.nlp:
            # Sentence embeddings using spaCy
            embeddings = []
            
            for text in texts:
                doc = self.nlp(text[:1000000])  # Limit text length
                # Get document vector
                embeddings.append(doc.vector)
            
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'embedding_{i}' for i in range(len(embeddings[0]))]
            )
            
            return embedding_df
        
        else:
            # Basic text statistics
            features = pd.DataFrame()
            features['text_length'] = [len(text) for text in texts]
            features['word_count'] = [len(text.split()) for text in texts]
            features['avg_word_length'] = [
                np.mean([len(word) for word in text.split()]) if text.split() else 0
                for text in texts
            ]
            features['sentence_count'] = [len(text.split('.')) for text in texts]
            features['exclamation_count'] = [text.count('!') for text in texts]
            features['question_count'] = [text.count('?') for text in texts]
            features['uppercase_ratio'] = [
                sum(1 for c in text if c.isupper()) / len(text) if text else 0
                for text in texts
            ]
            
            return features
    
    # Aggregate Features
    
    def compute_aggregate_features(self, df: pd.DataFrame, 
                                 group_by: str,
                                 time_window: str = '24H') -> pd.DataFrame:
        """Compute aggregate features for grouped data.
        
        Args:
            df: Input DataFrame
            group_by: Column to group by
            time_window: Time window for aggregation
            
        Returns:
            DataFrame with aggregate features
        """
        if group_by not in df.columns:
            return df
        
        # Ensure datetime index
        if 'time' in df.columns:
            df = df.set_index('time').sort_index()
        
        agg_features = []
        
        for name, group in df.groupby(group_by):
            group_features = {
                f'{group_by}': name,
                'timestamp': group.index[-1]  # Latest timestamp
            }
            
            # Volume aggregates
            if 'article_count' in group.columns:
                group_features.update({
                    f'{group_by}_total_articles': group['article_count'].sum(),
                    f'{group_by}_avg_articles': group['article_count'].mean(),
                    f'{group_by}_max_articles': group['article_count'].max(),
                    f'{group_by}_article_volatility': group['article_count'].std()
                })
            
            # Sentiment aggregates
            if 'avg_sentiment' in group.columns:
                group_features.update({
                    f'{group_by}_avg_sentiment': group['avg_sentiment'].mean(),
                    f'{group_by}_sentiment_trend': group['avg_sentiment'].iloc[-1] - group['avg_sentiment'].iloc[0],
                    f'{group_by}_sentiment_volatility': group['avg_sentiment'].std()
                })
            
            # Growth aggregates
            if 'growth_rate' in group.columns:
                group_features.update({
                    f'{group_by}_avg_growth': group['growth_rate'].mean(),
                    f'{group_by}_max_growth': group['growth_rate'].max(),
                    f'{group_by}_growth_acceleration': group['growth_rate'].diff().mean()
                })
            
            agg_features.append(group_features)
        
        return pd.DataFrame(agg_features)
    
    # Feature Scaling and Encoding
    
    def scale_features(self, features: pd.DataFrame, 
                      exclude_columns: List[str] = None) -> pd.DataFrame:
        """Scale numerical features using StandardScaler.
        
        Args:
            features: DataFrame with features
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            Scaled features DataFrame
        """
        exclude_columns = exclude_columns or []
        scaled_features = features.copy()
        
        # Identify numerical columns
        numerical_cols = [
            col for col in features.select_dtypes(include=[np.number]).columns
            if col not in exclude_columns
        ]
        
        if numerical_cols:
            # Initialize or use existing scaler
            if 'standard' not in self.scalers:
                self.scalers['standard'] = StandardScaler()
            
            scaler = self.scalers['standard']
            scaled_features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
        
        return scaled_features
    
    def encode_categorical_features(self, features: pd.DataFrame,
                                  categorical_columns: List[str] = None) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            features: DataFrame with features
            categorical_columns: List of categorical columns
            
        Returns:
            Encoded features DataFrame
        """
        encoded_features = features.copy()
        
        if categorical_columns is None:
            categorical_columns = features.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in features.columns:
                # Initialize or use existing encoder
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                
                encoder = self.encoders[col]
                
                # Handle unseen categories
                unique_values = features[col].unique()
                if hasattr(encoder, 'classes_'):
                    # Add new categories
                    new_categories = set(unique_values) - set(encoder.classes_)
                    if new_categories:
                        encoder.classes_ = np.append(encoder.classes_, list(new_categories))
                
                encoded_features[f'{col}_encoded'] = encoder.fit_transform(features[col])
        
        return encoded_features
    
    # Feature Selection
    
    def select_top_features(self, features: pd.DataFrame, 
                          target: pd.Series,
                          n_features: int = 50,
                          method: str = 'mutual_info') -> List[str]:
        """Select top features based on importance.
        
        Args:
            features: Features DataFrame
            target: Target variable
            n_features: Number of features to select
            method: Selection method (mutual_info, correlation, variance)
            
        Returns:
            List of selected feature names
        """
        numerical_features = features.select_dtypes(include=[np.number])
        
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            scores = mutual_info_regression(numerical_features, target)
            feature_scores = pd.Series(scores, index=numerical_features.columns)
            
        elif method == 'correlation':
            correlations = numerical_features.corrwith(target).abs()
            feature_scores = correlations
            
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            
            selector = VarianceThreshold()
            selector.fit(numerical_features)
            feature_scores = pd.Series(
                selector.variances_,
                index=numerical_features.columns
            )
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Sort and select top features
        top_features = feature_scores.nlargest(n_features).index.tolist()
        
        return top_features
    
    # Feature Validation
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature quality and identify issues.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Validation report
        """
        report = {
            'total_features': len(features.columns),
            'total_samples': len(features),
            'issues': []
        }
        
        # Check for missing values
        missing_counts = features.isnull().sum()
        if missing_counts.any():
            report['issues'].append({
                'type': 'missing_values',
                'details': missing_counts[missing_counts > 0].to_dict()
            })
        
        # Check for constant features
        constant_features = [
            col for col in features.columns
            if features[col].nunique() <= 1
        ]
        if constant_features:
            report['issues'].append({
                'type': 'constant_features',
                'details': constant_features
            })
        
        # Check for high correlation
        numerical_features = features.select_dtypes(include=[np.number])
        if len(numerical_features.columns) > 1:
            corr_matrix = numerical_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_pairs = [
                (col1, col2, corr)
                for col1 in corr_matrix.columns
                for col2 in corr_matrix.columns
                if col1 != col2 and upper_triangle.loc[col1, col2] > 0.95
            ]
            
            if high_corr_pairs:
                report['issues'].append({
                    'type': 'high_correlation',
                    'details': high_corr_pairs[:10]  # Limit to top 10
                })
        
        # Check for outliers
        outlier_features = {}
        for col in numerical_features.columns:
            q1 = features[col].quantile(0.25)
            q3 = features[col].quantile(0.75)
            iqr = q3 - q1
            
            outliers = features[
                (features[col] < q1 - 3 * iqr) | 
                (features[col] > q3 + 3 * iqr)
            ]
            
            if len(outliers) > 0:
                outlier_features[col] = len(outliers)
        
        if outlier_features:
            report['issues'].append({
                'type': 'outliers',
                'details': outlier_features
            })
        
        # Feature statistics
        report['feature_types'] = {
            'numerical': len(features.select_dtypes(include=[np.number]).columns),
            'categorical': len(features.select_dtypes(include=['object']).columns),
            'datetime': len(features.select_dtypes(include=['datetime']).columns)
        }
        
        return report