"""Predictive Analytics Engine for Mimir News Intelligence.

This module implements advanced predictive analytics including article volume
forecasting, topic trend prediction, sentiment forecasting, and adaptive
alert threshold learning.
"""

import logging
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    # Suppress Prophet warnings
    logging.getLogger('prophet').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore', category=FutureWarning)
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TF_AVAILABLE = False

from .data_warehouse import AnalyticsDataWarehouse
from .trend_analyzer import TrendAnalyzer

logger = logging.getLogger(__name__)


class PredictiveAnalytics:
    """Advanced predictive analytics engine for news intelligence."""

    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse):
        """Initialize the predictive analytics engine.
        
        Args:
            analytics_warehouse: Analytics data warehouse for data access
        """
        self.analytics_warehouse = analytics_warehouse
        self.trend_analyzer = TrendAnalyzer(analytics_warehouse)
        self.models_dir = Path("analytics/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def forecast_article_volume(self,
                               sources: Optional[List[str]] = None,
                               forecast_days: int = 7,
                               model_type: str = 'prophet') -> Dict[str, Any]:
        """Forecast article volume for specified sources.
        
        Args:
            sources: List of sources to forecast (None for all)
            forecast_days: Number of days to forecast ahead
            model_type: Model type ('prophet', 'arima', 'ml', 'ensemble')
            
        Returns:
            Comprehensive forecast results
        """
        try:
            # Get historical data (last 60 days for training)
            start_time = datetime.utcnow() - timedelta(days=60)
            end_time = datetime.utcnow()
            
            trend_data = self.analytics_warehouse.get_article_trends(
                sources=sources,
                start_time=start_time,
                end_time=end_time,
                interval='1 hour'
            )
            
            if len(trend_data) < 168:  # Need at least 1 week of hourly data
                self.logger.warning("Insufficient data for volume forecasting")
                return {'error': 'Insufficient historical data'}

            # Prepare data
            df = pd.DataFrame(trend_data)
            df['bucket'] = pd.to_datetime(df['bucket'])
            
            # Aggregate by time if multiple sources
            if sources and len(sources) > 1:
                df = df.groupby('bucket').agg({
                    'article_count': 'sum',
                    'avg_sentiment': 'mean'
                }).reset_index()
            
            df = df.sort_values('bucket')
            
            # Choose forecasting method
            if model_type == 'prophet' and PROPHET_AVAILABLE:
                forecast_result = self._prophet_volume_forecast(df, forecast_days)
            elif model_type == 'ml':
                forecast_result = self._ml_volume_forecast(df, forecast_days)
            elif model_type == 'ensemble':
                forecast_result = self._ensemble_volume_forecast(df, forecast_days)
            else:
                # Default to simple trend-based forecast
                forecast_result = self._simple_volume_forecast(df, forecast_days)

            # Add metadata
            forecast_result.update({
                'forecast_type': 'article_volume',
                'sources': sources or ['all'],
                'forecast_days': forecast_days,
                'model_used': model_type,
                'training_data_points': len(df),
                'forecast_generated_at': datetime.utcnow()
            })

            self.logger.info(f"Generated {forecast_days}-day volume forecast using {model_type}")
            return forecast_result

        except Exception as e:
            self.logger.error(f"Failed to forecast article volume: {e}")
            return {'error': str(e)}

    def predict_topic_trends(self,
                           keywords: Optional[List[str]] = None,
                           prediction_days: int = 7,
                           min_historical_days: int = 30) -> Dict[str, Any]:
        """Predict trending topics and their future popularity.
        
        Args:
            keywords: Specific keywords to predict (None for top keywords)
            prediction_days: Days to predict ahead
            min_historical_days: Minimum historical data required
            
        Returns:
            Topic trend predictions
        """
        try:
            start_time = datetime.utcnow() - timedelta(days=min_historical_days)
            end_time = datetime.utcnow()
            
            # Get keyword trend data
            keyword_data = self.analytics_warehouse.get_keyword_trends(
                keywords=keywords,
                start_time=start_time,
                end_time=end_time,
                limit=100 if not keywords else len(keywords)
            )
            
            if not keyword_data:
                return {'error': 'No keyword data available'}

            df = pd.DataFrame(keyword_data)
            df['bucket'] = pd.to_datetime(df['bucket'])
            
            predictions = {}
            
            # Analyze each keyword
            for keyword in df['keyword'].unique():
                keyword_df = df[df['keyword'] == keyword].sort_values('bucket')
                
                if len(keyword_df) < 10:  # Need minimum data
                    continue
                
                # Calculate trend features
                trend_features = self._extract_trend_features(keyword_df, 'total_mentions')
                
                # Predict future trend
                prediction = self._predict_keyword_trend(keyword_df, prediction_days)
                
                if prediction:
                    prediction.update(trend_features)
                    predictions[keyword] = prediction

            # Rank predictions by trend strength
            sorted_predictions = sorted(
                predictions.items(),
                key=lambda x: x[1].get('trend_strength', 0),
                reverse=True
            )

            result = {
                'prediction_type': 'topic_trends',
                'prediction_days': prediction_days,
                'predictions_count': len(predictions),
                'top_trending_keywords': [
                    {
                        'keyword': keyword,
                        'trend_direction': pred['trend_direction'],
                        'trend_strength': pred['trend_strength'],
                        'predicted_growth': pred.get('predicted_growth', 0),
                        'confidence': pred.get('confidence', 0)
                    }
                    for keyword, pred in sorted_predictions[:20]
                ],
                'detailed_predictions': dict(sorted_predictions),
                'generated_at': datetime.utcnow()
            }

            self.logger.info(f"Generated topic trend predictions for {len(predictions)} keywords")
            return result

        except Exception as e:
            self.logger.error(f"Failed to predict topic trends: {e}")
            return {'error': str(e)}

    def forecast_sentiment_trends(self,
                                sources: Optional[List[str]] = None,
                                forecast_days: int = 7) -> Dict[str, Any]:
        """Forecast sentiment trends for specified sources.
        
        Args:
            sources: Sources to analyze (None for all)
            forecast_days: Days to forecast ahead
            
        Returns:
            Sentiment trend forecasts
        """
        try:
            # Get historical sentiment data
            start_time = datetime.utcnow() - timedelta(days=30)
            end_time = datetime.utcnow()
            
            trend_data = self.analytics_warehouse.get_article_trends(
                sources=sources,
                start_time=start_time,
                end_time=end_time,
                interval='4 hours'  # 4-hour intervals for sentiment trends
            )
            
            df = pd.DataFrame(trend_data)
            df['bucket'] = pd.to_datetime(df['bucket'])
            
            # Filter out null sentiment values
            df = df[df['avg_sentiment'].notna()]
            
            if len(df) < 20:
                return {'error': 'Insufficient sentiment data for forecasting'}

            # Aggregate by time if multiple sources
            if sources and len(sources) > 1:
                df = df.groupby('bucket').agg({
                    'avg_sentiment': 'mean',
                    'article_count': 'sum'
                }).reset_index()
            
            df = df.sort_values('bucket')
            
            # Forecast sentiment using different approaches
            forecasts = {}
            
            # 1. Moving average forecast
            forecasts['moving_average'] = self._moving_average_sentiment_forecast(df, forecast_days)
            
            # 2. Linear trend forecast
            forecasts['linear_trend'] = self._linear_sentiment_forecast(df, forecast_days)
            
            # 3. Prophet forecast (if available)
            if PROPHET_AVAILABLE:
                forecasts['prophet'] = self._prophet_sentiment_forecast(df, forecast_days)
            
            # Calculate ensemble forecast
            ensemble_forecast = self._calculate_ensemble_sentiment_forecast(forecasts)
            
            # Analyze sentiment volatility
            sentiment_volatility = df['avg_sentiment'].std()
            sentiment_stability = 1 / (1 + sentiment_volatility)  # Higher = more stable
            
            result = {
                'forecast_type': 'sentiment_trends',
                'sources': sources or ['all'],
                'forecast_days': forecast_days,
                'current_sentiment': float(df['avg_sentiment'].iloc[-1]),
                'sentiment_volatility': float(sentiment_volatility),
                'sentiment_stability_score': float(sentiment_stability),
                'forecasts': forecasts,
                'ensemble_forecast': ensemble_forecast,
                'training_data_points': len(df),
                'generated_at': datetime.utcnow()
            }

            self.logger.info(f"Generated {forecast_days}-day sentiment forecast")
            return result

        except Exception as e:
            self.logger.error(f"Failed to forecast sentiment trends: {e}")
            return {'error': str(e)}

    def learn_alert_thresholds(self,
                             metric_type: str = 'article_volume',
                             learning_days: int = 60,
                             sensitivity: str = 'medium') -> Dict[str, Any]:
        """Learn adaptive alert thresholds based on historical patterns.
        
        Args:
            metric_type: Type of metric ('article_volume', 'sentiment', 'keyword_mentions')
            learning_days: Days of historical data to learn from
            sensitivity: Alert sensitivity ('low', 'medium', 'high')
            
        Returns:
            Learned threshold recommendations
        """
        try:
            start_time = datetime.utcnow() - timedelta(days=learning_days)
            end_time = datetime.utcnow()
            
            if metric_type == 'article_volume':
                data = self.analytics_warehouse.get_article_trends(
                    start_time=start_time,
                    end_time=end_time,
                    interval='1 hour'
                )
                metric_column = 'article_count'
            elif metric_type == 'sentiment':
                data = self.analytics_warehouse.get_article_trends(
                    start_time=start_time,
                    end_time=end_time,
                    interval='1 hour'
                )
                metric_column = 'avg_sentiment'
            elif metric_type == 'keyword_mentions':
                data = self.analytics_warehouse.get_keyword_trends(
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                metric_column = 'total_mentions'
            else:
                return {'error': f'Unsupported metric type: {metric_type}'}

            if not data:
                return {'error': 'No historical data available for threshold learning'}

            df = pd.DataFrame(data)
            
            # Extract the metric values
            if metric_type == 'keyword_mentions':
                # For keywords, aggregate all mentions
                values = df.groupby('bucket')[metric_column].sum().values
            else:
                values = df[metric_column].dropna().values
            
            if len(values) < 100:
                return {'error': 'Insufficient data for threshold learning'}

            # Calculate statistical thresholds
            thresholds = self._calculate_adaptive_thresholds(values, sensitivity)
            
            # Calculate seasonal adjustments
            seasonal_adjustments = self._calculate_seasonal_adjustments(df, metric_column)
            
            # Learn anomaly patterns
            anomaly_patterns = self._learn_anomaly_patterns(values)
            
            result = {
                'metric_type': metric_type,
                'learning_period_days': learning_days,
                'sensitivity': sensitivity,
                'data_points_analyzed': len(values),
                'statistical_summary': {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                },
                'recommended_thresholds': thresholds,
                'seasonal_adjustments': seasonal_adjustments,
                'anomaly_patterns': anomaly_patterns,
                'generated_at': datetime.utcnow()
            }

            self.logger.info(f"Learned alert thresholds for {metric_type} with {sensitivity} sensitivity")
            return result

        except Exception as e:
            self.logger.error(f"Failed to learn alert thresholds: {e}")
            return {'error': str(e)}

    def _prophet_volume_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate volume forecast using Prophet."""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df['bucket'],
                'y': df['article_count']
            })
            
            # Configure Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05  # More conservative changepoints
            )
            
            # Fit model
            model.fit(prophet_df)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=forecast_days * 24, freq='H')
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast['yhat'][-forecast_days * 24:].values
            lower_bound = forecast['yhat_lower'][-forecast_days * 24:].values
            upper_bound = forecast['yhat_upper'][-forecast_days * 24:].values
            
            # Calculate accuracy on historical data
            historical_actual = prophet_df['y'].values
            historical_predicted = forecast['yhat'][:len(historical_actual)].values
            
            mape = np.mean(np.abs((historical_actual - historical_predicted) / np.maximum(historical_actual, 1))) * 100
            accuracy = max(0, 1 - mape / 100)
            
            return {
                'forecast_values': forecast_values.tolist(),
                'confidence_lower': lower_bound.tolist(),
                'confidence_upper': upper_bound.tolist(),
                'forecast_accuracy': float(accuracy),
                'model_components': {
                    'trend': forecast['trend'][-forecast_days * 24:].mean(),
                    'weekly': forecast['weekly'][-forecast_days * 24:].mean() if 'weekly' in forecast else 0,
                    'daily': forecast['daily'][-forecast_days * 24:].mean() if 'daily' in forecast else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prophet volume forecast failed: {e}")
            return self._simple_volume_forecast(df, forecast_days)

    def _ml_volume_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate volume forecast using machine learning models."""
        try:
            # Prepare features
            df_features = self._create_volume_features(df)
            
            if len(df_features) < 24:  # Need minimum data
                return self._simple_volume_forecast(df, forecast_days)
            
            # Prepare training data
            X = df_features.drop(['article_count', 'bucket'], axis=1).values
            y = df_features['article_count'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train ensemble of models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            model_scores = {}
            trained_models = {}
            
            for name, model in models.items():
                scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    scores.append(score)
                
                model_scores[name] = np.mean(scores)
                
                # Train on full dataset
                model.fit(X_scaled, y)
                trained_models[name] = model
            
            # Select best model
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = trained_models[best_model_name]
            
            # Generate forecast features
            forecast_features = self._generate_forecast_features(df_features, forecast_days)
            X_forecast = scaler.transform(forecast_features)
            
            # Make predictions
            predictions = best_model.predict(X_forecast)
            
            return {
                'forecast_values': predictions.tolist(),
                'model_used': best_model_name,
                'model_score': float(model_scores[best_model_name]),
                'all_model_scores': model_scores
            }
            
        except Exception as e:
            self.logger.error(f"ML volume forecast failed: {e}")
            return self._simple_volume_forecast(df, forecast_days)

    def _ensemble_volume_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate ensemble forecast combining multiple methods."""
        forecasts = []
        
        # Get individual forecasts
        if PROPHET_AVAILABLE:
            prophet_forecast = self._prophet_volume_forecast(df, forecast_days)
            if 'forecast_values' in prophet_forecast:
                forecasts.append(('prophet', prophet_forecast['forecast_values'], prophet_forecast.get('forecast_accuracy', 0.5)))
        
        ml_forecast = self._ml_volume_forecast(df, forecast_days)
        if 'forecast_values' in ml_forecast:
            forecasts.append(('ml', ml_forecast['forecast_values'], ml_forecast.get('model_score', 0.5)))
        
        simple_forecast = self._simple_volume_forecast(df, forecast_days)
        if 'forecast_values' in simple_forecast:
            forecasts.append(('simple', simple_forecast['forecast_values'], 0.3))
        
        if not forecasts:
            return {'error': 'No forecasts generated for ensemble'}
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in forecasts)
        ensemble_values = np.zeros(forecast_days * 24)
        
        for name, values, weight in forecasts:
            ensemble_values += np.array(values) * (weight / total_weight)
        
        return {
            'forecast_values': ensemble_values.tolist(),
            'ensemble_components': [name for name, _, _ in forecasts],
            'component_weights': {name: weight/total_weight for name, _, weight in forecasts}
        }

    def _simple_volume_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Generate simple trend-based forecast."""
        try:
            # Calculate recent trend
            recent_data = df.tail(168)  # Last week
            values = recent_data['article_count'].values
            
            # Linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Generate forecast
            forecast_hours = forecast_days * 24
            forecast_x = np.arange(len(values), len(values) + forecast_hours)
            forecast_values = slope * forecast_x + intercept
            
            # Apply weekly seasonality (simple)
            if len(values) >= 168:  # Full week of data
                weekly_pattern = []
                for i in range(168):
                    hour_values = values[i::168]
                    weekly_pattern.append(np.mean(hour_values) if len(hour_values) > 0 else np.mean(values))
                
                # Apply pattern to forecast
                for i, val in enumerate(forecast_values):
                    pattern_factor = weekly_pattern[i % 168] / np.mean(weekly_pattern)
                    forecast_values[i] = val * pattern_factor
            
            # Ensure non-negative values
            forecast_values = np.maximum(forecast_values, 0)
            
            return {
                'forecast_values': forecast_values.tolist(),
                'trend_slope': float(slope),
                'trend_intercept': float(intercept)
            }
            
        except Exception as e:
            self.logger.error(f"Simple volume forecast failed: {e}")
            return {'error': str(e)}

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for volume forecasting."""
        df_features = df.copy()
        df_features['hour'] = df_features['bucket'].dt.hour
        df_features['day_of_week'] = df_features['bucket'].dt.dayofweek
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 6, 12, 24]:
            df_features[f'article_count_lag_{lag}'] = df_features['article_count'].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24]:
            df_features[f'article_count_rolling_mean_{window}'] = df_features['article_count'].rolling(window).mean()
            df_features[f'article_count_rolling_std_{window}'] = df_features['article_count'].rolling(window).std()
        
        # Drop rows with NaN values
        df_features = df_features.dropna()
        
        return df_features

    def _generate_forecast_features(self, df_features: pd.DataFrame, forecast_days: int) -> np.ndarray:
        """Generate features for forecasting future periods."""
        # This is a simplified version - in practice, you'd need more sophisticated feature generation
        last_row = df_features.iloc[-1]
        feature_columns = [col for col in df_features.columns if col not in ['article_count', 'bucket']]
        
        forecast_features = []
        for hour in range(forecast_days * 24):
            # Simple feature generation - extend the pattern
            features = []
            for col in feature_columns:
                if 'lag' in col:
                    # For lag features, use recent values
                    features.append(last_row[col])
                elif 'rolling' in col:
                    # For rolling features, use recent values
                    features.append(last_row[col])
                else:
                    # For time features, calculate for future time
                    features.append(last_row[col])
            forecast_features.append(features)
        
        return np.array(forecast_features)

    def _extract_trend_features(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Extract trend features for keyword analysis."""
        values = df[column].values
        
        # Linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        # Recent vs overall average
        recent_avg = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
        overall_avg = np.mean(values)
        relative_recent = recent_avg / overall_avg if overall_avg > 0 else 1
        
        return {
            'trend_direction': 'up' if slope > 0 else 'down',
            'trend_strength': abs(slope) * 100,
            'volatility': float(volatility),
            'recent_vs_overall': float(relative_recent)
        }

    def _predict_keyword_trend(self, df: pd.DataFrame, prediction_days: int) -> Optional[Dict[str, Any]]:
        """Predict future trend for a keyword."""
        try:
            values = df['total_mentions'].values
            
            if len(values) < 5:
                return None
            
            # Simple linear prediction
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Predict future values
            future_x = np.arange(len(values), len(values) + prediction_days)
            future_values = slope * future_x + intercept
            
            # Calculate confidence
            residuals = values - (slope * x + intercept)
            mse = np.mean(residuals ** 2)
            confidence = 1 / (1 + mse)  # Simple confidence metric
            
            # Calculate predicted growth
            current_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
            predicted_avg = np.mean(future_values)
            growth_rate = (predicted_avg - current_avg) / current_avg if current_avg > 0 else 0
            
            return {
                'trend_direction': 'up' if slope > 0 else 'down',
                'trend_strength': abs(slope) * 100,
                'predicted_growth': float(growth_rate * 100),
                'confidence': float(min(confidence, 1.0)),
                'predicted_values': future_values.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to predict keyword trend: {e}")
            return None

    def _moving_average_sentiment_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Forecast sentiment using moving averages."""
        values = df['avg_sentiment'].values
        
        # Calculate different moving averages
        ma_short = np.mean(values[-12:])  # Last 12 periods
        ma_long = np.mean(values[-48:]) if len(values) >= 48 else np.mean(values)  # Last 48 periods or all
        
        # Trend direction
        trend = ma_short - ma_long
        
        # Simple forecast: extend the recent average with slight trend adjustment
        forecast_value = ma_short + (trend * 0.1)  # Conservative trend extension
        
        return {
            'forecast_value': float(forecast_value),
            'method': 'moving_average',
            'confidence': 0.6
        }

    def _linear_sentiment_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Forecast sentiment using linear trend."""
        values = df['avg_sentiment'].values
        x = np.arange(len(values))
        
        slope, intercept = np.polyfit(x, values, 1)
        
        # Forecast future value
        future_x = len(values) + (forecast_days * 6)  # Assuming 4-hour intervals
        forecast_value = slope * future_x + intercept
        
        # Clamp to valid sentiment range
        forecast_value = np.clip(forecast_value, -1.0, 1.0)
        
        # Calculate confidence based on R²
        predicted = slope * x + intercept
        r2 = r2_score(values, predicted)
        confidence = max(0.1, r2)
        
        return {
            'forecast_value': float(forecast_value),
            'method': 'linear_trend',
            'confidence': float(confidence),
            'trend_slope': float(slope)
        }

    def _prophet_sentiment_forecast(self, df: pd.DataFrame, forecast_days: int) -> Dict[str, Any]:
        """Forecast sentiment using Prophet."""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df['bucket'],
                'y': df['avg_sentiment']
            })
            
            # Configure Prophet for sentiment (no strong seasonality expected)
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.80
            )
            
            model.fit(prophet_df)
            
            # Generate forecast
            future_periods = forecast_days * 6  # 4-hour intervals
            future = model.make_future_dataframe(periods=future_periods, freq='4H')
            forecast = model.predict(future)
            
            # Get the final forecast value
            forecast_value = forecast['yhat'].iloc[-1]
            lower_bound = forecast['yhat_lower'].iloc[-1]
            upper_bound = forecast['yhat_upper'].iloc[-1]
            
            # Clamp to valid sentiment range
            forecast_value = np.clip(forecast_value, -1.0, 1.0)
            
            # Calculate confidence based on prediction interval width
            interval_width = upper_bound - lower_bound
            confidence = max(0.1, 1 - interval_width / 2)  # Narrower interval = higher confidence
            
            return {
                'forecast_value': float(forecast_value),
                'method': 'prophet',
                'confidence': float(confidence),
                'confidence_interval': [float(lower_bound), float(upper_bound)]
            }
            
        except Exception as e:
            self.logger.error(f"Prophet sentiment forecast failed: {e}")
            return self._linear_sentiment_forecast(df, forecast_days)

    def _calculate_ensemble_sentiment_forecast(self, forecasts: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate ensemble sentiment forecast."""
        valid_forecasts = {k: v for k, v in forecasts.items() if 'forecast_value' in v}
        
        if not valid_forecasts:
            return {'forecast_value': 0.0, 'confidence': 0.0}
        
        # Weight by confidence
        total_weight = sum(f['confidence'] for f in valid_forecasts.values())
        
        if total_weight == 0:
            # Equal weights if no confidence scores
            ensemble_value = np.mean([f['forecast_value'] for f in valid_forecasts.values()])
            ensemble_confidence = 0.5
        else:
            ensemble_value = sum(
                f['forecast_value'] * f['confidence'] for f in valid_forecasts.values()
            ) / total_weight
            ensemble_confidence = total_weight / len(valid_forecasts)  # Average confidence
        
        return {
            'forecast_value': float(ensemble_value),
            'confidence': float(min(ensemble_confidence, 1.0)),
            'component_methods': list(valid_forecasts.keys())
        }

    def _calculate_adaptive_thresholds(self, values: np.ndarray, sensitivity: str) -> Dict[str, float]:
        """Calculate adaptive alert thresholds."""
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Define sensitivity multipliers
        sensitivity_multipliers = {
            'low': {'upper': 3.0, 'lower': 3.0},
            'medium': {'upper': 2.0, 'lower': 2.0},
            'high': {'upper': 1.5, 'lower': 1.5}
        }
        
        multiplier = sensitivity_multipliers.get(sensitivity, sensitivity_multipliers['medium'])
        
        # Calculate percentile-based thresholds as well
        p95 = np.percentile(values, 95)
        p5 = np.percentile(values, 5)
        
        return {
            'upper_threshold': float(mean_val + multiplier['upper'] * std_val),
            'lower_threshold': float(mean_val - multiplier['lower'] * std_val),
            'upper_percentile_95': float(p95),
            'lower_percentile_5': float(p5),
            'mean': float(mean_val),
            'std': float(std_val),
            'recommended_upper': float(min(mean_val + multiplier['upper'] * std_val, p95 * 1.1)),
            'recommended_lower': float(max(mean_val - multiplier['lower'] * std_val, p5 * 0.9))
        }

    def _calculate_seasonal_adjustments(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Calculate seasonal adjustment factors."""
        try:
            if 'bucket' not in df.columns:
                return {'seasonal_factors': {}}
            
            df['hour'] = pd.to_datetime(df['bucket']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['bucket']).dt.dayofweek
            
            # Calculate hourly factors
            hourly_avg = df.groupby('hour')[column].mean()
            overall_avg = df[column].mean()
            hourly_factors = (hourly_avg / overall_avg).to_dict()
            
            # Calculate daily factors
            daily_avg = df.groupby('day_of_week')[column].mean()
            daily_factors = (daily_avg / overall_avg).to_dict()
            
            return {
                'seasonal_factors': {
                    'hourly': {int(k): float(v) for k, v in hourly_factors.items()},
                    'daily': {int(k): float(v) for k, v in daily_factors.items()}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate seasonal adjustments: {e}")
            return {'seasonal_factors': {}}

    def _learn_anomaly_patterns(self, values: np.ndarray) -> Dict[str, Any]:
        """Learn patterns in anomalous data."""
        try:
            # Detect anomalies using z-score
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            anomaly_threshold = 2.5
            anomalies = z_scores > anomaly_threshold
            
            anomaly_count = np.sum(anomalies)
            anomaly_rate = anomaly_count / len(values)
            
            if anomaly_count > 0:
                anomaly_values = values[anomalies]
                anomaly_severity = np.mean(z_scores[anomalies])
            else:
                anomaly_severity = 0
            
            return {
                'historical_anomaly_rate': float(anomaly_rate),
                'historical_anomaly_count': int(anomaly_count),
                'avg_anomaly_severity': float(anomaly_severity),
                'anomaly_threshold_zscore': float(anomaly_threshold)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to learn anomaly patterns: {e}")
            return {'historical_anomaly_rate': 0.0}