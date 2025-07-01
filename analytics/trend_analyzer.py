"""Trend Analysis Engine for Analytics Pipeline.

This module implements sophisticated trend detection algorithms including
moving averages, anomaly detection with Prophet, seasonality analysis,
and breakout detection for news analytics.
"""

import logging
import math
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Some forecasting features will be disabled.")

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logging.warning("Ruptures not available. Breakout detection will use basic methods.")

from .data_warehouse import AnalyticsDataWarehouse
from .exceptions import (
    TrendAnalysisException, DataValidationException,
    AnalyticsBaseException
)

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Advanced trend analysis engine for time-series data."""

    def __init__(self, analytics_warehouse: AnalyticsDataWarehouse) -> None:
        """Initialize the trend analyzer.
        
        Args:
            analytics_warehouse: Analytics data warehouse for data access
            
        Raises:
            TrendAnalysisException: If initialization fails
        """
        self.analytics_warehouse = analytics_warehouse
        self.logger = logging.getLogger(__name__)

    def analyze_article_volume_trends(self,
                                    sources: Optional[List[str]] = None,
                                    days_back: int = 30,
                                    interval: str = '1 hour') -> Dict[str, Any]:
        """Analyze article volume trends over time.
        
        Args:
            sources: Optional list of sources to analyze
            days_back: Number of days to look back
            interval: Time bucket interval
            
        Returns:
            Comprehensive trend analysis results
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)
        end_time = datetime.utcnow()
        
        try:
            # Get trend data
            trend_data = self.analytics_warehouse.get_article_trends(
                sources=sources,
                start_time=start_time,
                end_time=end_time,
                interval=interval
            )
            
            if not trend_data:
                self.logger.warning("No trend data available for analysis")
                return {}

            # Convert to DataFrame for analysis
            df = pd.DataFrame(trend_data)
            df['bucket'] = pd.to_datetime(df['bucket'])
            df = df.sort_values('bucket')
            
            # Aggregate by time if multiple sources
            if sources and len(sources) > 1:
                df_agg = df.groupby('bucket').agg({
                    'article_count': 'sum',
                    'avg_sentiment': 'mean',
                    'avg_content_length': 'mean',
                    'total_views': 'sum'
                }).reset_index()
            else:
                df_agg = df

            if len(df_agg) < 10:
                self.logger.warning("Insufficient data points for trend analysis")
                return {}

            # Perform comprehensive trend analysis
            analysis_results = {
                'analysis_type': 'article_volume_trend',
                'subject': ', '.join(sources) if sources else 'all_sources',
                'time_period_start': start_time,
                'time_period_end': end_time,
                'data_points': len(df_agg),
                'interval': interval
            }

            # 1. Moving Average Analysis
            ma_results = self._calculate_moving_averages(df_agg, 'article_count')
            analysis_results.update(ma_results)

            # 2. Linear Trend Analysis
            trend_results = self._analyze_linear_trend(df_agg, 'article_count')
            analysis_results.update(trend_results)

            # 3. Anomaly Detection
            anomaly_results = self._detect_anomalies(df_agg, 'article_count')
            analysis_results.update(anomaly_results)

            # 4. Seasonality Analysis
            seasonal_results = self._analyze_seasonality(df_agg, 'article_count')
            analysis_results.update(seasonal_results)

            # 5. Breakout Detection
            breakout_results = self._detect_breakouts(df_agg, 'article_count')
            analysis_results.update(breakout_results)

            # 6. Prophet Forecasting (if available)
            if PROPHET_AVAILABLE:
                forecast_results = self._prophet_forecast(df_agg, 'article_count')
                analysis_results.update(forecast_results)

            # Store analysis results
            analysis_id = self.analytics_warehouse.store_trend_analysis(analysis_results)
            if analysis_id:
                analysis_results['analysis_id'] = analysis_id

            self.logger.info(f"Completed article volume trend analysis: {analysis_results.get('trend_direction', 'unknown')} trend")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Failed to analyze article volume trends: {e}")
            return {}

    def analyze_keyword_trends(self,
                              keywords: Optional[List[str]] = None,
                              days_back: int = 30,
                              min_mentions: int = 10) -> List[Dict[str, Any]]:
        """Analyze trends for specific keywords.
        
        Args:
            keywords: List of keywords to analyze
            days_back: Number of days to look back
            min_mentions: Minimum mentions required for analysis
            
        Returns:
            List of trend analysis results for each keyword
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)
        end_time = datetime.utcnow()
        
        try:
            # Get keyword trend data
            keyword_data = self.analytics_warehouse.get_keyword_trends(
                keywords=keywords,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not keyword_data:
                self.logger.warning("No keyword data available for analysis")
                return []

            # Group by keyword
            df = pd.DataFrame(keyword_data)
            df['bucket'] = pd.to_datetime(df['bucket'])
            
            results = []
            
            for keyword in df['keyword'].unique():
                keyword_df = df[df['keyword'] == keyword].sort_values('bucket')
                
                # Skip keywords with insufficient data
                total_mentions = keyword_df['total_mentions'].sum()
                if total_mentions < min_mentions:
                    continue

                if len(keyword_df) < 5:
                    continue

                # Analyze this keyword's trend
                analysis_results = {
                    'analysis_type': 'keyword_trend',
                    'subject': keyword,
                    'time_period_start': start_time,
                    'time_period_end': end_time,
                    'total_mentions': int(total_mentions),
                    'data_points': len(keyword_df)
                }

                # Linear trend analysis
                trend_results = self._analyze_linear_trend(keyword_df, 'total_mentions')
                analysis_results.update(trend_results)

                # Moving averages
                ma_results = self._calculate_moving_averages(keyword_df, 'total_mentions')
                analysis_results.update(ma_results)

                # Anomaly detection
                anomaly_results = self._detect_anomalies(keyword_df, 'total_mentions')
                analysis_results.update(anomaly_results)

                # Sentiment trend analysis
                if 'avg_sentiment' in keyword_df.columns:
                    sentiment_trend = self._analyze_linear_trend(keyword_df, 'avg_sentiment')
                    analysis_results['sentiment_trend_direction'] = sentiment_trend.get('trend_direction')
                    analysis_results['sentiment_slope'] = sentiment_trend.get('slope')

                # Store analysis results
                analysis_id = self.analytics_warehouse.store_trend_analysis(analysis_results)
                if analysis_id:
                    analysis_results['analysis_id'] = analysis_id

                results.append(analysis_results)

            self.logger.info(f"Analyzed trends for {len(results)} keywords")
            return results

        except Exception as e:
            self.logger.error(f"Failed to analyze keyword trends: {e}")
            return []

    def analyze_source_performance_trends(self,
                                        days_back: int = 14) -> List[Dict[str, Any]]:
        """Analyze performance trends for news sources.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of trend analysis results for each source
        """
        start_time = datetime.utcnow() - timedelta(days=days_back)
        end_time = datetime.utcnow()
        
        try:
            # Get source performance data
            source_data = self.analytics_warehouse.get_source_performance(
                start_time=start_time,
                end_time=end_time
            )
            
            if not source_data:
                self.logger.warning("No source performance data available")
                return []

            results = []
            
            for source_info in source_data:
                source = source_info['source']
                
                # Get time-series data for this source
                source_trends = self.analytics_warehouse.get_article_trends(
                    sources=[source],
                    start_time=start_time,
                    end_time=end_time,
                    interval='1 day'
                )
                
                if len(source_trends) < 5:
                    continue

                df = pd.DataFrame(source_trends)
                df['bucket'] = pd.to_datetime(df['bucket'])
                df = df.sort_values('bucket')

                # Analyze source performance trends
                analysis_results = {
                    'analysis_type': 'source_performance_trend',
                    'subject': source,
                    'time_period_start': start_time,
                    'time_period_end': end_time,
                    'total_articles': int(df['article_count'].sum()),
                    'avg_sentiment': float(df['avg_sentiment'].mean()) if df['avg_sentiment'].notna().any() else None
                }

                # Article volume trend
                volume_trend = self._analyze_linear_trend(df, 'article_count')
                analysis_results.update(volume_trend)

                # Quality trends (sentiment, content length)
                if df['avg_sentiment'].notna().any():
                    sentiment_trend = self._analyze_linear_trend(df, 'avg_sentiment')
                    analysis_results['sentiment_trend_direction'] = sentiment_trend.get('trend_direction')
                    analysis_results['sentiment_slope'] = sentiment_trend.get('slope')

                if df['avg_content_length'].notna().any():
                    length_trend = self._analyze_linear_trend(df, 'avg_content_length')
                    analysis_results['content_length_trend'] = length_trend.get('trend_direction')

                # Consistency analysis (coefficient of variation)
                cv = df['article_count'].std() / df['article_count'].mean() if df['article_count'].mean() > 0 else 0
                analysis_results['consistency_score'] = max(0, 1 - cv)  # Higher is more consistent

                # Store analysis results
                analysis_id = self.analytics_warehouse.store_trend_analysis(analysis_results)
                if analysis_id:
                    analysis_results['analysis_id'] = analysis_id

                results.append(analysis_results)

            self.logger.info(f"Analyzed performance trends for {len(results)} sources")
            return results

        except Exception as e:
            self.logger.error(f"Failed to analyze source performance trends: {e}")
            return []

    def _calculate_moving_averages(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Calculate various moving averages."""
        try:
            df = df.copy()
            
            # Calculate different moving averages
            df['ma_3'] = df[column].rolling(window=3, min_periods=1).mean()
            df['ma_7'] = df[column].rolling(window=7, min_periods=1).mean()
            df['ma_14'] = df[column].rolling(window=min(14, len(df)), min_periods=1).mean()
            
            # Calculate current trend based on moving averages
            if len(df) >= 3:
                current_value = df[column].iloc[-1]
                ma_3_current = df['ma_3'].iloc[-1]
                ma_7_current = df['ma_7'].iloc[-1]
                
                # Determine trend direction
                if current_value > ma_3_current > ma_7_current:
                    trend_direction = 'strong_up'
                elif current_value > ma_3_current:
                    trend_direction = 'up'
                elif current_value < ma_3_current < ma_7_current:
                    trend_direction = 'strong_down'
                elif current_value < ma_3_current:
                    trend_direction = 'down'
                else:
                    trend_direction = 'stable'
            else:
                trend_direction = 'unknown'

            return {
                'ma_trend_direction': trend_direction,
                'current_value': float(df[column].iloc[-1]),
                'ma_3_current': float(df['ma_3'].iloc[-1]),
                'ma_7_current': float(df['ma_7'].iloc[-1]) if len(df) >= 7 else None,
                'ma_14_current': float(df['ma_14'].iloc[-1]) if len(df) >= 14 else None
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate moving averages: {e}")
            return {}

    def _analyze_linear_trend(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze linear trend using regression."""
        try:
            if len(df) < 3:
                return {}

            # Prepare data for regression
            x = np.arange(len(df))
            y = df[column].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(y)
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) < 3:
                return {}

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction and strength
            if p_value < 0.05:  # Statistically significant
                if slope > 0:
                    trend_direction = 'up'
                    trend_strength = min(abs(slope) * 10, 100)  # Scale appropriately
                else:
                    trend_direction = 'down'
                    trend_strength = min(abs(slope) * 10, 100)
            else:
                trend_direction = 'stable'
                trend_strength = 0

            # Calculate trend confidence based on R-squared
            r_squared = r_value ** 2
            trend_confidence = r_squared

            return {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 2),
                'trend_confidence': round(trend_confidence, 4),
                'slope': round(slope, 6),
                'r_squared': round(r_squared, 4),
                'p_value': round(p_value, 8)
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze linear trend: {e}")
            return {}

    def _detect_anomalies(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Detect anomalies using statistical methods."""
        try:
            if len(df) < 10:
                return {'anomalies_detected': 0}

            values = df[column].values
            timestamps = df['bucket'].values if 'bucket' in df.columns else df.index
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            timestamps = timestamps[valid_mask]
            
            if len(values) < 10:
                return {'anomalies_detected': 0}

            # Method 1: Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(values))
            z_threshold = 2.5
            z_anomalies = z_scores > z_threshold

            # Method 2: Interquartile Range (IQR) method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            iqr_threshold = 1.5
            iqr_anomalies = (values < Q1 - iqr_threshold * IQR) | (values > Q3 + iqr_threshold * IQR)

            # Combine both methods
            combined_anomalies = z_anomalies | iqr_anomalies
            
            anomaly_indices = np.where(combined_anomalies)[0]
            anomaly_dates = [timestamps[i] for i in anomaly_indices]
            anomaly_scores = [float(z_scores[i]) for i in anomaly_indices]

            return {
                'anomalies_detected': len(anomaly_indices),
                'anomaly_dates': anomaly_dates,
                'anomaly_scores': anomaly_scores
            }

        except Exception as e:
            self.logger.error(f"Failed to detect anomalies: {e}")
            return {'anomalies_detected': 0}

    def _analyze_seasonality(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        try:
            if len(df) < 24:  # Need at least 24 data points
                return {'has_seasonality': False}

            values = df[column].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            
            if len(values) < 24:
                return {'has_seasonality': False}

            # Simple seasonality detection using autocorrelation
            # Check for common seasonal periods
            seasonal_periods = [7, 24, 168]  # weekly, daily, weekly (hours)
            max_correlation = 0
            best_period = None
            
            for period in seasonal_periods:
                if len(values) > period * 2:
                    # Calculate autocorrelation at this lag
                    correlation = self._calculate_autocorrelation(values, period)
                    if abs(correlation) > max_correlation:
                        max_correlation = abs(correlation)
                        best_period = period

            # Consider seasonal if correlation > 0.3
            has_seasonality = max_correlation > 0.3
            
            return {
                'has_seasonality': has_seasonality,
                'seasonal_period_days': best_period if has_seasonality else None,
                'seasonal_strength': round(max_correlation, 3) if has_seasonality else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze seasonality: {e}")
            return {'has_seasonality': False}

    def _calculate_autocorrelation(self, values: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        try:
            if len(values) <= lag:
                return 0.0
            
            # Center the data
            values_centered = values - np.mean(values)
            
            # Calculate autocorrelation
            n = len(values_centered)
            autocorr = np.correlate(values_centered[:-lag], values_centered[lag:], mode='valid')[0]
            autocorr /= np.sum(values_centered ** 2)
            
            return float(autocorr)
            
        except Exception:
            return 0.0

    def _detect_breakouts(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Detect breakout points (sudden changes in trend)."""
        try:
            if len(df) < 10:
                return {'breakouts_detected': 0}

            values = df[column].values
            timestamps = df['bucket'].values if 'bucket' in df.columns else df.index
            
            # Remove NaN values
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            timestamps = timestamps[valid_mask]
            
            if len(values) < 10:
                return {'breakouts_detected': 0}

            if RUPTURES_AVAILABLE:
                # Use ruptures library for change point detection
                algo = rpt.Pelt(model="rbf").fit(values.reshape(-1, 1))
                breakpoints = algo.predict(pen=len(values) * 0.1)
                
                # Remove the last point (end of series)
                if breakpoints and breakpoints[-1] == len(values):
                    breakpoints = breakpoints[:-1]
                    
                breakout_dates = [timestamps[bp] for bp in breakpoints if bp < len(timestamps)]
                
            else:
                # Simple breakout detection using moving statistics
                window = max(5, len(values) // 10)
                breakpoints = []
                
                for i in range(window, len(values) - window):
                    before_mean = np.mean(values[i-window:i])
                    after_mean = np.mean(values[i:i+window])
                    
                    # Check for significant change
                    if abs(after_mean - before_mean) > np.std(values) * 1.5:
                        breakpoints.append(i)
                
                breakout_dates = [timestamps[bp] for bp in breakpoints]

            return {
                'breakouts_detected': len(breakout_dates),
                'breakout_dates': breakout_dates
            }

        except Exception as e:
            self.logger.error(f"Failed to detect breakouts: {e}")
            return {'breakouts_detected': 0}

    def _prophet_forecast(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Generate forecast using Prophet."""
        try:
            if not PROPHET_AVAILABLE or len(df) < 10:
                return {}

            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df['bucket'] if 'bucket' in df.columns else pd.date_range(start='2024-01-01', periods=len(df), freq='H'),
                'y': df[column]
            })
            
            # Remove NaN values
            prophet_df = prophet_df.dropna()
            
            if len(prophet_df) < 10:
                return {}

            # Fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )
            
            # Suppress Prophet logging
            import logging
            prophet_logger = logging.getLogger('prophet')
            prophet_logger.setLevel(logging.WARNING)
            
            model.fit(prophet_df)
            
            # Make forecast for next period
            future_periods = max(1, len(prophet_df) // 10)  # Forecast 10% ahead
            future = model.make_future_dataframe(periods=future_periods, freq='H')
            forecast = model.predict(future)
            
            # Extract forecast results
            next_value = forecast['yhat'].iloc[-1]
            confidence_lower = forecast['yhat_lower'].iloc[-1]
            confidence_upper = forecast['yhat_upper'].iloc[-1]
            
            # Calculate forecast accuracy on historical data
            historical_forecast = forecast['yhat'].iloc[:-future_periods]
            historical_actual = prophet_df['y'].iloc[:len(historical_forecast)]
            
            if len(historical_actual) > 0:
                mape = np.mean(np.abs((historical_actual - historical_forecast) / historical_actual)) * 100
                accuracy_score = max(0, 1 - mape / 100)
            else:
                accuracy_score = 0.5

            return {
                'next_period_forecast': round(float(next_value), 2),
                'forecast_confidence_interval': [round(float(confidence_lower), 2), round(float(confidence_upper), 2)],
                'forecast_accuracy_score': round(accuracy_score, 3),
                'model_used': 'Prophet'
            }

        except Exception as e:
            self.logger.error(f"Failed to generate Prophet forecast: {e}")
            return {}

    def run_comprehensive_analysis(self, 
                                  days_back: int = 30,
                                  include_keywords: bool = True,
                                  include_sources: bool = True,
                                  keyword_limit: int = 50) -> Dict[str, Any]:
        """Run comprehensive trend analysis across all metrics.
        
        Args:
            days_back: Number of days to analyze
            include_keywords: Whether to include keyword trend analysis
            include_sources: Whether to include source performance analysis
            keyword_limit: Maximum number of keywords to analyze
            
        Returns:
            Comprehensive analysis results
        """
        start_time = time.time()
        
        results = {
            'analysis_timestamp': datetime.utcnow(),
            'analysis_period_days': days_back,
            'analysis_duration_seconds': 0,
            'article_volume_analysis': {},
            'keyword_analyses': [],
            'source_analyses': []
        }
        
        try:
            # 1. Overall article volume trend analysis
            self.logger.info("Starting article volume trend analysis")
            volume_analysis = self.analyze_article_volume_trends(days_back=days_back)
            results['article_volume_analysis'] = volume_analysis

            # 2. Keyword trend analysis
            if include_keywords:
                self.logger.info("Starting keyword trend analysis")
                keyword_analyses = self.analyze_keyword_trends(days_back=days_back)
                # Limit to top keywords by total mentions
                keyword_analyses.sort(key=lambda x: x.get('total_mentions', 0), reverse=True)
                results['keyword_analyses'] = keyword_analyses[:keyword_limit]

            # 3. Source performance trend analysis
            if include_sources:
                self.logger.info("Starting source performance trend analysis")
                source_analyses = self.analyze_source_performance_trends(days_back=days_back)
                results['source_analyses'] = source_analyses

            # Calculate analysis duration
            results['analysis_duration_seconds'] = int(time.time() - start_time)
            
            self.logger.info(f"Comprehensive trend analysis completed in {results['analysis_duration_seconds']} seconds")
            
            return results

        except Exception as e:
            self.logger.error(f"Failed to run comprehensive trend analysis: {e}")
            results['analysis_duration_seconds'] = int(time.time() - start_time)
            results['error'] = str(e)
            return results