#!/usr/bin/env python3
"""
Trend Prediction Module for Mimir
Analyzes patterns across multiple sources to predict industry trends
"""

import json
import sqlite3
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import logging
import re

# For trend analysis
try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None

# For sentiment analysis
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# For keyword extraction
try:
    import yake
except ImportError:
    yake = None

logger = logging.getLogger(__name__)

class TrendPredictor:
    """Analyzes news patterns to predict industry trends"""
    
    def __init__(self, db_path='mimir_news.db'):
        self.db_path = db_path
        self.pytrends = TrendReq() if TrendReq else None
        
    def extract_trending_keywords(self, days_back: int = 30, min_frequency: int = 3) -> List[Tuple[str, int]]:
        """Extract trending keywords from recent articles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent articles
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        cursor.execute("""
            SELECT title, content_summary 
            FROM articles 
            WHERE publication_date > ? 
            ORDER BY publication_date DESC
        """, (cutoff_date,))
        
        articles = cursor.fetchall()
        conn.close()
        
        # Extract keywords from all articles
        all_keywords = []
        
        if yake:
            kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7)
            
            for title, content in articles:
                text = f"{title} {content or ''}"
                keywords = kw_extractor.extract_keywords(text)
                all_keywords.extend([kw[0].lower() for kw in keywords[:10]])
        else:
            # Fallback: simple word frequency
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            
            for title, content in articles:
                text = f"{title} {content or ''}".lower()
                words = re.findall(r'\b\w+\b', text)
                meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
                all_keywords.extend(meaningful_words)
        
        # Count frequencies
        keyword_counts = Counter(all_keywords)
        
        # Filter by minimum frequency
        trending = [(kw, count) for kw, count in keyword_counts.most_common(50) 
                   if count >= min_frequency]
        
        return trending
    
    def analyze_keyword_velocity(self, keyword: str, days: int = 30) -> Dict:
        """Analyze how quickly a keyword is gaining mentions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = {
            'keyword': keyword,
            'total_mentions': 0,
            'daily_mentions': {},
            'velocity': 0,
            'acceleration': 0
        }
        
        # Get mentions by day
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).date()
            next_date = date + timedelta(days=1)
            
            cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE (title LIKE ? OR content_summary LIKE ?)
                AND publication_date >= ? AND publication_date < ?
            """, (f'%{keyword}%', f'%{keyword}%', date.isoformat(), next_date.isoformat()))
            
            count = cursor.fetchone()[0]
            results['daily_mentions'][date.isoformat()] = count
            results['total_mentions'] += count
        
        conn.close()
        
        # Calculate velocity (change in mentions over time)
        daily_counts = list(results['daily_mentions'].values())
        if len(daily_counts) > 7:
            # Compare last week to previous week
            last_week = sum(daily_counts[:7])
            prev_week = sum(daily_counts[7:14])
            if prev_week > 0:
                results['velocity'] = ((last_week - prev_week) / prev_week) * 100
        
        return results
    
    def get_google_trends(self, keywords: List[str], timeframe: str = 'today 3-m') -> Dict:
        """Get Google Trends data for keywords"""
        if not self.pytrends:
            return {}
        
        try:
            # Build payload
            self.pytrends.build_payload(keywords[:5], timeframe=timeframe)
            
            # Get interest over time
            interest = self.pytrends.interest_over_time()
            
            # Get related queries
            related = self.pytrends.related_queries()
            
            return {
                'interest_over_time': interest.to_dict() if not interest.empty else {},
                'related_queries': related
            }
        except Exception as e:
            logger.error(f"Error fetching Google Trends: {e}")
            return {}
    
    def analyze_sentiment_trends(self, days_back: int = 30) -> Dict:
        """Analyze sentiment trends in recent articles"""
        if not TextBlob:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        cursor.execute("""
            SELECT title, content_summary, publication_date, source
            FROM articles 
            WHERE publication_date > ? 
            ORDER BY publication_date DESC
        """, (cutoff_date,))
        
        articles = cursor.fetchall()
        conn.close()
        
        # Analyze sentiment by day and source
        daily_sentiment = defaultdict(list)
        source_sentiment = defaultdict(list)
        
        for title, content, pub_date, source in articles:
            text = f"{title} {content or ''}"
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            date = pub_date.split('T')[0]
            daily_sentiment[date].append(sentiment)
            source_sentiment[source].append(sentiment)
        
        # Calculate averages
        results = {
            'daily_average': {},
            'source_average': {},
            'overall_trend': 'neutral'
        }
        
        for date, sentiments in daily_sentiment.items():
            results['daily_average'][date] = sum(sentiments) / len(sentiments)
        
        for source, sentiments in source_sentiment.items():
            results['source_average'][source] = sum(sentiments) / len(sentiments)
        
        # Determine overall trend
        if results['daily_average']:
            recent_dates = sorted(results['daily_average'].keys())[-7:]
            recent_avg = sum(results['daily_average'][d] for d in recent_dates) / len(recent_dates)
            
            if recent_avg > 0.1:
                results['overall_trend'] = 'positive'
            elif recent_avg < -0.1:
                results['overall_trend'] = 'negative'
        
        return results
    
    def predict_emerging_topics(self, min_velocity: float = 50.0) -> List[Dict]:
        """Predict topics that are likely to become major trends"""
        predictions = []
        
        # Get trending keywords
        trending_keywords = self.extract_trending_keywords(days_back=14, min_frequency=2)
        
        # Analyze each keyword's velocity
        for keyword, frequency in trending_keywords[:20]:
            velocity_data = self.analyze_keyword_velocity(keyword, days=14)
            
            if velocity_data['velocity'] >= min_velocity:
                prediction = {
                    'topic': keyword,
                    'current_frequency': frequency,
                    'velocity': velocity_data['velocity'],
                    'confidence': min(velocity_data['velocity'] / 100, 1.0),
                    'prediction': f"{keyword} is gaining {velocity_data['velocity']:.0f}% momentum"
                }
                predictions.append(prediction)
        
        # Sort by velocity
        predictions.sort(key=lambda x: x['velocity'], reverse=True)
        
        return predictions
    
    def generate_trend_report(self) -> Dict:
        """Generate comprehensive trend analysis report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'trending_keywords': self.extract_trending_keywords(days_back=7),
            'emerging_topics': self.predict_emerging_topics(),
            'sentiment_analysis': self.analyze_sentiment_trends(days_back=7),
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if report['emerging_topics']:
            top_topic = report['emerging_topics'][0]
            report['recommendations'].append(
                f"Focus on '{top_topic['topic']}' - showing {top_topic['velocity']:.0f}% growth"
            )
        
        if report['sentiment_analysis'].get('overall_trend') == 'negative':
            report['recommendations'].append(
                "Industry sentiment is negative - consider defensive positioning"
            )
        
        return report

def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    predictor = TrendPredictor()
    
    print("🔮 Mimir Trend Prediction Analysis\n")
    
    # Get trending keywords
    print("📈 Trending Keywords (Last 30 Days):")
    trending = predictor.extract_trending_keywords()
    for keyword, count in trending[:10]:
        print(f"  - {keyword}: {count} mentions")
    
    # Analyze emerging topics
    print("\n🚀 Emerging Topics:")
    emerging = predictor.predict_emerging_topics()
    for topic in emerging[:5]:
        print(f"  - {topic['topic']}: {topic['velocity']:.0f}% growth velocity")
    
    # Generate full report
    print("\n📊 Generating comprehensive report...")
    report = predictor.generate_trend_report()
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Save report
    with open('trend_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n✅ Full report saved to trend_report.json")

if __name__ == "__main__":
    main()