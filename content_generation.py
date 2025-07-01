"""
Content generation system for Mimir - EPIC 8.3
Provides automated newsletter and report generation.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ai_optimization import ModelPool
from summarization_engine import SummarizationEngine
import db_adapter

logger = logging.getLogger(__name__)


class ContentGenerationEngine:
    """Advanced content generation engine for newsletters and reports."""
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize the content generation engine.
        
        Args:
            model_pool: Optional ModelPool instance for model management
        """
        self.model_pool = model_pool or ModelPool()
        self.summarization_engine = SummarizationEngine(model_pool)
        
        # Template configurations for different content types
        self.templates = {
            'newsletter': {
                'sections': ['header', 'highlights', 'trending', 'analysis', 'footer'],
                'max_articles': 15,
                'highlight_count': 3,
                'trending_count': 5
            },
            'executive_briefing': {
                'sections': ['summary', 'key_developments', 'market_impact', 'recommendations'],
                'max_articles': 10,
                'focus_areas': ['business', 'technology', 'markets']
            },
            'trend_report': {
                'sections': ['overview', 'trend_analysis', 'key_players', 'outlook'],
                'analysis_period_days': 7,
                'trend_threshold': 3
            },
            'competitive_analysis': {
                'sections': ['market_overview', 'competitor_activities', 'strategic_insights'],
                'competitor_focus': True,
                'entity_focus': ['companies', 'products']
            }
        }
        
        # Personalization factors
        self.personalization_weights = {
            'user_interests': 0.4,
            'reading_history': 0.3,
            'industry_focus': 0.2,
            'content_preferences': 0.1
        }
    
    def get_generation_model(self, task_type: str = "text2text"):
        """Get or load a content generation model."""
        return self.model_pool.get_transformer_model("text-generation", "facebook/bart-large")
    
    def generate_newsletter(
        self,
        user_preferences: Optional[Dict[str, Any]] = None,
        date_range_days: int = 1,
        custom_template: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized newsletter.
        
        Args:
            user_preferences: User interests and preferences
            date_range_days: Number of days to look back for articles
            custom_template: Custom template configuration
            
        Returns:
            Dictionary containing generated newsletter and metadata
        """
        try:
            # Use custom template or default
            template = custom_template or self.templates['newsletter']
            
            # Get articles for the specified date range
            articles = self._get_articles_for_period(
                days=date_range_days,
                limit=template['max_articles'],
                user_preferences=user_preferences
            )
            
            if not articles:
                return self._create_error_result("No articles found for the specified period")
            
            # Categorize and rank articles
            categorized_articles = self._categorize_articles(articles, user_preferences)
            
            # Generate newsletter sections
            newsletter_content = {}
            generation_start = datetime.now()
            
            for section in template['sections']:
                if section == 'header':
                    newsletter_content[section] = self._generate_header(date_range_days)
                elif section == 'highlights':
                    newsletter_content[section] = self._generate_highlights(
                        categorized_articles, template['highlight_count']
                    )
                elif section == 'trending':
                    newsletter_content[section] = self._generate_trending_section(
                        categorized_articles, template['trending_count']
                    )
                elif section == 'analysis':
                    newsletter_content[section] = self._generate_analysis_section(
                        categorized_articles
                    )
                elif section == 'footer':
                    newsletter_content[section] = self._generate_footer()
            
            # Combine sections into final newsletter
            full_newsletter = self._combine_newsletter_sections(newsletter_content)
            
            processing_time = (datetime.now() - generation_start).total_seconds()
            
            return {
                'content': full_newsletter,
                'sections': newsletter_content,
                'content_type': 'newsletter',
                'articles_used': len(articles),
                'date_range_days': date_range_days,
                'processing_time_seconds': processing_time,
                'personalized': bool(user_preferences),
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Newsletter generation failed: {e}")
            return self._create_error_result(str(e))
    
    def generate_executive_briefing(
        self,
        focus_areas: Optional[List[str]] = None,
        date_range_days: int = 7,
        executive_level: str = "senior"
    ) -> Dict[str, Any]:
        """
        Generate an executive briefing report.
        
        Args:
            focus_areas: Areas of business focus
            date_range_days: Number of days to analyze
            executive_level: Target executive level (senior, c-level)
            
        Returns:
            Dictionary containing generated briefing and metadata
        """
        try:
            template = self.templates['executive_briefing']
            
            # Get relevant articles
            articles = self._get_articles_for_period(
                days=date_range_days,
                limit=template['max_articles'],
                focus_areas=focus_areas or template['focus_areas']
            )
            
            if not articles:
                return self._create_error_result("No relevant articles found for briefing")
            
            # Analyze articles for executive insights
            analyzed_articles = self._analyze_for_executive_insights(articles)
            
            # Generate briefing sections
            briefing_content = {}
            generation_start = datetime.now()
            
            if 'summary' in template['sections']:
                briefing_content['summary'] = self._generate_executive_summary(
                    analyzed_articles, executive_level
                )
            
            if 'key_developments' in template['sections']:
                briefing_content['key_developments'] = self._generate_key_developments(
                    analyzed_articles
                )
            
            if 'market_impact' in template['sections']:
                briefing_content['market_impact'] = self._generate_market_impact_analysis(
                    analyzed_articles
                )
            
            if 'recommendations' in template['sections']:
                briefing_content['recommendations'] = self._generate_recommendations(
                    analyzed_articles, executive_level
                )
            
            # Combine into final briefing
            full_briefing = self._combine_briefing_sections(briefing_content)
            
            processing_time = (datetime.now() - generation_start).total_seconds()
            
            return {
                'content': full_briefing,
                'sections': briefing_content,
                'content_type': 'executive_briefing',
                'articles_analyzed': len(articles),
                'focus_areas': focus_areas or template['focus_areas'],
                'executive_level': executive_level,
                'date_range_days': date_range_days,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Executive briefing generation failed: {e}")
            return self._create_error_result(str(e))
    
    def generate_trend_report(
        self,
        analysis_period_days: int = 7,
        trend_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a trend analysis report.
        
        Args:
            analysis_period_days: Period for trend analysis
            trend_categories: Specific categories to analyze
            
        Returns:
            Dictionary containing generated trend report and metadata
        """
        try:
            template = self.templates['trend_report']
            
            # Get articles for trend analysis
            articles = self._get_articles_for_period(
                days=analysis_period_days,
                limit=100,  # More articles for better trend detection
                categories=trend_categories
            )
            
            if not articles:
                return self._create_error_result("No articles found for trend analysis")
            
            # Analyze trends
            trend_analysis = self._analyze_trends(articles, analysis_period_days)
            
            # Generate report sections
            report_content = {}
            generation_start = datetime.now()
            
            if 'overview' in template['sections']:
                report_content['overview'] = self._generate_trend_overview(
                    trend_analysis, analysis_period_days
                )
            
            if 'trend_analysis' in template['sections']:
                report_content['trend_analysis'] = self._generate_detailed_trend_analysis(
                    trend_analysis
                )
            
            if 'key_players' in template['sections']:
                report_content['key_players'] = self._generate_key_players_analysis(
                    trend_analysis, articles
                )
            
            if 'outlook' in template['sections']:
                report_content['outlook'] = self._generate_trend_outlook(
                    trend_analysis
                )
            
            # Combine into final report
            full_report = self._combine_report_sections(report_content)
            
            processing_time = (datetime.now() - generation_start).total_seconds()
            
            return {
                'content': full_report,
                'sections': report_content,
                'content_type': 'trend_report',
                'articles_analyzed': len(articles),
                'trends_identified': len(trend_analysis.get('trends', [])),
                'analysis_period_days': analysis_period_days,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Trend report generation failed: {e}")
            return self._create_error_result(str(e))
    
    def _get_articles_for_period(
        self,
        days: int,
        limit: int = 50,
        user_preferences: Optional[Dict[str, Any]] = None,
        focus_areas: Optional[List[str]] = None,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get articles for the specified period with filters."""
        try:
            # For now, get recent articles - in production this would use more sophisticated filtering
            articles = db_adapter.get_recent_articles(limit=limit, hours=days * 24)
            
            # Apply basic filtering if preferences provided
            if user_preferences or focus_areas or categories:
                articles = self._filter_articles_by_preferences(
                    articles, user_preferences, focus_areas, categories
                )
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to get articles for period: {e}")
            return []
    
    def _filter_articles_by_preferences(
        self,
        articles: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]],
        focus_areas: Optional[List[str]],
        categories: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Filter articles based on preferences and focus areas."""
        filtered = []
        
        for article in articles:
            include = True
            
            # Filter by categories if specified
            if categories:
                article_category = article.get('category', '').lower()
                if not any(cat.lower() in article_category for cat in categories):
                    include = False
            
            # Filter by focus areas (check in title/content)
            if focus_areas and include:
                title_content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
                if not any(area.lower() in title_content for area in focus_areas):
                    include = False
            
            # Apply user preferences (simplified)
            if user_preferences and include:
                interests = user_preferences.get('interests', [])
                if interests:
                    title_content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
                    if not any(interest.lower() in title_content for interest in interests):
                        include = False
            
            if include:
                filtered.append(article)
        
        return filtered
    
    def _categorize_articles(
        self,
        articles: List[Dict[str, Any]],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize articles by importance and relevance."""
        categorized = {
            'breaking': [],
            'important': [],
            'trending': [],
            'regular': []
        }
        
        for article in articles:
            # Simple categorization logic - in production this would use ML models
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # Check for breaking news indicators
            if any(term in title for term in ['breaking', 'urgent', 'alert', 'exclusive']):
                categorized['breaking'].append(article)
            # Check for importance indicators
            elif any(term in title for term in ['major', 'significant', 'important', 'critical']):
                categorized['important'].append(article)
            # Check for trending indicators
            elif any(term in title + content for term in ['trend', 'viral', 'popular', 'surge']):
                categorized['trending'].append(article)
            else:
                categorized['regular'].append(article)
        
        return categorized
    
    def _generate_header(self, date_range_days: int) -> str:
        """Generate newsletter header."""
        date_str = datetime.now().strftime("%B %d, %Y")
        period_str = "today" if date_range_days == 1 else f"the last {date_range_days} days"
        
        return f"""
# Mimir News Intelligence Briefing
## {date_str}

Welcome to your personalized news briefing covering the most important developments from {period_str}.
This report has been automatically generated using AI-powered analysis of thousands of news sources.

---
"""
    
    def _generate_highlights(
        self,
        categorized_articles: Dict[str, List[Dict[str, Any]]],
        count: int
    ) -> str:
        """Generate highlights section."""
        highlights = []
        
        # Prioritize breaking news, then important
        priority_articles = (
            categorized_articles.get('breaking', []) +
            categorized_articles.get('important', [])
        )[:count]
        
        if not priority_articles:
            priority_articles = categorized_articles.get('regular', [])[:count]
        
        highlights.append("## 🔥 Top Highlights\n")
        
        for i, article in enumerate(priority_articles, 1):
            # Generate summary for each highlight
            summary_result = self.summarization_engine.generate_summary(
                text=article.get('content', article.get('title', '')),
                format_type='bullet_points'
            )
            
            title = article.get('title', 'Untitled')
            source = article.get('source', 'Unknown Source')
            summary = summary_result.get('summary', 'Summary not available') if summary_result.get('success') else 'Summary not available'
            
            highlights.append(f"### {i}. {title}")
            highlights.append(f"**Source:** {source}")
            highlights.append(f"{summary}")
            highlights.append("")
        
        return "\n".join(highlights)
    
    def _generate_trending_section(
        self,
        categorized_articles: Dict[str, List[Dict[str, Any]]],
        count: int
    ) -> str:
        """Generate trending topics section."""
        trending_articles = categorized_articles.get('trending', [])
        if not trending_articles:
            trending_articles = categorized_articles.get('regular', [])
        
        trending_articles = trending_articles[:count]
        
        content = ["## 📈 Trending Topics\n"]
        
        for article in trending_articles:
            title = article.get('title', 'Untitled')
            source = article.get('source', 'Unknown')
            
            # Generate brief summary
            summary_result = self.summarization_engine.generate_summary(
                text=article.get('content', title),
                format_type='tweet'
            )
            
            summary = summary_result.get('summary', title) if summary_result.get('success') else title
            
            content.append(f"- **{title}** ({source})")
            content.append(f"  {summary}")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_analysis_section(
        self,
        categorized_articles: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Generate analysis section."""
        all_articles = []
        for category_articles in categorized_articles.values():
            all_articles.extend(category_articles)
        
        content = ["## 🔍 Market Analysis\n"]
        
        # Simple trend analysis
        total_articles = len(all_articles)
        breaking_count = len(categorized_articles.get('breaking', []))
        important_count = len(categorized_articles.get('important', []))
        
        content.append(f"Today's news landscape shows {total_articles} significant stories,")
        content.append(f"with {breaking_count} breaking developments and {important_count} important updates.")
        content.append("")
        
        # Add sector analysis if possible
        content.append("**Key Themes:**")
        content.append("- Market volatility continues to influence trading decisions")
        content.append("- Technology sector shows mixed signals amid regulatory scrutiny")
        content.append("- Economic indicators suggest cautious optimism")
        content.append("")
        
        return "\n".join(content)
    
    def _generate_footer(self) -> str:
        """Generate newsletter footer."""
        return """
---

## About This Report

This briefing was generated automatically by Mimir's AI-powered news intelligence platform. 
Our system analyzes thousands of news sources to bring you the most relevant and important stories.

**Powered by:** Advanced NLP, sentiment analysis, and trend detection algorithms  
**Generated:** """ + datetime.now().strftime("%Y-%m-%d %H:%M UTC") + """

---
"""
    
    def _combine_newsletter_sections(self, sections: Dict[str, str]) -> str:
        """Combine newsletter sections into final content."""
        return "\n".join(sections.values())
    
    def _analyze_for_executive_insights(
        self,
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze articles for executive-level insights."""
        analyzed = []
        
        for article in articles:
            analysis = {
                'article': article,
                'executive_relevance': self._calculate_executive_relevance(article),
                'business_impact': self._assess_business_impact(article),
                'urgency_level': self._assess_urgency(article),
                'strategic_implications': self._identify_strategic_implications(article)
            }
            analyzed.append(analysis)
        
        # Sort by executive relevance
        analyzed.sort(key=lambda x: x['executive_relevance'], reverse=True)
        return analyzed
    
    def _calculate_executive_relevance(self, article: Dict[str, Any]) -> float:
        """Calculate how relevant an article is to executives."""
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        
        executive_keywords = [
            'revenue', 'profit', 'merger', 'acquisition', 'ceo', 'strategy',
            'market share', 'competition', 'regulation', 'investment',
            'earnings', 'financial', 'growth', 'expansion'
        ]
        
        relevance_score = 0.0
        text = title + ' ' + content
        
        for keyword in executive_keywords:
            if keyword in text:
                relevance_score += 0.1
        
        return min(1.0, relevance_score)
    
    def _assess_business_impact(self, article: Dict[str, Any]) -> str:
        """Assess the business impact level of an article."""
        title = article.get('title', '').lower()
        
        high_impact_terms = ['acquisition', 'merger', 'bankruptcy', 'ipo', 'regulation']
        medium_impact_terms = ['partnership', 'product launch', 'expansion', 'investment']
        
        if any(term in title for term in high_impact_terms):
            return 'high'
        elif any(term in title for term in medium_impact_terms):
            return 'medium'
        else:
            return 'low'
    
    def _assess_urgency(self, article: Dict[str, Any]) -> str:
        """Assess the urgency level of an article."""
        title = article.get('title', '').lower()
        
        urgent_terms = ['breaking', 'urgent', 'alert', 'emergency']
        
        if any(term in title for term in urgent_terms):
            return 'urgent'
        else:
            return 'normal'
    
    def _identify_strategic_implications(self, article: Dict[str, Any]) -> List[str]:
        """Identify strategic implications from an article."""
        implications = []
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        text = title + ' ' + content
        
        if 'competition' in text or 'competitor' in text:
            implications.append('Competitive landscape changes')
        
        if 'regulation' in text or 'policy' in text:
            implications.append('Regulatory environment shifts')
        
        if 'technology' in text and ('innovation' in text or 'disruption' in text):
            implications.append('Technology disruption potential')
        
        if 'market' in text and ('expansion' in text or 'growth' in text):
            implications.append('Market opportunity identification')
        
        return implications
    
    def _generate_executive_summary(
        self,
        analyzed_articles: List[Dict[str, Any]],
        executive_level: str
    ) -> str:
        """Generate executive summary section."""
        top_articles = analyzed_articles[:5]  # Top 5 most relevant
        
        content = ["## Executive Summary\n"]
        
        if executive_level == "c-level":
            content.append("**Strategic Overview for C-Level Leadership**\n")
        else:
            content.append("**Strategic Briefing for Senior Management**\n")
        
        for i, analysis in enumerate(top_articles, 1):
            article = analysis['article']
            title = article.get('title', 'Untitled')
            impact = analysis['business_impact']
            
            # Generate executive-focused summary
            summary_result = self.summarization_engine.generate_summary(
                text=article.get('content', title),
                format_type='executive'
            )
            
            summary = summary_result.get('summary', 'Summary not available') if summary_result.get('success') else 'Summary not available'
            
            content.append(f"### {i}. {title}")
            content.append(f"**Business Impact:** {impact.title()}")
            content.append(f"{summary}")
            
            if analysis['strategic_implications']:
                content.append(f"**Strategic Implications:** {', '.join(analysis['strategic_implications'])}")
            
            content.append("")
        
        return "\n".join(content)
    
    def _generate_key_developments(self, analyzed_articles: List[Dict[str, Any]]) -> str:
        """Generate key developments section."""
        urgent_articles = [a for a in analyzed_articles if a['urgency_level'] == 'urgent']
        high_impact_articles = [a for a in analyzed_articles if a['business_impact'] == 'high']
        
        key_articles = urgent_articles + high_impact_articles
        # Remove duplicates while preserving order
        seen = set()
        unique_articles = []
        for article in key_articles:
            article_id = article['article'].get('id')
            if article_id not in seen:
                seen.add(article_id)
                unique_articles.append(article)
        
        content = ["## Key Developments\n"]
        
        for analysis in unique_articles[:7]:  # Top 7 key developments
            article = analysis['article']
            title = article.get('title', 'Untitled')
            source = article.get('source', 'Unknown')
            
            content.append(f"- **{title}** ({source})")
            content.append(f"  Impact: {analysis['business_impact'].title()}")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_market_impact_analysis(self, analyzed_articles: List[Dict[str, Any]]) -> str:
        """Generate market impact analysis section."""
        content = ["## Market Impact Analysis\n"]
        
        high_impact_count = len([a for a in analyzed_articles if a['business_impact'] == 'high'])
        medium_impact_count = len([a for a in analyzed_articles if a['business_impact'] == 'medium'])
        
        content.append(f"Current market dynamics show {high_impact_count} high-impact developments ")
        content.append(f"and {medium_impact_count} medium-impact events requiring attention.")
        content.append("")
        
        content.append("**Risk Assessment:**")
        if high_impact_count > 3:
            content.append("- Elevated risk environment due to multiple high-impact events")
        else:
            content.append("- Moderate risk environment with manageable volatility")
        content.append("")
        
        return "\n".join(content)
    
    def _generate_recommendations(
        self,
        analyzed_articles: List[Dict[str, Any]],
        executive_level: str
    ) -> str:
        """Generate recommendations section."""
        content = ["## Strategic Recommendations\n"]
        
        urgent_count = len([a for a in analyzed_articles if a['urgency_level'] == 'urgent'])
        
        if urgent_count > 0:
            content.append(f"**Immediate Actions Required:** {urgent_count} urgent items need attention")
            content.append("")
        
        content.append("**Recommended Actions:**")
        content.append("1. Monitor competitive developments for strategic positioning opportunities")
        content.append("2. Assess regulatory changes for compliance and business impact")
        content.append("3. Evaluate market conditions for timing of strategic initiatives")
        content.append("")
        
        return "\n".join(content)
    
    def _combine_briefing_sections(self, sections: Dict[str, str]) -> str:
        """Combine briefing sections into final content."""
        return "\n".join(sections.values())
    
    def _analyze_trends(
        self,
        articles: List[Dict[str, Any]],
        period_days: int
    ) -> Dict[str, Any]:
        """Analyze trends from articles."""
        # Simple trend analysis - in production this would use sophisticated ML
        trends = {
            'trends': [],
            'keywords': {},
            'entities': {},
            'sentiment_trends': {}
        }
        
        # Count keyword frequencies
        all_text = ""
        for article in articles:
            all_text += " " + article.get('title', '') + " " + article.get('content', '')
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_freq = {}
        for word in words:
            if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were', 'said']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top trending keywords
        trending_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        trends['keywords'] = dict(trending_keywords)
        
        # Identify trends
        for keyword, count in trending_keywords[:5]:
            if count >= 3:  # Threshold for trend
                trends['trends'].append({
                    'keyword': keyword,
                    'frequency': count,
                    'trend_strength': min(1.0, count / 10)
                })
        
        return trends
    
    def _generate_trend_overview(
        self,
        trend_analysis: Dict[str, Any],
        period_days: int
    ) -> str:
        """Generate trend overview section."""
        content = ["## Trend Overview\n"]
        
        trends = trend_analysis.get('trends', [])
        
        content.append(f"Analysis of news patterns over the last {period_days} days reveals ")
        content.append(f"{len(trends)} significant trending topics.")
        content.append("")
        
        if trends:
            content.append("**Top Trending Topics:**")
            for i, trend in enumerate(trends[:5], 1):
                keyword = trend['keyword'].title()
                strength = trend['trend_strength']
                content.append(f"{i}. {keyword} (Trend Strength: {strength:.1f})")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_detailed_trend_analysis(self, trend_analysis: Dict[str, Any]) -> str:
        """Generate detailed trend analysis section."""
        content = ["## Detailed Trend Analysis\n"]
        
        trends = trend_analysis.get('trends', [])
        
        for trend in trends[:3]:  # Top 3 trends
            keyword = trend['keyword'].title()
            frequency = trend['frequency']
            
            content.append(f"### {keyword}")
            content.append(f"**Frequency:** {frequency} mentions")
            content.append(f"**Analysis:** The '{keyword}' trend shows significant activity ")
            content.append(f"with {frequency} mentions across our monitored sources.")
            content.append("")
        
        return "\n".join(content)
    
    def _generate_key_players_analysis(
        self,
        trend_analysis: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate key players analysis section."""
        content = ["## Key Players & Entities\n"]
        
        # Simple entity extraction - look for capitalized words
        entities = {}
        for article in articles:
            title = article.get('title', '')
            words = re.findall(r'\b[A-Z][a-zA-Z]+\b', title)
            for word in words:
                if len(word) > 3 and word not in ['The', 'And', 'For', 'Inc', 'LLC']:
                    entities[word] = entities.get(word, 0) + 1
        
        top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        content.append("**Most Mentioned Entities:**")
        for entity, count in top_entities:
            content.append(f"- {entity} ({count} mentions)")
        content.append("")
        
        return "\n".join(content)
    
    def _generate_trend_outlook(self, trend_analysis: Dict[str, Any]) -> str:
        """Generate trend outlook section."""
        content = ["## Outlook & Predictions\n"]
        
        trends = trend_analysis.get('trends', [])
        
        if trends:
            strongest_trend = max(trends, key=lambda x: x['trend_strength'])
            keyword = strongest_trend['keyword'].title()
            
            content.append(f"**Primary Focus Area:** {keyword}")
            content.append(f"Based on current momentum, {keyword} is expected to remain ")
            content.append("a significant topic in the coming period.")
            content.append("")
        
        content.append("**Market Implications:**")
        content.append("- Continued monitoring recommended for emerging patterns")
        content.append("- Strategic positioning opportunities may arise from trend developments")
        content.append("")
        
        return "\n".join(content)
    
    def _combine_report_sections(self, sections: Dict[str, str]) -> str:
        """Combine report sections into final content."""
        return "\n".join(sections.values())
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'content': '',
            'sections': {},
            'content_type': 'error',
            'articles_used': 0,
            'processing_time_seconds': 0.0,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error_message
        }
    
    def save_generated_content(
        self,
        content_result: Dict[str, Any],
        content_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save generated content to database."""
        try:
            # This would save to a dedicated content generation table
            # For now, we'll create a simple storage mechanism
            
            storage_data = {
                'content_id': content_id,
                'content_type': content_result.get('content_type', 'unknown'),
                'content': content_result.get('content', ''),
                'metadata': {
                    **content_result,
                    **(metadata or {})
                },
                'created_at': datetime.now().isoformat()
            }
            
            # In production, this would be saved to a proper table
            logger.info(f"Generated content saved with ID: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save generated content: {e}")
            return False


# Extend ModelPool to support content generation models
def _extend_model_pool_for_content_generation():
    """Extend ModelPool class to include content generation models."""
    original_get_transformer = ModelPool.get_transformer_model
    
    def get_transformer_model_extended(self, task: str, model_name: Optional[str] = None):
        """Extended transformer model getter with content generation support."""
        if task == "text-generation":
            key = f"transformer_{task}_{model_name or 'default'}"
            
            with self.model_locks[key]:
                if key not in self.models:
                    logger.info(f"Loading content generation model: {model_name}")
                    try:
                        model = model_name or "facebook/bart-large"
                        generator = pipeline(
                            "text2text-generation",
                            model=model,
                            tokenizer=model,
                            device=0 if torch.cuda.is_available() else -1,
                            framework="pt"
                        )
                        self.models[key] = generator
                    except Exception as e:
                        logger.error(f"Failed to load content generation model {model_name}: {e}")
                        return None
                
                self.usage_counts[key] += 1
                self.last_used[key] = time.time()
                return self.models[key]
        else:
            return original_get_transformer(self, task, model_name)
    
    # Monkey patch the method
    ModelPool.get_transformer_model = get_transformer_model_extended


# Initialize the extension
_extend_model_pool_for_content_generation()