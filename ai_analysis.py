"""
Main AI analysis module for Mimir.
Integrates all NLP components and provides a unified analysis interface.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Database adapters
import db_adapter
from ai_optimization import (BatchProcessor, EntityExtractor, EntityLinker,
                             IndustryClassifier, KeywordExtractor,
                             MemoryOptimizedPipeline, ModelPool,
                             SentimentAnalyzer, StreamingProcessor,
                             TextPreprocessor, TopicModeler,
                             analyze_article_sentiment, analyze_article_topics,
                             batch_preprocess_articles,
                             classify_article_industry, entity_extraction,
                             entity_linking, extract_entities_from_article,
                             extract_keywords_from_article, from, import,
                             industry_classification, keyword_extraction,
                             link_article_entities, nlp_preprocessing,
                             optimize_model_loading, sentiment_analysis,
                             topic_modeling)

logger = logging.getLogger(__name__)


class AIAnalysisEngine:
    """Main AI analysis engine that coordinates all NLP tasks."""
    
    def __init__(self, use_optimization: bool = True, max_memory_mb: int = 1024):
        """
        Initialize the AI analysis engine with all components.
        
        Args:
            use_optimization: Whether to use memory optimization
            max_memory_mb: Maximum memory usage in MB
        """
        logger.info("Initializing AI Analysis Engine...")
        
        # Initialize optimization components
        self.use_optimization = use_optimization
        if use_optimization:
            self.model_pool = ModelPool()
            self.batch_processor = BatchProcessor()
            self.memory_pipeline = MemoryOptimizedPipeline(max_memory_mb)
            # Pre-load common models
            optimize_model_loading()
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.keyword_extractor = KeywordExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.industry_classifier = IndustryClassifier()
        self.entity_linker = EntityLinker()
        
        # Database connection
        self.db = db_adapter.get_connection()
        
        logger.info("AI Analysis Engine initialized successfully")
    
    def analyze_article(self, article: Dict, 
                       components: Optional[List[str]] = None) -> Dict:
        """
        Perform comprehensive AI analysis on a single article.
        
        Args:
            article: Article dictionary with at least 'content' and optionally 'title'
            components: List of components to run (defaults to all)
                       Options: ['preprocess', 'entities', 'keywords', 'sentiment']
        
        Returns:
            Dictionary with all analysis results
        """
        start_time = time.time()
        
        # Default to all components
        if not components:
            components = ['preprocess', 'entities', 'keywords', 'sentiment', 'industry', 'topics']
        
        results = {
            'article_id': article.get('id', 'unknown'),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'components_run': components,
            'errors': []
        }
        
        try:
            # Step 1: Preprocessing (required for other components)
            if 'preprocess' in components or len(components) > 1:
                logger.info(f"Preprocessing article {article.get('id', 'unknown')}")
                preprocessed = self.preprocessor.preprocess_article(article)
                results['preprocessing'] = preprocessed['preprocessed']
                article['preprocessed'] = preprocessed['preprocessed']
            
            # Step 2: Entity extraction
            if 'entities' in components:
                logger.info(f"Extracting entities from article {article.get('id', 'unknown')}")
                try:
                    entity_results = extract_entities_from_article(article)
                    results['entities'] = entity_results
                except Exception as e:
                    logger.error(f"Entity extraction failed: {e}")
                    results['errors'].append(f"Entity extraction error: {str(e)}")
            
            # Step 3: Keyword extraction
            if 'keywords' in components:
                logger.info(f"Extracting keywords from article {article.get('id', 'unknown')}")
                try:
                    keyword_results = extract_keywords_from_article(article)
                    results['keywords'] = keyword_results
                except Exception as e:
                    logger.error(f"Keyword extraction failed: {e}")
                    results['errors'].append(f"Keyword extraction error: {str(e)}")
            
            # Step 4: Sentiment analysis
            if 'sentiment' in components:
                logger.info(f"Analyzing sentiment for article {article.get('id', 'unknown')}")
                try:
                    sentiment_results = analyze_article_sentiment(article)
                    results['sentiment'] = sentiment_results
                except Exception as e:
                    logger.error(f"Sentiment analysis failed: {e}")
                    results['errors'].append(f"Sentiment analysis error: {str(e)}")
            
            # Step 5: Industry classification
            if 'industry' in components:
                logger.info(f"Classifying industry for article {article.get('id', 'unknown')}")
                try:
                    industry_results = classify_article_industry(article)
                    results['industry'] = industry_results
                except Exception as e:
                    logger.error(f"Industry classification failed: {e}")
                    results['errors'].append(f"Industry classification error: {str(e)}")
            
            # Step 6: Topic modeling (if analyzing multiple articles)
            if 'topics' in components and isinstance(article, list):
                logger.info(f"Analyzing topics across {len(article)} articles")
                try:
                    topic_results = analyze_article_topics(article)
                    results['topics'] = topic_results
                except Exception as e:
                    logger.error(f"Topic modeling failed: {e}")
                    results['errors'].append(f"Topic modeling error: {str(e)}")
            
            # Step 7: Entity linking (optional, resource-intensive)
            if 'entity_linking' in components and 'entities' in results:
                logger.info(f"Linking entities for article {article.get('id', 'unknown')}")
                try:
                    linking_results = link_article_entities(
                        article, 
                        results['entities'].get('entities', {})
                    )
                    results['entity_linking'] = linking_results
                except Exception as e:
                    logger.error(f"Entity linking failed: {e}")
                    results['errors'].append(f"Entity linking error: {str(e)}")
            
            # Calculate processing time
            results['processing_time'] = time.time() - start_time
            
            # Generate summary insights
            results['insights'] = self._generate_insights(results)
            
            return results
            
        except Exception as e:
            logger.error(f"AI analysis failed for article {article.get('id', 'unknown')}: {e}")
            results['errors'].append(f"General analysis error: {str(e)}")
            results['processing_time'] = time.time() - start_time
            return results
    
    def _generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from analysis results."""
        insights = {
            'summary': '',
            'key_topics': [],
            'main_entities': [],
            'sentiment_summary': '',
            'relevance_indicators': []
        }
        
        try:
            # Extract key topics from keywords
            if 'keywords' in analysis_results:
                keywords = analysis_results['keywords'].get('keywords', {})
                if 'ensemble' in keywords:
                    insights['key_topics'] = [
                        kw[0] for kw in keywords['ensemble'][:5]
                    ]
            
            # Extract main entities
            if 'entities' in analysis_results:
                entities = analysis_results['entities'].get('entities', {})
                # Get most mentioned organizations and people
                for entity_type in ['organization', 'person']:
                    if entity_type in entities:
                        entity_list = entities[entity_type]
                        if entity_list:
                            insights['main_entities'].extend([
                                e['normalized'] for e in entity_list[:3]
                            ])
            
            # Summarize sentiment
            if 'sentiment' in analysis_results:
                sentiment = analysis_results['sentiment']
                sentiment_label = sentiment.get('overall_sentiment', 'neutral')
                confidence = sentiment.get('confidence', 0)
                impact = sentiment.get('market_impact', 'unknown')
                
                insights['sentiment_summary'] = (
                    f"{sentiment_label} sentiment "
                    f"(confidence: {confidence:.0%}, "
                    f"market impact: {impact})"
                )
            
            # Add industry classification
            if 'industry' in analysis_results:
                industry = analysis_results['industry']
                insights['primary_industry'] = industry.get('primary_industry')
                insights['industry_confidence'] = industry.get('confidence', 0)
            
            # Determine relevance indicators
            if 'preprocessing' in analysis_results:
                word_count = analysis_results['preprocessing'].get('word_count', 0)
                if word_count > 500:
                    insights['relevance_indicators'].append('detailed_article')
                elif word_count < 100:
                    insights['relevance_indicators'].append('brief_update')
            
            # Generate summary
            insights['summary'] = self._create_summary(insights)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _create_summary(self, insights: Dict[str, Any]) -> str:
        """Create a human-readable summary from insights."""
        parts = []
        
        if insights['key_topics']:
            topics = ', '.join(insights['key_topics'][:3])
            parts.append(f"Key topics: {topics}")
        
        if insights['main_entities']:
            entities = ', '.join(insights['main_entities'][:3])
            parts.append(f"Main entities: {entities}")
        
        if insights['sentiment_summary']:
            parts.append(insights['sentiment_summary'])
        
        return '. '.join(parts) if parts else "No significant insights extracted."
    
    def analyze_batch(self, articles: List[Dict], 
                     components: Optional[List[str]] = None,
                     save_to_db: bool = True,
                     use_optimization: Optional[bool] = None) -> List[Dict]:
        """
        Analyze multiple articles in batch.
        
        Args:
            articles: List of article dictionaries
            components: Components to run (defaults to all)
            save_to_db: Whether to save results to database
            use_optimization: Override optimization setting
            
        Returns:
            List of analysis results
        """
        # Use optimization if enabled and not overridden
        use_opt = use_optimization if use_optimization is not None else self.use_optimization
        
        if use_opt and hasattr(self, 'memory_pipeline'):
            return self._analyze_batch_optimized(articles, components, save_to_db)
        else:
            return self._analyze_batch_simple(articles, components, save_to_db)
    
    def _analyze_batch_simple(self, articles: List[Dict], 
                             components: Optional[List[str]],
                             save_to_db: bool) -> List[Dict]:
        """Simple batch processing without optimization."""
        results = []
        total = len(articles)
        
        logger.info(f"Starting batch analysis of {total} articles")
        
        for i, article in enumerate(articles):
            logger.info(f"Analyzing article {i+1}/{total}")
            
            # Analyze article
            result = self.analyze_article(article, components)
            results.append(result)
            
            # Save to database if requested
            if save_to_db and not result.get('errors'):
                self._save_analysis_to_db(article.get('id'), result)
        
        logger.info(f"Completed batch analysis of {total} articles")
        return results
    
    def _analyze_batch_optimized(self, articles: List[Dict],
                                components: Optional[List[str]],
                                save_to_db: bool) -> List[Dict]:
        """Optimized batch processing with memory management."""
        logger.info(f"Starting optimized batch analysis of {len(articles)} articles")
        
        # Define components to run
        if components is None:
            components = ['preprocess', 'entities', 'keywords', 'sentiment', 
                         'topics', 'industry', 'entity_linking']
        
        results = []
        
        with self.memory_pipeline.memory_efficient_context():
            # Process in optimized batches
            text_lengths = [len(a.get('content', '') + a.get('title', '')) 
                          for a in articles]
            batches = self.batch_processor.dynamic_batch_size(text_lengths)
            
            for batch_indices in batches:
                batch_articles = [articles[i] for i in batch_indices]
                
                # Process batch efficiently
                batch_results = []
                for article in batch_articles:
                    result = self.analyze_article(article, components)
                    batch_results.append(result)
                    
                    # Save to database if requested
                    if save_to_db and not result.get('errors'):
                        self._save_analysis_to_db(article.get('id'), result)
                
                # Map results back to original order
                for idx, result in zip(batch_indices, batch_results):
                    results.append((idx, result))
                
                # Check memory and cleanup if needed
                memory_stats = self.model_pool.get_memory_usage()
                logger.info(f"Memory usage: {memory_stats['rss_mb']:.2f} MB")
        
        # Sort results back to original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def _save_analysis_to_db(self, article_id: str, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to database."""
        try:
            # Prepare data for storage
            metadata = {
                'ai_analysis': {
                    'timestamp': analysis_results['analysis_timestamp'],
                    'components': analysis_results['components_run'],
                    'processing_time': analysis_results['processing_time']
                }
            }
            
            # Add preprocessing results
            if 'preprocessing' in analysis_results:
                metadata['language'] = analysis_results['preprocessing'].get('language')
                metadata['word_count'] = analysis_results['preprocessing'].get('word_count')
            
            # Add entity results
            if 'entities' in analysis_results:
                entities = analysis_results['entities'].get('entities', {})
                metadata['entities'] = {
                    entity_type: [e['normalized'] for e in entity_list]
                    for entity_type, entity_list in entities.items()
                }
            
            # Add keyword results
            if 'keywords' in analysis_results:
                keywords = analysis_results['keywords'].get('keywords', {})
                if 'ensemble' in keywords:
                    metadata['keywords'] = [
                        kw[0] for kw in keywords['ensemble'][:MAX_KEYWORDS_TO_STORE]
                    ]
            
            # Add sentiment results
            if 'sentiment' in analysis_results:
                sentiment = analysis_results['sentiment']
                metadata['sentiment'] = {
                    'label': sentiment.get('overall_sentiment'),
                    'score': sentiment.get('overall_score'),
                    'confidence': sentiment.get('confidence')
                }
            
            # Add insights
            metadata['insights'] = analysis_results.get('insights', {})
            
            # Update article in database
            self.db.update_article_metadata(article_id, metadata)
            logger.info(f"Saved AI analysis for article {article_id}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis for article {article_id}: {e}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about AI analysis performed."""
        try:
            stats = {
                'total_analyzed': self.db.count_analyzed_articles(),
                'by_language': self.db.get_article_count_by_language(),
                'sentiment_distribution': self.db.get_sentiment_distribution(),
                'top_entities': self.db.get_top_entities(limit=10),
                'top_keywords': self.db.get_top_keywords(limit=20)
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get analysis stats: {e}")
            return {}


def analyze_new_articles(limit: int = 100) -> None:
    """
    Analyze new articles that haven't been processed yet.
    
    Args:
        limit: Maximum number of articles to process
    """
    engine = AIAnalysisEngine()
    
    # Get unanalyzed articles from database
    db = db_adapter.get_connection()
    articles = db.get_unanalyzed_articles(limit=limit)
    
    if not articles:
        logger.info("No new articles to analyze")
        return
    
    logger.info(f"Found {len(articles)} articles to analyze")
    
    # Analyze in batches
    for i in range(0, len(articles), DEFAULT_BATCH_SIZE):
        batch = articles[i:i+DEFAULT_BATCH_SIZE]
        engine.analyze_batch(batch, save_to_db=True)
    
    # Print summary statistics
    stats = engine.get_analysis_stats()
    logger.info(f"Analysis complete. Total analyzed: {stats.get('total_analyzed', 0)}")


def analyze_single_article(article_id: str) -> Optional[Dict[str, Any]]:
    """
    Analyze a specific article by ID.
    
    Args:
        article_id: ID of the article to analyze
    """
    engine = AIAnalysisEngine()
    
    # Get article from database
    db = db_adapter.get_connection()
    article = db.get_article_by_id(article_id)
    
    if not article:
        logger.error(f"Article {article_id} not found")
        return None
    
    # Perform analysis
    result = engine.analyze_article(article)
    
    # Save to database
    if not result.get('errors'):
        engine._save_analysis_to_db(article_id, result)
    
    return result


if __name__ == "__main__ff":
    # Test the AI analysis engine
    logging.basicConfig(
        level=logging.INFO,
        format=ff'{asctime}'s - {name}"s - {levelname}'s - {message}"s'
    )
    
    # Test article
    test_article = {
        'id': 'test-001',
        'title': 'Tech Giant Announces Revolutionary AI Breakthrough',
        'content': """
        In a groundbreaking announcement today, TechCorp unveiled its latest 
        artificial intelligence system that promises to transform how businesses 
        operate. CEO John Smith stated that the new AI platform, called "NexGen AI", 
        can process natural language with unprecedented accuracy and speed.
        
        The company's stock surged 15% following the announcement, as investors 
        showed strong confidence in the technology's potential. Industry analysts 
        predict this could disrupt multiple sectors including healthcare, finance, 
        and retail.
        
        "This is not just an incremental improvement," said Dr. Sarah Johnson, 
        TechCorp's Chief AI Officer. "We've achieved a fundamental breakthrough 
        in how AI understands and processes human language."
        
        The system will be available for enterprise customers starting Q2 2024, 
        with pricing starting at $10,000 per month for basic access.
        """,
        'source': 'TechNews Daily',
        'published_date': datetime.utcnow().isoformat()
    }
    
    # Initialize engine
    engine = AIAnalysisEngine()
    
    # Analyze the article
    print("Analyzing test article...")
    result = engine.analyze_article(test_article)
    
    # Print results
    print("\n" + "="*60)
    print("AI ANALYSIS RESULTS")
    print("="*60)
    
    # Print insights
    print("\nINSIGHTS:")
    print(json.dumps(result.get('insights', {}), indent=2))
    
    # Print sentiment
    if 'sentiment' in result:
        print("\nSENTIMENT ANALYSIS:")
        sentiment = result['sentiment']
        print(f"Overall: {sentiment.get('overall_sentiment')} "
              f"(score: {sentiment.get('overall_score', 0):.3f})")
        print(f"Interpretation: {sentiment.get('interpretation', 'N/A')}")
    
    # Print top entities
    if 'entities' in result:
        print("\nTOP ENTITIES:")
        entities = result['entities'].get('entities', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"\n{entity_type.upper()}:")
                for entity in entity_list[:3]:
                    print(f"  - {entity['normalized']}")
    
    # Print top keywords
    if 'keywords' in result:
        print("\nTOP KEYWORDS:")
        keywords = result['keywords'].get('keywords', {})
        if 'ensemble' in keywords:
            for keyword, score in keywords['ensemble'][:MAX_TOP_KEYWORDS]:
                print(f"  - {keyword}: {score:.3f}")
    
    print(f"\nProcessing time: {result.get('processing_time', 0):.2f} seconds")