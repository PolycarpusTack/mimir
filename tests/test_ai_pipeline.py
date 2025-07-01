"""
Comprehensive tests for the AI analysis pipeline.
Tests all components and benchmarks performance.
"""

import logging
import os
import sys
import time

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_analysis import AIAnalysisEngine
from entity_extraction import EntityExtractor
from industry_classification import IndustryClassifier
from keyword_extraction import KeywordExtractor
from nlp_preprocessing import TextPreprocessor
from sentiment_analysis import SentimentAnalyzer
from topic_modeling import TopicModeler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAIPipeline:
    """Test suite for AI analysis pipeline."""

    @pytest.fixture
    def sample_articles(self):
        """Sample articles for testing."""
        return [
            {
                "id": "test-001",
                "title": "Apple Unveils Revolutionary AI Technology",
                "content": """Apple Inc. announced today a breakthrough in artificial intelligence 
                that promises to transform how users interact with their devices. CEO Tim Cook 
                presented the new technology at the company's headquarters in Cupertino, California.
                
                The technology uses advanced machine learning algorithms to understand user behavior
                and provide personalized experiences. Industry analysts are calling it a game-changer
                for the tech industry. The stock market responded positively, with Apple shares 
                rising 5% in after-hours trading.""",
                "source": "TechNews",
                "published_date": "2024-01-15T10:00:00Z",
            },
            {
                "id": "test-002",
                "title": "Major Bank Faces Regulatory Scrutiny",
                "content": """JPMorgan Chase is under investigation by federal regulators for 
                potential violations of consumer protection laws. The bank's CEO Jamie Dimon 
                stated that the company is cooperating fully with authorities.
                
                The investigation centers around the bank's lending practices and fee structures.
                Consumer advocacy groups have welcomed the scrutiny, saying it's long overdue.
                The bank's stock fell 3% on the news, reflecting investor concerns about 
                potential fines and regulatory actions.""",
                "source": "Financial Times",
                "published_date": "2024-01-15T14:30:00Z",
            },
            {
                "id": "test-003",
                "title": "Breakthrough in Cancer Treatment Shows Promise",
                "content": """Researchers at Johns Hopkins University have developed a new 
                immunotherapy treatment that shows remarkable results in fighting aggressive 
                forms of cancer. Dr. Sarah Johnson, lead researcher, called it "a significant 
                step forward in our battle against cancer."
                
                The treatment works by enhancing the body's natural immune response to target
                cancer cells more effectively. Early clinical trials have shown a 70% response
                rate in patients with advanced melanoma. The FDA has granted fast-track status
                for the treatment's approval process.""",
                "source": "Medical News Today",
                "published_date": "2024-01-15T09:00:00Z",
            },
        ]

    @pytest.fixture
    def ai_engine(self):
        """Initialize AI analysis engine."""
        return AIAnalysisEngine()

    def test_preprocessing(self, sample_articles):
        """Test text preprocessing component."""
        preprocessor = TextPreprocessor()

        for article in sample_articles:
            start_time = time.time()
            result = preprocessor.preprocess_article(article)
            processing_time = time.time() - start_time

            assert "preprocessed" in result
            assert result["preprocessed"]["language"] in ["en", "nl", "de", "fr", "unknown"]
            assert result["preprocessed"]["word_count"] > 0
            assert result["preprocessed"]["sentence_count"] > 0
            assert processing_time < 1.0  # Should process in under 1 second

            logger.info(f"Preprocessed article {article['id']} in {processing_time:.3f}s")

    def test_entity_extraction(self, sample_articles):
        """Test entity extraction component."""
        extractor = EntityExtractor()

        for article in sample_articles:
            start_time = time.time()
            text = f"{article['title']} {article['content']}"
            entities = extractor.extract_entities(text)
            processing_time = time.time() - start_time

            assert isinstance(entities, dict)
            assert any(len(entities.get(etype, [])) > 0 for etype in ["organization", "person", "location"])
            assert processing_time < 2.0  # Should process in under 2 seconds

            # Verify specific entities
            if article["id"] == "test-001":
                org_names = [e["normalized"] for e in entities.get("organization", [])]
                assert "Apple" in org_names or "Apple Inc" in org_names
                person_names = [e["normalized"] for e in entities.get("person", [])]
                assert "Tim Cook" in person_names

            logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities in {processing_time:.3f}s")

    def test_keyword_extraction(self, sample_articles):
        """Test keyword extraction component."""
        extractor = KeywordExtractor()

        for article in sample_articles:
            start_time = time.time()
            text = f"{article['title']} {article['content']}"
            keywords = extractor.extract_keywords(text, method="all", num_keywords=10)
            processing_time = time.time() - start_time

            assert "tfidf" in keywords
            assert "yake" in keywords
            assert "textrank" in keywords
            assert "ensemble" in keywords

            # Check ensemble keywords
            ensemble_kws = keywords["ensemble"]
            assert ensemble_kws
            assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in ensemble_kws)
            assert processing_time < 3.0  # Should process in under 3 seconds

            logger.info(f"Extracted keywords using all methods in {processing_time:.3f}s")

    def test_sentiment_analysis(self, sample_articles):
        """Test sentiment analysis component."""
        analyzer = SentimentAnalyzer()

        for article in sample_articles:
            start_time = time.time()
            text = f"{article['title']} {article['content']}"
            result = analyzer.analyze(text, model_type="general")
            processing_time = time.time() - start_time

            assert result.label in ["very_negative", "negative", "neutral", "positive", "very_positive"]
            assert 0 <= result.confidence <= 1
            assert -1 <= result.score <= 1
            assert processing_time < 2.0  # Should process in under 2 seconds

            # Check expected sentiments
            if article["id"] == "test-001":
                assert result.label in ["positive", "very_positive"]
            elif article["id"] == "test-002":
                assert result.label in ["negative", "very_negative"]

            logger.info(f"Sentiment: {result.label} (confidence: {result.confidence:.2f}) in {processing_time:.3f}s")

    def test_industry_classification(self, sample_articles):
        """Test industry classification component."""
        classifier = IndustryClassifier()

        for article in sample_articles:
            start_time = time.time()
            text = f"{article['title']} {article['content']}"
            result = classifier.classify(text)
            processing_time = time.time() - start_time

            assert result.primary_industry in [ind.value for ind in classifier.industries]
            assert 0 <= result.confidence <= 1
            assert result.all_scores
            assert processing_time < 2.0  # Should process in under 2 seconds

            # Check expected industries
            if article["id"] == "test-001":
                assert result.primary_industry == "Technology"
            elif article["id"] == "test-002":
                assert result.primary_industry == "Finance & Banking"
            elif article["id"] == "test-003":
                assert result.primary_industry == "Healthcare & Pharmaceuticals"

            logger.info(
                f"Industry: {result.primary_industry} (confidence: {result.confidence:.2f}) in {processing_time:.3f}s"
            )

    def test_topic_modeling(self, sample_articles):
        """Test topic modeling component."""
        modeler = TopicModeler(num_topics=2)

        # Extract content from articles
        documents = [f"{a['title']} {a['content']}" for a in sample_articles]

        start_time = time.time()
        result = modeler.train_lda_model(documents, optimize_topics=False)
        processing_time = time.time() - start_time

        assert "topics" in result
        assert "topic_labels" in result
        assert "coherence_score" in result
        assert len(result["topics"]) == 2
        assert processing_time < 5.0  # Should process in under 5 seconds

        logger.info(
            f"Discovered {result['num_topics']} topics with coherence {result['coherence_score']:.3f} in {processing_time:.3f}s"
        )

    def test_full_pipeline(self, ai_engine, sample_articles):
        """Test complete AI analysis pipeline."""
        total_start = time.time()

        for article in sample_articles:
            start_time = time.time()
            result = ai_engine.analyze_article(article)
            processing_time = time.time() - start_time

            # Check all components ran
            assert "preprocessing" in result
            assert "entities" in result
            assert "keywords" in result
            assert "sentiment" in result
            assert "industry" in result
            assert "insights" in result

            # Check no critical errors
            assert len(result.get("errors", [])) == 0

            # Check performance
            assert processing_time < 10.0  # Should process in under 10 seconds

            logger.info(f"\nArticle: {article['title']}")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Language: {result['preprocessing']['language']}")
            logger.info(f"Entities: {result['entities']['statistics']['total_entities']}")
            logger.info(f"Sentiment: {result['sentiment']['overall_sentiment']}")
            logger.info(f"Industry: {result['industry']['primary_industry']}")
            logger.info(f"Summary: {result['insights']['summary']}")

        total_time = time.time() - total_start
        avg_time = total_time / len(sample_articles)
        logger.info(f"\nTotal pipeline time: {total_time:.2f}s")
        logger.info(f"Average per article: {avg_time:.2f}s")

        assert avg_time < 10.0  # Average should be under 10 seconds per article

    def test_batch_processing(self, ai_engine, sample_articles):
        """Test batch processing performance."""
        # Test with more articles
        batch_articles = sample_articles * 3  # 9 articles

        start_time = time.time()
        results = ai_engine.analyze_batch(batch_articles, save_to_db=False)
        processing_time = time.time() - start_time

        assert len(results) == len(batch_articles)
        assert all("preprocessing" in r for r in results)

        avg_time = processing_time / len(batch_articles)
        logger.info(f"Batch processed {len(batch_articles)} articles in {processing_time:.2f}s")
        logger.info(f"Average time per article: {avg_time:.2f}s")

        # Batch processing should be more efficient
        assert avg_time < 8.0  # Should be faster than individual processing

    def test_memory_usage(self, ai_engine, sample_articles):
        """Test memory usage during processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        # Process articles multiple times
        for _ in range(5):
            for article in sample_articles:
                ai_engine.analyze_article(article)

        # Check memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase

    def test_error_handling(self, ai_engine):
        """Test error handling with invalid inputs."""
        # Test with empty article
        result = ai_engine.analyze_article({})
        assert "errors" in result
        assert result["article_id"] == "unknown"

        # Test with missing content
        result = ai_engine.analyze_article({"id": "test", "title": "Test"})
        assert "preprocessing" in result  # Should still work with title only

        # Test with invalid content
        result = ai_engine.analyze_article({"id": "test", "content": None, "title": 123})  # Invalid type
        assert isinstance(result, dict)  # Should handle gracefully


def run_benchmarks():
    """Run performance benchmarks."""
    logger.info("Running AI Pipeline Benchmarks...")

    # Create test instance
    test = TestAIPipeline()
    engine = AIAnalysisEngine()

    # Generate sample articles
    sample_articles = test.sample_articles()

    # Run benchmarks
    logger.info("\n=== Component Benchmarks ===")
    test.test_preprocessing(sample_articles)
    test.test_entity_extraction(sample_articles)
    test.test_keyword_extraction(sample_articles)
    test.test_sentiment_analysis(sample_articles)
    test.test_industry_classification(sample_articles)
    test.test_topic_modeling(sample_articles)

    logger.info("\n=== Pipeline Benchmarks ===")
    test.test_full_pipeline(engine, sample_articles)
    test.test_batch_processing(engine, sample_articles)
    test.test_memory_usage(engine, sample_articles)

    logger.info("\n=== Error Handling Tests ===")
    test.test_error_handling(engine)

    logger.info("\nBenchmarks completed successfully!")


if __name__ == "__main__":
    run_benchmarks()
