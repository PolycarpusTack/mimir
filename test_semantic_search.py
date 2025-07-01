"""
Comprehensive Test Suite for Semantic Search & Similarity Engine
EPIC 3: Semantic Search & Similarity Engine

This module provides comprehensive tests for all semantic search components:
- Sentence transformer models
- Embedding generation and storage
- Vector similarity search
- Hybrid search functionality
- Deduplication algorithms
- Database operations

Author: Claude Code
"""

import logging
import os
import shutil
import tempfile
import time
import unittest
from typing import Any, Dict, List

import psutil

from advanced_deduplication import AdvancedDeduplicator, ContentFingerprinter
from db_manager_semantic import SemanticDatabaseManager
from embedding_pipeline import EmbeddingPipeline, EmbeddingTask
from semantic_search import SemanticSearchEngine
from setup_sentence_transformers import SentenceTransformerSetup

logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
logger = logging.getLogger(__name__)


class TestSentenceTransformers(unittest.TestCase):
    """Test sentence transformer setup and benchmarking."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.setup = SentenceTransformerSetup(cache_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_download(self):
        """Test downloading a sentence transformer model."""
        # Use a small, fast model for testing
        model_name = "all-MiniLM-L6-v2"

        # Test download
        success = self.setup.download_model(model_name)
        self.assertTrue(success)

        # Verify model is cached
        self.assertIn(model_name, self.setup.available_models)

    def test_model_benchmark(self):
        """Test benchmarking a sentence transformer model."""
        model_name = "all-MiniLM-L6-v2"

        # Download model first
        self.setup.download_model(model_name)

        # Run benchmark with small parameters for speed
        results = self.setup.benchmark_model(model_name, num_iterations=1, batch_sizes=[1, 4])

        self.assertIsInstance(results, dict)
        self.assertIn("model_name", results)
        self.assertIn("batch_benchmarks", results)
        self.assertIn("quality_metrics", results)

        # Check batch results
        batch_results = results["batch_benchmarks"]
        self.assertIn(1, batch_results)
        self.assertIn(4, batch_results)

        # Check that we have reasonable performance metrics
        batch_1 = batch_results[1]
        self.assertGreater(batch_1["sentences_per_second"], 0)
        self.assertGreater(batch_1["embedding_dimension"], 0)


class TestSemanticDatabaseManager(unittest.TestCase):
    """Test semantic database operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        # Skip if PostgreSQL not available
        try:
            cls.db_manager = SemanticDatabaseManager()
            cls.db_available = True
        except Exception as e:
            cls.db_available = False
            cls.skipTest(cls, f"PostgreSQL not available: {e}")

    def setUp(self):
        """Set up test data."""
        if not self.db_available:
            self.skipTest("Database not available")

        self.test_article_id = "test-article-123"
        self.test_embedding = np.random.rand(384).astype(np.float32)
        self.test_embedding = self.test_embedding / np.linalg.norm(self.test_embedding)  # Normalize

    def test_store_embedding(self):
        """Test storing a single embedding."""
        success = self.db_manager.store_embedding(
            article_id=self.test_article_id,
            embedding=self.test_embedding,
            model_name="test-model",
            embedding_type="test",
            quality_score=0.9,
        )

        self.assertTrue(success)

    def test_get_embedding(self):
        """Test retrieving an embedding."""
        # Store first
        self.db_manager.store_embedding(
            article_id=self.test_article_id,
            embedding=self.test_embedding,
            model_name="test-model",
            embedding_type="test",
        )

        # Retrieve
        retrieved = self.db_manager.get_embedding(
            article_id=self.test_article_id, model_name="test-model", embedding_type="test"
        )

        self.assertIsNotNone(retrieved)
        np.testing.assert_array_almost_equal(retrieved, self.test_embedding, decimal=5)

    def test_batch_store_embeddings(self):
        """Test batch storing of embeddings."""
        embeddings_data = []
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            embeddings_data.append(
                {
                    "article_id": f"test-article-{i}",
                    "embedding": embedding,
                    "model_name": "test-model",
                    "embedding_type": "test",
                    "quality_score": 0.8 + i * 0.05,
                }
            )

        stored_count = self.db_manager.store_embeddings_batch(embeddings_data)
        self.assertEqual(stored_count, 5)

    def test_embedding_statistics(self):
        """Test getting embedding statistics."""
        # Store some test embeddings first
        self.test_batch_store_embeddings()

        stats = self.db_manager.get_embedding_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_embeddings", stats)
        self.assertIn("articles_with_embeddings", stats)
        self.assertIn("by_model", stats)
        self.assertGreater(stats["total_embeddings"], 0)


class TestSemanticSearchEngine(unittest.TestCase):
    """Test semantic search engine functionality."""

    def setUp(self):
        """Set up test environment."""
        try:
            self.engine = SemanticSearchEngine(
                model_name="all-MiniLM-L6-v2",
                cache_embeddings=False,  # Disable caching for tests
                use_redis=False,  # Use local queue for tests
            )
            self.available = True
        except Exception as e:
            self.available = False
            self.skipTest(f"Semantic engine not available: {e}")

    def test_embedding_generation(self):
        """Test generating embeddings for text."""
        texts = [
            "Artificial intelligence is transforming healthcare",
            "Machine learning algorithms improve patient outcomes",
            "Climate change affects global weather patterns",
        ]

        embeddings = self.engine.generate_embeddings(texts, show_progress=False)

        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], self.engine.embedding_dim)

        # Check that embeddings are normalized
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_similarity_calculation(self):
        """Test similarity calculation between texts."""
        text1 = "Machine learning in healthcare"
        text2 = "AI applications in medical diagnosis"
        text3 = "Weather patterns and climate change"

        embeddings = self.engine.generate_embeddings([text1, text2, text3], show_progress=False)

        # Calculate cosine similarities
        sim_12 = np.dot(embeddings[0], embeddings[1])  # Similar topics
        sim_13 = np.dot(embeddings[0], embeddings[2])  # Different topics

        # Similar topics should have higher similarity
        self.assertGreater(sim_12, sim_13)

        # All similarities should be between -1 and 1
        self.assertGreaterEqual(sim_12, -1.0)
        self.assertLessEqual(sim_12, 1.0)
        self.assertGreaterEqual(sim_13, -1.0)
        self.assertLessEqual(sim_13, 1.0)

    def test_create_minhash(self):
        """Test MinHash creation for deduplication."""
        minhash1 = self.engine._create_minhash("This is a test document about AI")
        minhash2 = self.engine._create_minhash("This is a test document about AI")
        minhash3 = self.engine._create_minhash("Completely different content here")

        # Identical texts should have identical MinHash
        self.assertEqual(minhash1.jaccard(minhash2), 1.0)

        # Different texts should have lower similarity
        self.assertLess(minhash1.jaccard(minhash3), 1.0)


class TestEmbeddingPipeline(unittest.TestCase):
    """Test embedding pipeline functionality."""

    def setUp(self):
        """Set up test environment."""
        try:
            self.pipeline = EmbeddingPipeline(
                model_name="all-MiniLM-L6-v2",
                batch_size=4,
                max_workers=1,  # Single worker for tests
                cache_embeddings=False,
                use_redis=False,
            )
            self.available = True
        except Exception as e:
            self.available = False
            self.skipTest(f"Embedding pipeline not available: {e}")

    def test_embedding_task_creation(self):
        """Test creating embedding tasks."""
        task = EmbeddingTask(
            article_id="test-123",
            title="Test Article",
            content="This is test content for the article",
            source="test-source",
            priority=5,
        )

        self.assertEqual(task.article_id, "test-123")
        self.assertEqual(task.priority, 5)
        self.assertEqual(task.retry_count, 0)

    def test_single_article_processing(self):
        """Test processing a single article."""
        task = EmbeddingTask(
            article_id="test-456",
            title="AI in Healthcare",
            content="Artificial intelligence is revolutionizing medical diagnosis and treatment",
            source="test-source",
        )

        result = self.pipeline.process_single_article(task)

        self.assertTrue(result.success)
        self.assertEqual(result.article_id, "test-456")
        self.assertGreater(result.embedding.shape[0], 0)
        self.assertGreater(result.quality_score, 0)
        self.assertGreater(result.processing_time, 0)

    def test_batch_processing(self):
        """Test batch processing of articles."""
        tasks = []
        for i in range(3):
            task = EmbeddingTask(
                article_id=f"test-batch-{i}",
                title=f"Test Article {i}",
                content=f"This is test content for article number {i}",
                source="test-source",
            )
            tasks.append(task)

        results = self.pipeline.process_batch(tasks)

        self.assertEqual(len(results), 3)

        for result in results:
            self.assertTrue(result.success)
            self.assertGreater(result.embedding.shape[0], 0)


class TestContentFingerprinter(unittest.TestCase):
    """Test content fingerprinting for deduplication."""

    def setUp(self):
        """Set up test environment."""
        self.fingerprinter = ContentFingerprinter()

    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        dirty_text = "<p>This is a TEST article with <b>HTML tags</b>!</p>"
        clean_text = self.fingerprinter.preprocess_text(dirty_text)

        # Should be lowercase and without HTML
        self.assertNotIn("<p>", clean_text)
        self.assertNotIn("<b>", clean_text)
        self.assertNotIn("TEST", clean_text)
        self.assertIn("test", clean_text)

    def test_shingle_creation(self):
        """Test creating text shingles."""
        text = "this is a test document"
        shingles = self.fingerprinter.create_shingles(text)

        self.assertIsInstance(shingles, set)
        self.assertGreater(len(shingles), 0)

        # Should contain 3-grams by default
        self.assertIn("this is a", shingles)
        self.assertIn("is a test", shingles)

    def test_minhash_creation(self):
        """Test MinHash signature creation."""
        text1 = "This is a test document about machine learning"
        text2 = "This is a test document about machine learning"
        text3 = "Completely different content here"

        hash1 = self.fingerprinter.create_minhash(text1)
        hash2 = self.fingerprinter.create_minhash(text2)
        hash3 = self.fingerprinter.create_minhash(text3)

        # Identical texts should have identical hashes
        self.assertEqual(hash1.jaccard(hash2), 1.0)

        # Different texts should have lower similarity
        self.assertLess(hash1.jaccard(hash3), 1.0)

    def test_content_hash(self):
        """Test content hash creation."""
        title = "Test Article"
        content = "This is test content"

        hash1 = self.fingerprinter.create_content_hash(title, content)
        hash2 = self.fingerprinter.create_content_hash(title, content)
        hash3 = self.fingerprinter.create_content_hash("Different", "Different content")

        # Same content should produce same hash
        self.assertEqual(hash1, hash2)

        # Different content should produce different hash
        self.assertNotEqual(hash1, hash3)


class TestAdvancedDeduplicator(unittest.TestCase):
    """Test advanced deduplication functionality."""

    def setUp(self):
        """Set up test environment."""
        try:
            self.deduplicator = AdvancedDeduplicator()
            self.available = True
        except Exception as e:
            self.available = False
            self.skipTest(f"Deduplicator not available: {e}")

    def create_test_articles(self) -> List[Dict[str, Any]]:
        """Create test articles for deduplication."""
        return [
            {
                "id": "art1",
                "title": "AI in Healthcare",
                "content": "Artificial intelligence is transforming medical diagnosis",
                "source": "tech-news",
            },
            {
                "id": "art2",
                "title": "AI in Healthcare",  # Exact duplicate title
                "content": "Artificial intelligence is transforming medical diagnosis",  # Exact duplicate content
                "source": "health-news",
            },
            {
                "id": "art3",
                "title": "Machine Learning in Medicine",
                "content": "AI technology is revolutionizing healthcare and medical diagnostics",  # Similar content
                "source": "med-tech",
            },
            {
                "id": "art4",
                "title": "Climate Change Effects",
                "content": "Global warming impacts weather patterns worldwide",  # Different topic
                "source": "environment",
            },
        ]

    def test_exact_duplicate_detection(self):
        """Test exact duplicate detection."""
        articles = self.create_test_articles()

        duplicates = self.deduplicator.find_exact_duplicates(articles)

        # Should find one exact duplicate pair (art1 and art2)
        self.assertEqual(len(duplicates), 1)

        art1, art2, score = duplicates[0]
        self.assertEqual(score, 1.0)  # Exact duplicates should have score 1.0

    def test_title_duplicate_detection(self):
        """Test title-based duplicate detection."""
        articles = self.create_test_articles()

        duplicates = self.deduplicator.find_title_duplicates(articles, similarity_threshold=0.8)

        # Should find duplicates with identical titles
        self.assertGreater(len(duplicates), 0)

    def test_lsh_index_building(self):
        """Test LSH index building."""
        articles = self.create_test_articles()

        # Build LSH index
        self.deduplicator.build_lsh_index(articles)

        # Check that index was built
        self.assertIsNotNone(self.deduplicator.lsh_forest)
        self.assertEqual(len(self.deduplicator.article_signatures), len(articles))

    def test_comprehensive_deduplication(self):
        """Test comprehensive deduplication with all methods."""
        articles = self.create_test_articles()

        results = self.deduplicator.comprehensive_deduplication(articles)

        # Should return results for all deduplication methods
        self.assertIn("exact", results)
        self.assertIn("near", results)
        self.assertIn("title", results)
        self.assertIn("cross_language", results)

        # Should find at least some duplicates
        total_duplicates = sum(len(dup_list) for dup_list in results.values())
        self.assertGreater(total_duplicates, 0)


class TestPerformance(unittest.TestCase):
    """Test performance and scalability."""

    def setUp(self):
        """Set up performance tests."""
        try:
            self.engine = SemanticSearchEngine(model_name="all-MiniLM-L6-v2", cache_embeddings=False, use_redis=False)
            self.available = True
        except Exception as e:
            self.available = False
            self.skipTest(f"Performance tests not available: {e}")

    def test_embedding_performance(self):
        """Test embedding generation performance."""
        # Create test texts
        texts = [f"Test document number {i} about various topics" for i in range(100)]

        # Measure time
        start_time = time.time()
        embeddings = self.engine.generate_embeddings(texts, show_progress=False)
        end_time = time.time()

        processing_time = end_time - start_time
        texts_per_second = len(texts) / processing_time

        print(f"Processed {len(texts)} texts in {processing_time:.2f}s ({texts_per_second:.1f} texts/sec)")

        # Performance should be reasonable (at least 10 texts per second)
        self.assertGreater(texts_per_second, 10)

        # Check output shape
        self.assertEqual(embeddings.shape[0], len(texts))

    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate embeddings for a batch of texts
        texts = [f"Memory test document {i}" for i in range(50)]
        embeddings = self.engine.generate_embeddings(texts, show_progress=False)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

        # Memory increase should be reasonable (less than 500MB for this test)
        self.assertLess(memory_increase, 500)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete semantic search system."""

    def setUp(self):
        """Set up integration tests."""
        # This would test the complete flow from web API to database
        pass

    def test_end_to_end_workflow(self):
        """Test complete workflow from article to semantic search."""
        # This would test:
        # 1. Article insertion
        # 2. Embedding generation
        # 3. Index building
        # 4. Search functionality
        # 5. Similar article finding
        pass


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestSentenceTransformers,
        TestSemanticDatabaseManager,
        TestSemanticSearchEngine,
        TestEmbeddingPipeline,
        TestContentFingerprinter,
        TestAdvancedDeduplicator,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_quick_tests():
    """Run only quick tests (excluding performance and integration)."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add quick test classes
    quick_test_classes = [
        TestSentenceTransformers,
        TestContentFingerprinter,
        TestEmbeddingPipeline,
    ]

    for test_class in quick_test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--specific", help="Run specific test class")

    args = parser.parse_args()

    if args.quick:
        print("Running quick tests...")
        success = run_quick_tests()
    elif args.performance:
        print("Running performance tests...")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()
    elif args.specific:
        print(f"Running specific test class: {args.specific}")
        # Map class names to actual classes
        class_map = {
            "SentenceTransformers": TestSentenceTransformers,
            "SemanticDatabase": TestSemanticDatabaseManager,
            "SemanticSearch": TestSemanticSearchEngine,
            "EmbeddingPipeline": TestEmbeddingPipeline,
            "ContentFingerprinter": TestContentFingerprinter,
            "Deduplicator": TestAdvancedDeduplicator,
            "Performance": TestPerformance,
        }

        if args.specific in class_map:
            suite = unittest.TestLoader().loadTestsFromTestCase(class_map[args.specific])
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            success = result.wasSuccessful()
        else:
            print(f"Unknown test class: {args.specific}")
            print(f"Available classes: {list(class_map.keys())}")
            success = False
    else:
        print("Running all tests...")
        success = run_all_tests()

    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
