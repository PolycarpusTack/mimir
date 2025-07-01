"""
AI memory optimization and batch processing utilities for Mimir.
Optimizes model loading, memory usage, and batch processing efficiency.
"""

import gc
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import psutil
import spacy
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class ModelPool:
    """Singleton pool for sharing models across processes using shared memory."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models = {}
        self.model_locks = defaultdict(threading.Lock)
        self.usage_counts = defaultdict(int)
        self.last_used = defaultdict(lambda: time.time())
        self._initialized = True

        # Configure memory settings
        self._configure_memory_settings()

    def _configure_memory_settings(self):
        """Configure memory optimization settings."""
        # Set environment variables for memory efficiency
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Limit PyTorch threads for CPU inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            torch.set_num_threads(2)

        # Configure spaCy memory settings
        spacy.prefer_gpu()

    @lru_cache(maxsize=4)
    def get_spacy_model(self, model_name: str):
        """Get or load a spaCy model with caching."""
        with self.model_locks[f"spacy_{model_name}"]:
            key = f"spacy_{model_name}"

            if key not in self.models:
                logger.info(f"Loading spaCy model: {model_name}")
                try:
                    # Disable unnecessary components for efficiency
                    nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
                    # Add sentencizer for sentence segmentation
                    nlp.add_pipe("sentencizer")
                    self.models[key] = nlp
                except Exception as e:
                    logger.error(f"Failed to load spaCy model {model_name}: {e}")
                    return None

            self.usage_counts[key] += 1
            self.last_used[key] = time.time()
            return self.models[key]

    @lru_cache(maxsize=3)
    def get_transformer_model(self, task: str, model_name: Optional[str] = None):
        """Get or load a transformer model with caching."""
        key = f"transformer_{task}_{model_name or 'default'}"

        with self.model_locks[key]:
            if key not in self.models:
                logger.info(f"Loading transformer model for {task}")
                try:
                    # Use smaller models for efficiency
                    if task == "sentiment-analysis":
                        model = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
                    elif task == "zero-shot-classification":
                        model = model_name or "facebook/bart-large-mnli"
                    else:
                        model = model_name

                    # Load with reduced precision for memory efficiency
                    pipe = pipeline(
                        task,
                        model=model,
                        device=-1,  # CPU
                        model_kwargs={"torchscript": True} if torch.cuda.is_available() else {},
                    )
                    self.models[key] = pipe
                except Exception as e:
                    logger.error(f"Failed to load transformer model for {task}: {e}")
                    return None

            self.usage_counts[key] += 1
            self.last_used[key] = time.time()
            return self.models[key]

    def cleanup_unused_models(self, max_age_seconds: int = 3600):
        """Remove models that haven't been used recently."""
        current_time = time.time()
        models_to_remove = []

        for key, last_used_time in self.last_used.items():
            if current_time - last_used_time > max_age_seconds:
                models_to_remove.append(key)

        for key in models_to_remove:
            with self.model_locks[key]:
                if key in self.models:
                    del self.models[key]
                    logger.info(f"Removed unused model: {key}")

        # Force garbage collection
        gc.collect()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "model_count": len(self.models),
        }


class BatchProcessor:
    """Efficient batch processing for AI analysis."""

    def __init__(self, batch_size: int = 10, max_workers: int = None):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items to process in a batch
            max_workers: Maximum number of worker processes
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
        self.model_pool = ModelPool()

    def process_batch_spacy(self, texts: List[str], model_name: str = "en_core_web_md") -> List[Any]:
        """
        Process a batch of texts with spaCy.

        Args:
            texts: List of texts to process
            model_name: spaCy model to use

        Returns:
            List of processed documents
        """
        nlp = self.model_pool.get_spacy_model(model_name)
        if not nlp:
            return []

        # Use pipe for efficient batch processing
        docs = list(nlp.pipe(texts, batch_size=self.batch_size, n_process=1))
        return docs

    def process_batch_transformer(
        self, texts: List[str], task: str, model_name: Optional[str] = None, **kwargs
    ) -> List[Any]:
        """
        Process a batch of texts with transformers.

        Args:
            texts: List of texts to process
            task: Transformer task
            model_name: Optional model name
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of results
        """
        pipe = self.model_pool.get_transformer_model(task, model_name)
        if not pipe:
            return []

        # Process in chunks to avoid memory issues
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            chunk_results = pipe(chunk, **kwargs)
            results.extend(chunk_results)

            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def dynamic_batch_size(self, text_lengths: List[int]) -> List[List[int]]:
        """
        Create dynamic batches based on text length.

        Args:
            text_lengths: List of text lengths

        Returns:
            List of batch indices
        """
        # Sort indices by text length
        sorted_indices = sorted(range(len(text_lengths)), key=lambda i: text_lengths[i])

        batches = []
        current_batch = []
        current_size = 0
        max_batch_chars = 10000  # Maximum characters per batch

        for idx in sorted_indices:
            text_len = text_lengths[idx]

            if current_size + text_len > max_batch_chars and current_batch:
                batches.append(current_batch)
                current_batch = [idx]
                current_size = text_len
            else:
                current_batch.append(idx)
                current_size += text_len

        if current_batch:
            batches.append(current_batch)

        return batches


class MemoryOptimizedPipeline:
    """Memory-optimized AI analysis pipeline."""

    def __init__(self, max_memory_mb: int = 1024):
        """
        Initialize memory-optimized pipeline.

        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.model_pool = ModelPool()
        self.batch_processor = BatchProcessor()
        self._running = True
        self._setup_memory_monitoring()

    def _setup_memory_monitoring(self):
        """Setup memory monitoring thread."""

        def monitor_memory() -> None:
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            while getattr(self, "_running", True):
                memory_stats = self.model_pool.get_memory_usage()
                if memory_stats["rss_mb"] > self.max_memory_mb:
                    logger.warning(
                        f"Memory usage ({memory_stats['rss_mb']:.2f} MB) " f"exceeds limit ({self.max_memory_mb} MB)"
                    )
                    self.model_pool.cleanup_unused_models(max_age_seconds=1800)
                    gc.collect()
                time.sleep(60)  # Check every minute

        # Start monitoring in background
        self.monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.monitor_thread.name = "MemoryMonitor"
        self.monitor_thread.start()

    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient processing."""
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)

        # Clear any existing caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        yield

        # Cleanup after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def process_articles_memory_efficient(
        self, articles: List[Dict[str, Any]], components: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Process articles with memory optimization.

        Args:
            articles: List of articles to process
            components: List of components to run

        Returns:
            List of results
        """
        results = []

        with self.memory_efficient_context():
            # Create dynamic batches based on article length
            text_lengths = [len(a.get("content", "") + a.get("title", "")) for a in articles]
            batches = self.batch_processor.dynamic_batch_size(text_lengths)

            for batch_indices in batches:
                batch_articles = [articles[i] for i in batch_indices]
                batch_results = self._process_batch(batch_articles, components)

                # Map results back to original order
                for idx, result in zip(batch_indices, batch_results):
                    results.append((idx, result))

                # Check memory usage after each batch
                memory_stats = self.model_pool.get_memory_usage()
                if memory_stats["rss_mb"] > self.max_memory_mb * 0.9:
                    logger.info("Approaching memory limit, running cleanup")
                    self.model_pool.cleanup_unused_models(max_age_seconds=600)
                    gc.collect()

            # Sort results back to original order
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

    def _process_batch(self, articles: List[Dict[str, Any]], components: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of articles."""
        # This is a simplified version - in production, integrate with ai_analysis.py
        results = []

        for article in articles:
            result = {"article_id": article.get("id", "unknown"), "components_run": components, "processing_time": 0}

            # Add placeholder for actual processing
            # In production, call actual AI components here

            results.append(result)

        return results


class StreamingProcessor:
    """Process articles in a streaming fashion to minimize memory usage."""

    def __init__(self, chunk_size: int = 100):
        """
        Initialize streaming processor.

        Args:
            chunk_size: Number of articles to process at once
        """
        self.chunk_size = chunk_size
        self.pipeline = MemoryOptimizedPipeline()

    def process_stream(self, article_generator, components: List[str], callback=None) -> int:
        """
        Process articles from a generator/iterator.

        Args:
            article_generator: Generator yielding articles
            components: List of AI components to run
            callback: Optional callback for each processed chunk

        Returns:
            Total number of articles processed
        """
        total_processed = 0
        chunk = []

        for article in article_generator:
            chunk.append(article)

            if len(chunk) >= self.chunk_size:
                results = self.pipeline.process_articles_memory_efficient(chunk, components)

                if callback:
                    callback(results)

                total_processed += len(chunk)
                chunk = []

                # Log progress
                logger.info(f"Processed {total_processed} articles")

        # Process remaining articles
        if chunk:
            results = self.pipeline.process_articles_memory_efficient(chunk, components)

            if callback:
                callback(results)

            total_processed += len(chunk)

        return total_processed


def optimize_model_loading():
    """Optimize model loading for production use."""
    logger.info("Optimizing model loading...")

    # Pre-load commonly used models
    model_pool = ModelPool()

    # Load spaCy models with error handling
    for lang in ["en_core_web_md", "nl_core_news_md"]:
        try:
            model_pool.get_spacy_model(lang)
        except ModelLoadingError as e:
            logger.warning(f"Failed to preload spaCy model {lang}: {e}")

    # Load transformer models with error handling
    try:
        model_pool.get_transformer_model("sentiment-analysis")
    except ModelLoadingError as e:
        logger.warning(f"Failed to preload sentiment model: {e}")

    try:
        model_pool.get_transformer_model("zero-shot-classification")
    except ModelLoadingError as e:
        logger.warning(f"Failed to preload classification model: {e}")

    logger.info("Model pre-loading complete")


def benchmark_batch_sizes():
    """Benchmark different batch sizes for optimal performance."""
    import time

    # Generate test data
    test_texts = ["This is a test article about technology and innovation." * 10 for _ in range(100)]

    batch_processor = BatchProcessor()
    results = {}

    for batch_size in [1, 5, 10, 20, 50]:
        batch_processor.batch_size = batch_size

        start_time = time.time()
        batch_processor.process_batch_spacy(test_texts)
        processing_time = time.time() - start_time

        results[batch_size] = {"time": processing_time, "articles_per_second": len(test_texts) / processing_time}

        logger.info(
            f"Batch size {batch_size}: {processing_time:.2f}s "
            f"({results[batch_size]['articles_per_second']:.2f} articles/s)"
        )

    # Find optimal batch size
    optimal_batch = max(results.items(), key=lambda x: x[1]["articles_per_second"])[0]
    logger.info(f"Optimal batch size: {optimal_batch}")

    return results


if __name__ == "__main__":
    # Test memory optimization
    logging.basicConfig(level=logging.INFO)

    # Run benchmarks with error handling
    logger.info("Running batch size benchmarks...")
    try:
        benchmark_results = benchmark_batch_sizes()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        benchmark_results = {}

    # Test memory-efficient pipeline
    logger.info("\nTesting memory-efficient pipeline...")
    pipeline = MemoryOptimizedPipeline(max_memory_mb=512)

    # Generate test articles
    test_articles = [
        {
            "id": f"test-{i}",
            "title": f"Test Article {i}",
            "content": "This is a test article about AI and technology. " * 50,
        }
        for i in range(50)
    ]

    # Process with memory monitoring
    try:
        start_mem = pipeline.model_pool.get_memory_usage()
        logger.info(f"Initial memory: {start_mem['rss_mb']:.2f} MB")

        results = pipeline.process_articles_memory_efficient(test_articles, ["preprocess", "entities", "sentiment"])

        end_mem = pipeline.model_pool.get_memory_usage()
        logger.info(f"Final memory: {end_mem['rss_mb']:.2f} MB")
        logger.info(f"Memory increase: {end_mem['rss_mb'] - start_mem['rss_mb']:.2f} MB")
    except Exception as e:
        logger.error(f"Memory-efficient processing failed: {e}")
    finally:
        # Stop the monitoring thread
        pipeline._running = False

    # Test streaming processor
    logger.info("\nTesting streaming processor...")

    def article_generator():
        """TODO: Add docstring."""
        """TODO: Add docstring."""
        for i in range(200):
            yield {
                "id": f"stream-{i}",
                "title": f"Streaming Article {i}",
                "content": "Content for streaming test. " * 20,
            }

    streaming_processor = StreamingProcessor(chunk_size=25)
    total = streaming_processor.process_stream(
        article_generator(),
        ["preprocess", "keywords"],
        callback=lambda results: logger.info(f"Processed chunk of {len(results)} articles"),
    )

    logger.info(f"Total articles processed: {total}")
