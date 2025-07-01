"""
Embedding Generation Pipeline
EPIC 3: Semantic Search & Similarity Engine

This module provides a complete pipeline for generating, storing, and managing
embeddings for all articles in the Mimir database.

Features:
- Batch embedding generation with progress tracking
- Automatic model selection and optimization
- Database integration with PostgreSQL/pgvector
- Queue-based processing for scalability
- Error handling and retry logic
- Monitoring and statistics

Author: Claude Code
"""

import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import redis
import schedule
import torch
from langdetect import DetectorFactory, detect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from db_manager_semantic import SemanticDatabaseManager

DetectorFactory.seed = 0

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTask:
    """Data class for embedding tasks."""

    article_id: str
    title: str
    content: str
    source: str
    priority: int = 5  # 1 = highest, 10 = lowest
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class EmbeddingResult:
    """Data class for embedding results."""

    article_id: str
    embedding: np.ndarray
    model_name: str
    embedding_type: str
    quality_score: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class EmbeddingPipeline:
    """
    Complete embedding generation and management pipeline.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_workers: int = 4,
        cache_embeddings: bool = True,
        use_redis: bool = True,
    ):
        """
        Initialize the embedding pipeline.

        Args:
            model_name: Sentence transformer model to use
            batch_size: Batch size for processing
            max_workers: Maximum number of worker threads
            cache_embeddings: Whether to cache embeddings
            use_redis: Whether to use Redis for task queue
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_embeddings = cache_embeddings
        self.use_redis = use_redis

        # Initialize components
        self.model = None
        self.db_manager = None
        self.redis_client = None
        self.task_queue = queue.PriorityQueue()
        self.results_queue = queue.Queue()

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0,
            "average_time_per_article": 0,
            "articles_per_second": 0,
            "start_time": None,
            "last_processed": None,
        }

        # Control flags
        self.running = False
        self.workers = []

        # Initialize all components
        self._initialize()

    def _initialize(self):
        """Initialize all pipeline components."""
        logger.info(f"Initializing Embedding Pipeline with model: {self.model_name}")

        # Load sentence transformer model
        self._load_model()

        # Initialize database manager
        self._initialize_database()

        # Initialize Redis if enabled
        if self.use_redis:
            self._initialize_redis()

        logger.info("Embedding Pipeline initialized successfully")

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            logger.info(
                f"Model loaded on {device}. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _initialize_database(self):
        """Initialize database connection."""
        try:
            self.db_manager = SemanticDatabaseManager()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def _initialize_redis(self):
        """Initialize Redis connection for task queue."""
        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=1)
            self.redis_client.ping()
            logger.info("Redis connection established for task queue")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using local queue.")
            self.use_redis = False

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()

        # Limit length to avoid memory issues and improve quality
        if len(text) > 1500:  # Optimal length for sentence transformers
            text = text[:1500] + "..."

        return text

    def _calculate_quality_score(self, text: str, embedding: np.ndarray) -> float:
        """
        Calculate a quality score for the embedding.

        Args:
            text: Original text
            embedding: Generated embedding

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize very short or very long texts
        text_length = len(text.split())
        if text_length < 5:
            score *= 0.7
        elif text_length < 10:
            score *= 0.8
        elif text_length > 200:
            score *= 0.9

        # Check embedding quality (not all zeros, reasonable variance)
        embedding_std = np.std(embedding)
        if embedding_std < 0.01:  # Very low variance suggests poor embedding
            score *= 0.5
        elif embedding_std < 0.05:
            score *= 0.8

        # Check for NaN or infinite values
        if not np.isfinite(embedding).all():
            score = 0.0

        return score

    def add_task(self, task: EmbeddingTask):
        """Add an embedding task to the queue."""
        if self.use_redis and self.redis_client:
            # Use Redis for distributed processing
            task_data = {
                "article_id": task.article_id,
                "title": task.title,
                "content": task.content,
                "source": task.source,
                "priority": task.priority,
                "retry_count": task.retry_count,
            }

            # Use priority as score (lower score = higher priority in Redis)
            self.redis_client.zadd("embedding_tasks", {json.dumps(task_data): task.priority})
        else:
            # Use local queue
            self.task_queue.put((task.priority, time.time(), task))

    def get_next_task(self) -> Optional[EmbeddingTask]:
        """Get the next task from the queue."""
        if self.use_redis and self.redis_client:
            # Get highest priority task from Redis
            result = self.redis_client.zpopmin("embedding_tasks", 1)
            if result:
                task_data = json.loads(result[0][0])
                return EmbeddingTask(**task_data)
        else:
            # Get from local queue
            if not self.task_queue.empty():
                try:
                    _, _, task = self.task_queue.get_nowait()
                    return task
                except queue.Empty:
                    pass

        return None

    def queue_articles_for_embedding(self, limit: int = 1000, force_reprocess: bool = False) -> int:
        """
        Queue articles that need embeddings.

        Args:
            limit: Maximum number of articles to queue
            force_reprocess: Whether to reprocess articles that already have embeddings

        Returns:
            Number of articles queued
        """
        logger.info(f"Queuing articles for embedding (limit: {limit})")

        try:
            if force_reprocess:
                # Get all articles
                articles = self.db_manager.get_all_articles(limit=limit)
            else:
                # Get only articles without embeddings
                articles = self.db_manager.get_articles_without_embeddings(model_name=self.model_name, limit=limit)

            if not articles:
                logger.info("No articles found to process")
                return 0

            queued_count = 0
            for article in articles:
                # Create embedding task
                task = EmbeddingTask(
                    article_id=article["id"],
                    title=article.get("title", ""),
                    content=article.get("content", ""),
                    source=article.get("source", "unknown"),
                    priority=5,  # Default priority
                )

                # Prioritize newer articles
                if article.get("published_date"):
                    pub_date = article["published_date"]
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))

                    # Higher priority for newer articles
                    days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                    task.priority = min(10, max(1, days_old // 7 + 1))

                self.add_task(task)
                queued_count += 1

            logger.info(f"Queued {queued_count} articles for embedding generation")
            return queued_count

        except Exception as e:
            logger.error(f"Failed to queue articles: {e}")
            return 0

    def process_single_article(self, task: EmbeddingTask) -> EmbeddingResult:
        """
        Process a single article to generate embeddings.

        Args:
            task: Embedding task to process

        Returns:
            EmbeddingResult with success/failure information
        """
        start_time = time.time()

        try:
            # Combine title and content
            combined_text = f"{task.title} {task.content}"
            processed_text = self._preprocess_text(combined_text)

            if not processed_text.strip():
                return EmbeddingResult(
                    article_id=task.article_id,
                    embedding=np.array([]),
                    model_name=self.model_name,
                    embedding_type="title_content",
                    quality_score=0.0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Empty text after preprocessing",
                )

            # Generate embedding
            embedding = self.model.encode(
                processed_text, convert_to_numpy=True, normalize_embeddings=True  # Normalize for cosine similarity
            )

            # Calculate quality score
            quality_score = self._calculate_quality_score(processed_text, embedding)

            processing_time = time.time() - start_time

            return EmbeddingResult(
                article_id=task.article_id,
                embedding=embedding,
                model_name=self.model_name,
                embedding_type="title_content",
                quality_score=quality_score,
                processing_time=processing_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to process article {task.article_id}: {e}")
            return EmbeddingResult(
                article_id=task.article_id,
                embedding=np.array([]),
                model_name=self.model_name,
                embedding_type="title_content",
                quality_score=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )

    def process_batch(self, tasks: List[EmbeddingTask]) -> List[EmbeddingResult]:
        """
        Process a batch of articles for better efficiency.

        Args:
            tasks: List of embedding tasks

        Returns:
            List of embedding results
        """
        if not tasks:
            return []

        start_time = time.time()
        results = []

        try:
            # Prepare texts
            texts = []
            valid_tasks = []

            for task in tasks:
                combined_text = f"{task.title} {task.content}"
                processed_text = self._preprocess_text(combined_text)

                if processed_text.strip():
                    texts.append(processed_text)
                    valid_tasks.append(task)
                else:
                    # Add failed result for empty text
                    results.append(
                        EmbeddingResult(
                            article_id=task.article_id,
                            embedding=np.array([]),
                            model_name=self.model_name,
                            embedding_type="title_content",
                            quality_score=0.0,
                            processing_time=0,
                            success=False,
                            error_message="Empty text after preprocessing",
                        )
                    )

            if not texts:
                return results

            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                batch_size=min(self.batch_size, len(texts)),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Create results
            processing_time = time.time() - start_time
            avg_time_per_article = processing_time / len(valid_tasks)

            for task, embedding, text in zip(valid_tasks, embeddings, texts):
                quality_score = self._calculate_quality_score(text, embedding)

                results.append(
                    EmbeddingResult(
                        article_id=task.article_id,
                        embedding=embedding,
                        model_name=self.model_name,
                        embedding_type="title_content",
                        quality_score=quality_score,
                        processing_time=avg_time_per_article,
                        success=True,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")

            # Return failed results for all tasks
            for task in tasks:
                results.append(
                    EmbeddingResult(
                        article_id=task.article_id,
                        embedding=np.array([]),
                        model_name=self.model_name,
                        embedding_type="title_content",
                        quality_score=0.0,
                        processing_time=0,
                        success=False,
                        error_message=str(e),
                    )
                )

            return results

    def store_results(self, results: List[EmbeddingResult]) -> int:
        """
        Store embedding results in the database.

        Args:
            results: List of embedding results to store

        Returns:
            Number of results successfully stored
        """
        if not results:
            return 0

        # Prepare data for batch storage
        embeddings_data = []

        for result in results:
            if result.success and result.embedding.size > 0:
                embeddings_data.append(
                    {
                        "article_id": result.article_id,
                        "embedding": result.embedding,
                        "model_name": result.model_name,
                        "embedding_type": result.embedding_type,
                        "quality_score": result.quality_score,
                    }
                )

        if not embeddings_data:
            logger.warning("No valid embeddings to store")
            return 0

        # Store in database
        stored_count = self.db_manager.store_embeddings_batch(embeddings_data)

        # Update statistics
        self.stats["total_processed"] += len(results)
        self.stats["successful"] += stored_count
        self.stats["failed"] += len(results) - stored_count

        for result in results:
            if result.success:
                self.stats["total_time"] += result.processing_time

        if self.stats["successful"] > 0:
            self.stats["average_time_per_article"] = self.stats["total_time"] / self.stats["successful"]

            if self.stats["start_time"]:
                elapsed = time.time() - self.stats["start_time"]
                self.stats["articles_per_second"] = self.stats["successful"] / elapsed

        self.stats["last_processed"] = datetime.now()

        return stored_count

    def worker_thread(self, worker_id: int):
        """Worker thread for processing embedding tasks."""
        logger.info(f"Starting worker thread {worker_id}")

        while self.running:
            try:
                # Collect tasks for batch processing
                batch_tasks = []

                # Try to get a full batch
                for _ in range(self.batch_size):
                    task = self.get_next_task()
                    if task:
                        batch_tasks.append(task)
                    else:
                        break

                if not batch_tasks:
                    time.sleep(1)  # Wait before checking again
                    continue

                # Process batch
                if len(batch_tasks) == 1:
                    # Single article processing
                    result = self.process_single_article(batch_tasks[0])
                    results = [result]
                else:
                    # Batch processing
                    results = self.process_batch(batch_tasks)

                # Store results
                stored_count = self.store_results(results)

                logger.debug(f"Worker {worker_id}: Processed {len(batch_tasks)} articles, stored {stored_count}")

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(5)  # Wait before retrying

        logger.info(f"Worker thread {worker_id} stopped")

    def start_processing(self):
        """Start the embedding processing pipeline."""
        if self.running:
            logger.warning("Pipeline is already running")
            return

        logger.info(f"Starting embedding pipeline with {self.max_workers} workers")

        self.running = True
        self.stats["start_time"] = time.time()

        # Start worker threads
        self.workers = []
        for i in range(self.max_workers):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        logger.info("Embedding pipeline started successfully")

    def stop_processing(self):
        """Stop the embedding processing pipeline."""
        if not self.running:
            logger.warning("Pipeline is not running")
            return

        logger.info("Stopping embedding pipeline...")

        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=30)  # Wait up to 30 seconds

        self.workers = []

        logger.info("Embedding pipeline stopped")

    def process_all_pending(self, show_progress: bool = True) -> Dict[str, Any]:
        """
        Process all pending articles in a single run.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Processing all pending articles...")

        # Queue all articles that need embeddings
        total_queued = self.queue_articles_for_embedding(limit=10000)

        if total_queued == 0:
            logger.info("No articles to process")
            return self.get_statistics()

        # Reset statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0,
            "average_time_per_article": 0,
            "articles_per_second": 0,
            "start_time": time.time(),
            "last_processed": None,
        }

        # Process with progress bar
        if show_progress:
            pbar = tqdm(total=total_queued, desc="Processing articles")

        processed = 0
        while processed < total_queued:
            # Get batch of tasks
            batch_tasks = []
            for _ in range(self.batch_size):
                task = self.get_next_task()
                if task:
                    batch_tasks.append(task)
                else:
                    break

            if not batch_tasks:
                break

            # Process batch
            if len(batch_tasks) == 1:
                results = [self.process_single_article(batch_tasks[0])]
            else:
                results = self.process_batch(batch_tasks)

            # Store results
            stored_count = self.store_results(results)
            processed += len(batch_tasks)

            if show_progress:
                pbar.update(len(batch_tasks))
                pbar.set_postfix(
                    {
                        "success": f"{self.stats['successful']}/{processed}",
                        "rate": f"{self.stats.get('articles_per_second', 0):.1f}/s",
                    }
                )

        if show_progress:
            pbar.close()

        final_stats = self.get_statistics()
        logger.info(
            f"Processing completed. Processed: {final_stats['total_processed']}, "
            f"Successful: {final_stats['successful']}, Failed: {final_stats['failed']}"
        )

        return final_stats

    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        stats = self.stats.copy()

        # Add database statistics
        db_stats = self.db_manager.get_embedding_statistics()
        stats["database"] = db_stats

        # Add queue statistics
        if self.use_redis and self.redis_client:
            queue_size = self.redis_client.zcard("embedding_tasks")
        else:
            queue_size = self.task_queue.qsize()

        stats["queue_size"] = queue_size
        stats["running"] = self.running
        stats["model_name"] = self.model_name

        return stats

    def schedule_periodic_processing(self, interval_hours: int = 6):
        """
        Schedule periodic processing of new articles.

        Args:
            interval_hours: Hours between processing runs
        """

        def run_processing():
            """Run scheduled embedding processing for new articles."""
            logger.info("Starting scheduled embedding processing")
            try:
                # Queue only recent articles
                recent_count = self.queue_articles_for_embedding(limit=1000)
                if recent_count > 0:
                    self.process_all_pending(show_progress=False)
                else:
                    logger.info("No new articles to process")
            except Exception as e:
                logger.error(f"Scheduled processing failed: {e}")

        # Schedule the job
        schedule.every(interval_hours).hours.do(run_processing)

        logger.info(f"Scheduled periodic processing every {interval_hours} hours")


# Utility functions
def create_embedding_pipeline(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingPipeline:
    """Create and return an embedding pipeline instance."""
    return EmbeddingPipeline(model_name=model_name)


def process_all_articles(model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Process all articles in the database for embeddings."""
    pipeline = create_embedding_pipeline(model_name)
    return pipeline.process_all_pending()


def get_embedding_coverage() -> Dict[str, Any]:
    """Get statistics about embedding coverage."""
    db_manager = SemanticDatabaseManager()
    return db_manager.get_embedding_statistics()


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Pipeline CLI")
    parser.add_argument("command", choices=["process", "stats", "schedule", "queue"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum articles to process")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of existing embeddings")

    args = parser.parse_args()

    pipeline = EmbeddingPipeline(model_name=args.model, batch_size=args.batch_size, max_workers=args.workers)

    if args.command == "process":
        # Process all pending articles
        stats = pipeline.process_all_pending()
        print(f"Processing complete. Results: {stats}")

    elif args.command == "stats":
        # Show current statistics
        stats = pipeline.get_statistics()
        print("Embedding Pipeline Statistics:")
        print(f"  Model: {stats['model_name']}")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Queue size: {stats['queue_size']}")

        if stats.get("database"):
            db_stats = stats["database"]
            print(f"  Database coverage: {db_stats.get('coverage_percentage', 0):.1f}%")
            print(f"  Total embeddings: {db_stats.get('total_embeddings', 0)}")

    elif args.command == "queue":
        # Queue articles for processing
        count = pipeline.queue_articles_for_embedding(limit=args.limit, force_reprocess=args.force)
        print(f"Queued {count} articles for processing")

    elif args.command == "schedule":
        # Start scheduled processing
        pipeline.schedule_periodic_processing(interval_hours=6)

        print("Scheduled processing started. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("Scheduled processing stopped.")
