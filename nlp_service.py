"""
NLP Service Architecture for Mimir.
Implements queue-based processing with Redis and worker processes.
"""

import json
import logging
import multiprocessing
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import redis
from redis.exceptions import RedisError

from nlp_preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
TASK_RESULT_TTL_SECONDS = 86400  # 24 hours
RESULT_CHECK_INTERVAL = 0.1  # 100ms
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_NUM_WORKERS = 4
MAX_RETRIES = 3
DEFAULT_PRIORITY = 10


@dataclass
class NLPTask:
    """Represents an NLP processing task."""

    task_id: str
    task_type: str  # 'preprocess', 'entities', 'sentiment', 'keywords', 'full_analysis'
    article_id: str
    article_data: Dict
    priority: int = 5  # 1-10, higher is more priority
    created_at: float = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class NLPResult:
    """Represents the result of NLP processing."""

    task_id: str
    article_id: str
    task_type: str
    status: str  # 'success', 'failed', 'partial'
    results: Dict
    processing_time: float
    error: Optional[str] = None
    worker_id: Optional[str] = None


class NLPService:
    """Main NLP service that manages task queue and workers."""

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, num_workers: int = None
    ):
        """
        Initialize the NLP service.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            num_workers: Number of worker processes (defaults to CPU count - 1)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)

        # Redis keys
        self.task_queue_key = "mimir:nlp:tasks"
        self.result_queue_key = "mimir:nlp:results"
        self.processing_key = "mimir:nlp:processing"
        self.failed_key = "mimir:nlp:failed"
        self.cache_key_prefix = "mimir:nlp:cache:"

        # Connect to Redis
        self.redis_client = self._connect_redis()

        # Worker processes
        self.workers = []
        self.running = False

    def _connect_redis(self) -> redis.Redis:
        """Create Redis connection with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    decode_responses=False,  # We'll handle encoding/decoding
                )
                # Test connection
                client.ping()
                logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
                return client
            except RedisError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to Redis after {max_retries} attempts")
                    raise

    def submit_task(self, task: NLPTask) -> str:
        """
        Submit a task to the processing queue.

        Args:
            task: NLPTask to process

        Returns:
            Task ID
        """
        try:
            # Serialize task to JSON
            task_dict = asdict(task)
            task_dict["timestamp"] = task_dict["timestamp"].isoformat()
            task_data = json.dumps(task_dict)

            # Add to queue with priority (lower score = higher priority)
            score = -task.priority  # Negative so higher priority has lower score
            self.redis_client.zadd(self.task_queue_key, {task_data: score})

            logger.info(f"Submitted task {task.task_id} for article {task.article_id}")
            return task.task_id

        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise

    def get_result(self, task_id: str, timeout: int = 0) -> Optional[NLPResult]:
        """
        Get the result of a processed task.

        Args:
            task_id: Task ID to retrieve
            timeout: How long to wait for result (0 = don't wait)

        Returns:
            NLPResult if available, None otherwise
        """
        result_key = f"{self.result_queue_key}:{task_id}"

        if timeout > 0:
            # Wait for result with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                result_data = self.redis_client.get(result_key)
                if result_data:
                    result_dict = json.loads(result_data)
                    result_dict["timestamp"] = datetime.fromisoformat(result_dict["timestamp"])
                    return NLPResult(**result_dict)
                time.sleep(RESULT_CHECK_INTERVAL)
            return None
        else:
            # Try to get result immediately
            result_data = self.redis_client.get(result_key)
            if result_data:
                result_dict = json.loads(result_data)
                result_dict["timestamp"] = datetime.fromisoformat(result_dict["timestamp"])
                return NLPResult(**result_dict)
            return None

    def start_workers(self):
        """Start worker processes."""
        logger.info(f"Starting {self.num_workers} NLP workers...")
        self.running = True

        for i in range(self.num_workers):
            worker = NLPWorker(
                worker_id=f"worker-{i}", redis_host=self.redis_host, redis_port=self.redis_port, redis_db=self.redis_db
            )
            process = multiprocessing.Process(target=worker.run)
            process.start()
            self.workers.append(process)
            logger.info(f"Started worker {i} (PID: {process.pid})")

    def stop_workers(self):
        """Stop all worker processes."""
        logger.info("Stopping NLP workers...")
        self.running = False

        # Send stop signal to workers
        for _, _ in enumerate(self.workers):
            self.redis_client.zadd(self.task_queue_key, {b"STOP": -999})

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Worker {worker.pid} did not stop gracefully, terminating...")
                worker.terminate()

        self.workers.clear()
        logger.info("All workers stopped")

    def get_queue_stats(self) -> Dict:
        """Get statistics about the processing queues."""
        try:
            stats = {
                "pending_tasks": self.redis_client.zcard(self.task_queue_key),
                "processing_tasks": self.redis_client.hlen(self.processing_key),
                "failed_tasks": self.redis_client.llen(self.failed_key),
                "workers": len(self.workers),
                "workers_alive": sum(1 for w in self.workers if w.is_alive()),
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}


class NLPWorker:
    """Worker process that processes NLP tasks from the queue."""

    def __init__(self, worker_id: str, redis_host: str, redis_port: int, redis_db: int):
        """Initialize NLP worker."""
        self.worker_id = worker_id
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client = None
        self.preprocessor = None
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.running = False

    def _connect_redis(self):
        """Connect to Redis."""
        self.redis_client = redis.Redis(
            host=self.redis_host, port=self.redis_port, db=self.redis_db, decode_responses=False
        )

    def _initialize_models(self):
        """Initialize NLP models and processors."""
        logger.info(f"Worker {self.worker_id} initializing NLP models...")
        self.preprocessor = TextPreprocessor()
        # Additional model initialization will be added here
        logger.info(f"Worker {self.worker_id} ready")

    def run(self):
        """Main worker loop."""
        try:
            # Initialize connections and models
            self._connect_redis()
            self._initialize_models()

            while self.running:
                try:
                    # Get next task from queue (blocking with timeout)
                    result = self.redis_client.bzpopmin(self.task_queue_key, timeout=5)

                    if not result:
                        continue

                    _, task_data, score = result

                    # Check for stop signal
                    if task_data == b"STOP":
                        logger.info(f"Worker {self.worker_id} received stop signal")
                        break

                    # Deserialize task
                    task_dict = json.loads(task_data)
                    task_dict["timestamp"] = datetime.fromisoformat(task_dict["timestamp"])
                    task = NLPTask(**task_dict)

                    # Process task
                    self._process_task(task)

                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error in main loop: {e}")
                    time.sleep(1)

        except Exception as e:
            logger.error(f"Worker {self.worker_id} fatal error: {e}")
        finally:
            logger.info(f"Worker {self.worker_id} shutting down")

    def _process_task(self, task: NLPTask):
        """Process a single NLP task."""
        start_time = time.time()
        logger.info(f"Worker {self.worker_id} processing task {task.task_id}")

        try:
            # Mark task as being processed
            self.redis_client.hset(
                self.processing_key, task.task_id, json.dumps({"worker_id": self.worker_id, "started_at": time.time()})
            )

            # Process based on task type
            if task.task_type == "preprocess":
                result_data = self._preprocess_article(task.article_data)
            elif task.task_type == "full_analysis":
                result_data = self._full_analysis(task.article_data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

            # Create result
            processing_time = time.time() - start_time
            result = NLPResult(
                task_id=task.task_id,
                article_id=task.article_id,
                task_type=task.task_type,
                status="success",
                results=result_data,
                processing_time=processing_time,
                worker_id=self.worker_id,
            )

            # Store result
            result_key = f"mimir:nlp:results:{task.task_id}"
            result_dict = asdict(result)
            result_dict["timestamp"] = result_dict["timestamp"].isoformat()
            self.redis_client.setex(result_key, TASK_RESULT_TTL_SECONDS, json.dumps(result_dict))

            # Remove from processing
            self.redis_client.hdel(self.processing_key, task.task_id)

            logger.info(f"Worker {self.worker_id} completed task {task.task_id} " f"in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to process task {task.task_id}: {e}")

            # Handle retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.created_at = time.time()  # Reset timestamp

                # Re-queue with lower priority
                task.priority = max(1, task.priority - 1)
                task_dict = asdict(task)
                task_dict["timestamp"] = task_dict["timestamp"].isoformat()
                task_data = json.dumps(task_dict)
                self.redis_client.zadd("mimir:nlp:tasks", {task_data: -task.priority})
                logger.info(f"Re-queued task {task.task_id} (attempt {task.retry_count})")
            else:
                # Max retries exceeded, mark as failed
                failed_result = NLPResult(
                    task_id=task.task_id,
                    article_id=task.article_id,
                    task_type=task.task_type,
                    status="failed",
                    results={},
                    processing_time=time.time() - start_time,
                    error=str(e),
                    worker_id=self.worker_id,
                )

                # Store failed result
                failed_dict = asdict(failed_result)
                failed_dict["timestamp"] = failed_dict["timestamp"].isoformat()
                self.redis_client.lpush("mimir:nlp:failed", json.dumps(failed_dict))

            # Remove from processing
            self.redis_client.hdel(self.processing_key, task.task_id)

    def _preprocess_article(self, article_data: Dict) -> Dict:
        """Preprocess an article."""
        return self.preprocessor.preprocess_article(article_data)["preprocessed"]

    def _full_analysis(self, article_data: Dict) -> Dict:
        """Perform full NLP analysis on an article."""
        # For now, just do preprocessing
        # We'll add more analysis steps as we implement them
        results = {
            "preprocessing": self._preprocess_article(article_data),
            "entities": {},  # To be implemented
            "keywords": {},  # To be implemented
            "sentiment": {},  # To be implemented
            "topics": {},  # To be implemented
        }
        return results


if __name__ == "__main__":
    # Test the NLP service
    import uuid

    # Create service
    service = NLPService(num_workers=2)

    try:
        # Start workers
        service.start_workers()

        # Submit a test task
        test_article = {
            "id": "test-001",
            "title": "AI Technology Advances in 2024",
            "content": "Major breakthroughs in artificial intelligence...",
        }

        task = NLPTask(
            task_id=str(uuid.uuid4()),
            task_type="preprocess",
            article_id=test_article["id"],
            article_data=test_article,
            priority=8,
        )

        task_id = service.submit_task(task)
        logger.info(f"Submitted task: {task_id}")

        # Wait for result
        time.sleep(2)
        result = service.get_result(task_id, timeout=10)

        if result:
            logger.info(f"Got result: {result.status}")
            logger.info(f"Processing time: {result.processing_time:.2f}s")
            logger.info(f"Results: {json.dumps(result.results, indent=2)}")
        else:
            logger.warning("No result received")

        # Check queue stats
        stats = service.get_queue_stats()
        logger.info(f"Queue stats: {stats}")

    finally:
        # Stop workers
        service.stop_workers()
