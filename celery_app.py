"""
Celery Application Configuration for Mimir Async Scraper
EPIC 4 - Task 4.2.1: Celery infrastructure with Redis broker
"""

import json
import logging
import os
from datetime import datetime, timedelta

import redis
from celery import Celery
from celery.signals import task_postrun, task_prerun, worker_ready
from celery.utils.log import get_task_logger
from kombu import Queue

# Load configuration
CONFIG_FILE = "config.json"
try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", CONFIG.get("redis", {}).get("host", "localhost"))
REDIS_PORT = os.getenv("REDIS_PORT", CONFIG.get("redis", {}).get("port", 6379))
REDIS_DB = os.getenv("REDIS_DB", CONFIG.get("redis", {}).get("db", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", CONFIG.get("redis", {}).get("password", None))

# Construct Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Create Celery app
app = Celery("mimir_scraper")

# Celery configuration
app.conf.update(
    # Broker settings
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    # Task routing
    task_routes={
        "mimir_scraper.scrape_site": {"queue": "scraping"},
        "mimir_scraper.scrape_article": {"queue": "scraping"},
        "mimir_scraper.process_ai_analysis": {"queue": "ai_processing"},
        "mimir_scraper.send_notification": {"queue": "notifications"},
        "mimir_scraper.cleanup_old_data": {"queue": "maintenance"},
        "mimir_scraper.generate_report": {"queue": "reports"},
    },
    # Task queues with different priorities
    task_default_queue="default",
    task_queues=(
        Queue("scraping", routing_key="scraping"),
        Queue("ai_processing", routing_key="ai_processing"),
        Queue("notifications", routing_key="notifications"),
        Queue("maintenance", routing_key="maintenance"),
        Queue("reports", routing_key="reports"),
        Queue("default", routing_key="default"),
    ),
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_transport_options={"master_name": "mimir"},
    # Beat scheduler settings (for periodic tasks)
    beat_schedule={
        "scheduled-scraping": {
            "task": "mimir_scraper.run_scheduled_scraping",
            "schedule": timedelta(hours=4),  # Run every 4 hours
            "options": {"queue": "scraping"},
        },
        "cleanup-old-data": {
            "task": "mimir_scraper.cleanup_old_data",
            "schedule": timedelta(days=1),  # Run daily
            "options": {"queue": "maintenance"},
        },
        "generate-daily-report": {
            "task": "mimir_scraper.generate_daily_report",
            "schedule": timedelta(days=1),  # Run daily
            "options": {"queue": "reports"},
        },
    },
    # Monitoring and logging
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Error handling
    task_annotations={
        "*": {
            "rate_limit": "100/m",  # 100 tasks per minute default limit
            "retry_policy": {
                "max_retries": 3,
                "interval_start": 0,
                "interval_step": 0.2,
                "interval_max": 0.2,
            },
        },
        "mimir_scraper.scrape_site": {
            "rate_limit": "50/m",  # Lower rate for scraping tasks
            "time_limit": 1800,  # 30 minutes for site scraping
        },
        "mimir_scraper.process_ai_analysis": {
            "rate_limit": "20/m",  # Even lower for AI processing
            "time_limit": 300,  # 5 minutes for AI tasks
        },
    },
)


# Task tracking and metrics
class TaskMetrics:
    """Track task execution metrics."""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(REDIS_URL)
        self.metrics_key_prefix = "mimir:task_metrics"

    def record_task_start(self, task_name: str, task_id: str):
        """Record task start."""
        metrics_key = f"{self.metrics_key_prefix}:{task_name}"

        # Increment counters
        pipe = self.redis_client.pipeline()
        pipe.hincrby(metrics_key, "started", 1)
        pipe.hset(metrics_key, "last_started", datetime.now().isoformat())
        pipe.sadd(f"{metrics_key}:active", task_id)
        pipe.expire(metrics_key, 86400)  # Expire after 24 hours
        pipe.execute()

    def record_task_end(self, task_name: str, task_id: str, success: bool, duration: float):
        """Record task completion."""
        metrics_key = f"{self.metrics_key_prefix}:{task_name}"

        # Update counters
        pipe = self.redis_client.pipeline()
        if success:
            pipe.hincrby(metrics_key, "completed", 1)
        else:
            pipe.hincrby(metrics_key, "failed", 1)

        pipe.hset(metrics_key, "last_completed", datetime.now().isoformat())
        pipe.lpush(f"{metrics_key}:durations", duration)
        pipe.ltrim(f"{metrics_key}:durations", 0, 99)  # Keep last 100 durations
        pipe.srem(f"{metrics_key}:active", task_id)
        pipe.execute()

    def get_task_metrics(self, task_name: str) -> dict:
        """Get metrics for a specific task."""
        metrics_key = f"{self.metrics_key_prefix}:{task_name}"

        # Get basic metrics
        metrics = self.redis_client.hgetall(metrics_key)
        if not metrics:
            return {}

        # Convert bytes to strings and numbers
        result = {}
        for key, value in metrics.items():
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            value_str = value.decode("utf-8") if isinstance(value, bytes) else value

            if key_str in ["started", "completed", "failed"]:
                result[key_str] = int(value_str) if value_str else 0
            else:
                result[key_str] = value_str

        # Get active task count
        active_count = self.redis_client.scard(f"{metrics_key}:active")
        result["active"] = active_count

        # Get recent durations for average
        durations = self.redis_client.lrange(f"{metrics_key}:durations", 0, -1)
        if durations:
            duration_values = [float(d) for d in durations]
            result["avg_duration"] = sum(duration_values) / len(duration_values)
            result["last_duration"] = duration_values[0] if duration_values else 0
        else:
            result["avg_duration"] = 0
            result["last_duration"] = 0

        return result


# Global metrics instance
task_metrics = TaskMetrics()


# Celery signals for monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Called before task execution."""
    logger = get_task_logger(__name__)
    logger.info(f"Starting task {task.name} [{task_id}]")
    task_metrics.record_task_start(task.name, task_id)


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Called after task execution."""
    logger = get_task_logger(__name__)

    # Calculate duration
    duration = 0  # Would need to track start time for accurate duration
    success = state == "SUCCESS"

    logger.info(f"Finished task {task.name} [{task_id}] - State: {state}")
    task_metrics.record_task_end(task.name, task_id, success, duration)


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker is ready."""
    logger = get_task_logger(__name__)
    logger.info(f"Worker {sender.hostname} is ready")


# Health check task
@app.task(bind=True)
def health_check(self):
    """Health check task for monitoring."""
    try:
        # Test Redis connection
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()

        # Test database connection
        import db_manager

        db_manager.init_db()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "worker_id": self.request.id,
            "redis_connected": True,
            "database_connected": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "worker_id": self.request.id,
        }


# Utility functions for task management
def get_queue_length(queue_name: str) -> int:
    """Get the number of tasks in a queue."""
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        return redis_client.llen(f"celery:{queue_name}")
    except Exception:
        return -1


def get_active_tasks() -> list:
    """Get list of currently active tasks."""
    try:
        i = app.control.inspect()
        active = i.active()
        return active if active else {}
    except Exception:
        return {}


def get_worker_stats() -> dict:
    """Get worker statistics."""
    try:
        i = app.control.inspect()
        stats = i.stats()
        return stats if stats else {}
    except Exception:
        return {}


def purge_queue(queue_name: str) -> int:
    """Purge all tasks from a queue."""
    try:
        return app.control.purge()
    except Exception:
        return 0


# Configure logging for Celery
def setup_celery_logging():
    """Setup logging for Celery workers."""
    log_config = CONFIG.get("logging", {})

    # Configure Celery logger
    celery_logger = logging.getLogger("celery")
    celery_logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

    # Add file handler if configured
    if log_config.get("file"):
        handler = logging.FileHandler(log_config["file"].replace(".log", "_celery.log"))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        celery_logger.addHandler(handler)


# Initialize logging
setup_celery_logging()

if __name__ == "__main__":
    app.start()
