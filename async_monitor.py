"""
Async Resource Monitor and Performance Metrics for Mimir Scraper
EPIC 4 - Task 4.1.3: Concurrent processing with resource monitoring
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional

import psutil


class ResourceSnapshot(NamedTuple):
    """Snapshot of system resources at a point in time."""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_connections: int
    active_tasks: int


@dataclass
class PerformanceMetrics:
    """Performance metrics for scraping operations."""

    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    total_bytes_downloaded: int = 0
    total_articles_processed: int = 0
    total_time_seconds: float = 0.0
    avg_response_time: float = 0.0
    peak_concurrent_requests: int = 0

    @property
    def success_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return (self.requests_successful / self.requests_total) * 100

    @property
    def throughput_requests_per_second(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return self.requests_total / self.total_time_seconds

    @property
    def throughput_articles_per_second(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_articles_processed / self.total_time_seconds


class AsyncResourceMonitor:
    """Monitor system resources and performance during async scraping."""

    def __init__(self, max_history_size: int = 1000, sample_interval: float = 1.0):
        self.max_history_size = max_history_size
        self.sample_interval = sample_interval
        self.logger = logging.getLogger(__name__)

        # Resource monitoring
        self.resource_history: deque = deque(maxlen=max_history_size)
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.response_times: deque = deque(maxlen=1000)  # Keep last 1000 response times
        self.current_concurrent_requests = 0

        # Alerts and thresholds
        self.cpu_threshold = 80.0  # CPU usage percentage
        self.memory_threshold = 80.0  # Memory usage percentage
        self.response_time_threshold = 10.0  # Response time in seconds

        # Process info
        self.process = psutil.Process()
        self.start_time = time.time()

    async def start_monitoring(self):
        """Start background monitoring task."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Resource monitoring started")

    async def stop_monitoring(self):
        """Stop background monitoring task."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self):
        """Background loop for collecting resource metrics."""
        try:
            while self.is_monitoring:
                await self._collect_resource_snapshot()

                # Prevent memory leaks by cleaning up old data
                await self._cleanup_old_data()

                await asyncio.sleep(self.sample_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")

    async def _cleanup_old_data(self):
        """Clean up old monitoring data to prevent memory leaks."""
        try:
            # Clean up old response times (keep only last 1000)
            if len(self.response_times) > 1000:
                # Keep last 1000 entries
                self.response_times = deque(list(self.response_times)[-1000:], maxlen=1000)

            # Clean up old resource history if too large
            if len(self.resource_history) > self.max_history_size:
                # Remove oldest 10% of entries
                cleanup_count = max(1, self.max_history_size // 10)
                for _ in range(cleanup_count):
                    if self.resource_history:
                        self.resource_history.popleft()
        except Exception as e:
            self.logger.warning(f"Error cleaning up monitoring data: {e}")

    async def _collect_resource_snapshot(self):
        """Collect current resource usage snapshot."""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Process-specific metrics
            process_memory = self.process.memory_info().rss / 1024 / 1024  # MB

            # Network connections
            try:
                connections = len(self.process.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0

            # Active tasks
            active_tasks = len([task for task in asyncio.all_tasks() if not task.done()])

            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=process_memory,
                memory_percent=memory.percent,
                active_connections=connections,
                active_tasks=active_tasks,
            )

            self.resource_history.append(snapshot)

            # Check for alerts
            await self._check_resource_alerts(snapshot)

        except Exception as e:
            self.logger.error(f"Error collecting resource snapshot: {e}")

    async def _check_resource_alerts(self, snapshot: ResourceSnapshot):
        """Check if any resource thresholds are exceeded."""
        alerts = []

        if snapshot.cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {snapshot.cpu_percent:.1f}%")

        if snapshot.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {snapshot.memory_percent:.1f}%")

        if alerts:
            self.logger.warning(f"Resource alerts: {', '.join(alerts)}")

    def record_request_start(self):
        """Record the start of a new request."""
        self.current_concurrent_requests += 1
        self.metrics.peak_concurrent_requests = max(
            self.metrics.peak_concurrent_requests, self.current_concurrent_requests
        )

    def record_request_end(self, success: bool, response_time: float, bytes_downloaded: int = 0):
        """Record the completion of a request."""
        self.current_concurrent_requests = max(0, self.current_concurrent_requests - 1)
        self.metrics.requests_total += 1

        if success:
            self.metrics.requests_successful += 1
        else:
            self.metrics.requests_failed += 1

        self.metrics.total_bytes_downloaded += bytes_downloaded
        self.response_times.append(response_time)

        # Update average response time
        if self.response_times:
            self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)

        # Check for slow response alert
        if response_time > self.response_time_threshold:
            self.logger.warning(f"Slow response detected: {response_time:.2f}s")

    def record_article_processed(self):
        """Record that an article has been processed."""
        self.metrics.total_articles_processed += 1

    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        current_time = time.time()
        self.metrics.total_time_seconds = current_time - self.start_time

        latest_snapshot = self.resource_history[-1] if self.resource_history else None

        return {
            "performance": {
                "requests_total": self.metrics.requests_total,
                "requests_successful": self.metrics.requests_successful,
                "requests_failed": self.metrics.requests_failed,
                "success_rate": self.metrics.success_rate,
                "total_bytes_downloaded": self.metrics.total_bytes_downloaded,
                "total_articles_processed": self.metrics.total_articles_processed,
                "total_time_seconds": self.metrics.total_time_seconds,
                "avg_response_time": self.metrics.avg_response_time,
                "throughput_rps": self.metrics.throughput_requests_per_second,
                "throughput_aps": self.metrics.throughput_articles_per_second,
                "peak_concurrent_requests": self.metrics.peak_concurrent_requests,
                "current_concurrent_requests": self.current_concurrent_requests,
            },
            "resources": {
                "cpu_percent": latest_snapshot.cpu_percent if latest_snapshot else 0,
                "memory_mb": latest_snapshot.memory_mb if latest_snapshot else 0,
                "memory_percent": latest_snapshot.memory_percent if latest_snapshot else 0,
                "active_connections": latest_snapshot.active_connections if latest_snapshot else 0,
                "active_tasks": latest_snapshot.active_tasks if latest_snapshot else 0,
            }
            if latest_snapshot
            else {},
            "timestamp": current_time,
        }

    def get_resource_history_summary(self, minutes: int = 5) -> Dict:
        """Get resource usage summary for the last N minutes."""
        if not self.resource_history:
            return {}

        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.resource_history if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {}

        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_mb for s in recent_snapshots]
        connection_values = [s.active_connections for s in recent_snapshots]

        return {
            "time_window_minutes": minutes,
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
            },
            "memory_mb": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
            },
            "connections": {
                "avg": sum(connection_values) / len(connection_values),
                "min": min(connection_values),
                "max": max(connection_values),
            },
            "samples_count": len(recent_snapshots),
        }

    def export_metrics(self, filename: str):
        """Export current metrics to JSON file."""
        metrics_data = {
            "export_timestamp": datetime.now().isoformat(),
            "current_metrics": self.get_current_metrics(),
            "resource_history_summary": self.get_resource_history_summary(),
            "response_time_percentiles": self._calculate_response_time_percentiles(),
        }

        try:
            with open(filename, "w") as f:
                json.dump(metrics_data, f, indent=2)
            self.logger.info(f"Metrics exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")

    def _calculate_response_time_percentiles(self) -> Dict:
        """Calculate response time percentiles."""
        if not self.response_times:
            return {}

        sorted_times = sorted(self.response_times)
        length = len(sorted_times)

        percentiles = [50, 75, 90, 95, 99]
        result = {}

        for p in percentiles:
            index = int((p / 100) * length) - 1
            if index < 0:
                index = 0
            elif index >= length:
                index = length - 1
            result[f"p{p}"] = sorted_times[index]

        return result

    def print_status_report(self):
        """Print a comprehensive status report."""
        metrics = self.get_current_metrics()
        history = self.get_resource_history_summary()
        percentiles = self._calculate_response_time_percentiles()

        print("\n" + "=" * 60)
        print("ASYNC SCRAPER PERFORMANCE REPORT")
        print("=" * 60)

        # Performance metrics
        perf = metrics["performance"]
        print(f"Requests: {perf['requests_total']} total, {perf['requests_successful']} successful")
        print(f"Success Rate: {perf['success_rate']:.1f}%")
        print(f"Articles Processed: {perf['total_articles_processed']}")
        print(f"Throughput: {perf['throughput_rps']:.2f} req/s, {perf['throughput_aps']:.2f} articles/s")
        print(f"Avg Response Time: {perf['avg_response_time']:.2f}s")
        print(f"Concurrent Requests: {perf['current_concurrent_requests']} (peak: {perf['peak_concurrent_requests']})")
        print(f"Data Downloaded: {perf['total_bytes_downloaded'] / 1024 / 1024:.2f} MB")

        # Resource usage
        if metrics["resources"]:
            res = metrics["resources"]
            print(f"\nCurrent Resources:")
            print(f"CPU: {res['cpu_percent']:.1f}%, Memory: {res['memory_mb']:.1f} MB ({res['memory_percent']:.1f}%)")
            print(f"Connections: {res['active_connections']}, Tasks: {res['active_tasks']}")

        # Resource history
        if history:
            print(f"\nLast 5 Minutes:")
            print(f"CPU: {history['cpu']['avg']:.1f}% avg ({history['cpu']['min']:.1f}-{history['cpu']['max']:.1f}%)")
            print(f"Memory: {history['memory_mb']['avg']:.1f} MB avg")

        # Response time percentiles
        if percentiles:
            print(f"\nResponse Time Percentiles:")
            for p, time_val in percentiles.items():
                print(f"{p}: {time_val:.2f}s")

        print("=" * 60 + "\n")


class ConcurrencyController:
    """Advanced concurrency control with adaptive limits."""

    def __init__(self, initial_limit: int = 10, min_limit: int = 2, max_limit: int = 50):
        self.initial_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.current_limit = initial_limit

        self.semaphore = asyncio.Semaphore(initial_limit)
        self.monitor = AsyncResourceMonitor()
        self.logger = logging.getLogger(__name__)

        # Adaptive control parameters
        self.error_rate_threshold = 0.1  # 10% error rate
        self.response_time_threshold = 5.0  # 5 seconds
        self.adjustment_interval = 30  # seconds
        self.last_adjustment = time.time()

    async def acquire(self):
        """Acquire semaphore with adaptive control."""
        await self.semaphore.acquire()
        await self._maybe_adjust_limits()

    def release(self):
        """Release semaphore."""
        self.semaphore.release()

    async def _maybe_adjust_limits(self):
        """Adjust concurrency limits based on performance metrics."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return

        metrics = self.monitor.get_current_metrics()
        perf = metrics["performance"]

        # Skip if not enough data
        if perf["requests_total"] < 10:
            return

        # Calculate current error rate
        error_rate = perf["requests_failed"] / perf["requests_total"]
        avg_response_time = perf["avg_response_time"]

        # Decide on adjustment
        should_decrease = error_rate > self.error_rate_threshold or avg_response_time > self.response_time_threshold

        should_increase = (
            error_rate < 0.05
            and avg_response_time < 2.0  # Less than 5% error rate
            and self.current_limit < self.max_limit  # Fast responses
        )

        if should_decrease and self.current_limit > self.min_limit:
            new_limit = max(self.min_limit, int(self.current_limit * 0.8))
            await self._adjust_semaphore_limit(new_limit)
            self.logger.info(
                f"Decreased concurrency limit to {new_limit} (error_rate={error_rate:.3f}, resp_time={avg_response_time:.2f}s)"
            )

        elif should_increase:
            new_limit = min(self.max_limit, self.current_limit + 2)
            await self._adjust_semaphore_limit(new_limit)
            self.logger.info(f"Increased concurrency limit to {new_limit}")

        self.last_adjustment = now

    async def _adjust_semaphore_limit(self, new_limit: int):
        """Adjust semaphore limit by creating a new semaphore."""
        # This is a simplified approach - in production you might want a more sophisticated method
        old_limit = self.current_limit
        self.current_limit = new_limit

        # Create new semaphore with new limit
        # Note: This doesn't preserve current waiters, which is a limitation
        # In production, consider using a more sophisticated semaphore implementation
        self.semaphore = asyncio.Semaphore(new_limit)


# Context manager for easy monitoring
class MonitoredAsyncSession:
    """Context manager that combines async session with monitoring."""

    def __init__(self, monitor: AsyncResourceMonitor):
        self.monitor = monitor

    async def __aenter__(self):
        await self.monitor.start_monitoring()
        return self.monitor

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.monitor.stop_monitoring()

        # Print final report
        self.monitor.print_status_report()

        # Export metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor.export_metrics(f"logs/scraper_metrics_{timestamp}.json")
