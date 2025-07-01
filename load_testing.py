"""
Load Testing Framework for Mimir Async Scraper
EPIC 4 - Task 4.5.1: Locust scenarios, stress testing, performance regression tests
"""

import asyncio
import json
import random
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import psutil
import redis
import requests
from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

# Configuration for load testing
LOAD_TEST_CONFIG = {
    "base_urls": [
        "http://localhost:5000",  # Flask web interface
        "http://localhost:8000",  # FastAPI if implemented
    ],
    "redis_url": "redis://localhost:6379/0",
    "test_duration": 300,  # 5 minutes
    "max_users": 100,
    "spawn_rate": 5,  # users per second
}


class ScraperAPIUser(HttpUser):
    """Load test user for Scraper API endpoints."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Called when user starts."""
        # Initialize any user-specific data
        self.session_id = f"session_{random.randint(1000, 9999)}"
        self.start_time = time.time()

    @task(3)
    def get_articles(self):
        """Test getting articles list."""
        params = {"limit": random.randint(10, 50), "offset": random.randint(0, 100)}

        with self.client.get("/api/articles", params=params, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "articles" in data:
                        response.success()
                    else:
                        response.failure("Missing articles in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def search_articles(self):
        """Test article search functionality."""
        search_terms = ["technology", "news", "business", "python", "AI", "machine learning"]
        search_term = random.choice(search_terms)

        params = {"q": search_term, "limit": 20}

        with self.client.get("/api/search", params=params, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "results" in data:
                        response.success()
                    else:
                        response.failure("Missing results in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_statistics(self):
        """Test statistics endpoint."""
        with self.client.get("/api/statistics", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "total_articles" in data:
                        response.success()
                    else:
                        response.failure("Missing statistics in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def get_health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class CeleryTaskUser(HttpUser):
    """Load test user for Celery task endpoints."""

    wait_time = between(2, 5)

    @task(1)
    def trigger_scraping_task(self):
        """Test triggering a scraping task."""
        payload = {"sites": ["test_site_1", "test_site_2"], "priority": random.choice(["low", "medium", "high"])}

        with self.client.post("/api/scrape", json=payload, catch_response=True) as response:
            if response.status_code in [200, 202]:  # Accepted for async processing
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def check_task_status(self):
        """Test checking task status."""
        # Generate a random task ID for testing
        task_id = f"task_{random.randint(1000, 9999)}"

        with self.client.get(f"/api/tasks/{task_id}", catch_response=True) as response:
            if response.status_code in [200, 404]:  # 404 is acceptable for non-existent tasks
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class AsyncScraperLoadTest:
    """Comprehensive load testing for async scraper components."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or LOAD_TEST_CONFIG
        self.redis_client = None
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "total_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "max_response_time": 0,
            "min_response_time": float("inf"),
            "response_times": [],
            "system_metrics": [],
            "error_details": [],
        }

    def setup(self):
        """Setup load testing environment."""
        try:
            self.redis_client = redis.from_url(self.config["redis_url"])
            self.redis_client.ping()
            print("✓ Redis connection established")
        except Exception as e:
            print(f"✗ Redis connection failed: {e}")
            self.redis_client = None

        # Clear any existing test data
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                print("✓ Redis test database cleared")
            except Exception as e:
                print(f"✗ Failed to clear Redis: {e}")

    def collect_system_metrics(self):
        """Collect system performance metrics during test."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "network_io": psutil.net_io_counters()._asdict(),
            "process_count": len(psutil.pids()),
        }

        # Add Redis metrics if available
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                metrics["redis_memory"] = redis_info.get("used_memory", 0)
                metrics["redis_connections"] = redis_info.get("connected_clients", 0)
                metrics["redis_commands"] = redis_info.get("total_commands_processed", 0)
            except Exception:
                pass

        return metrics

    async def stress_test_async_components(self):
        """Stress test async scraper components directly."""
        print("\n=== Async Components Stress Test ===")

        from async_scraper import AsyncNewsScraper

        # Load test configuration
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            with open("sites_to_scrape.json", "r") as f:
                sites = json.load(f)
        except FileNotFoundError:
            print("✗ Configuration files not found")
            return

        # Create multiple concurrent scraper instances
        concurrent_scrapers = 5
        tasks = []

        for i in range(concurrent_scrapers):
            # Create a subset of sites for each scraper
            scraper_sites = sites[:2]  # Limit to 2 sites per scraper for testing
            task = self.run_single_scraper(config, scraper_sites, f"scraper_{i}")
            tasks.append(task)

        # Run all scrapers concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        # Analyze results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        print(f"\nStress Test Results:")
        print(f"Concurrent scrapers: {concurrent_scrapers}")
        print(f"Duration: {duration:.2f}s")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        return {
            "concurrent_scrapers": concurrent_scrapers,
            "duration": duration,
            "successful": successful,
            "failed": failed,
            "results": [r for r in results if not isinstance(r, Exception)],
        }

    async def run_single_scraper(self, config: Dict, sites: List[Dict], scraper_id: str):
        """Run a single scraper instance for testing."""
        try:
            from async_scraper import AsyncNewsScraper

            async with AsyncNewsScraper(config, sites) as scraper:
                return await scraper.run_scraper()
        except Exception as e:
            print(f"Scraper {scraper_id} failed: {e}")
            raise e

    def performance_regression_test(self):
        """Run performance regression tests."""
        print("\n=== Performance Regression Test ===")

        # Test scenarios with increasing load
        scenarios = [
            {"users": 10, "duration": 60},
            {"users": 25, "duration": 60},
            {"users": 50, "duration": 60},
            {"users": 100, "duration": 60},
        ]

        regression_results = []

        for i, scenario in enumerate(scenarios):
            print(f"\nRunning scenario {i+1}: {scenario['users']} users for {scenario['duration']}s")

            # Simulate load test results (in real implementation, this would trigger actual Locust tests)
            result = self.simulate_load_test_result(scenario)
            regression_results.append(result)

            # Check for performance degradation
            if i > 0:
                prev_result = regression_results[i - 1]
                current_result = result

                # Calculate performance change
                response_time_change = (
                    (current_result["avg_response_time"] - prev_result["avg_response_time"])
                    / prev_result["avg_response_time"]
                    * 100
                )
                throughput_change = (
                    (current_result["throughput"] - prev_result["throughput"]) / prev_result["throughput"] * 100
                )

                print(f"  Response time change: {response_time_change:+.1f}%")
                print(f"  Throughput change: {throughput_change:+.1f}%")

                # Alert on significant degradation
                if response_time_change > 50:  # 50% increase in response time
                    print(f"  ⚠️  WARNING: Significant response time degradation detected!")
                if throughput_change < -25:  # 25% decrease in throughput
                    print(f"  ⚠️  WARNING: Significant throughput degradation detected!")

        return regression_results

    def simulate_load_test_result(self, scenario: Dict) -> Dict:
        """Simulate load test results (replace with actual Locust integration)."""
        # This is a simulation - in real implementation, this would run actual Locust tests
        base_response_time = 0.5  # 500ms base
        base_throughput = 100  # 100 requests/second base

        # Simulate performance degradation with increased load
        load_factor = scenario["users"] / 10
        response_time = base_response_time * (1 + load_factor * 0.1)
        throughput = base_throughput / (1 + load_factor * 0.05)

        # Add some randomness
        response_time *= random.uniform(0.9, 1.1)
        throughput *= random.uniform(0.95, 1.05)

        return {
            "users": scenario["users"],
            "duration": scenario["duration"],
            "avg_response_time": response_time,
            "throughput": throughput,
            "total_requests": int(throughput * scenario["duration"]),
            "failed_requests": random.randint(0, int(throughput * scenario["duration"] * 0.02)),  # 2% max failure rate
        }

    def redis_performance_test(self):
        """Test Redis performance under load."""
        print("\n=== Redis Performance Test ===")

        if not self.redis_client:
            print("✗ Redis not available for testing")
            return None

        # Test Redis operations
        operations = 1000
        start_time = time.time()

        # Test SET operations
        for i in range(operations):
            self.redis_client.set(f"test_key_{i}", f"test_value_{i}")

        set_duration = time.time() - start_time

        # Test GET operations
        start_time = time.time()
        for i in range(operations):
            self.redis_client.get(f"test_key_{i}")

        get_duration = time.time() - start_time

        # Test pipeline operations
        start_time = time.time()
        pipe = self.redis_client.pipeline()
        for i in range(operations):
            pipe.set(f"pipeline_key_{i}", f"pipeline_value_{i}")
        pipe.execute()

        pipeline_duration = time.time() - start_time

        # Cleanup
        self.redis_client.flushdb()

        redis_results = {
            "operations_count": operations,
            "set_ops_per_second": operations / set_duration,
            "get_ops_per_second": operations / get_duration,
            "pipeline_ops_per_second": operations / pipeline_duration,
            "set_avg_latency": set_duration / operations * 1000,  # ms
            "get_avg_latency": get_duration / operations * 1000,  # ms
            "pipeline_avg_latency": pipeline_duration / operations * 1000,  # ms
        }

        print(f"Redis Performance Results ({operations} operations):")
        print(f"  SET: {redis_results['set_ops_per_second']:.0f} ops/sec, {redis_results['set_avg_latency']:.2f}ms avg")
        print(f"  GET: {redis_results['get_ops_per_second']:.0f} ops/sec, {redis_results['get_avg_latency']:.2f}ms avg")
        print(
            f"  Pipeline: {redis_results['pipeline_ops_per_second']:.0f} ops/sec, {redis_results['pipeline_avg_latency']:.2f}ms avg"
        )

        return redis_results

    def run_comprehensive_test(self):
        """Run comprehensive load and performance tests."""
        print("🚀 Starting Comprehensive Load Testing")
        print("=" * 50)

        self.setup()

        # Collect baseline system metrics
        baseline_metrics = self.collect_system_metrics()
        print(f"Baseline CPU: {baseline_metrics['cpu_percent']:.1f}%")
        print(f"Baseline Memory: {baseline_metrics['memory_percent']:.1f}%")

        # Run tests
        results = {
            "baseline_metrics": baseline_metrics,
            "redis_performance": self.redis_performance_test(),
            "regression_test": self.performance_regression_test(),
            "test_timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_file = f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"reports/{results_file}", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Load testing completed. Results saved to {results_file}")

        return results


# Event handlers for Locust
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Handle Locust request events."""
    if exception:
        print(f"Request failed: {name} - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Handle test start event."""
    print(f"Load test started with {environment.runner.user_count} users")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Handle test stop event."""
    print(f"Load test stopped. Total requests: {environment.runner.stats.total.num_requests}")


# CLI for running load tests
def run_locust_tests():
    """Run Locust load tests via command line."""
    import os
    import subprocess

    # Locust configuration
    locust_config = [
        "locust",
        "-f",
        __file__,
        "--host",
        LOAD_TEST_CONFIG["base_urls"][0],
        "--users",
        str(LOAD_TEST_CONFIG["max_users"]),
        "--spawn-rate",
        str(LOAD_TEST_CONFIG["spawn_rate"]),
        "--run-time",
        f"{LOAD_TEST_CONFIG['test_duration']}s",
        "--headless",  # Run without web UI
        "--html",
        f"reports/locust_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    ]

    print("Running Locust load tests...")
    try:
        result = subprocess.run(locust_config, capture_output=True, text=True)
        print("Locust output:", result.stdout)
        if result.stderr:
            print("Locust errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run Locust: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mimir Load Testing")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--async-stress", action="store_true", help="Run async stress tests")
    parser.add_argument("--redis-only", action="store_true", help="Run Redis tests only")
    parser.add_argument("--locust", action="store_true", help="Run Locust load tests")

    args = parser.parse_args()

    # Create reports directory
    import os

    os.makedirs("reports", exist_ok=True)

    tester = AsyncScraperLoadTest()

    if args.comprehensive:
        tester.run_comprehensive_test()
    elif args.async_stress:
        asyncio.run(tester.stress_test_async_components())
    elif args.redis_only:
        tester.setup()
        tester.redis_performance_test()
    elif args.locust:
        run_locust_tests()
    else:
        print("Please specify a test type. Use --help for options.")
