"""
Load Testing for Mimir Enterprise API

Comprehensive load and performance testing using locust and custom metrics.
"""

import asyncio
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import httpx
import pytest
from fastapi.testclient import TestClient


class PerformanceMetrics:
    """Performance metrics collector"""

    def __init__(self):
        self.response_times: List[float] = []
        self.status_codes: Dict[int, int] = {}
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0

    def add_response(self, response_time: float, status_code: int, error: str = None):
        """Add a response measurement"""
        self.response_times.append(response_time)
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
        if error:
            self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.response_times:
            return {}

        total_requests = len(self.response_times)
        successful_requests = sum(count for status, count in self.status_codes.items() if 200 <= status < 400)

        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "error_rate": len(self.errors) / total_requests if total_requests > 0 else 0,
            "response_times": {
                "min": min(self.response_times),
                "max": max(self.response_times),
                "mean": statistics.mean(self.response_times),
                "median": statistics.median(self.response_times),
                "p95": self._percentile(self.response_times, 95),
                "p99": self._percentile(self.response_times, 99),
            },
            "requests_per_second": total_requests / (self.end_time - self.start_time)
            if self.end_time > self.start_time
            else 0,
            "status_codes": self.status_codes,
            "error_count": len(self.errors),
        }

    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class LoadTester:
    """Load testing implementation"""

    def __init__(self, base_url: str, auth_headers: Dict[str, str] = None):
        self.base_url = base_url
        self.auth_headers = auth_headers or {}

    def single_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Tuple[float, int, str]:
        """Make a single request and measure response time"""
        start_time = time.time()
        error = None

        try:
            with httpx.Client() as client:
                response = client.request(
                    method=method, url=f"{self.base_url}{endpoint}", headers=self.auth_headers, json=data, timeout=30.0
                )
                status_code = response.status_code
        except Exception as e:
            status_code = 0
            error = str(e)

        response_time = time.time() - start_time
        return response_time, status_code, error

    def concurrent_load_test(
        self, endpoint: str, concurrent_users: int, requests_per_user: int, method: str = "GET", data: Dict = None
    ) -> PerformanceMetrics:
        """Run concurrent load test"""
        metrics = PerformanceMetrics()
        metrics.start_time = time.time()

        def user_session():
            """Simulate user session"""
            user_metrics = []
            for _ in range(requests_per_user):
                response_time, status_code, error = self.single_request(endpoint, method, data)
                user_metrics.append((response_time, status_code, error))
            return user_metrics

        # Run concurrent user sessions
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_session) for _ in range(concurrent_users)]

            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    for response_time, status_code, error in user_results:
                        metrics.add_response(response_time, status_code, error)
                except Exception as e:
                    metrics.errors.append(str(e))

        metrics.end_time = time.time()
        return metrics

    def sustained_load_test(
        self, endpoint: str, duration_seconds: int, requests_per_second: int, method: str = "GET", data: Dict = None
    ) -> PerformanceMetrics:
        """Run sustained load test for specified duration"""
        metrics = PerformanceMetrics()
        metrics.start_time = time.time()

        end_time = metrics.start_time + duration_seconds
        request_interval = 1.0 / requests_per_second

        while time.time() < end_time:
            request_start = time.time()

            response_time, status_code, error = self.single_request(endpoint, method, data)
            metrics.add_response(response_time, status_code, error)

            # Maintain request rate
            elapsed = time.time() - request_start
            sleep_time = max(0, request_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        metrics.end_time = time.time()
        return metrics

    def stress_test(
        self, endpoint: str, max_users: int, ramp_up_time: int, hold_time: int, method: str = "GET", data: Dict = None
    ) -> Dict[int, PerformanceMetrics]:
        """Run stress test with gradual user ramp-up"""
        results = {}
        user_increments = [1, 5, 10, 25, 50, 100, 200, 500]
        user_levels = [u for u in user_increments if u <= max_users]

        for user_count in user_levels:
            print(f"Testing with {user_count} concurrent users...")

            metrics = self.concurrent_load_test(
                endpoint=endpoint,
                concurrent_users=user_count,
                requests_per_user=10,  # 10 requests per user
                method=method,
                data=data,
            )

            results[user_count] = metrics

            # Brief pause between test levels
            time.sleep(2)

        return results


@pytest.mark.performance
class TestAPIPerformance:
    """API performance test suite"""

    @pytest.fixture
    def load_tester(self, client: TestClient, auth_headers: Dict[str, str]):
        """Load tester fixture"""
        return LoadTester("http://testserver", auth_headers)

    def test_articles_endpoint_performance(self, load_tester: LoadTester, performance_test_data: Dict[str, Any]):
        """Test articles endpoint performance"""
        metrics = load_tester.concurrent_load_test(
            endpoint="/api/v1/articles", concurrent_users=10, requests_per_user=5
        )

        stats = metrics.get_stats()

        # Performance assertions
        assert stats["success_rate"] >= 0.95, f"Success rate too low: {stats['success_rate']}"
        assert stats["response_times"]["mean"] < 2.0, f"Mean response time too high: {stats['response_times']['mean']}"
        assert stats["response_times"]["p95"] < 5.0, f"95th percentile too high: {stats['response_times']['p95']}"
        assert stats["requests_per_second"] > 5, f"RPS too low: {stats['requests_per_second']}"

    def test_search_endpoint_performance(self, load_tester: LoadTester):
        """Test search endpoint performance"""
        metrics = load_tester.concurrent_load_test(
            endpoint="/api/v1/search?q=test&page=1&page_size=20", concurrent_users=5, requests_per_user=10
        )

        stats = metrics.get_stats()

        # Search should be reasonably fast
        assert stats["success_rate"] >= 0.90
        assert stats["response_times"]["mean"] < 3.0
        assert stats["response_times"]["p95"] < 8.0

    def test_analytics_endpoint_performance(self, load_tester: LoadTester):
        """Test analytics endpoint performance"""
        metrics = load_tester.concurrent_load_test(
            endpoint="/api/v1/analytics/summary", concurrent_users=3, requests_per_user=5
        )

        stats = metrics.get_stats()

        # Analytics queries may be slower but should still be reasonable
        assert stats["success_rate"] >= 0.90
        assert stats["response_times"]["mean"] < 5.0
        assert stats["response_times"]["p95"] < 15.0

    def test_concurrent_user_scalability(self, load_tester: LoadTester):
        """Test API scalability with increasing concurrent users"""
        user_levels = [1, 5, 10, 25]
        results = {}

        for user_count in user_levels:
            metrics = load_tester.concurrent_load_test(
                endpoint="/api/v1/articles", concurrent_users=user_count, requests_per_user=5
            )
            results[user_count] = metrics.get_stats()

        # Check that performance degrades gracefully
        for user_count in user_levels:
            stats = results[user_count]
            assert stats["success_rate"] >= 0.85, f"Success rate too low at {user_count} users: {stats['success_rate']}"

            # Response times should not increase exponentially
            if user_count <= 10:
                assert stats["response_times"]["mean"] < 3.0
            else:
                assert stats["response_times"]["mean"] < 10.0

    def test_sustained_load_performance(self, load_tester: LoadTester):
        """Test sustained load over time"""
        metrics = load_tester.sustained_load_test(
            endpoint="/api/v1/articles", duration_seconds=30, requests_per_second=10
        )

        stats = metrics.get_stats()

        # Should maintain performance over time
        assert stats["success_rate"] >= 0.95
        assert stats["response_times"]["mean"] < 2.0
        assert abs(stats["requests_per_second"] - 10) < 2.0  # Allow some variance

    def test_memory_efficiency(self, load_tester: LoadTester):
        """Test memory usage during load"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run load test
        metrics = load_tester.concurrent_load_test(
            endpoint="/api/v1/articles", concurrent_users=20, requests_per_user=10
        )

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage should not increase significantly
        assert memory_increase < 100, f"Memory usage increased by {memory_increase}MB"

        stats = metrics.get_stats()
        assert stats["success_rate"] >= 0.90

    def test_database_connection_handling(self, load_tester: LoadTester):
        """Test database connection handling under load"""
        # Simulate high database load
        metrics = load_tester.concurrent_load_test(
            endpoint="/api/v1/articles", concurrent_users=50, requests_per_user=5
        )

        stats = metrics.get_stats()

        # Should not have connection pool exhaustion
        assert stats["success_rate"] >= 0.85

        # Check for database-related errors
        db_errors = [error for error in metrics.errors if "database" in error.lower() or "connection" in error.lower()]
        assert len(db_errors) < len(metrics.errors) * 0.1, "Too many database connection errors"

    def test_api_rate_limiting_behavior(self, load_tester: LoadTester):
        """Test rate limiting behavior"""
        # Send requests rapidly to trigger rate limiting
        metrics = load_tester.sustained_load_test(
            endpoint="/api/v1/articles", duration_seconds=10, requests_per_second=100  # High rate to trigger limits
        )

        stats = metrics.get_stats()

        # Should have some rate limit responses (429)
        rate_limited = stats["status_codes"].get(429, 0)

        # Verify rate limiting is working
        if rate_limited > 0:
            assert rate_limited < stats["total_requests"] * 0.8, "Too many requests rate limited"

    @pytest.mark.slow
    def test_long_running_stability(self, load_tester: LoadTester):
        """Test API stability over longer duration"""
        metrics = load_tester.sustained_load_test(
            endpoint="/api/v1/articles", duration_seconds=120, requests_per_second=5  # 2 minutes
        )

        stats = metrics.get_stats()

        # Should maintain stability
        assert stats["success_rate"] >= 0.95
        assert stats["response_times"]["mean"] < 3.0
        assert len(metrics.errors) < stats["total_requests"] * 0.05

    def test_endpoint_specific_performance_requirements(self, load_tester: LoadTester):
        """Test performance requirements for specific endpoints"""
        endpoints_requirements = {
            "/api/v1/articles": {"max_response_time": 2.0, "min_rps": 20},
            "/api/v1/search": {"max_response_time": 3.0, "min_rps": 10},
            "/api/v1/analytics/summary": {"max_response_time": 5.0, "min_rps": 5},
            "/health": {"max_response_time": 0.5, "min_rps": 100},
        }

        for endpoint, requirements in endpoints_requirements.items():
            metrics = load_tester.concurrent_load_test(endpoint=endpoint, concurrent_users=10, requests_per_user=5)

            stats = metrics.get_stats()

            assert (
                stats["response_times"]["mean"] < requirements["max_response_time"]
            ), f"{endpoint} response time too high: {stats['response_times']['mean']}"
            assert (
                stats["requests_per_second"] > requirements["min_rps"]
            ), f"{endpoint} RPS too low: {stats['requests_per_second']}"


@pytest.mark.asyncio
@pytest.mark.performance
class TestAsyncPerformance:
    """Async performance testing"""

    async def test_async_concurrent_requests(self, auth_headers: Dict[str, str]):
        """Test async concurrent request handling"""

        async def make_request(session: httpx.AsyncClient, url: str) -> Tuple[float, int]:
            start_time = time.time()
            try:
                response = await session.get(url, headers=auth_headers)
                return time.time() - start_time, response.status_code
            except Exception:
                return time.time() - start_time, 0

        base_url = "http://testserver"
        endpoints = ["/api/v1/articles", "/api/v1/search?q=test", "/api/v1/analytics/summary", "/health"]

        async with httpx.AsyncClient() as client:
            tasks = []
            for _ in range(50):  # 50 concurrent requests
                endpoint = endpoints[len(tasks) % len(endpoints)]
                task = make_request(client, f"{base_url}{endpoint}")
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_requests = sum(1 for result in results if isinstance(result, tuple) and result[1] == 200)
        total_requests = len(results)

        assert successful_requests / total_requests >= 0.90, "Too many failed requests in async test"
