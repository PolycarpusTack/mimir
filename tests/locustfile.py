"""
Locust Load Testing Configuration for Mimir Enterprise API

High-scale load testing scenarios using Locust framework.
"""

import json
import random
from typing import Any, Dict

from locust import HttpUser, between, events, task
from locust.contrib.fasthttp import FastHttpUser


class MimirAPIUser(FastHttpUser):
    """Base API user for load testing"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Setup for each user session"""
        # Authenticate (in real implementation)
        self.auth_headers = {
            "Authorization": "Bearer test-token",
            "X-Organization-ID": "test-org-id",
            "Content-Type": "application/json",
        }

        # Test data for creating articles/searches
        self.test_data = {
            "search_terms": [
                "artificial intelligence",
                "machine learning",
                "data science",
                "technology",
                "automation",
                "cloud computing",
                "cybersecurity",
                "blockchain",
            ],
            "categories": ["technology", "business", "science", "news"],
            "sources": ["TechCrunch", "MIT News", "BBC Tech", "Wired"],
        }

    @task(10)
    def read_articles(self):
        """Read articles endpoint (most common operation)"""
        params = {"page": random.randint(1, 10), "page_size": random.choice([10, 20, 50])}

        with self.client.get(
            "/api/v1/articles", params=params, headers=self.auth_headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(8)
    def search_articles(self):
        """Search articles (second most common)"""
        search_term = random.choice(self.test_data["search_terms"])
        params = {"q": search_term, "page": random.randint(1, 5), "page_size": 20}

        with self.client.get(
            "/api/v1/search", params=params, headers=self.auth_headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "items" in data and "total" in data:
                    response.success()
                else:
                    response.failure("Invalid search response")
            else:
                response.failure(f"Search failed: {response.status_code}")

    @task(3)
    def get_analytics_summary(self):
        """Get analytics summary"""
        params = {"time_range": random.choice(["1d", "7d", "30d"])}

        with self.client.get(
            "/api/v1/analytics/summary", params=params, headers=self.auth_headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analytics failed: {response.status_code}")

    @task(2)
    def create_article(self):
        """Create new article (less frequent)"""
        article_data = {
            "title": f"Load Test Article {random.randint(1, 10000)}",
            "content": "This is content generated during load testing.",
            "url": f"https://example.com/article-{random.randint(1, 10000)}",
            "source": random.choice(self.test_data["sources"]),
            "category": random.choice(self.test_data["categories"]),
            "tags": random.sample(["loadtest", "api", "automation", "test"], 2),
        }

        with self.client.post(
            "/api/v1/articles", json=article_data, headers=self.auth_headers, catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Article creation failed: {response.status_code}")

    @task(2)
    def get_saved_searches(self):
        """Get saved searches"""
        with self.client.get("/api/v1/saved-searches", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Saved searches failed: {response.status_code}")

    @task(1)
    def create_saved_search(self):
        """Create saved search (infrequent)"""
        search_data = {
            "name": f"Load Test Search {random.randint(1, 1000)}",
            "description": "Search created during load testing",
            "query": {
                "terms": [random.choice(self.test_data["search_terms"])],
                "categories": [random.choice(self.test_data["categories"])],
            },
            "alert_enabled": random.choice([True, False]),
        }

        with self.client.post(
            "/api/v1/saved-searches", json=search_data, headers=self.auth_headers, catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Saved search creation failed: {response.status_code}")

    @task(1)
    def get_dashboards(self):
        """Get dashboards"""
        with self.client.get("/api/v1/dashboards", headers=self.auth_headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Dashboards failed: {response.status_code}")

    @task(5)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Unhealthy status")
            else:
                response.failure(f"Health check failed: {response.status_code}")


class ReadOnlyUser(FastHttpUser):
    """User that only performs read operations"""

    wait_time = between(0.5, 2)
    weight = 3  # More read-only users

    def on_start(self):
        self.auth_headers = {"Authorization": "Bearer readonly-token", "X-Organization-ID": "readonly-org-id"}

    @task(15)
    def read_articles(self):
        """Read articles frequently"""
        params = {"page": random.randint(1, 20), "page_size": 20}
        self.client.get("/api/v1/articles", params=params, headers=self.auth_headers)

    @task(10)
    def search_articles(self):
        """Search articles frequently"""
        search_terms = ["AI", "tech", "news", "data", "cloud"]
        params = {"q": random.choice(search_terms), "page": 1, "page_size": 10}
        self.client.get("/api/v1/search", params=params, headers=self.auth_headers)

    @task(5)
    def get_analytics(self):
        """View analytics"""
        self.client.get("/api/v1/analytics/summary", headers=self.auth_headers)

    @task(3)
    def health_check(self):
        """Health checks"""
        self.client.get("/health")


class AdminUser(FastHttpUser):
    """Admin user performing management operations"""

    wait_time = between(2, 5)
    weight = 1  # Fewer admin users

    def on_start(self):
        self.auth_headers = {"Authorization": "Bearer admin-token", "X-Organization-ID": "admin-org-id"}

    @task(5)
    def manage_users(self):
        """User management operations"""
        self.client.get("/api/v1/users", headers=self.auth_headers)

    @task(3)
    def view_analytics(self):
        """View detailed analytics"""
        endpoints = ["/api/v1/analytics/summary", "/api/v1/analytics/sources", "/api/v1/analytics/trends"]
        endpoint = random.choice(endpoints)
        self.client.get(endpoint, headers=self.auth_headers)

    @task(2)
    def manage_subscriptions(self):
        """Subscription management"""
        self.client.get("/api/v1/subscriptions/current", headers=self.auth_headers)

    @task(2)
    def manage_webhooks(self):
        """Webhook management"""
        self.client.get("/api/v1/webhooks", headers=self.auth_headers)

    @task(1)
    def create_dashboard(self):
        """Create dashboard"""
        dashboard_data = {
            "name": f"Admin Dashboard {random.randint(1, 100)}",
            "description": "Dashboard created during load testing",
            "layout": {
                "widgets": [
                    {
                        "id": "widget1",
                        "type": "article_count",
                        "title": "Article Count",
                        "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                    }
                ]
            },
        }
        self.client.post("/api/v1/dashboards", json=dashboard_data, headers=self.auth_headers)


class StressTestUser(FastHttpUser):
    """User for stress testing scenarios"""

    wait_time = between(0.1, 0.5)  # Very aggressive timing

    def on_start(self):
        self.auth_headers = {"Authorization": "Bearer stress-token", "X-Organization-ID": "stress-org-id"}

    @task(20)
    def rapid_requests(self):
        """Rapid fire requests"""
        endpoints = ["/health", "/api/v1/articles", "/api/v1/search?q=test"]
        endpoint = random.choice(endpoints)

        if endpoint == "/api/v1/articles":
            self.client.get(endpoint, headers=self.auth_headers)
        elif endpoint.startswith("/api/v1/search"):
            self.client.get(endpoint, headers=self.auth_headers)
        else:
            self.client.get(endpoint)


# Event handlers for custom metrics
@events.request.add_listener
def request_handler(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    """Custom request handler for additional metrics"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response and hasattr(response, "status_code"):
        if response.status_code >= 400:
            print(f"HTTP error: {name} - {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Test start handler"""
    print("Starting Mimir API load test...")
    print(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test stop handler"""
    print("Mimir API load test completed.")

    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")


# Load test scenarios as classes
class LightLoadTest(MimirAPIUser):
    """Light load test scenario"""

    wait_time = between(2, 5)
    weight = 1


class NormalLoadTest(MimirAPIUser):
    """Normal load test scenario"""

    wait_time = between(1, 3)
    weight = 3


class HeavyLoadTest(MimirAPIUser):
    """Heavy load test scenario"""

    wait_time = between(0.5, 2)
    weight = 5


# Custom task sets for specific testing scenarios
from locust import TaskSet


class SearchTaskSet(TaskSet):
    """Task set focused on search operations"""

    @task(10)
    def basic_search(self):
        terms = ["AI", "machine learning", "technology", "news"]
        params = {"q": random.choice(terms)}
        self.client.get("/api/v1/search", params=params, headers=self.user.auth_headers)

    @task(5)
    def advanced_search(self):
        params = {"q": "artificial intelligence", "category": "technology", "date_from": "2024-01-01", "page_size": 50}
        self.client.get("/api/v1/search", params=params, headers=self.user.auth_headers)

    @task(2)
    def search_analytics(self):
        self.client.get("/api/v1/analytics/search", headers=self.user.auth_headers)


class AnalyticsTaskSet(TaskSet):
    """Task set focused on analytics operations"""

    @task(5)
    def summary_analytics(self):
        params = {"time_range": random.choice(["1d", "7d", "30d"])}
        self.client.get("/api/v1/analytics/summary", params=params, headers=self.user.auth_headers)

    @task(3)
    def source_analytics(self):
        self.client.get("/api/v1/analytics/sources", headers=self.user.auth_headers)

    @task(2)
    def trend_analytics(self):
        self.client.get("/api/v1/analytics/trends", headers=self.user.auth_headers)


# Example usage:
# locust -f locustfile.py --host=http://localhost:8000 -u 50 -r 5 -t 300s MimirAPIUser
# locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 600s ReadOnlyUser AdminUser
# locust -f locustfile.py --host=http://localhost:8000 -u 500 -r 50 -t 1800s StressTestUser
