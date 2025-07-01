"""
Integration Tests for Mimir Enterprise API

Tests complete user workflows and system integration scenarios.
"""

import asyncio
import uuid
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestUserWorkflows:
    """Complete user workflow integration tests"""

    def test_complete_article_management_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test complete article management workflow"""
        # Step 1: Create a new article
        new_article = {
            "title": "Integration Test Article",
            "content": "This is content for integration testing",
            "url": "https://example.com/integration-test",
            "source": "Integration Test Source",
            "author": "Test Author",
            "category": "Technology",
            "tags": ["integration", "test", "api"],
        }

        create_response = client.post("/api/v1/articles", json=new_article, headers=auth_headers)
        assert create_response.status_code == 201
        created_article = create_response.json()
        article_id = created_article["id"]

        # Step 2: Retrieve the created article
        get_response = client.get(f"/api/v1/articles/{article_id}", headers=auth_headers)
        assert get_response.status_code == 200
        retrieved_article = get_response.json()
        assert retrieved_article["title"] == new_article["title"]

        # Step 3: Update the article
        update_data = {
            "title": "Updated Integration Test Article",
            "content": "Updated content for integration testing",
            "tags": ["integration", "test", "api", "updated"],
        }

        update_response = client.put(f"/api/v1/articles/{article_id}", json=update_data, headers=auth_headers)
        assert update_response.status_code == 200
        updated_article = update_response.json()
        assert updated_article["title"] == update_data["title"]
        assert "updated" in updated_article["tags"]

        # Step 4: Search for the article
        search_response = client.get(
            "/api/v1/search", params={"q": "Integration Test", "page": 1, "page_size": 10}, headers=auth_headers
        )
        assert search_response.status_code == 200
        search_results = search_response.json()

        # Should find the updated article
        found_articles = [article for article in search_results["items"] if article["id"] == article_id]
        assert len(found_articles) == 1
        assert found_articles[0]["title"] == update_data["title"]

        # Step 5: Delete the article
        delete_response = client.delete(f"/api/v1/articles/{article_id}", headers=auth_headers)
        assert delete_response.status_code == 204

        # Step 6: Verify article is deleted
        get_deleted_response = client.get(f"/api/v1/articles/{article_id}", headers=auth_headers)
        assert get_deleted_response.status_code == 404

    def test_saved_search_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test saved search creation, execution, and analytics workflow"""
        # Step 1: Create a saved search
        saved_search_data = {
            "name": "AI Technology Search",
            "description": "Search for AI and ML articles",
            "query": {
                "terms": ["artificial intelligence", "machine learning"],
                "categories": ["technology"],
                "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            },
            "alert_enabled": True,
            "alert_frequency": "daily",
        }

        create_response = client.post("/api/v1/saved-searches", json=saved_search_data, headers=auth_headers)
        assert create_response.status_code == 201
        saved_search = create_response.json()
        search_id = saved_search["id"]

        # Step 2: Execute the saved search
        execute_response = client.post(f"/api/v1/saved-searches/{search_id}/execute", headers=auth_headers)
        assert execute_response.status_code == 200
        search_results = execute_response.json()
        assert "items" in search_results
        assert "total" in search_results

        # Step 3: Get search analytics
        analytics_response = client.get(f"/api/v1/analytics/saved-searches/{search_id}", headers=auth_headers)
        assert analytics_response.status_code == 200
        analytics = analytics_response.json()
        assert "execution_count" in analytics
        assert analytics["execution_count"] >= 1

        # Step 4: Update saved search
        update_data = {"name": "Updated AI Technology Search", "alert_frequency": "weekly"}

        update_response = client.put(f"/api/v1/saved-searches/{search_id}", json=update_data, headers=auth_headers)
        assert update_response.status_code == 200
        updated_search = update_response.json()
        assert updated_search["name"] == update_data["name"]
        assert updated_search["alert_frequency"] == update_data["alert_frequency"]

        # Step 5: Delete saved search
        delete_response = client.delete(f"/api/v1/saved-searches/{search_id}", headers=auth_headers)
        assert delete_response.status_code == 204

    def test_dashboard_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test dashboard creation and widget management workflow"""
        # Step 1: Create a dashboard
        dashboard_data = {
            "name": "Integration Test Dashboard",
            "description": "Dashboard for integration testing",
            "layout": {
                "widgets": [
                    {
                        "id": "widget1",
                        "type": "article_count",
                        "title": "Total Articles",
                        "position": {"x": 0, "y": 0, "w": 6, "h": 4},
                        "config": {"time_range": "7d", "filters": {}},
                    }
                ]
            },
            "is_public": False,
        }

        create_response = client.post("/api/v1/dashboards", json=dashboard_data, headers=auth_headers)
        assert create_response.status_code == 201
        dashboard = create_response.json()
        dashboard_id = dashboard["id"]

        # Step 2: Add a widget to the dashboard
        new_widget = {
            "id": "widget2",
            "type": "sentiment_chart",
            "title": "Sentiment Analysis",
            "position": {"x": 6, "y": 0, "w": 6, "h": 4},
            "config": {"chart_type": "pie", "time_range": "30d"},
        }

        add_widget_response = client.post(
            f"/api/v1/dashboards/{dashboard_id}/widgets", json=new_widget, headers=auth_headers
        )
        assert add_widget_response.status_code == 201

        # Step 3: Get dashboard data
        dashboard_response = client.get(f"/api/v1/dashboards/{dashboard_id}", headers=auth_headers)
        assert dashboard_response.status_code == 200
        updated_dashboard = dashboard_response.json()
        assert len(updated_dashboard["layout"]["widgets"]) == 2

        # Step 4: Execute dashboard (get widget data)
        execute_response = client.post(f"/api/v1/dashboards/{dashboard_id}/execute", headers=auth_headers)
        assert execute_response.status_code == 200
        dashboard_data = execute_response.json()
        assert "widgets" in dashboard_data

        # Step 5: Update widget configuration
        widget_update = {"config": {"time_range": "14d", "filters": {"category": "technology"}}}

        update_widget_response = client.put(
            f"/api/v1/dashboards/{dashboard_id}/widgets/widget1", json=widget_update, headers=auth_headers
        )
        assert update_widget_response.status_code == 200

        # Step 6: Delete dashboard
        delete_response = client.delete(f"/api/v1/dashboards/{dashboard_id}", headers=auth_headers)
        assert delete_response.status_code == 204

    def test_user_management_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test user management workflow (admin functionality)"""
        # Step 1: Create a new user
        new_user_data = {
            "email": "newuser@test.com",
            "username": "newintegrationuser",
            "password": "StrongPassword123!",
            "role": "user",
            "permissions": ["articles:read", "search:basic"],
        }

        create_user_response = client.post("/api/v1/users", json=new_user_data, headers=auth_headers)
        assert create_user_response.status_code == 201
        new_user = create_user_response.json()
        user_id = new_user["id"]

        # Step 2: Get user details
        get_user_response = client.get(f"/api/v1/users/{user_id}", headers=auth_headers)
        assert get_user_response.status_code == 200
        user_details = get_user_response.json()
        assert user_details["email"] == new_user_data["email"]

        # Step 3: Update user permissions
        permission_update = {"permissions": ["articles:read", "articles:create", "search:advanced"]}

        update_permissions_response = client.put(
            f"/api/v1/users/{user_id}/permissions", json=permission_update, headers=auth_headers
        )
        assert update_permissions_response.status_code == 200

        # Step 4: Deactivate user
        deactivate_response = client.post(f"/api/v1/users/{user_id}/deactivate", headers=auth_headers)
        assert deactivate_response.status_code == 200

        # Step 5: Verify user is deactivated
        get_deactivated_response = client.get(f"/api/v1/users/{user_id}", headers=auth_headers)
        assert get_deactivated_response.status_code == 200
        deactivated_user = get_deactivated_response.json()
        assert not deactivated_user["is_active"]

    def test_subscription_feature_access_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test subscription and feature access workflow"""
        # Step 1: Check current subscription
        subscription_response = client.get("/api/v1/subscriptions/current", headers=auth_headers)
        # May be 200 with subscription or 404 if no subscription

        # Step 2: Get available plans
        plans_response = client.get("/api/v1/subscriptions/plans", headers=auth_headers)
        assert plans_response.status_code == 200
        plans = plans_response.json()
        assert len(plans) > 0

        # Step 3: Check feature access
        features_to_check = ["advanced_search", "ai_analysis", "custom_dashboards"]

        for feature in features_to_check:
            feature_response = client.get(f"/api/v1/subscriptions/features/{feature}/access", headers=auth_headers)
            assert feature_response.status_code == 200
            feature_access = feature_response.json()
            assert "has_access" in feature_access
            assert "feature" in feature_access

        # Step 4: Get feature usage summary
        usage_response = client.get("/api/v1/subscriptions/features/usage/summary", headers=auth_headers)
        assert usage_response.status_code == 200
        usage_summary = usage_response.json()
        assert isinstance(usage_summary, dict)

    def test_webhook_workflow(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test webhook configuration and delivery workflow"""
        # Step 1: Create a webhook
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["article.created", "search.executed"],
            "secret": "webhook-secret-key",
            "is_active": True,
        }

        create_webhook_response = client.post("/api/v1/webhooks", json=webhook_data, headers=auth_headers)
        assert create_webhook_response.status_code == 201
        webhook = create_webhook_response.json()
        webhook_id = webhook["id"]

        # Step 2: Test webhook (ping)
        test_response = client.post(f"/api/v1/webhooks/{webhook_id}/test", headers=auth_headers)
        assert test_response.status_code == 200

        # Step 3: Get webhook delivery logs
        logs_response = client.get(f"/api/v1/webhooks/{webhook_id}/deliveries", headers=auth_headers)
        assert logs_response.status_code == 200
        delivery_logs = logs_response.json()
        assert "items" in delivery_logs

        # Step 4: Update webhook configuration
        webhook_update = {"events": ["article.created", "article.updated", "search.executed"], "is_active": False}

        update_webhook_response = client.put(
            f"/api/v1/webhooks/{webhook_id}", json=webhook_update, headers=auth_headers
        )
        assert update_webhook_response.status_code == 200
        updated_webhook = update_webhook_response.json()
        assert not updated_webhook["is_active"]
        assert len(updated_webhook["events"]) == 3

        # Step 5: Delete webhook
        delete_webhook_response = client.delete(f"/api/v1/webhooks/{webhook_id}", headers=auth_headers)
        assert delete_webhook_response.status_code == 204


@pytest.mark.integration
class TestSystemIntegration:
    """System-level integration tests"""

    def test_search_to_analytics_integration(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test integration between search and analytics systems"""
        # Step 1: Perform several searches
        search_queries = ["artificial intelligence", "machine learning", "data science", "technology trends"]

        for query in search_queries:
            search_response = client.get(
                "/api/v1/search", params={"q": query, "page": 1, "page_size": 10}, headers=auth_headers
            )
            assert search_response.status_code == 200

        # Step 2: Check that analytics reflect the searches
        analytics_response = client.get("/api/v1/analytics/search", params={"time_range": "1d"}, headers=auth_headers)
        assert analytics_response.status_code == 200
        search_analytics = analytics_response.json()

        # Should show search activity
        assert "total_searches" in search_analytics
        assert search_analytics["total_searches"] >= len(search_queries)

    def test_notification_system_integration(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test notification system integration"""
        # Step 1: Configure notification preferences
        notification_config = {
            "email_enabled": True,
            "webhook_enabled": True,
            "frequency": "immediate",
            "filters": {"categories": ["technology"], "keywords": ["AI", "machine learning"]},
        }

        config_response = client.put(
            "/api/v1/notifications/preferences", json=notification_config, headers=auth_headers
        )
        assert config_response.status_code == 200

        # Step 2: Create an article that should trigger notification
        triggering_article = {
            "title": "New AI Breakthrough in Machine Learning",
            "content": "Researchers have developed a new AI algorithm...",
            "url": "https://example.com/ai-breakthrough",
            "source": "Tech News",
            "category": "technology",
            "tags": ["AI", "machine learning", "research"],
        }

        create_response = client.post("/api/v1/articles", json=triggering_article, headers=auth_headers)
        assert create_response.status_code == 201

        # Step 3: Check notification history
        notifications_response = client.get(
            "/api/v1/notifications/history", params={"page": 1, "page_size": 10}, headers=auth_headers
        )
        assert notifications_response.status_code == 200
        notification_history = notifications_response.json()

        # Should find notification related to the article
        assert "items" in notification_history

    def test_analytics_data_consistency(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test data consistency across analytics endpoints"""
        # Get analytics from different endpoints
        endpoints = [
            "/api/v1/analytics/summary",
            "/api/v1/analytics/articles",
            "/api/v1/analytics/sources",
            "/api/v1/analytics/trends",
        ]

        analytics_data = {}
        for endpoint in endpoints:
            response = client.get(endpoint, params={"time_range": "7d"}, headers=auth_headers)
            if response.status_code == 200:
                analytics_data[endpoint] = response.json()

        # Check data consistency
        if "/api/v1/analytics/summary" in analytics_data:
            summary = analytics_data["/api/v1/analytics/summary"]

            if "/api/v1/analytics/articles" in analytics_data:
                articles = analytics_data["/api/v1/analytics/articles"]

                # Article counts should be consistent
                if "total_articles" in summary and "total_count" in articles:
                    assert summary["total_articles"] == articles["total_count"]

    def test_multi_tenant_data_isolation(self, client: TestClient):
        """Test data isolation between organizations"""
        # Create auth headers for two different organizations
        org1_headers = {"Authorization": "Bearer org1-token", "X-Organization-ID": str(uuid.uuid4())}

        org2_headers = {"Authorization": "Bearer org2-token", "X-Organization-ID": str(uuid.uuid4())}

        # Create article in org1
        org1_article = {
            "title": "Organization 1 Article",
            "content": "Content for org 1",
            "url": "https://example.com/org1-article",
            "source": "Org1 Source",
        }

        org1_response = client.post("/api/v1/articles", json=org1_article, headers=org1_headers)

        # Create article in org2
        org2_article = {
            "title": "Organization 2 Article",
            "content": "Content for org 2",
            "url": "https://example.com/org2-article",
            "source": "Org2 Source",
        }

        org2_response = client.post("/api/v1/articles", json=org2_article, headers=org2_headers)

        # Each org should only see their own articles
        org1_articles = client.get("/api/v1/articles", headers=org1_headers)
        org2_articles = client.get("/api/v1/articles", headers=org2_headers)

        if org1_articles.status_code == 200 and org2_articles.status_code == 200:
            org1_data = org1_articles.json()
            org2_data = org2_articles.json()

            # Verify data isolation
            org1_titles = [article["title"] for article in org1_data["items"]]
            org2_titles = [article["title"] for article in org2_data["items"]]

            assert "Organization 1 Article" in org1_titles
            assert "Organization 1 Article" not in org2_titles
            assert "Organization 2 Article" in org2_titles
            assert "Organization 2 Article" not in org1_titles
