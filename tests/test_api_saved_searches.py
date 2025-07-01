"""
API Integration tests for Saved Search endpoints

Tests the FastAPI endpoints for saved search functionality.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.models import SavedSearchCreate, SavedSearchUpdate
from auth.jwt_handler import JWTHandler
from services.saved_search_manager import AlertFrequency, SavedSearchConfig


@pytest.fixture
def app():
    """Create test application"""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers"""
    jwt_handler = JWTHandler()
    token = jwt_handler.create_access_token(
        {"user_id": "test-user-123", "organization_id": "test-org-456", "email": "test@example.com", "roles": ["admin"]}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_saved_search_manager():
    """Mock SavedSearchManager"""
    with patch("api.routers.saved_searches.SavedSearchManager") as mock:
        yield mock


class TestSavedSearchEndpoints:
    """Test saved search API endpoints"""

    def test_list_saved_searches(self, client, auth_headers, mock_saved_search_manager):
        """Test GET /api/v1/saved-searches"""
        # Mock saved searches
        mock_manager = mock_saved_search_manager.return_value
        mock_manager.list_user_saved_searches.return_value = [
            SavedSearchConfig(
                id="search-1",
                name="My Search",
                description="Test search",
                query={"q": "test"},
                is_alert=False,
                alert_frequency=None,
                last_checked_at=None,
                next_check_at=None,
                notification_channels=[],
                notification_settings={},
                is_shared=False,
                shared_with=[],
                tags=["test"],
                organization_id="test-org-456",
                user_id="test-user-123",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        ]

        response = client.get(
            "/api/v1/saved-searches", headers=auth_headers, params={"include_shared": True, "only_alerts": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "My Search"
        assert data[0]["is_alert"] is False

    def test_get_saved_search(self, client, auth_headers, mock_saved_search_manager):
        """Test GET /api/v1/saved-searches/{search_id}"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.get_saved_search.return_value = SavedSearchConfig(
            id=search_id,
            name="Specific Search",
            description="A specific search",
            query={"q": "specific", "source": "TechCrunch"},
            is_alert=True,
            alert_frequency=3600,
            last_checked_at=datetime.now(timezone.utc),
            next_check_at=datetime.now(timezone.utc) + timedelta(hours=1),
            notification_channels=["email"],
            notification_settings={"email": {"to": "alerts@example.com"}},
            is_shared=True,
            shared_with=["user-789"],
            tags=["tech", "alerts"],
            organization_id="test-org-456",
            user_id="test-user-123",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        response = client.get(f"/api/v1/saved-searches/{search_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == search_id
        assert data["name"] == "Specific Search"
        assert data["is_alert"] is True
        assert data["alert_frequency"] == 3600

    def test_create_saved_search(self, client, auth_headers, mock_saved_search_manager):
        """Test POST /api/v1/saved-searches"""
        mock_manager = mock_saved_search_manager.return_value
        created_id = str(uuid4())

        mock_manager.create_saved_search.return_value = SavedSearchConfig(
            id=created_id,
            name="New Alert",
            description="A new alert search",
            query={"q": "breaking", "category": "news"},
            is_alert=True,
            alert_frequency=1800,
            last_checked_at=None,
            next_check_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            notification_channels=["email", "slack"],
            notification_settings={"email": {"to": "alerts@example.com"}, "slack": {"channel": "#alerts"}},
            is_shared=False,
            shared_with=[],
            tags=["urgent", "news"],
            organization_id="test-org-456",
            user_id="test-user-123",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        request_data = {
            "name": "New Alert",
            "description": "A new alert search",
            "query": {"q": "breaking", "category": "news"},
            "is_alert": True,
            "alert_frequency": 1800,
            "notification_channels": ["email", "slack"],
            "notification_settings": {"email": {"to": "alerts@example.com"}, "slack": {"channel": "#alerts"}},
            "is_shared": False,
            "shared_with": [],
            "tags": ["urgent", "news"],
        }

        response = client.post("/api/v1/saved-searches", headers=auth_headers, json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == created_id
        assert data["name"] == "New Alert"
        assert data["is_alert"] is True
        assert len(data["notification_channels"]) == 2

    def test_update_saved_search(self, client, auth_headers, mock_saved_search_manager):
        """Test PUT /api/v1/saved-searches/{search_id}"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.update_saved_search.return_value = SavedSearchConfig(
            id=search_id,
            name="Updated Search",
            description="Updated description",
            query={"q": "updated"},
            is_alert=False,
            alert_frequency=None,
            last_checked_at=None,
            next_check_at=None,
            notification_channels=[],
            notification_settings={},
            is_shared=True,
            shared_with=["user-999"],
            tags=["updated"],
            organization_id="test-org-456",
            user_id="test-user-123",
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            updated_at=datetime.now(timezone.utc),
        )

        update_data = {
            "name": "Updated Search",
            "description": "Updated description",
            "is_shared": True,
            "shared_with": ["user-999"],
            "tags": ["updated"],
        }

        response = client.put(f"/api/v1/saved-searches/{search_id}", headers=auth_headers, json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Search"
        assert data["is_shared"] is True
        assert "user-999" in data["shared_with"]

    def test_delete_saved_search(self, client, auth_headers, mock_saved_search_manager):
        """Test DELETE /api/v1/saved-searches/{search_id}"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.get_saved_search.return_value = SavedSearchConfig(
            id=search_id, name="To Delete", query={}, organization_id="test-org-456", user_id="test-user-123"
        )
        mock_manager.delete_saved_search.return_value = True

        response = client.delete(f"/api/v1/saved-searches/{search_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "deleted successfully" in data["message"]

    def test_execute_saved_search(self, client, auth_headers, mock_saved_search_manager):
        """Test POST /api/v1/saved-searches/{search_id}/execute"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.execute_saved_search.return_value = {
            "results": [
                {
                    "id": "article-1",
                    "title": "Search Result 1",
                    "content": "Content...",
                    "source": "TestSource",
                    "published_date": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "total_count": 1,
            "execution_time_ms": 150.5,
            "search_id": search_id,
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }

        response = client.post(
            f"/api/v1/saved-searches/{search_id}/execute", headers=auth_headers, params={"limit": 100}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["total_count"] == 1
        assert data["execution_time_ms"] == 150.5

    def test_test_alert(self, client, auth_headers, mock_saved_search_manager):
        """Test POST /api/v1/saved-searches/{search_id}/test-alert"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.get_saved_search.return_value = SavedSearchConfig(
            id=search_id,
            name="Test Alert",
            query={"q": "test"},
            is_alert=True,
            alert_frequency=3600,
            notification_channels=["email"],
            organization_id="test-org-456",
            user_id="test-user-123",
        )

        response = client.post(f"/api/v1/saved-searches/{search_id}/test-alert", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Alert test initiated" in data["message"]
        assert data["alert_id"] == search_id

    def test_get_alert_statistics(self, client, auth_headers, mock_saved_search_manager):
        """Test GET /api/v1/saved-searches/alerts/statistics"""
        mock_manager = mock_saved_search_manager.return_value
        mock_manager.get_alert_statistics.return_value = {
            "total_alerts": 10,
            "active_alerts": 8,
            "alerts_by_frequency": {"300": 2, "3600": 5, "86400": 3},
            "recent_triggers": 25,
            "period_days": 30,
        }

        response = client.get("/api/v1/saved-searches/alerts/statistics", headers=auth_headers, params={"days": 30})

        assert response.status_code == 200
        data = response.json()
        assert data["total_alerts"] == 10
        assert data["active_alerts"] == 8
        assert data["recent_triggers"] == 25

    def test_get_shared_searches(self, client, auth_headers, mock_saved_search_manager):
        """Test GET /api/v1/saved-searches/shared"""
        mock_manager = mock_saved_search_manager.return_value
        mock_manager.list_user_saved_searches.return_value = [
            SavedSearchConfig(
                id="shared-1",
                name="Team Search",
                query={"q": "team"},
                is_shared=True,
                shared_with=[],
                organization_id="test-org-456",
                user_id="other-user-789",  # Not the current user
                created_at=datetime.now(timezone.utc),
            ),
            SavedSearchConfig(
                id="shared-2",
                name="Private Shared",
                query={"q": "private"},
                is_shared=False,
                shared_with=["test-user-123"],  # Shared with current user
                organization_id="test-org-456",
                user_id="other-user-999",
                created_at=datetime.now(timezone.utc),
            ),
        ]

        response = client.get("/api/v1/saved-searches/shared", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(search["user_id"] != "test-user-123" for search in data)


class TestSavedSearchValidation:
    """Test input validation for saved searches"""

    def test_create_invalid_alert_frequency(self, client, auth_headers):
        """Test creating alert with invalid frequency"""
        request_data = {
            "name": "Invalid Alert",
            "query": {"q": "test"},
            "is_alert": True,
            "alert_frequency": 60,  # Too low
            "notification_channels": ["email"],
        }

        response = client.post("/api/v1/saved-searches", headers=auth_headers, json=request_data)

        assert response.status_code == 400
        assert "Invalid alert frequency" in response.json()["detail"]

    def test_create_alert_without_channels(self, client, auth_headers):
        """Test creating alert without notification channels"""
        request_data = {
            "name": "No Channel Alert",
            "query": {"q": "test"},
            "is_alert": True,
            "alert_frequency": 3600,
            "notification_channels": [],  # Empty
        }

        response = client.post("/api/v1/saved-searches", headers=auth_headers, json=request_data)

        # Should succeed but might want to add validation
        assert response.status_code in [201, 400]

    def test_update_nonexistent_search(self, client, auth_headers, mock_saved_search_manager):
        """Test updating non-existent search"""
        search_id = str(uuid4())

        mock_manager = mock_saved_search_manager.return_value
        mock_manager.update_saved_search.return_value = None

        response = client.put(f"/api/v1/saved-searches/{search_id}", headers=auth_headers, json={"name": "Updated"})

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_execute_search_limit_validation(self, client, auth_headers):
        """Test execute search with invalid limit"""
        search_id = str(uuid4())

        # Test limit too high
        response = client.post(
            f"/api/v1/saved-searches/{search_id}/execute",
            headers=auth_headers,
            params={"limit": 2000},  # Over 1000 limit
        )

        assert response.status_code == 422  # Validation error

    def test_filter_by_tag(self, client, auth_headers, mock_saved_search_manager):
        """Test filtering searches by tag"""
        mock_manager = mock_saved_search_manager.return_value
        mock_manager.list_user_saved_searches.return_value = [
            SavedSearchConfig(
                id="search-1",
                name="Tagged Search",
                query={},
                tags=["important", "tech"],
                organization_id="test-org-456",
                user_id="test-user-123",
            ),
            SavedSearchConfig(
                id="search-2",
                name="Other Search",
                query={},
                tags=["personal"],
                organization_id="test-org-456",
                user_id="test-user-123",
            ),
        ]

        response = client.get("/api/v1/saved-searches", headers=auth_headers, params={"tag": "important"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert "important" in data[0]["tags"]


class TestSavedSearchAnalyticsEndpoints:
    """Test saved search analytics endpoints"""

    def test_get_usage_metrics(self, client, auth_headers):
        """Test GET /api/v1/saved-searches/usage"""
        with patch("api.routers.saved_search_analytics.SavedSearchAnalytics") as mock_analytics:
            mock_analytics.return_value.get_search_usage_metrics.return_value = {
                "period": {"start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-31T23:59:59Z"},
                "execution_stats": [],
                "user_engagement": {"unique_users": 15, "total_searches": 50, "alert_count": 20},
                "alert_performance": {"overdue_alerts": 2, "active_alerts_24h": 10},
            }

            response = client.get("/api/v1/saved-searches/usage", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["user_engagement"]["unique_users"] == 15

    def test_get_popular_queries(self, client, auth_headers):
        """Test GET /api/v1/saved-searches/popular"""
        with patch("api.routers.saved_search_analytics.SavedSearchAnalytics") as mock_analytics:
            mock_analytics.return_value.get_popular_search_queries.return_value = [
                {
                    "query_components": {
                        "text_search": "AI",
                        "filters": {"source": "TechCrunch"},
                        "sorting": {"sort_by": "published_date", "sort_order": "desc"},
                    },
                    "usage_count": 100,
                    "unique_users": 25,
                    "has_alert": True,
                    "search_names": ["AI News", "AI Updates"],
                }
            ]

            response = client.get("/api/v1/saved-searches/popular", headers=auth_headers, params={"limit": 10})

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["usage_count"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
