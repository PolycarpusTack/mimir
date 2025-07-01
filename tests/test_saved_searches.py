"""
Integration tests for Saved Searches functionality

Tests the complete saved search system including creation, execution, alerts, and analytics.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from db_manager_postgres import PostgreSQLManager
from services.alert_scheduler import AlertJob, AlertScheduler
from services.saved_search_analytics import SavedSearchAnalytics
from services.saved_search_manager import AlertFrequency, SavedSearchConfig, SavedSearchManager


@pytest.fixture
def mock_db():
    """Create mock database manager"""
    db = Mock(spec=PostgreSQLManager)
    db.fetch_one = Mock()
    db.fetch_all = Mock()
    db.execute_query = Mock()
    return db


@pytest.fixture
def saved_search_manager(mock_db):
    """Create SavedSearchManager instance"""
    return SavedSearchManager(mock_db)


@pytest.fixture
def alert_scheduler(saved_search_manager):
    """Create AlertScheduler instance"""
    notification_service = Mock()
    return AlertScheduler(saved_search_manager, notification_service)


@pytest.fixture
def analytics_service(mock_db):
    """Create SavedSearchAnalytics instance"""
    return SavedSearchAnalytics(mock_db)


class TestSavedSearchManager:
    """Test SavedSearchManager functionality"""

    def test_create_saved_search(self, saved_search_manager, mock_db):
        """Test creating a new saved search"""
        # Mock database response
        mock_db.fetch_one.return_value = {
            "id": "test-id-123",
            "name": "Tech News Alert",
            "description": "Alert for tech news",
            "query": json.dumps({"q": "technology", "source": "TechCrunch"}),
            "is_alert": True,
            "alert_frequency": 3600,
            "last_checked_at": None,
            "next_check_at": datetime.now(timezone.utc) + timedelta(hours=1),
            "notification_channels": ["email"],
            "notification_settings": {"email": {"to": "user@example.com"}},
            "is_shared": False,
            "shared_with": [],
            "tags": ["tech", "news"],
            "organization_id": "org-123",
            "user_id": "user-456",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        # Create saved search
        result = saved_search_manager.create_saved_search(
            organization_id="org-123",
            user_id="user-456",
            name="Tech News Alert",
            query={"q": "technology", "source": "TechCrunch"},
            description="Alert for tech news",
            is_alert=True,
            alert_frequency=3600,
            notification_channels=["email"],
            notification_settings={"email": {"to": "user@example.com"}},
            is_shared=False,
            shared_with=[],
            tags=["tech", "news"],
        )

        assert isinstance(result, SavedSearchConfig)
        assert result.name == "Tech News Alert"
        assert result.is_alert is True
        assert result.alert_frequency == 3600
        assert "email" in result.notification_channels

        # Verify database call
        mock_db.execute_query.assert_called_once()

    def test_execute_saved_search(self, saved_search_manager, mock_db):
        """Test executing a saved search"""
        # Mock saved search
        mock_db.fetch_one.side_effect = [
            # First call - get saved search
            {
                "id": "search-123",
                "query": json.dumps({"q": "AI", "limit": 10}),
                "user_id": "user-456",
                "organization_id": "org-123",
            },
            # Second call - count results
            {"count": 5},
        ]

        # Mock search results
        mock_db.fetch_all.return_value = [
            {
                "id": "article-1",
                "title": "AI Breakthrough",
                "content": "New AI model...",
                "source": "TechNews",
                "published_date": datetime.now(timezone.utc),
                "scraped_at": datetime.now(timezone.utc),
            }
        ]

        # Execute search
        results = saved_search_manager.execute_saved_search("search-123", "user-456", limit=10)

        assert "results" in results
        assert "total_count" in results
        assert "execution_time_ms" in results
        assert results["total_count"] == 5
        assert len(results["results"]) == 1

    def test_list_user_saved_searches(self, saved_search_manager, mock_db):
        """Test listing user's saved searches"""
        # Mock database response
        mock_db.fetch_all.return_value = [
            {
                "id": "search-1",
                "name": "My Search 1",
                "is_alert": False,
                "is_shared": False,
                "tags": ["personal"],
                "user_id": "user-456",
            },
            {
                "id": "search-2",
                "name": "Shared Search",
                "is_alert": True,
                "is_shared": True,
                "tags": ["team"],
                "user_id": "user-789",
            },
        ]

        # List searches
        searches = saved_search_manager.list_user_saved_searches(
            organization_id="org-123", user_id="user-456", include_shared=True, only_alerts=False
        )

        assert len(searches) == 2
        assert searches[0].name == "My Search 1"
        assert searches[1].is_shared is True

    def test_check_alerts(self, saved_search_manager, mock_db):
        """Test checking and processing alerts"""
        # Mock due alerts
        mock_db.fetch_all.side_effect = [
            # First call - get due alerts
            [
                {
                    "id": "alert-1",
                    "query": json.dumps({"q": "urgent"}),
                    "last_checked_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    "notification_channels": ["email"],
                    "notification_settings": {"email": {"to": "alerts@example.com"}},
                }
            ],
            # Second call - search results
            [{"id": "article-new", "title": "Urgent News", "scraped_at": datetime.now(timezone.utc)}],
        ]

        mock_db.fetch_one.return_value = {"count": 1}

        # Mock notification service
        with patch.object(saved_search_manager, "notification_service") as mock_notify:
            processed = saved_search_manager.check_alerts()

            assert len(processed) == 1
            assert processed[0]["alert_id"] == "alert-1"
            assert processed[0]["new_results_count"] == 1

            # Verify notification was attempted
            if mock_notify:
                mock_notify.send_notification.assert_called()


class TestAlertScheduler:
    """Test AlertScheduler functionality"""

    @pytest.mark.asyncio
    async def test_execute_alert(self, alert_scheduler, saved_search_manager, mock_db):
        """Test executing a single alert"""
        # Create test alert
        alert = AlertJob(
            alert_id="alert-123",
            organization_id="org-123",
            user_id="user-456",
            name="Test Alert",
            query={"q": "test"},
            notification_channels=["email"],
            notification_settings={"email": {"to": "test@example.com"}},
            next_check_at=datetime.now(timezone.utc),
            frequency=3600,
        )

        # Mock saved search execution
        saved_search_manager.execute_saved_search = Mock(
            return_value={
                "results": [{"id": "1", "title": "New Article", "scraped_at": datetime.now(timezone.utc)}],
                "total_count": 1,
            }
        )

        # Mock last checked timestamp
        mock_db.fetch_one.return_value = {"last_checked_at": datetime.now(timezone.utc) - timedelta(hours=1)}

        # Execute alert
        await alert_scheduler._execute_alert(alert)

        # Verify execution
        saved_search_manager.execute_saved_search.assert_called_with("alert-123", "user-456", 100)

        # Verify schedule update
        mock_db.execute_query.assert_called()

    @pytest.mark.asyncio
    async def test_filter_new_results(self, alert_scheduler, mock_db):
        """Test filtering new results"""
        alert = AlertJob(
            alert_id="alert-123",
            organization_id="org-123",
            user_id="user-456",
            name="Test Alert",
            query={},
            notification_channels=[],
            notification_settings={},
            next_check_at=datetime.now(timezone.utc),
            frequency=3600,
        )

        last_checked = datetime.now(timezone.utc) - timedelta(hours=1)
        mock_db.fetch_one.return_value = {"last_checked_at": last_checked}

        results = [
            {"id": "1", "scraped_at": last_checked - timedelta(hours=2)},  # Old
            {"id": "2", "scraped_at": last_checked + timedelta(minutes=30)},  # New
            {"id": "3", "scraped_at": last_checked + timedelta(minutes=45)},  # New
        ]

        new_results = await alert_scheduler._filter_new_results(alert, results)

        assert len(new_results) == 2
        assert new_results[0]["id"] == "2"
        assert new_results[1]["id"] == "3"

    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self, alert_scheduler):
        """Test starting and stopping the scheduler"""
        # Start scheduler
        await alert_scheduler.start()
        assert alert_scheduler._running is True
        assert alert_scheduler._task is not None

        # Stop scheduler
        await alert_scheduler.stop()
        assert alert_scheduler._running is False


class TestSavedSearchAnalytics:
    """Test SavedSearchAnalytics functionality"""

    def test_get_search_usage_metrics(self, analytics_service, mock_db):
        """Test getting usage metrics"""
        # Mock database responses
        mock_db.fetch_all.return_value = [
            {
                "id": "search-1",
                "name": "Daily Report",
                "is_alert": True,
                "days_executed": 25,
                "total_executions": 100,
                "first_execution": datetime.now(timezone.utc) - timedelta(days=30),
                "last_execution": datetime.now(timezone.utc),
            }
        ]

        mock_db.fetch_one.side_effect = [
            # User stats
            {"unique_users": 5, "total_searches": 20, "alert_count": 8, "shared_count": 3, "avg_tags_per_search": 2.5},
            # Alert stats
            {"overdue_alerts": 1, "active_alerts_24h": 6, "avg_check_interval_seconds": 3600},
        ]

        # Get metrics
        metrics = analytics_service.get_search_usage_metrics(
            organization_id="org-123", start_date=datetime.now(timezone.utc) - timedelta(days=30)
        )

        assert "execution_stats" in metrics
        assert "user_engagement" in metrics
        assert "alert_performance" in metrics
        assert metrics["user_engagement"]["unique_users"] == 5

    def test_get_popular_search_queries(self, analytics_service, mock_db):
        """Test getting popular queries"""
        mock_db.fetch_all.return_value = [
            {
                "query": json.dumps({"q": "AI", "source": "all"}),
                "usage_count": 50,
                "unique_users": 10,
                "last_used": datetime.now(timezone.utc),
                "has_alert": True,
                "search_names": ["AI News", "AI Updates", "AI Research"],
            }
        ]

        popular = analytics_service.get_popular_search_queries(organization_id="org-123", limit=10)

        assert len(popular) == 1
        assert popular[0]["usage_count"] == 50
        assert popular[0]["query_components"]["text_search"] == "AI"

    def test_calculate_effectiveness_score(self, analytics_service):
        """Test effectiveness score calculation"""
        alert_stats = {"trigger_count": 50, "avg_results_per_trigger": 5, "max_results": 20}

        score = analytics_service._calculate_effectiveness_score(alert_stats)

        assert 0 <= score <= 1.0
        assert score > 0.5  # Should be relatively effective

    def test_get_tag_analytics(self, analytics_service, mock_db):
        """Test tag analytics"""
        mock_db.fetch_all.side_effect = [
            # Tag frequency
            [
                {"tag": "urgent", "usage_count": 25, "unique_users": 8, "used_in_alerts": True, "used_in_shared": True},
                {"tag": "tech", "usage_count": 20, "unique_users": 5, "used_in_alerts": False, "used_in_shared": True},
            ],
            # Co-occurrence
            [{"tag1": "urgent", "tag2": "tech", "cooccurrence_count": 5}],
        ]

        tag_analytics = analytics_service.get_tag_analytics("org-123")

        assert "tag_frequency" in tag_analytics
        assert "tag_relationships" in tag_analytics
        assert "summary" in tag_analytics
        assert tag_analytics["summary"]["unique_tags"] == 2


class TestEndToEndIntegration:
    """Test complete saved search workflow"""

    @pytest.mark.asyncio
    async def test_complete_alert_workflow(self, saved_search_manager, alert_scheduler, analytics_service, mock_db):
        """Test creating, executing, and analyzing an alert"""
        # Step 1: Create saved search alert
        mock_db.fetch_one.return_value = {
            "id": "alert-test-123",
            "name": "Integration Test Alert",
            "query": json.dumps({"q": "breaking news"}),
            "is_alert": True,
            "alert_frequency": 300,  # 5 minutes
            "notification_channels": ["email"],
            "created_at": datetime.now(timezone.utc),
        }

        search_config = saved_search_manager.create_saved_search(
            organization_id="org-test",
            user_id="user-test",
            name="Integration Test Alert",
            query={"q": "breaking news"},
            is_alert=True,
            alert_frequency=300,
            notification_channels=["email"],
        )

        assert search_config.id == "alert-test-123"

        # Step 2: Execute the alert
        mock_db.fetch_all.return_value = [
            {"id": "article-123", "title": "Breaking: Test News", "scraped_at": datetime.now(timezone.utc)}
        ]

        result = await alert_scheduler.execute_alert_immediately("alert-test-123")
        assert result["status"] == "success"

        # Step 3: Analyze the alert
        mock_db.fetch_all.return_value = [
            {
                "id": "alert-test-123",
                "name": "Integration Test Alert",
                "alert_frequency": 300,
                "trigger_count": 10,
                "avg_results_per_trigger": 3,
                "max_results": 8,
            }
        ]

        effectiveness = analytics_service.get_alert_effectiveness("org-test", days=7)
        assert "summary" in effectiveness
        assert effectiveness["summary"]["total_alerts"] > 0


@pytest.mark.asyncio
async def test_concurrent_alert_execution(alert_scheduler, mock_db):
    """Test concurrent execution of multiple alerts"""
    # Create multiple alerts
    alerts = [
        AlertJob(
            alert_id=f"alert-{i}",
            organization_id="org-123",
            user_id=f"user-{i}",
            name=f"Alert {i}",
            query={"q": f"query-{i}"},
            notification_channels=["email"],
            notification_settings={},
            next_check_at=datetime.now(timezone.utc),
            frequency=3600,
        )
        for i in range(5)
    ]

    # Mock database to return these alerts
    mock_db.fetch_all.return_value = [
        {
            "id": alert.alert_id,
            "organization_id": alert.organization_id,
            "user_id": alert.user_id,
            "name": alert.name,
            "query": json.dumps(alert.query),
            "notification_channels": alert.notification_channels,
            "notification_settings": alert.notification_settings,
            "next_check_at": alert.next_check_at,
            "alert_frequency": alert.frequency,
        }
        for alert in alerts
    ]

    # Execute alerts concurrently
    alert_scheduler._get_due_alerts = AsyncMock(return_value=alerts)
    alert_scheduler._execute_alert = AsyncMock()

    await alert_scheduler._check_and_execute_alerts()

    # Verify all alerts were executed
    assert alert_scheduler._execute_alert.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
