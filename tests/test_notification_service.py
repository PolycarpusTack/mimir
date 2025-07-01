"""
Tests for the Notification Service
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from services.notification_service import (
    EmailHandler,
    NotificationChannel,
    NotificationPriority,
    NotificationService,
    NotificationTemplate,
    SlackHandler,
    TeamsHandler,
    WebhookHandler,
)


@pytest.fixture
def mock_db_manager():
    """Mock database manager"""
    db = Mock()
    db.fetch_one = Mock()
    db.fetch_all = Mock()
    db.execute_query = Mock()
    return db


@pytest.fixture
def notification_service(mock_db_manager):
    """Create notification service instance"""
    return NotificationService(mock_db_manager)


class TestEmailHandler:
    """Test email notification handler"""

    @pytest.fixture
    def email_config(self):
        return {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "smtp_user": "test@example.com",
            "smtp_password": "password",
            "from_email": "noreply@example.com",
            "from_name": "Mimir Alerts",
            "use_tls": True,
        }

    @pytest.fixture
    def email_handler(self, email_config):
        return EmailHandler(email_config)

    def test_validate_config_success(self, email_handler, email_config):
        """Test successful config validation"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(email_handler.validate_config(email_config))
        assert result is True

    def test_validate_config_missing_fields(self, email_handler):
        """Test config validation with missing fields"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        invalid_config = {"smtp_host": "smtp.example.com"}
        result = loop.run_until_complete(email_handler.validate_config(invalid_config))
        assert result is False

    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp, email_handler):
        """Test successful email sending"""
        # Mock SMTP
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            email_handler.send(
                recipient="user@example.com",
                subject="Test Alert",
                body="<h1>Test Body</h1>",
                metadata={"headers": {"X-Priority": "High"}},
            )
        )

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()


class TestSlackHandler:
    """Test Slack notification handler"""

    @pytest.fixture
    def slack_config(self):
        return {"webhook_url": "https://hooks.slack.com/test", "default_channel": "#alerts"}

    @pytest.fixture
    def slack_handler(self, slack_config):
        return SlackHandler(slack_config)

    @pytest.mark.asyncio
    async def test_send_slack_webhook(self, slack_handler):
        """Test sending Slack notification via webhook"""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session

            # Force handler to use our mock session
            slack_handler.session = mock_session

            result = await slack_handler.send(
                recipient="#general", subject="Test Alert", body="Test notification body", metadata={"attachments": []}
            )

            assert result is True
            mock_session.post.assert_called_once()

            # Verify the data sent
            call_args = mock_session.post.call_args
            assert call_args[0][0] == slack_handler.config["webhook_url"]
            data = call_args[1]["json"]
            assert data["channel"] == "#general"
            assert "blocks" in data


class TestNotificationService:
    """Test main notification service"""

    @pytest.mark.asyncio
    async def test_configure_channel(self, notification_service):
        """Test channel configuration"""
        config = {
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_email": "test@example.com",
            "from_name": "Test",
        }

        notification_service.configure_channel(
            channel=NotificationChannel.EMAIL, config=config, enabled=True, rate_limit=100
        )

        assert NotificationChannel.EMAIL in notification_service.handlers
        assert NotificationChannel.EMAIL in notification_service.channel_configs

        channel_config = notification_service.channel_configs[NotificationChannel.EMAIL]
        assert channel_config.enabled is True
        assert channel_config.rate_limit == 100

    @pytest.mark.asyncio
    async def test_send_notification_success(self, notification_service, mock_db_manager):
        """Test successful notification sending"""
        # Configure mock handler
        mock_handler = AsyncMock()
        mock_handler.send = AsyncMock(return_value=True)

        notification_service.handlers[NotificationChannel.EMAIL] = mock_handler
        notification_service.channel_configs[NotificationChannel.EMAIL] = Mock(enabled=True, rate_limit=None)

        # Mock rate limit check
        mock_db_manager.fetch_one.return_value = {"count": 0}

        # Mock template
        mock_db_manager.fetch_one.return_value = {
            "subject_template": "Alert: {{ alert_name }}",
            "body_template": "<h1>{{ alert_name }}</h1><p>{{ result_count }} results</p>",
        }

        result = await notification_service.send_notification(
            channel=NotificationChannel.EMAIL,
            recipient="user@example.com",
            notification_type="saved_search_alert",
            data={"alert_name": "Test Alert", "result_count": 5},
            template_id="test-template",
        )

        assert result is True
        mock_handler.send.assert_called_once()

        # Verify log entry
        mock_db_manager.execute_query.assert_called()

    @pytest.mark.asyncio
    async def test_send_notification_channel_disabled(self, notification_service):
        """Test sending to disabled channel"""
        notification_service.channel_configs[NotificationChannel.EMAIL] = Mock(enabled=False)

        result = await notification_service.send_notification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com", notification_type="test", data={}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_notification_rate_limited(self, notification_service, mock_db_manager):
        """Test rate limiting"""
        notification_service.channel_configs[NotificationChannel.EMAIL] = Mock(enabled=True, rate_limit=10)

        # Mock rate limit exceeded
        mock_db_manager.fetch_one.return_value = {"count": 15}

        result = await notification_service.send_notification(
            channel=NotificationChannel.EMAIL, recipient="user@example.com", notification_type="test", data={}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_multi_channel(self, notification_service, mock_db_manager):
        """Test sending to multiple channels"""
        # Configure mock handlers
        email_handler = AsyncMock()
        email_handler.send = AsyncMock(return_value=True)

        slack_handler = AsyncMock()
        slack_handler.send = AsyncMock(return_value=False)

        notification_service.handlers = {
            NotificationChannel.EMAIL: email_handler,
            NotificationChannel.SLACK: slack_handler,
        }

        notification_service.channel_configs = {
            NotificationChannel.EMAIL: Mock(enabled=True, rate_limit=None),
            NotificationChannel.SLACK: Mock(enabled=True, rate_limit=None),
        }

        # Mock rate limit check
        mock_db_manager.fetch_one.return_value = {"count": 0}

        results = await notification_service.send_multi_channel(
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            recipient="user@example.com",
            notification_type="test",
            data={"test": "data"},
        )

        assert results[NotificationChannel.EMAIL] is True
        assert results[NotificationChannel.SLACK] is False

        email_handler.send.assert_called_once()
        slack_handler.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_content_saved_search_alert(self, notification_service):
        """Test content preparation for saved search alerts"""
        data = {
            "alert_name": "Tech News Alert",
            "result_count": 3,
            "query": {"q": "technology"},
            "results": [
                {"title": "Article 1", "source": "TechCrunch", "published_date": "2024-01-01"},
                {"title": "Article 2", "source": "Wired", "published_date": "2024-01-02"},
            ],
        }

        subject, body = notification_service._prepare_content("saved_search_alert", data)

        assert "Tech News Alert" in subject
        assert "Tech News Alert" in body
        assert "3" in body
        assert "Article 1" in body
        assert "TechCrunch" in body

    @pytest.mark.asyncio
    async def test_get_notification_history(self, notification_service, mock_db_manager):
        """Test retrieving notification history"""
        mock_history = [
            {
                "id": "123",
                "channel": "email",
                "recipient": "user@example.com",
                "notification_type": "saved_search_alert",
                "status": "sent",
                "created_at": datetime.now(timezone.utc),
            }
        ]

        mock_db_manager.fetch_all.return_value = mock_history

        history = await notification_service.get_notification_history(
            recipient="user@example.com", channel=NotificationChannel.EMAIL, days=7
        )

        assert len(history) == 1
        assert history[0]["channel"] == "email"
        mock_db_manager.fetch_all.assert_called_once()


class TestNotificationTemplate:
    """Test notification template model"""

    def test_extract_variables(self):
        """Test variable extraction from template"""
        template = NotificationTemplate(
            id="test",
            name="Test Template",
            channel=NotificationChannel.EMAIL,
            subject_template="Alert: {{ alert_name }} - {{ count }} items",
            body_template="<h1>{{ alert_name }}</h1><p>Found {{ count }} results for {{ query }}</p>",
        )

        expected_vars = {"alert_name", "count", "query"}
        assert set(template.variables) == expected_vars

    def test_template_without_variables(self):
        """Test template without variables"""
        template = NotificationTemplate(
            id="test",
            name="Static Template",
            channel=NotificationChannel.EMAIL,
            body_template="<h1>Static Alert</h1><p>This is a static notification.</p>",
        )

        assert template.variables == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
