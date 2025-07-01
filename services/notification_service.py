"""
Notification Service for Mimir Enterprise

Handles multi-channel notifications including Email, Slack, Microsoft Teams, and Webhooks.
"""

import asyncio
import json
import logging
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import aiohttp
import jinja2
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Available notification channels"""

    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"


class NotificationPriority(str, Enum):
    """Notification priority levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    """Notification delivery status"""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel"""

    channel: NotificationChannel
    enabled: bool
    settings: Dict[str, Any]
    rate_limit: Optional[int] = None  # Messages per hour
    retry_count: int = 3
    retry_delay: int = 60  # seconds


class NotificationTemplate(BaseModel):
    """Notification template model"""

    id: str
    name: str
    channel: NotificationChannel
    subject_template: Optional[str] = None
    body_template: str
    variables: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @validator("variables", pre=True)
    def extract_variables(cls, v, values):
        """Extract variables from template"""
        if not v and "body_template" in values:
            import re

            # Find all {{ variable }} patterns
            pattern = r"\{\{\s*(\w+)\s*\}\}"
            variables = re.findall(pattern, values["body_template"])
            if "subject_template" in values and values["subject_template"]:
                variables.extend(re.findall(pattern, values["subject_template"]))
            return list(set(variables))
        return v


class NotificationHandler(ABC):
    """Abstract base class for notification handlers"""

    @abstractmethod
    async def send(self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification to recipient"""
        pass

    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate channel configuration"""
        pass


class EmailHandler(NotificationHandler):
    """Email notification handler"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email handler

        Config should include:
        - smtp_host: SMTP server hostname
        - smtp_port: SMTP server port
        - smtp_user: SMTP username
        - smtp_password: SMTP password
        - from_email: Sender email address
        - from_name: Sender name
        - use_tls: Whether to use TLS
        """
        self.config = config
        self.template_engine = jinja2.Environment(autoescape=jinja2.select_autoescape(["html", "xml"]))

    async def send(self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send email notification"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._send_email_sync, recipient, subject, body, metadata)
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            return False

    def _send_email_sync(
        self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Synchronous email sending"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"{self.config['from_name']} <{self.config['from_email']}>"
            msg["To"] = recipient

            # Add custom headers if provided
            if metadata and "headers" in metadata:
                for key, value in metadata["headers"].items():
                    msg[key] = value

            # Create HTML and text parts
            text_part = MIMEText(self._html_to_text(body), "plain")
            html_part = MIMEText(body, "html")

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.config["smtp_host"], self.config["smtp_port"]) as server:
                if self.config.get("use_tls", True):
                    server.starttls()

                if self.config.get("smtp_user") and self.config.get("smtp_password"):
                    server.login(self.config["smtp_user"], self.config["smtp_password"])

                server.send_message(msg)

            logger.info(f"Email sent successfully to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text"""
        # Simple conversion - in production, use html2text library
        import re

        text = re.sub("<[^<]+?>", "", html)
        return text.strip()

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate email configuration"""
        required_fields = ["smtp_host", "smtp_port", "from_email", "from_name"]
        return all(field in config for field in required_fields)


class SlackHandler(NotificationHandler):
    """Slack notification handler"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Slack handler

        Config should include:
        - webhook_url: Slack webhook URL (for simple notifications)
        - bot_token: Slack bot token (for advanced features)
        - default_channel: Default channel to post to
        """
        self.config = config
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send(self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send Slack notification"""
        try:
            session = await self._get_session()

            # Prepare Slack message
            slack_data = {
                "channel": recipient or self.config.get("default_channel", "#general"),
                "username": "Mimir Alerts",
                "icon_emoji": ":newspaper:",
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": subject}},
                    {"type": "section", "text": {"type": "mrkdwn", "text": body}},
                ],
            }

            # Add attachments if provided
            if metadata and "attachments" in metadata:
                slack_data["attachments"] = metadata["attachments"]

            # Send via webhook or API
            if self.config.get("webhook_url"):
                async with session.post(self.config["webhook_url"], json=slack_data) as response:
                    success = response.status == 200
            else:
                # Use bot token API
                headers = {"Authorization": f"Bearer {self.config['bot_token']}", "Content-Type": "application/json"}
                async with session.post(
                    "https://slack.com/api/chat.postMessage", headers=headers, json=slack_data
                ) as response:
                    result = await response.json()
                    success = result.get("ok", False)

            if success:
                logger.info(f"Slack notification sent to {recipient}")
            else:
                logger.error(f"Slack notification failed for {recipient}")

            return success

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Slack configuration"""
        return "webhook_url" in config or "bot_token" in config

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()


class TeamsHandler(NotificationHandler):
    """Microsoft Teams notification handler"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Teams handler

        Config should include:
        - webhook_url: Teams incoming webhook URL
        """
        self.config = config
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send(self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send Teams notification"""
        try:
            session = await self._get_session()

            # Prepare Teams card
            teams_data = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": subject,
                "themeColor": "0076D7",
                "sections": [{"activityTitle": subject, "text": body, "markdown": True}],
            }

            # Add actions if provided
            if metadata and "actions" in metadata:
                teams_data["potentialAction"] = metadata["actions"]

            # Send to webhook
            async with session.post(self.config["webhook_url"], json=teams_data) as response:
                success = response.status == 200

            if success:
                logger.info(f"Teams notification sent: {subject}")
            else:
                logger.error(f"Teams notification failed: {response.status}")

            return success

        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Teams configuration"""
        return "webhook_url" in config

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()


class WebhookHandler(NotificationHandler):
    """Generic webhook notification handler"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize webhook handler

        Config should include:
        - url: Webhook URL
        - method: HTTP method (default: POST)
        - headers: Additional headers
        - auth_type: Authentication type (basic, bearer, api_key)
        - auth_value: Authentication value
        """
        self.config = config
        self.session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def send(self, recipient: str, subject: str, body: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send webhook notification"""
        try:
            session = await self._get_session()

            # Prepare payload
            payload = {
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            # Prepare headers
            headers = self.config.get("headers", {}).copy()

            # Add authentication
            auth_type = self.config.get("auth_type")
            auth_value = self.config.get("auth_value")

            if auth_type == "bearer":
                headers["Authorization"] = f"Bearer {auth_value}"
            elif auth_type == "api_key":
                headers["X-API-Key"] = auth_value
            elif auth_type == "basic":
                import base64

                auth_str = base64.b64encode(auth_value.encode()).decode()
                headers["Authorization"] = f"Basic {auth_str}"

            # Send webhook
            method = self.config.get("method", "POST").upper()
            async with session.request(method, self.config["url"], json=payload, headers=headers) as response:
                success = 200 <= response.status < 300

            if success:
                logger.info(f"Webhook notification sent to {self.config['url']}")
            else:
                logger.error(f"Webhook notification failed: {response.status}")

            return success

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate webhook configuration"""
        return "url" in config

    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()


class NotificationService:
    """Main notification service orchestrator"""

    def __init__(self, db_manager, template_manager=None):
        """
        Initialize notification service

        Args:
            db_manager: Database manager for storing notification history
            template_manager: Optional template manager
        """
        self.db = db_manager
        self.template_manager = template_manager
        self.handlers: Dict[NotificationChannel, NotificationHandler] = {}
        self.channel_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self._template_engine = jinja2.Environment(autoescape=jinja2.select_autoescape(["html", "xml"]))

    def configure_channel(
        self,
        channel: NotificationChannel,
        config: Dict[str, Any],
        enabled: bool = True,
        rate_limit: Optional[int] = None,
    ):
        """Configure a notification channel"""
        try:
            # Create handler based on channel type
            if channel == NotificationChannel.EMAIL:
                handler = EmailHandler(config)
            elif channel == NotificationChannel.SLACK:
                handler = SlackHandler(config)
            elif channel == NotificationChannel.TEAMS:
                handler = TeamsHandler(config)
            elif channel == NotificationChannel.WEBHOOK:
                handler = WebhookHandler(config)
            else:
                raise ValueError(f"Unsupported channel: {channel}")

            # Store handler and config
            self.handlers[channel] = handler
            self.channel_configs[channel] = NotificationConfig(
                channel=channel, enabled=enabled, settings=config, rate_limit=rate_limit
            )

            logger.info(f"Configured notification channel: {channel}")

        except Exception as e:
            logger.error(f"Failed to configure channel {channel}: {e}")
            raise

    async def send_notification(
        self,
        channel: NotificationChannel,
        recipient: str,
        notification_type: str,
        data: Dict[str, Any],
        template_id: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ) -> bool:
        """
        Send notification through specified channel

        Args:
            channel: Notification channel
            recipient: Recipient identifier (email, channel name, etc.)
            notification_type: Type of notification
            data: Notification data/context
            template_id: Optional template ID
            priority: Notification priority

        Returns:
            True if sent successfully
        """
        try:
            # Check if channel is configured and enabled
            if channel not in self.channel_configs:
                logger.error(f"Channel {channel} not configured")
                return False

            config = self.channel_configs[channel]
            if not config.enabled:
                logger.warning(f"Channel {channel} is disabled")
                return False

            # Check rate limits
            if config.rate_limit and not await self._check_rate_limit(channel, recipient, config.rate_limit):
                logger.warning(f"Rate limit exceeded for {channel} to {recipient}")
                return False

            # Get handler
            handler = self.handlers.get(channel)
            if not handler:
                logger.error(f"No handler for channel {channel}")
                return False

            # Prepare notification content
            if template_id:
                subject, body = await self._render_template(template_id, data)
            else:
                subject, body = self._prepare_content(notification_type, data)

            # Send notification
            success = await handler.send(
                recipient=recipient,
                subject=subject,
                body=body,
                metadata={"notification_type": notification_type, "priority": priority, "data": data},
            )

            # Log notification
            await self._log_notification(
                channel=channel,
                recipient=recipient,
                notification_type=notification_type,
                subject=subject,
                status=NotificationStatus.SENT if success else NotificationStatus.FAILED,
                priority=priority,
            )

            return success

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def send_multi_channel(
        self,
        channels: List[NotificationChannel],
        recipient: str,
        notification_type: str,
        data: Dict[str, Any],
        template_id: Optional[str] = None,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
    ) -> Dict[NotificationChannel, bool]:
        """Send notification to multiple channels"""
        results = {}

        for channel in channels:
            results[channel] = await self.send_notification(
                channel=channel,
                recipient=recipient,
                notification_type=notification_type,
                data=data,
                template_id=template_id,
                priority=priority,
            )

        return results

    async def _check_rate_limit(self, channel: NotificationChannel, recipient: str, limit: int) -> bool:
        """Check if rate limit is exceeded"""
        try:
            # Count recent notifications
            query = """
            SELECT COUNT(*) as count
            FROM notification_log
            WHERE channel = %s
              AND recipient = %s
              AND created_at > %s
            """

            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            result = self.db.fetch_one(query, (channel.value, recipient, one_hour_ago))

            return result["count"] < limit if result else True

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error

    async def _render_template(self, template_id: str, data: Dict[str, Any]) -> tuple[str, str]:
        """Render notification template"""
        try:
            # Get template from database
            query = """
            SELECT * FROM notification_templates
            WHERE id = %s
            """

            template_data = self.db.fetch_one(query, (template_id,))
            if not template_data:
                raise ValueError(f"Template {template_id} not found")

            # Render template
            subject_template = self._template_engine.from_string(template_data.get("subject_template", "Notification"))
            body_template = self._template_engine.from_string(template_data["body_template"])

            subject = subject_template.render(**data)
            body = body_template.render(**data)

            return subject, body

        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return self._prepare_content("default", data)

    def _prepare_content(self, notification_type: str, data: Dict[str, Any]) -> tuple[str, str]:
        """Prepare notification content based on type"""

        if notification_type == "saved_search_alert":
            subject = f"Alert: {data.get('alert_name', 'Saved Search')}"
            body = f"""
            <h2>Your saved search alert has new results</h2>
            <p><strong>Alert:</strong> {data.get('alert_name')}</p>
            <p><strong>New Results:</strong> {data.get('result_count', 0)}</p>
            <p><strong>Query:</strong> {data.get('query', {})}</p>
            
            <h3>Top Results:</h3>
            <ul>
            """

            for result in data.get("results", [])[:5]:
                body += f"""
                <li>
                    <strong>{result.get('title', 'Untitled')}</strong><br>
                    Source: {result.get('source', 'Unknown')}<br>
                    Date: {result.get('published_date', 'Unknown')}
                </li>
                """

            body += "</ul>"

        elif notification_type == "dashboard_share":
            subject = f"Dashboard shared with you: {data.get('dashboard_name')}"
            body = f"""
            <h2>A dashboard has been shared with you</h2>
            <p><strong>Dashboard:</strong> {data.get('dashboard_name')}</p>
            <p><strong>Shared by:</strong> {data.get('shared_by')}</p>
            <p><strong>Description:</strong> {data.get('description', 'No description')}</p>
            """

        elif notification_type == "api_key_expiry":
            subject = "API Key Expiring Soon"
            body = f"""
            <h2>Your API key is expiring soon</h2>
            <p><strong>Key Name:</strong> {data.get('key_name')}</p>
            <p><strong>Expires:</strong> {data.get('expires_at')}</p>
            <p>Please generate a new API key to continue accessing the service.</p>
            """

        else:
            subject = f"Notification: {notification_type}"
            body = f"<pre>{json.dumps(data, indent=2)}</pre>"

        return subject, body

    async def _log_notification(
        self,
        channel: NotificationChannel,
        recipient: str,
        notification_type: str,
        subject: str,
        status: NotificationStatus,
        priority: NotificationPriority,
    ):
        """Log notification to database"""
        try:
            query = """
            INSERT INTO notification_log (
                channel, recipient, notification_type, subject,
                status, priority, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute_query(
                query,
                (
                    channel.value,
                    recipient,
                    notification_type,
                    subject,
                    status.value,
                    priority.value,
                    datetime.now(timezone.utc),
                ),
            )

        except Exception as e:
            logger.error(f"Error logging notification: {e}")

    async def get_notification_history(
        self, recipient: Optional[str] = None, channel: Optional[NotificationChannel] = None, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get notification history"""
        try:
            conditions = []
            params = []

            if recipient:
                conditions.append("recipient = %s")
                params.append(recipient)

            if channel:
                conditions.append("channel = %s")
                params.append(channel.value)

            conditions.append("created_at > %s")
            params.append(datetime.now(timezone.utc) - timedelta(days=days))

            query = f"""
            SELECT * FROM notification_log
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT 100
            """

            return self.db.fetch_all(query, params)

        except Exception as e:
            logger.error(f"Error getting notification history: {e}")
            return []

    async def close(self):
        """Close all handlers"""
        for handler in self.handlers.values():
            if hasattr(handler, "close"):
                await handler.close()
