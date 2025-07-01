"""
Redis Streams Event System for Mimir Async Scraper
EPIC 4 - Task 4.4.1: Event streaming with Redis Streams and real-time notifications
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import aioredis
import redis

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the streaming system."""

    ARTICLE_DISCOVERED = "article.discovered"
    ARTICLE_PROCESSED = "article.processed"
    ARTICLE_FAILED = "article.failed"
    SCRAPE_STARTED = "scrape.started"
    SCRAPE_COMPLETED = "scrape.completed"
    SCRAPE_FAILED = "scrape.failed"
    SITE_PROCESSED = "site.processed"
    KEYWORD_ALERT = "keyword.alert"
    AI_ANALYSIS_COMPLETED = "ai.analysis.completed"
    NOTIFICATION_SENT = "notification.sent"
    CACHE_MISS = "cache.miss"
    CACHE_HIT = "cache.hit"
    SYSTEM_HEALTH = "system.health"
    RATE_LIMIT_HIT = "rate_limit.hit"


@dataclass
class Event:
    """Base event structure."""

    event_type: EventType
    timestamp: float
    event_id: str
    source: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "source": self.source,
            "data": self.data,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            event_id=data["event_id"],
            source=data["source"],
            data=data["data"],
            correlation_id=data.get("correlation_id"),
        )


class EventPublisher:
    """Publisher for Redis Streams events."""

    def __init__(self, redis_url: str, stream_prefix: str = "mimir"):
        self.redis_url = redis_url
        self.stream_prefix = stream_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.async_client: Optional[aioredis.Redis] = None

        # Stream names for different event types
        self.streams = {
            "articles": f"{stream_prefix}:articles",
            "scraping": f"{stream_prefix}:scraping",
            "notifications": f"{stream_prefix}:notifications",
            "system": f"{stream_prefix}:system",
            "ai": f"{stream_prefix}:ai",
        }

        # Published events counter
        self.published_count = 0

    def connect(self):
        """Connect to Redis (sync)."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Event publisher connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect event publisher: {e}")
            raise

    async def async_connect(self):
        """Connect to Redis (async)."""
        try:
            self.async_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.async_client.ping()
            logger.info("Async event publisher connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect async event publisher: {e}")
            raise

    def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            self.redis_client.close()

    async def async_disconnect(self):
        """Disconnect from Redis (async)."""
        if self.async_client:
            await self.async_client.close()

    def _get_stream_name(self, event_type: EventType) -> str:
        """Get stream name for event type."""
        if event_type.value.startswith("article"):
            return self.streams["articles"]
        elif event_type.value.startswith("scrape"):
            return self.streams["scraping"]
        elif event_type.value.startswith("notification"):
            return self.streams["notifications"]
        elif event_type.value.startswith("ai"):
            return self.streams["ai"]
        else:
            return self.streams["system"]

    def publish(self, event: Event) -> str:
        """Publish event to Redis Stream (sync)."""
        if not self.redis_client:
            raise RuntimeError("Publisher not connected")

        try:
            stream_name = self._get_stream_name(event.event_type)
            event_data = event.to_dict()

            # Add event to stream
            stream_id = self.redis_client.xadd(stream_name, event_data)
            self.published_count += 1

            logger.debug(f"Published event {event.event_id} to {stream_name}")
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            raise

    async def async_publish(self, event: Event) -> str:
        """Publish event to Redis Stream (async)."""
        if not self.async_client:
            raise RuntimeError("Async publisher not connected")

        try:
            stream_name = self._get_stream_name(event.event_type)
            event_data = event.to_dict()

            # Add event to stream
            stream_id = await self.async_client.xadd(stream_name, event_data)
            self.published_count += 1

            logger.debug(f"Published event {event.event_id} to {stream_name}")
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish async event: {e}")
            raise

    def create_event(
        self, event_type: EventType, data: Dict[str, Any], source: str = "scraper", correlation_id: Optional[str] = None
    ) -> Event:
        """Create a new event."""
        return Event(
            event_type=event_type,
            timestamp=time.time(),
            event_id=str(uuid.uuid4()),
            source=source,
            data=data,
            correlation_id=correlation_id,
        )

    # Convenience methods for common events
    def publish_article_discovered(self, article_url: str, source_site: str, correlation_id: str = None):
        """Publish article discovered event."""
        event = self.create_event(
            EventType.ARTICLE_DISCOVERED,
            {"url": article_url, "source_site": source_site, "discovered_at": datetime.now().isoformat()},
            correlation_id=correlation_id,
        )
        return self.publish(event)

    def publish_article_processed(
        self, article_id: int, article_url: str, processing_time: float, correlation_id: str = None
    ):
        """Publish article processed event."""
        event = self.create_event(
            EventType.ARTICLE_PROCESSED,
            {
                "article_id": article_id,
                "url": article_url,
                "processing_time": processing_time,
                "processed_at": datetime.now().isoformat(),
            },
            correlation_id=correlation_id,
        )
        return self.publish(event)

    def publish_scrape_started(self, run_id: int, sites_count: int):
        """Publish scrape run started event."""
        event = self.create_event(
            EventType.SCRAPE_STARTED,
            {"run_id": run_id, "sites_count": sites_count, "started_at": datetime.now().isoformat()},
        )
        return self.publish(event)

    def publish_scrape_completed(self, run_id: int, new_articles: int, errors: int, duration: float):
        """Publish scrape run completed event."""
        event = self.create_event(
            EventType.SCRAPE_COMPLETED,
            {
                "run_id": run_id,
                "new_articles": new_articles,
                "errors": errors,
                "duration": duration,
                "completed_at": datetime.now().isoformat(),
            },
        )
        return self.publish(event)

    def publish_keyword_alert(self, keyword: str, article_id: int, article_title: str, context: str):
        """Publish keyword alert event."""
        event = self.create_event(
            EventType.KEYWORD_ALERT,
            {
                "keyword": keyword,
                "article_id": article_id,
                "article_title": article_title,
                "context": context,
                "alert_time": datetime.now().isoformat(),
            },
        )
        return self.publish(event)


class EventConsumer:
    """Consumer for Redis Streams events."""

    def __init__(self, redis_url: str, consumer_group: str, consumer_name: str, stream_prefix: str = "mimir"):
        self.redis_url = redis_url
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.stream_prefix = stream_prefix

        self.redis_client: Optional[redis.Redis] = None
        self.async_client: Optional[aioredis.Redis] = None

        # Stream names
        self.streams = {
            "articles": f"{stream_prefix}:articles",
            "scraping": f"{stream_prefix}:scraping",
            "notifications": f"{stream_prefix}:notifications",
            "system": f"{stream_prefix}:system",
            "ai": f"{stream_prefix}:ai",
        }

        # Event handlers
        self.handlers: Dict[EventType, List[Callable]] = {}

        # Consumer state
        self.is_consuming = False
        self.processed_count = 0
        self.error_count = 0

    def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Event consumer {self.consumer_name} connected")
            self._create_consumer_groups()
        except Exception as e:
            logger.error(f"Failed to connect event consumer: {e}")
            raise

    async def async_connect(self):
        """Connect to Redis (async)."""
        try:
            self.async_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.async_client.ping()
            logger.info(f"Async event consumer {self.consumer_name} connected")
            await self._async_create_consumer_groups()
        except Exception as e:
            logger.error(f"Failed to connect async event consumer: {e}")
            raise

    def _create_consumer_groups(self):
        """Create consumer groups for all streams."""
        for stream_name in self.streams.values():
            try:
                self.redis_client.xgroup_create(stream_name, self.consumer_group, "0", mkstream=True)
                logger.debug(f"Created consumer group {self.consumer_group} for {stream_name}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Error creating consumer group: {e}")

    async def _async_create_consumer_groups(self):
        """Create consumer groups for all streams (async)."""
        for stream_name in self.streams.values():
            try:
                await self.async_client.xgroup_create(stream_name, self.consumer_group, "0", mkstream=True)
                logger.debug(f"Created consumer group {self.consumer_group} for {stream_name}")
            except aioredis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.error(f"Error creating consumer group: {e}")

    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]):
        """Register event handler."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type.value}")

    def remove_handler(self, event_type: EventType, handler: Callable):
        """Remove event handler."""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(handler)
            except ValueError:
                pass

    async def consume_events(self, block_time: int = 1000, count: int = 10) -> AsyncGenerator[Event, None]:
        """Consume events from all streams."""
        if not self.async_client:
            raise RuntimeError("Consumer not connected")

        self.is_consuming = True
        logger.info(f"Starting event consumption for {self.consumer_name}")

        try:
            while self.is_consuming:
                try:
                    # Read from all streams
                    stream_list = {stream: ">" for stream in self.streams.values()}

                    messages = await self.async_client.xreadgroup(
                        self.consumer_group, self.consumer_name, stream_list, count=count, block=block_time
                    )

                    for stream_name, events in messages:
                        for event_id, fields in events:
                            try:
                                # Convert Redis response to Event object
                                event = Event.from_dict(fields)

                                # Yield the event
                                yield event

                                # Acknowledge the message
                                await self.async_client.xack(stream_name, self.consumer_group, event_id)
                                self.processed_count += 1

                            except Exception as e:
                                self.error_count += 1
                                logger.error(f"Error processing event {event_id}: {e}")

                except asyncio.TimeoutError:
                    # Normal timeout, continue consuming
                    continue
                except Exception as e:
                    logger.error(f"Error in event consumption: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Event consumption cancelled")
        finally:
            self.is_consuming = False

    async def process_events(self, block_time: int = 1000):
        """Process events using registered handlers."""
        async for event in self.consume_events(block_time):
            try:
                # Call all registered handlers for this event type
                if event.event_type in self.handlers:
                    for handler in self.handlers[event.event_type]:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as e:
                            logger.error(f"Error in event handler: {e}")

            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def stop_consuming(self):
        """Stop event consumption."""
        self.is_consuming = False

    def get_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        return {
            "consumer_name": self.consumer_name,
            "consumer_group": self.consumer_group,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "is_consuming": self.is_consuming,
            "timestamp": datetime.now().isoformat(),
        }


class RealTimeNotificationSystem:
    """Real-time notification system using WebSockets and SSE."""

    def __init__(self, event_consumer: EventConsumer):
        self.consumer = event_consumer
        self.connected_clients: List[Any] = []  # WebSocket connections
        self.sse_clients: List[Any] = []  # SSE connections

        # Register event handlers for notifications
        self.consumer.register_handler(EventType.ARTICLE_PROCESSED, self._handle_article_event)
        self.consumer.register_handler(EventType.KEYWORD_ALERT, self._handle_keyword_alert)
        self.consumer.register_handler(EventType.SCRAPE_COMPLETED, self._handle_scrape_completed)
        self.consumer.register_handler(EventType.SYSTEM_HEALTH, self._handle_system_health)

    async def _handle_article_event(self, event: Event):
        """Handle article-related events."""
        notification = {"type": "article_update", "data": event.data, "timestamp": event.timestamp}
        await self._broadcast_notification(notification)

    async def _handle_keyword_alert(self, event: Event):
        """Handle keyword alerts."""
        notification = {"type": "keyword_alert", "data": event.data, "timestamp": event.timestamp, "priority": "high"}
        await self._broadcast_notification(notification)

    async def _handle_scrape_completed(self, event: Event):
        """Handle scrape completion events."""
        notification = {"type": "scrape_update", "data": event.data, "timestamp": event.timestamp}
        await self._broadcast_notification(notification)

    async def _handle_system_health(self, event: Event):
        """Handle system health events."""
        notification = {"type": "system_health", "data": event.data, "timestamp": event.timestamp}
        await self._broadcast_notification(notification)

    async def _broadcast_notification(self, notification: Dict[str, Any]):
        """Broadcast notification to all connected clients."""
        message = json.dumps(notification)

        # Broadcast to WebSocket clients
        disconnected_ws = []
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_ws.append(client)

        # Remove disconnected clients
        for client in disconnected_ws:
            self.connected_clients.remove(client)

        # Broadcast to SSE clients
        disconnected_sse = []
        for client in self.sse_clients:
            try:
                await client.put(f"data: {message}\n\n")
            except Exception:
                disconnected_sse.append(client)

        # Remove disconnected SSE clients
        for client in disconnected_sse:
            self.sse_clients.remove(client)

    def add_websocket_client(self, client):
        """Add WebSocket client."""
        self.connected_clients.append(client)

    def remove_websocket_client(self, client):
        """Remove WebSocket client."""
        if client in self.connected_clients:
            self.connected_clients.remove(client)

    def add_sse_client(self, client):
        """Add SSE client."""
        self.sse_clients.append(client)

    def remove_sse_client(self, client):
        """Remove SSE client."""
        if client in self.sse_clients:
            self.sse_clients.remove(client)

    def get_client_count(self) -> Dict[str, int]:
        """Get count of connected clients."""
        return {
            "websocket_clients": len(self.connected_clients),
            "sse_clients": len(self.sse_clients),
            "total_clients": len(self.connected_clients) + len(self.sse_clients),
        }


# Global event system instances
class EventSystem:
    """Global event system manager."""

    def __init__(self):
        self.publisher: Optional[EventPublisher] = None
        self.consumer: Optional[EventConsumer] = None
        self.notification_system: Optional[RealTimeNotificationSystem] = None

    def initialize(self, redis_url: str, consumer_group: str = "mimir_consumers", consumer_name: str = None):
        """Initialize the event system."""
        if consumer_name is None:
            consumer_name = f"consumer_{int(time.time())}"

        self.publisher = EventPublisher(redis_url)
        self.consumer = EventConsumer(redis_url, consumer_group, consumer_name)
        self.notification_system = RealTimeNotificationSystem(self.consumer)

    async def start(self):
        """Start the event system."""
        if self.publisher:
            await self.publisher.async_connect()
        if self.consumer:
            await self.consumer.async_connect()

    async def stop(self):
        """Stop the event system."""
        if self.consumer:
            self.consumer.stop_consuming()
        if self.publisher:
            await self.publisher.async_disconnect()


# Global instance
event_system = EventSystem()
