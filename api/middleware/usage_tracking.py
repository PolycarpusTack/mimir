"""
Usage Tracking Middleware for Mimir Enterprise

Automatically tracks API usage for billing and analytics.
"""

import asyncio
import time
from decimal import Decimal
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...services.usage_metering import UsageMeteringService, UsageMetric


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track API usage"""

    def __init__(self, app, usage_service: Optional[UsageMeteringService] = None):
        """
        Initialize usage tracking middleware

        Args:
            app: FastAPI application
            usage_service: Optional usage metering service
        """
        super().__init__(app)
        self.usage_service = usage_service

        # Endpoints that should not be tracked
        self.excluded_paths = {"/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}

        # Map endpoints to usage metrics
        self.metric_mappings = {
            "/api/v1/articles": UsageMetric.ARTICLES_PROCESSED,
            "/api/v1/search": UsageMetric.SEARCHES,
            "/api/v1/dashboards": UsageMetric.DASHBOARD_VIEWS,
            "/api/v1/notifications/send": UsageMetric.NOTIFICATIONS_SENT,
            "/api/v1/webhooks": UsageMetric.WEBHOOKS,
            "/api/v1/analytics/export": UsageMetric.DATA_EXPORTS,
            "/api/v1/ai/analyze": UsageMetric.AI_ANALYSIS,
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track usage"""
        # Skip tracking for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Skip if no usage service
        if not self.usage_service:
            return await call_next(request)

        # Track request timing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate response time
        duration = time.time() - start_time

        # Track usage if successful
        if 200 <= response.status_code < 300:
            await self._track_usage(request, response, duration)

        return response

    async def _track_usage(self, request: Request, response: Response, duration: float):
        """Track API usage"""
        try:
            # Get user context
            user_id = None
            organization_id = None

            if hasattr(request.state, "user"):
                user_id = request.state.user.get("id")
                organization_id = request.state.user.get("organization_id")
            elif hasattr(request.state, "api_key"):
                organization_id = request.state.api_key.get("organization_id")
                user_id = request.state.api_key.get("user_id")

            if not organization_id:
                return

            # Determine metric based on endpoint
            metric = self._get_metric_for_endpoint(request.url.path, request.method)

            # Track API call
            asyncio.create_task(
                self._record_usage(
                    organization_id=organization_id,
                    metric=UsageMetric.API_CALLS,
                    value=Decimal("1"),
                    user_id=user_id,
                    metadata={
                        "endpoint": request.url.path,
                        "method": request.method,
                        "duration_ms": int(duration * 1000),
                        "status_code": response.status_code,
                    },
                )
            )

            # Track specific metric if applicable
            if metric and metric != UsageMetric.API_CALLS:
                # Get value based on response
                value = await self._get_metric_value(metric, request, response)
                if value:
                    asyncio.create_task(
                        self._record_usage(
                            organization_id=organization_id,
                            metric=metric,
                            value=value,
                            user_id=user_id,
                            metadata={"endpoint": request.url.path, "method": request.method},
                        )
                    )

        except Exception as e:
            # Log error but don't fail the request
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error tracking usage: {e}")

    def _get_metric_for_endpoint(self, path: str, method: str) -> Optional[UsageMetric]:
        """Determine usage metric based on endpoint"""
        # Check exact matches first
        for endpoint, metric in self.metric_mappings.items():
            if path.startswith(endpoint):
                return metric

        # Check patterns
        if "/articles" in path and method == "POST":
            return UsageMetric.ARTICLES_PROCESSED
        elif "/saved-searches" in path and method == "POST":
            return UsageMetric.SAVED_SEARCHES
        elif "/ai/" in path:
            return UsageMetric.AI_ANALYSIS
        elif "/export" in path:
            return UsageMetric.DATA_EXPORTS

        return None

    async def _get_metric_value(self, metric: UsageMetric, request: Request, response: Response) -> Optional[Decimal]:
        """Extract metric value from request/response"""
        try:
            if metric == UsageMetric.ARTICLES_PROCESSED:
                # Check if bulk operation
                body = await request.body()
                if body and b'"articles"' in body:
                    import json

                    data = json.loads(body)
                    if isinstance(data, dict) and "articles" in data:
                        return Decimal(len(data["articles"]))
                return Decimal("1")

            elif metric == UsageMetric.NOTIFICATIONS_SENT:
                # Check for bulk notifications
                body = await request.body()
                if body and b'"recipients"' in body:
                    import json

                    data = json.loads(body)
                    if isinstance(data, dict) and "recipients" in data:
                        return Decimal(len(data["recipients"]))
                return Decimal("1")

            else:
                # Default to 1 for most metrics
                return Decimal("1")

        except Exception:
            return Decimal("1")

    async def _record_usage(
        self,
        organization_id: str,
        metric: UsageMetric,
        value: Decimal,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Record usage asynchronously"""
        try:
            self.usage_service.record_usage(
                organization_id=organization_id, metric=metric, value=value, user_id=user_id, metadata=metadata
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to record usage: {e}")


class StorageTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track storage usage"""

    def __init__(self, app, db_manager=None):
        """
        Initialize storage tracking middleware

        Args:
            app: FastAPI application
            db_manager: Database manager for storage queries
        """
        super().__init__(app)
        self.db = db_manager
        self._last_check = {}
        self._check_interval = 3600  # Check every hour

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and track storage if needed"""
        response = await call_next(request)

        # Check storage periodically
        if hasattr(request.state, "user"):
            org_id = request.state.user.get("organization_id")
            if org_id:
                await self._check_storage(org_id)

        return response

    async def _check_storage(self, organization_id: str):
        """Check and record storage usage"""
        try:
            # Check if we've checked recently
            last_check = self._last_check.get(organization_id, 0)
            if time.time() - last_check < self._check_interval:
                return

            # Update last check time
            self._last_check[organization_id] = time.time()

            # Query storage usage
            if self.db:
                query = """
                SELECT 
                    SUM(LENGTH(content)) + 
                    SUM(LENGTH(title)) + 
                    SUM(LENGTH(summary)) as total_bytes
                FROM articles
                WHERE organization_id = %s
                """

                result = self.db.fetch_one(query, (organization_id,))
                if result and result["total_bytes"]:
                    # Record storage usage
                    usage_service = self.app.state.get("usage_service")
                    if usage_service:
                        asyncio.create_task(
                            self._record_storage(usage_service, organization_id, Decimal(result["total_bytes"]))
                        )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error checking storage: {e}")

    async def _record_storage(self, usage_service: UsageMeteringService, organization_id: str, bytes_used: Decimal):
        """Record storage usage"""
        try:
            usage_service.record_usage(
                organization_id=organization_id,
                metric=UsageMetric.STORAGE_BYTES,
                value=bytes_used,
                metadata={"type": "articles"},
            )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to record storage usage: {e}")
