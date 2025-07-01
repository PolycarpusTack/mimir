"""
Usage Metering Service for Mimir Enterprise

Tracks API usage, storage consumption, and feature utilization for billing purposes.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


class UsageMetric(str, Enum):
    """Types of usage metrics tracked"""

    API_CALLS = "api_calls"
    STORAGE_BYTES = "storage_bytes"
    ARTICLES_PROCESSED = "articles_processed"
    AI_ANALYSIS = "ai_analysis"
    SEARCHES = "searches"
    SAVED_SEARCHES = "saved_searches"
    DASHBOARD_VIEWS = "dashboard_views"
    NOTIFICATIONS_SENT = "notifications_sent"
    WEBHOOKS = "webhooks"
    DATA_EXPORTS = "data_exports"
    USERS_ACTIVE = "users_active"
    CUSTOM_MODELS = "custom_models"


class BillingPeriod(str, Enum):
    """Billing period types"""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class UsageRecord:
    """Individual usage record"""

    organization_id: str
    metric: UsageMetric
    value: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    resource_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "organization_id": self.organization_id,
            "metric": self.metric.value,
            "value": str(self.value),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
        }


@dataclass
class PricingTier:
    """Pricing tier configuration"""

    name: str
    base_price: Decimal
    included_limits: Dict[UsageMetric, Decimal]
    overage_rates: Dict[UsageMetric, Decimal]
    features: List[str] = field(default_factory=list)

    def calculate_overage(self, metric: UsageMetric, usage: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Calculate overage amount and cost

        Returns:
            Tuple of (overage_amount, overage_cost)
        """
        included = self.included_limits.get(metric, Decimal("0"))
        if usage <= included:
            return Decimal("0"), Decimal("0")

        overage = usage - included
        rate = self.overage_rates.get(metric, Decimal("0"))
        cost = overage * rate

        return overage, cost


class UsageMeteringService:
    """Main usage metering service"""

    def __init__(self, db_manager, cache_manager=None):
        """
        Initialize usage metering service

        Args:
            db_manager: Database manager instance
            cache_manager: Optional cache manager for performance
        """
        self.db = db_manager
        self.cache = cache_manager

        # Default pricing tiers
        self.pricing_tiers = self._initialize_pricing_tiers()

        # Metric aggregation rules
        self.aggregation_rules = {
            UsageMetric.API_CALLS: "sum",
            UsageMetric.STORAGE_BYTES: "max",  # Peak storage
            UsageMetric.ARTICLES_PROCESSED: "sum",
            UsageMetric.AI_ANALYSIS: "sum",
            UsageMetric.SEARCHES: "sum",
            UsageMetric.SAVED_SEARCHES: "max",  # Active count
            UsageMetric.DASHBOARD_VIEWS: "sum",
            UsageMetric.NOTIFICATIONS_SENT: "sum",
            UsageMetric.WEBHOOKS: "sum",
            UsageMetric.DATA_EXPORTS: "sum",
            UsageMetric.USERS_ACTIVE: "max",  # Peak users
            UsageMetric.CUSTOM_MODELS: "max",  # Active models
        }

    def _initialize_pricing_tiers(self) -> Dict[str, PricingTier]:
        """Initialize default pricing tiers"""
        return {
            "free": PricingTier(
                name="Free",
                base_price=Decimal("0"),
                included_limits={
                    UsageMetric.API_CALLS: Decimal("10000"),
                    UsageMetric.STORAGE_BYTES: Decimal("1073741824"),  # 1GB
                    UsageMetric.ARTICLES_PROCESSED: Decimal("1000"),
                    UsageMetric.SEARCHES: Decimal("100"),
                    UsageMetric.USERS_ACTIVE: Decimal("3"),
                },
                overage_rates={},  # No overages on free tier
                features=["basic_search", "email_notifications"],
            ),
            "starter": PricingTier(
                name="Starter",
                base_price=Decimal("99.00"),
                included_limits={
                    UsageMetric.API_CALLS: Decimal("100000"),
                    UsageMetric.STORAGE_BYTES: Decimal("10737418240"),  # 10GB
                    UsageMetric.ARTICLES_PROCESSED: Decimal("10000"),
                    UsageMetric.AI_ANALYSIS: Decimal("5000"),
                    UsageMetric.SEARCHES: Decimal("1000"),
                    UsageMetric.SAVED_SEARCHES: Decimal("10"),
                    UsageMetric.USERS_ACTIVE: Decimal("10"),
                },
                overage_rates={
                    UsageMetric.API_CALLS: Decimal("0.001"),  # $0.001 per call
                    UsageMetric.STORAGE_BYTES: Decimal("0.00000001"),  # $0.10 per GB
                    UsageMetric.ARTICLES_PROCESSED: Decimal("0.01"),
                    UsageMetric.AI_ANALYSIS: Decimal("0.02"),
                },
                features=["advanced_search", "ai_analysis", "slack_integration", "api_access"],
            ),
            "professional": PricingTier(
                name="Professional",
                base_price=Decimal("299.00"),
                included_limits={
                    UsageMetric.API_CALLS: Decimal("500000"),
                    UsageMetric.STORAGE_BYTES: Decimal("107374182400"),  # 100GB
                    UsageMetric.ARTICLES_PROCESSED: Decimal("50000"),
                    UsageMetric.AI_ANALYSIS: Decimal("25000"),
                    UsageMetric.SEARCHES: Decimal("5000"),
                    UsageMetric.SAVED_SEARCHES: Decimal("50"),
                    UsageMetric.DASHBOARD_VIEWS: Decimal("10000"),
                    UsageMetric.USERS_ACTIVE: Decimal("50"),
                },
                overage_rates={
                    UsageMetric.API_CALLS: Decimal("0.0008"),
                    UsageMetric.STORAGE_BYTES: Decimal("0.000000008"),
                    UsageMetric.ARTICLES_PROCESSED: Decimal("0.008"),
                    UsageMetric.AI_ANALYSIS: Decimal("0.015"),
                },
                features=[
                    "all_starter_features",
                    "custom_dashboards",
                    "webhooks",
                    "team_collaboration",
                    "priority_support",
                    "custom_models",
                ],
            ),
            "enterprise": PricingTier(
                name="Enterprise",
                base_price=Decimal("999.00"),
                included_limits={
                    UsageMetric.API_CALLS: Decimal("2000000"),
                    UsageMetric.STORAGE_BYTES: Decimal("1099511627776"),  # 1TB
                    UsageMetric.ARTICLES_PROCESSED: Decimal("200000"),
                    UsageMetric.AI_ANALYSIS: Decimal("100000"),
                    UsageMetric.SEARCHES: Decimal("20000"),
                    UsageMetric.SAVED_SEARCHES: Decimal("200"),
                    UsageMetric.DASHBOARD_VIEWS: Decimal("50000"),
                    UsageMetric.USERS_ACTIVE: Decimal("200"),
                },
                overage_rates={
                    UsageMetric.API_CALLS: Decimal("0.0005"),
                    UsageMetric.STORAGE_BYTES: Decimal("0.000000005"),
                    UsageMetric.ARTICLES_PROCESSED: Decimal("0.005"),
                    UsageMetric.AI_ANALYSIS: Decimal("0.01"),
                },
                features=[
                    "all_professional_features",
                    "sso",
                    "audit_logs",
                    "sla",
                    "dedicated_support",
                    "custom_integrations",
                ],
            ),
        }

    def record_usage(
        self,
        organization_id: str,
        metric: UsageMetric,
        value: Decimal,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Record a usage event

        Args:
            organization_id: Organization ID
            metric: Type of metric
            value: Metric value
            user_id: Optional user ID
            resource_id: Optional resource ID
            metadata: Optional metadata

        Returns:
            Created usage record
        """
        try:
            record = UsageRecord(
                organization_id=organization_id,
                metric=metric,
                value=value,
                timestamp=datetime.now(timezone.utc),
                user_id=user_id,
                resource_id=resource_id,
                metadata=metadata or {},
            )

            # Store in database
            query = """
            INSERT INTO usage_records (
                organization_id, metric, value, timestamp,
                user_id, resource_id, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute_query(
                query,
                (
                    record.organization_id,
                    record.metric.value,
                    str(record.value),
                    record.timestamp,
                    record.user_id,
                    record.resource_id,
                    json.dumps(record.metadata),
                ),
            )

            # Update cache if available
            if self.cache:
                self._update_usage_cache(record)

            logger.debug(f"Recorded usage: {record.metric.value} = {record.value} for org {record.organization_id}")

            return record

        except Exception as e:
            logger.error(f"Error recording usage: {e}")
            raise

    def get_usage_summary(
        self,
        organization_id: str,
        period: BillingPeriod = BillingPeriod.MONTHLY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage summary for an organization

        Args:
            organization_id: Organization ID
            period: Billing period
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Usage summary with metrics and costs
        """
        try:
            # Default to current period if dates not specified
            if not end_date:
                end_date = datetime.now(timezone.utc)

            if not start_date:
                start_date = self._get_period_start(end_date, period)

            # Get aggregated usage
            usage_data = self._get_aggregated_usage(organization_id, start_date, end_date)

            # Get organization's pricing tier
            org_tier = self._get_organization_tier(organization_id)
            pricing_tier = self.pricing_tiers.get(org_tier, self.pricing_tiers["free"])

            # Calculate costs
            total_cost = pricing_tier.base_price
            overage_details = {}

            for metric, usage in usage_data.items():
                overage, cost = pricing_tier.calculate_overage(metric, usage)
                if overage > 0:
                    overage_details[metric.value] = {
                        "usage": str(usage),
                        "included": str(pricing_tier.included_limits.get(metric, 0)),
                        "overage": str(overage),
                        "cost": str(cost),
                    }
                    total_cost += cost

            return {
                "organization_id": organization_id,
                "period": period.value,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "pricing_tier": org_tier,
                "base_price": str(pricing_tier.base_price),
                "usage": {k.value: str(v) for k, v in usage_data.items()},
                "overages": overage_details,
                "total_cost": str(total_cost),
            }

        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            raise

    def get_usage_details(
        self,
        organization_id: str,
        metric: UsageMetric,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "daily",
    ) -> List[Dict[str, Any]]:
        """
        Get detailed usage data for a specific metric

        Args:
            organization_id: Organization ID
            metric: Usage metric
            start_date: Start date
            end_date: End date
            granularity: Data granularity (hourly, daily, weekly)

        Returns:
            Time series usage data
        """
        try:
            # Determine time truncation based on granularity
            if granularity == "hourly":
                trunc_func = "DATE_TRUNC('hour', timestamp)"
            elif granularity == "weekly":
                trunc_func = "DATE_TRUNC('week', timestamp)"
            else:
                trunc_func = "DATE_TRUNC('day', timestamp)"

            # Get aggregation rule
            agg_rule = self.aggregation_rules.get(metric, "sum")
            agg_func = "MAX(value)" if agg_rule == "max" else "SUM(value)"

            query = f"""
            SELECT 
                {trunc_func} as period,
                {agg_func} as value,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(*) as event_count
            FROM usage_records
            WHERE organization_id = %s
              AND metric = %s
              AND timestamp >= %s
              AND timestamp < %s
            GROUP BY period
            ORDER BY period
            """

            results = self.db.fetch_all(query, (organization_id, metric.value, start_date, end_date))

            return [
                {
                    "period": r["period"].isoformat(),
                    "value": str(r["value"]),
                    "unique_users": r["unique_users"],
                    "event_count": r["event_count"],
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error getting usage details: {e}")
            raise

    def check_usage_limits(
        self, organization_id: str, metric: UsageMetric, additional_usage: Decimal = Decimal("1")
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if organization can perform an action within limits

        Args:
            organization_id: Organization ID
            metric: Usage metric to check
            additional_usage: Amount to be added

        Returns:
            Tuple of (allowed, limit_info)
        """
        try:
            # Get current usage
            current_usage = self._get_current_period_usage(organization_id, metric)

            # Get organization's limits
            org_tier = self._get_organization_tier(organization_id)
            pricing_tier = self.pricing_tiers.get(org_tier, self.pricing_tiers["free"])

            # Check hard limits (no overages allowed on free tier)
            if org_tier == "free" and metric in pricing_tier.included_limits:
                limit = pricing_tier.included_limits[metric]
                projected_usage = current_usage + additional_usage

                if projected_usage > limit:
                    return False, {
                        "current_usage": str(current_usage),
                        "limit": str(limit),
                        "requested": str(additional_usage),
                        "message": f"Usage limit exceeded for {metric.value}",
                    }

            # For paid tiers, always allow but track overages
            return True, {
                "current_usage": str(current_usage),
                "limit": str(pricing_tier.included_limits.get(metric, "unlimited")),
                "tier": org_tier,
            }

        except Exception as e:
            logger.error(f"Error checking usage limits: {e}")
            # Allow on error to prevent service disruption
            return True, None

    def _get_aggregated_usage(
        self, organization_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[UsageMetric, Decimal]:
        """Get aggregated usage for all metrics"""
        try:
            usage_data = {}

            for metric in UsageMetric:
                agg_rule = self.aggregation_rules.get(metric, "sum")
                agg_func = "MAX(value)" if agg_rule == "max" else "SUM(value)"

                query = f"""
                SELECT {agg_func} as total
                FROM usage_records
                WHERE organization_id = %s
                  AND metric = %s
                  AND timestamp >= %s
                  AND timestamp < %s
                """

                result = self.db.fetch_one(query, (organization_id, metric.value, start_date, end_date))

                usage_data[metric] = Decimal(result["total"] or "0")

            return usage_data

        except Exception as e:
            logger.error(f"Error getting aggregated usage: {e}")
            return {}

    def _get_current_period_usage(self, organization_id: str, metric: UsageMetric) -> Decimal:
        """Get current billing period usage for a metric"""
        try:
            # Get current month start
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            agg_rule = self.aggregation_rules.get(metric, "sum")
            agg_func = "MAX(value)" if agg_rule == "max" else "SUM(value)"

            query = f"""
            SELECT {agg_func} as total
            FROM usage_records
            WHERE organization_id = %s
              AND metric = %s
              AND timestamp >= %s
            """

            result = self.db.fetch_one(query, (organization_id, metric.value, period_start))

            return Decimal(result["total"] or "0")

        except Exception as e:
            logger.error(f"Error getting current period usage: {e}")
            return Decimal("0")

    def _get_organization_tier(self, organization_id: str) -> str:
        """Get organization's pricing tier"""
        try:
            query = "SELECT tier FROM organizations WHERE id = %s"
            result = self.db.fetch_one(query, (organization_id,))
            return result["tier"] if result else "free"
        except Exception as e:
            logger.error(f"Error getting organization tier: {e}")
            return "free"

    def _get_period_start(self, end_date: datetime, period: BillingPeriod) -> datetime:
        """Calculate period start date"""
        if period == BillingPeriod.HOURLY:
            return end_date - timedelta(hours=1)
        elif period == BillingPeriod.DAILY:
            return end_date - timedelta(days=1)
        elif period == BillingPeriod.WEEKLY:
            return end_date - timedelta(weeks=1)
        elif period == BillingPeriod.YEARLY:
            return end_date - timedelta(days=365)
        else:  # Monthly
            # Go back to same day previous month
            if end_date.month == 1:
                return end_date.replace(year=end_date.year - 1, month=12)
            else:
                return end_date.replace(month=end_date.month - 1)

    def _update_usage_cache(self, record: UsageRecord):
        """Update usage cache for real-time tracking"""
        if not self.cache:
            return

        try:
            # Cache key for current period usage
            cache_key = f"usage:{record.organization_id}:{record.metric.value}:current"

            # Increment cached value
            current = self.cache.get(cache_key) or 0
            self.cache.set(cache_key, current + float(record.value), expire=3600)

        except Exception as e:
            logger.error(f"Error updating usage cache: {e}")

    def generate_usage_report(
        self, organization_id: str, start_date: datetime, end_date: datetime, format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive usage report

        Args:
            organization_id: Organization ID
            start_date: Report start date
            end_date: Report end date
            format: Report format (json, csv, pdf)

        Returns:
            Usage report data
        """
        try:
            # Get usage summary
            summary = self.get_usage_summary(organization_id, BillingPeriod.MONTHLY, start_date, end_date)

            # Get detailed usage for each metric
            details = {}
            for metric in UsageMetric:
                metric_details = self.get_usage_details(organization_id, metric, start_date, end_date, "daily")
                if metric_details:
                    details[metric.value] = metric_details

            # Get top users
            top_users_query = """
            SELECT 
                user_id,
                COUNT(*) as event_count,
                SUM(CASE WHEN metric = %s THEN value ELSE 0 END) as api_calls
            FROM usage_records
            WHERE organization_id = %s
              AND timestamp >= %s
              AND timestamp < %s
              AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY api_calls DESC
            LIMIT 10
            """

            top_users = self.db.fetch_all(
                top_users_query, (UsageMetric.API_CALLS.value, organization_id, start_date, end_date)
            )

            report = {
                "organization_id": organization_id,
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "summary": summary,
                "details": details,
                "top_users": [
                    {"user_id": u["user_id"], "event_count": u["event_count"], "api_calls": str(u["api_calls"])}
                    for u in top_users
                ],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Convert to requested format
            if format == "csv":
                # TODO: Implement CSV generation
                pass
            elif format == "pdf":
                # TODO: Implement PDF generation
                pass

            return report

        except Exception as e:
            logger.error(f"Error generating usage report: {e}")
            raise
