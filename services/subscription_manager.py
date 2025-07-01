"""
Subscription Management Service for Mimir Enterprise

Handles subscription plans, feature flags, and usage limit enforcement.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class SubscriptionStatus(str, Enum):
    """Subscription status types"""

    ACTIVE = "active"
    TRIAL = "trial"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    PENDING = "pending"


class FeatureType(str, Enum):
    """Types of features that can be controlled"""

    BOOLEAN = "boolean"  # On/off feature
    NUMERIC = "numeric"  # Numeric limit
    LIST = "list"  # List of allowed values
    CUSTOM = "custom"  # Custom logic


@dataclass
class Feature:
    """Feature definition"""

    name: str
    display_name: str
    description: str
    feature_type: FeatureType
    default_value: Any
    available_in_tiers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available_for_tier(self, tier: str) -> bool:
        """Check if feature is available for a tier"""
        return tier in self.available_in_tiers or "all" in self.available_in_tiers


@dataclass
class SubscriptionPlan:
    """Subscription plan definition"""

    id: str
    name: str
    display_name: str
    description: str
    tier: str
    price_monthly: Decimal
    price_yearly: Decimal
    trial_days: int = 0
    features: Dict[str, Any] = field(default_factory=dict)
    usage_limits: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def get_feature_value(self, feature_name: str, default: Any = None) -> Any:
        """Get feature value for this plan"""
        return self.features.get(feature_name, default)


@dataclass
class Subscription:
    """Active subscription for an organization"""

    id: str
    organization_id: str
    plan_id: str
    status: SubscriptionStatus
    started_at: datetime
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    custom_features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if subscription is active"""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]

    def is_in_trial(self) -> bool:
        """Check if subscription is in trial period"""
        return (
            self.status == SubscriptionStatus.TRIAL and self.trial_end and datetime.now(timezone.utc) < self.trial_end
        )


class FeatureFlagService:
    """Service for managing feature flags"""

    def __init__(self):
        """Initialize feature flag service"""
        self.features = self._initialize_features()

    def _initialize_features(self) -> Dict[str, Feature]:
        """Initialize available features"""
        return {
            # Search features
            "advanced_search": Feature(
                name="advanced_search",
                display_name="Advanced Search",
                description="Advanced search with filters and operators",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["starter", "professional", "enterprise"],
            ),
            "semantic_search": Feature(
                name="semantic_search",
                display_name="Semantic Search",
                description="AI-powered semantic search",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "saved_searches": Feature(
                name="saved_searches",
                display_name="Saved Searches",
                description="Save and manage search queries",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["starter", "professional", "enterprise"],
            ),
            "search_alerts": Feature(
                name="search_alerts",
                display_name="Search Alerts",
                description="Get notified when new results match saved searches",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            # AI features
            "ai_analysis": Feature(
                name="ai_analysis",
                display_name="AI Analysis",
                description="AI-powered content analysis",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["starter", "professional", "enterprise"],
            ),
            "sentiment_analysis": Feature(
                name="sentiment_analysis",
                display_name="Sentiment Analysis",
                description="Analyze sentiment of articles",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "entity_extraction": Feature(
                name="entity_extraction",
                display_name="Entity Extraction",
                description="Extract companies, people, and locations",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "custom_models": Feature(
                name="custom_models",
                display_name="Custom AI Models",
                description="Train and use custom AI models",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            # Dashboard features
            "dashboards": Feature(
                name="dashboards",
                display_name="Custom Dashboards",
                description="Create custom dashboards",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "dashboard_sharing": Feature(
                name="dashboard_sharing",
                display_name="Dashboard Sharing",
                description="Share dashboards with team members",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "real_time_updates": Feature(
                name="real_time_updates",
                display_name="Real-time Updates",
                description="Real-time dashboard updates",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            # Integration features
            "api_access": Feature(
                name="api_access",
                display_name="API Access",
                description="REST API access",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["starter", "professional", "enterprise"],
            ),
            "webhooks": Feature(
                name="webhooks",
                display_name="Webhooks",
                description="Webhook notifications",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "slack_integration": Feature(
                name="slack_integration",
                display_name="Slack Integration",
                description="Send notifications to Slack",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["starter", "professional", "enterprise"],
            ),
            "teams_integration": Feature(
                name="teams_integration",
                display_name="Teams Integration",
                description="Send notifications to Microsoft Teams",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            # Export features
            "data_export": Feature(
                name="data_export",
                display_name="Data Export",
                description="Export data in various formats",
                feature_type=FeatureType.BOOLEAN,
                default_value=True,
                available_in_tiers=["all"],
            ),
            "scheduled_reports": Feature(
                name="scheduled_reports",
                display_name="Scheduled Reports",
                description="Schedule automated reports",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            # Security features
            "sso": Feature(
                name="sso",
                display_name="Single Sign-On",
                description="SSO authentication",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            "audit_logs": Feature(
                name="audit_logs",
                display_name="Audit Logs",
                description="Detailed audit logging",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            "ip_whitelist": Feature(
                name="ip_whitelist",
                display_name="IP Whitelist",
                description="IP-based access control",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            # Support features
            "priority_support": Feature(
                name="priority_support",
                display_name="Priority Support",
                description="Priority customer support",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["professional", "enterprise"],
            ),
            "dedicated_support": Feature(
                name="dedicated_support",
                display_name="Dedicated Support",
                description="Dedicated support representative",
                feature_type=FeatureType.BOOLEAN,
                default_value=False,
                available_in_tiers=["enterprise"],
            ),
            # Numeric limits
            "max_users": Feature(
                name="max_users",
                display_name="Maximum Users",
                description="Maximum number of users",
                feature_type=FeatureType.NUMERIC,
                default_value=3,
                available_in_tiers=["all"],
            ),
            "max_saved_searches": Feature(
                name="max_saved_searches",
                display_name="Maximum Saved Searches",
                description="Maximum number of saved searches",
                feature_type=FeatureType.NUMERIC,
                default_value=5,
                available_in_tiers=["all"],
            ),
            "max_dashboards": Feature(
                name="max_dashboards",
                display_name="Maximum Dashboards",
                description="Maximum number of dashboards",
                feature_type=FeatureType.NUMERIC,
                default_value=1,
                available_in_tiers=["all"],
            ),
            "data_retention_days": Feature(
                name="data_retention_days",
                display_name="Data Retention",
                description="Days of data retention",
                feature_type=FeatureType.NUMERIC,
                default_value=30,
                available_in_tiers=["all"],
            ),
        }

    def get_feature(self, feature_name: str) -> Optional[Feature]:
        """Get feature definition"""
        return self.features.get(feature_name)

    def get_all_features(self) -> Dict[str, Feature]:
        """Get all feature definitions"""
        return self.features.copy()

    def get_features_for_tier(self, tier: str) -> Dict[str, Feature]:
        """Get features available for a specific tier"""
        return {name: feature for name, feature in self.features.items() if feature.is_available_for_tier(tier)}


class SubscriptionManager:
    """Main subscription management service"""

    def __init__(self, db_manager, usage_service=None):
        """
        Initialize subscription manager

        Args:
            db_manager: Database manager instance
            usage_service: Optional usage metering service
        """
        self.db = db_manager
        self.usage_service = usage_service
        self.feature_service = FeatureFlagService()
        self.plans = self._initialize_plans()

    def _initialize_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize subscription plans"""
        return {
            "free": SubscriptionPlan(
                id="free",
                name="free",
                display_name="Free",
                description="Get started with basic features",
                tier="free",
                price_monthly=Decimal("0"),
                price_yearly=Decimal("0"),
                trial_days=0,
                features={
                    "data_export": True,
                    "max_users": 3,
                    "max_saved_searches": 5,
                    "max_dashboards": 0,
                    "data_retention_days": 7,
                },
                usage_limits={
                    "api_calls": Decimal("10000"),
                    "storage_bytes": Decimal("1073741824"),  # 1GB
                    "articles_processed": Decimal("1000"),
                },
            ),
            "starter": SubscriptionPlan(
                id="starter",
                name="starter",
                display_name="Starter",
                description="Perfect for small teams",
                tier="starter",
                price_monthly=Decimal("99"),
                price_yearly=Decimal("990"),  # 2 months free
                trial_days=14,
                features={
                    "advanced_search": True,
                    "saved_searches": True,
                    "ai_analysis": True,
                    "api_access": True,
                    "slack_integration": True,
                    "data_export": True,
                    "max_users": 10,
                    "max_saved_searches": 25,
                    "max_dashboards": 5,
                    "data_retention_days": 90,
                },
                usage_limits={
                    "api_calls": Decimal("100000"),
                    "storage_bytes": Decimal("10737418240"),  # 10GB
                    "articles_processed": Decimal("10000"),
                    "ai_analysis": Decimal("5000"),
                },
            ),
            "professional": SubscriptionPlan(
                id="professional",
                name="professional",
                display_name="Professional",
                description="Advanced features for growing teams",
                tier="professional",
                price_monthly=Decimal("299"),
                price_yearly=Decimal("2990"),  # 2 months free
                trial_days=14,
                features={
                    "advanced_search": True,
                    "semantic_search": True,
                    "saved_searches": True,
                    "search_alerts": True,
                    "ai_analysis": True,
                    "sentiment_analysis": True,
                    "entity_extraction": True,
                    "dashboards": True,
                    "dashboard_sharing": True,
                    "api_access": True,
                    "webhooks": True,
                    "slack_integration": True,
                    "teams_integration": True,
                    "data_export": True,
                    "scheduled_reports": True,
                    "priority_support": True,
                    "max_users": 50,
                    "max_saved_searches": 100,
                    "max_dashboards": 25,
                    "data_retention_days": 365,
                },
                usage_limits={
                    "api_calls": Decimal("500000"),
                    "storage_bytes": Decimal("107374182400"),  # 100GB
                    "articles_processed": Decimal("50000"),
                    "ai_analysis": Decimal("25000"),
                },
            ),
            "enterprise": SubscriptionPlan(
                id="enterprise",
                name="enterprise",
                display_name="Enterprise",
                description="Full platform access with enterprise features",
                tier="enterprise",
                price_monthly=Decimal("999"),
                price_yearly=Decimal("9990"),  # 2 months free
                trial_days=30,
                features={
                    # All features enabled
                    **{
                        feature.name: True
                        for feature in self.feature_service.get_all_features().values()
                        if feature.feature_type == FeatureType.BOOLEAN
                    },
                    # Numeric limits
                    "max_users": 500,
                    "max_saved_searches": 1000,
                    "max_dashboards": 100,
                    "data_retention_days": 730,  # 2 years
                },
                usage_limits={
                    "api_calls": Decimal("2000000"),
                    "storage_bytes": Decimal("1099511627776"),  # 1TB
                    "articles_processed": Decimal("200000"),
                    "ai_analysis": Decimal("100000"),
                },
                metadata={"sla": "99.9%", "support_response_time": "1 hour"},
            ),
        }

    def create_subscription(
        self, organization_id: str, plan_id: str, trial: bool = True, metadata: Optional[Dict[str, Any]] = None
    ) -> Subscription:
        """
        Create a new subscription for an organization

        Args:
            organization_id: Organization ID
            plan_id: Subscription plan ID
            trial: Whether to start with trial period
            metadata: Optional metadata

        Returns:
            Created subscription
        """
        try:
            plan = self.plans.get(plan_id)
            if not plan:
                raise ValueError(f"Invalid plan ID: {plan_id}")

            # Calculate dates
            now = datetime.now(timezone.utc)
            subscription_id = str(uuid4())

            # Determine trial period
            trial_end = None
            status = SubscriptionStatus.ACTIVE

            if trial and plan.trial_days > 0:
                trial_end = now + timedelta(days=plan.trial_days)
                status = SubscriptionStatus.TRIAL

            # Calculate billing period
            period_start = now
            period_end = self._calculate_period_end(period_start, "monthly")

            # Create subscription
            subscription = Subscription(
                id=subscription_id,
                organization_id=organization_id,
                plan_id=plan_id,
                status=status,
                started_at=now,
                current_period_start=period_start,
                current_period_end=period_end,
                trial_end=trial_end,
                metadata=metadata or {},
            )

            # Store in database
            query = """
            INSERT INTO subscriptions (
                id, organization_id, plan_id, status,
                started_at, current_period_start, current_period_end,
                trial_end, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute_query(
                query,
                (
                    subscription.id,
                    subscription.organization_id,
                    subscription.plan_id,
                    subscription.status.value,
                    subscription.started_at,
                    subscription.current_period_start,
                    subscription.current_period_end,
                    subscription.trial_end,
                    json.dumps(subscription.metadata),
                ),
            )

            # Update organization tier
            self._update_organization_tier(organization_id, plan.tier)

            logger.info(f"Created subscription {subscription_id} for org {organization_id} with plan {plan_id}")

            return subscription

        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            raise

    def get_subscription(self, organization_id: str) -> Optional[Subscription]:
        """Get active subscription for an organization"""
        try:
            query = """
            SELECT * FROM subscriptions
            WHERE organization_id = %s
              AND status IN ('active', 'trial')
            ORDER BY started_at DESC
            LIMIT 1
            """

            result = self.db.fetch_one(query, (organization_id,))

            if result:
                return Subscription(
                    id=result["id"],
                    organization_id=result["organization_id"],
                    plan_id=result["plan_id"],
                    status=SubscriptionStatus(result["status"]),
                    started_at=result["started_at"],
                    current_period_start=result["current_period_start"],
                    current_period_end=result["current_period_end"],
                    trial_end=result.get("trial_end"),
                    cancelled_at=result.get("cancelled_at"),
                    custom_features=result.get("custom_features", {}),
                    metadata=result.get("metadata", {}),
                )

            return None

        except Exception as e:
            logger.error(f"Error getting subscription: {e}")
            return None

    def check_feature_access(
        self, organization_id: str, feature_name: str, user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Any]]:
        """
        Check if organization has access to a feature

        Args:
            organization_id: Organization ID
            feature_name: Feature name to check
            user_context: Optional user context for feature checks

        Returns:
            Tuple of (has_access, feature_value)
        """
        try:
            # Get subscription
            subscription = self.get_subscription(organization_id)
            if not subscription or not subscription.is_active():
                # Default to free plan
                plan = self.plans["free"]
            else:
                plan = self.plans.get(subscription.plan_id, self.plans["free"])

            # Get feature definition
            feature = self.feature_service.get_feature(feature_name)
            if not feature:
                logger.warning(f"Unknown feature: {feature_name}")
                return False, None

            # Check if feature is available for plan tier
            if not feature.is_available_for_tier(plan.tier):
                return False, None

            # Get feature value from plan
            feature_value = plan.get_feature_value(feature_name, feature.default_value)

            # Check custom features from subscription
            if subscription and feature_name in subscription.custom_features:
                feature_value = subscription.custom_features[feature_name]

            # For boolean features, return access directly
            if feature.feature_type == FeatureType.BOOLEAN:
                return bool(feature_value), feature_value

            # For numeric features, check against current usage if applicable
            elif feature.feature_type == FeatureType.NUMERIC:
                if user_context and "current_value" in user_context:
                    current = user_context["current_value"]
                    return current < feature_value, feature_value
                return True, feature_value

            # For other types, return the value
            return True, feature_value

        except Exception as e:
            logger.error(f"Error checking feature access: {e}")
            return False, None

    def get_feature_usage(self, organization_id: str, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get current usage for a numeric feature

        Args:
            organization_id: Organization ID
            feature_name: Feature name

        Returns:
            Usage information
        """
        try:
            # Map features to usage queries
            usage_queries = {
                "max_users": """
                    SELECT COUNT(*) as current_value
                    FROM users
                    WHERE organization_id = %s AND is_active = true
                """,
                "max_saved_searches": """
                    SELECT COUNT(*) as current_value
                    FROM saved_searches
                    WHERE organization_id = %s
                """,
                "max_dashboards": """
                    SELECT COUNT(*) as current_value
                    FROM dashboards
                    WHERE organization_id = %s
                """,
            }

            if feature_name not in usage_queries:
                return None

            # Get current usage
            result = self.db.fetch_one(usage_queries[feature_name], (organization_id,))
            current_value = result["current_value"] if result else 0

            # Get limit
            has_access, limit = self.check_feature_access(organization_id, feature_name)

            return {
                "feature": feature_name,
                "current_value": current_value,
                "limit": limit,
                "available": limit - current_value if limit else None,
                "percentage_used": (current_value / limit * 100) if limit else 0,
            }

        except Exception as e:
            logger.error(f"Error getting feature usage: {e}")
            return None

    def update_subscription(
        self,
        organization_id: str,
        plan_id: Optional[str] = None,
        custom_features: Optional[Dict[str, Any]] = None,
        status: Optional[SubscriptionStatus] = None,
    ) -> bool:
        """Update subscription details"""
        try:
            subscription = self.get_subscription(organization_id)
            if not subscription:
                logger.error(f"No active subscription found for org {organization_id}")
                return False

            # Build update query
            updates = []
            params = []

            if plan_id:
                updates.append("plan_id = %s")
                params.append(plan_id)

                # Update organization tier
                if plan_id in self.plans:
                    self._update_organization_tier(organization_id, self.plans[plan_id].tier)

            if custom_features is not None:
                updates.append("custom_features = %s")
                params.append(json.dumps(custom_features))

            if status:
                updates.append("status = %s")
                params.append(status.value)

                if status == SubscriptionStatus.CANCELLED:
                    updates.append("cancelled_at = %s")
                    params.append(datetime.now(timezone.utc))

            if not updates:
                return True

            updates.append("updated_at = %s")
            params.append(datetime.now(timezone.utc))

            params.append(subscription.id)

            query = f"""
            UPDATE subscriptions
            SET {', '.join(updates)}
            WHERE id = %s
            """

            self.db.execute_query(query, params)

            logger.info(f"Updated subscription {subscription.id}")
            return True

        except Exception as e:
            logger.error(f"Error updating subscription: {e}")
            return False

    def renew_subscription(self, organization_id: str) -> bool:
        """Renew subscription for next period"""
        try:
            subscription = self.get_subscription(organization_id)
            if not subscription:
                return False

            # Calculate new period
            new_period_start = subscription.current_period_end
            new_period_end = self._calculate_period_end(new_period_start, "monthly")

            # Update subscription
            query = """
            UPDATE subscriptions
            SET current_period_start = %s,
                current_period_end = %s,
                status = %s,
                updated_at = %s
            WHERE id = %s
            """

            # Convert trial to active if trial period ended
            new_status = SubscriptionStatus.ACTIVE
            if subscription.status == SubscriptionStatus.TRIAL:
                if subscription.trial_end and datetime.now(timezone.utc) >= subscription.trial_end:
                    new_status = SubscriptionStatus.ACTIVE
                else:
                    new_status = SubscriptionStatus.TRIAL

            self.db.execute_query(
                query, (new_period_start, new_period_end, new_status.value, datetime.now(timezone.utc), subscription.id)
            )

            # Record billing event if needed
            if new_status == SubscriptionStatus.ACTIVE:
                self._record_billing_event(subscription, new_period_start, new_period_end)

            logger.info(f"Renewed subscription {subscription.id} for org {organization_id}")
            return True

        except Exception as e:
            logger.error(f"Error renewing subscription: {e}")
            return False

    def cancel_subscription(self, organization_id: str, immediate: bool = False, reason: Optional[str] = None) -> bool:
        """Cancel subscription"""
        try:
            subscription = self.get_subscription(organization_id)
            if not subscription:
                return False

            if immediate:
                # Immediate cancellation
                status = SubscriptionStatus.CANCELLED
                end_date = datetime.now(timezone.utc)
            else:
                # Cancel at end of period
                status = SubscriptionStatus.ACTIVE
                end_date = subscription.current_period_end

            # Update subscription
            query = """
            UPDATE subscriptions
            SET status = %s,
                cancelled_at = %s,
                metadata = metadata || %s,
                updated_at = %s
            WHERE id = %s
            """

            cancellation_data = {
                "cancellation_reason": reason,
                "cancellation_date": datetime.now(timezone.utc).isoformat(),
                "immediate": immediate,
            }

            self.db.execute_query(
                query,
                (status.value, end_date, json.dumps(cancellation_data), datetime.now(timezone.utc), subscription.id),
            )

            # Downgrade to free tier if immediate
            if immediate:
                self._update_organization_tier(organization_id, "free")

            logger.info(f"Cancelled subscription {subscription.id} for org {organization_id}")
            return True

        except Exception as e:
            logger.error(f"Error cancelling subscription: {e}")
            return False

    def get_available_plans(self, organization_id: Optional[str] = None) -> List[SubscriptionPlan]:
        """Get available subscription plans"""
        plans = list(self.plans.values())

        # Filter based on organization if provided
        if organization_id:
            current_sub = self.get_subscription(organization_id)
            if current_sub:
                # Could filter based on current plan, upgrade paths, etc.
                pass

        return [plan for plan in plans if plan.is_active]

    def _calculate_period_end(self, start_date: datetime, billing_cycle: str) -> datetime:
        """Calculate billing period end date"""
        if billing_cycle == "monthly":
            # Add one month
            if start_date.month == 12:
                return start_date.replace(year=start_date.year + 1, month=1)
            else:
                return start_date.replace(month=start_date.month + 1)
        elif billing_cycle == "yearly":
            return start_date.replace(year=start_date.year + 1)
        else:
            # Default to monthly
            return self._calculate_period_end(start_date, "monthly")

    def _update_organization_tier(self, organization_id: str, tier: str):
        """Update organization tier"""
        try:
            query = "UPDATE organizations SET tier = %s WHERE id = %s"
            self.db.execute_query(query, (tier, organization_id))
        except Exception as e:
            logger.error(f"Error updating organization tier: {e}")

    def _record_billing_event(self, subscription: Subscription, period_start: datetime, period_end: datetime):
        """Record billing event for subscription renewal"""
        try:
            if self.usage_service:
                # Get usage for the period
                usage_summary = self.usage_service.get_usage_summary(
                    subscription.organization_id,
                    start_date=subscription.current_period_start,
                    end_date=subscription.current_period_end,
                )

                # Record in billing history
                plan = self.plans.get(subscription.plan_id)
                if plan:
                    query = """
                    INSERT INTO billing_history (
                        organization_id, period_start, period_end,
                        tier, base_cost, overage_cost, total_cost,
                        usage_summary, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    base_cost = plan.price_monthly
                    overage_cost = Decimal(usage_summary.get("total_cost", "0")) - base_cost
                    total_cost = Decimal(usage_summary.get("total_cost", str(base_cost)))

                    self.db.execute_query(
                        query,
                        (
                            subscription.organization_id,
                            period_start,
                            period_end,
                            plan.tier,
                            str(base_cost),
                            str(max(overage_cost, Decimal("0"))),
                            str(total_cost),
                            json.dumps(usage_summary),
                            "pending",
                        ),
                    )

        except Exception as e:
            logger.error(f"Error recording billing event: {e}")
