"""
Subscription Management API Router for Mimir Enterprise

Provides endpoints for managing subscriptions, feature flags, and billing.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ...auth.auth_manager import AuthManager
from ...services.subscription_manager import (
    Feature,
    FeatureType,
    Subscription,
    SubscriptionManager,
    SubscriptionPlan,
    SubscriptionStatus,
)
from ..dependencies import get_auth_manager, get_current_user, get_db_manager
from ..models.common import PaginatedResponse

router = APIRouter(prefix="/api/v1/subscriptions", tags=["subscriptions"])


# Request/Response Models
class SubscriptionPlanResponse(BaseModel):
    """Subscription plan response"""

    id: str
    name: str
    display_name: str
    description: str
    tier: str
    price_monthly: str
    price_yearly: str
    trial_days: int
    features: Dict[str, Any]
    usage_limits: Dict[str, str]
    metadata: Dict[str, Any]
    is_active: bool


class SubscriptionResponse(BaseModel):
    """Subscription response"""

    id: UUID
    organization_id: UUID
    plan_id: str
    status: SubscriptionStatus
    started_at: datetime
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime]
    cancelled_at: Optional[datetime]
    custom_features: Dict[str, Any]
    metadata: Dict[str, Any]


class CreateSubscriptionRequest(BaseModel):
    """Create subscription request"""

    plan_id: str = Field(..., min_length=1)
    trial: bool = True
    metadata: Optional[Dict[str, Any]] = None


class UpdateSubscriptionRequest(BaseModel):
    """Update subscription request"""

    plan_id: Optional[str] = None
    custom_features: Optional[Dict[str, Any]] = None
    status: Optional[SubscriptionStatus] = None


class FeatureResponse(BaseModel):
    """Feature response"""

    name: str
    display_name: str
    description: str
    feature_type: FeatureType
    default_value: Any
    available_in_tiers: List[str]
    metadata: Dict[str, Any]


class FeatureAccessResponse(BaseModel):
    """Feature access response"""

    feature: str
    has_access: bool
    value: Any
    source: str  # plan, override, custom, etc.


class FeatureUsageResponse(BaseModel):
    """Feature usage response"""

    feature: str
    current_value: int
    limit: Optional[int]
    available: Optional[int]
    percentage_used: float


class FeatureOverrideRequest(BaseModel):
    """Feature override request"""

    feature_name: str = Field(..., min_length=1)
    value: Any
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None


class SubscriptionHistoryResponse(BaseModel):
    """Subscription history response"""

    id: UUID
    action: str
    old_plan_id: Optional[str]
    new_plan_id: Optional[str]
    old_status: Optional[str]
    new_status: Optional[str]
    metadata: Dict[str, Any]
    performed_by: Optional[UUID]
    created_at: datetime


# Initialize service
def get_subscription_service(db_manager=Depends(get_db_manager)) -> SubscriptionManager:
    """Get subscription manager service"""
    return SubscriptionManager(db_manager)


# Subscription Management Endpoints
@router.get("/plans", response_model=List[SubscriptionPlanResponse])
async def get_subscription_plans(
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Get available subscription plans"""
    try:
        plans = subscription_service.get_available_plans(organization_id=current_user["organization_id"])

        return [
            SubscriptionPlanResponse(
                id=plan.id,
                name=plan.name,
                display_name=plan.display_name,
                description=plan.description,
                tier=plan.tier,
                price_monthly=str(plan.price_monthly),
                price_yearly=str(plan.price_yearly),
                trial_days=plan.trial_days,
                features=plan.features,
                usage_limits={k: str(v) for k, v in plan.usage_limits.items()},
                metadata=plan.metadata,
                is_active=plan.is_active,
            )
            for plan in plans
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current", response_model=Optional[SubscriptionResponse])
async def get_current_subscription(
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Get current subscription for organization"""
    try:
        subscription = subscription_service.get_subscription(current_user["organization_id"])

        if subscription:
            return SubscriptionResponse(
                id=UUID(subscription.id),
                organization_id=UUID(subscription.organization_id),
                plan_id=subscription.plan_id,
                status=subscription.status,
                started_at=subscription.started_at,
                current_period_start=subscription.current_period_start,
                current_period_end=subscription.current_period_end,
                trial_end=subscription.trial_end,
                cancelled_at=subscription.cancelled_at,
                custom_features=subscription.custom_features,
                metadata=subscription.metadata,
            )

        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create", response_model=SubscriptionResponse)
async def create_subscription(
    request: CreateSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Create new subscription"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:create"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Check if organization already has an active subscription
        existing = subscription_service.get_subscription(current_user["organization_id"])
        if existing and existing.is_active():
            raise HTTPException(status_code=400, detail="Organization already has an active subscription")

        subscription = subscription_service.create_subscription(
            organization_id=current_user["organization_id"],
            plan_id=request.plan_id,
            trial=request.trial,
            metadata=request.metadata,
        )

        return SubscriptionResponse(
            id=UUID(subscription.id),
            organization_id=UUID(subscription.organization_id),
            plan_id=subscription.plan_id,
            status=subscription.status,
            started_at=subscription.started_at,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            trial_end=subscription.trial_end,
            cancelled_at=subscription.cancelled_at,
            custom_features=subscription.custom_features,
            metadata=subscription.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/update")
async def update_subscription(
    request: UpdateSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Update current subscription"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:update"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        success = subscription_service.update_subscription(
            organization_id=current_user["organization_id"],
            plan_id=request.plan_id,
            custom_features=request.custom_features,
            status=request.status,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to update subscription")

        return {"message": "Subscription updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/cancel")
async def cancel_subscription(
    immediate: bool = Query(False),
    reason: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Cancel subscription"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:cancel"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        success = subscription_service.cancel_subscription(
            organization_id=current_user["organization_id"], immediate=immediate, reason=reason
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel subscription")

        return {"message": "Subscription cancelled successfully", "immediate": immediate}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/renew")
async def renew_subscription(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Renew subscription (admin use)"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:renew"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        success = subscription_service.renew_subscription(current_user["organization_id"])

        if not success:
            raise HTTPException(status_code=400, detail="Failed to renew subscription")

        return {"message": "Subscription renewed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Feature Flag Endpoints
@router.get("/features", response_model=List[FeatureResponse])
async def get_features(
    tier: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Get available features"""
    try:
        if tier:
            features = subscription_service.feature_service.get_features_for_tier(tier)
        else:
            features = subscription_service.feature_service.get_all_features()

        return [
            FeatureResponse(
                name=feature.name,
                display_name=feature.display_name,
                description=feature.description,
                feature_type=feature.feature_type,
                default_value=feature.default_value,
                available_in_tiers=feature.available_in_tiers,
                metadata=feature.metadata,
            )
            for feature in features.values()
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_name}/access", response_model=FeatureAccessResponse)
async def check_feature_access(
    feature_name: str = Path(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Check access to a specific feature"""
    try:
        has_access, value = subscription_service.check_feature_access(
            organization_id=current_user["organization_id"], feature_name=feature_name
        )

        # Determine source
        subscription = subscription_service.get_subscription(current_user["organization_id"])
        source = "plan"
        if subscription and feature_name in subscription.custom_features:
            source = "custom"

        return FeatureAccessResponse(feature=feature_name, has_access=has_access, value=value, source=source)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{feature_name}/usage", response_model=Optional[FeatureUsageResponse])
async def get_feature_usage(
    feature_name: str = Path(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Get usage information for a numeric feature"""
    try:
        usage = subscription_service.get_feature_usage(
            organization_id=current_user["organization_id"], feature_name=feature_name
        )

        if usage:
            return FeatureUsageResponse(**usage)

        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/override")
async def create_feature_override(
    request: FeatureOverrideRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Create feature override (admin only)"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:feature_override"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        query = """
        INSERT INTO feature_overrides (
            organization_id, feature_name, value, reason, expires_at, created_by
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (organization_id, feature_name)
        DO UPDATE SET
            value = EXCLUDED.value,
            reason = EXCLUDED.reason,
            expires_at = EXCLUDED.expires_at,
            updated_at = CURRENT_TIMESTAMP
        """

        db_manager.execute_query(
            query,
            (
                current_user["organization_id"],
                request.feature_name,
                request.value,
                request.reason,
                request.expires_at,
                current_user["id"],
            ),
        )

        return {"message": "Feature override created successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/features/overrides")
async def list_feature_overrides(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """List feature overrides for organization"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:feature_override:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        query = """
        SELECT fo.*, fd.display_name, fd.description
        FROM feature_overrides fo
        JOIN feature_definitions fd ON fo.feature_name = fd.name
        WHERE fo.organization_id = %s
          AND (fo.expires_at IS NULL OR fo.expires_at > CURRENT_TIMESTAMP)
        ORDER BY fo.created_at DESC
        """

        results = db_manager.fetch_all(query, (current_user["organization_id"],))

        return [
            {
                "id": r["id"],
                "feature_name": r["feature_name"],
                "feature_display_name": r["display_name"],
                "feature_description": r["description"],
                "value": r["value"],
                "reason": r["reason"],
                "expires_at": r["expires_at"],
                "created_at": r["created_at"],
            }
            for r in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/features/overrides/{feature_name}")
async def delete_feature_override(
    feature_name: str = Path(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Delete feature override"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:feature_override:delete"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        query = """
        DELETE FROM feature_overrides
        WHERE organization_id = %s AND feature_name = %s
        """

        result = db_manager.execute_query(query, (current_user["organization_id"], feature_name))

        if not result:
            raise HTTPException(status_code=404, detail="Feature override not found")

        return {"message": "Feature override deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Subscription History
@router.get("/history", response_model=PaginatedResponse[SubscriptionHistoryResponse])
async def get_subscription_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get subscription history"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:history:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Count total
        count_query = """
        SELECT COUNT(*) as total FROM subscription_history
        WHERE organization_id = %s
        """
        total_result = db_manager.fetch_one(count_query, (current_user["organization_id"],))
        total = total_result["total"]

        # Get history
        offset = (page - 1) * page_size
        query = """
        SELECT * FROM subscription_history
        WHERE organization_id = %s
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """

        results = db_manager.fetch_all(query, (current_user["organization_id"], page_size, offset))

        history = [
            SubscriptionHistoryResponse(
                id=r["id"],
                action=r["action"],
                old_plan_id=r["old_plan_id"],
                new_plan_id=r["new_plan_id"],
                old_status=r["old_status"],
                new_status=r["new_status"],
                metadata=r["metadata"],
                performed_by=r["performed_by"],
                created_at=r["created_at"],
            )
            for r in results
        ]

        return PaginatedResponse(
            items=history, total=total, page=page, page_size=page_size, pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Feature Usage Summary
@router.get("/features/usage/summary")
async def get_feature_usage_summary(
    current_user: Dict[str, Any] = Depends(get_current_user),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
):
    """Get usage summary for all numeric features"""
    try:
        # Get numeric features
        features = subscription_service.feature_service.get_all_features()
        numeric_features = [name for name, feature in features.items() if feature.feature_type == FeatureType.NUMERIC]

        usage_summary = {}
        for feature_name in numeric_features:
            usage = subscription_service.get_feature_usage(
                organization_id=current_user["organization_id"], feature_name=feature_name
            )
            if usage:
                usage_summary[feature_name] = usage

        return usage_summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Subscription Analytics
@router.get("/analytics")
async def get_subscription_analytics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    subscription_service: SubscriptionManager = Depends(get_subscription_service),
    db_manager=Depends(get_db_manager),
):
    """Get subscription analytics"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "subscriptions:analytics:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        subscription = subscription_service.get_subscription(current_user["organization_id"])

        if not subscription:
            return {"message": "No active subscription"}

        # Feature usage stats
        feature_usage = {}
        numeric_features = ["max_users", "max_saved_searches", "max_dashboards"]

        for feature in numeric_features:
            usage = subscription_service.get_feature_usage(
                organization_id=current_user["organization_id"], feature_name=feature
            )
            if usage:
                feature_usage[feature] = usage

        # Subscription timeline
        timeline_query = """
        SELECT action, old_plan_id, new_plan_id, created_at
        FROM subscription_history
        WHERE organization_id = %s
        ORDER BY created_at ASC
        """

        timeline = db_manager.fetch_all(timeline_query, (current_user["organization_id"],))

        return {
            "subscription": {
                "plan_id": subscription.plan_id,
                "status": subscription.status.value,
                "started_at": subscription.started_at.isoformat(),
                "current_period_start": subscription.current_period_start.isoformat(),
                "current_period_end": subscription.current_period_end.isoformat(),
                "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None,
            },
            "feature_usage": feature_usage,
            "timeline": [
                {
                    "action": t["action"],
                    "old_plan": t["old_plan_id"],
                    "new_plan": t["new_plan_id"],
                    "date": t["created_at"].isoformat(),
                }
                for t in timeline
            ],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
