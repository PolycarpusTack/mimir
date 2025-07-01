"""
Usage and Billing API Router for Mimir Enterprise

Provides endpoints for usage tracking, billing information, and cost management.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from ...auth.auth_manager import AuthManager
from ...services.usage_metering import BillingPeriod, PricingTier, UsageMeteringService, UsageMetric
from ..dependencies import get_auth_manager, get_current_user, get_db_manager
from ..models.common import PaginatedResponse

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


# Request/Response Models
class UsageRecordRequest(BaseModel):
    """Record usage request"""

    metric: UsageMetric
    value: Decimal = Field(..., gt=0)
    resource_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageSummaryResponse(BaseModel):
    """Usage summary response"""

    organization_id: UUID
    period: str
    start_date: datetime
    end_date: datetime
    pricing_tier: str
    base_price: str
    usage: Dict[str, str]
    overages: Dict[str, Dict[str, str]]
    total_cost: str


class UsageDetailResponse(BaseModel):
    """Usage detail response"""

    period: datetime
    value: str
    unique_users: int
    event_count: int


class UsageAlertRequest(BaseModel):
    """Create usage alert request"""

    metric: UsageMetric
    threshold_value: Decimal = Field(..., gt=0)
    threshold_type: str = Field(..., pattern="^(absolute|percentage)$")
    alert_channels: List[str] = Field(default_factory=list)


class UsageAlertResponse(BaseModel):
    """Usage alert response"""

    id: UUID
    organization_id: UUID
    metric: UsageMetric
    threshold_value: str
    threshold_type: str
    alert_channels: List[str]
    is_active: bool
    last_triggered_at: Optional[datetime]
    created_at: datetime


class BillingHistoryResponse(BaseModel):
    """Billing history response"""

    id: UUID
    period_start: datetime
    period_end: datetime
    tier: str
    base_cost: str
    overage_cost: str
    total_cost: str
    status: str
    invoice_number: Optional[str]
    paid_at: Optional[datetime]
    created_at: datetime


class PricingTierResponse(BaseModel):
    """Pricing tier information"""

    name: str
    base_price: str
    included_limits: Dict[str, str]
    overage_rates: Dict[str, str]
    features: List[str]


# Initialize service
def get_usage_service(db_manager=Depends(get_db_manager)) -> UsageMeteringService:
    """Get usage metering service"""
    return UsageMeteringService(db_manager)


# Usage Summary Endpoints
@router.get("/summary", response_model=UsageSummaryResponse)
async def get_usage_summary(
    period: BillingPeriod = Query(BillingPeriod.MONTHLY),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Get usage summary for current organization"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        summary = usage_service.get_usage_summary(
            organization_id=current_user["organization_id"], period=period, start_date=start_date, end_date=end_date
        )

        return UsageSummaryResponse(**summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/details/{metric}", response_model=List[UsageDetailResponse])
async def get_usage_details(
    metric: UsageMetric = Path(...),
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    granularity: str = Query("daily", pattern="^(hourly|daily|weekly)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Get detailed usage data for a specific metric"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        details = usage_service.get_usage_details(
            organization_id=current_user["organization_id"],
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )

        return [UsageDetailResponse(**d) for d in details]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record")
async def record_usage(
    usage_record: UsageRecordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Record usage event (internal use)"""
    # This endpoint should only be used by internal services
    # In production, use authentication for service accounts

    try:
        record = usage_service.record_usage(
            organization_id=current_user["organization_id"],
            metric=usage_record.metric,
            value=usage_record.value,
            user_id=current_user["id"],
            resource_id=usage_record.resource_id,
            metadata=usage_record.metadata,
        )

        return {"message": "Usage recorded", "id": record.timestamp.isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def generate_usage_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    format: str = Query("json", pattern="^(json|csv|pdf)$"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Generate comprehensive usage report"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:export"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        report = usage_service.generate_usage_report(
            organization_id=current_user["organization_id"], start_date=start_date, end_date=end_date, format=format
        )

        if format == "csv":
            # TODO: Return CSV file
            raise HTTPException(status_code=501, detail="CSV export not yet implemented")
        elif format == "pdf":
            # TODO: Return PDF file
            raise HTTPException(status_code=501, detail="PDF export not yet implemented")

        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Usage Alerts Endpoints
@router.post("/alerts", response_model=UsageAlertResponse)
async def create_usage_alert(
    alert: UsageAlertRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Create usage alert"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:alerts:create"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        query = """
        INSERT INTO usage_alerts (
            organization_id, metric, threshold_value, threshold_type,
            alert_channels, created_by
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING *
        """

        result = db_manager.fetch_one(
            query,
            (
                current_user["organization_id"],
                alert.metric.value,
                str(alert.threshold_value),
                alert.threshold_type,
                alert.alert_channels,
                current_user["id"],
            ),
        )

        return UsageAlertResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/alerts", response_model=List[UsageAlertResponse])
async def list_usage_alerts(
    active_only: bool = Query(True),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """List usage alerts"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:alerts:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        conditions = ["organization_id = %s"]
        params = [current_user["organization_id"]]

        if active_only:
            conditions.append("is_active = true")

        query = f"""
        SELECT * FROM usage_alerts
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        """

        results = db_manager.fetch_all(query, params)

        return [UsageAlertResponse(**r) for r in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/alerts/{alert_id}")
async def update_usage_alert(
    alert_id: UUID,
    threshold_value: Optional[Decimal] = None,
    alert_channels: Optional[List[str]] = None,
    is_active: Optional[bool] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Update usage alert"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:alerts:update"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Build update query
        update_fields = []
        params = []

        if threshold_value is not None:
            update_fields.append("threshold_value = %s")
            params.append(str(threshold_value))

        if alert_channels is not None:
            update_fields.append("alert_channels = %s")
            params.append(alert_channels)

        if is_active is not None:
            update_fields.append("is_active = %s")
            params.append(is_active)

        if not update_fields:
            raise HTTPException(status_code=400, detail="No fields to update")

        update_fields.append("updated_at = %s")
        params.append(datetime.now(timezone.utc))

        params.extend([str(alert_id), current_user["organization_id"]])

        query = f"""
        UPDATE usage_alerts
        SET {', '.join(update_fields)}
        WHERE id = %s AND organization_id = %s
        """

        result = db_manager.execute_query(query, params)

        if not result:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"message": "Alert updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/alerts/{alert_id}")
async def delete_usage_alert(
    alert_id: UUID,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Delete usage alert"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:alerts:delete"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    query = """
    DELETE FROM usage_alerts
    WHERE id = %s AND organization_id = %s
    """

    result = db_manager.execute_query(query, (str(alert_id), current_user["organization_id"]))

    if not result:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"message": "Alert deleted successfully"}


# Billing Endpoints
@router.get("/billing/history", response_model=PaginatedResponse[BillingHistoryResponse])
async def get_billing_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get billing history"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "billing:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        conditions = ["organization_id = %s"]
        params = [current_user["organization_id"]]

        if status:
            conditions.append("status = %s")
            params.append(status)

        # Count total
        count_query = f"SELECT COUNT(*) as total FROM billing_history WHERE {' AND '.join(conditions)}"
        total_result = db_manager.fetch_one(count_query, params)
        total = total_result["total"]

        # Get history
        offset = (page - 1) * page_size
        query = f"""
        SELECT * FROM billing_history
        WHERE {' AND '.join(conditions)}
        ORDER BY period_start DESC
        LIMIT %s OFFSET %s
        """

        params.extend([page_size, offset])
        results = db_manager.fetch_all(query, params)

        history = [BillingHistoryResponse(**r) for r in results]

        return PaginatedResponse(
            items=history, total=total, page=page, page_size=page_size, pages=(total + page_size - 1) // page_size
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/billing/current")
async def get_current_billing_period(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    usage_service: UsageMeteringService = Depends(get_usage_service),
    db_manager=Depends(get_db_manager),
):
    """Get current billing period usage and estimated cost"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "billing:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Get organization's billing cycle day
        org_query = "SELECT billing_cycle_day FROM organizations WHERE id = %s"
        org_result = db_manager.fetch_one(org_query, (current_user["organization_id"],))
        billing_day = org_result["billing_cycle_day"] if org_result else 1

        # Calculate current billing period
        now = datetime.now(timezone.utc)
        if now.day >= billing_day:
            period_start = now.replace(day=billing_day, hour=0, minute=0, second=0, microsecond=0)
        else:
            # Previous month
            if now.month == 1:
                period_start = now.replace(
                    year=now.year - 1, month=12, day=billing_day, hour=0, minute=0, second=0, microsecond=0
                )
            else:
                period_start = now.replace(
                    month=now.month - 1, day=billing_day, hour=0, minute=0, second=0, microsecond=0
                )

        # Get usage summary
        summary = usage_service.get_usage_summary(
            organization_id=current_user["organization_id"],
            period=BillingPeriod.MONTHLY,
            start_date=period_start,
            end_date=now,
        )

        # Add days remaining
        next_billing = period_start + timedelta(days=30)  # Approximate
        days_remaining = (next_billing - now).days

        return {**summary, "days_remaining": days_remaining, "period_progress": ((now - period_start).days / 30) * 100}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pricing", response_model=Dict[str, PricingTierResponse])
async def get_pricing_tiers(
    current_user: Dict[str, Any] = Depends(get_current_user),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Get all pricing tier information"""
    try:
        tiers = {}
        for tier_name, tier in usage_service.pricing_tiers.items():
            tiers[tier_name] = PricingTierResponse(
                name=tier.name,
                base_price=str(tier.base_price),
                included_limits={k.value: str(v) for k, v in tier.included_limits.items()},
                overage_rates={k.value: str(v) for k, v in tier.overage_rates.items()},
                features=tier.features,
            )

        return tiers

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits/check/{metric}")
async def check_usage_limit(
    metric: UsageMetric = Path(...),
    additional_usage: Decimal = Query(Decimal("1"), gt=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
    usage_service: UsageMeteringService = Depends(get_usage_service),
):
    """Check if organization can perform an action within limits"""
    try:
        allowed, limit_info = usage_service.check_usage_limits(
            organization_id=current_user["organization_id"], metric=metric, additional_usage=additional_usage
        )

        return {"allowed": allowed, "metric": metric.value, "limit_info": limit_info}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_usage_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager),
    db_manager=Depends(get_db_manager),
):
    """Get usage statistics and trends"""
    # Check permissions
    if not auth_manager.check_permission(current_user["role"], "usage:statistics:read"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Overall statistics
        stats_query = """
        SELECT 
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(DISTINCT DATE(timestamp)) as active_days,
            COUNT(*) as total_events,
            MIN(timestamp) as first_event,
            MAX(timestamp) as last_event
        FROM usage_records
        WHERE organization_id = %s 
          AND timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
        """

        overall_stats = db_manager.fetch_one(stats_query, (current_user["organization_id"], days))

        # Top metrics
        metrics_query = """
        SELECT 
            metric,
            COUNT(*) as event_count,
            SUM(value) as total_value,
            AVG(value) as avg_value,
            MAX(value) as max_value
        FROM usage_records
        WHERE organization_id = %s 
          AND timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY metric
        ORDER BY total_value DESC
        """

        top_metrics = db_manager.fetch_all(metrics_query, (current_user["organization_id"], days))

        # Daily trend
        trend_query = """
        SELECT 
            DATE(timestamp) as date,
            metric,
            SUM(value) as value
        FROM usage_records
        WHERE organization_id = %s 
          AND timestamp > CURRENT_TIMESTAMP - INTERVAL '%s days'
        GROUP BY DATE(timestamp), metric
        ORDER BY date DESC, metric
        """

        daily_trend = db_manager.fetch_all(trend_query, (current_user["organization_id"], days))

        # Group trend by date
        trend_by_date = {}
        for row in daily_trend:
            date_str = row["date"].isoformat()
            if date_str not in trend_by_date:
                trend_by_date[date_str] = {}
            trend_by_date[date_str][row["metric"]] = str(row["value"])

        return {
            "period_days": days,
            "overall": overall_stats,
            "by_metric": [
                {
                    "metric": m["metric"],
                    "event_count": m["event_count"],
                    "total_value": str(m["total_value"]),
                    "avg_value": str(m["avg_value"]),
                    "max_value": str(m["max_value"]),
                }
                for m in top_metrics
            ],
            "daily_trend": trend_by_date,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
