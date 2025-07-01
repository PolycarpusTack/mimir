"""
Feature Flag Decorators for Mimir Enterprise

Provides decorators to check feature access in API endpoints.
"""

from functools import wraps
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException

from ...services.subscription_manager import SubscriptionManager


def require_feature(feature_name: str, error_message: Optional[str] = None, check_usage: bool = False):
    """
    Decorator to require a specific feature

    Args:
        feature_name: Name of the feature to check
        error_message: Custom error message
        check_usage: Whether to check usage limits for numeric features
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get subscription service from dependencies
            subscription_service = None
            for arg in args:
                if isinstance(arg, SubscriptionManager):
                    subscription_service = arg
                    break

            if not subscription_service:
                # Try to get from kwargs
                subscription_service = kwargs.get("subscription_service")

            if not subscription_service:
                raise HTTPException(status_code=500, detail="Subscription service not available")

            # Check feature access
            has_access, value = subscription_service.check_feature_access(
                organization_id=current_user["organization_id"], feature_name=feature_name
            )

            if not has_access:
                message = error_message or f"Feature '{feature_name}' not available in your plan"
                raise HTTPException(status_code=403, detail=message)

            # Check usage limits for numeric features if requested
            if check_usage:
                feature = subscription_service.feature_service.get_feature(feature_name)
                if feature and feature.feature_type.value == "numeric":
                    usage = subscription_service.get_feature_usage(
                        organization_id=current_user["organization_id"], feature_name=feature_name
                    )

                    if usage and usage["available"] is not None and usage["available"] <= 0:
                        raise HTTPException(status_code=403, detail=f"Usage limit exceeded for {feature_name}")

            # Add feature info to request context
            kwargs["feature_access"] = {"feature": feature_name, "has_access": has_access, "value": value}

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_feature(*feature_names: str, error_message: Optional[str] = None):
    """
    Decorator to require any of the specified features

    Args:
        feature_names: Names of features (requires at least one)
        error_message: Custom error message
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get subscription service
            subscription_service = None
            for arg in args:
                if isinstance(arg, SubscriptionManager):
                    subscription_service = arg
                    break

            if not subscription_service:
                subscription_service = kwargs.get("subscription_service")

            if not subscription_service:
                raise HTTPException(status_code=500, detail="Subscription service not available")

            # Check if any feature is available
            available_features = []
            for feature_name in feature_names:
                has_access, value = subscription_service.check_feature_access(
                    organization_id=current_user["organization_id"], feature_name=feature_name
                )

                if has_access:
                    available_features.append({"feature": feature_name, "value": value})

            if not available_features:
                message = error_message or f"None of the required features are available: {', '.join(feature_names)}"
                raise HTTPException(status_code=403, detail=message)

            # Add feature info to request context
            kwargs["available_features"] = available_features

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_tier(min_tier: str, error_message: Optional[str] = None):
    """
    Decorator to require a minimum subscription tier

    Args:
        min_tier: Minimum required tier (free, starter, professional, enterprise)
        error_message: Custom error message
    """
    tier_hierarchy = ["free", "starter", "professional", "enterprise"]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get subscription service
            subscription_service = None
            for arg in args:
                if isinstance(arg, SubscriptionManager):
                    subscription_service = arg
                    break

            if not subscription_service:
                subscription_service = kwargs.get("subscription_service")

            if not subscription_service:
                raise HTTPException(status_code=500, detail="Subscription service not available")

            # Get current subscription
            subscription = subscription_service.get_subscription(current_user["organization_id"])

            if not subscription:
                current_tier = "free"
            else:
                plan = subscription_service.plans.get(subscription.plan_id)
                current_tier = plan.tier if plan else "free"

            # Check tier hierarchy
            if current_tier not in tier_hierarchy or min_tier not in tier_hierarchy:
                raise HTTPException(status_code=500, detail="Invalid tier configuration")

            current_index = tier_hierarchy.index(current_tier)
            required_index = tier_hierarchy.index(min_tier)

            if current_index < required_index:
                message = error_message or f"This feature requires {min_tier} tier or higher"
                raise HTTPException(status_code=403, detail=message)

            # Add tier info to request context
            kwargs["tier_info"] = {"current_tier": current_tier, "required_tier": min_tier}

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def usage_limit_check(metric: str, additional_usage: int = 1):
    """
    Decorator to check usage limits before allowing action

    Args:
        metric: Usage metric to check
        additional_usage: Amount to be added
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")

            # Get usage service
            usage_service = None
            for arg in args:
                if hasattr(arg, "check_usage_limits"):
                    usage_service = arg
                    break

            if usage_service:
                # Check usage limits
                allowed, limit_info = usage_service.check_usage_limits(
                    organization_id=current_user["organization_id"], metric=metric, additional_usage=additional_usage
                )

                if not allowed:
                    message = limit_info.get("message", f"Usage limit exceeded for {metric}")
                    raise HTTPException(status_code=429, detail=message)

                # Add limit info to request context
                kwargs["usage_limit_info"] = limit_info

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators for common features
def require_api_access(func: Callable) -> Callable:
    """Require API access feature"""
    return require_feature("api_access", "API access not available in your plan")(func)


def require_advanced_search(func: Callable) -> Callable:
    """Require advanced search feature"""
    return require_feature("advanced_search", "Advanced search not available in your plan")(func)


def require_dashboards(func: Callable) -> Callable:
    """Require dashboards feature with usage check"""
    return require_feature("dashboards", "Custom dashboards not available in your plan", check_usage=True)(func)


def require_saved_searches(func: Callable) -> Callable:
    """Require saved searches feature with usage check"""
    return require_feature("saved_searches", "Saved searches not available in your plan", check_usage=True)(func)


def require_ai_analysis(func: Callable) -> Callable:
    """Require AI analysis feature"""
    return require_feature("ai_analysis", "AI analysis not available in your plan")(func)


def require_webhooks(func: Callable) -> Callable:
    """Require webhooks feature"""
    return require_feature("webhooks", "Webhooks not available in your plan")(func)


def require_professional_tier(func: Callable) -> Callable:
    """Require professional tier or higher"""
    return require_tier("professional", "This feature requires Professional tier or higher")(func)


def require_enterprise_tier(func: Callable) -> Callable:
    """Require enterprise tier"""
    return require_tier("enterprise", "This feature requires Enterprise tier")(func)


# Usage limit decorators
def check_user_limit(func: Callable) -> Callable:
    """Check user creation limit"""
    return usage_limit_check("max_users", 1)(func)


def check_saved_search_limit(func: Callable) -> Callable:
    """Check saved search creation limit"""
    return usage_limit_check("max_saved_searches", 1)(func)


def check_dashboard_limit(func: Callable) -> Callable:
    """Check dashboard creation limit"""
    return usage_limit_check("max_dashboards", 1)(func)
