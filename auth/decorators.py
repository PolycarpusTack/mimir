"""
Authentication and Authorization Decorators for Mimir Enterprise

Provides decorators for protecting API endpoints and functions with
authentication, role-based access control, and permission checks.
"""

import functools
import logging
from typing import Any, Callable, List, Optional, Union

from flask import g, jsonify, request

from .api_key_manager import APIKeyManager
from .jwt_handler import JWTHandler
from .rbac import Role, role_manager
from .tenant_context import TenantContext, tenant_context_manager

logger = logging.getLogger(__name__)


def extract_token_from_header() -> Optional[str]:
    """Extract JWT token from Authorization header"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    return None


def extract_api_key_from_header() -> Optional[str]:
    """Extract API key from X-API-Key header"""
    return request.headers.get("X-API-Key")


def require_auth(jwt_handler: JWTHandler, api_key_manager: APIKeyManager) -> Any:
    """
    Decorator that requires authentication via JWT token or API key

    Args:
        jwt_handler: JWT handler instance
        api_key_manager: API key manager instance
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            # Try JWT authentication first
            token = extract_token_from_header()
            if token:
                payload = jwt_handler.verify_token(token)
                if payload:
                    # Set tenant context for JWT authentication
                    context = TenantContext(
                        organization_id=payload.organization_id,
                        user_id=payload.user_id,
                        user_role=payload.role,
                        user_email=payload.email,
                        permissions=payload.permissions,
                        session_id=payload.token_id,
                    )
                    tenant_context_manager.set_context(context)

                    # Store auth info in Flask g for access in view functions
                    g.current_user = {
                        "id": payload.user_id,
                        "email": payload.email,
                        "role": payload.role,
                        "organization_id": payload.organization_id,
                        "permissions": payload.permissions,
                        "auth_method": "jwt",
                    }

                    return func(*args, **kwargs)

            # Try API key authentication
            api_key = extract_api_key_from_header()
            if api_key:
                client_ip = request.remote_addr
                key_info = api_key_manager.validate_api_key(api_key, client_ip)
                if key_info:
                    # Check rate limiting
                    within_limit, rate_info = api_key_manager.check_rate_limit(key_info)
                    if not within_limit:
                        return jsonify({"error": "Rate limit exceeded", "rate_limit": rate_info}), 429

                    # Set tenant context for API key authentication
                    context = TenantContext(
                        organization_id=key_info.organization_id,
                        user_id=key_info.user_id,
                        permissions=key_info.permissions,
                        api_key_id=key_info.id,
                    )
                    tenant_context_manager.set_context(context)

                    # Store auth info in Flask g
                    g.current_user = {
                        "id": key_info.user_id,
                        "organization_id": key_info.organization_id,
                        "permissions": key_info.permissions,
                        "api_key_id": key_info.id,
                        "api_key_name": key_info.name,
                        "api_key_type": key_info.key_type.value,
                        "auth_method": "api_key",
                    }

                    # Record API call
                    api_key_manager.record_api_call(
                        key_info, request.endpoint or request.path, 200  # Will be updated by response handler
                    )

                    return func(*args, **kwargs)

            # No valid authentication found
            return jsonify({"error": "Authentication required"}), 401

        return wrapper

    return decorator


def require_role(*allowed_roles: Union[str, Role]) -> Any:
    """
    Decorator that requires specific roles

    Args:
        allowed_roles: List of allowed roles (strings or Role enums)
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            if not hasattr(g, "current_user"):
                return jsonify({"error": "Authentication required"}), 401

            user_role = g.current_user.get("role")
            if not user_role:
                # For API keys, check permissions instead
                if g.current_user.get("auth_method") == "api_key":
                    return func(*args, **kwargs)  # Let permission decorator handle it
                return jsonify({"error": "No role assigned"}), 403

            # Convert string roles to Role enums for comparison
            allowed_role_values = []
            for role in allowed_roles:
                if isinstance(role, Role):
                    allowed_role_values.append(role.value)
                else:
                    allowed_role_values.append(role)

            if user_role not in allowed_role_values:
                return (
                    jsonify(
                        {
                            "error": "Insufficient privileges",
                            "required_roles": allowed_role_values,
                            "user_role": user_role,
                        }
                    ),
                    403,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_permission(*required_permissions: str) -> Any:
    """
    Decorator that requires specific permissions

    Args:
        required_permissions: List of required permissions (e.g., 'articles:read')
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            if not hasattr(g, "current_user"):
                return jsonify({"error": "Authentication required"}), 401

            user_permissions = g.current_user.get("permissions", [])
            user_role = g.current_user.get("role")

            # Check if user has any of the required permissions
            has_permission = False

            # For JWT users, check role-based permissions
            if user_role:
                try:
                    role_enum = Role(user_role)
                    for permission in required_permissions:
                        resource, action = permission.split(":", 1)
                        if role_manager.has_permission(role_enum, resource, action):
                            has_permission = True
                            break
                except ValueError:
                    pass

            # For API keys or additional permissions, check direct permissions
            if not has_permission:
                for permission in required_permissions:
                    if permission in user_permissions:
                        has_permission = True
                        break
                    # Check for wildcard permissions
                    if "*" in user_permissions:
                        has_permission = True
                        break
                    # Check for resource-level wildcard (e.g., 'articles:*')
                    resource = permission.split(":")[0]
                    if f"{resource}:*" in user_permissions:
                        has_permission = True
                        break

            if not has_permission:
                return (
                    jsonify(
                        {
                            "error": "Insufficient permissions",
                            "required_permissions": required_permissions,
                            "user_permissions": user_permissions,
                        }
                    ),
                    403,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_admin(func: Callable) -> Callable:
    """Decorator that requires admin role"""
    return require_role(Role.ADMIN)(func)


def require_organization_access(organization_field: str = "organization_id") -> Any:
    """
    Decorator that ensures user can only access their organization's data

    Args:
        organization_field: Field name containing organization ID in request
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            if not hasattr(g, "current_user"):
                return jsonify({"error": "Authentication required"}), 401

            user_org_id = g.current_user.get("organization_id")
            if not user_org_id:
                return jsonify({"error": "No organization context"}), 403

            # Get organization ID from request
            request_org_id = None

            # Check URL parameters
            if organization_field in kwargs:
                request_org_id = kwargs[organization_field]

            # Check JSON body
            elif request.is_json:
                data = request.get_json()
                if data and organization_field in data:
                    request_org_id = data[organization_field]

            # Check query parameters
            else:
                request_org_id = request.args.get(organization_field)

            # If no organization ID specified, use user's organization
            if not request_org_id:
                kwargs[organization_field] = user_org_id
                return func(*args, **kwargs)

            # Verify user can access the requested organization
            if request_org_id != user_org_id:
                # Allow admins to access any organization (optional)
                user_role = g.current_user.get("role")
                if user_role != "admin":
                    return jsonify({"error": "Access denied to organization"}), 403

            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(calls_per_hour: int = 1000) -> Any:
    """
    Decorator for API rate limiting

    Args:
        calls_per_hour: Maximum calls per hour
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            # Rate limiting is handled in require_auth for API keys
            # For JWT users, implement separate rate limiting if needed
            if hasattr(g, "current_user") and g.current_user.get("auth_method") == "jwt":
                # Implement JWT-based rate limiting here if needed
                pass

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_api_access(func: Callable) -> Callable:
    """Decorator for logging API access"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """TODO: Add docstring for wrapper"""
        # Log the API access
        user_info = getattr(g, "current_user", {})
        logger.info(
            f"API access: {request.method} {request.path} - "
            f"User: {user_info.get('id', 'unknown')} - "
            f"Org: {user_info.get('organization_id', 'unknown')} - "
            f"Auth: {user_info.get('auth_method', 'unknown')}"
        )

        return func(*args, **kwargs)

    return wrapper


def validate_organization_limits(limit_type: str) -> Any:
    """
    Decorator that checks organization limits before allowing operations

    Args:
        limit_type: Type of limit to check ('sources', 'keywords', 'users')
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """TODO: Add docstring for wrapper"""
            if not hasattr(g, "current_user"):
                return jsonify({"error": "Authentication required"}), 401

            org_id = g.current_user.get("organization_id")
            if not org_id:
                return jsonify({"error": "No organization context"}), 403

            # This would need to be implemented with the database
            # For now, just continue
            # TODO: Implement organization limit checking

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorator combinations
def api_endpoint(
    jwt_handler: JWTHandler,
    api_key_manager: APIKeyManager,
    required_permissions: Optional[List[str]] = None,
    required_roles: Optional[List[Union[str, Role]]] = None,
    log_access: bool = True,
) -> Any:
    """
    Comprehensive decorator for API endpoints

    Args:
        jwt_handler: JWT handler instance
        api_key_manager: API key manager instance
        required_permissions: Optional list of required permissions
        required_roles: Optional list of required roles
        log_access: Whether to log API access
    """

    def decorator(func: Callable) -> Callable:
        """TODO: Add docstring for decorator"""
        # Build decorator chain
        decorated_func = func

        # Add logging if requested
        if log_access:
            decorated_func = log_api_access(decorated_func)

        # Add permission checks
        if required_permissions:
            decorated_func = require_permission(*required_permissions)(decorated_func)

        # Add role checks
        if required_roles:
            decorated_func = require_role(*required_roles)(decorated_func)

        # Add authentication (must be last/outermost)
        decorated_func = require_auth(jwt_handler, api_key_manager)(decorated_func)

        return decorated_func

    return decorator
