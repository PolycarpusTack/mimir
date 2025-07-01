"""
FastAPI Dependencies for Mimir Enterprise

Provides dependency injection for authentication, database access,
and tenant context management.
"""

import logging
from typing import Annotated, Any, Dict, Optional

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.auth_manager import AuthManager
from ..auth.tenant_context import TenantContext
from ..db_manager_postgres import PostgreSQLManager

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


async def get_db() -> PostgreSQLManager:
    """Get database manager instance"""
    # In a real application, this would be injected from app state
    # or use connection pooling
    return PostgreSQLManager()


async def get_auth_manager(request: Request) -> AuthManager:
    """Get authentication manager from app state"""
    if not hasattr(request.app.state, "auth_manager"):
        raise HTTPException(status_code=500, detail="Authentication manager not initialized")
    return request.app.state.auth_manager


async def get_api_key(x_api_key: Annotated[Optional[str], Header()] = None) -> Optional[str]:
    """Extract API key from headers"""
    return x_api_key


async def get_organization_context(
    request: Request,
    x_organization_id: Annotated[Optional[str], Header()] = None,
    x_organization_slug: Annotated[Optional[str], Header()] = None,
) -> Dict[str, Optional[str]]:
    """Get organization context from headers or request state"""
    return {
        "organization_id": x_organization_id or getattr(request.state, "organization_id", None),
        "organization_slug": x_organization_slug or getattr(request.state, "organization_slug", None),
    }


async def authenticate_request(
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[str] = Depends(get_api_key),
    org_context: Dict[str, Optional[str]] = Depends(get_organization_context),
) -> Dict[str, Any]:
    """
    Authenticate request using JWT token or API key

    Returns:
        User/API key information if authenticated
    """

    # Try JWT authentication first
    if credentials and credentials.credentials:
        token_payload = auth_manager.jwt_handler.verify_token(credentials.credentials)
        if token_payload:
            # Set tenant context
            context = TenantContext(
                organization_id=token_payload.organization_id,
                user_id=token_payload.user_id,
                user_role=token_payload.role,
                user_email=token_payload.email,
                permissions=token_payload.permissions,
                session_id=token_payload.token_id,
            )
            auth_manager.set_tenant_context(context)

            return {
                "auth_type": "jwt",
                "user_id": token_payload.user_id,
                "email": token_payload.email,
                "role": token_payload.role,
                "organization_id": token_payload.organization_id,
                "permissions": token_payload.permissions,
                "token_id": token_payload.token_id,
            }

    # Try API key authentication
    if api_key:
        client_ip = request.client.host if request.client else None
        key_info = auth_manager.api_key_manager.validate_api_key(api_key, client_ip)

        if key_info:
            # Check rate limiting
            within_limit, rate_info = auth_manager.api_key_manager.check_rate_limit(key_info)
            if not within_limit:
                raise HTTPException(
                    status_code=429, detail={"error": "Rate limit exceeded", "rate_limit_info": rate_info}
                )

            # Set tenant context
            context = TenantContext(
                organization_id=key_info.organization_id,
                user_id=key_info.user_id,
                permissions=key_info.permissions,
                api_key_id=key_info.id,
            )
            auth_manager.set_tenant_context(context)

            # Record API call
            auth_manager.api_key_manager.record_api_call(
                key_info, request.url.path, 200  # Status will be updated by middleware
            )

            return {
                "auth_type": "api_key",
                "user_id": key_info.user_id,
                "organization_id": key_info.organization_id,
                "permissions": key_info.permissions,
                "api_key_id": key_info.id,
                "api_key_name": key_info.name,
                "api_key_type": key_info.key_type.value,
            }

    # No valid authentication found
    raise HTTPException(
        status_code=401, detail="Authentication required. Provide either Bearer token or X-API-Key header."
    )


async def get_current_user(auth_info: Dict[str, Any] = Depends(authenticate_request)) -> Dict[str, Any]:
    """Get current authenticated user information"""
    return auth_info


async def require_role(required_roles: list):
    """Dependency factory for requiring specific roles"""

    async def check_role(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = user.get("role")

        # API keys don't have roles, check permissions instead
        if user.get("auth_type") == "api_key":
            return user

        if not user_role or user_role not in required_roles:
            raise HTTPException(
                status_code=403,
                detail={"error": "Insufficient privileges", "required_roles": required_roles, "user_role": user_role},
            )

        return user

    return check_role


async def require_permission(required_permissions: list):
    """Dependency factory for requiring specific permissions"""

    async def check_permission(
        user: Dict[str, Any] = Depends(get_current_user), auth_manager: AuthManager = Depends(get_auth_manager)
    ) -> Dict[str, Any]:
        user_permissions = user.get("permissions", [])
        user_role = user.get("role")

        has_permission = False

        # Check role-based permissions for JWT users
        if user_role:
            for permission in required_permissions:
                if auth_manager.check_permission(user_role, permission):
                    has_permission = True
                    break

        # Check direct permissions for API keys or additional permissions
        if not has_permission:
            for permission in required_permissions:
                if permission in user_permissions or "*" in user_permissions:
                    has_permission = True
                    break

                # Check resource-level wildcards
                resource = permission.split(":")[0]
                if f"{resource}:*" in user_permissions:
                    has_permission = True
                    break

        if not has_permission:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Insufficient permissions",
                    "required_permissions": required_permissions,
                    "user_permissions": user_permissions,
                },
            )

        return user

    return check_permission


async def require_admin(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require admin role"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user


async def require_organization_access(
    user: Dict[str, Any] = Depends(get_current_user), organization_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ensure user can only access their organization's data"""
    user_org_id = user.get("organization_id")

    if not user_org_id:
        raise HTTPException(status_code=403, detail="No organization context")

    # If organization_id is specified, verify access
    if organization_id and organization_id != user_org_id:
        # Allow admins to access any organization
        if user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied to organization")

    return user


async def get_pagination_params(page: int = 1, page_size: int = 50, max_page_size: int = 1000) -> Dict[str, int]:
    """Get pagination parameters with validation"""
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")

    if page_size < 1:
        raise HTTPException(status_code=400, detail="Page size must be >= 1")

    if page_size > max_page_size:
        raise HTTPException(status_code=400, detail=f"Page size must be <= {max_page_size}")

    offset = (page - 1) * page_size

    return {"page": page, "page_size": page_size, "offset": offset, "limit": page_size}


async def get_search_params(
    q: Optional[str] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
    language: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: str = "published_date",
    sort_order: str = "desc",
) -> Dict[str, Any]:
    """Get search parameters with validation"""

    # Validate sort parameters
    valid_sort_fields = ["published_date", "scraped_at", "title", "source", "relevance"]
    if sort_by not in valid_sort_fields:
        raise HTTPException(status_code=400, detail=f"Invalid sort field. Must be one of: {valid_sort_fields}")

    if sort_order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Sort order must be 'asc' or 'desc'")

    return {
        "query": q,
        "source": source,
        "category": category,
        "language": language,
        "start_date": start_date,
        "end_date": end_date,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }


async def get_request_context(request: Request) -> Dict[str, Any]:
    """Get request context information"""
    return {
        "request_id": getattr(request.state, "request_id", None),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
    }


# Convenience dependency combinations
RequireAdmin = Depends(require_admin)
RequireEditor = Depends(require_role(["admin", "editor"]))
RequireAnalyst = Depends(require_role(["admin", "editor", "analyst"]))

# Permission-based dependencies
RequireArticlesRead = Depends(require_permission(["articles:read"]))
RequireArticlesWrite = Depends(require_permission(["articles:write", "articles:*"]))
RequireSourcesWrite = Depends(require_permission(["sources:write", "sources:*"]))
RequireKeywordsWrite = Depends(require_permission(["keywords:write", "keywords:*"]))
RequireUsersRead = Depends(require_permission(["users:read", "users:*"]))
RequireAnalyticsRead = Depends(require_permission(["analytics:read", "analytics:*"]))
