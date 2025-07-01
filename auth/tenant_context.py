"""
Tenant Context Manager for Mimir Enterprise

Manages multi-tenant context throughout the application lifecycle,
ensuring proper tenant isolation and row-level security.
"""

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TenantContext:
    """Represents the current tenant context"""

    organization_id: str
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_email: Optional[str] = None
    permissions: Optional[list] = None
    api_key_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "user_email": self.user_email,
            "permissions": self.permissions or [],
            "api_key_id": self.api_key_id,
            "session_id": self.session_id,
        }

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission"""
        if not self.permissions:
            return False
        return permission in self.permissions

    def is_admin(self) -> bool:
        """Check if user has admin role"""
        return self.user_role == "admin"

    def is_same_organization(self, org_id: str) -> bool:
        """Check if given org ID matches current context"""
        return self.organization_id == org_id


class TenantContextManager:
    """Thread-safe tenant context manager"""

    def __init__(self):
        self._local = threading.local()

    def set_context(self, context: TenantContext) -> None:
        """Set the current tenant context"""
        self._local.context = context
        logger.debug(f"Set tenant context for org: {context.organization_id}")

    def get_context(self) -> Optional[TenantContext]:
        """Get the current tenant context"""
        return getattr(self._local, "context", None)

    def clear_context(self) -> None:
        """Clear the current tenant context"""
        if hasattr(self._local, "context"):
            del self._local.context
            logger.debug("Cleared tenant context")

    def get_organization_id(self) -> Optional[str]:
        """Get current organization ID"""
        context = self.get_context()
        return context.organization_id if context else None

    def get_user_id(self) -> Optional[str]:
        """Get current user ID"""
        context = self.get_context()
        return context.user_id if context else None

    def get_user_role(self) -> Optional[str]:
        """Get current user role"""
        context = self.get_context()
        return context.user_role if context else None

    def has_permission(self, permission: str) -> bool:
        """Check if current context has permission"""
        context = self.get_context()
        return context.has_permission(permission) if context else False

    def is_admin(self) -> bool:
        """Check if current user is admin"""
        context = self.get_context()
        return context.is_admin() if context else False

    def require_context(self) -> TenantContext:
        """Get context or raise exception if not set"""
        context = self.get_context()
        if not context:
            raise RuntimeError("No tenant context set")
        return context

    @contextmanager
    def use_context(self, context: TenantContext) -> ContextManager[TenantContext]:
        """Context manager for temporary context switching"""
        original_context = self.get_context()
        try:
            self.set_context(context)
            yield context
        finally:
            if original_context:
                self.set_context(original_context)
            else:
                self.clear_context()


class DatabaseContextManager:
    """Manages database-level tenant context using PostgreSQL session variables"""

    def __init__(self, db_manager):
        self.db = db_manager

    def set_tenant_context(
        self, organization_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None
    ) -> None:
        """
        Set PostgreSQL session variables for RLS

        Args:
            organization_id: Organization UUID
            user_id: Optional user UUID
            user_role: Optional user role
        """
        try:
            # Set organization context for RLS
            self.db.execute_query("SELECT set_config('app.current_organization_id', %s, true)", (organization_id,))

            if user_id:
                self.db.execute_query("SELECT set_config('app.current_user_id', %s, true)", (user_id,))

            if user_role:
                self.db.execute_query("SELECT set_config('app.current_user_role', %s, true)", (user_role,))

            logger.debug(f"Set database context for org: {organization_id}")

        except Exception as e:
            logger.error(f"Error setting database context: {e}")
            raise

    def clear_tenant_context(self) -> None:
        """Clear PostgreSQL session variables"""
        try:
            self.db.execute_query("SELECT set_config('app.current_organization_id', '', true)")
            self.db.execute_query("SELECT set_config('app.current_user_id', '', true)")
            self.db.execute_query("SELECT set_config('app.current_user_role', '', true)")
            logger.debug("Cleared database context")
        except Exception as e:
            logger.error(f"Error clearing database context: {e}")

    def get_tenant_context(self) -> Dict[str, Optional[str]]:
        """Get current database context"""
        try:
            result = self.db.fetch_one(
                "SELECT current_setting('app.current_organization_id', true) as org_id, "
                "current_setting('app.current_user_id', true) as user_id, "
                "current_setting('app.current_user_role', true) as user_role"
            )

            return {
                "organization_id": result["org_id"] if result["org_id"] else None,
                "user_id": result["user_id"] if result["user_id"] else None,
                "user_role": result["user_role"] if result["user_role"] else None,
            }
        except Exception as e:
            logger.error(f"Error getting database context: {e}")
            return {"organization_id": None, "user_id": None, "user_role": None}

    @contextmanager
    def tenant_context(
        self, organization_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None
    ) -> ContextManager[None]:
        """Context manager for database tenant context"""
        original_context = self.get_tenant_context()
        try:
            self.set_tenant_context(organization_id, user_id, user_role)
            yield
        finally:
            # Restore original context
            if original_context["organization_id"]:
                self.set_tenant_context(
                    original_context["organization_id"], original_context["user_id"], original_context["user_role"]
                )
            else:
                self.clear_tenant_context()


class TenantMiddleware:
    """Middleware for automatic tenant context management"""

    def __init__(self, tenant_manager: TenantContextManager, db_context_manager: DatabaseContextManager):
        self.tenant_manager = tenant_manager
        self.db_context_manager = db_context_manager

    def set_request_context(self, context: TenantContext) -> None:
        """Set context for the current request"""
        # Set application context
        self.tenant_manager.set_context(context)

        # Set database context
        self.db_context_manager.set_tenant_context(context.organization_id, context.user_id, context.user_role)

    def clear_request_context(self) -> None:
        """Clear context after request"""
        self.tenant_manager.clear_context()
        self.db_context_manager.clear_tenant_context()

    @contextmanager
    def request_context(self, context: TenantContext) -> ContextManager[None]:
        """Context manager for request-scoped tenant context"""
        try:
            self.set_request_context(context)
            yield
        finally:
            self.clear_request_context()


class OrganizationManager:
    """Manages organization-related operations"""

    def __init__(self, db_manager):
        self.db = db_manager

    def get_organization_by_id(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization by ID"""
        try:
            query = """
            SELECT id, name, slug, tier, subscription_status, 
                   max_sources, max_keywords, max_users, max_api_calls_per_hour,
                   current_sources, current_keywords, current_users,
                   settings, features, created_at
            FROM organizations
            WHERE id = %s
            """
            return self.db.fetch_one(query, (org_id,))
        except Exception as e:
            logger.error(f"Error getting organization: {e}")
            return None

    def get_organization_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get organization by slug"""
        try:
            query = """
            SELECT id, name, slug, tier, subscription_status,
                   max_sources, max_keywords, max_users, max_api_calls_per_hour,
                   current_sources, current_keywords, current_users,
                   settings, features, created_at
            FROM organizations
            WHERE slug = %s
            """
            return self.db.fetch_one(query, (slug,))
        except Exception as e:
            logger.error(f"Error getting organization by slug: {e}")
            return None

    def check_organization_limits(self, org_id: str, limit_type: str) -> bool:
        """Check if organization is within limits"""
        try:
            query = "SELECT check_organization_limits(%s, %s)"
            result = self.db.fetch_one(query, (org_id, limit_type))
            return result[0] if result else False
        except Exception as e:
            logger.error(f"Error checking organization limits: {e}")
            return False

    def get_organization_usage(self, org_id: str) -> Dict[str, Any]:
        """Get current organization usage"""
        try:
            query = """
            SELECT current_sources, current_keywords, current_users,
                   max_sources, max_keywords, max_users, max_api_calls_per_hour,
                   current_storage_gb, max_storage_gb
            FROM organizations
            WHERE id = %s
            """
            result = self.db.fetch_one(query, (org_id,))

            if result:
                return {
                    "sources": {
                        "current": result["current_sources"],
                        "max": result["max_sources"],
                        "usage_percent": (result["current_sources"] / result["max_sources"]) * 100,
                    },
                    "keywords": {
                        "current": result["current_keywords"],
                        "max": result["max_keywords"],
                        "usage_percent": (result["current_keywords"] / result["max_keywords"]) * 100,
                    },
                    "users": {
                        "current": result["current_users"],
                        "max": result["max_users"],
                        "usage_percent": (result["current_users"] / result["max_users"]) * 100,
                    },
                    "storage": {
                        "current_gb": float(result["current_storage_gb"]),
                        "max_gb": float(result["max_storage_gb"]),
                        "usage_percent": (float(result["current_storage_gb"]) / float(result["max_storage_gb"])) * 100,
                    },
                    "api_calls_limit": result["max_api_calls_per_hour"],
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting organization usage: {e}")
            return {}


# Global instances
tenant_context_manager = TenantContextManager()
