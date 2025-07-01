"""
Role-Based Access Control (RBAC) System for Mimir Enterprise

Implements comprehensive permission management with roles, resources, and actions.
Supports hierarchical roles and dynamic permission evaluation.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Action(Enum):
    """Standard CRUD actions"""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class Resource(Enum):
    """System resources"""

    ARTICLES = "articles"
    SOURCES = "sources"
    KEYWORDS = "keywords"
    USERS = "users"
    ORGANIZATIONS = "organizations"
    API_KEYS = "api_keys"
    WEBHOOKS = "webhooks"
    SAVED_SEARCHES = "saved_searches"
    ANALYTICS = "analytics"
    SETTINGS = "settings"
    BILLING = "billing"


@dataclass
class Permission:
    """Represents a specific permission"""

    resource: Resource
    action: Action
    conditions: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.resource.value}:{self.action.value}"

    def matches(self, resource: str, action: str) -> bool:
        """Check if this permission matches a resource:action pattern"""
        return self.resource.value == resource and (self.action.value == action or self.action == Action.ADMIN)

    @classmethod
    def from_string(cls, permission_str: str) -> "Permission":
        """Create permission from string like 'articles:read'"""
        try:
            resource_str, action_str = permission_str.split(":", 1)

            # Handle wildcard actions
            if action_str == "*":
                action = Action.ADMIN
            else:
                action = Action(action_str)

            resource = Resource(resource_str)
            return cls(resource=resource, action=action)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid permission format: {permission_str}") from e


class Role(Enum):
    """Predefined system roles"""

    ADMIN = "admin"
    EDITOR = "editor"
    ANALYST = "analyst"
    VIEWER = "viewer"


class RoleManager:
    """Manages roles and permissions"""

    def __init__(self):
        """Initialize role manager with default role definitions"""
        self.role_permissions = self._define_default_roles()
        self.role_hierarchy = self._define_role_hierarchy()

    def _define_default_roles(self) -> Dict[Role, List[Permission]]:
        """Define default permissions for each role"""
        return {
            Role.ADMIN: [
                # Full access to all resources
                Permission(Resource.ARTICLES, Action.ADMIN),
                Permission(Resource.SOURCES, Action.ADMIN),
                Permission(Resource.KEYWORDS, Action.ADMIN),
                Permission(Resource.USERS, Action.ADMIN),
                Permission(Resource.ORGANIZATIONS, Action.ADMIN),
                Permission(Resource.API_KEYS, Action.ADMIN),
                Permission(Resource.WEBHOOKS, Action.ADMIN),
                Permission(Resource.SAVED_SEARCHES, Action.ADMIN),
                Permission(Resource.ANALYTICS, Action.ADMIN),
                Permission(Resource.SETTINGS, Action.ADMIN),
                Permission(Resource.BILLING, Action.ADMIN),
            ],
            Role.EDITOR: [
                # Can manage content and configuration
                Permission(Resource.ARTICLES, Action.CREATE),
                Permission(Resource.ARTICLES, Action.READ),
                Permission(Resource.ARTICLES, Action.UPDATE),
                Permission(Resource.ARTICLES, Action.DELETE),
                Permission(Resource.SOURCES, Action.ADMIN),
                Permission(Resource.KEYWORDS, Action.ADMIN),
                Permission(Resource.WEBHOOKS, Action.ADMIN),
                Permission(Resource.SAVED_SEARCHES, Action.ADMIN),
                Permission(Resource.ANALYTICS, Action.READ),
                Permission(Resource.USERS, Action.READ),
                Permission(Resource.API_KEYS, Action.CREATE),
                Permission(Resource.API_KEYS, Action.READ),
                Permission(Resource.API_KEYS, Action.DELETE),
            ],
            Role.ANALYST: [
                # Focus on data analysis and reporting
                Permission(Resource.ARTICLES, Action.READ),
                Permission(Resource.SOURCES, Action.READ),
                Permission(Resource.KEYWORDS, Action.READ),
                Permission(Resource.KEYWORDS, Action.CREATE),
                Permission(Resource.KEYWORDS, Action.UPDATE),
                Permission(Resource.SAVED_SEARCHES, Action.ADMIN),
                Permission(Resource.ANALYTICS, Action.READ),
                Permission(Resource.WEBHOOKS, Action.READ),
                Permission(Resource.API_KEYS, Action.READ),
            ],
            Role.VIEWER: [
                # Read-only access to basic resources
                Permission(Resource.ARTICLES, Action.READ),
                Permission(Resource.SOURCES, Action.READ),
                Permission(Resource.KEYWORDS, Action.READ),
                Permission(Resource.SAVED_SEARCHES, Action.CREATE),
                Permission(Resource.SAVED_SEARCHES, Action.READ),
                Permission(Resource.SAVED_SEARCHES, Action.UPDATE),
                Permission(Resource.SAVED_SEARCHES, Action.DELETE, conditions={"owner_only": True}),
                Permission(Resource.ANALYTICS, Action.READ),
            ],
        }

    def _define_role_hierarchy(self) -> Dict[Role, List[Role]]:
        """Define role hierarchy for inheritance"""
        return {
            Role.ADMIN: [Role.EDITOR, Role.ANALYST, Role.VIEWER],
            Role.EDITOR: [Role.ANALYST, Role.VIEWER],
            Role.ANALYST: [Role.VIEWER],
            Role.VIEWER: [],
        }

    def get_role_permissions(self, role: Role) -> List[Permission]:
        """Get all permissions for a role (including inherited)"""
        permissions = set()

        # Add direct permissions
        permissions.update(self.role_permissions.get(role, []))

        # Add inherited permissions
        for inherited_role in self.role_hierarchy.get(role, []):
            permissions.update(self.role_permissions.get(inherited_role, []))

        return list(permissions)

    def has_permission(
        self, user_role: Role, resource: str, action: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a role has permission for a specific action

        Args:
            user_role: User's role
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for condition evaluation

        Returns:
            True if permission granted
        """
        permissions = self.get_role_permissions(user_role)

        for permission in permissions:
            if permission.matches(resource, action):
                # Check conditions if any
                if permission.conditions:
                    if not self._evaluate_conditions(permission.conditions, context):
                        continue
                return True

        return False

    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate permission conditions"""
        if not context:
            return False

        for condition_key, condition_value in conditions.items():
            if condition_key == "owner_only":
                if condition_value and context.get("user_id") != context.get("resource_owner_id"):
                    return False
            elif condition_key == "organization_only":
                if condition_value and context.get("user_org_id") != context.get("resource_org_id"):
                    return False
            # Add more condition types as needed

        return True

    def get_accessible_resources(self, user_role: Role) -> Dict[str, List[str]]:
        """Get all resources and actions accessible to a role"""
        permissions = self.get_role_permissions(user_role)
        accessible = {}

        for permission in permissions:
            resource = permission.resource.value
            action = permission.action.value

            if resource not in accessible:
                accessible[resource] = []

            if action not in accessible[resource]:
                accessible[resource].append(action)

        return accessible

    def can_access_endpoint(
        self, user_role: Role, method: str, endpoint: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if role can access a specific API endpoint

        Args:
            user_role: User's role
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            context: Additional context

        Returns:
            True if access granted
        """
        # Map HTTP methods to actions
        method_to_action = {
            "GET": Action.READ,
            "POST": Action.CREATE,
            "PUT": Action.UPDATE,
            "PATCH": Action.UPDATE,
            "DELETE": Action.DELETE,
        }

        action = method_to_action.get(method.upper())
        if not action:
            return False

        # Extract resource from endpoint
        resource = self._extract_resource_from_endpoint(endpoint)
        if not resource:
            return False

        return self.has_permission(user_role, resource, action.value, context)

    def _extract_resource_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract resource name from API endpoint"""
        # Remove leading slash and split by /
        parts = endpoint.lstrip("/").split("/")

        if not parts:
            return None

        # Map API endpoints to resources
        endpoint_mappings = {
            "articles": Resource.ARTICLES.value,
            "sources": Resource.SOURCES.value,
            "keywords": Resource.KEYWORDS.value,
            "users": Resource.USERS.value,
            "organizations": Resource.ORGANIZATIONS.value,
            "api-keys": Resource.API_KEYS.value,
            "webhooks": Resource.WEBHOOKS.value,
            "saved-searches": Resource.SAVED_SEARCHES.value,
            "analytics": Resource.ANALYTICS.value,
            "settings": Resource.SETTINGS.value,
            "billing": Resource.BILLING.value,
            "search": Resource.ARTICLES.value,  # Search is reading articles
        }

        # Get the first part of the endpoint (resource identifier)
        resource_path = parts[0]
        return endpoint_mappings.get(resource_path)


class PermissionChecker:
    """Utility class for checking permissions in application code"""

    def __init__(self, role_manager: RoleManager):
        self.role_manager = role_manager

    def check_permission(
        self, user_role: str, permission_string: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check permission using string format

        Args:
            user_role: User role as string
            permission_string: Permission in format 'resource:action'
            context: Additional context

        Returns:
            True if permission granted
        """
        try:
            role = Role(user_role)
            resource, action = permission_string.split(":", 1)
            return self.role_manager.has_permission(role, resource, action, context)
        except (ValueError, KeyError):
            return False

    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get all permissions for a user role as strings"""
        try:
            role = Role(user_role)
            permissions = self.role_manager.get_role_permissions(role)
            return [str(p) for p in permissions]
        except (ValueError, KeyError):
            return []

    def filter_by_permissions(
        self,
        user_role: str,
        items: List[Dict[str, Any]],
        resource: str,
        action: str = "read",
        owner_field: str = "user_id",
        org_field: str = "organization_id",
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of items based on permissions

        Args:
            user_role: User role
            items: List of items to filter
            resource: Resource type
            action: Action being performed
            owner_field: Field containing owner ID
            org_field: Field containing organization ID
            user_id: Current user ID
            organization_id: Current organization ID

        Returns:
            Filtered list of items
        """
        try:
            role = Role(user_role)
            filtered_items = []

            for item in items:
                context = {
                    "user_id": user_id,
                    "organization_id": organization_id,
                    "resource_owner_id": item.get(owner_field),
                    "resource_org_id": item.get(org_field),
                }

                if self.role_manager.has_permission(role, resource, action, context):
                    filtered_items.append(item)

            return filtered_items

        except (ValueError, KeyError):
            return []


# Global instances
role_manager = RoleManager()
permission_checker = PermissionChecker(role_manager)
