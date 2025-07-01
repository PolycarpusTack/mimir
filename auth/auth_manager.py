"""
Main Authentication Manager for Mimir Enterprise

Orchestrates all authentication and authorization components,
providing a unified interface for auth operations.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .api_key_manager import APIKeyManager, APIKeyType
from .jwt_handler import JWTHandler, TokenPayload
from .rbac import Role, RoleManager, permission_checker
from .tenant_context import DatabaseContextManager, OrganizationManager, TenantContext, TenantContextManager

logger = logging.getLogger(__name__)


class AuthManager:
    """Main authentication and authorization manager"""

    def __init__(self, db_manager, secret_key: Optional[str] = None):
        """
        Initialize authentication manager

        Args:
            db_manager: Database manager instance
            secret_key: Optional secret key for JWT (will generate if not provided)
        """
        self.db = db_manager

        # Initialize secret key
        self.secret_key = secret_key or self._generate_secret_key()

        # Initialize component managers
        self.jwt_handler = JWTHandler(self.secret_key)
        self.api_key_manager = APIKeyManager(db_manager)
        self.role_manager = RoleManager()
        self.tenant_manager = TenantContextManager()
        self.db_context_manager = DatabaseContextManager(db_manager)
        self.org_manager = OrganizationManager(db_manager)

        logger.info("AuthManager initialized successfully")

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for JWT signing"""
        # Try to get from environment first
        secret = os.getenv("MIMIR_JWT_SECRET")
        if secret:
            return secret

        # Generate a new secret (in production, this should be stored securely)
        logger.warning("No JWT secret found in environment, generating new one")
        return secrets.token_urlsafe(64)

    # User Authentication Methods

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with email and password

        Args:
            email: User email
            password: User password

        Returns:
            User data if authenticated, None if failed
        """
        try:
            # Hash password for comparison
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            query = """
            SELECT u.id, u.email, u.name, u.role, u.organization_id, u.is_active,
                   o.name as organization_name, o.slug as organization_slug,
                   o.tier, o.subscription_status
            FROM users u
            JOIN organizations o ON o.id = u.organization_id
            WHERE u.email = %s AND u.password_hash = %s AND u.is_active = true
            """

            result = self.db.fetch_one(query, (email, password_hash))

            if result:
                # Update last login
                self.db.execute_query(
                    "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = %s", (result["id"],)
                )

                logger.info(f"User authenticated: {email}")
                return dict(result)

            logger.warning(f"Authentication failed for: {email}")
            return None

        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None

    def create_user_session(self, user_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Create JWT tokens for authenticated user

        Args:
            user_data: User data from authentication

        Returns:
            Token response with access and refresh tokens
        """
        try:
            # Get user permissions based on role
            permissions = permission_checker.get_user_permissions(user_data["role"])

            # Generate tokens
            tokens = self.jwt_handler.generate_tokens(
                user_id=user_data["id"],
                organization_id=user_data["organization_id"],
                email=user_data["email"],
                role=user_data["role"],
                permissions=permissions,
            )

            # Log activity
            self._log_activity(
                organization_id=user_data["organization_id"],
                user_id=user_data["id"],
                action="login",
                details={"auth_method": "password"},
            )

            return tokens

        except Exception as e:
            logger.error(f"Error creating user session: {e}")
            raise

    def refresh_user_session(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Refresh user session with refresh token

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair if successful
        """
        try:
            tokens = self.jwt_handler.refresh_access_token(refresh_token)
            if tokens:
                logger.info("Session refreshed successfully")
            return tokens
        except Exception as e:
            logger.error(f"Error refreshing session: {e}")
            return None

    def logout_user(self, access_token: str) -> bool:
        """
        Logout user by revoking token

        Args:
            access_token: User's access token

        Returns:
            True if successful
        """
        try:
            payload = self.jwt_handler.verify_token(access_token)
            if payload:
                self._log_activity(
                    organization_id=payload.organization_id,
                    user_id=payload.user_id,
                    action="logout",
                    details={"auth_method": "jwt"},
                )

            return self.jwt_handler.revoke_token(access_token)
        except Exception as e:
            logger.error(f"Error logging out user: {e}")
            return False

    # API Key Management

    def create_api_key(
        self,
        organization_id: str,
        user_id: str,
        name: str,
        key_type: str = "read_only",
        expires_in_days: Optional[int] = None,
        permissions: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create new API key

        Args:
            organization_id: Organization ID
            user_id: User creating the key
            name: Human-readable name
            key_type: Type of key ('read_only', 'read_write', 'admin')
            expires_in_days: Optional expiration
            permissions: Optional custom permissions

        Returns:
            Tuple of (api_key, key_info)
        """
        try:
            key_type_enum = APIKeyType(key_type)
            api_key, key_info = self.api_key_manager.generate_api_key(
                organization_id=organization_id,
                name=name,
                key_type=key_type_enum,
                user_id=user_id,
                expires_in_days=expires_in_days,
                custom_permissions=permissions,
                created_by=user_id,
            )

            # Log activity
            self._log_activity(
                organization_id=organization_id,
                user_id=user_id,
                action="create_api_key",
                resource_type="api_key",
                resource_id=key_info.id,
                details={"key_name": name, "key_type": key_type},
            )

            return api_key, key_info.to_dict()

        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise

    def validate_api_key(self, api_key: str, client_ip: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return key information

        Args:
            api_key: API key to validate
            client_ip: Optional client IP

        Returns:
            Key information if valid
        """
        try:
            key_info = self.api_key_manager.validate_api_key(api_key, client_ip)
            return key_info.to_dict() if key_info else None
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    # Organization Management

    def get_organization(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get organization by ID"""
        return self.org_manager.get_organization_by_id(org_id)

    def get_organization_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get organization by slug"""
        return self.org_manager.get_organization_by_slug(slug)

    def check_organization_limits(self, org_id: str, limit_type: str) -> bool:
        """Check organization limits"""
        return self.org_manager.check_organization_limits(org_id, limit_type)

    def get_organization_usage(self, org_id: str) -> Dict[str, Any]:
        """Get organization usage statistics"""
        return self.org_manager.get_organization_usage(org_id)

    # User Management

    def create_user(
        self,
        organization_id: str,
        email: str,
        name: str,
        password: str,
        role: str = "viewer",
        created_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create new user

        Args:
            organization_id: Organization ID
            email: User email
            name: User name
            password: User password
            role: User role
            created_by: ID of user creating this user

        Returns:
            Created user data
        """
        try:
            # Check organization user limit
            if not self.check_organization_limits(organization_id, "users"):
                raise ValueError("Organization user limit exceeded")

            # Hash password
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            # Generate API key for the user
            api_key = secrets.token_urlsafe(32)
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Create user
            query = """
            INSERT INTO users (organization_id, email, name, password_hash, role, api_key, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, email, name, role, organization_id, created_at
            """

            result = self.db.fetch_one(
                query, (organization_id, email, name, password_hash, role, api_key_hash, created_by)
            )

            if result:
                # Log activity
                self._log_activity(
                    organization_id=organization_id,
                    user_id=created_by,
                    action="create_user",
                    resource_type="user",
                    resource_id=result["id"],
                    details={"email": email, "role": role},
                )

                user_data = dict(result)
                user_data["api_key"] = api_key  # Return the actual key once
                return user_data

            return None

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            query = """
            SELECT u.id, u.email, u.name, u.role, u.organization_id, u.is_active,
                   u.created_at, u.last_login_at,
                   o.name as organization_name, o.slug as organization_slug
            FROM users u
            JOIN organizations o ON o.id = u.organization_id
            WHERE u.id = %s
            """
            return self.db.fetch_one(query, (user_id,))
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None

    def update_user_role(self, user_id: str, new_role: str, updated_by: str) -> bool:
        """Update user role"""
        try:
            query = "UPDATE users SET role = %s WHERE id = %s"
            result = self.db.execute_query(query, (new_role, user_id))

            if result:
                # Get user for logging
                user = self.get_user_by_id(user_id)
                if user:
                    self._log_activity(
                        organization_id=user["organization_id"],
                        user_id=updated_by,
                        action="update_user_role",
                        resource_type="user",
                        resource_id=user_id,
                        details={"new_role": new_role, "email": user["email"]},
                    )
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            return False

    # Permission Checking

    def check_permission(self, user_role: str, permission: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if user role has permission"""
        return permission_checker.check_permission(user_role, permission, context)

    def get_user_permissions(self, user_role: str) -> List[str]:
        """Get all permissions for user role"""
        return permission_checker.get_user_permissions(user_role)

    # Context Management

    def set_tenant_context(self, context: TenantContext) -> None:
        """Set current tenant context"""
        self.tenant_manager.set_context(context)
        self.db_context_manager.set_tenant_context(context.organization_id, context.user_id, context.user_role)

    def get_tenant_context(self) -> Optional[TenantContext]:
        """Get current tenant context"""
        return self.tenant_manager.get_context()

    def clear_tenant_context(self) -> None:
        """Clear current tenant context"""
        self.tenant_manager.clear_context()
        self.db_context_manager.clear_tenant_context()

    # Activity Logging

    def _log_activity(
        self,
        organization_id: str,
        user_id: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """Log user activity"""
        try:
            query = """
            INSERT INTO activity_log (
                organization_id, user_id, action, resource_type, resource_id,
                ip_address, user_agent, details, status, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            self.db.execute_query(
                query,
                (
                    organization_id,
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    ip_address,
                    user_agent,
                    details,
                    status,
                    error_message,
                ),
            )

        except Exception as e:
            logger.error(f"Error logging activity: {e}")

    # Utility Methods

    def get_auth_stats(self, organization_id: str, days: int = 30) -> Dict[str, Any]:
        """Get authentication statistics for organization"""
        try:
            # Get login statistics
            query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as logins,
                COUNT(DISTINCT user_id) as unique_users
            FROM activity_log
            WHERE organization_id = %s 
              AND action = 'login'
              AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """

            login_stats = self.db.fetch_all(query, (organization_id, days))

            # Get API usage statistics
            api_stats = self.api_key_manager.get_api_key_usage_stats(organization_id, days)

            return {"login_stats": login_stats, "api_stats": api_stats}

        except Exception as e:
            logger.error(f"Error getting auth stats: {e}")
            return {}

    def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens and sessions"""
        try:
            count = 0

            # Cleanup expired API keys
            query = """
            UPDATE api_keys 
            SET is_active = false 
            WHERE expires_at < CURRENT_TIMESTAMP AND is_active = true
            """
            result = self.db.execute_query(query)
            if result:
                count += result

            logger.info(f"Cleaned up {count} expired auth tokens")
            return count

        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {e}")
            return 0
