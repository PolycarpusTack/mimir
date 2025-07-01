"""
API Key Manager for Mimir Enterprise

Handles API key generation, validation, and management for programmatic access
with support for different key types, rate limiting, and usage tracking.
"""

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class APIKeyType(Enum):
    """API Key types with different permission levels"""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"


@dataclass
class APIKeyInfo:
    """API Key information structure"""

    id: str
    organization_id: str
    user_id: Optional[str]
    name: str
    key_prefix: str
    key_type: APIKeyType
    permissions: List[str]
    rate_limit_per_hour: Optional[int]
    allowed_ips: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    created_at: datetime
    created_by: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "key_type": self.key_type.value,
            "permissions": self.permissions,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "allowed_ips": self.allowed_ips,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


class APIKeyManager:
    """Manages API key operations"""

    def __init__(self, db_manager):
        """
        Initialize API key manager

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.key_prefix = "mk_"  # Mimir Key prefix
        self.key_length = 32  # Length of the secret part

        # Default permissions for each key type
        self.default_permissions = {
            APIKeyType.READ_ONLY: ["articles:read", "search:read", "sources:read", "keywords:read"],
            APIKeyType.READ_WRITE: [
                "articles:read",
                "articles:write",
                "search:read",
                "sources:read",
                "sources:write",
                "keywords:read",
                "keywords:write",
                "webhooks:read",
                "webhooks:write",
            ],
            APIKeyType.ADMIN: [
                "articles:*",
                "search:*",
                "sources:*",
                "keywords:*",
                "webhooks:*",
                "users:read",
                "analytics:read",
                "settings:read",
            ],
        }

    def generate_api_key(
        self,
        organization_id: str,
        name: str,
        key_type: APIKeyType,
        user_id: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_per_hour: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        custom_permissions: Optional[List[str]] = None,
        created_by: Optional[str] = None,
    ) -> Tuple[str, APIKeyInfo]:
        """
        Generate a new API key

        Args:
            organization_id: Organization UUID
            name: Human-readable name for the key
            key_type: Type of API key
            user_id: Optional user who owns the key
            expires_in_days: Optional expiration in days
            rate_limit_per_hour: Optional rate limit
            allowed_ips: Optional list of allowed IP addresses
            custom_permissions: Optional custom permissions (overrides defaults)
            created_by: UUID of user creating the key

        Returns:
            Tuple of (full_api_key, api_key_info)
        """
        try:
            # Generate the secret key
            secret = secrets.token_urlsafe(self.key_length)
            full_key = f"{self.key_prefix}{secret}"

            # Create hash for storage
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            key_prefix = full_key[:8]  # First 8 characters for identification

            # Set permissions
            permissions = custom_permissions or self.default_permissions.get(key_type, [])

            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

            # Set default rate limit based on key type
            if rate_limit_per_hour is None:
                rate_limits = {APIKeyType.READ_ONLY: 1000, APIKeyType.READ_WRITE: 500, APIKeyType.ADMIN: 2000}
                rate_limit_per_hour = rate_limits.get(key_type, 1000)

            # Create API key record
            key_id = str(uuid.uuid4())

            # Store in database
            query = """
            INSERT INTO api_keys (
                id, organization_id, user_id, name, key_hash, key_prefix,
                type, permissions, rate_limit_per_hour, allowed_ips,
                expires_at, created_by
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            params = (
                key_id,
                organization_id,
                user_id,
                name,
                key_hash,
                key_prefix,
                key_type.value,
                custom_permissions or permissions,
                rate_limit_per_hour,
                allowed_ips or [],
                expires_at,
                created_by,
            )

            self.db.execute_query(query, params)

            # Create API key info object
            api_key_info = APIKeyInfo(
                id=key_id,
                organization_id=organization_id,
                user_id=user_id,
                name=name,
                key_prefix=key_prefix,
                key_type=key_type,
                permissions=permissions,
                rate_limit_per_hour=rate_limit_per_hour,
                allowed_ips=allowed_ips or [],
                is_active=True,
                expires_at=expires_at,
                last_used_at=None,
                usage_count=0,
                created_at=datetime.now(timezone.utc),
                created_by=created_by,
            )

            logger.info(f"Generated API key '{name}' for organization {organization_id}")
            return full_key, api_key_info

        except Exception as e:
            logger.error(f"Error generating API key: {e}")
            raise

    def validate_api_key(self, api_key: str, client_ip: Optional[str] = None) -> Optional[APIKeyInfo]:
        """
        Validate an API key and return its information

        Args:
            api_key: The full API key
            client_ip: Optional client IP for validation

        Returns:
            APIKeyInfo if valid, None if invalid
        """
        try:
            # Check key format
            if not api_key.startswith(self.key_prefix):
                return None

            # Create hash for lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Query database
            query = """
            SELECT 
                id, organization_id, user_id, name, key_prefix, type,
                permissions, rate_limit_per_hour, allowed_ips, is_active,
                expires_at, last_used_at, usage_count, created_at, created_by
            FROM api_keys
            WHERE key_hash = %s AND is_active = true
            """

            result = self.db.fetch_one(query, (key_hash,))
            if not result:
                return None

            # Check expiration
            if result["expires_at"] and result["expires_at"] < datetime.now(timezone.utc):
                logger.warning(f"API key {result['key_prefix']}... has expired")
                return None

            # Check IP restrictions
            if client_ip and result["allowed_ips"]:
                if client_ip not in result["allowed_ips"]:
                    logger.warning(f"API key {result['key_prefix']}... used from unauthorized IP: {client_ip}")
                    return None

            # Update last used timestamp and usage count
            self._update_key_usage(result["id"])

            # Create APIKeyInfo object
            api_key_info = APIKeyInfo(
                id=result["id"],
                organization_id=result["organization_id"],
                user_id=result["user_id"],
                name=result["name"],
                key_prefix=result["key_prefix"],
                key_type=APIKeyType(result["type"]),
                permissions=result["permissions"],
                rate_limit_per_hour=result["rate_limit_per_hour"],
                allowed_ips=result["allowed_ips"],
                is_active=result["is_active"],
                expires_at=result["expires_at"],
                last_used_at=result["last_used_at"],
                usage_count=result["usage_count"] + 1,  # Include current usage
                created_at=result["created_at"],
                created_by=result["created_by"],
            )

            return api_key_info

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    def _update_key_usage(self, key_id: str) -> None:
        """Update API key usage statistics"""
        try:
            query = """
            UPDATE api_keys 
            SET last_used_at = CURRENT_TIMESTAMP, usage_count = usage_count + 1
            WHERE id = %s
            """
            self.db.execute_query(query, (key_id,))
        except Exception as e:
            logger.error(f"Error updating key usage: {e}")

    def list_api_keys(self, organization_id: str) -> List[APIKeyInfo]:
        """
        List all API keys for an organization

        Args:
            organization_id: Organization UUID

        Returns:
            List of APIKeyInfo objects
        """
        try:
            query = """
            SELECT 
                id, organization_id, user_id, name, key_prefix, type,
                permissions, rate_limit_per_hour, allowed_ips, is_active,
                expires_at, last_used_at, usage_count, created_at, created_by
            FROM api_keys
            WHERE organization_id = %s
            ORDER BY created_at DESC
            """

            results = self.db.fetch_all(query, (organization_id,))

            return [
                APIKeyInfo(
                    id=row["id"],
                    organization_id=row["organization_id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    key_prefix=row["key_prefix"],
                    key_type=APIKeyType(row["type"]),
                    permissions=row["permissions"],
                    rate_limit_per_hour=row["rate_limit_per_hour"],
                    allowed_ips=row["allowed_ips"],
                    is_active=row["is_active"],
                    expires_at=row["expires_at"],
                    last_used_at=row["last_used_at"],
                    usage_count=row["usage_count"],
                    created_at=row["created_at"],
                    created_by=row["created_by"],
                )
                for row in results
            ]

        except Exception as e:
            logger.error(f"Error listing API keys: {e}")
            return []

    def revoke_api_key(self, key_id: str, revoked_by: Optional[str] = None) -> bool:
        """
        Revoke an API key

        Args:
            key_id: API key UUID
            revoked_by: UUID of user revoking the key

        Returns:
            True if successful
        """
        try:
            query = """
            UPDATE api_keys 
            SET is_active = false, revoked_at = CURRENT_TIMESTAMP, revoked_by = %s
            WHERE id = %s
            """

            result = self.db.execute_query(query, (revoked_by, key_id))
            if result:
                logger.info(f"Revoked API key {key_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False

    def get_api_key_usage_stats(self, organization_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get API key usage statistics

        Args:
            organization_id: Organization UUID
            days: Number of days to look back

        Returns:
            Usage statistics dictionary
        """
        try:
            # Get basic stats
            query = """
            SELECT 
                COUNT(*) as total_keys,
                COUNT(*) FILTER (WHERE is_active = true) as active_keys,
                COUNT(*) FILTER (WHERE expires_at < CURRENT_TIMESTAMP) as expired_keys,
                SUM(usage_count) as total_usage,
                AVG(usage_count) as avg_usage_per_key
            FROM api_keys
            WHERE organization_id = %s
            """

            basic_stats = self.db.fetch_one(query, (organization_id,))

            # Get usage by key type
            query = """
            SELECT 
                type,
                COUNT(*) as count,
                SUM(usage_count) as total_usage
            FROM api_keys
            WHERE organization_id = %s AND is_active = true
            GROUP BY type
            """

            usage_by_type = self.db.fetch_all(query, (organization_id,))

            # Get recent usage (last 30 days)
            query = """
            SELECT 
                DATE(last_used_at) as date,
                COUNT(DISTINCT id) as keys_used,
                SUM(usage_count) as total_calls
            FROM api_keys
            WHERE organization_id = %s 
              AND last_used_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
            GROUP BY DATE(last_used_at)
            ORDER BY date DESC
            """

            recent_usage = self.db.fetch_all(query, (organization_id, days))

            return {"basic_stats": basic_stats, "usage_by_type": usage_by_type, "recent_usage": recent_usage}

        except Exception as e:
            logger.error(f"Error getting API key usage stats: {e}")
            return {}

    def check_rate_limit(self, api_key_info: APIKeyInfo) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if API key has exceeded rate limit

        Args:
            api_key_info: API key information

        Returns:
            Tuple of (within_limit, rate_limit_info)
        """
        try:
            if not api_key_info.rate_limit_per_hour:
                return True, {"unlimited": True}

            # Check usage in the last hour
            query = """
            SELECT COUNT(*) as calls_last_hour
            FROM usage_metrics
            WHERE organization_id = %s
              AND metric_type = 'api_calls'
              AND recorded_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
              AND metadata->>'api_key_id' = %s
            """

            result = self.db.fetch_one(query, (api_key_info.organization_id, api_key_info.id))
            calls_last_hour = result["calls_last_hour"] if result else 0

            within_limit = calls_last_hour < api_key_info.rate_limit_per_hour

            return within_limit, {
                "limit": api_key_info.rate_limit_per_hour,
                "used": calls_last_hour,
                "remaining": max(0, api_key_info.rate_limit_per_hour - calls_last_hour),
                "reset_at": datetime.now(timezone.utc) + timedelta(hours=1),
            }

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True, {"error": str(e)}

    def record_api_call(self, api_key_info: APIKeyInfo, endpoint: str, status_code: int) -> None:
        """
        Record an API call for usage tracking

        Args:
            api_key_info: API key information
            endpoint: API endpoint called
            status_code: HTTP status code
        """
        try:
            # Record usage metric
            query = """
            INSERT INTO usage_metrics (organization_id, metric_type, metric_value, metadata)
            VALUES (%s, 'api_calls', 1, %s)
            """

            metadata = {
                "api_key_id": api_key_info.id,
                "api_key_name": api_key_info.name,
                "endpoint": endpoint,
                "status_code": status_code,
                "key_type": api_key_info.key_type.value,
            }

            self.db.execute_query(query, (api_key_info.organization_id, metadata))

        except Exception as e:
            logger.error(f"Error recording API call: {e}")
