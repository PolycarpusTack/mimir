"""
JWT Token Handler for Mimir Enterprise Authentication

Handles JWT token creation, validation, and refresh functionality
with support for multi-tenant contexts and role-based claims.
"""

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT token payload structure"""

    user_id: str
    organization_id: str
    email: str
    role: str
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_id: str

    def to_dict(self) -> Dict[str, Any]:
        """TODO: Add docstring for to_dict"""
        return {
            "sub": self.user_id,
            "org_id": self.organization_id,
            "email": self.email,
            "role": self.role,
            "permissions": self.permissions,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "jti": self.token_id,
            "iss": "mimir-enterprise",
            "aud": "mimir-api",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        """TODO: Add docstring for from_dict"""
        return cls(
            user_id=data["sub"],
            organization_id=data["org_id"],
            email=data["email"],
            role=data["role"],
            permissions=data.get("permissions", []),
            issued_at=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            token_id=data["jti"],
        )


class JWTHandler:
    """Handles JWT token operations for authentication"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """
        Initialize JWT handler

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT signing algorithm (default: HS256)
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 60  # 1 hour
        self.refresh_token_expire_days = 30  # 30 days

        # Token blacklist for logout/revocation
        self._blacklisted_tokens = set()

    def generate_tokens(
        self, user_id: str, organization_id: str, email: str, role: str, permissions: List[str]
    ) -> Dict[str, str]:
        """
        Generate access and refresh tokens

        Args:
            user_id: User UUID
            organization_id: Organization UUID
            email: User email
            role: User role
            permissions: List of permissions

        Returns:
            Dictionary with access_token and refresh_token
        """
        now = datetime.now(timezone.utc)

        # Generate access token
        access_payload = TokenPayload(
            user_id=user_id,
            organization_id=organization_id,
            email=email,
            role=role,
            permissions=permissions,
            issued_at=now,
            expires_at=now + timedelta(minutes=self.access_token_expire_minutes),
            token_id=str(uuid.uuid4()),
        )

        # Generate refresh token (longer expiry, minimal claims)
        refresh_payload = TokenPayload(
            user_id=user_id,
            organization_id=organization_id,
            email=email,
            role="refresh",  # Special role for refresh tokens
            permissions=[],
            issued_at=now,
            expires_at=now + timedelta(days=self.refresh_token_expire_days),
            token_id=str(uuid.uuid4()),
        )

        access_token = jwt.encode(access_payload.to_dict(), self.secret_key, algorithm=self.algorithm)

        refresh_token = jwt.encode(refresh_payload.to_dict(), self.secret_key, algorithm=self.algorithm)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
        }

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            TokenPayload if valid, None if invalid
        """
        try:
            # Check if token is blacklisted
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in self._blacklisted_tokens:
                logger.warning("Attempted use of blacklisted token")
                return None

            # Decode and verify token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], audience="mimir-api", issuer="mimir-enterprise"
            )

            return TokenPayload.from_dict(payload)

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return None

    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Generate new access token from refresh token

        Args:
            refresh_token: Valid refresh token

        Returns:
            New token pair if successful, None if invalid
        """
        payload = self.verify_token(refresh_token)
        if not payload or payload.role != "refresh":
            return None

        # Here you would typically fetch fresh user data from database
        # For now, we'll use the data from the refresh token
        return self.generate_tokens(
            user_id=payload.user_id,
            organization_id=payload.organization_id,
            email=payload.email,
            role="viewer",  # Default role, should be fetched from DB
            permissions=[],  # Should be fetched from DB
        )

    def revoke_token(self, token: str) -> bool:
        """
        Revoke/blacklist a token

        Args:
            token: Token to revoke

        Returns:
            True if successful
        """
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self._blacklisted_tokens.add(token_hash)
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get token information without verification (for debugging)

        Args:
            token: JWT token

        Returns:
            Token payload if decodable
        """
        try:
            # Decode without verification for inspection
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception as e:
            logger.error(f"Error decoding token: {e}")
            return None


class TokenManager:
    """Manages token storage and retrieval for sessions"""

    def __init__(self):
        # In production, this should use Redis or database
        self._active_tokens: Dict[str, TokenPayload] = {}
        self._user_sessions: Dict[str, List[str]] = {}

    def store_token(self, payload: TokenPayload) -> None:
        """Store active token"""
        self._active_tokens[payload.token_id] = payload

        # Track user sessions
        if payload.user_id not in self._user_sessions:
            self._user_sessions[payload.user_id] = []
        self._user_sessions[payload.user_id].append(payload.token_id)

    def get_token(self, token_id: str) -> Optional[TokenPayload]:
        """Retrieve token by ID"""
        return self._active_tokens.get(token_id)

    def remove_token(self, token_id: str) -> bool:
        """Remove token from active storage"""
        payload = self._active_tokens.pop(token_id, None)
        if payload:
            user_tokens = self._user_sessions.get(payload.user_id, [])
            if token_id in user_tokens:
                user_tokens.remove(token_id)
            return True
        return False

    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        user_tokens = self._user_sessions.get(user_id, [])
        count = 0
        for token_id in user_tokens:
            if self.remove_token(token_id):
                count += 1
        self._user_sessions[user_id] = []
        return count

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage"""
        now = datetime.now(timezone.utc)
        expired_tokens = []

        for token_id, payload in self._active_tokens.items():
            if payload.expires_at < now:
                expired_tokens.append(token_id)

        for token_id in expired_tokens:
            self.remove_token(token_id)

        return len(expired_tokens)


# Global token manager instance
token_manager = TokenManager()
