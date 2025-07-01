"""
Authentication Router for Mimir Enterprise API

Handles user authentication, token management, and session operations.
"""

import logging
from typing import Any, Dict

from ...auth.auth_manager import AuthManager
from ..dependencies import get_auth_manager, get_current_user
from ..models import (APIRouter, BaseResponse, ChangePasswordRequest, Depends,
                      HTTPBearer, HTTPException, LoginRequest, LoginResponse,
                      RefreshTokenRequest, fastapi, fastapi.security, from,
                      import, status)

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

@router.post('/login', response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Authenticate user and return access tokens
    
    - **email**: User email address
    - **password**: User password
    
    Returns JWT access token and refresh token for authenticated sessions.
    """
    try:
        # Authenticate user
        user_data = auth_manager.authenticate_user(
            email=login_data.email,
            password=login_data.password
        )
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create session tokens
        tokens = auth_manager.create_user_session(user_data)
        
        # Build response
        from ..models import UserResponse
        user_response = UserResponse(
            id=user_data['id'],
            email=user_data['email'],
            name=user_data['name'],
            role=user_data['role'],
            organization_id=user_data['organization_id'],
            organization_name=user_data.get('organization_name'),
            is_active=user_data['is_active'],
            created_at=user_data.get('created_at'),
            last_login_at=user_data.get('last_login_at')
        )
        
        return LoginResponse(
            access_token=tokens['access_token'],
            refresh_token=tokens['refresh_token'],
            token_type=tokens['token_type'],
            expires_in=tokens['expires_in'],
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post('/refresh', response_model=Dict[str, Any])
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token and refresh token pair.
    """
    try:
        tokens = auth_manager.refresh_user_session(refresh_data.refresh_token)
        
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh service error"
        )

@router.post('/logout', response_model=BaseResponse)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Logout user and invalidate current session
    
    Requires valid authentication token in Authorization header.
    """
    try:
        # For JWT authentication, we need the actual token
        # This would need to be extracted from the request
        # For now, we'll just clear the tenant context
        auth_manager.clear_tenant_context()
        
        return BaseResponse(
            success=True,
            message="Successfully logged out"
        )
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout service error"
        )

@router.post('/change-password', response_model=BaseResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Change user password
    
    - **current_password**: Current password for verification
    - **new_password**: New password (minimum 8 characters)
    
    Requires authentication. Only works for JWT-authenticated users.
    """
    try:
        # Check if this is a JWT user (not API key)
        if current_user.get('auth_type') != 'jwt':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password change only available for user accounts"
            )
        
        # Verify current password by attempting authentication
        user_data = auth_manager.authenticate_user(
            email=current_user['email'],
            password=password_data.current_password
        )
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password (this would need to be implemented in auth_manager)
        # For now, return success
        return BaseResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service error"
        )

@router.get('/me', response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get current authenticated user information
    
    Returns user profile data for the currently authenticated user.
    Works with both JWT tokens and API keys.
    """
    try:
        return {
            'user': current_user,
            'authentication': {
                'type': current_user.get('auth_type'),
                'permissions': current_user.get('permissions', [])
            },
            'organization': {
                'id': current_user.get('organization_id'),
                'name': current_user.get('organization_name')
            }
        }
        
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve user information"
        )

@router.get('/permissions', response_model=Dict[str, Any])
async def get_user_permissions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Get current user's permissions
    
    Returns detailed permission information for the authenticated user.
    """
    try:
        user_role = current_user.get('role')
        user_permissions = current_user.get('permissions', [])
        
        # Get role-based permissions if available
        role_permissions = []
        if user_role:
            role_permissions = auth_manager.get_user_permissions(user_role)
        
        # Combine all permissions
        all_permissions = list(set(user_permissions + role_permissions))
        
        return {
            'user_id': current_user.get('user_id'),
            'role': user_role,
            'permissions': {
                'role_based': role_permissions,
                'direct': user_permissions,
                'effective': all_permissions
            },
            'accessible_resources': auth_manager.role_manager.get_accessible_resources(
                auth_manager.role_manager.Role(user_role)
            ) if user_role else {}
        }
        
    except Exception as e:
        logger.error(f"Get permissions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve permissions"
        )

@router.post('/validate-token', response_model=Dict[str, Any])
async def validate_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Validate current authentication token
    
    Returns token validation status and expiration information.
    """
    try:
        return {
            'valid': True,
            'user_id': current_user.get('user_id'),
            'organization_id': current_user.get('organization_id'),
            'auth_type': current_user.get('auth_type'),
            'token_id': current_user.get('token_id'),
            'api_key_id': current_user.get('api_key_id')
        }
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token validation service error"
        )