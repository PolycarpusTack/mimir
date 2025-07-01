#!/usr/bin/env python3
"""
Secure Configuration Loader for Mimir News Scraper

Handles loading configuration from JSON files with environment variable substitution
and validation to prevent hardcoded secrets.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Secure configuration loader with environment variable substitution"""
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration loader
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self._config_cache = None
        
    def load_config(self, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration with environment variable substitution
        
        Args:
            reload: Force reload from file (ignore cache)
            
        Returns:
            Configuration dictionary with substituted values
        """
        if self._config_cache is not None and not reload:
            return self._config_cache
            
        try:
            with open(self.config_file, 'r') as f:
                config_text = f.read()
                
            # Substitute environment variables
            config_text = self._substitute_env_vars(config_text)
            
            # Parse JSON
            config = json.loads(config_text)
            
            # Validate configuration
            self._validate_config(config)
            
            # Cache configuration
            self._config_cache = config
            
            logger.info(f"Configuration loaded successfully from {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _substitute_env_vars(self, text: str) -> str:
        """
        Substitute ${VAR_NAME} patterns with environment variables
        
        Args:
            text: Configuration text with placeholders
            
        Returns:
            Text with environment variables substituted
        """
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            
            if env_value is None:
                logger.warning(f"Environment variable {var_name} not found, using empty string")
                return ""
                
            return env_value
            
        # Replace ${VAR_NAME} patterns
        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}'
        return re.sub(pattern, replace_var, text)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration for security issues
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If security issues are found
        """
        # Check for hardcoded passwords or secrets
        def check_for_secrets(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for suspicious keys with hardcoded values
                    if any(keyword in key.lower() for keyword in ['password', 'secret', 'key', 'token']):
                        if isinstance(value, str) and value and not value.startswith('${'):
                            # Check if it looks like a placeholder value
                            if value in ['your-password', 'your-secret', 'your-key', 'change-me']:
                                logger.warning(f"Placeholder value found for {current_path}: {value}")
                            else:
                                raise ValueError(f"Hardcoded secret detected at {current_path}")
                    
                    check_for_secrets(value, current_path)
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_for_secrets(item, f"{path}[{i}]")
                    
        check_for_secrets(config)
        
        # Validate required environment variables
        required_env_vars = [
            'POSTGRES_PASSWORD',
            'JWT_SECRET_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            # Don't raise in development mode, just warn
            if os.getenv('ENVIRONMENT') == 'production':
                raise ValueError(f"Missing required environment variables: {missing_vars}")
                
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        config = self.load_config()
        
        # Check if using PostgreSQL or SQLite
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return {
                'type': 'postgresql',
                'url': db_url,
                'pool_size': int(os.getenv('DB_POOL_SIZE', '20')),
                'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '10'))
            }
        else:
            return {
                'type': 'sqlite',
                'name': config.get('database_name', 'mimir_news.db')
            }
            
    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration"""
        config = self.load_config()
        email_config = config.get('notification_email', {})
        
        # Ensure environment variables are loaded
        return {
            'send_email': email_config.get('send_email', False),
            'smtp_server': os.getenv('SMTP_SERVER', email_config.get('smtp_server')),
            'smtp_port': int(os.getenv('SMTP_PORT', email_config.get('smtp_port', 587))),
            'smtp_user': os.getenv('SMTP_USER', email_config.get('smtp_user')),
            'smtp_password': os.getenv('SMTP_PASSWORD', email_config.get('smtp_password')),
            'recipient_email': os.getenv('RECIPIENT_EMAIL', email_config.get('recipient_email'))
        }
        
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'jwt_secret': os.getenv('JWT_SECRET_KEY'),
            'session_secret': os.getenv('SESSION_SECRET_KEY'),
            'cors_origins': os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else [],
            'allowed_hosts': os.getenv('ALLOWED_HOSTS', '').split(',') if os.getenv('ALLOWED_HOSTS') else []
        }


# Global configuration loader instance
config_loader = ConfigLoader()


def load_config(reload: bool = False) -> Dict[str, Any]:
    """Load configuration (convenience function)"""
    return config_loader.load_config(reload)


def get_database_config() -> Dict[str, Any]:
    """Get database configuration (convenience function)"""
    return config_loader.get_database_config()


def get_email_config() -> Dict[str, Any]:
    """Get email configuration (convenience function)"""
    return config_loader.get_email_config()


def get_security_config() -> Dict[str, Any]:
    """Get security configuration (convenience function)"""
    return config_loader.get_security_config()


if __name__ == "__main__":
    # Test configuration loading
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        print("Testing configuration loading...")
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        print("\nDatabase config:")
        print(json.dumps(get_database_config(), indent=2))
        
        print("\nEmail config:")
        email_config = get_email_config()
        # Mask sensitive data for display
        if email_config.get('smtp_password'):
            email_config['smtp_password'] = '***masked***'
        print(json.dumps(email_config, indent=2))
        
        print("\nSecurity config:")
        security_config = get_security_config()
        # Mask sensitive data for display
        if security_config.get('jwt_secret'):
            security_config['jwt_secret'] = '***masked***'
        if security_config.get('session_secret'):
            security_config['session_secret'] = '***masked***'
        print(json.dumps(security_config, indent=2))
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        sys.exit(1)