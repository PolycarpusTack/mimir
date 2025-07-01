"""
Proxy Provider Integration Templates for Mimir
Templates and utilities for integrating with major proxy service providers
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ProxyProviderConfig:
    """Configuration for a proxy provider"""
    name: str
    provider_type: str  # 'api', 'static', 'rotating'
    authentication: Dict[str, Any]
    endpoints: Dict[str, str]
    rate_limits: Dict[str, int]
    features: List[str]
    pricing_model: str


class ProxyProviderTemplate(ABC):
    """Abstract base class for proxy provider templates"""
    
    def __init__(self, config: ProxyProviderConfig):
        self.config = config
        self.session = requests.Session()
    
    @abstractmethod
    def get_proxy_list(self) -> List[Dict[str, Any]]:
        """Get list of available proxies from provider"""
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate provider credentials"""
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics from provider"""
        pass
    
    def generate_mimir_config(self) -> Dict[str, Any]:
        """Generate Mimir proxy configuration for this provider"""
        proxies = self.get_proxy_list()
        
        return {
            "type": "api" if self.config.provider_type == "api" else "static_list",
            "name": f"{self.config.name}_provider",
            "enabled": False,  # User must explicitly enable
            "description": f"Auto-generated configuration for {self.config.name}",
            "proxies": [proxy["url"] for proxy in proxies] if proxies else [],
            "provider_specific": {
                "provider": self.config.name,
                "features": self.config.features,
                "rate_limits": self.config.rate_limits,
                "pricing_model": self.config.pricing_model
            }
        }


class BrightDataTemplate(ProxyProviderTemplate):
    """Template for Bright Data (formerly Luminati) proxy service"""
    
    def __init__(self):
        config = ProxyProviderConfig(
            name="BrightData",
            provider_type="api",
            authentication={
                "username_env": "BRIGHTDATA_USERNAME",
                "password_env": "BRIGHTDATA_PASSWORD",
                "zone_env": "BRIGHTDATA_ZONE"
            },
            endpoints={
                "proxy_base": "zproxy.lum-superproxy.io",
                "api_base": "brightdata.com/api"
            },
            rate_limits={
                "requests_per_second": 10,
                "concurrent_sessions": 100
            },
            features=["residential", "datacenter", "mobile", "geo_targeting"],
            pricing_model="pay_per_gb"
        )
        super().__init__(config)
    
    def get_proxy_list(self) -> List[Dict[str, Any]]:
        """Generate Bright Data proxy endpoints"""
        username = os.getenv(self.config.authentication["username_env"])
        password = os.getenv(self.config.authentication["password_env"])
        zone = os.getenv(self.config.authentication["zone_env"], "residential")
        
        if not username or not password:
            logger.warning("Bright Data credentials not found in environment variables")
            return []
        
        # Bright Data uses different ports for different proxy types
        proxy_configs = [
            {
                "url": f"http://{username}-session-{i}:{password}@{self.config.endpoints['proxy_base']}:22225",
                "location": "global",
                "type": "residential",
                "session_id": i,
                "provider": "brightdata"
            }
            for i in range(1, 6)  # Create 5 session-based proxies
        ]
        
        return proxy_configs
    
    def validate_credentials(self) -> bool:
        """Validate Bright Data credentials"""
        try:
            username = os.getenv(self.config.authentication["username_env"])
            password = os.getenv(self.config.authentication["password_env"])
            
            if not username or not password:
                return False
            
            # Test with a simple request
            proxy_url = f"http://{username}:{password}@{self.config.endpoints['proxy_base']}:22225"
            test_session = requests.Session()
            test_session.proxies = {"http": proxy_url, "https": proxy_url}
            
            response = test_session.get("http://httpbin.org/ip", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Bright Data credential validation failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get Bright Data usage statistics"""
        # Note: Actual API integration would require API tokens
        return {
            "provider": "brightdata",
            "note": "API integration requires additional setup with Bright Data API tokens",
            "bandwidth_used": "Unknown - requires API access",
            "requests_made": "Unknown - requires API access"
        }


class OxylabsTemplate(ProxyProviderTemplate):
    """Template for Oxylabs proxy service"""
    
    def __init__(self):
        config = ProxyProviderConfig(
            name="Oxylabs",
            provider_type="static",
            authentication={
                "username_env": "OXYLABS_USERNAME",
                "password_env": "OXYLABS_PASSWORD"
            },
            endpoints={
                "residential": "pr.oxylabs.io:7777",
                "datacenter": "dc.oxylabs.io:8001"
            },
            rate_limits={
                "requests_per_second": 15,
                "concurrent_sessions": 200
            },
            features=["residential", "datacenter", "real_time_crawler"],
            pricing_model="pay_per_request"
        )
        super().__init__(config)
    
    def get_proxy_list(self) -> List[Dict[str, Any]]:
        """Generate Oxylabs proxy endpoints"""
        username = os.getenv(self.config.authentication["username_env"])
        password = os.getenv(self.config.authentication["password_env"])
        
        if not username or not password:
            logger.warning("Oxylabs credentials not found in environment variables")
            return []
        
        proxy_configs = [
            {
                "url": f"http://{username}:{password}@{self.config.endpoints['residential']}",
                "location": "global",
                "type": "residential",
                "provider": "oxylabs"
            },
            {
                "url": f"http://{username}:{password}@{self.config.endpoints['datacenter']}",
                "location": "global", 
                "type": "datacenter",
                "provider": "oxylabs"
            }
        ]
        
        return proxy_configs
    
    def validate_credentials(self) -> bool:
        """Validate Oxylabs credentials"""
        try:
            username = os.getenv(self.config.authentication["username_env"])
            password = os.getenv(self.config.authentication["password_env"])
            
            if not username or not password:
                return False
            
            # Test residential proxy
            proxy_url = f"http://{username}:{password}@{self.config.endpoints['residential']}"
            test_session = requests.Session()
            test_session.proxies = {"http": proxy_url, "https": proxy_url}
            
            response = test_session.get("http://httpbin.org/ip", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Oxylabs credential validation failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get Oxylabs usage statistics"""
        return {
            "provider": "oxylabs",
            "note": "Usage statistics require dashboard access or API integration",
            "requests_made": "Check Oxylabs dashboard",
            "bandwidth_used": "Check Oxylabs dashboard"
        }


class SmartProxyTemplate(ProxyProviderTemplate):
    """Template for SmartProxy service"""
    
    def __init__(self):
        config = ProxyProviderConfig(
            name="SmartProxy",
            provider_type="static",
            authentication={
                "username_env": "SMARTPROXY_USERNAME",
                "password_env": "SMARTPROXY_PASSWORD"
            },
            endpoints={
                "residential": "gate.smartproxy.com:7000",
                "datacenter": "dc.smartproxy.com:10000"
            },
            rate_limits={
                "requests_per_second": 5,
                "concurrent_sessions": 50
            },
            features=["residential", "datacenter", "mobile"],
            pricing_model="pay_per_gb"
        )
        super().__init__(config)
    
    def get_proxy_list(self) -> List[Dict[str, Any]]:
        """Generate SmartProxy endpoints"""
        username = os.getenv(self.config.authentication["username_env"])
        password = os.getenv(self.config.authentication["password_env"])
        
        if not username or not password:
            logger.warning("SmartProxy credentials not found in environment variables")
            return []
        
        proxy_configs = [
            {
                "url": f"http://{username}:{password}@{self.config.endpoints['residential']}",
                "location": "global",
                "type": "residential",
                "provider": "smartproxy"
            },
            {
                "url": f"http://{username}:{password}@{self.config.endpoints['datacenter']}",
                "location": "global",
                "type": "datacenter", 
                "provider": "smartproxy"
            }
        ]
        
        return proxy_configs
    
    def validate_credentials(self) -> bool:
        """Validate SmartProxy credentials"""
        try:
            username = os.getenv(self.config.authentication["username_env"])
            password = os.getenv(self.config.authentication["password_env"])
            
            if not username or not password:
                return False
            
            proxy_url = f"http://{username}:{password}@{self.config.endpoints['residential']}"
            test_session = requests.Session()
            test_session.proxies = {"http": proxy_url, "https": proxy_url}
            
            response = test_session.get("http://httpbin.org/ip", timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"SmartProxy credential validation failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get SmartProxy usage statistics"""
        return {
            "provider": "smartproxy",
            "note": "Usage statistics available in SmartProxy dashboard",
            "requests_made": "Check SmartProxy dashboard",
            "bandwidth_used": "Check SmartProxy dashboard"
        }


class ProxyProviderManager:
    """Manager for all proxy provider templates"""
    
    def __init__(self):
        self.providers = {
            "brightdata": BrightDataTemplate(),
            "oxylabs": OxylabsTemplate(),
            "smartproxy": SmartProxyTemplate()
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider templates"""
        return list(self.providers.keys())
    
    def validate_provider_credentials(self, provider_name: str) -> bool:
        """Validate credentials for a specific provider"""
        if provider_name not in self.providers:
            logger.error(f"Unknown provider: {provider_name}")
            return False
        
        return self.providers[provider_name].validate_credentials()
    
    def generate_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Generate Mimir configuration for a specific provider"""
        if provider_name not in self.providers:
            logger.error(f"Unknown provider: {provider_name}")
            return None
        
        try:
            return self.providers[provider_name].generate_mimir_config()
        except Exception as e:
            logger.error(f"Failed to generate config for {provider_name}: {e}")
            return None
    
    def generate_all_provider_configs(self) -> Dict[str, Any]:
        """Generate Mimir configurations for all providers"""
        configs = {}
        
        for provider_name in self.providers:
            config = self.generate_provider_config(provider_name)
            if config:
                configs[provider_name] = config
        
        return configs
    
    def validate_all_providers(self) -> Dict[str, bool]:
        """Validate credentials for all providers"""
        results = {}
        
        for provider_name in self.providers:
            try:
                results[provider_name] = self.validate_provider_credentials(provider_name)
            except Exception as e:
                logger.error(f"Error validating {provider_name}: {e}")
                results[provider_name] = False
        
        return results
    
    def create_production_config(self, enabled_providers: List[str]) -> Dict[str, Any]:
        """Create production-ready proxy configuration"""
        provider_configs = []
        
        for provider_name in enabled_providers:
            if provider_name in self.providers:
                config = self.generate_provider_config(provider_name)
                if config:
                    config["enabled"] = True  # Enable for production
                    provider_configs.append(config)
        
        return {
            "proxy_settings": {
                "enabled": len(provider_configs) > 0,
                "rotation_strategy": "health_based",
                "health_check_interval": 300,
                "failure_threshold": 3,
                "health_check_timeout": 10,
                "health_check_url": "http://httpbin.org/ip",
                "save_state_file": "proxy_state.json",
                "auto_save_interval": 600,
                "proxy_providers": provider_configs,
                "domain_specific": {
                    "enabled": True,
                    "assignments": {}
                },
                "tenant_isolation": {
                    "enabled": False,
                    "tenant_proxy_pools": {}
                }
            }
        }


def create_provider_setup_guide() -> str:
    """Create setup guide for proxy providers"""
    guide = """
# Proxy Provider Setup Guide

## Supported Providers

### 1. Bright Data (formerly Luminati)
- **Type**: Residential/Datacenter proxies
- **Setup**: 
  1. Sign up at brightdata.com
  2. Create a proxy zone
  3. Set environment variables:
     ```bash
     export BRIGHTDATA_USERNAME="your_username"
     export BRIGHTDATA_PASSWORD="your_password"
     export BRIGHTDATA_ZONE="your_zone"
     ```

### 2. Oxylabs
- **Type**: Residential/Datacenter proxies
- **Setup**:
  1. Sign up at oxylabs.io
  2. Get proxy credentials from dashboard
  3. Set environment variables:
     ```bash
     export OXYLABS_USERNAME="your_username"
     export OXYLABS_PASSWORD="your_password"
     ```

### 3. SmartProxy
- **Type**: Residential/Datacenter proxies
- **Setup**:
  1. Sign up at smartproxy.com
  2. Get endpoint credentials
  3. Set environment variables:
     ```bash
     export SMARTPROXY_USERNAME="your_username"
     export SMARTPROXY_PASSWORD="your_password"
     ```

## Usage

```python
from proxy_provider_templates import ProxyProviderManager

# Initialize manager
manager = ProxyProviderManager()

# Validate credentials
results = manager.validate_all_providers()
print("Validation results:", results)

# Generate configuration for working providers
working_providers = [name for name, valid in results.items() if valid]
config = manager.create_production_config(working_providers)

# Save to config.json
import json
with open('config.json', 'r') as f:
    current_config = json.load(f)

current_config.update(config)

with open('config.json', 'w') as f:
    json.dump(current_config, f, indent=4)
```

## Security Notes

- Never commit credentials to version control
- Use environment variables for all sensitive data
- Rotate credentials regularly
- Monitor usage to detect abuse
- Use HTTPS proxies when possible

## Cost Optimization

- Monitor proxy usage and costs
- Use health-based rotation for better performance
- Set appropriate rate limits
- Consider caching for repeated requests
- Use different proxy types for different use cases
"""
    return guide


if __name__ == "__main__":
    # Example usage
    manager = ProxyProviderManager()
    
    print("Available providers:", manager.get_available_providers())
    
    # Validate all providers
    validation_results = manager.validate_all_providers()
    print("Validation results:", validation_results)
    
    # Generate configs for validated providers
    working_providers = [name for name, valid in validation_results.items() if valid]
    if working_providers:
        config = manager.create_production_config(working_providers)
        print("Generated configuration for providers:", working_providers)
    else:
        print("No working providers found. Please set up credentials.")
        print("\nSetup guide:")
        print(create_provider_setup_guide())