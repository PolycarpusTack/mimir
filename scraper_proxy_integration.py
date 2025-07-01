"""
Enhanced Scraper-Proxy Integration for Mimir
Provides complete integration between scraper.py and proxy system
with proper error handling, configuration validation, and monitoring
"""

import logging
import time
from typing import Dict, Optional, Any, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from proxy_manager import ProxyManager
from proxy_integration import (
    create_domain_specific_session,
    get_global_proxy_manager,
    get_proxy_statistics,
    ProxyEnabledSession
)

logger = logging.getLogger(__name__)


class ScraperProxyManager:
    """
    Enhanced proxy management specifically designed for scraper.py integration.
    Provides centralized proxy management with proper error handling and monitoring.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ScraperProxyManager with configuration validation.
        
        Args:
            config: Full scraper configuration dictionary
        """
        self.config = config
        self.proxy_config = config.get("proxy_settings", {})
        self.enabled = self.proxy_config.get("enabled", False)
        self.proxy_manager: Optional[ProxyManager] = None
        self.initialization_error: Optional[str] = None
        
        # Performance tracking
        self.total_requests = 0
        self.proxy_requests = 0
        self.fallback_requests = 0
        
        # Initialize proxy system
        self._initialize_proxy_system()
    
    def _initialize_proxy_system(self):
        """Initialize the proxy system with proper error handling."""
        if not self.enabled:
            logger.info("Proxy system disabled in configuration")
            return
        
        try:
            # Validate configuration
            validation_result = self._validate_proxy_config()
            if not validation_result[0]:
                self.initialization_error = f"Configuration validation failed: {validation_result[1]}"
                logger.error(self.initialization_error)
                return
            
            # Initialize global proxy manager
            self.proxy_manager = get_global_proxy_manager(self.config)
            
            if self.proxy_manager is None:
                self.initialization_error = "Failed to create proxy manager"
                logger.error(self.initialization_error)
                return
            
            # Verify at least one proxy is available
            if not self.proxy_manager.proxies:
                self.initialization_error = "No proxies configured"
                logger.warning(self.initialization_error)
                return
            
            # Run initial health check
            self.proxy_manager.check_all_proxy_health()
            
            logger.info(f"Proxy system initialized successfully with {len(self.proxy_manager.proxies)} proxies")
            
        except Exception as e:
            self.initialization_error = f"Proxy system initialization failed: {str(e)}"
            logger.error(self.initialization_error, exc_info=True)
    
    def _validate_proxy_config(self) -> Tuple[bool, str]:
        """
        Validate proxy configuration for common issues.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(self.proxy_config, dict):
            return False, "proxy_settings must be a dictionary"
        
        # Check required fields
        required_fields = ["enabled", "rotation_strategy", "proxy_providers"]
        for field in required_fields:
            if field not in self.proxy_config:
                return False, f"Missing required field: {field}"
        
        # Validate rotation strategy
        valid_strategies = ["round_robin", "random", "health_based"]
        strategy = self.proxy_config.get("rotation_strategy")
        if strategy not in valid_strategies:
            return False, f"Invalid rotation_strategy: {strategy}. Must be one of {valid_strategies}"
        
        # Validate proxy providers
        providers = self.proxy_config.get("proxy_providers", [])
        if not isinstance(providers, list):
            return False, "proxy_providers must be a list"
        
        if not providers:
            return False, "At least one proxy provider must be configured"
        
        # Check if any provider is enabled
        enabled_providers = [p for p in providers if p.get("enabled", False)]
        if not enabled_providers:
            return False, "At least one proxy provider must be enabled"
        
        # Validate each provider
        for i, provider in enumerate(providers):
            if not isinstance(provider, dict):
                return False, f"Provider {i} must be a dictionary"
            
            if "type" not in provider:
                return False, f"Provider {i} missing 'type' field"
            
            provider_type = provider["type"]
            if provider_type == "static_list":
                proxies = provider.get("proxies", [])
                if not proxies:
                    return False, f"Static provider {i} has no proxies configured"
            elif provider_type == "file":
                file_path = provider.get("file_path")
                if not file_path:
                    return False, f"File provider {i} missing 'file_path'"
            elif provider_type == "api":
                api_url = provider.get("api_url")
                if not api_url:
                    return False, f"API provider {i} missing 'api_url'"
            else:
                return False, f"Provider {i} has invalid type: {provider_type}"
        
        return True, "Configuration is valid"
    
    def fetch_url_with_proxy(self, url: str, retries: int = 0) -> Optional[str]:
        """
        Enhanced fetch_url with domain-specific proxy support and proper error handling.
        
        Args:
            url: URL to fetch
            retries: Number of retries attempted
            
        Returns:
            Response text or None
        """
        max_retries = self.config.get("max_retries", 3)
        retry_delay = self.config.get("retry_delay_seconds", 5)
        request_delay = self.config.get("default_request_delay_seconds", 2)
        
        self.total_requests += 1
        
        try:
            # Create domain-specific session (with or without proxy)
            if self.enabled and self.proxy_manager and not self.initialization_error:
                session = create_domain_specific_session(self.config, url)
                self.proxy_requests += 1
                
                # Log domain-specific proxy usage
                domain = urlparse(url).netloc
                if isinstance(session, ProxyEnabledSession):
                    proxy_url = session.proxy_manager.get_proxy(domain=domain)
                    if proxy_url:
                        logger.debug(f"Using domain-specific proxy for {domain}: {proxy_url}")
                    else:
                        logger.debug(f"No proxy available for {domain}, proceeding without proxy")
            else:
                # Fallback to regular session
                session = self._create_fallback_session()
                self.fallback_requests += 1
                
                if self.enabled:
                    logger.warning(f"Using fallback session for {url}: {self.initialization_error or 'Proxy system unavailable'}")
            
            # Make the request
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            logger.debug(f"Successfully fetched: {url}")
            time.sleep(request_delay)
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            if retries < max_retries:
                logger.warning(f"Fetch error for {url}, attempt {retries + 1}/{max_retries}: {e}")
                time.sleep(retry_delay)
                return self.fetch_url_with_proxy(url, retries + 1)
            
            logger.error(f"Definitively failed to fetch {url}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
            return None
    
    def _create_fallback_session(self) -> requests.Session:
        """Create a fallback session when proxy system is unavailable."""
        session = requests.Session()
        
        # Apply same retry strategy as proxy-enabled sessions
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Apply user agent
        user_agent = self.config.get("user_agent", "Mimir-Scraper/1.0")
        session.headers.update({"User-Agent": user_agent})
        
        return session
    
    def get_proxy_status(self) -> Dict[str, Any]:
        """
        Get comprehensive proxy system status for monitoring.
        
        Returns:
            Dictionary containing proxy system status and statistics
        """
        status = {
            "enabled": self.enabled,
            "initialization_error": self.initialization_error,
            "total_requests": self.total_requests,
            "proxy_requests": self.proxy_requests,
            "fallback_requests": self.fallback_requests,
            "proxy_usage_rate": 0.0
        }
        
        if self.total_requests > 0:
            status["proxy_usage_rate"] = (self.proxy_requests / self.total_requests) * 100
        
        if self.proxy_manager:
            # Add detailed proxy statistics
            proxy_stats = get_proxy_statistics(self.config)
            status.update({
                "proxy_system": proxy_stats,
                "available_proxies": len([p for p in self.proxy_manager.proxies.values() if p.is_healthy]),
                "blacklisted_proxies": len(self.proxy_manager.blacklisted_proxies)
            })
        
        return status
    
    def trigger_health_check(self):
        """Manually trigger proxy health check."""
        if self.proxy_manager:
            logger.info("Triggering manual proxy health check...")
            self.proxy_manager.check_all_proxy_health()
        else:
            logger.warning("Cannot trigger health check: proxy manager not available")
    
    def reset_proxy(self, proxy_url: str):
        """Reset a specific proxy's metrics and blacklist status."""
        if self.proxy_manager:
            self.proxy_manager.reset_proxy(proxy_url)
            logger.info(f"Reset proxy: {proxy_url}")
        else:
            logger.warning("Cannot reset proxy: proxy manager not available")
    
    def save_proxy_state(self):
        """Save current proxy state to disk."""
        if self.proxy_manager:
            state_file = self.proxy_config.get("save_state_file", "proxy_state.json")
            self.proxy_manager.save_proxy_state(state_file)
            logger.info(f"Saved proxy state to {state_file}")
        else:
            logger.warning("Cannot save proxy state: proxy manager not available")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for logging and monitoring."""
        status = self.get_proxy_status()
        
        summary = {
            "proxy_enabled": status["enabled"],
            "total_requests": status["total_requests"],
            "proxy_usage_rate": status["proxy_usage_rate"],
            "has_errors": bool(status["initialization_error"])
        }
        
        if status.get("proxy_system"):
            proxy_stats = status["proxy_system"]
            summary.update({
                "active_proxies": proxy_stats.get("active_proxies", 0),
                "overall_success_rate": proxy_stats.get("overall_success_rate", 0)
            })
        
        return summary


# Global instance for use throughout scraper
_scraper_proxy_manager: Optional[ScraperProxyManager] = None


def initialize_scraper_proxy_system(config: Dict) -> ScraperProxyManager:
    """
    Initialize the global scraper proxy system.
    
    Args:
        config: Full scraper configuration
        
    Returns:
        ScraperProxyManager instance
    """
    global _scraper_proxy_manager
    
    if _scraper_proxy_manager is None:
        _scraper_proxy_manager = ScraperProxyManager(config)
        logger.info("Initialized scraper proxy system")
    
    return _scraper_proxy_manager


def get_scraper_proxy_manager() -> Optional[ScraperProxyManager]:
    """Get the global scraper proxy manager instance."""
    return _scraper_proxy_manager


def enhanced_fetch_url(url: str, config: Dict, retries: int = 0) -> Optional[str]:
    """
    Enhanced fetch_url function with full proxy support.
    Drop-in replacement for the original fetch_url in scraper.py.
    
    Args:
        url: URL to fetch
        config: Configuration dictionary (for compatibility)
        retries: Number of retries attempted
        
    Returns:
        Response text or None
    """
    proxy_manager = get_scraper_proxy_manager()
    
    if proxy_manager is None:
        # Initialize if not done yet
        proxy_manager = initialize_scraper_proxy_system(config)
    
    return proxy_manager.fetch_url_with_proxy(url, retries)


def log_proxy_performance_summary():
    """Log proxy performance summary for monitoring."""
    proxy_manager = get_scraper_proxy_manager()
    
    if proxy_manager:
        summary = proxy_manager.get_performance_summary()
        logger.info(f"Proxy Performance Summary: {summary}")
    else:
        logger.info("Proxy system not initialized")


def get_proxy_integration_status() -> Dict[str, Any]:
    """
    Get complete proxy integration status for debugging and monitoring.
    
    Returns:
        Dictionary with comprehensive status information
    """
    proxy_manager = get_scraper_proxy_manager()
    
    if proxy_manager is None:
        return {
            "initialized": False,
            "error": "Proxy manager not initialized"
        }
    
    status = proxy_manager.get_proxy_status()
    status["initialized"] = True
    
    return status