"""
Proxy Integration for Mimir News Scraper
Integrates ProxyManager with requests library and Playwright
"""

import logging
import time
from typing import Dict, Optional, Any
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from proxy_manager import ProxyManager, create_proxy_manager

logger = logging.getLogger(__name__)


class ProxyEnabledSession(requests.Session):
    """
    Enhanced requests session with automatic proxy rotation.
    Follows existing Mimir patterns for error handling and logging.
    """
    
    def __init__(self, proxy_manager: ProxyManager, domain: str = None):
        super().__init__()
        self.proxy_manager = proxy_manager
        self.domain = domain
        self.current_proxy = None
        
        # Configure retry strategy (same as existing scraper)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.mount("http://", adapter)
        self.mount("https://", adapter)
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Override request method to add proxy rotation.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            Response object
        """
        # Get proxy if manager is enabled
        if self.proxy_manager.enabled:
            proxy_url = self.proxy_manager.get_proxy(domain=self.domain)
            if proxy_url:
                self.current_proxy = proxy_url
                kwargs['proxies'] = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                logger.debug(f"Using proxy {proxy_url} for {url}")
            else:
                logger.warning("No proxy available, proceeding without proxy")
                self.current_proxy = None
        
        start_time = time.time()
        
        try:
            # Make the request
            response = super().request(method, url, **kwargs)
            
            # Record successful proxy usage
            if self.current_proxy:
                response_time = (time.time() - start_time) * 1000
                self.proxy_manager.record_proxy_result(
                    self.current_proxy, 
                    True, 
                    response_time
                )
            
            return response
            
        except Exception as e:
            # Record failed proxy usage
            if self.current_proxy:
                self.proxy_manager.record_proxy_result(self.current_proxy, False)
            
            logger.error(f"Request failed for {url}: {e}")
            raise


class ProxyPlaywrightIntegration:
    """
    Integration class to add proxy support to Playwright scraping.
    """
    
    def __init__(self, proxy_manager: ProxyManager):
        self.proxy_manager = proxy_manager
    
    def get_proxy_config_for_playwright(self, domain: str = None) -> Optional[Dict[str, Any]]:
        """
        Get proxy configuration for Playwright browser context.
        
        Args:
            domain: Optional domain for domain-specific proxy assignment
            
        Returns:
            Proxy configuration dict for Playwright or None
        """
        if not self.proxy_manager.enabled:
            return None
        
        proxy_url = self.proxy_manager.get_proxy(domain=domain)
        if not proxy_url:
            return None
        
        # Parse proxy URL
        parsed = urlparse(proxy_url)
        
        proxy_config = {
            'server': proxy_url,
            'bypass': None  # Can be configured for local addresses
        }
        
        # Add authentication if present in URL
        if parsed.username and parsed.password:
            proxy_config['username'] = parsed.username
            proxy_config['password'] = parsed.password
        
        logger.debug(f"Playwright proxy config: {proxy_config['server']}")
        return proxy_config
    
    def record_playwright_proxy_result(self, proxy_url: str, success: bool, response_time: float = 0.0):
        """
        Record the result of a Playwright request using a proxy.
        
        Args:
            proxy_url: The proxy URL that was used
            success: Whether the request was successful
            response_time: Response time in milliseconds
        """
        self.proxy_manager.record_proxy_result(proxy_url, success, response_time)


def create_proxy_enabled_session(config: Dict, domain: str = None) -> requests.Session:
    """
    Factory function to create a proxy-enabled requests session.
    
    Args:
        config: Configuration dictionary containing proxy settings
        domain: Optional domain for domain-specific proxy assignment
        
    Returns:
        ProxyEnabledSession or regular Session based on proxy configuration
    """
    proxy_config = config.get("proxy_settings", {})
    
    if not proxy_config.get("enabled", False):
        # Return regular session if proxies are disabled
        session = requests.Session()
        
        # Still configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    # Create proxy manager and proxy-enabled session
    proxy_manager = create_proxy_manager(proxy_config)
    return ProxyEnabledSession(proxy_manager, domain)


def update_scraper_session_with_proxy(config: Dict) -> requests.Session:
    """
    Update the global scraper session with proxy support.
    Maintains compatibility with existing scraper.py patterns.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated session with proxy support
    """
    session = create_proxy_enabled_session(config)
    
    # Apply existing user agent (maintaining compatibility)
    user_agent = config.get("user_agent", "Mimir-Scraper/1.0")
    session.headers.update({"User-Agent": user_agent})
    
    return session


def integrate_proxy_with_playwright_scraper(playwright_config: Dict, site_config: Dict, proxy_manager: ProxyManager) -> Dict:
    """
    Integrate proxy settings into Playwright configuration.
    
    Args:
        playwright_config: Playwright configuration dictionary
        site_config: Site configuration dictionary
        proxy_manager: ProxyManager instance
        
    Returns:
        Updated playwright_config with proxy settings
    """
    if not proxy_manager.enabled:
        return playwright_config
    
    # Get domain from site config for domain-specific proxy assignment
    domain = None
    site_url = site_config.get('url', '')
    if site_url:
        parsed_url = urlparse(site_url)
        domain = parsed_url.netloc
    
    # Create proxy integration
    proxy_integration = ProxyPlaywrightIntegration(proxy_manager)
    proxy_config = proxy_integration.get_proxy_config_for_playwright(domain)
    
    if proxy_config:
        # Add proxy to browser context defaults
        if 'defaults' not in playwright_config:
            playwright_config['defaults'] = {}
        
        playwright_config['defaults']['proxy'] = proxy_config
        logger.info(f"Added proxy {proxy_config['server']} to Playwright config for {domain}")
    
    return playwright_config


def create_domain_specific_session(config: Dict, url: str) -> requests.Session:
    """
    Create a session with domain-specific proxy assignment.
    
    Args:
        config: Configuration dictionary
        url: URL to extract domain from
        
    Returns:
        Session configured for the specific domain
    """
    domain = urlparse(url).netloc
    return create_proxy_enabled_session(config, domain)


# Global proxy manager instance for reuse
_global_proxy_manager: Optional[ProxyManager] = None


def get_global_proxy_manager(config: Dict) -> Optional[ProxyManager]:
    """
    Get or create global proxy manager instance.
    Follows singleton pattern for efficient resource usage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ProxyManager instance or None if disabled
    """
    global _global_proxy_manager
    
    proxy_config = config.get("proxy_settings", {})
    
    if not proxy_config.get("enabled", False):
        return None
    
    if _global_proxy_manager is None:
        _global_proxy_manager = create_proxy_manager(proxy_config)
        logger.info("Created global proxy manager")
    
    return _global_proxy_manager


def reset_global_proxy_manager():
    """Reset global proxy manager instance (useful for testing)"""
    global _global_proxy_manager
    _global_proxy_manager = None


# Compatibility functions for existing scraper integration

def proxy_aware_fetch_url(url: str, config: Dict, retries: int = 0) -> Optional[str]:
    """
    Proxy-aware version of fetch_url function.
    Drop-in replacement for existing fetch_url in scraper.py.
    
    Args:
        url: URL to fetch
        config: Configuration dictionary
        retries: Number of retries attempted
        
    Returns:
        Response text or None
    """
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay_seconds", 5)
    request_delay = config.get("default_request_delay_seconds", 2)
    
    try:
        session = create_domain_specific_session(config, url)
        
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        logger.debug(f"Successfully fetched: {url}")
        time.sleep(request_delay)
        
        return response.text
        
    except requests.exceptions.RequestException as e:
        if retries < max_retries:
            logger.warning(f"Fetch error for {url}, attempt {retries + 1}/{max_retries}: {e}")
            time.sleep(retry_delay)
            return proxy_aware_fetch_url(url, config, retries + 1)
        
        logger.error(f"Definitively failed to fetch {url}: {e}")
        return None


def get_proxy_statistics(config: Dict) -> Dict:
    """
    Get proxy statistics for monitoring and debugging.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing proxy statistics
    """
    proxy_manager = get_global_proxy_manager(config)
    
    if proxy_manager is None:
        return {"enabled": False, "message": "Proxy system is disabled"}
    
    return proxy_manager.get_proxy_statistics()


def health_check_all_proxies(config: Dict):
    """
    Manually trigger health check for all proxies.
    
    Args:
        config: Configuration dictionary
    """
    proxy_manager = get_global_proxy_manager(config)
    
    if proxy_manager is None:
        logger.info("Proxy system is disabled, skipping health check")
        return
    
    proxy_manager.check_all_proxy_health()