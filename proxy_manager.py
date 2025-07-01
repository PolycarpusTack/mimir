"""
Proxy Manager for Mimir News Scraper
Handles proxy rotation, health monitoring, and failover management
"""

import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import asyncio
import threading
from dataclasses import dataclass, asdict
from enum import Enum

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ProxyStatus(Enum):
    """Proxy status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BLACKLISTED = "blacklisted"
    TESTING = "testing"


@dataclass
class ProxyMetrics:
    """Proxy performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if data['last_success']:
            data['last_success'] = data['last_success'].isoformat()
        if data['last_failure']:
            data['last_failure'] = data['last_failure'].isoformat()
        return data


@dataclass
class ProxyInfo:
    """Proxy configuration and status"""
    url: str
    status: ProxyStatus = ProxyStatus.INACTIVE
    metrics: ProxyMetrics = None
    provider: str = "unknown"
    geographical_location: str = "unknown"
    last_health_check: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ProxyMetrics()
        if self.tags is None:
            self.tags = []
    
    @property
    def is_healthy(self) -> bool:
        """Check if proxy is considered healthy"""
        return (
            self.status == ProxyStatus.ACTIVE and
            self.metrics.consecutive_failures < 3 and
            self.metrics.success_rate >= 80.0
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['metrics'] = self.metrics.to_dict()
        if data['last_health_check']:
            data['last_health_check'] = data['last_health_check'].isoformat()
        return data


class ProxyManager:
    """
    Manages proxy rotation, health monitoring, and failover for web scraping.
    Follows existing Mimir patterns for configuration and error handling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ProxyManager with configuration.
        
        Args:
            config: Proxy configuration dictionary from config.json
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.proxies: Dict[str, ProxyInfo] = {}
        self.rotation_index = 0
        self.blacklisted_proxies = set()
        self.lock = threading.Lock()
        
        # Configuration settings
        self.rotation_strategy = config.get("rotation_strategy", "round_robin")
        self.health_check_interval = config.get("health_check_interval", 300)  # 5 minutes
        self.failure_threshold = config.get("failure_threshold", 3)
        self.health_check_timeout = config.get("health_check_timeout", 10)
        self.health_check_url = config.get("health_check_url", "http://httpbin.org/ip")
        
        # Performance tracking
        self.total_requests = 0
        self.total_failures = 0
        self.last_health_check = None
        
        # Load proxy providers
        self._load_proxy_providers()
        
        # Start background health checker if enabled
        if self.enabled and self.proxies:
            self._start_health_monitor()
            logger.info(f"ProxyManager initialized with {len(self.proxies)} proxies")
        elif self.enabled:
            logger.warning("ProxyManager enabled but no proxies configured")
        else:
            logger.info("ProxyManager disabled")
    
    def _load_proxy_providers(self):
        """Load proxies from configured providers"""
        providers = self.config.get("proxy_providers", [])
        
        for provider_config in providers:
            provider_type = provider_config.get("type", "")
            
            if provider_type == "static_list":
                self._load_static_proxies(provider_config)
            elif provider_type == "file":
                self._load_file_proxies(provider_config)
            elif provider_type == "api":
                self._load_api_proxies(provider_config)
            else:
                logger.warning(f"Unknown proxy provider type: {provider_type}")
    
    def _load_static_proxies(self, provider_config: Dict):
        """Load proxies from static list"""
        proxy_list = provider_config.get("proxies", [])
        provider_name = provider_config.get("name", "static")
        
        for proxy_url in proxy_list:
            proxy_info = ProxyInfo(
                url=proxy_url,
                provider=provider_name,
                status=ProxyStatus.INACTIVE
            )
            self.proxies[proxy_url] = proxy_info
            logger.debug(f"Added static proxy: {proxy_url}")
    
    def _load_file_proxies(self, provider_config: Dict):
        """Load proxies from file"""
        file_path = provider_config.get("file_path", "")
        if not file_path:
            logger.error("File proxy provider missing file_path")
            return
        
        try:
            with open(file_path, 'r') as f:
                proxy_data = json.load(f)
            
            provider_name = provider_config.get("name", "file")
            
            for proxy_entry in proxy_data.get("proxies", []):
                if isinstance(proxy_entry, str):
                    proxy_url = proxy_entry
                    proxy_info = ProxyInfo(url=proxy_url, provider=provider_name)
                else:
                    proxy_info = ProxyInfo(
                        url=proxy_entry.get("url", ""),
                        provider=provider_name,
                        geographical_location=proxy_entry.get("location", "unknown"),
                        tags=proxy_entry.get("tags", [])
                    )
                
                if proxy_info.url:
                    self.proxies[proxy_info.url] = proxy_info
                    logger.debug(f"Added file proxy: {proxy_info.url}")
                    
        except Exception as e:
            logger.error(f"Failed to load proxies from file {file_path}: {e}")
    
    def _load_api_proxies(self, provider_config: Dict):
        """Load proxies from API endpoint"""
        api_url = provider_config.get("api_url", "")
        if not api_url:
            logger.error("API proxy provider missing api_url")
            return
        
        try:
            headers = provider_config.get("headers", {})
            response = requests.get(api_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            proxy_data = response.json()
            provider_name = provider_config.get("name", "api")
            
            for proxy_entry in proxy_data.get("proxies", []):
                proxy_info = ProxyInfo(
                    url=proxy_entry.get("url", ""),
                    provider=provider_name,
                    geographical_location=proxy_entry.get("location", "unknown"),
                    tags=proxy_entry.get("tags", [])
                )
                
                if proxy_info.url:
                    self.proxies[proxy_info.url] = proxy_info
                    logger.debug(f"Added API proxy: {proxy_info.url}")
                    
        except Exception as e:
            logger.error(f"Failed to load proxies from API {api_url}: {e}")
    
    def get_proxy(self, domain: str = None, tags: List[str] = None) -> Optional[str]:
        """
        Get next proxy based on rotation strategy.
        
        Args:
            domain: Optional domain for domain-specific proxy assignment
            tags: Optional tags to filter proxies
            
        Returns:
            Proxy URL string or None if no proxy available
        """
        if not self.enabled or not self.proxies:
            return None
        
        with self.lock:
            available_proxies = self._get_available_proxies(tags)
            
            if not available_proxies:
                logger.warning("No available proxies found")
                return None
            
            if self.rotation_strategy == "round_robin":
                proxy_url = self._get_round_robin_proxy(available_proxies)
            elif self.rotation_strategy == "random":
                proxy_url = random.choice(available_proxies)
            elif self.rotation_strategy == "health_based":
                proxy_url = self._get_health_based_proxy(available_proxies)
            else:
                # Default to round robin
                proxy_url = self._get_round_robin_proxy(available_proxies)
            
            logger.debug(f"Selected proxy: {proxy_url}")
            return proxy_url
    
    def _get_available_proxies(self, tags: List[str] = None) -> List[str]:
        """Get list of available proxy URLs"""
        available = []
        
        for url, proxy_info in self.proxies.items():
            # Skip blacklisted proxies
            if url in self.blacklisted_proxies:
                continue
            
            # Skip unhealthy proxies
            if not proxy_info.is_healthy:
                continue
            
            # Filter by tags if specified
            if tags and not any(tag in proxy_info.tags for tag in tags):
                continue
            
            available.append(url)
        
        return available
    
    def _get_round_robin_proxy(self, available_proxies: List[str]) -> str:
        """Get proxy using round-robin strategy"""
        if self.rotation_index >= len(available_proxies):
            self.rotation_index = 0
        
        proxy_url = available_proxies[self.rotation_index]
        self.rotation_index += 1
        return proxy_url
    
    def _get_health_based_proxy(self, available_proxies: List[str]) -> str:
        """Get proxy based on health metrics (best performing first)"""
        # Sort by success rate and response time
        def proxy_score(url):
            proxy_info = self.proxies[url]
            success_rate = proxy_info.metrics.success_rate
            response_time = proxy_info.metrics.average_response_time
            
            # Higher success rate is better, lower response time is better
            return success_rate - (response_time / 1000)  # Normalize response time
        
        sorted_proxies = sorted(available_proxies, key=proxy_score, reverse=True)
        return sorted_proxies[0]
    
    def record_proxy_result(self, proxy_url: str, success: bool, response_time: float = 0.0):
        """
        Record the result of using a proxy.
        
        Args:
            proxy_url: The proxy URL that was used
            success: Whether the request was successful
            response_time: Response time in milliseconds
        """
        if not proxy_url or proxy_url not in self.proxies:
            return
        
        with self.lock:
            proxy_info = self.proxies[proxy_url]
            metrics = proxy_info.metrics
            
            # Update metrics
            metrics.total_requests += 1
            self.total_requests += 1
            
            if success:
                metrics.successful_requests += 1
                metrics.consecutive_failures = 0
                metrics.last_success = datetime.now()
                
                # Update average response time
                if metrics.average_response_time == 0:
                    metrics.average_response_time = response_time
                else:
                    # Exponential moving average
                    metrics.average_response_time = (
                        0.7 * metrics.average_response_time + 0.3 * response_time
                    )
            else:
                metrics.failed_requests += 1
                metrics.consecutive_failures += 1
                metrics.last_failure = datetime.now()
                self.total_failures += 1
                
                # Blacklist proxy if it exceeds failure threshold
                if metrics.consecutive_failures >= self.failure_threshold:
                    self._blacklist_proxy(proxy_url, "consecutive_failures")
    
    def _blacklist_proxy(self, proxy_url: str, reason: str):
        """Blacklist a proxy due to failures"""
        self.blacklisted_proxies.add(proxy_url)
        self.proxies[proxy_url].status = ProxyStatus.BLACKLISTED
        logger.warning(f"Blacklisted proxy {proxy_url}: {reason}")
    
    def check_proxy_health(self, proxy_url: str) -> bool:
        """
        Check if a single proxy is healthy.
        
        Args:
            proxy_url: Proxy URL to check
            
        Returns:
            True if proxy is healthy, False otherwise
        """
        if proxy_url not in self.proxies:
            return False
        
        try:
            start_time = time.time()
            
            # Configure session with proxy
            session = requests.Session()
            session.proxies = {"http": proxy_url, "https": proxy_url}
            
            # Set up retry strategy
            retry_strategy = Retry(
                total=1,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Make health check request
            response = session.get(
                self.health_check_url,
                timeout=self.health_check_timeout,
                headers={'User-Agent': 'Mimir-HealthCheck/1.0'}
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                # Update proxy status
                proxy_info = self.proxies[proxy_url]
                proxy_info.status = ProxyStatus.ACTIVE
                proxy_info.last_health_check = datetime.now()
                
                # Record successful health check
                self.record_proxy_result(proxy_url, True, response_time)
                
                logger.debug(f"Proxy {proxy_url} health check passed ({response_time:.2f}ms)")
                return True
            else:
                logger.warning(f"Proxy {proxy_url} health check failed: HTTP {response.status_code}")
                self.record_proxy_result(proxy_url, False)
                return False
                
        except Exception as e:
            logger.warning(f"Proxy {proxy_url} health check error: {e}")
            self.record_proxy_result(proxy_url, False)
            return False
    
    def check_all_proxy_health(self):
        """Check health of all configured proxies"""
        if not self.enabled:
            return
        
        logger.info("Starting proxy health check for all proxies")
        healthy_count = 0
        
        for proxy_url in self.proxies.keys():
            if self.check_proxy_health(proxy_url):
                healthy_count += 1
        
        self.last_health_check = datetime.now()
        logger.info(f"Proxy health check completed: {healthy_count}/{len(self.proxies)} healthy")
    
    def _start_health_monitor(self):
        """Start background thread for periodic health monitoring"""
        def health_monitor():
            while True:
                try:
                    time.sleep(self.health_check_interval)
                    self.check_all_proxy_health()
                except Exception as e:
                    logger.error(f"Error in proxy health monitor: {e}")
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
        logger.info("Proxy health monitor started")
    
    def get_proxy_statistics(self) -> Dict:
        """Get comprehensive proxy statistics"""
        if not self.enabled:
            return {"enabled": False}
        
        active_proxies = sum(1 for p in self.proxies.values() if p.status == ProxyStatus.ACTIVE)
        blacklisted_proxies = len(self.blacklisted_proxies)
        
        # Calculate overall success rate
        overall_success_rate = 0.0
        if self.total_requests > 0:
            overall_success_rate = ((self.total_requests - self.total_failures) / self.total_requests) * 100
        
        # Get top performing proxies
        proxy_performance = []
        for url, proxy_info in self.proxies.items():
            proxy_performance.append({
                "url": url,
                "status": proxy_info.status.value,
                "success_rate": proxy_info.metrics.success_rate,
                "avg_response_time": proxy_info.metrics.average_response_time,
                "total_requests": proxy_info.metrics.total_requests,
                "provider": proxy_info.provider
            })
        
        # Sort by success rate
        proxy_performance.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return {
            "enabled": True,
            "total_proxies": len(self.proxies),
            "active_proxies": active_proxies,
            "blacklisted_proxies": blacklisted_proxies,
            "rotation_strategy": self.rotation_strategy,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "overall_success_rate": overall_success_rate,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "top_proxies": proxy_performance[:10]  # Top 10 performing proxies
        }
    
    def reset_proxy(self, proxy_url: str):
        """Reset a proxy's metrics and remove from blacklist"""
        if proxy_url in self.proxies:
            self.blacklisted_proxies.discard(proxy_url)
            self.proxies[proxy_url].status = ProxyStatus.INACTIVE
            self.proxies[proxy_url].metrics = ProxyMetrics()
            logger.info(f"Reset proxy: {proxy_url}")
    
    def save_proxy_state(self, file_path: str = "proxy_state.json"):
        """Save current proxy state to file"""
        try:
            state_data = {
                "proxies": {url: info.to_dict() for url, info in self.proxies.items()},
                "blacklisted_proxies": list(self.blacklisted_proxies),
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "last_saved": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Proxy state saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save proxy state: {e}")
    
    def load_proxy_state(self, file_path: str = "proxy_state.json"):
        """Load proxy state from file"""
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
            
            # Restore proxy metrics
            for url, proxy_data in state_data.get("proxies", {}).items():
                if url in self.proxies:
                    proxy_info = self.proxies[url]
                    
                    # Restore metrics
                    metrics_data = proxy_data.get("metrics", {})
                    proxy_info.metrics.total_requests = metrics_data.get("total_requests", 0)
                    proxy_info.metrics.successful_requests = metrics_data.get("successful_requests", 0)
                    proxy_info.metrics.failed_requests = metrics_data.get("failed_requests", 0)
                    proxy_info.metrics.average_response_time = metrics_data.get("average_response_time", 0.0)
                    proxy_info.metrics.consecutive_failures = metrics_data.get("consecutive_failures", 0)
                    
                    # Parse datetime strings
                    last_success = metrics_data.get("last_success")
                    if last_success:
                        proxy_info.metrics.last_success = datetime.fromisoformat(last_success)
                    
                    last_failure = metrics_data.get("last_failure")
                    if last_failure:
                        proxy_info.metrics.last_failure = datetime.fromisoformat(last_failure)
                    
                    # Restore status
                    status_str = proxy_data.get("status", "inactive")
                    proxy_info.status = ProxyStatus(status_str)
            
            # Restore blacklisted proxies
            self.blacklisted_proxies = set(state_data.get("blacklisted_proxies", []))
            
            # Restore global stats
            self.total_requests = state_data.get("total_requests", 0)
            self.total_failures = state_data.get("total_failures", 0)
            
            logger.info(f"Proxy state loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load proxy state: {e}")


def create_proxy_manager(config: Dict) -> ProxyManager:
    """
    Factory function to create ProxyManager instance.
    Follows existing Mimir patterns for service creation.
    """
    return ProxyManager(config)