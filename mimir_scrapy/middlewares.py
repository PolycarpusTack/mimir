"""
Scrapy middlewares for Mimir news scraper
Handles user agents, proxy rotation, rate limiting, and error handling
"""

import random
import time
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import scrapy
from scrapy import signals
from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import HtmlResponse
from scrapy.spiders import Spider
from w3lib.http import basic_auth_header


class UserAgentMiddleware:
    """Rotates user agents to avoid detection"""
    
    def __init__(self, user_agents: Optional[List[str]] = None):
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mimir Broadcast Intelligence Bot 1.0 (+https://mimir.ai/bot)'
        ]
    
    @classmethod
    def from_crawler(cls, crawler):
        user_agents = crawler.settings.getlist('USER_AGENT_LIST')
        if not user_agents:
            user_agents = None
        return cls(user_agents)
    
    def process_request(self, request, spider):
        """Set random user agent for each request"""
        ua = random.choice(self.user_agents)
        request.headers['User-Agent'] = ua
        return None


class ProxyMiddleware:
    """Handles proxy rotation and authentication"""
    
    def __init__(self, proxy_settings: Optional[Dict] = None):
        if not proxy_settings or not proxy_settings.get('enabled', False):
            raise NotConfigured('Proxy middleware disabled')
        
        self.proxy_settings = proxy_settings
        self.proxy_list = self._load_proxy_list()
        self.current_proxy_index = 0
        self.proxy_failures = {}
        self.max_failures = proxy_settings.get('max_failures_per_proxy', 3)
        self.rotation_strategy = proxy_settings.get('rotation_strategy', 'round_robin')
        
    @classmethod
    def from_crawler(cls, crawler):
        proxy_settings = crawler.settings.get('PROXY_SETTINGS', {})
        return cls(proxy_settings)
    
    def _load_proxy_list(self) -> List[Dict]:
        """Load proxy list from configuration"""
        proxies = []
        
        for provider in self.proxy_settings.get('proxy_providers', []):
            if provider.get('type') == 'static_list':
                for proxy_url in provider.get('proxies', []):
                    proxies.append({
                        'url': proxy_url,
                        'provider': provider.get('name', 'static'),
                        'auth': provider.get('auth'),
                        'failures': 0
                    })
        
        return proxies
    
    def process_request(self, request, spider):
        """Set proxy for request"""
        if not self.proxy_list:
            return None
        
        proxy = self._get_next_proxy()
        if not proxy:
            spider.logger.warning("No available proxies")
            return None
        
        request.meta['proxy'] = proxy['url']
        
        # Set authentication if required
        if proxy.get('auth'):
            auth = proxy['auth']
            if isinstance(auth, dict):
                username = auth.get('username')
                password = auth.get('password')
                if username and password:
                    request.headers['Proxy-Authorization'] = basic_auth_header(username, password)
        
        request.meta['proxy_info'] = proxy
        spider.logger.debug(f"Using proxy: {proxy['url']}")
        return None
    
    def process_response(self, request, response, spider):
        """Handle proxy response"""
        proxy_info = request.meta.get('proxy_info')
        if proxy_info and response.status == 200:
            # Reset failure count on success
            proxy_info['failures'] = 0
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Handle proxy failures"""
        proxy_info = request.meta.get('proxy_info')
        if proxy_info:
            proxy_info['failures'] += 1
            spider.logger.warning(
                f"Proxy failure for {proxy_info['url']}: {exception} "
                f"(failures: {proxy_info['failures']})"
            )
            
            # Blacklist proxy if too many failures
            if proxy_info['failures'] >= self.max_failures:
                spider.logger.error(f"Blacklisting proxy: {proxy_info['url']}")
        
        return None
    
    def _get_next_proxy(self) -> Optional[Dict]:
        """Get next available proxy based on rotation strategy"""
        available_proxies = [p for p in self.proxy_list if p['failures'] < self.max_failures]
        
        if not available_proxies:
            return None
        
        if self.rotation_strategy == 'round_robin':
            proxy = available_proxies[self.current_proxy_index % len(available_proxies)]
            self.current_proxy_index += 1
        elif self.rotation_strategy == 'random':
            proxy = random.choice(available_proxies)
        else:
            proxy = available_proxies[0]
        
        return proxy


class RateLimitMiddleware:
    """Implements smart rate limiting per domain"""
    
    def __init__(self, default_delay: float = 2.0, domain_delays: Optional[Dict[str, float]] = None):
        self.default_delay = default_delay
        self.domain_delays = domain_delays or {}
        self.last_request_times = {}
        
    @classmethod
    def from_crawler(cls, crawler):
        default_delay = crawler.settings.getfloat('DOWNLOAD_DELAY', 2.0)
        domain_delays = crawler.settings.get('DOMAIN_DELAYS', {})
        return cls(default_delay, domain_delays)
    
    def process_request(self, request, spider):
        """Apply rate limiting before request"""
        domain = urlparse(request.url).netloc
        delay = self.domain_delays.get(domain, self.default_delay)
        
        if domain in self.last_request_times:
            time_since_last = time.time() - self.last_request_times[domain]
            if time_since_last < delay:
                sleep_time = delay - time_since_last
                spider.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                time.sleep(sleep_time)
        
        self.last_request_times[domain] = time.time()
        return None


class ErrorHandlingMiddleware:
    """Enhanced error handling and retry logic"""
    
    def __init__(self, max_retry_times: int = 3, retry_http_codes: Optional[List[int]] = None):
        self.max_retry_times = max_retry_times
        self.retry_http_codes = retry_http_codes or [500, 502, 503, 504, 408, 429]
        
    @classmethod
    def from_crawler(cls, crawler):
        max_retry_times = crawler.settings.getint('RETRY_TIMES', 3)
        retry_http_codes = crawler.settings.getlist('RETRY_HTTP_CODES')
        if retry_http_codes:
            retry_http_codes = [int(code) for code in retry_http_codes]
        return cls(max_retry_times, retry_http_codes)
    
    def process_response(self, request, response, spider):
        """Handle response errors"""
        if response.status in self.retry_http_codes:
            retry_times = request.meta.get('retry_times', 0)
            
            if retry_times < self.max_retry_times:
                spider.logger.warning(
                    f"Retrying {request.url} (status: {response.status}, "
                    f"attempt: {retry_times + 1}/{self.max_retry_times})"
                )
                
                retry_req = request.copy()
                retry_req.meta['retry_times'] = retry_times + 1
                retry_req.dont_filter = True
                
                # Add exponential backoff
                delay = 2 ** retry_times
                time.sleep(delay)
                
                return retry_req
            else:
                spider.logger.error(
                    f"Max retries exceeded for {request.url} (status: {response.status})"
                )
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Handle request exceptions"""
        retry_times = request.meta.get('retry_times', 0)
        
        if retry_times < self.max_retry_times:
            spider.logger.warning(
                f"Retrying {request.url} due to exception: {exception} "
                f"(attempt: {retry_times + 1}/{self.max_retry_times})"
            )
            
            retry_req = request.copy()
            retry_req.meta['retry_times'] = retry_times + 1
            retry_req.dont_filter = True
            
            # Add exponential backoff
            delay = 2 ** retry_times
            time.sleep(delay)
            
            return retry_req
        else:
            spider.logger.error(f"Max retries exceeded for {request.url}: {exception}")
        
        return None


class ResourceBlockingMiddleware:
    """Blocks unnecessary resources to improve performance"""
    
    def __init__(self, blocked_resources: Optional[List[str]] = None):
        self.blocked_resources = blocked_resources or [
            'image', 'stylesheet', 'font', 'media'
        ]
    
    @classmethod
    def from_crawler(cls, crawler):
        blocked_resources = crawler.settings.getlist('BLOCKED_RESOURCES')
        return cls(blocked_resources)
    
    def process_request(self, request, spider):
        """Block unnecessary resource requests"""
        # This would work with Playwright middleware
        resource_type = request.meta.get('resource_type')
        
        if resource_type in self.blocked_resources:
            spider.logger.debug(f"Blocking {resource_type} resource: {request.url}")
            raise IgnoreRequest(f"Blocked {resource_type} resource")
        
        return None


class CacheMiddleware:
    """Simple response caching middleware"""
    
    def __init__(self, cache_enabled: bool = True, cache_ttl: int = 3600):
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache = {}
        
    @classmethod
    def from_crawler(cls, crawler):
        cache_enabled = crawler.settings.getbool('CACHE_ENABLED', True)
        cache_ttl = crawler.settings.getint('CACHE_TTL', 3600)
        return cls(cache_enabled, cache_ttl)
    
    def process_request(self, request, spider):
        """Check cache before making request"""
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(request)
        cached_response = self.cache.get(cache_key)
        
        if cached_response:
            cache_time, response_data = cached_response
            if time.time() - cache_time < self.cache_ttl:
                spider.logger.debug(f"Cache hit for: {request.url}")
                return HtmlResponse(
                    url=request.url,
                    body=response_data.encode('utf-8'),
                    encoding='utf-8'
                )
        
        return None
    
    def process_response(self, request, response, spider):
        """Cache successful responses"""
        if self.cache_enabled and response.status == 200:
            cache_key = self._get_cache_key(request)
            self.cache[cache_key] = (time.time(), response.text)
            spider.logger.debug(f"Cached response for: {request.url}")
        
        return response
    
    def _get_cache_key(self, request):
        """Generate cache key for request"""
        return f"{request.method}:{request.url}"


class StatsMiddleware:
    """Collects detailed scraping statistics"""
    
    def __init__(self):
        self.stats = {
            'requests_total': 0,
            'responses_total': 0,
            'errors_total': 0,
            'retries_total': 0,
            'cache_hits': 0,
            'proxy_switches': 0,
            'domains': set(),
            'status_codes': {},
            'start_time': None,
            'end_time': None
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(o.spider_closed, signal=signals.spider_closed)
        return o
    
    def spider_opened(self, spider):
        """Initialize stats when spider opens"""
        self.stats['start_time'] = time.time()
        spider.logger.info("Stats middleware initialized")
    
    def spider_closed(self, spider):
        """Report final stats when spider closes"""
        self.stats['end_time'] = time.time()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        # Convert set to list for JSON serialization
        self.stats['domains'] = list(self.stats['domains'])
        self.stats['duration_seconds'] = duration
        
        spider.logger.info(f"Scraping stats: {self.stats}")
        spider.crawler.stats.set_value('mimir_middleware_stats', self.stats)
    
    def process_request(self, request, spider):
        """Track request statistics"""
        self.stats['requests_total'] += 1
        domain = urlparse(request.url).netloc
        self.stats['domains'].add(domain)
        
        # Track retries
        if request.meta.get('retry_times', 0) > 0:
            self.stats['retries_total'] += 1
        
        return None
    
    def process_response(self, request, response, spider):
        """Track response statistics"""
        self.stats['responses_total'] += 1
        
        # Track status codes
        status = response.status
        self.stats['status_codes'][status] = self.stats['status_codes'].get(status, 0) + 1
        
        return response
    
    def process_exception(self, request, exception, spider):
        """Track error statistics"""
        self.stats['errors_total'] += 1
        return None