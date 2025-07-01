"""
Scrapy settings for Mimir project
Production-ready configuration with security and performance optimizations
"""

import os

# Scrapy settings for mimir_scrapy project
BOT_NAME = 'mimir_scrapy'

SPIDER_MODULES = ['mimir_scrapy.spiders']
NEWSPIDER_MODULE = 'mimir_scrapy.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure a delay for requests for the same website
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = 0.5

# Configure concurrent requests
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# Request timeout settings
DOWNLOAD_TIMEOUT = 30

# AutoThrottle settings
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 60
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = False

# Retry settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429, 403]

# Pipelines configuration
ITEM_PIPELINES = {
    'mimir_scrapy.pipelines.ValidationPipeline': 100,
    'mimir_scrapy.pipelines.DuplicationPipeline': 200,
    'mimir_scrapy.pipelines.KeywordPipeline': 300,
    'mimir_scrapy.pipelines.DatabasePipeline': 400,
    'mimir_scrapy.pipelines.StatsPipeline': 500,
}

# Downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'mimir_scrapy.middlewares.UserAgentMiddleware': 400,
    'mimir_scrapy.middlewares.RateLimitMiddleware': 450,
    'mimir_scrapy.middlewares.ErrorHandlingMiddleware': 500,
    'mimir_scrapy.middlewares.StatsMiddleware': 550,
    'mimir_scrapy.middlewares.CacheMiddleware': 600,
}

# User agent list for rotation
USER_AGENT_LIST = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mimir Broadcast Intelligence Bot 1.0 (+https://mimir.ai/bot)'
]

# Default request headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Enable and configure HTTP caching
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 3600
HTTPCACHE_DIR = 'httpcache'
HTTPCACHE_IGNORE_HTTP_CODES = [503, 504, 505, 500, 403, 404, 408, 429]

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(levelname)s: %(message)s'

# Telnet Console (disabled for security)
TELNETCONSOLE_ENABLED = False

# Memory usage monitoring
MEMUSAGE_ENABLED = True
MEMUSAGE_LIMIT_MB = 2048
MEMUSAGE_WARNING_MB = 1024

# DNS timeout
DNSCACHE_ENABLED = True
DNSCACHE_SIZE = 10000
DNS_TIMEOUT = 60

# Custom settings that can be overridden by environment
CACHE_ENABLED = os.getenv('SCRAPY_CACHE_ENABLED', 'true').lower() == 'true'
CACHE_TTL = int(os.getenv('SCRAPY_CACHE_TTL', '3600'))

# Domain-specific delays (can be customized per deployment)
DOMAIN_DELAYS = {
    'example.com': 3.0,
    'heavy-site.com': 5.0,
}

# Resources to block for performance
BLOCKED_RESOURCES = ['image', 'stylesheet', 'font', 'media']

# Database settings (will be populated by ScrapyAdapter)
DATABASE_SETTINGS = {
    'database_name': 'mimir_news.db',
    'use_postgres': False
}

# Keyword monitoring settings (will be populated by ScrapyAdapter)
KEYWORD_SETTINGS = {
    'keywords_to_monitor': [],
    'alert_keywords': [],
    'case_sensitive': False
}

# Email notification settings (will be populated by ScrapyAdapter)
EMAIL_SETTINGS = {
    'send_email': False
}

# Proxy settings (will be populated by ScrapyAdapter if enabled)
PROXY_SETTINGS = {
    'enabled': False
}

# Security settings
COOKIES_ENABLED = True
COOKIES_DEBUG = False

# Reactor settings for async compatibility
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'

# Feed exports (for debugging and data export)
FEEDS = {
    'output.json': {
        'format': 'json',
        'encoding': 'utf8',
        'store_empty': False,
        'overwrite': True,
    },
}

# Extensions
EXTENSIONS = {
    'scrapy.extensions.telnet.TelnetConsole': None,  # Disabled for security
    'scrapy.extensions.memusage.MemoryUsage': 500,
    'scrapy.extensions.logstats.LogStats': 500,
}

# Request fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'

# Enable compression
COMPRESSION_ENABLED = True

# Handle duplicate requests
DUPEFILTER_DEBUG = False

# Stats collection
STATS_CLASS = 'scrapy.statscollectors.MemoryStatsCollector'

# Reactor settings
REACTOR_THREADPOOL_MAXSIZE = 20

# Download handlers for special protocols
DOWNLOAD_HANDLERS = {
    'http': 'scrapy.core.downloader.handlers.http.HTTPDownloadHandler',
    'https': 'scrapy.core.downloader.handlers.http.HTTPDownloadHandler',
}

# Media pipeline settings (if needed for images/files)
MEDIA_ALLOW_REDIRECTS = True
FILES_STORE = 'files'
IMAGES_STORE = 'images'

# Item processor settings
ITEM_PROCESSOR = 'scrapy.pipelines.ItemPipelineManager'

# Close spider settings
CLOSESPIDER_TIMEOUT = 3600  # 1 hour max
CLOSESPIDER_ITEMCOUNT = 1000  # Max items per spider
CLOSESPIDER_PAGECOUNT = 100   # Max pages per spider
CLOSESPIDER_ERRORCOUNT = 10   # Max errors before stopping

# Job directory for persistence
JOBDIR = 'jobs'

# Spider middleware
SPIDER_MIDDLEWARES = {
    'scrapy.spidermiddlewares.httperror.HttpErrorMiddleware': 50,
    'scrapy.spidermiddlewares.offsite.OffsiteMiddleware': 500,
    'scrapy.spidermiddlewares.referer.RefererMiddleware': 700,
    'scrapy.spidermiddlewares.urllength.UrlLengthMiddleware': 800,
    'scrapy.spidermiddlewares.depth.DepthMiddleware': 900,
}

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = '2.7'
TWISTED_REACTOR = 'twisted.internet.asyncioreactor.AsyncioSelectorReactor'
FEED_EXPORT_ENCODING = 'utf-8'