"""
Redis Caching Layer for Mimir Async Scraper
EPIC 4 - Task 4.3.1: Multi-level caching with Redis
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import aioredis
import redis

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache settings."""

    default_ttl: int = 3600  # 1 hour
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    health_check_interval: int = 30


class RedisCache:
    """Multi-level Redis caching system for Mimir scraper."""

    def __init__(self, redis_url: str, config: CacheConfig = None):
        self.redis_url = redis_url
        self.config = config or CacheConfig()

        # Redis clients
        self.sync_client: Optional[redis.Redis] = None
        self.async_client: Optional[aioredis.Redis] = None

        # Cache key prefixes for different data types
        self.prefixes = {
            "article": "mimir:cache:article",
            "rss_feed": "mimir:cache:rss",
            "html_page": "mimir:cache:html",
            "robots_txt": "mimir:cache:robots",
            "site_config": "mimir:cache:config",
            "search_result": "mimir:cache:search",
            "ai_analysis": "mimir:cache:ai",
            "statistics": "mimir:cache:stats",
        }

        # Default TTL values for different data types
        self.ttl_settings = {
            "article": 86400,  # 24 hours
            "rss_feed": 1800,  # 30 minutes
            "html_page": 3600,  # 1 hour
            "robots_txt": 86400,  # 24 hours
            "site_config": 300,  # 5 minutes
            "search_result": 600,  # 10 minutes
            "ai_analysis": 604800,  # 1 week
            "statistics": 300,  # 5 minutes
        }

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}

    def connect(self):
        """Establish synchronous Redis connection."""
        try:
            self.sync_client = redis.from_url(
                self.redis_url,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,  # Keep binary for pickle support
            )

            # Test connection
            self.sync_client.ping()
            logger.info("Redis sync connection established")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def async_connect(self):
        """Establish asynchronous Redis connection."""
        try:
            self.async_client = aioredis.from_url(
                self.redis_url,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,
            )

            # Test connection
            await self.async_client.ping()
            logger.info("Redis async connection established")

        except Exception as e:
            logger.error(f"Failed to connect to async Redis: {e}")
            raise

    def disconnect(self):
        """Close Redis connections."""
        if self.sync_client:
            self.sync_client.close()
        logger.info("Redis sync connection closed")

    async def async_disconnect(self):
        """Close async Redis connection."""
        if self.async_client:
            await self.async_client.close()
        logger.info("Redis async connection closed")

    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix and hash for long identifiers."""
        if len(identifier) > 200:  # Hash long identifiers
            identifier = hashlib.md5(identifier.encode()).hexdigest()
        return f"{prefix}:{identifier}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        if isinstance(value, (str, int, float)):
            return json.dumps(value).encode()
        else:
            return pickle.dumps(value)

    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        try:
            # Try JSON first (faster)
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)

    # Synchronous cache operations
    def get(self, cache_type: str, identifier: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.sync_client:
            return None

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            data = self.sync_client.get(key)

            if data is None:
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return self._deserialize_value(data)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error: {e}")
            return None

    def set(self, cache_type: str, identifier: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self.sync_client:
            return False

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            data = self._serialize_value(value)
            ttl = ttl or self.ttl_settings.get(cache_type, self.config.default_ttl)

            result = self.sync_client.setex(key, ttl, data)
            if result:
                self.stats["sets"] += 1
            return bool(result)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, cache_type: str, identifier: str) -> bool:
        """Delete value from cache."""
        if not self.sync_client:
            return False

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            result = self.sync_client.delete(key)
            if result:
                self.stats["deletes"] += 1
            return bool(result)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache delete error: {e}")
            return False

    def exists(self, cache_type: str, identifier: str) -> bool:
        """Check if key exists in cache."""
        if not self.sync_client:
            return False

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            return bool(self.sync_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    # Asynchronous cache operations
    async def async_get(self, cache_type: str, identifier: str) -> Optional[Any]:
        """Async get value from cache."""
        if not self.async_client:
            return None

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            data = await self.async_client.get(key)

            if data is None:
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return self._deserialize_value(data)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Async cache get error: {e}")
            return None

    async def async_set(self, cache_type: str, identifier: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Async set value in cache."""
        if not self.async_client:
            return False

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            data = self._serialize_value(value)
            ttl = ttl or self.ttl_settings.get(cache_type, self.config.default_ttl)

            result = await self.async_client.setex(key, ttl, data)
            if result:
                self.stats["sets"] += 1
            return bool(result)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Async cache set error: {e}")
            return False

    async def async_delete(self, cache_type: str, identifier: str) -> bool:
        """Async delete value from cache."""
        if not self.async_client:
            return False

        try:
            key = self._generate_key(self.prefixes[cache_type], identifier)
            result = await self.async_client.delete(key)
            if result:
                self.stats["deletes"] += 1
            return bool(result)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Async cache delete error: {e}")
            return False

    # Batch operations
    def get_multi(self, cache_type: str, identifiers: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.sync_client or not identifiers:
            return {}

        try:
            keys = [self._generate_key(self.prefixes[cache_type], id) for id in identifiers]
            values = self.sync_client.mget(keys)

            result = {}
            for identifier, data in zip(identifiers, values):
                if data is not None:
                    result[identifier] = self._deserialize_value(data)
                    self.stats["hits"] += 1
                else:
                    self.stats["misses"] += 1

            return result

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get_multi error: {e}")
            return {}

    def set_multi(self, cache_type: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not self.sync_client or not data:
            return False

        try:
            ttl = ttl or self.ttl_settings.get(cache_type, self.config.default_ttl)
            pipe = self.sync_client.pipeline()

            for identifier, value in data.items():
                key = self._generate_key(self.prefixes[cache_type], identifier)
                serialized_value = self._serialize_value(value)
                pipe.setex(key, ttl, serialized_value)

            results = pipe.execute()
            successful = sum(1 for r in results if r)
            self.stats["sets"] += successful

            return successful == len(data)

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set_multi error: {e}")
            return False

    # Cache warming and invalidation
    def warm_cache(self, cache_type: str, data_loader: Callable) -> int:
        """Warm cache with data from a loader function."""
        try:
            data = data_loader()
            if isinstance(data, dict):
                self.set_multi(cache_type, data)
                return len(data)
            else:
                logger.warning("Data loader must return a dictionary")
                return 0
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
            return 0

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching a pattern."""
        if not self.sync_client:
            return 0

        try:
            keys = self.sync_client.keys(pattern)
            if keys:
                deleted = self.sync_client.delete(*keys)
                self.stats["deletes"] += deleted
                return deleted
            return 0
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache invalidation error: {e}")
            return 0

    def invalidate_cache_type(self, cache_type: str) -> int:
        """Invalidate all cache entries of a specific type."""
        pattern = f"{self.prefixes[cache_type]}:*"
        return self.invalidate_pattern(pattern)

    # Cache statistics and monitoring
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_operations = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_operations * 100) if total_operations > 0 else 0

        stats = self.stats.copy()
        stats.update(
            {"hit_rate": hit_rate, "total_operations": total_operations, "timestamp": datetime.now().isoformat()}
        )

        # Add Redis info if available
        if self.sync_client:
            try:
                redis_info = self.sync_client.info()
                stats.update(
                    {
                        "redis_memory_used": redis_info.get("used_memory_human"),
                        "redis_connected_clients": redis_info.get("connected_clients"),
                        "redis_uptime": redis_info.get("uptime_in_seconds"),
                    }
                )
            except Exception:
                pass

        return stats

    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}

    def cleanup_expired_keys(self, pattern: str = "*") -> int:
        """Cleanup expired keys to prevent memory leaks."""
        if not self.sync_client:
            return 0

        try:
            # Get all keys matching pattern
            keys = self.sync_client.keys(pattern)
            expired_count = 0

            # Check each key and remove if expired
            for key in keys:
                ttl = self.sync_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_count += 1

            logger.debug(f"Cleaned up {expired_count} expired cache keys")
            return expired_count

        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if not self.sync_client:
            return {}

        try:
            info = self.sync_client.info("memory")
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "maxmemory": info.get("maxmemory", 0),
                "maxmemory_human": info.get("maxmemory_human", "unlimited"),
            }
        except Exception:
            return {}

    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on Redis connection."""
        try:
            if not self.sync_client:
                return {"healthy": False, "error": "No connection"}

            start_time = time.time()
            self.sync_client.ping()
            response_time = time.time() - start_time

            return {"healthy": True, "response_time": response_time, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"healthy": False, "error": str(e), "timestamp": datetime.now().isoformat()}


# Decorators for caching
def cache_result(cache: RedisCache, cache_type: str, ttl: Optional[int] = None):
    """Decorator to cache function results."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            result = cache.get(cache_type, cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_type, cache_key, result, ttl)
            return result

        return wrapper

    return decorator


def async_cache_result(cache: RedisCache, cache_type: str, ttl: Optional[int] = None):
    """Decorator to cache async function results."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            result = await cache.async_get(cache_type, cache_key)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.async_set(cache_type, cache_key, result, ttl)
            return result

        return wrapper

    return decorator


# Cache manager for easy setup
class CacheManager:
    """Singleton cache manager for the application."""

    _instance = None
    _cache = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, redis_url: str, config: CacheConfig = None):
        """Initialize the cache system."""
        if self._cache is None:
            self._cache = RedisCache(redis_url, config)
            self._cache.connect()

    def get_cache(self) -> RedisCache:
        """Get the cache instance."""
        if self._cache is None:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        return self._cache

    def shutdown(self):
        """Shutdown the cache system."""
        if self._cache:
            self._cache.disconnect()
            self._cache = None


# Global cache instance
cache_manager = CacheManager()
