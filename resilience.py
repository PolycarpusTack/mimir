"""
Resilience and Error Handling Module for Mimir Async Scraper
EPIC 4 - Critical Fix: Circuit breakers, retry logic, resilience patterns
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import aiohttp

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to open circuit
    timeout: float = 60.0  # Timeout in seconds before trying again
    expected_exception: tuple = (Exception,)  # Exceptions that trigger circuit
    recovery_timeout: float = 30.0  # Timeout for half-open state
    success_threshold: int = 3  # Successes needed to close circuit in half-open


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_exceptions: tuple = (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                return time.time() - self.last_failure_time >= self.config.timeout
        elif self.state == CircuitState.HALF_OPEN:
            if self.last_failure_time:
                return time.time() - self.last_failure_time >= self.config.recovery_timeout
        return False

    def _record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.last_success_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed after successful recovery")

    def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened after failure in half-open state")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        # Check if we should attempt reset
        if self.state == CircuitState.OPEN and self._should_attempt_reset():
            self.state = CircuitState.HALF_OPEN
            logger.info("Circuit breaker moving to half-open state")

        # Fail fast if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._record_success()
            return result

        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exception, don't trigger circuit breaker
            raise


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class AsyncRetry:
    """Async retry mechanism with exponential backoff and jitter."""

    def __init__(self, config: RetryConfig):
        self.config = config

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt."""
        delay = min(self.config.base_delay * (self.config.exponential_base**attempt), self.config.max_delay)

        if self.config.jitter:
            # Add jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except self.config.retry_exceptions as e:
                last_exception = e

                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
                    break
            except Exception as e:
                # Non-retryable exception
                logger.error(f"Non-retryable exception: {e}")
                raise

        # If we get here, all retries failed
        raise last_exception


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests = 0
        self.max_concurrent = max_concurrent

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead isolation."""
        async with self.semaphore:
            self.active_requests += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            finally:
                self.active_requests -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "active_requests": self.active_requests,
            "max_concurrent": self.max_concurrent,
            "available_slots": self.max_concurrent - self.active_requests,
        }


class TimeoutWrapper:
    """Timeout wrapper for async operations."""

    def __init__(self, timeout: float):
        self.timeout = timeout

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                # For sync functions, run in executor with timeout
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(loop.run_in_executor(None, func, *args, **kwargs), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {self.timeout}s")
            raise


class ResilientHttpClient:
    """HTTP client with built-in resilience patterns."""

    def __init__(
        self,
        circuit_breaker_config: CircuitBreakerConfig = None,
        retry_config: RetryConfig = None,
        timeout: float = 30.0,
        max_concurrent: int = 10,
    ):
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.retry = AsyncRetry(retry_config or RetryConfig())
        self.timeout_wrapper = TimeoutWrapper(timeout)
        self.bulkhead = BulkheadIsolation(max_concurrent)

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, keepalive_timeout=30, enable_cleanup_closed=True)

        timeout = aiohttp.ClientTimeout(total=30)

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers={"User-Agent": "Mimir-ResilientScraper/1.0"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get(self, url: str, **kwargs) -> str:
        """Resilient HTTP GET request."""

        async def _get():
            async with self.session.get(url, **kwargs) as response:
                response.raise_for_status()
                return await response.text()

        # Apply resilience patterns
        return await self.bulkhead.execute(
            self.timeout_wrapper.execute, self.retry.execute, self.circuit_breaker.call, _get
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all resilience components."""
        return {
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "success_count": self.circuit_breaker.success_count,
            },
            "bulkhead": self.bulkhead.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }


class ResilienceManager:
    """Central manager for all resilience patterns."""

    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, AsyncRetry] = {}
        self.bulkheads: Dict[str, BulkheadIsolation] = {}

    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config or CircuitBreakerConfig())
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker

    def create_retry_handler(self, name: str, config: RetryConfig = None) -> AsyncRetry:
        """Create and register a retry handler."""
        retry_handler = AsyncRetry(config or RetryConfig())
        self.retry_handlers[name] = retry_handler
        return retry_handler

    def create_bulkhead(self, name: str, max_concurrent: int = 10) -> BulkheadIsolation:
        """Create and register a bulkhead."""
        bulkhead = BulkheadIsolation(max_concurrent)
        self.bulkheads[name] = bulkhead
        return bulkhead

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def get_retry_handler(self, name: str) -> Optional[AsyncRetry]:
        """Get retry handler by name."""
        return self.retry_handlers.get(name)

    def get_bulkhead(self, name: str) -> Optional[BulkheadIsolation]:
        """Get bulkhead by name."""
        return self.bulkheads.get(name)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        return {
            "circuit_breakers": {
                name: {"state": cb.state.value, "failure_count": cb.failure_count, "success_count": cb.success_count}
                for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {name: bh.get_stats() for name, bh in self.bulkheads.items()},
            "timestamp": datetime.now().isoformat(),
        }


# Decorators for easy application of resilience patterns
def circuit_breaker(config: CircuitBreakerConfig = None, name: str = None):
    """Decorator to apply circuit breaker pattern."""

    def decorator(func):
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        cb = CircuitBreaker(config or CircuitBreakerConfig())

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)

        return wrapper

    return decorator


def retry(config: RetryConfig = None):
    """Decorator to apply retry pattern."""

    def decorator(func):
        retry_handler = AsyncRetry(config or RetryConfig())

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def timeout(seconds: float):
    """Decorator to apply timeout pattern."""

    def decorator(func):
        timeout_wrapper = TimeoutWrapper(seconds)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await timeout_wrapper.execute(func, *args, **kwargs)

        return wrapper

    return decorator


def bulkhead(max_concurrent: int = 10, name: str = None):
    """Decorator to apply bulkhead pattern."""

    def decorator(func):
        bulkhead_name = name or f"{func.__module__}.{func.__name__}"
        bh = BulkheadIsolation(max_concurrent)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await bh.execute(func, *args, **kwargs)

        return wrapper

    return decorator


# Global resilience manager instance
resilience_manager = ResilienceManager()
