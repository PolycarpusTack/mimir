"""
Custom exception hierarchy for Mimir Playwright integration.
Provides specific exception types for browser-based scraping errors.
"""

from typing import Any, Dict, Optional
from ai_exceptions import MimirAIException


class PlaywrightException(MimirAIException):
    """Base exception for all Playwright-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message, details)


class BrowserLaunchError(PlaywrightException):
    """Raised when browser fails to launch."""
    
    def __init__(self, browser_type: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize browser launch error.
        
        Args:
            browser_type: Type of browser (chromium, firefox, webkit)
            reason: Reason for the failure
            details: Optional additional details
        """
        message = f"Failed to launch {browser_type} browser: {reason}"
        super().__init__(message, details)
        self.browser_type = browser_type
        self.reason = reason


class PageLoadError(PlaywrightException):
    """Raised when page fails to load."""
    
    def __init__(self, url: str, reason: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize page load error.
        
        Args:
            url: URL that failed to load
            reason: Reason for the failure
            status_code: Optional HTTP status code
            details: Optional additional details
        """
        message = f"Failed to load page {url}: {reason}"
        if status_code:
            message += f" (status: {status_code})"
        
        if details is None:
            details = {}
        details['url'] = url
        if status_code:
            details['status_code'] = status_code
            
        super().__init__(message, details)
        self.url = url
        self.reason = reason
        self.status_code = status_code


class ElementNotFoundError(PlaywrightException):
    """Raised when expected element is not found on page."""
    
    def __init__(self, selector: str, url: str, timeout: Optional[int] = None):
        """
        Initialize element not found error.
        
        Args:
            selector: CSS selector that failed
            url: URL where element was not found
            timeout: Optional timeout value in milliseconds
        """
        message = f"Element '{selector}' not found on {url}"
        if timeout:
            message += f" after {timeout}ms"
            
        details = {
            'selector': selector,
            'url': url,
            'timeout': timeout
        }
        super().__init__(message, details)
        self.selector = selector
        self.url = url
        self.timeout = timeout


class JavaScriptError(PlaywrightException):
    """Raised when JavaScript execution fails."""
    
    def __init__(self, script: str, error_message: str, url: Optional[str] = None):
        """
        Initialize JavaScript error.
        
        Args:
            script: JavaScript code that failed
            error_message: Error message from browser
            url: Optional URL where error occurred
        """
        # Truncate script for security
        script_sample = script[:100] + "..." if len(script) > 100 else script
        
        message = f"JavaScript execution failed: {error_message}"
        if url:
            message += f" on {url}"
            
        details = {
            'script_sample': script_sample,
            'error': error_message,
            'url': url
        }
        super().__init__(message, details)
        self.script = script
        self.error_message = error_message
        self.url = url


class BrowserContextError(PlaywrightException):
    """Raised when browser context operations fail."""
    
    def __init__(self, operation: str, reason: str):
        """
        Initialize browser context error.
        
        Args:
            operation: Operation that failed (e.g., 'create_context', 'close_context')
            reason: Reason for the failure
        """
        message = f"Browser context operation '{operation}' failed: {reason}"
        super().__init__(message)
        self.operation = operation
        self.reason = reason


class NetworkError(PlaywrightException):
    """Raised when network-related operations fail."""
    
    def __init__(self, operation: str, url: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize network error.
        
        Args:
            operation: Network operation (e.g., 'intercept', 'wait_for_response')
            url: URL involved in the operation
            reason: Reason for the failure
            details: Optional additional details
        """
        message = f"Network operation '{operation}' failed for {url}: {reason}"
        if details is None:
            details = {}
        details.update({
            'operation': operation,
            'url': url
        })
        super().__init__(message, details)
        self.operation = operation
        self.url = url
        self.reason = reason


class AuthenticationError(PlaywrightException):
    """Raised when authentication fails."""
    
    def __init__(self, auth_type: str, url: str, reason: str):
        """
        Initialize authentication error.
        
        Args:
            auth_type: Type of authentication (e.g., 'basic', 'cookie', 'oauth')
            url: URL requiring authentication
            reason: Reason for the failure
        """
        message = f"{auth_type} authentication failed for {url}: {reason}"
        super().__init__(message)
        self.auth_type = auth_type
        self.url = url
        self.reason = reason


class ScreenshotError(PlaywrightException):
    """Raised when screenshot capture fails."""
    
    def __init__(self, path: str, reason: str):
        """
        Initialize screenshot error.
        
        Args:
            path: Path where screenshot was to be saved
            reason: Reason for the failure
        """
        message = f"Failed to capture screenshot to {path}: {reason}"
        super().__init__(message)
        self.path = path
        self.reason = reason


class ResourceBlockError(PlaywrightException):
    """Raised when resource blocking fails."""
    
    def __init__(self, resource_type: str, reason: str):
        """
        Initialize resource block error.
        
        Args:
            resource_type: Type of resource (e.g., 'image', 'stylesheet', 'font')
            reason: Reason for the failure
        """
        message = f"Failed to block {resource_type} resources: {reason}"
        super().__init__(message)
        self.resource_type = resource_type
        self.reason = reason


class PlaywrightTimeoutError(PlaywrightException):
    """Raised when Playwright operation times out."""
    
    def __init__(self, operation: str, timeout_ms: int, url: Optional[str] = None):
        """
        Initialize timeout error.
        
        Args:
            operation: Operation that timed out
            timeout_ms: Timeout value in milliseconds
            url: Optional URL where timeout occurred
        """
        message = f"Playwright operation '{operation}' timed out after {timeout_ms}ms"
        if url:
            message += f" on {url}"
            
        details = {
            'operation': operation,
            'timeout_ms': timeout_ms,
            'url': url
        }
        super().__init__(message, details)
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.url = url


# Utility functions for error handling

def handle_playwright_error(e: Exception, context: Dict[str, Any]) -> None:
    """
    Convert generic exceptions to specific Playwright errors.
    
    Args:
        e: Original exception
        context: Context information (url, operation, etc.)
        
    Raises:
        Appropriate PlaywrightException subclass
    """
    error_str = str(e).lower()
    
    if "timeout" in error_str:
        raise PlaywrightTimeoutError(
            context.get('operation', 'unknown'),
            context.get('timeout', 30000),
            context.get('url')
        )
    elif "net::err" in error_str or "network" in error_str:
        raise NetworkError(
            context.get('operation', 'request'),
            context.get('url', 'unknown'),
            str(e)
        )
    elif "not found" in error_str and context.get('selector'):
        raise ElementNotFoundError(
            context.get('selector'),
            context.get('url', 'unknown'),
            context.get('timeout')
        )
    elif "crashed" in error_str or "closed" in error_str:
        raise BrowserContextError(
            context.get('operation', 'unknown'),
            str(e)
        )
    else:
        # Generic Playwright exception
        raise PlaywrightException(str(e), context)