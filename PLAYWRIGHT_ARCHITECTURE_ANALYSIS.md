# Mimir Codebase Architecture Analysis for Playwright Integration

## Executive Summary

This document provides a comprehensive architectural analysis of the Mimir news scraper codebase, focusing on how to seamlessly integrate Playwright for JavaScript-heavy website scraping while maintaining compatibility with the existing architecture.

## Current Architecture Overview

### 1. Core Scraping Flow

The scraping architecture follows a modular pattern with clear separation of concerns:

```
[Configuration] → [Scraper Engine] → [Parser] → [Database Adapter] → [Storage]
                        ↓
                  [Error Handler]
                        ↓
                  [Email/Alerts]
```

#### Key Components:

1. **NewsScraper Class** (`scraper.py`):
   - Main orchestrator for scraping operations
   - Handles both RSS and HTML scraping
   - Uses `requests.Session` for HTTP operations
   - Implements rate limiting and retry logic
   - Respects robots.txt

2. **AsyncNewsScraper Class** (`async_scraper.py`):
   - High-performance async implementation
   - Uses `aiohttp` for concurrent requests
   - Includes Redis caching and event streaming
   - Implements circuit breakers and resilience patterns

### 2. Database Interaction Patterns

#### Database Abstraction Layer (`db_adapter.py`):
- Provides unified interface for SQLite and PostgreSQL
- Key pattern: All database operations go through the adapter
- Supports both sync and async operations
- Handles schema differences transparently

#### Key Database Operations:
```python
# Article storage pattern
article_id = db_adapter.add_article(
    url=url,
    title=title,
    publication_date=publication_date,
    source_website=source_name,
    category=category,
    content_summary=content_summary,
    full_content=full_content,
    author=author,
    keywords_found=keywords_found
)
```

### 3. Configuration Management

#### ConfigLoader Pattern (`config_loader.py`):
- Secure configuration loading with environment variable substitution
- Pattern: `${ENV_VAR}` in JSON files
- Validates against hardcoded secrets
- Provides typed configuration getters

#### Configuration Structure:
```json
{
    "scraping": {
        "user_agent": "Mimir-Scraper/1.0",
        "default_request_delay_seconds": 2,
        "max_retries": 3,
        "concurrent_requests": 10
    },
    "playwright": {
        "enabled": false,
        "headless": true,
        "timeout": 30000,
        "viewport": {"width": 1920, "height": 1080}
    }
}
```

### 4. Error Handling Standards

#### Exception Hierarchy (`ai_exceptions.py`):
- Base exception: `MimirAIException`
- Pattern: Specific exceptions for each error type
- Rich error context with details dictionary
- Consistent error message formatting

#### Error Handling Pattern:
```python
try:
    # Operation
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    db_manager.log_scrape_error(run_id, source, type(e).__name__, str(e), url)
    # Graceful degradation
```

### 5. Testing Patterns

#### Test Structure:
- Pytest-based testing framework
- Fixtures for common test data (`tests/conftest.py`)
- Async test support
- Mock external dependencies

#### Key Testing Patterns:
- Integration tests with in-memory databases
- Performance testing scenarios
- Contract testing for APIs
- Load testing configurations

## Playwright Integration Architecture

### 1. Design Principles

To ensure seamless integration, the Playwright implementation should:

1. **Follow Existing Patterns**: Mirror the structure of existing scrapers
2. **Maintain Compatibility**: Work through the same database adapter
3. **Respect Configuration**: Use the existing configuration system
4. **Handle Errors Consistently**: Use the established exception hierarchy
5. **Support Both Sync/Async**: Provide both implementations like existing code

### 2. Proposed Architecture

#### Component Structure:
```
playwright_scraper.py         # Sync implementation
async_playwright_scraper.py   # Async implementation
playwright_manager.py         # Browser lifecycle management
playwright_exceptions.py      # Playwright-specific exceptions
playwright_utils.py          # Helper functions
```

#### Integration Points:

1. **Configuration Extension**:
   - Add `playwright` section to config.json
   - Use ConfigLoader for settings
   - Support environment variable substitution

2. **Database Integration**:
   - Use existing `db_adapter` for all storage
   - Add metadata for JS-rendered content
   - Track rendering metrics

3. **Error Handling**:
   - Extend exception hierarchy for browser-specific errors
   - Log browser console errors
   - Handle timeout and navigation errors

4. **Scraper Integration**:
   - Add `"type": "playwright"` site configuration
   - Fallback to regular scraping on failure
   - Reuse existing parsing logic

### 3. Implementation Patterns

#### Browser Management Pattern:
```python
class PlaywrightManager:
    def __init__(self, config: dict):
        self.config = config
        self.browser = None
        self.context = None
        
    async def __aenter__(self):
        # Initialize browser with config
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup browser resources
```

#### Site Scraping Pattern:
```python
async def scrape_with_playwright(site_config: dict, manager: PlaywrightManager):
    page = await manager.new_page()
    try:
        # Navigate and wait for content
        await page.goto(site_config['url'])
        await page.wait_for_selector(site_config['wait_selector'])
        
        # Extract content using existing selectors
        content = await page.content()
        
        # Use existing parsing logic
        return parse_content(content, site_config)
    finally:
        await page.close()
```

### 4. Configuration Schema

#### sites_to_scrape.json Extension:
```json
{
    "name": "JavaScript Heavy Site",
    "url": "https://example.com",
    "type": "playwright",
    "enabled": true,
    "playwright_config": {
        "wait_selector": ".article-content",
        "wait_timeout": 10000,
        "scroll_to_load": true,
        "screenshot_on_error": true
    },
    "selectors": {
        // Existing selector configuration
    }
}
```

### 5. Database Schema Extensions

No schema changes needed, but utilize metadata fields:
- Store JavaScript execution metrics
- Track rendering time
- Log browser console errors

### 6. Testing Strategy

#### Unit Tests:
- Mock Playwright API
- Test error handling
- Validate configuration parsing

#### Integration Tests:
- Use Playwright in headless mode
- Test against local HTML fixtures
- Validate database storage

#### Performance Tests:
- Measure browser resource usage
- Compare with regular scraping
- Test concurrent browser instances

## Implementation Roadmap

### Phase 1: Core Implementation
1. Create PlaywrightManager for browser lifecycle
2. Implement basic scraping functionality
3. Add error handling and logging
4. Create unit tests

### Phase 2: Integration
1. Integrate with existing scraper flow
2. Add configuration support
3. Implement database storage
4. Add integration tests

### Phase 3: Advanced Features
1. Add screenshot capture on errors
2. Implement JavaScript execution
3. Add browser console log capture
4. Performance optimization

### Phase 4: Production Readiness
1. Add monitoring and metrics
2. Implement resource limits
3. Add circuit breakers
4. Documentation and examples

## Best Practices and Recommendations

### 1. Resource Management
- Reuse browser contexts when possible
- Implement proper cleanup in all error paths
- Set memory and CPU limits
- Monitor browser process health

### 2. Performance Optimization
- Use browser connection pooling
- Parallelize page operations carefully
- Cache rendered content when appropriate
- Implement request interception for efficiency

### 3. Error Handling
- Capture browser console errors
- Take screenshots on failure
- Implement retry logic with backoff
- Graceful fallback to regular scraping

### 4. Security Considerations
- Run browsers in sandboxed environment
- Disable unnecessary browser features
- Validate all executed JavaScript
- Monitor for suspicious activity

### 5. Monitoring and Observability
- Track rendering times
- Monitor resource usage
- Log all browser errors
- Create performance dashboards

## Conclusion

The Mimir codebase follows well-established patterns that make Playwright integration straightforward. By following the existing architecture patterns and respecting the established interfaces, we can add powerful JavaScript rendering capabilities while maintaining system stability and consistency.

The modular design, comprehensive error handling, and flexible configuration system provide an excellent foundation for extending the scraper with Playwright support. The proposed implementation maintains backward compatibility while adding new capabilities for modern JavaScript-heavy websites.