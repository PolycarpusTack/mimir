# Playwright Integration Guide for Mimir

## Overview

This guide explains how to use the new Playwright integration in Mimir to scrape JavaScript-heavy websites that don't work with traditional HTML parsing.

## What is Playwright?

Playwright is a browser automation framework that can:
- Execute JavaScript on pages
- Wait for dynamic content to load
- Handle Single Page Applications (SPAs)
- Take screenshots of pages
- Work with React, Vue, Angular, and other JS frameworks

## Installation

### 1. Install Dependencies

```bash
# Install Playwright and related packages
pip install -r requirements.txt

# This installs:
# - playwright>=1.40.0
# - playwright-stealth>=1.0.6
```

### 2. Install Browsers

```bash
# Run the setup script
python setup_playwright.py

# This will:
# - Install Chromium, Firefox, and WebKit browsers
# - Create configuration files
# - Set up directories for screenshots
# - Run a test to verify installation
```

### 3. Enable Playwright

Edit `config.json` and set Playwright to enabled:

```json
{
    "playwright_scraping": {
        "enabled": true,  // Change from false to true
        "config_file": "playwright_config.json",
        "fallback_to_beautifulsoup": true
    }
}
```

## Configuration

### Main Configuration (config.json)

```json
{
    "playwright_scraping": {
        "enabled": true,
        "config_file": "playwright_config.json",
        "fallback_to_beautifulsoup": true,
        "js_site_indicators": [
            "react-root",
            "ng-app",
            "vue-app",
            "__NEXT_DATA__",
            "window.React"
        ]
    }
}
```

### Playwright Configuration (playwright_config.json)

The setup script creates this file with sensible defaults:

```json
{
    "playwright_settings": {
        "enabled": true,
        "browsers": {
            "chromium": {
                "enabled": true,
                "headless": true,
                "args": ["--disable-blink-features=AutomationControlled"]
            }
        },
        "timeouts": {
            "page_load": 30000,
            "script": 10000,
            "navigation": 30000
        },
        "viewport": {
            "width": 1920,
            "height": 1080
        },
        "stealth_mode": true,
        "block_resources": ["image", "font", "media"],
        "max_concurrent_contexts": 3,
        "screenshot_on_error": true
    }
}
```

## Adding JavaScript Sites

### Method 1: Mark Site as Requiring JavaScript

In `sites_to_scrape.json`, add the `js_required` flag:

```json
{
    "name": "React News Site",
    "url": "https://news.example.com",
    "type": "html",
    "enabled": true,
    "js_required": true,  // This forces Playwright usage
    "playwright_settings": {
        "wait_for_selector": ".article-list",
        "wait_for_load_state": "networkidle"
    },
    "selectors": {
        "overview_article_link": "a.article-link",
        "detail_title": "h1.title",
        "detail_content": "div.content"
    }
}
```

### Method 2: Automatic Detection

If `js_required` is not set, the system automatically detects JavaScript frameworks by looking for indicators like:
- `react-root`
- `ng-app`
- `vue-app`
- `__NEXT_DATA__`

## Playwright-Specific Settings

### Wait Strategies

```json
{
    "playwright_settings": {
        // Wait for specific element
        "wait_for_selector": ".content-loaded",
        
        // Wait for network to be idle
        "wait_for_load_state": "networkidle",
        
        // Wait for custom JavaScript function
        "wait_for_function": "() => document.querySelectorAll('.article').length > 0",
        
        // Handle infinite scroll
        "scroll_to_load": true,
        "scroll_pause_time": 2000,
        "max_scrolls": 5
    }
}
```

### Resource Blocking (Performance)

Block unnecessary resources to speed up scraping:

```json
{
    "playwright_settings": {
        "block_resources": ["image", "stylesheet", "font", "media"]
    }
}
```

### Authentication

For sites requiring login:

```json
{
    "playwright_settings": {
        "authentication": {
            "required": true,
            "type": "basic",
            "username": "${AUTH_USERNAME}",
            "password": "${AUTH_PASSWORD}"
        }
    }
}
```

## Running the Scraper

### Option 1: Use Enhanced Scraper (Recommended)

```bash
# The enhanced scraper automatically uses Playwright when needed
python scraper_enhanced.py --run

# Check Playwright status
python scraper_enhanced.py --check-playwright
```

### Option 2: Replace Original Scraper

```bash
# Backup original
mv scraper.py scraper_original.py

# Use enhanced version
mv scraper_enhanced.py scraper.py

# Run as normal
python scraper.py --run
```

## Database Schema

When using PostgreSQL, Playwright adds these columns to the articles table:

- `rendering_method` - How the page was rendered ('html' or 'playwright')
- `screenshot_path` - Path to screenshot if captured
- `js_errors` - JavaScript errors encountered (JSONB)
- `page_metrics` - Performance metrics (JSONB)

Run the migration:

```bash
alembic upgrade head
```

## Monitoring & Debugging

### View Playwright Statistics

```sql
-- PostgreSQL
SELECT * FROM playwright_scraping_stats 
ORDER BY date DESC;

-- See which sites use Playwright
SELECT DISTINCT source, rendering_method, COUNT(*) 
FROM articles 
GROUP BY source, rendering_method;
```

### Check Logs

Playwright activities are logged to:
- `logs/mimir_scraper.log` - General scraping logs
- `logs/playwright_screenshots/` - Error screenshots

### Debug Mode

To see the browser in action (non-headless):

```json
{
    "browsers": {
        "chromium": {
            "headless": false  // Change to false
        }
    }
}
```

## Performance Considerations

1. **Playwright is slower than BeautifulSoup**
   - Only use for sites that require JavaScript
   - The system automatically falls back when JS isn't needed

2. **Resource Usage**
   - Each browser context uses memory
   - Limit concurrent contexts in config
   - Block unnecessary resources (images, fonts)

3. **Timeouts**
   - Adjust timeouts based on site speed
   - Shorter timeouts = faster failures
   - Longer timeouts = better success rate

## Troubleshooting

### "Browser not found"
```bash
# Re-run setup
python setup_playwright.py
```

### "Timeout waiting for selector"
- Check if the selector is correct
- Increase timeout in playwright_config.json
- Try a different wait strategy

### "JavaScript error on page"
- Check `js_errors` in the database
- Some JS errors are normal and don't affect scraping
- Enable screenshots to see what the page looks like

### High Memory Usage
- Reduce `max_concurrent_contexts`
- Enable more resource blocking
- Restart the scraper periodically

## Examples

### React News Site
```json
{
    "name": "TechCrunch (React)",
    "url": "https://techcrunch.com",
    "type": "html",
    "js_required": true,
    "playwright_settings": {
        "wait_for_selector": "[data-module='latest-news']",
        "wait_for_load_state": "domcontentloaded"
    }
}
```

### Infinite Scroll Site
```json
{
    "name": "Infinite News",
    "url": "https://infinite-news.example.com",
    "type": "html",
    "js_required": true,
    "playwright_settings": {
        "scroll_to_load": true,
        "scroll_pause_time": 3000,
        "max_scrolls": 10,
        "wait_for_function": "() => document.querySelectorAll('.article').length >= 50"
    }
}
```

### Site with Lazy Loading
```json
{
    "name": "Lazy News",
    "url": "https://lazy-news.example.com",
    "type": "html",
    "js_required": true,
    "playwright_settings": {
        "wait_for_selector": "img[data-loaded='true']",
        "execute_script": "window.scrollTo(0, document.body.scrollHeight);",
        "wait_for_load_state": "networkidle"
    }
}
```

## Best Practices

1. **Start with `js_required: false`**
   - Let the system auto-detect
   - Only force Playwright if auto-detection fails

2. **Use specific selectors**
   - More specific = faster and more reliable
   - Test selectors in browser console first

3. **Monitor performance**
   - Check `playwright_scraping_stats` table
   - Watch for high error rates
   - Adjust timeouts as needed

4. **Screenshots for debugging**
   - Enable `screenshot_on_error`
   - Check screenshots when scraping fails
   - Disable in production to save disk space

5. **Fallback strategy**
   - Keep `fallback_to_beautifulsoup: true`
   - System automatically falls back on errors
   - No data loss if Playwright fails

## Testing

Run the comprehensive test suite:

```bash
# Run all Playwright integration tests
python test_playwright_integration.py

# Test a specific site
python -c "
from playwright_scraper import PlaywrightScraper
import asyncio

async def test():
    async with PlaywrightScraper() as scraper:
        result = await scraper.scrape_with_playwright(
            'https://example.com',
            {'name': 'Test', 'selectors': {}}
        )
        print(result)

asyncio.run(test())
"
```

## Migration Checklist

- [ ] Install Playwright dependencies
- [ ] Run setup_playwright.py
- [ ] Enable in config.json
- [ ] Run database migration (if using PostgreSQL)
- [ ] Test with one JS-heavy site
- [ ] Monitor performance and errors
- [ ] Adjust timeouts and settings
- [ ] Roll out to more sites gradually

## Support

If you encounter issues:

1. Check the logs in `logs/mimir_scraper.log`
2. Look for error screenshots in `logs/playwright_screenshots/`
3. Run the test suite to verify installation
4. Try with `headless: false` to see what's happening
5. Check the browser console for JavaScript errors

Remember: Playwright is powerful but slower than regular scraping. Use it only when necessary!