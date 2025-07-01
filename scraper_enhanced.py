#!/usr/bin/env python
"""
Enhanced version of scraper.py with Playwright support.
This is a drop-in replacement that adds JavaScript rendering capabilities.

To use this instead of the original scraper.py:
1. Rename scraper.py to scraper_original.py
2. Rename this file to scraper.py
3. Run setup_playwright.py to install browser dependencies
4. Set "enabled": true in config.json under playwright_scraping
"""

# Import everything from the original scraper
import sys
import os

# Add the current directory to the path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all functions and variables from the original scraper
from scraper import *

# Import Playwright integration
from scraper_playwright_integration import run_with_playwright

# Store original functions
_original_run_scraper = run_scraper
_original_parse_html_site = parse_html_site

# Flag to check if Playwright is available and enabled
try:
    from playwright.async_api import async_playwright
    from config_loader import config_loader
    
    config = config_loader.load_config()
    PLAYWRIGHT_AVAILABLE = True
    PLAYWRIGHT_ENABLED = config.get('playwright_scraping', {}).get('enabled', False)
    
    if PLAYWRIGHT_ENABLED:
        logger.info("Playwright support is ENABLED")
    else:
        logger.info("Playwright support is available but DISABLED")
        
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PLAYWRIGHT_ENABLED = False
    logger.warning("Playwright not installed. Run: pip install playwright playwright-stealth")


# Enhanced parse_html_site with Playwright support
async def parse_html_site_with_playwright_async(site_config: Dict, run_id: int) -> int:
    """
    Async version of parse_html_site that uses Playwright for JS-heavy sites.
    """
    from playwright_scraper import PlaywrightScraper
    from db_adapter_playwright import add_article_with_playwright_data
    
    base_url = site_config["url"]
    new_articles_count = 0
    source_name = site_config["name"]
    category = site_config.get("category", "general")
    
    # Keywords configuration
    keywords_config = CONFIG.get("keywords_monitoring", {})
    monitor_keywords = keywords_config.get("enabled", False)
    keywords_list = keywords_config.get("keywords", [])
    
    # Initialize Playwright scraper
    async with PlaywrightScraper() as pw_scraper:
        # Check if JS is required
        initial_html = fetch_url(base_url)
        
        if pw_scraper.is_javascript_required(site_config, initial_html):
            logger.info(f"JavaScript detected for {source_name}, using Playwright")
            
            # Scrape with Playwright
            content_data = await pw_scraper.scrape_with_playwright(base_url, site_config)
            
            # Process article links
            for article_url in content_data.get('article_links', []):
                # Check if article exists
                if db_manager.article_exists(article_url):
                    continue
                    
                # Get article details
                article_data = await pw_scraper.scrape_with_playwright(article_url, site_config)
                
                # Extract fields
                title = article_data.get('title', 'No title')
                content = article_data.get('content', '')
                author = article_data.get('author', '')
                date_str = article_data.get('date', '')
                
                # Parse date
                publication_date = parse_date(date_str, site_config.get("date_format"))
                
                # Extract keywords
                keywords_found = []
                if monitor_keywords:
                    combined_text = f"{title} {content}"
                    keyword_matches = extract_keywords(combined_text, keywords_list)
                    keywords_found = [kw[0] for kw in keyword_matches]
                    
                # Add article
                article_id = add_article_with_playwright_data(
                    url=article_url,
                    title=title,
                    publication_date=publication_date,
                    source_website=source_name,
                    category=category,
                    content_summary=content[:1000],
                    full_content=content[:5000],
                    author=author,
                    keywords_found=keywords_found,
                    rendering_method='playwright',
                    screenshot_path=article_data.get('screenshot_path'),
                    js_errors=article_data.get('js_errors'),
                    page_metrics=article_data.get('metrics')
                )
                
                if article_id:
                    new_articles_count += 1
                    
        else:
            # Fall back to regular parsing
            logger.debug(f"No JavaScript required for {source_name}, using regular parser")
            return _original_parse_html_site(site_config, run_id)
            
    return new_articles_count


# Wrapper for synchronous context
def parse_html_site_enhanced(site_config: Dict, run_id: int) -> int:
    """
    Enhanced HTML parser that uses Playwright when needed.
    """
    if PLAYWRIGHT_ENABLED and PLAYWRIGHT_AVAILABLE:
        # Check if this site might need Playwright
        if site_config.get('js_required', False):
            # Run async function in sync context
            import asyncio
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    parse_html_site_with_playwright_async(site_config, run_id)
                )
                return result
            except Exception as e:
                logger.error(f"Playwright failed for {site_config['name']}: {e}")
                logger.info("Falling back to regular parser")
            finally:
                loop.close()
                
    # Use original parser
    return _original_parse_html_site(site_config, run_id)


# Override the parse_html_site function
parse_html_site = parse_html_site_enhanced


# Enhanced run_scraper with Playwright statistics
@run_with_playwright
def run_scraper_enhanced():
    """Enhanced scraper with Playwright support."""
    return _original_run_scraper()


# Override the run_scraper function if Playwright is available
if PLAYWRIGHT_AVAILABLE and PLAYWRIGHT_ENABLED:
    run_scraper = run_scraper_enhanced
    logger.info("Playwright enhancement activated for run_scraper")


# Add new command line option for Playwright setup
def main():
    """Enhanced main function with Playwright options."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--setup-playwright":
            print("Setting up Playwright browsers...")
            import subprocess
            subprocess.run([sys.executable, "setup_playwright.py"])
            return
        elif sys.argv[1] == "--check-playwright":
            print(f"Playwright available: {PLAYWRIGHT_AVAILABLE}")
            print(f"Playwright enabled: {PLAYWRIGHT_ENABLED}")
            
            if PLAYWRIGHT_AVAILABLE:
                from playwright_scraper import PlaywrightScraper
                scraper = PlaywrightScraper()
                print(f"Browsers configured: {list(scraper.browser_configs.keys())}")
            return
            
    # Run original main
    if __name__ == "__main__":
        original_args = sys.argv[1:]
        
        if "--run" in original_args:
            run_scraper()
        elif "--schedule" in original_args:
            print("Scheduler gestart. Druk Ctrl+C om te stoppen.")
            schedule.every(4).hours.do(run_scraper)
            try:
                while True:
                    schedule.run_pending()
                    time.sleep(60)
            except KeyboardInterrupt:
                print("\nScheduler gestopt.")
        elif "--stats" in original_args:
            show_statistics()
        elif "--test-email" in original_args:
            test_email()
        else:
            print("Gebruik:")
            print("  python scraper.py --run           # Eenmalig draaien")
            print("  python scraper.py --schedule      # Elke 4 uur draaien")
            print("  python scraper.py --stats         # Toon statistieken")
            print("  python scraper.py --test-email    # Test email configuratie")
            print("\nPlaywright opties:")
            print("  python scraper.py --setup-playwright  # Setup Playwright browsers")
            print("  python scraper.py --check-playwright  # Check Playwright status")


if __name__ == "__main__":
    main()