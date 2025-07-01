#!/usr/bin/env python
"""
Playwright Setup Script for Mimir
Installs Playwright browsers and validates the installation
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_playwright_installed():
    """Check if playwright is installed via pip."""
    try:
        import playwright
        logger.info(f"✅ Playwright version {playwright.__version__} is installed")
        return True
    except ImportError:
        logger.error("❌ Playwright is not installed. Run: pip install -r requirements.txt")
        return False


def install_browsers():
    """Install Playwright browsers (Chromium, Firefox, WebKit)."""
    logger.info("🌐 Installing Playwright browsers...")
    
    try:
        # Install browsers using playwright CLI
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Browsers installed successfully")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Failed to install browsers: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing browsers: {str(e)}")
        return False


def install_system_dependencies():
    """Install system dependencies for Playwright (Linux only)."""
    if sys.platform.startswith('linux'):
        logger.info("🔧 Installing system dependencies for Linux...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install-deps"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ System dependencies installed")
                return True
            else:
                logger.warning(f"⚠️  Could not install system dependencies: {result.stderr}")
                logger.info("   You may need to run with sudo or install manually")
                return True  # Non-fatal on failure
                
        except Exception as e:
            logger.warning(f"⚠️  Error installing system dependencies: {str(e)}")
            return True  # Non-fatal
    
    return True


def create_playwright_config():
    """Create playwright configuration file."""
    # Check if file already exists
    config_path = Path("playwright_config.json")
    
    if config_path.exists():
        logger.info("ℹ️  playwright_config.json already exists, verifying content...")
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
            # Verify it has required structure
            if 'browsers' in existing_config and 'defaults' in existing_config:
                logger.info("✅ playwright_config.json is valid")
                return True
            else:
                logger.warning("⚠️  Existing config invalid, will recreate")
        except Exception as e:
            logger.warning(f"⚠️  Could not read existing config: {e}, will recreate")
    
    # Create the config with proper structure
    config = {
        "browsers": {
            "chromium": {
                "enabled": True,
                "headless": True,
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-gpu"
                ]
            },
            "firefox": {
                "enabled": False,
                "headless": True
            },
            "webkit": {
                "enabled": False,
                "headless": True
            }
        },
        "defaults": {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "locale": "en-US",
            "timezone": "America/New_York"
        },
        "timeouts": {
            "page_load": 30000,
            "element_wait": 10000,
            "script_execution": 5000,
            "navigation": 30000
        },
        "performance": {
            "block_resources": ["image", "font", "media"],
            "intercept_requests": True
        },
        "stealth": {
            "enabled": True
        },
        "screenshots": {
            "enabled": True,
            "on_error": True,
            "path": "logs/playwright_screenshots"
        },
        "error_handling": {
            "max_retries": 3,
            "retry_delay": 2000,
            "continue_on_error": True
        }
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("✅ Created playwright_config.json")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create config: {str(e)}")
        return False


def update_main_config():
    """Add Playwright settings to main config.json if not present."""
    config_path = Path("config.json")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if Playwright settings already exist
        if 'playwright' in config:
            logger.info("ℹ️  Playwright settings already exist in config.json")
            return True
        
        # Add Playwright settings with same structure as other features
        config['playwright'] = {
            "enabled": False,  # Disabled by default for safety
            "config_file": "playwright_config.json",
            "browser_type": "chromium",
            "headless": True,
            "timeout_seconds": 30,
            "max_concurrent_contexts": 3,
            "fallback_to_html": True,
            "screenshot_on_error": True,
            "auto_detect_js": True,
            "js_indicators": [
                "react-root",
                "ng-app", 
                "vue-app",
                "__NEXT_DATA__",
                "window.React",
                "window.angular"
            ],
            "default_wait_strategy": "domcontentloaded",
            "page_load_timeout": 30000,
            "element_timeout": 10000
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("✅ Updated config.json with Playwright settings")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config.json: {str(e)}")
        return False


def test_playwright():
    """Run a simple test to verify Playwright works."""
    logger.info("🧪 Testing Playwright installation...")
    
    test_script = '''
import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://example.com")
        title = await page.title()
        await browser.close()
        return title

try:
    title = asyncio.run(test())
    print(f"✅ Test successful! Page title: {title}")
    exit(0)
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
'''
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(result.stdout.strip())
            return True
        else:
            logger.error(f"Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run test: {str(e)}")
        return False


def create_directories():
    """Create necessary directories for Playwright."""
    directories = [
        "logs/playwright_screenshots",
        "logs/playwright_traces"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Created Playwright directories")
    return True


def main():
    """Main setup function."""
    print("\n🎭 PLAYWRIGHT SETUP FOR MIMIR")
    print("=" * 50)
    
    steps = [
        ("Checking Playwright installation", check_playwright_installed),
        ("Installing browsers", install_browsers),
        ("Installing system dependencies", install_system_dependencies),
        ("Creating configuration file", create_playwright_config),
        ("Updating main config", update_main_config),
        ("Creating directories", create_directories),
        ("Testing Playwright", test_playwright)
    ]
    
    all_success = True
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        success = step_func()
        if not success and step_name not in ["Installing system dependencies"]:
            all_success = False
            break
    
    print("\n" + "=" * 50)
    
    if all_success:
        print("✅ PLAYWRIGHT SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Review playwright_config.json for your needs")
        print("2. Set 'enabled': true in config.json when ready")
        print("3. Run 'python playwright_scraper.py' to test")
    else:
        print("❌ SETUP FAILED - Please check the errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()