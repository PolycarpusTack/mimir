#!/usr/bin/env python3
"""
Robots.txt Checker for Mimir News Scraper
Checks if URLs are allowed by robots.txt before adding them to scraper
"""

import sys
import urllib.robotparser
from urllib.parse import urlparse, urljoin
import requests
import json

def check_robots_txt(url, user_agent='*'):
    """Check if a URL is allowed by robots.txt"""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        print(f"Checking robots.txt: {robots_url}")
        
        # Download robots.txt
        response = requests.get(robots_url, timeout=10)
        if response.status_code == 404:
            print("✅ No robots.txt found - scraping allowed")
            return True
            
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        
        can_fetch = rp.can_fetch(user_agent, url)
        
        if can_fetch:
            print("✅ URL is allowed by robots.txt")
        else:
            print("❌ URL is BLOCKED by robots.txt")
            
        # Show relevant robots.txt rules
        print(f"\nRobots.txt content preview:")
        lines = response.text.split('\n')[:20]  # First 20 lines
        for line in lines:
            if line.strip():
                print(f"  {line}")
                
        return can_fetch
        
    except Exception as e:
        print(f"⚠️  Error checking robots.txt: {e}")
        return None

def suggest_rss_feeds():
    """Suggest robots.txt-friendly RSS feeds"""
    
    reliable_feeds = [
        {
            "name": "Ars Technica",
            "url": "https://feeds.arstechnica.com/arstechnica/index",
            "category": "technology"
        },
        {
            "name": "Wired",
            "url": "https://www.wired.com/feed/",
            "category": "technology"
        },
        {
            "name": "Engadget",
            "url": "https://www.engadget.com/rss.xml",
            "category": "technology"
        },
        {
            "name": "TechRadar",
            "url": "https://www.techradar.com/rss",
            "category": "technology"
        },
        {
            "name": "ZDNet",
            "url": "https://www.zdnet.com/news/rss.xml",
            "category": "technology"
        },
        {
            "name": "Hacker News",
            "url": "https://hnrss.org/frontpage",
            "category": "technology"
        },
        {
            "name": "MIT Technology Review",
            "url": "https://www.technologyreview.com/feed/",
            "category": "technology"
        },
        {
            "name": "Slashdot",
            "url": "http://rss.slashdot.org/Slashdot/slashdotMain",
            "category": "technology"
        }
    ]
    
    print("\n🔍 Checking reliability of suggested RSS feeds:\n")
    
    for feed in reliable_feeds:
        print(f"📰 {feed['name']}")
        print(f"   URL: {feed['url']}")
        result = check_robots_txt(feed['url'])
        if result is True:
            print("   Status: ✅ SAFE TO USE")
        elif result is False:
            print("   Status: ❌ BLOCKED")
        else:
            print("   Status: ⚠️  UNKNOWN")
        print()

def check_sites_file():
    """Check robots.txt status for all sites in sites_to_scrape.json"""
    try:
        with open('sites_to_scrape.json', 'r') as f:
            data = json.load(f)
            sites = data.get('sites', [])
            
        print("🔍 Checking robots.txt for all configured sites:\n")
        
        for site in sites:
            print(f"📰 {site['name']}")
            print(f"   URL: {site['url']}")
            print(f"   Enabled: {site.get('enabled', 'unknown')}")
            
            result = check_robots_txt(site['url'])
            if result is True:
                print("   Robots.txt: ✅ ALLOWED")
            elif result is False:
                print("   Robots.txt: ❌ BLOCKED")
            else:
                print("   Robots.txt: ⚠️  UNKNOWN")
            print()
            
    except FileNotFoundError:
        print("❌ sites_to_scrape.json not found")
    except Exception as e:
        print(f"❌ Error reading sites file: {e}")

if __name__ == "__main__":
    print("🤖 Mimir Robots.txt Checker\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "suggest":
            suggest_rss_feeds()
        elif sys.argv[1] == "check":
            check_sites_file()
        else:
            # Check specific URL
            url = sys.argv[1]
            check_robots_txt(url)
    else:
        print("Usage:")
        print("  python3 check_robots.py <url>          # Check specific URL")
        print("  python3 check_robots.py suggest        # Show reliable RSS feeds")
        print("  python3 check_robots.py check          # Check all configured sites")