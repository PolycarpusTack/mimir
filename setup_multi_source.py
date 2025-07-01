#!/usr/bin/env python3
"""
Setup script for Mimir Multi-Source Aggregator
Installs required packages and configures APIs
"""

import subprocess
import sys
import os
import json

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])
        print(f"✅ Installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def main():
    print("🚀 Mimir Multi-Source Aggregator Setup\n")
    
    # Required packages
    packages = {
        # Core packages (already installed)
        'feedparser': 'RSS feed parsing',
        'requests': 'HTTP requests',
        'beautifulsoup4': 'HTML parsing',
        
        # API packages
        'newsapi-python': 'NewsAPI client',
        'praw': 'Reddit API wrapper',
        'arxiv': 'arXiv research papers',
        
        # Analysis packages
        'yake': 'Keyword extraction',
        'textblob': 'Sentiment analysis',
        'pytrends': 'Google Trends unofficial API',
        
        # Optional but recommended
        'pandas': 'Data analysis',
        'matplotlib': 'Visualization'
    }
    
    print("📦 Installing required packages...\n")
    
    failed = []
    for package, description in packages.items():
        print(f"Installing {package} ({description})...")
        if not install_package(package):
            failed.append(package)
    
    print("\n📋 Installation Summary:")
    print(f"✅ Successfully installed: {len(packages) - len(failed)} packages")
    if failed:
        print(f"❌ Failed to install: {', '.join(failed)}")
    
    # Create API configuration template
    print("\n⚙️  Creating API configuration template...")
    
    api_config = {
        "newsapi": {
            "api_key": "YOUR_NEWSAPI_KEY_HERE",
            "enabled": False,
            "docs": "Get free key at https://newsapi.org/register"
        },
        "reddit": {
            "client_id": "YOUR_REDDIT_CLIENT_ID",
            "client_secret": "YOUR_REDDIT_CLIENT_SECRET",
            "user_agent": "Mimir News Aggregator 1.0",
            "enabled": False,
            "docs": "Create app at https://www.reddit.com/prefs/apps"
        },
        "rss_feeds": [
            {"name": "TechCrunch", "url": "https://techcrunch.com/feed/"},
            {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
            {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
            {"name": "Engadget", "url": "https://www.engadget.com/rss.xml"},
            {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
            {"name": "BBC Technology", "url": "http://feeds.bbci.co.uk/news/technology/rss.xml"},
            {"name": "Reuters Technology", "url": "https://www.reuters.com/technology/rss"},
            {"name": "The Hollywood Reporter", "url": "https://www.hollywoodreporter.com/feed"},
            {"name": "Variety", "url": "https://variety.com/feed/"},
            {"name": "Deadline", "url": "https://deadline.com/feed/"},
            {"name": "TV Technology", "url": "https://www.tvtechnology.com/rss"},
            {"name": "Radio World", "url": "https://www.radioworld.com/feed"},
            {"name": "Digital TV Europe", "url": "https://www.digitaltveurope.com/feed/"},
            {"name": "C21Media", "url": "https://www.c21media.net/feed/"}
        ]
    }
    
    config_file = 'api_config.json'
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(api_config, f, indent=2)
        print(f"✅ Created {config_file}")
    else:
        print(f"⚠️  {config_file} already exists, skipping...")
    
    print("\n🎯 Next Steps:")
    print("1. Get free API keys:")
    print("   - NewsAPI: https://newsapi.org/register (100 requests/day free)")
    print("   - Reddit: https://www.reddit.com/prefs/apps (create 'script' app)")
    print("\n2. Edit api_config.json with your API keys")
    print("\n3. Test the multi-source aggregator:")
    print("   python multi_source_aggregator.py")
    print("\n4. Run trend analysis:")
    print("   python trend_predictor.py")
    
    print("\n💡 Pro Tips:")
    print("- RSS feeds work without any API keys")
    print("- Start with RSS feeds, then add APIs as needed")
    print("- NewsAPI covers 80,000+ sources with one key")
    print("- Reddit API is great for real-time community sentiment")

if __name__ == "__main__":
    main()