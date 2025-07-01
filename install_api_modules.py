#!/usr/bin/env python3
"""
Install API modules for Mimir
Installs optional API packages for NewsAPI, Reddit, etc.
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])
        print(f"✅ Installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        print(f"   Try: pip install --user {package_name}")
        return False

def main():
    print("🚀 Mimir API Modules Installation\n")
    print("This will install optional API packages for enhanced news aggregation.\n")
    
    packages = {
        'newsapi-python': 'NewsAPI.org client (80,000+ news sources)',
        'praw': 'Reddit API wrapper (community discussions)',
        'yake': 'Keyword extraction (better filtering)',
        'pytrends': 'Google Trends data (trend analysis)'
    }
    
    print("📦 Packages to install:")
    for pkg, desc in packages.items():
        print(f"   - {pkg}: {desc}")
    
    response = input("\nProceed with installation? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    print("\n📥 Installing packages...\n")
    
    success = []
    failed = []
    
    for package, description in packages.items():
        print(f"Installing {package}...")
        if install_package(package):
            success.append(package)
        else:
            failed.append(package)
    
    print("\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {len(success)} packages")
    if success:
        print(f"   {', '.join(success)}")
    
    if failed:
        print(f"\n❌ Failed to install: {len(failed)} packages")
        print(f"   {', '.join(failed)}")
        print("\n💡 Manual installation:")
        for pkg in failed:
            print(f"   pip install --user {pkg}")
    
    print("\n⚙️  Next steps:")
    print("1. Get your free API keys:")
    print("   - NewsAPI: https://newsapi.org/register")
    print("   - Reddit: https://www.reddit.com/prefs/apps")
    print("\n2. Configure APIs in the web interface:")
    print("   - Start Mimir: python start_netflix_ui.py")
    print("   - Go to Settings > API Sources")
    print("   - Enter your API keys")
    print("\n3. The scraper will automatically use configured APIs!")

if __name__ == "__main__":
    main()