#!/usr/bin/env python3
"""
Quick start script for the Netflix-style Mimir interface
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Timer

def check_requirements():
    """Check if all required packages are installed"""
    # Web interface requirements
    web_required = ['flask', 'flask-cors']
    # Scraper requirements
    scraper_required = ['feedparser', 'beautifulsoup4', 'requests', 'lxml']
    
    all_required = web_required + scraper_required
    missing = []
    
    for package in all_required:
        try:
            # Handle package name variations
            import_name = package.replace('-', '_')
            if package == 'beautifulsoup4':
                import_name = 'bs4'
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Attempting to install missing packages...")
        
        try:
            # Try with --user flag first
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user'] + missing)
            print("Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("\n⚠️  Could not install packages automatically.")
            print("\nPlease install manually with one of these commands:")
            print(f"\n1. With user flag (recommended):")
            print(f"   python -m pip install --user {' '.join(missing)}")
            print(f"\n2. With admin privileges:")
            print(f"   python -m pip install {' '.join(missing)}")
            print(f"\n3. In a virtual environment:")
            print(f"   python -m venv venv")
            print(f"   venv\\Scripts\\activate  (Windows)")
            print(f"   pip install {' '.join(missing)}")
            print("\nPress Enter to exit...")
            input()
            sys.exit(1)

def check_database():
    """Check if database exists, create if not"""
    if not os.path.exists('news_scraper.db'):
        print("Database not found. Creating new database...")
        subprocess.run([sys.executable, 'db_manager.py'], check=True)
        print("Database created successfully!")
    else:
        print("Database found.")

def open_browser():
    """Open the browser after a delay"""
    print("\nOpening browser...")
    webbrowser.open('http://localhost:5000')

def main():
    print("=" * 60)
    print("🎬 Mimir Netflix-Style Interface Launcher")
    print("=" * 60)
    
    # Check requirements
    print("\nChecking requirements...")
    check_requirements()
    
    # Check database
    print("\nChecking database...")
    check_database()
    
    # Start the web interface
    print("\nStarting Netflix-style web interface...")
    print("-" * 60)
    print("Access the interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Open browser after 2 seconds
    timer = Timer(2.0, open_browser)
    timer.daemon = True
    timer.start()
    
    # Run the Flask app
    try:
        # Check if fixed version exists
        if os.path.exists('web_interface_fixed.py'):
            print("\n✅ Using fixed web interface (no JavaScript errors)")
            subprocess.run([sys.executable, 'web_interface_fixed.py'])
        else:
            subprocess.run([sys.executable, 'web_interface.py'])
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("Thank you for using Mimir!")

if __name__ == '__main__':
    main()