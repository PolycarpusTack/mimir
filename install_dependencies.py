#!/usr/bin/env python3
"""
Quick dependency installer for Mimir
Installs only the essential packages needed to run the basic scraper and web interface
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("Mimir Dependency Installer")
    print("=" * 60)
    
    # Essential packages only
    essential_packages = [
        'flask',
        'flask-cors', 
        'requests',
        'beautifulsoup4',
        'lxml',
        'feedparser',
        'python-dateutil',
        'schedule'
    ]
    
    print("\nInstalling essential packages for Mimir...")
    print("Packages:", ', '.join(essential_packages))
    print()
    
    try:
        # Try with --user flag first
        cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + essential_packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All essential packages installed successfully!")
            print("\nYou can now run:")
            print("  python start_netflix_ui.py")
        else:
            print("❌ Installation failed. Trying without --user flag...")
            cmd = [sys.executable, '-m', 'pip', 'install'] + essential_packages
            subprocess.check_call(cmd)
            print("✅ Packages installed successfully!")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("\nPlease install manually:")
        print(f"  pip install {' '.join(essential_packages)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()