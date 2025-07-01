#!/usr/bin/env python3
"""
Quick script to populate some data for testing the Netflix UI
"""

import subprocess
import sys
import time

def main():
    print("=" * 60)
    print("Populating Mimir with initial data...")
    print("=" * 60)
    
    print("\nRunning scraper to fetch articles...")
    print("This may take a minute...\n")
    
    try:
        # Run the scraper
        result = subprocess.run([sys.executable, 'scraper.py', '--run'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Scraper completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Scraper encountered an error:")
            print(result.stderr)
            
    except FileNotFoundError:
        print("❌ Could not find scraper.py")
        print("Make sure you're in the Mimir directory")
    except Exception as e:
        print(f"❌ Error running scraper: {e}")
    
    print("\n" + "=" * 60)
    print("Data population complete!")
    print("You can now run: python start_netflix_ui.py")
    print("=" * 60)

if __name__ == '__main__':
    main()