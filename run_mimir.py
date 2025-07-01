#!/usr/bin/env python3
"""
Mimir Startup Script

Simple script to start Mimir with common configurations.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main entry point."""
    # Change to the project directory
    project_dir = Path(__file__).parent
    
    if len(sys.argv) == 1:
        # Default: Start the API server in development mode
        print("🚀 Starting Mimir API server in development mode...")
        print("   Access the API at: http://localhost:8000")
        print("   API Documentation: http://localhost:8000/docs")
        print("   Press Ctrl+C to stop")
        print()
        
        cmd = [sys.executable, "-m", "mimir.cli.main", "serve", "--reload", "--host", "0.0.0.0", "--port", "8000"]
        
    elif sys.argv[1] == "scrape":
        # Run scraping operation
        print("📰 Running scraping operation...")
        cmd = [sys.executable, "-m", "mimir.cli.main", "scrape"]
        
    elif sys.argv[1] == "init":
        # Initialize database
        print("🔧 Initializing database...")
        cmd = [sys.executable, "-m", "mimir.cli.main", "init-db"]
        
    elif sys.argv[1] == "health":
        # Health check
        print("🏥 Running health check...")
        cmd = [sys.executable, "-m", "mimir.cli.main", "health"]
        
    elif sys.argv[1] == "config":
        # Show configuration
        cmd = [sys.executable, "-m", "mimir.cli.main", "config"]
        
    elif sys.argv[1] == "production":
        # Start in production mode
        print("🏭 Starting Mimir API server in production mode...")
        print("   Access the API at: http://localhost:8000")
        print("   Press Ctrl+C to stop")
        print()
        
        cmd = [sys.executable, "-m", "mimir.cli.main", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
        
    elif sys.argv[1] == "help":
        # Show help
        print_help()
        return
        
    else:
        print(f"❌ Unknown command: {sys.argv[1]}")
        print_help()
        return
    
    try:
        subprocess.run(cmd, cwd=project_dir, check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def print_help():
    """Print help message."""
    print("Mimir Broadcasting Intelligence Platform")
    print("=" * 50)
    print()
    print("Usage: python run_mimir.py [command]")
    print()
    print("Commands:")
    print("  (no command)  Start API server in development mode")
    print("  scrape        Run scraping operation")
    print("  init          Initialize database")
    print("  health        Run health check")
    print("  config        Show current configuration")
    print("  production    Start API server in production mode")
    print("  help          Show this help message")
    print()
    print("Examples:")
    print("  python run_mimir.py              # Start dev server")
    print("  python run_mimir.py scrape       # Run scraping")
    print("  python run_mimir.py init         # Initialize database")
    print("  python run_mimir.py production   # Start production server")


if __name__ == "__main__":
    main()