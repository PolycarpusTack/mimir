#!/usr/bin/env python3
"""
Quick start script for Mimir News Scraper - Windows optimized version
Provides an interactive menu to run different components of the system.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python():
    """Check if Python is available and get version."""
    try:
        version = sys.version.split()[0]
        print(f"✅ Python {version} found")
        return True
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False

def install_minimal_packages():
    """Install minimal required packages."""
    print("\n📦 Installing minimal required packages...")
    
    try:
        # Check if requirements_minimal.txt exists
        req_file = Path("requirements_minimal.txt")
        if not req_file.exists():
            print("❌ requirements_minimal.txt not found")
            return False
        
        # Install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_minimal.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Minimal packages installed successfully")
            return True
        else:
            print(f"❌ Package installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_database():
    """Set up the database."""
    print("\n🗄️ Setting up database...")
    
    # Check if we should use SQLite or PostgreSQL
    use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
    
    if use_postgres:
        print("📊 PostgreSQL mode detected")
        try:
            from db_manager_postgres import PostgreSQLManager
            db = PostgreSQLManager()
            if db.test_connection():
                print("✅ PostgreSQL connection successful")
                return True
            else:
                print("❌ PostgreSQL connection failed")
                return False
        except ImportError:
            print("❌ PostgreSQL dependencies not available")
            print("💡 Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"❌ PostgreSQL setup failed: {e}")
            return False
    else:
        print("📁 SQLite mode - creating database...")
        try:
            from db_manager import DatabaseManager
            db = DatabaseManager()
            db.create_tables()
            print("✅ SQLite database created successfully")
            return True
        except Exception as e:
            print(f"❌ SQLite setup failed: {e}")
            return False

def run_scraper():
    """Run the news scraper."""
    print("\n🔍 Starting news scraper...")
    
    try:
        import scraper
        print("✅ Scraper module loaded")
        
        # Run scraper with basic configuration
        news_scraper = scraper.NewsScraper()
        
        # Check if sites configuration exists
        sites_file = Path("sites_to_scrape.json")
        if not sites_file.exists():
            create_sample_sites_config()
        
        print("🚀 Running scraper...")
        articles = news_scraper.scrape_all_sites()
        print(f"✅ Scraped {len(articles)} articles")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Try installing: pip install requests beautifulsoup4 feedparser")
        return False
    except Exception as e:
        print(f"❌ Scraper failed: {e}")
        return False

def start_web_interface():
    """Start the Flask web interface."""
    print("\n🌐 Starting web interface...")
    
    try:
        # Set environment variables
        os.environ["FLASK_ENV"] = "development"
        os.environ["FLASK_DEBUG"] = "1"
        
        # Import and run
        import web_interface
        print("✅ Web interface module loaded")
        print("🚀 Starting Flask server on http://localhost:5000")
        print("📝 Press Ctrl+C to stop the server")
        
        web_interface.app.run(host='0.0.0.0', port=5000, debug=True)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Try installing: pip install flask flask-cors")
        return False
    except Exception as e:
        print(f"❌ Web interface failed: {e}")
        return False

def start_api_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting API server...")
    
    try:
        # Check if FastAPI is available
        import uvicorn
        print("✅ FastAPI/Uvicorn available")
        
        print("🚀 Starting API server on http://localhost:8000")
        print("📚 API docs will be available at http://localhost:8000/docs")
        print("📝 Press Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ FastAPI not available: {e}")
        print("💡 Try installing: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ API server failed: {e}")
        return False

def test_clustering():
    """Test the clustering functionality."""
    print("\n🔍 Testing clustering functionality...")
    
    try:
        from article_clustering import ArticleClusteringEngine
        
        engine = ArticleClusteringEngine()
        print("✅ Clustering engine loaded")
        
        # Test with sample data
        sample_articles = [
            {
                'id': 'test_1',
                'title': 'AI Technology Advances',
                'content': 'Artificial intelligence and machine learning continue to advance...',
                'source': 'TechNews',
                'published_date': '2024-01-01T00:00:00'
            },
            {
                'id': 'test_2', 
                'title': 'Business Market Update',
                'content': 'Stock markets showed significant movement this week...',
                'source': 'BusinessDaily',
                'published_date': '2024-01-01T00:00:00'
            }
        ]
        
        result = engine.cluster_by_topic(sample_articles, n_clusters=2, method='kmeans')
        
        if result['success']:
            print(f"✅ Clustering test successful: {result['n_clusters']} clusters created")
            return True
        else:
            print(f"❌ Clustering test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Clustering dependencies missing: {e}")
        print("💡 Try installing: pip install numpy pandas scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Clustering test failed: {e}")
        return False

def test_recommendations():
    """Test the recommendation functionality."""
    print("\n🎯 Testing recommendation functionality...")
    
    try:
        from recommendation_engine import RecommendationEngine
        
        engine = RecommendationEngine()
        print("✅ Recommendation engine loaded")
        
        # Build models with sample data
        result = engine.build_comprehensive_recommendation_system()
        
        if result['success']:
            methods = result.get('available_methods', [])
            print(f"✅ Recommendation test successful: {', '.join(methods)} methods available")
            return True
        else:
            print(f"❌ Recommendation test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except ImportError as e:
        print(f"❌ Recommendation dependencies missing: {e}")
        print("💡 Try installing: pip install numpy pandas scikit-learn")
        return False
    except Exception as e:
        print(f"❌ Recommendation test failed: {e}")
        return False

def view_statistics():
    """View database statistics."""
    print("\n📊 Database Statistics:")
    
    try:
        # Try PostgreSQL first, then SQLite
        use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
        
        if use_postgres:
            from db_manager_postgres import PostgreSQLManager
            db = PostgreSQLManager()
        else:
            from db_manager import DatabaseManager
            db = DatabaseManager()
        
        # Get article count
        try:
            if hasattr(db, 'count_articles'):
                article_count = db.count_articles()
            else:
                # Fallback query
                result = db.execute_query("SELECT COUNT(*) FROM articles")
                article_count = result[0][0] if result else 0
            
            print(f"📰 Total articles: {article_count}")
            
            # Get source count
            try:
                result = db.execute_query("SELECT COUNT(DISTINCT source) FROM articles")
                source_count = result[0][0] if result else 0
                print(f"📡 Total sources: {source_count}")
            except:
                print("📡 Source count: Unable to retrieve")
            
            # Get recent articles
            try:
                result = db.execute_query(
                    "SELECT COUNT(*) FROM articles WHERE scraped_at > datetime('now', '-24 hours')"
                )
                recent_count = result[0][0] if result else 0
                print(f"🕐 Articles in last 24h: {recent_count}")
            except:
                print("🕐 Recent articles: Unable to retrieve")
            
            return True
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Database module not available: {e}")
        return False

def create_sample_sites_config():
    """Create a sample sites configuration file."""
    print("📝 Creating sample sites configuration...")
    
    sample_config = {
        "sites": [
            {
                "name": "Reuters Technology",
                "url": "https://www.reuters.com/technology/",
                "type": "rss",
                "enabled": True,
                "category": "technology"
            },
            {
                "name": "BBC News",
                "url": "http://feeds.bbci.co.uk/news/rss.xml",
                "type": "rss", 
                "enabled": True,
                "category": "general"
            }
        ]
    }
    
    try:
        import json
        with open("sites_to_scrape.json", "w") as f:
            json.dump(sample_config, f, indent=2)
        print("✅ Sample sites configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        return False

def create_sample_env():
    """Create a sample .env file."""
    print("📝 Creating sample environment configuration...")
    
    env_content = """# Database Configuration
USE_POSTGRES=false
SQLITE_DB_PATH=./mimir.db

# For PostgreSQL (when USE_POSTGRES=true):
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=mimir_news
# POSTGRES_USER=mimir_user
# POSTGRES_PASSWORD=your_password

# API Configuration
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# Scraping Configuration
USER_AGENT=Mimir News Scraper 1.0
REQUEST_DELAY=1
MAX_RETRIES=3
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Sample .env file created")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def show_menu():
    """Show the main menu."""
    print("\n" + "="*50)
    print("🚀 MIMIR NEWS SCRAPER - QUICK START")
    print("="*50)
    
    print("\n📋 Available Options:")
    print("1. 📦 Install minimal packages")
    print("2. 🗄️ Setup database")
    print("3. 📝 Create sample configuration")
    print("4. 🔍 Run news scraper")
    print("5. 🌐 Start web interface")
    print("6. 🚀 Start API server")
    print("7. 🧪 Test clustering")
    print("8. 🎯 Test recommendations")
    print("9. 📊 View statistics")
    print("0. ❌ Exit")
    
    return input("\n🔢 Choose an option (0-9): ").strip()

def main():
    """Main function."""
    # Check Python first
    if not check_python():
        input("\n❌ Press Enter to exit...")
        return
    
    # Show system info
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    while True:
        try:
            choice = show_menu()
            
            if choice == "0":
                print("\n👋 Goodbye!")
                break
            elif choice == "1":
                install_minimal_packages()
            elif choice == "2":
                setup_database()
            elif choice == "3":
                create_sample_sites_config()
                create_sample_env()
            elif choice == "4":
                run_scraper()
            elif choice == "5":
                start_web_interface()
            elif choice == "6":
                start_api_server()
            elif choice == "7":
                test_clustering()
            elif choice == "8":
                test_recommendations()
            elif choice == "9":
                view_statistics()
            else:
                print("❌ Invalid option. Please choose 0-9.")
            
            if choice in ["5", "6"]:  # Web interface or API server
                break  # These are blocking operations
            
            input("\n⏸️ Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            input("⏸️ Press Enter to continue...")

if __name__ == "__main__":
    main()