#!/usr/bin/env python3
"""Setup script for Analytics & Business Intelligence (EPIC 6).

This script automates the setup of analytics infrastructure including
TimescaleDB, required Python packages, and database initialization.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyticsSetup:
    """Setup manager for analytics infrastructure."""

    def __init__(self):
        """Initialize the setup manager."""
        self.project_root = Path(__file__).parent
        self.docker_compose_file = self.project_root / "docker-compose.timescale.yml"
        
    def check_requirements(self) -> bool:
        """Check if all required tools are available."""
        logger.info("Checking requirements...")
        
        required_tools = ['docker', 'docker-compose', 'python3', 'pip']
        missing_tools = []
        
        for tool in required_tools:
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
                if result.returncode != 0:
                    missing_tools.append(tool)
                else:
                    logger.info(f"✅ {tool} is available")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            logger.error(f"❌ Missing required tools: {', '.join(missing_tools)}")
            logger.error("Please install the missing tools and try again.")
            return False
            
        logger.info("✅ All requirements satisfied")
        return True
    
    def install_python_dependencies(self) -> bool:
        """Install required Python packages."""
        logger.info("Installing Python dependencies...")
        
        try:
            # Install analytics-specific dependencies
            analytics_packages = [
                'prophet>=1.1.5',
                'ruptures>=1.1.9', 
                'scipy>=1.11.4',
                'statsmodels>=0.14.1',
                'matplotlib>=3.8.2',
                'seaborn>=0.13.0',
                'plotly>=5.17.0',
                'umap-learn>=0.5.5',
                'hdbscan>=0.8.33',
                'reportlab>=4.0.7',
                'jinja2>=3.1.2',
                'weasyprint>=60.2',
                'category-encoders>=2.6.3',
                'feature-engine>=1.6.2',
                'imbalanced-learn>=0.11.0'
            ]
            
            for package in analytics_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"Failed to install {package}: {result.stderr}")
                    # Continue with other packages
                else:
                    logger.info(f"✅ Installed {package}")
            
            logger.info("✅ Python dependencies installation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install Python dependencies: {e}")
            return False
    
    def setup_timescaledb(self) -> bool:
        """Setup TimescaleDB using Docker Compose."""
        logger.info("Setting up TimescaleDB...")
        
        try:
            # Check if docker-compose file exists
            if not self.docker_compose_file.exists():
                logger.error(f"❌ Docker Compose file not found: {self.docker_compose_file}")
                return False
            
            # Stop existing containers if any
            logger.info("Stopping any existing containers...")
            subprocess.run([
                'docker-compose', '-f', str(self.docker_compose_file), 'down'
            ], capture_output=True)
            
            # Start TimescaleDB and other services
            logger.info("Starting TimescaleDB and supporting services...")
            result = subprocess.run([
                'docker-compose', '-f', str(self.docker_compose_file), 'up', '-d'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to start services: {result.stderr}")
                return False
            
            logger.info("✅ Services started successfully")
            
            # Wait for services to be ready
            logger.info("Waiting for services to be ready...")
            max_retries = 30
            retry_count = 0
            
            while retry_count < max_retries:
                result = subprocess.run([
                    'docker-compose', '-f', str(self.docker_compose_file), 
                    'exec', '-T', 'timescaledb', 
                    'pg_isready', '-U', 'mimir_user', '-d', 'mimir_analytics'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ TimescaleDB is ready")
                    break
                    
                retry_count += 1
                logger.info(f"Waiting for TimescaleDB... ({retry_count}/{max_retries})")
                time.sleep(2)
            
            if retry_count >= max_retries:
                logger.error("❌ TimescaleDB failed to become ready")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup TimescaleDB: {e}")
            return False
    
    def initialize_database_schema(self) -> bool:
        """Initialize the analytics database schema."""
        logger.info("Initializing database schema...")
        
        try:
            # The schema should be automatically initialized via docker-entrypoint-initdb.d
            # Let's verify that the schema exists
            
            # Test database connection and schema
            test_script = '''
import sys
sys.path.append(".")
try:
    from analytics.data_warehouse import AnalyticsDataWarehouse
    warehouse = AnalyticsDataWarehouse()
    with warehouse.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'analytics'")
        if cursor.fetchone():
            print("SUCCESS: Analytics schema found")
        else:
            print("ERROR: Analytics schema not found")
            sys.exit(1)
        
        # Check if tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'analytics' 
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(tables)} analytics tables: {', '.join(tables)}")
        
        # Check TimescaleDB hypertables
        cursor.execute("""
            SELECT hypertable_name FROM timescaledb_information.hypertables
            WHERE hypertable_schema = 'analytics'
        """)
        hypertables = [row[0] for row in cursor.fetchall()]
        print(f"Found {len(hypertables)} hypertables: {', '.join(hypertables)}")
        
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run([
                sys.executable, '-c', test_script
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.error(f"❌ Database schema validation failed: {result.stderr}")
                logger.error(f"Output: {result.stdout}")
                return False
            
            logger.info("✅ Database schema initialized successfully")
            logger.info(result.stdout)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database schema: {e}")
            return False
    
    def create_analytics_directories(self) -> bool:
        """Create necessary directories for analytics."""
        logger.info("Creating analytics directories...")
        
        try:
            directories = [
                self.project_root / "logs",
                self.project_root / "analytics" / "reports",
                self.project_root / "analytics" / "exports",
                self.project_root / "analytics" / "models"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def run_initial_data_collection(self) -> bool:
        """Run initial metrics collection to populate analytics database."""
        logger.info("Running initial data collection...")
        
        try:
            # Run analytics manager to collect initial metrics
            result = subprocess.run([
                sys.executable, 'analytics_manager.py', 'collect'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Initial data collection had issues: {result.stderr}")
                logger.info("This is normal if no articles exist yet in the main database")
            else:
                logger.info("✅ Initial data collection completed")
            
            logger.info(result.stdout)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to run initial data collection: {e}")
            return False
    
    def display_completion_message(self):
        """Display setup completion message with next steps."""
        logger.info("🎉 Analytics setup completed!")
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎉 EPIC 6: Analytics Setup Complete! 🎉                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Your Mimir analytics infrastructure is now ready for business intelligence! ║
║                                                                              ║
║  🗄️  TimescaleDB:     Running on port 5433                                  ║ 
║  📊  Analytics Schema: Initialized with hypertables                         ║
║  🐍  Python Packages: Analytics dependencies installed                      ║
║                                                                              ║
║  Next Steps:                                                                 ║
║  ┌─────────────────────────────────────────────────────────────────────────┐║
║  │ 1. Start metrics collection:                                            │║
║  │    python analytics_manager.py continuous --interval 30                 │║
║  │                                                                         │║
║  │ 2. Run trend analysis:                                                  │║
║  │    python analytics_manager.py analyze --days 7                        │║
║  │                                                                         │║
║  │ 3. Check system status:                                                 │║
║  │    python analytics_manager.py status                                   │║
║  │                                                                         │║
║  │ 4. View TimescaleDB admin:                                              │║
║  │    http://localhost:8080 (Adminer)                                      │║
║  └─────────────────────────────────────────────────────────────────────────┘║
║                                                                              ║
║  📚 Documentation: Check /docs/analytics/ for detailed guides               ║
║  🐛 Issues: Report at https://github.com/your-repo/issues                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def run_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("🚀 Starting Analytics & Business Intelligence Setup (EPIC 6)")
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Creating directories", self.create_analytics_directories),
            ("Setting up TimescaleDB", self.setup_timescaledb),
            ("Initializing database schema", self.initialize_database_schema),
            ("Running initial data collection", self.run_initial_data_collection)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Setup failed at step: {step_name}")
                return False
            logger.info(f"✅ Completed: {step_name}")
        
        self.display_completion_message()
        return True


def main():
    """Main setup function."""
    setup = AnalyticsSetup()
    
    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()