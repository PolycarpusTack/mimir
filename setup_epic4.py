#!/usr/bin/env python3
"""
EPIC 4 Setup Script - High-Performance Async Architecture
Automated setup for async scraping, Celery, Redis, and monitoring systems
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_step(step_num, total_steps, description):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total_steps}] {description}")


def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"  Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        if e.stderr:
            print(f"  Error details: {e.stderr.strip()}")
        return False


def check_redis():
    """Check if Redis is running."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
        print("  ✅ Redis is running")
        return True
    except Exception as e:
        print(f"  ❌ Redis not available: {e}")
        return False


def create_config_updates():
    """Create or update configuration files for EPIC 4."""
    config_updates = {
        "redis": {"host": "localhost", "port": 6379, "db": 0, "password": None},
        "enable_cache": True,
        "cache_ttl_seconds": 3600,
        "enable_events": True,
        "concurrent_requests": 20,
        "request_timeout_seconds": 30,
        "connector_limit": 100,
        "connector_limit_per_host": 30,
    }

    # Read existing config
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Update with EPIC 4 settings
    config.update(config_updates)

    # Write updated config
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  ✅ Updated {config_file} with EPIC 4 settings")


def create_celery_config():
    """Create Celery configuration files."""
    celery_beat_schedule = {
        "scheduled-scraping": {
            "task": "mimir_scraper.run_scheduled_scraping",
            "schedule": 'crontab(minute=0, hour="*/4")',  # Every 4 hours
        },
        "cleanup-old-data": {
            "task": "mimir_scraper.cleanup_old_data",
            "schedule": "crontab(minute=0, hour=2)",  # Daily at 2 AM
        },
        "generate-daily-report": {
            "task": "mimir_scraper.generate_daily_report",
            "schedule": "crontab(minute=30, hour=8)",  # Daily at 8:30 AM
        },
    }

    with open("celerybeat-schedule.json", "w") as f:
        json.dump(celery_beat_schedule, f, indent=2)

    print("  ✅ Created Celery Beat schedule configuration")


def create_docker_compose():
    """Create Docker Compose file for Redis and monitoring."""
    docker_compose_content = """version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: mimir-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: mimir-redis-commander
    ports:
      - "8081:8081"
    environment:
      - REDIS_HOSTS=local:redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  flower:
    image: mher/flower:latest
    container_name: mimir-flower
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FLOWER_PORT=5555
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./:/app
    working_dir: /app

volumes:
  redis_data:
"""

    with open("docker-compose.epic4.yml", "w") as f:
        f.write(docker_compose_content)

    print("  ✅ Created Docker Compose file for EPIC 4 services")


def create_startup_scripts():
    """Create startup scripts for EPIC 4 components."""

    # Celery worker startup script
    celery_worker_script = """#!/bin/bash
# Start Celery Worker for Mimir Scraper

echo "Starting Celery Worker..."
celery -A celery_app worker \\
    --loglevel=info \\
    --queues=scraping,ai_processing,notifications,maintenance,reports,default \\
    --concurrency=4 \\
    --max-tasks-per-child=1000 \\
    --prefetch-multiplier=1
"""

    with open("start_celery_worker.sh", "w") as f:
        f.write(celery_worker_script)
    os.chmod("start_celery_worker.sh", 0o755)

    # Celery beat startup script
    celery_beat_script = """#!/bin/bash
# Start Celery Beat Scheduler for Mimir Scraper

echo "Starting Celery Beat Scheduler..."
celery -A celery_app beat \\
    --loglevel=info \\
    --schedule=celerybeat-schedule
"""

    with open("start_celery_beat.sh", "w") as f:
        f.write(celery_beat_script)
    os.chmod("start_celery_beat.sh", 0o755)

    # Async scraper startup script
    async_scraper_script = """#!/bin/bash
# Start Async Scraper for Mimir

echo "Starting Async Scraper..."
python async_scraper.py --run
"""

    with open("start_async_scraper.sh", "w") as f:
        f.write(async_scraper_script)
    os.chmod("start_async_scraper.sh", 0o755)

    print("  ✅ Created startup scripts")


def create_monitoring_script():
    """Create monitoring and health check script."""
    monitoring_script = """#!/usr/bin/env python3
import redis
import json
import time
from celery_app import app as celery_app
from async_monitor import AsyncResourceMonitor

def check_redis():
    try:
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        return {'status': 'healthy', 'details': 'Redis is responding'}
    except Exception as e:
        return {'status': 'unhealthy', 'details': str(e)}

def check_celery():
    try:
        i = celery_app.control.inspect()
        stats = i.stats()
        if stats:
            return {'status': 'healthy', 'details': f'Workers: {len(stats)}'}
        else:
            return {'status': 'unhealthy', 'details': 'No workers responding'}
    except Exception as e:
        return {'status': 'unhealthy', 'details': str(e)}

def main():
    print("EPIC 4 Health Check")
    print("==================")
    
    checks = {
        'Redis': check_redis(),
        'Celery': check_celery(),
    }
    
    for service, result in checks.items():
        status = "✅" if result['status'] == 'healthy' else "❌"
        print(f"{status} {service}: {result['details']}")
    
    # Save health check results
    with open('health_check_results.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'checks': checks
        }, f, indent=2)

if __name__ == '__main__':
    main()
"""

    with open("health_check.py", "w") as f:
        f.write(monitoring_script)
    os.chmod("health_check.py", 0o755)

    print("  ✅ Created health check script")


def setup_directories():
    """Create necessary directories."""
    directories = ["logs", "reports", "data/cache", "data/events"]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"  ✅ Created directories: {', '.join(directories)}")


def main():
    """Main setup function."""
    print_header("MIMIR EPIC 4 SETUP")
    print("Setting up High-Performance Async Architecture")

    total_steps = 8

    # Step 1: Check Python dependencies
    print_step(1, total_steps, "Installing Python dependencies")
    if not run_command("pip install -r requirements.txt", "Install requirements", check=False):
        print("  ⚠️  Some dependencies may have failed to install")

    # Step 2: Create directories
    print_step(2, total_steps, "Creating directories")
    setup_directories()

    # Step 3: Update configuration
    print_step(3, total_steps, "Updating configuration files")
    create_config_updates()
    create_celery_config()

    # Step 4: Create Docker Compose
    print_step(4, total_steps, "Creating Docker Compose configuration")
    create_docker_compose()

    # Step 5: Create startup scripts
    print_step(5, total_steps, "Creating startup scripts")
    create_startup_scripts()

    # Step 6: Create monitoring tools
    print_step(6, total_steps, "Creating monitoring tools")
    create_monitoring_script()

    # Step 7: Start Redis (optional)
    print_step(7, total_steps, "Starting Redis (optional)")
    if run_command("docker-compose -f docker-compose.epic4.yml up -d redis", "Start Redis with Docker", check=False):
        time.sleep(5)  # Wait for Redis to start
        check_redis()
    else:
        print("  ℹ️  Redis not started automatically. You can start it manually with:")
        print("     docker-compose -f docker-compose.epic4.yml up -d")

    # Step 8: Final validation
    print_step(8, total_steps, "Running final validation")
    try:
        import async_scraper
        import celery_app
        import event_streaming
        import load_testing
        import redis_cache

        print("  ✅ All EPIC 4 modules imported successfully")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")

    # Final instructions
    print_header("EPIC 4 SETUP COMPLETE")

    print("\n🚀 Next Steps:")
    print("1. Start Redis (if not already running):")
    print("   docker-compose -f docker-compose.epic4.yml up -d")

    print("\n2. Start Celery Worker:")
    print("   ./start_celery_worker.sh")

    print("\n3. Start Celery Beat (for scheduled tasks):")
    print("   ./start_celery_beat.sh")

    print("\n4. Run the async scraper:")
    print("   ./start_async_scraper.sh")

    print("\n5. Monitor system health:")
    print("   python health_check.py")

    print("\n6. Run load tests:")
    print("   python load_testing.py --comprehensive")

    print("\n📊 Monitoring URLs:")
    print("   - Flower (Celery): http://localhost:5555")
    print("   - Redis Commander: http://localhost:8081")

    print("\n✅ EPIC 4 High-Performance Async Architecture is ready!")


if __name__ == "__main__":
    main()
