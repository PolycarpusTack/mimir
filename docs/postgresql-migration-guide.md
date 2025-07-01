# PostgreSQL Migration Guide for Mimir

This guide covers the complete process of migrating Mimir from SQLite to PostgreSQL.

## Prerequisites

- Docker and Docker Compose (for development)
- Python 3.8+
- Existing Mimir installation with SQLite database

## Quick Start (Development)

### 1. Install PostgreSQL Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start PostgreSQL with Docker

```bash
# Start PostgreSQL, pgBouncer, and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Initialize PostgreSQL Schema

```bash
# Create database and apply migrations
python db_manager_postgres.py
alembic upgrade head
```

### 4. Migrate Data from SQLite

```bash
# Run the migration script
python migrate_to_postgres.py

# Optional: specify custom SQLite path
python migrate_to_postgres.py --sqlite-db /path/to/mimir_news.db
```

### 5. Test the Migration

```bash
# Set environment variable to use PostgreSQL
export USE_POSTGRES=true

# Run a test scrape
python scraper.py --run

# Check statistics
python scraper.py --stats
```

## Production Setup

### 1. Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-16 postgresql-contrib-16 pgbouncer

# Configure PostgreSQL (see docs/postgres-setup.md for detailed configuration)
```

### 2. Create Production Database

```sql
-- As postgres user
CREATE USER mimir_app WITH PASSWORD 'secure_password_here';
CREATE DATABASE mimir_news OWNER mimir_app;
GRANT ALL PRIVILEGES ON DATABASE mimir_news TO mimir_app;
```

### 3. Set Environment Variables

Create a `.env` file:

```env
# Database Configuration
USE_POSTGRES=true
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mimir_news
POSTGRES_USER=mimir_app
POSTGRES_PASSWORD=your_secure_password

# Connection Pool
DB_POOL_MIN=5
DB_POOL_MAX=20

# Use pgBouncer (recommended for production)
USE_PGBOUNCER=true
PGBOUNCER_PORT=6432
```

### 4. Run Migrations

```bash
# Export environment variables
export $(cat .env | xargs)

# Initialize database
python db_manager_postgres.py

# Apply schema
alembic upgrade head

# Migrate data
python migrate_to_postgres.py
```

## Using the Database Adapter

The codebase includes a database adapter (`db_adapter.py`) that provides backward compatibility:

```python
# In your code, replace:
import db_manager

# With:
import db_adapter as db_manager

# The adapter will automatically use PostgreSQL if USE_POSTGRES=true
```

## Features Comparison

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Full-text search | Limited | ✅ Advanced with ranking |
| Concurrent writes | ❌ | ✅ |
| JSON support | Basic | ✅ JSONB with indexing |
| Source management | ❌ | ✅ |
| Performance monitoring | ❌ | ✅ |
| Scalability | Limited | ✅ Excellent |
| Connection pooling | ❌ | ✅ pgBouncer |

## New PostgreSQL Features

### 1. Full-Text Search

```python
# Search articles with ranking
from db_manager_postgres import search_articles

results = search_articles("artificial intelligence machine learning", limit=10)
for result in results:
    print(f"{result['title']} (relevance: {result['rank']})")
    print(f"Snippet: {result['snippet']}")
```

### 2. Source Management

```python
# Get active sources from database
from db_manager_postgres import get_active_sources, update_source_stats

sources = get_active_sources()
for source in sources:
    print(f"{source['name']}: {source['url']}")
    
# Update source statistics
update_source_stats("TechCrunch", success=True)
```

### 3. Advanced Queries

```python
# Get articles with keyword matches
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.*, array_agg(k.keyword) as keywords
        FROM articles a
        JOIN keyword_alerts ka ON ka.article_id = a.id
        JOIN keywords k ON k.id = ka.keyword_id
        WHERE a.published_date > CURRENT_DATE - INTERVAL '7 days'
        GROUP BY a.id
        ORDER BY a.published_date DESC
    """)
    articles = cursor.fetchall()
```

### 4. Performance Monitoring

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Monitor connections
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state;
```

## Troubleshooting

### Connection Issues

```bash
# Test PostgreSQL connection
psql -h localhost -U mimir_user -d mimir_news -c "SELECT version();"

# Check pgBouncer
psql -h localhost -p 6432 -U mimir_user -d pgbouncer -c "SHOW POOLS;"
```

### Migration Errors

```python
# Check migration status
import db_adapter
status = db_adapter.check_migration_status()
print(status)

# Verify data integrity
python -c "
import db_manager_postgres as db
stats = db.get_statistics()
print(f'Total articles: {stats['total_articles']}')
print(f'Sources: {len(stats['articles_per_source'])}')
"
```

### Performance Issues

```bash
# Analyze and vacuum database
psql -U mimir_user -d mimir_news -c "VACUUM ANALYZE;"

# Check index usage
psql -U mimir_user -d mimir_news -c "
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan;
"
```

## Rollback Procedure

If you need to rollback to SQLite:

```bash
# 1. Stop using PostgreSQL
export USE_POSTGRES=false

# 2. Ensure SQLite database is intact
python -c "import db_manager; print(db_manager.get_statistics())"

# 3. Update configuration to use SQLite
# Remove or comment out PostgreSQL environment variables
```

## Monitoring and Maintenance

### Daily Tasks

```bash
# Refresh materialized view
psql -U mimir_user -d mimir_news -c "REFRESH MATERIALIZED VIEW CONCURRENTLY article_stats;"

# Clean up old data (archives articles older than 30 days)
python -c "import db_manager_postgres as db; print(db.cleanup_old_data(30))"
```

### Weekly Tasks

```bash
# Full vacuum and analyze
psql -U mimir_user -d mimir_news -c "VACUUM FULL ANALYZE;"

# Check for unused indexes
psql -U mimir_user -d mimir_news -f /path/to/maintenance/unused_indexes.sql
```

### Backup

```bash
# Backup database
pg_dump -U mimir_user -d mimir_news | gzip > mimir_backup_$(date +%Y%m%d).sql.gz

# Restore from backup
gunzip -c mimir_backup_20240101.sql.gz | psql -U mimir_user -d mimir_news
```

## Next Steps

After successful migration:

1. Update cron jobs to set `USE_POSTGRES=true`
2. Configure monitoring alerts
3. Set up automated backups
4. Plan for implementing EPIC 2: AI-Powered Content Analysis

For more details on PostgreSQL configuration and optimization, see `docs/postgres-setup.md`.