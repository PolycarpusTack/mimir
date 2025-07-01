# PostgreSQL Setup Guide for Mimir

## Development Setup

### Quick Start with Docker Compose

```bash
# Start PostgreSQL, pgBouncer, Redis, and Adminer
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f postgres
```

### Access Points
- PostgreSQL: `localhost:5432`
- pgBouncer (connection pooling): `localhost:6432`
- Redis: `localhost:6379`
- Adminer (DB GUI): `http://localhost:8080`

### Connection Details
- Database: `mimir_news`
- Username: `mimir_user`
- Password: `mimir_dev_password` (change in production!)

## Production Deployment

### 1. PostgreSQL Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-16 postgresql-contrib-16

# RHEL/CentOS
sudo yum install postgresql16-server postgresql16-contrib
sudo postgresql-16-setup initdb
```

### 2. PostgreSQL Configuration

Edit `/etc/postgresql/16/main/postgresql.conf`:

```conf
# Connection Settings
listen_addresses = 'localhost'
max_connections = 200
shared_buffers = 256MB

# Performance Tuning
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1

# Logging
log_statement = 'mod'
log_duration = on
log_min_duration_statement = 100ms

# Full-text Search
default_text_search_config = 'pg_catalog.english'
```

### 3. pgBouncer Configuration

Create `/etc/pgbouncer/pgbouncer.ini`:

```ini
[databases]
mimir_news = host=localhost port=5432 dbname=mimir_news

[pgbouncer]
listen_port = 6432
listen_addr = localhost
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 3
server_lifetime = 3600
server_idle_timeout = 600
```

### 4. Security Setup

```bash
# Create application user
sudo -u postgres psql <<EOF
CREATE USER mimir_app WITH PASSWORD 'secure_password_here';
CREATE DATABASE mimir_news OWNER mimir_app;
GRANT CONNECT ON DATABASE mimir_news TO mimir_app;
EOF

# Configure pg_hba.conf for secure access
echo "host mimir_news mimir_app 127.0.0.1/32 scram-sha-256" >> /etc/postgresql/16/main/pg_hba.conf
```

### 5. Backup Strategy

Create `/etc/cron.d/mimir-backup`:

```cron
# Daily backup at 2 AM
0 2 * * * postgres pg_dump mimir_news | gzip > /backup/mimir_news_$(date +\%Y\%m\%d).sql.gz

# Weekly full backup
0 3 * * 0 postgres pg_dumpall | gzip > /backup/mimir_full_$(date +\%Y\%m\%d).sql.gz

# Clean up old backups (keep 30 days)
0 4 * * * find /backup -name "mimir_*.sql.gz" -mtime +30 -delete
```

## Environment Variables

Create `.env` file:

```env
# Database Configuration
DATABASE_URL=postgresql://mimir_user:password@localhost:5432/mimir_news
PGBOUNCER_URL=postgresql://mimir_user:password@localhost:6432/mimir_news

# Connection Pool Settings
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
```

## Monitoring

### 1. Connection Monitoring

```sql
-- Active connections
SELECT count(*) FROM pg_stat_activity;

-- Connections by state
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;

-- Long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

### 2. Performance Monitoring

```sql
-- Table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname = 'public' 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Troubleshooting

### Common Issues

1. **Connection Pool Exhaustion**
   ```bash
   # Check pgBouncer stats
   psql -h localhost -p 6432 pgbouncer -c "SHOW POOLS;"
   ```

2. **Slow Queries**
   ```sql
   -- Enable query logging
   ALTER SYSTEM SET log_min_duration_statement = 100;
   SELECT pg_reload_conf();
   ```

3. **Disk Space**
   ```bash
   # Check database size
   psql -c "SELECT pg_database_size('mimir_news');"
   
   # Vacuum and analyze
   psql -c "VACUUM ANALYZE;"
   ```