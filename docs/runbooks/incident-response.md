# Mimir Incident Response Runbook

This runbook provides step-by-step procedures for responding to incidents in the Mimir News Scraper system.

## 🚨 Emergency Contacts

| Role | Contact | Phone | Escalation Time |
|------|---------|-------|-----------------|
| On-Call Engineer | +1-XXX-XXX-XXXX | 24/7 | Immediate |
| Platform Team Lead | +1-XXX-XXX-XXXX | Business Hours | 15 minutes |
| Security Team | security@company.com | 24/7 | 30 minutes |
| Management | management@company.com | Business Hours | 1 hour |

## 📊 Severity Levels

### SEV-1 (Critical)
- **Description**: Complete service outage or data breach
- **Response Time**: 15 minutes
- **Examples**: All APIs down, database corruption, security incident

### SEV-2 (High)
- **Description**: Significant feature degradation
- **Response Time**: 1 hour
- **Examples**: Single service down, high error rates, performance degradation

### SEV-3 (Medium)
- **Description**: Minor feature issues
- **Response Time**: 4 hours
- **Examples**: Non-critical feature broken, elevated error rates

### SEV-4 (Low)
- **Description**: Cosmetic issues or minor bugs
- **Response Time**: Next business day
- **Examples**: UI glitches, documentation issues

## 🔍 Initial Assessment

### Step 1: Acknowledge the Incident
```bash
# Check overall system health
kubectl get pods -n mimir-production
curl https://mimir.example.com/health

# Check monitoring dashboards
# - Grafana: https://grafana.example.com
# - Prometheus: https://prometheus.example.com
# - Kibana: https://kibana.example.com
```

### Step 2: Determine Severity
Use the severity matrix above to classify the incident.

### Step 3: Notify Stakeholders
```bash
# Post in incident channel
# Template:
# 🚨 SEV-X: [Brief Description]
# Impact: [User-facing impact]
# Time: [When started]
# Lead: [Your name]
# Status: Investigating
```

## 🔧 Common Incident Scenarios

### Application Pod Crashes

#### Symptoms
- HTTP 503 errors
- Pods in CrashLoopBackOff state
- High error rates in logs

#### Investigation
```bash
# Check pod status
kubectl get pods -n mimir-production -l app=mimir-api

# Check recent events
kubectl get events -n mimir-production --sort-by='.lastTimestamp' | tail -20

# Check pod logs
kubectl logs -f deployment/mimir-api -n mimir-production --tail=100

# Check resource usage
kubectl top pods -n mimir-production
```

#### Resolution
```bash
# Quick fix: Restart deployment
kubectl rollout restart deployment/mimir-api -n mimir-production

# Check if fix worked
kubectl rollout status deployment/mimir-api -n mimir-production

# If issue persists, check resource limits
kubectl describe deployment mimir-api -n mimir-production

# Scale up if needed
kubectl scale deployment mimir-api --replicas=5 -n mimir-production
```

### Database Connection Issues

#### Symptoms
- Database connection timeouts
- High database error rates
- Applications unable to start

#### Investigation
```bash
# Check PostgreSQL status
kubectl get pods -n mimir-production -l app=postgres

# Check database connectivity
kubectl exec -it postgres-primary-0 -n mimir-production -- pg_isready

# Check connection pool status
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"

# Check database logs
kubectl logs postgres-primary-0 -n mimir-production --tail=50
```

#### Resolution
```bash
# Restart database pod if necessary
kubectl delete pod postgres-primary-0 -n mimir-production

# Check for connection leaks
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "
    SELECT count(*), state 
    FROM pg_stat_activity 
    GROUP BY state;
    "

# Terminate long-running connections if needed
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "
    SELECT pg_terminate_backend(pid) 
    FROM pg_stat_activity 
    WHERE state = 'idle' 
    AND query_start < now() - interval '1 hour';
    "
```

### High Memory Usage

#### Symptoms
- Pods being OOMKilled
- Slow response times
- Memory alerts firing

#### Investigation
```bash
# Check current memory usage
kubectl top pods -n mimir-production

# Check memory limits
kubectl describe pods -n mimir-production | grep -A 5 -B 5 "memory"

# Check for memory leaks in application
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### Resolution
```bash
# Increase memory limits temporarily
kubectl patch deployment mimir-api -n mimir-production -p='
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "mimir-api",
            "resources": {
              "limits": {
                "memory": "2Gi"
              },
              "requests": {
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  }
}'

# Restart deployment to apply changes
kubectl rollout restart deployment/mimir-api -n mimir-production

# Monitor memory usage
kubectl top pods -n mimir-production -l app=mimir-api --watch
```

### High Error Rates

#### Symptoms
- Increased 4xx/5xx responses
- Error rate alerts
- User complaints

#### Investigation
```bash
# Check error rates in logs
kubectl logs deployment/mimir-api -n mimir-production | grep ERROR | tail -20

# Check specific error patterns
kubectl logs deployment/mimir-api -n mimir-production | \
    grep -E "(500|503|504)" | tail -10

# Check application metrics
curl http://localhost:8000/metrics | grep http_requests_total

# Check external dependencies
curl -I https://external-api.example.com/
```

#### Resolution
```bash
# Check recent deployments
kubectl rollout history deployment/mimir-api -n mimir-production

# Rollback if recent deployment caused issues
kubectl rollout undo deployment/mimir-api -n mimir-production

# Verify rollback
kubectl rollout status deployment/mimir-api -n mimir-production

# Check if error rates improved
kubectl logs deployment/mimir-api -n mimir-production --since=5m | grep ERROR
```

### Storage Issues

#### Symptoms
- Disk space alerts
- Database write failures
- Pod evictions

#### Investigation
```bash
# Check persistent volume usage
kubectl exec -it postgres-primary-0 -n mimir-production -- df -h

# Check pod storage usage
kubectl exec -it deployment/mimir-api -n mimir-production -- df -h

# Check for large log files
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    find /app/logs -name "*.log" -exec ls -lh {} \;
```

#### Resolution
```bash
# Clean up logs temporarily
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    find /app/logs -name "*.log" -mtime +7 -delete

# Increase persistent volume size (if supported)
kubectl patch pvc postgres-data-postgres-primary-0 -n mimir-production -p='
{
  "spec": {
    "resources": {
      "requests": {
        "storage": "200Gi"
      }
    }
  }
}'

# Clean up old database logs
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "SELECT pg_switch_wal();"
```

## 🔐 Security Incidents

### Suspected Data Breach

#### Immediate Actions
1. **Isolate affected systems**
```bash
# Scale down affected services
kubectl scale deployment mimir-api --replicas=0 -n mimir-production

# Block external traffic
kubectl patch ingress mimir-ingress -n mimir-production -p='
{
  "metadata": {
    "annotations": {
      "nginx.ingress.kubernetes.io/whitelist-source-range": "10.0.0.0/8"
    }
  }
}'
```

2. **Preserve evidence**
```bash
# Capture logs
kubectl logs deployment/mimir-api -n mimir-production > incident-logs.txt

# Capture system state
kubectl get all -n mimir-production > incident-state.yaml

# Backup current database
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    pg_dump -U mimir_user mimir_news > incident-backup-$(date +%Y%m%d).sql
```

3. **Notify security team**
- Email: security@company.com
- Include: Timeline, affected systems, initial assessment

### Unauthorized Access

#### Investigation
```bash
# Check access logs
kubectl logs deployment/mimir-api -n mimir-production | grep -E "(401|403)"

# Check authentication failures
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "
    SELECT * FROM activity_log 
    WHERE action = 'login' 
    AND status = 'failed' 
    AND created_at > NOW() - INTERVAL '1 hour'
    ORDER BY created_at DESC;
    "

# Check unusual API patterns
kubectl logs deployment/mimir-api -n mimir-production | \
    grep -E "POST|PUT|DELETE" | tail -20
```

#### Immediate Response
```bash
# Rotate API keys
kubectl delete secret api-keys -n mimir-production
kubectl create secret generic api-keys \
    --from-literal=master-key="$(openssl rand -base64 32)" \
    -n mimir-production

# Force password reset for affected users
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "
    UPDATE users SET password_reset_required = true 
    WHERE last_login_at > NOW() - INTERVAL '24 hours';
    "

# Restart services to pick up new secrets
kubectl rollout restart deployment/mimir-api -n mimir-production
```

## 📈 Performance Incidents

### High Latency

#### Investigation
```bash
# Check response times
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/articles

# Check database performance
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "
    SELECT query, mean_exec_time, calls 
    FROM pg_stat_statements 
    ORDER BY mean_exec_time DESC 
    LIMIT 10;
    "

# Check Redis performance
kubectl exec -it redis-0 -n mimir-production -- \
    redis-cli --latency-history
```

#### Resolution
```bash
# Scale up application
kubectl scale deployment mimir-api --replicas=10 -n mimir-production

# Enable caching if not already enabled
kubectl patch configmap mimir-config -n mimir-production -p='
{
  "data": {
    "CACHE_ENABLED": "true",
    "CACHE_TTL": "300"
  }
}'

# Restart to apply changes
kubectl rollout restart deployment/mimir-api -n mimir-production
```

## 🔄 Recovery Procedures

### Database Recovery

#### Point-in-Time Recovery
```bash
# Stop all write operations
kubectl scale deployment mimir-api --replicas=0 -n mimir-production
kubectl scale deployment mimir-worker --replicas=0 -n mimir-production

# Create backup of current state
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    pg_dump -U mimir_user mimir_news > pre-recovery-backup.sql

# Restore from specific point in time
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    pg_restore -U mimir_user -d mimir_news --clean backup-file.sql

# Verify data integrity
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user -d mimir_news -c "SELECT COUNT(*) FROM articles;"

# Restart services
kubectl scale deployment mimir-api --replicas=3 -n mimir-production
kubectl scale deployment mimir-worker --replicas=2 -n mimir-production
```

### Disaster Recovery

#### Failover to DR Site
```bash
# Run DR test to verify readiness
./scripts/testing/dr-test.sh full-test

# Activate DR environment
kubectl apply -k k8s/overlays/dr

# Update DNS to point to DR site
# (This step depends on your DNS provider)

# Verify DR services
curl https://dr.mimir.example.com/health

# Monitor DR performance
kubectl logs -f deployment/mimir-api -n mimir-dr
```

## 📝 Post-Incident Procedures

### Incident Review Checklist

1. **Timeline Documentation**
   - When was the incident first detected?
   - What actions were taken and when?
   - When was the incident resolved?

2. **Root Cause Analysis**
   - What was the immediate cause?
   - What were the contributing factors?
   - What could have prevented this?

3. **Impact Assessment**
   - How many users were affected?
   - What was the duration of impact?
   - What data was compromised?

4. **Action Items**
   - What improvements need to be made?
   - Who is responsible for each action?
   - What are the deadlines?

### Incident Report Template

```markdown
# Incident Report: [Brief Description]

## Summary
- **Date**: [YYYY-MM-DD]
- **Duration**: [X hours Y minutes]
- **Severity**: [SEV-X]
- **Impact**: [Brief description of user impact]

## Timeline
- **HH:MM** - [Event description]
- **HH:MM** - [Action taken]
- **HH:MM** - [Resolution achieved]

## Root Cause
[Detailed explanation of what caused the incident]

## Resolution
[What was done to resolve the incident]

## Lessons Learned
[What we learned and how to prevent similar incidents]

## Action Items
1. [ ] [Action description] - [Owner] - [Deadline]
2. [ ] [Action description] - [Owner] - [Deadline]
```

## 🔧 Useful Commands Reference

### Quick Health Checks
```bash
# Overall cluster health
kubectl get nodes
kubectl get pods --all-namespaces | grep -v Running

# Application health
curl https://mimir.example.com/health
curl https://mimir.example.com/api/health

# Database health
kubectl exec -it postgres-primary-0 -n mimir-production -- pg_isready

# Redis health
kubectl exec -it redis-0 -n mimir-production -- redis-cli ping
```

### Log Analysis
```bash
# Get recent errors
kubectl logs deployment/mimir-api -n mimir-production --since=1h | grep ERROR

# Follow live logs
kubectl logs -f deployment/mimir-api -n mimir-production

# Search for specific patterns
kubectl logs deployment/mimir-api -n mimir-production | grep -i "timeout\|connection\|error"
```

### Resource Monitoring
```bash
# Resource usage
kubectl top nodes
kubectl top pods -n mimir-production

# Describe resources
kubectl describe deployment mimir-api -n mimir-production
kubectl describe pod <pod-name> -n mimir-production
```

### Quick Fixes
```bash
# Restart deployment
kubectl rollout restart deployment/mimir-api -n mimir-production

# Scale deployment
kubectl scale deployment mimir-api --replicas=5 -n mimir-production

# Delete problematic pod
kubectl delete pod <pod-name> -n mimir-production

# Rollback deployment
kubectl rollout undo deployment/mimir-api -n mimir-production
```

## 📞 Escalation Matrix

| Time Elapsed | Action |
|--------------|--------|
| 0-15 minutes | On-call engineer investigates |
| 15-30 minutes | Escalate to team lead |
| 30-60 minutes | Escalate to platform team |
| 1-2 hours | Escalate to management |
| 2+ hours | All hands incident response |

Remember: **Communication is key**. Keep stakeholders updated every 30 minutes during active incidents.