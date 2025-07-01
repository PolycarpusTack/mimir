# Complete Region Failover Runbook

**Last Updated**: 2024-01-19
**Severity**: CRITICAL
**Expected Duration**: 30-45 minutes
**Business Impact**: Complete service disruption during failover

## Decision Criteria

Initiate region failover when:
- Multiple availability zones in primary region are unavailable
- Network connectivity to primary region is severed
- Major cloud provider outage affecting primary region
- Catastrophic infrastructure failure

## Pre-Failover Assessment

### Step 1: Confirm Region Failure

```bash
# Check primary region health
aws ec2 describe-regions --region-names us-east-1

# Test connectivity to primary cluster
kubectl --context primary-cluster get nodes

# Verify application endpoints
curl -f https://api.mimir.example.com/health || echo "Primary API unreachable"

# Check Route53 health checks
aws route53 get-health-check-status --health-check-id <health-check-id>
```

### Step 2: Assess DR Readiness

```bash
# Switch to DR context
kubectl config use-context dr-cluster

# Verify DR cluster health
kubectl get nodes
kubectl get pods -n mimir

# Check database replication lag
kubectl exec -n mimir postgres-standby-0 -- psql -U postgres -c \
  "SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;"

# Verify recent backups
aws s3 ls s3://mimir-backups-dr/postgres/ --recursive | tail -10
```

## Failover Execution

### Phase 1: Prepare DR Environment

```bash
# 1. Scale up DR cluster if needed
kubectl --context dr-cluster scale deployment cluster-autoscaler --replicas=1

# 2. Verify all secrets are present
kubectl --context dr-cluster get secrets -n mimir

# 3. Check persistent volumes
kubectl --context dr-cluster get pv
kubectl --context dr-cluster get pvc -n mimir
```

### Phase 2: Database Failover

```bash
# 1. Promote PostgreSQL standby to primary
kubectl --context dr-cluster exec -n mimir postgres-standby-0 -- pg_ctl promote

# 2. Verify promotion
kubectl --context dr-cluster exec -n mimir postgres-standby-0 -- \
  psql -U postgres -c "SELECT pg_is_in_recovery();"

# 3. Update database service endpoints
kubectl --context dr-cluster patch service postgres-primary-service -n mimir \
  -p '{"spec":{"selector":{"app.kubernetes.io/name":"postgres","app.kubernetes.io/role":"standby"}}}'
```

### Phase 3: Redis Failover

```bash
# 1. Make DR Redis master
kubectl --context dr-cluster exec -n mimir redis-0 -- \
  redis-cli -a $REDIS_PASSWORD SLAVEOF NO ONE

# 2. Update Sentinel configuration
for i in 0 1 2; do
  kubectl --context dr-cluster exec -n mimir redis-sentinel-$i -- \
    redis-cli -p 26379 -a $REDIS_PASSWORD SENTINEL REMOVE mymaster
  
  kubectl --context dr-cluster exec -n mimir redis-sentinel-$i -- \
    redis-cli -p 26379 -a $REDIS_PASSWORD \
    SENTINEL MONITOR mymaster redis-0.redis-service 6379 2
done
```

### Phase 4: Application Deployment

```bash
# 1. Update configuration for DR environment
kubectl --context dr-cluster patch configmap mimir-config -n mimir --type merge \
  -p '{"data":{"REGION":"us-west-2","ENVIRONMENT":"dr-production"}}'

# 2. Scale up applications
kubectl --context dr-cluster scale deployment mimir-api -n mimir --replicas=3
kubectl --context dr-cluster scale deployment mimir-worker -n mimir --replicas=3
kubectl --context dr-cluster scale deployment mimir-analytics -n mimir --replicas=2
kubectl --context dr-cluster scale deployment mimir-web -n mimir --replicas=2

# 3. Wait for pods to be ready
kubectl --context dr-cluster wait --for=condition=ready pods -n mimir \
  -l app.kubernetes.io/part-of=mimir --timeout=300s
```

### Phase 5: Network Cutover

```bash
# 1. Get DR load balancer endpoints
DR_API_LB=$(kubectl --context dr-cluster get service mimir-api-service -n mimir \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
DR_WEB_LB=$(kubectl --context dr-cluster get service mimir-web-service -n mimir \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# 2. Update Route53 records
aws route53 change-resource-record-sets --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.mimir.example.com",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "'$DR_API_LB'"}]
      }
    }]
  }'

# 3. Update health checks
aws route53 update-health-check --health-check-id <health-check-id> \
  --ip-address <dr-endpoint-ip>
```

### Phase 6: Validation

```bash
# 1. Test application endpoints
curl -f https://api.mimir.example.com/health
curl -f https://app.mimir.example.com/

# 2. Verify database connectivity
kubectl --context dr-cluster exec -n mimir mimir-api-0 -- \
  python -c "import db_adapter; print(db_adapter.test_connection())"

# 3. Check application logs
kubectl --context dr-cluster logs -n mimir \
  -l app.kubernetes.io/name=mimir-api --tail=100

# 4. Monitor key metrics
kubectl --context dr-cluster port-forward -n mimir svc/grafana 3000:3000
# Open http://localhost:3000 and check dashboards
```

## Post-Failover Tasks

### Immediate Actions

1. **Update Status Page**
   ```bash
   # Update public status page
   curl -X POST https://status.mimir.example.com/api/v1/incidents \
     -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
     -d '{"name":"Region Failover Complete","status":"resolved","message":"Services restored in DR region"}'
   ```

2. **Notify Stakeholders**
   - Send all-hands notification
   - Update internal wiki
   - Schedule emergency post-mortem

3. **Monitor Stability**
   ```bash
   # Watch error rates
   watch -n 10 'kubectl --context dr-cluster top pods -n mimir'
   
   # Check for any issues
   kubectl --context dr-cluster get events -n mimir --sort-by='.lastTimestamp' | tail -20
   ```

### Within 1 Hour

1. **Verify Data Integrity**
   ```bash
   # Run data validation
   kubectl --context dr-cluster exec -n mimir postgres-primary-0 -- \
     psql -U postgres -f /scripts/data-validation.sql
   ```

2. **Check Backup Procedures**
   ```bash
   # Ensure backups are running in DR
   kubectl --context dr-cluster get cronjobs -n mimir
   ```

3. **Update Monitoring**
   - Adjust alert thresholds for DR capacity
   - Update dashboard to show DR metrics
   - Verify all integrations are working

### Within 24 Hours

1. **Plan Primary Region Recovery**
2. **Document Lessons Learned**
3. **Update Runbooks**
4. **Review and Adjust DR Capacity**

## Rollback to Primary Region

When primary region is recovered:

### Step 1: Prepare Primary Region

```bash
# Restore primary cluster
kubectl --context primary-cluster get nodes

# Restore data from DR
./scripts/restore-to-primary.sh
```

### Step 2: Sync Data

```bash
# Set up reverse replication
kubectl --context primary-cluster exec -n mimir postgres-0 -- \
  pg_basebackup -h $DR_POSTGRES_ENDPOINT -U replicator -D /tmp/restore -P

# Apply incremental changes
kubectl --context primary-cluster exec -n mimir postgres-0 -- \
  psql -U postgres -c "SELECT pg_start_backup('failback');"
```

### Step 3: Failback

```bash
# Reverse the failover process
# Update DNS back to primary
# Scale down DR region
```

## Emergency Contacts

- **Incident Commander**: ic@example.com / +1-555-0100
- **VP Engineering**: vp-eng@example.com / +1-555-0101
- **Cloud Provider Support**: [Support Case Link]
- **Executive Team**: exec-team@example.com

## Appendix: Common Issues

### DNS Not Updating
```bash
# Force DNS refresh
aws route53 get-change --id <change-id>

# Use regional endpoints temporarily
echo "Temporary endpoints:"
echo "API: $DR_API_LB"
echo "Web: $DR_WEB_LB"
```

### Database Won't Promote
```bash
# Force promotion
kubectl --context dr-cluster exec -n mimir postgres-standby-0 -- \
  touch /tmp/postgresql.trigger.5432

# If still failing, restore from backup
kubectl --context dr-cluster apply -f backup-restore-job.yaml
```

### Application Won't Start
```bash
# Check for missing configurations
kubectl --context dr-cluster describe pods -n mimir | grep -A 10 "Events:"

# Verify all required services
kubectl --context dr-cluster get svc -n mimir
```