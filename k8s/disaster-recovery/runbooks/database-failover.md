# PostgreSQL Database Failover Runbook

**Last Updated**: 2024-01-19
**Severity**: CRITICAL
**Expected Duration**: 10-15 minutes

## Prerequisites

- kubectl access to both primary and DR clusters
- AWS CLI configured with appropriate permissions
- PostgreSQL client tools installed
- Access to monitoring dashboards

## Pre-Failover Checklist

- [ ] Confirm primary database is truly unavailable
- [ ] Check replication lag on standby databases
- [ ] Notify stakeholders via incident channel
- [ ] Ensure DR cluster is healthy
- [ ] Verify recent backup availability

## Automatic Failover (Primary to Standby in Same Region)

This should happen automatically, but verify:

```bash
# Check current primary
kubectl get pods -n mimir -l app.kubernetes.io/role=primary

# Check standby status
kubectl get pods -n mimir -l app.kubernetes.io/role=standby

# Verify new primary is accepting connections
kubectl exec -n mimir postgres-standby-0 -- pg_isready
```

## Manual Failover (Cross-Region)

### Step 1: Stop Applications in Primary Region

```bash
# Scale down applications to prevent split-brain
kubectl scale deployment -n mimir mimir-api --replicas=0
kubectl scale deployment -n mimir mimir-worker --replicas=0
kubectl scale deployment -n mimir mimir-analytics --replicas=0
```

### Step 2: Promote Standby in DR Region

```bash
# Connect to DR cluster
kubectl config use-context dr-cluster

# Promote standby to primary
kubectl exec -n mimir postgres-standby-0 -- pg_ctl promote

# Verify promotion
kubectl exec -n mimir postgres-standby-0 -- psql -U postgres -c "SELECT pg_is_in_recovery();"
# Should return 'f' (false)
```

### Step 3: Update Service Endpoints

```bash
# Update the primary service selector
kubectl patch service postgres-primary-service -n mimir -p '{"spec":{"selector":{"app.kubernetes.io/role":"standby"}}}'

# Verify service endpoints
kubectl get endpoints postgres-primary-service -n mimir
```

### Step 4: Start Applications in DR Region

```bash
# Scale up applications in DR
kubectl scale deployment -n mimir mimir-api --replicas=3
kubectl scale deployment -n mimir mimir-worker --replicas=3
kubectl scale deployment -n mimir mimir-analytics --replicas=2

# Verify applications are running
kubectl get pods -n mimir
```

### Step 5: Update DNS

```bash
# Update Route53 to point to DR load balancer
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://dns-failover.json

# Verify DNS propagation
dig api.mimir.example.com
```

### Step 6: Validate Application

```bash
# Test database connectivity
kubectl exec -n mimir mimir-api-xyz -- python -c "import db_adapter; print(db_adapter.test_connection())"

# Test API endpoint
curl -f https://api.mimir.example.com/health

# Check metrics
kubectl port-forward -n mimir svc/prometheus 9090:9090
# Open http://localhost:9090
```

## Post-Failover Tasks

1. **Monitor Application**
   ```bash
   # Watch logs
   kubectl logs -n mimir -l app.kubernetes.io/name=mimir-api -f
   
   # Check error rates
   kubectl exec -n mimir prometheus-0 -- promtool query instant \
     'rate(http_requests_total{status=~"5.."}[5m])'
   ```

2. **Verify Data Integrity**
   ```bash
   # Run data validation script
   kubectl exec -n mimir postgres-primary-0 -- psql -U postgres -f /scripts/validate-data.sql
   ```

3. **Update Monitoring**
   - Update alerting rules to reflect new primary
   - Adjust dashboards for DR region metrics
   - Verify all alerts are functioning

4. **Communication**
   - Update status page
   - Send notification to users
   - Schedule post-mortem

## Rollback Procedure

If failover needs to be reversed:

```bash
# Demote current primary back to standby
kubectl exec -n mimir postgres-primary-0 -- pg_ctl -D $PGDATA demote

# Reestablish replication
kubectl exec -n mimir postgres-primary-0 -- /scripts/setup-replication.sh

# Reverse DNS changes
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://dns-rollback.json
```

## Troubleshooting

### Issue: Replication Lag Too High
```bash
# Check replication lag
kubectl exec -n mimir postgres-standby-0 -- psql -U postgres -c \
  "SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;"

# If lag > 5 minutes, consider restoring from backup instead
```

### Issue: Split Brain Scenario
```bash
# Immediately stop one side
kubectl scale deployment -n mimir mimir-api --replicas=0

# Check both databases for divergence
# Use the backup-restore procedure to reconcile
```

### Issue: Connection Timeouts
```bash
# Check network connectivity
kubectl exec -n mimir mimir-api-xyz -- nc -zv postgres-primary-service 5432

# Verify security groups and network policies
kubectl get networkpolicies -n mimir
```

## Recovery Validation

After successful failover:

1. Run full application test suite
2. Verify all integrations are working
3. Check performance metrics
4. Validate backup procedures in new primary
5. Update documentation with lessons learned

## Contacts

- **Database Team**: db-oncall@example.com
- **Infrastructure**: infra-oncall@example.com
- **Application**: app-oncall@example.com
- **Management Escalation**: cto@example.com