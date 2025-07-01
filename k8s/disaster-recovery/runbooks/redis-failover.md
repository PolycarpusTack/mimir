# Redis Cache Failover Runbook

**Last Updated**: 2024-01-19
**Severity**: HIGH
**Expected Duration**: 2-5 minutes

## Overview

Redis Sentinel should handle most failovers automatically. This runbook covers manual intervention scenarios and verification procedures.

## Automatic Failover Verification

When Redis Sentinel detects a master failure:

```bash
# Check Sentinel logs
kubectl logs -n mimir -l app.kubernetes.io/component=sentinel --tail=50

# Verify new master
kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 sentinel masters

# Check which instance is now master
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD info replication
```

## Manual Failover Procedure

### Step 1: Identify Current Master

```bash
# Get current master from Sentinel
MASTER=$(kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD sentinel get-master-addr-by-name mymaster | head -1)
echo "Current master: $MASTER"

# Check all Redis instances
for i in 0 1 2; do
  echo "=== redis-$i ==="
  kubectl exec -n mimir redis-$i -- redis-cli -a $REDIS_PASSWORD info replication | grep "role:"
done
```

### Step 2: Force Failover

```bash
# Trigger manual failover via Sentinel
kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD sentinel failover mymaster

# Monitor failover progress
kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD sentinel masters
```

### Step 3: Verify New Master

```bash
# Wait for failover to complete (usually < 30 seconds)
sleep 30

# Check new master
NEW_MASTER=$(kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD sentinel get-master-addr-by-name mymaster | head -1)
echo "New master: $NEW_MASTER"

# Verify replication setup
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD info replication
```

### Step 4: Update Application Configuration

```bash
# Most applications should auto-discover via Sentinel
# If manual update needed:
kubectl set env deployment/mimir-api -n mimir REDIS_HOST=$NEW_MASTER
kubectl set env deployment/mimir-worker -n mimir REDIS_HOST=$NEW_MASTER

# Restart pods to pick up changes
kubectl rollout restart deployment/mimir-api -n mimir
kubectl rollout restart deployment/mimir-worker -n mimir
```

## Cache Rebuild Procedure

If Redis data is lost completely:

```bash
# Step 1: Clear any corrupted data
for i in 0 1 2; do
  kubectl exec -n mimir redis-$i -- redis-cli -a $REDIS_PASSWORD FLUSHALL
done

# Step 2: Trigger cache warming
kubectl exec -n mimir mimir-api-0 -- python -c "
from cache_warmer import warm_cache
warm_cache(full=True)
"

# Step 3: Monitor cache hit rates
kubectl exec -n mimir prometheus-0 -- promtool query instant \
  'redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)'
```

## Cross-Region Redis Failover

For disaster recovery scenarios:

### Step 1: Stop Replication

```bash
# On DR cluster
kubectl config use-context dr-cluster

# Stop replication from primary region
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD SLAVEOF NO ONE
```

### Step 2: Promote DR Redis

```bash
# Make DR Redis master
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD CONFIG SET min-replicas-to-write 0

# Update Sentinel configuration
kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD \
  SENTINEL REMOVE mymaster

kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD \
  SENTINEL MONITOR mymaster redis-0.redis-service 6379 2
```

### Step 3: Update Applications

```bash
# Point applications to DR Redis
kubectl patch configmap mimir-config -n mimir --type merge \
  -p '{"data":{"REDIS_SENTINEL_HOSTS":"redis-sentinel-0.redis-sentinel-service:26379,redis-sentinel-1.redis-sentinel-service:26379,redis-sentinel-2.redis-sentinel-service:26379"}}'

# Restart applications
kubectl rollout restart deployment -n mimir
```

## Monitoring During Failover

```bash
# Watch Redis metrics
watch -n 1 'kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD info stats | grep -E "instantaneous_ops_per_sec|used_memory_human"'

# Monitor application errors
kubectl logs -n mimir -l app.kubernetes.io/name=mimir-api --tail=100 -f | grep -i "redis\|cache"

# Check Sentinel decisions
kubectl logs -n mimir -l app.kubernetes.io/component=sentinel -f
```

## Troubleshooting

### Issue: Sentinel Can't Agree on Master

```bash
# Reset all Sentinels
for i in 0 1 2; do
  kubectl exec -n mimir redis-sentinel-$i -- redis-cli -p 26379 -a $REDIS_PASSWORD SENTINEL RESET mymaster
done

# Reconfigure with known good master
kubectl exec -n mimir redis-sentinel-0 -- redis-cli -p 26379 -a $REDIS_PASSWORD \
  SENTINEL MONITOR mymaster redis-0.redis-service 6379 2
```

### Issue: High Memory Usage

```bash
# Check memory usage
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD info memory

# Enable eviction if needed
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD CONFIG SET maxmemory-policy allkeys-lru

# Clear specific key patterns if needed
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD --scan --pattern "temp:*" | \
  xargs -L 100 kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD DEL
```

### Issue: Replication Broken

```bash
# Check replication status
kubectl exec -n mimir redis-1 -- redis-cli -a $REDIS_PASSWORD info replication

# Resync replica
kubectl exec -n mimir redis-1 -- redis-cli -a $REDIS_PASSWORD SLAVEOF redis-0.redis-service 6379

# Force full resync if needed
kubectl exec -n mimir redis-1 -- redis-cli -a $REDIS_PASSWORD PSYNC
```

## Performance Validation

After failover:

```bash
# Run Redis benchmark
kubectl exec -n mimir redis-0 -- redis-benchmark -a $REDIS_PASSWORD -q -n 10000

# Check slow queries
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD SLOWLOG GET 10

# Monitor client connections
kubectl exec -n mimir redis-0 -- redis-cli -a $REDIS_PASSWORD CLIENT LIST | wc -l
```

## Recovery Checklist

- [ ] Verify master election completed
- [ ] Check all replicas are syncing
- [ ] Confirm Sentinels agree on master
- [ ] Test application connectivity
- [ ] Monitor error rates
- [ ] Check cache hit rates
- [ ] Validate performance metrics
- [ ] Update documentation

## Contacts

- **Cache Team**: cache-oncall@example.com
- **Application Team**: app-oncall@example.com
- **Infrastructure**: infra-oncall@example.com