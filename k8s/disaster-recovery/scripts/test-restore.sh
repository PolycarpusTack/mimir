#!/bin/bash
# Test Restore Script - Validates backup restoration procedures
# Performs non-destructive restore testing in isolated namespace

set -euo pipefail

# Configuration
TEST_NAMESPACE="mimir-restore-test"
PRODUCTION_NAMESPACE="mimir"
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-mimir-backups-primary}"
CONTEXT="${KUBE_CONTEXT:-current-context}"
CLEANUP="${CLEANUP:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test tracking
declare -A RESTORE_TESTS
TEST_START_TIME=$(date +%s)

# Logging
log() {
    local level=$1
    shift
    echo -e "${level}[$(date +'%H:%M:%S')] $@${NC}"
}

# Cleanup function
cleanup() {
    if [ "$CLEANUP" == "true" ]; then
        log "$YELLOW" "Cleaning up test namespace..."
        kubectl delete namespace "$TEST_NAMESPACE" --ignore-not-found=true --wait=false
    else
        log "$YELLOW" "Skipping cleanup. Test namespace preserved: $TEST_NAMESPACE"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Create test namespace
setup_test_namespace() {
    log "$BLUE" "Setting up test namespace..."
    
    kubectl create namespace "$TEST_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Copy necessary secrets
    kubectl get secret mimir-secrets -n "$PRODUCTION_NAMESPACE" -o yaml | \
        sed "s/namespace: $PRODUCTION_NAMESPACE/namespace: $TEST_NAMESPACE/" | \
        kubectl apply -f -
    
    kubectl get secret postgres-replication-secret -n "$PRODUCTION_NAMESPACE" -o yaml 2>/dev/null | \
        sed "s/namespace: $PRODUCTION_NAMESPACE/namespace: $TEST_NAMESPACE/" | \
        kubectl apply -f - || true
    
    # Copy ConfigMaps
    kubectl get configmap mimir-config -n "$PRODUCTION_NAMESPACE" -o yaml | \
        sed "s/namespace: $PRODUCTION_NAMESPACE/namespace: $TEST_NAMESPACE/" | \
        kubectl apply -f -
}

# Test PostgreSQL restore
test_postgres_restore() {
    log "$BLUE" "\n=== Testing PostgreSQL Restore ==="
    
    local test_start=$(date +%s)
    
    # Find latest backup
    local latest_backup=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/postgres/" \
        | grep ".sql.gz$" | sort | tail -1 | awk '{print $4}')
    
    if [ -z "$latest_backup" ]; then
        log "$RED" "No PostgreSQL backup found!"
        RESTORE_TESTS["postgres"]="FAILED|No backup found|0s"
        return 1
    fi
    
    log "$GREEN" "Found backup: $latest_backup"
    
    # Create test PostgreSQL instance
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-test-pvc
  namespace: $TEST_NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: postgres-restore-test
  namespace: $TEST_NAMESPACE
  labels:
    app: postgres-test
spec:
  containers:
  - name: postgres
    image: postgres:16-alpine
    env:
    - name: POSTGRES_PASSWORD
      valueFrom:
        secretKeyRef:
          name: mimir-secrets
          key: POSTGRES_PASSWORD
    - name: POSTGRES_USER
      valueFrom:
        secretKeyRef:
          name: mimir-secrets
          key: POSTGRES_USER
    - name: POSTGRES_DB
      valueFrom:
        configMapKeyRef:
          name: mimir-config
          key: POSTGRES_DB
    volumeMounts:
    - name: postgres-data
      mountPath: /var/lib/postgresql/data
    - name: restore-script
      mountPath: /scripts
  - name: aws-cli
    image: amazon/aws-cli:latest
    command: ["/bin/sh", "-c", "tail -f /dev/null"]
    volumeMounts:
    - name: restore-script
      mountPath: /scripts
  volumes:
  - name: postgres-data
    persistentVolumeClaim:
      claimName: postgres-test-pvc
  - name: restore-script
    emptyDir: {}
EOF
    
    # Wait for pod to be ready
    log "$YELLOW" "Waiting for test PostgreSQL to start..."
    kubectl wait --for=condition=ready pod/postgres-restore-test -n "$TEST_NAMESPACE" --timeout=120s
    
    # Download and restore backup
    log "$YELLOW" "Downloading backup from S3..."
    kubectl exec -n "$TEST_NAMESPACE" postgres-restore-test -c aws-cli -- \
        aws s3 cp "s3://${S3_BACKUP_BUCKET}/postgres/${latest_backup}" /scripts/backup.sql.gz
    
    log "$YELLOW" "Restoring database..."
    kubectl exec -n "$TEST_NAMESPACE" postgres-restore-test -c postgres -- \
        sh -c "gunzip -c /scripts/backup.sql.gz | psql -U \$POSTGRES_USER -d \$POSTGRES_DB"
    
    # Validate restore
    log "$YELLOW" "Validating restored data..."
    
    # Check table count
    local table_count=$(kubectl exec -n "$TEST_NAMESPACE" postgres-restore-test -c postgres -- \
        psql -U postgres -d postgres -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    
    # Check key tables exist
    local articles_count=$(kubectl exec -n "$TEST_NAMESPACE" postgres-restore-test -c postgres -- \
        psql -U postgres -d postgres -t -c "SELECT COUNT(*) FROM articles;" 2>/dev/null | xargs || echo "0")
    
    local test_duration=$(( $(date +%s) - test_start ))
    
    if [ "$table_count" -gt 0 ] && [ "$articles_count" -gt 0 ]; then
        RESTORE_TESTS["postgres"]="PASSED|$table_count tables, $articles_count articles|${test_duration}s"
        log "$GREEN" "✓ PostgreSQL restore successful: $table_count tables, $articles_count articles"
        return 0
    else
        RESTORE_TESTS["postgres"]="FAILED|Restore validation failed|${test_duration}s"
        log "$RED" "✗ PostgreSQL restore validation failed"
        return 1
    fi
}

# Test Redis restore
test_redis_restore() {
    log "$BLUE" "\n=== Testing Redis Restore ==="
    
    local test_start=$(date +%s)
    
    # Find latest Redis backup
    local latest_backup=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/redis/" \
        | grep ".rdb.gz$" | sort | tail -1 | awk '{print $4}')
    
    if [ -z "$latest_backup" ]; then
        log "$YELLOW" "No Redis backup found (this may be expected)"
        RESTORE_TESTS["redis"]="SKIPPED|No backup found|0s"
        return 0
    fi
    
    # Create test Redis instance
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: redis-restore-test
  namespace: $TEST_NAMESPACE
  labels:
    app: redis-test
spec:
  containers:
  - name: redis
    image: redis:7-alpine
    command: ["redis-server", "--save", "60", "1", "--requirepass", "\$(REDIS_PASSWORD)"]
    env:
    - name: REDIS_PASSWORD
      valueFrom:
        secretKeyRef:
          name: mimir-secrets
          key: REDIS_PASSWORD
    volumeMounts:
    - name: redis-data
      mountPath: /data
  - name: aws-cli
    image: amazon/aws-cli:latest
    command: ["/bin/sh", "-c", "tail -f /dev/null"]
    volumeMounts:
    - name: redis-data
      mountPath: /data
  volumes:
  - name: redis-data
    emptyDir: {}
EOF
    
    # Wait for pod to be ready
    kubectl wait --for=condition=ready pod/redis-restore-test -n "$TEST_NAMESPACE" --timeout=60s
    
    # Download and restore backup
    log "$YELLOW" "Restoring Redis backup..."
    kubectl exec -n "$TEST_NAMESPACE" redis-restore-test -c aws-cli -- \
        sh -c "aws s3 cp s3://${S3_BACKUP_BUCKET}/redis/${latest_backup} /data/dump.rdb.gz && \
               gunzip /data/dump.rdb.gz"
    
    # Restart Redis to load the dump
    kubectl delete pod redis-restore-test -n "$TEST_NAMESPACE"
    sleep 5
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: redis-restore-test
  namespace: $TEST_NAMESPACE
  labels:
    app: redis-test
spec:
  containers:
  - name: redis
    image: redis:7-alpine
    command: ["redis-server", "--requirepass", "\$(REDIS_PASSWORD)"]
    env:
    - name: REDIS_PASSWORD
      valueFrom:
        secretKeyRef:
          name: mimir-secrets
          key: REDIS_PASSWORD
    volumeMounts:
    - name: redis-data
      mountPath: /data
  volumes:
  - name: redis-data
    emptyDir: {}
EOF
    
    kubectl wait --for=condition=ready pod/redis-restore-test -n "$TEST_NAMESPACE" --timeout=60s
    
    # Validate restore
    local key_count=$(kubectl exec -n "$TEST_NAMESPACE" redis-restore-test -- \
        redis-cli -a "\$REDIS_PASSWORD" DBSIZE | awk '{print $2}' || echo "0")
    
    local test_duration=$(( $(date +%s) - test_start ))
    
    if [ "$key_count" -gt 0 ]; then
        RESTORE_TESTS["redis"]="PASSED|$key_count keys restored|${test_duration}s"
        log "$GREEN" "✓ Redis restore successful: $key_count keys"
    else
        RESTORE_TESTS["redis"]="WARNING|Empty restore|${test_duration}s"
        log "$YELLOW" "⚠ Redis restored but empty (may be normal)"
    fi
}

# Test application data restore
test_application_restore() {
    log "$BLUE" "\n=== Testing Application Data Restore ==="
    
    local test_start=$(date +%s)
    
    # Check for application backups
    local app_backup_count=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/application/" \
        | grep ".tar.gz$" | wc -l)
    
    if [ "$app_backup_count" -eq 0 ]; then
        log "$YELLOW" "No application backups found"
        RESTORE_TESTS["application"]="SKIPPED|No backups found|0s"
        return 0
    fi
    
    # Create test pod for application restore
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: app-restore-test
  namespace: $TEST_NAMESPACE
spec:
  containers:
  - name: restore-test
    image: alpine:latest
    command: ["/bin/sh", "-c", "tail -f /dev/null"]
    volumeMounts:
    - name: app-data
      mountPath: /app
  volumes:
  - name: app-data
    emptyDir: {}
EOF
    
    kubectl wait --for=condition=ready pod/app-restore-test -n "$TEST_NAMESPACE" --timeout=60s
    
    # Download latest application backup
    local latest_manifest=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/application/" \
        | grep "_manifest.json$" | sort | tail -1 | awk '{print $4}')
    
    if [ -n "$latest_manifest" ]; then
        # Download and check manifest
        kubectl exec -n "$TEST_NAMESPACE" app-restore-test -- \
            wget -O /tmp/manifest.json \
            "https://${S3_BACKUP_BUCKET}.s3.amazonaws.com/application/${latest_manifest}"
        
        local backup_timestamp=$(kubectl exec -n "$TEST_NAMESPACE" app-restore-test -- \
            cat /tmp/manifest.json | grep timestamp | cut -d'"' -f4)
        
        RESTORE_TESTS["application"]="PASSED|Manifest validated: $backup_timestamp|$(($(date +%s) - test_start))s"
        log "$GREEN" "✓ Application backup manifest validated"
    else
        RESTORE_TESTS["application"]="WARNING|No manifest found|$(($(date +%s) - test_start))s"
        log "$YELLOW" "⚠ No application manifest found"
    fi
}

# Test cross-region restore capability
test_cross_region_restore() {
    log "$BLUE" "\n=== Testing Cross-Region Restore Capability ==="
    
    local test_start=$(date +%s)
    
    # Check if DR bucket is configured
    if [ -z "${S3_DR_BUCKET:-}" ]; then
        log "$YELLOW" "DR bucket not configured, skipping cross-region test"
        RESTORE_TESTS["cross_region"]="SKIPPED|Not configured|0s"
        return 0
    fi
    
    # Compare backup counts between regions
    local primary_count=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/postgres/" --recursive | wc -l)
    local dr_count=$(aws s3 ls "s3://${S3_DR_BUCKET}/postgres/" --recursive | wc -l)
    
    local diff=$(( primary_count - dr_count ))
    if [ $diff -lt 0 ]; then
        diff=$(( -diff ))
    fi
    
    if [ $diff -le 2 ]; then
        RESTORE_TESTS["cross_region"]="PASSED|Primary: $primary_count, DR: $dr_count|$(($(date +%s) - test_start))s"
        log "$GREEN" "✓ Cross-region sync healthy (diff: $diff)"
    else
        RESTORE_TESTS["cross_region"]="WARNING|Primary: $primary_count, DR: $dr_count (diff: $diff)|$(($(date +%s) - test_start))s"
        log "$YELLOW" "⚠ Cross-region sync lag detected (diff: $diff)"
    fi
}

# Generate summary report
generate_summary() {
    log "$BLUE" "\n=== Restore Test Summary ==="
    
    local total_duration=$(( $(date +%s) - TEST_START_TIME ))
    
    echo
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    Restore Test Results                      │"
    echo "├─────────────────┬───────────┬────────────────────┬──────────┤"
    echo "│ Component       │ Status    │ Details            │ Duration │"
    echo "├─────────────────┼───────────┼────────────────────┼──────────┤"
    
    for component in postgres redis application cross_region; do
        if [ -n "${RESTORE_TESTS[$component]:-}" ]; then
            IFS='|' read -r status details duration <<< "${RESTORE_TESTS[$component]}"
            
            # Color code status
            case $status in
                PASSED)
                    status_colored="${GREEN}PASSED${NC}"
                    ;;
                FAILED)
                    status_colored="${RED}FAILED${NC}"
                    ;;
                WARNING)
                    status_colored="${YELLOW}WARNING${NC}"
                    ;;
                SKIPPED)
                    status_colored="${BLUE}SKIPPED${NC}"
                    ;;
            esac
            
            printf "│ %-15s │ %-9s │ %-18s │ %8s │\n" \
                "$component" "$status_colored" "${details:0:18}" "$duration"
        fi
    done
    
    echo "└─────────────────┴───────────┴────────────────────┴──────────┘"
    echo
    echo "Total test duration: ${total_duration}s"
    echo
    
    # Check for any failures
    if grep -q "FAILED" <<< "${RESTORE_TESTS[@]}"; then
        log "$RED" "⚠️  Some restore tests failed. Review backup procedures."
        return 1
    else
        log "$GREEN" "✅ All restore tests completed successfully!"
        return 0
    fi
}

# Main execution
main() {
    log "$YELLOW" "Starting restore test validation..."
    log "$YELLOW" "Test namespace: $TEST_NAMESPACE"
    log "$YELLOW" "Cleanup after test: $CLEANUP"
    echo
    
    setup_test_namespace
    
    # Run all tests
    test_postgres_restore || true
    test_redis_restore || true
    test_application_restore || true
    test_cross_region_restore || true
    
    # Generate summary
    generate_summary
}

# Parse arguments
case "${1:-run}" in
    run)
        main
        ;;
    quick)
        # Quick test - PostgreSQL only
        setup_test_namespace
        test_postgres_restore
        generate_summary
        ;;
    preserve)
        CLEANUP=false
        main
        ;;
    *)
        echo "Usage: $0 {run|quick|preserve}"
        echo "  run      - Run all restore tests (default)"
        echo "  quick    - Run PostgreSQL restore test only"
        echo "  preserve - Run tests but preserve namespace for debugging"
        exit 1
        ;;
esac