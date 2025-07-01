#!/bin/bash
# Database Failover Automation Script
# Handles both automatic detection and manual failover scenarios

set -euo pipefail

# Configuration
NAMESPACE="mimir"
PRIMARY_CONTEXT="${PRIMARY_CONTEXT:-primary-cluster}"
DR_CONTEXT="${DR_CONTEXT:-dr-cluster}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
LOG_FILE="/tmp/database-failover-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    echo -e "${level}[$(date +'%Y-%m-%d %H:%M:%S')] $message${NC}" | tee -a "$LOG_FILE"
}

# Send notification
notify() {
    local message=$1
    local severity=${2:-info}
    
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST "$SLACK_WEBHOOK" \
            -H 'Content-Type: application/json' \
            -d "{\"text\":\":database: Database Failover Alert\",\"attachments\":[{\"color\":\"$severity\",\"text\":\"$message\"}]}" \
            2>/dev/null || true
    fi
}

# Check database health
check_database_health() {
    local context=$1
    local role=$2
    
    kubectl --context "$context" exec -n "$NAMESPACE" postgres-${role}-0 -- \
        pg_isready -U postgres -d postgres -t 5 2>/dev/null
}

# Get replication lag
get_replication_lag() {
    local context=$1
    local pod=$2
    
    kubectl --context "$context" exec -n "$NAMESPACE" "$pod" -- \
        psql -U postgres -t -c \
        "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int AS lag_seconds;" \
        2>/dev/null || echo "999999"
}

# Promote standby to primary
promote_standby() {
    local context=$1
    local standby_pod=$2
    
    log "$GREEN" "Promoting $standby_pod to primary..."
    
    # Create promotion trigger file
    kubectl --context "$context" exec -n "$NAMESPACE" "$standby_pod" -- \
        touch /tmp/postgresql.trigger.5432
    
    # Wait for promotion
    local max_wait=30
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if kubectl --context "$context" exec -n "$NAMESPACE" "$standby_pod" -- \
            psql -U postgres -t -c "SELECT pg_is_in_recovery();" | grep -q "f"; then
            log "$GREEN" "Promotion successful!"
            return 0
        fi
        sleep 1
        ((waited++))
    done
    
    log "$RED" "Promotion failed after ${max_wait} seconds"
    return 1
}

# Update service endpoints
update_service_endpoints() {
    local context=$1
    local new_role=$2
    
    log "$GREEN" "Updating service endpoints..."
    
    kubectl --context "$context" patch service postgres-primary-service -n "$NAMESPACE" \
        -p "{\"spec\":{\"selector\":{\"app.kubernetes.io/role\":\"$new_role\"}}}"
}

# Scale applications
scale_applications() {
    local context=$1
    local replicas=$2
    
    log "$GREEN" "Scaling applications to $replicas replicas..."
    
    for app in api worker analytics; do
        kubectl --context "$context" scale deployment "mimir-$app" -n "$NAMESPACE" \
            --replicas="$replicas"
    done
}

# Validate failover
validate_failover() {
    local context=$1
    
    log "$GREEN" "Validating failover..."
    
    # Check new primary
    if check_database_health "$context" "primary"; then
        log "$GREEN" "✓ New primary is healthy"
    else
        log "$RED" "✗ New primary is not responding"
        return 1
    fi
    
    # Test write capability
    if kubectl --context "$context" exec -n "$NAMESPACE" postgres-primary-0 -- \
        psql -U postgres -c "CREATE TABLE failover_test (id int); DROP TABLE failover_test;" 2>/dev/null; then
        log "$GREEN" "✓ Write test successful"
    else
        log "$RED" "✗ Write test failed"
        return 1
    fi
    
    # Check application connectivity
    local api_pod=$(kubectl --context "$context" get pod -n "$NAMESPACE" \
        -l app.kubernetes.io/name=mimir-api -o jsonpath='{.items[0].metadata.name}')
    
    if [ -n "$api_pod" ]; then
        if kubectl --context "$context" exec -n "$NAMESPACE" "$api_pod" -- \
            python -c "import db_adapter; print(db_adapter.test_connection())" 2>/dev/null | grep -q "True"; then
            log "$GREEN" "✓ Application connectivity verified"
        else
            log "$RED" "✗ Application cannot connect to database"
            return 1
        fi
    fi
    
    return 0
}

# Main failover logic
main() {
    local mode=${1:-auto}
    local target_context=${2:-$DR_CONTEXT}
    
    log "$YELLOW" "Starting database failover in $mode mode..."
    log "$YELLOW" "Log file: $LOG_FILE"
    
    notify "Database failover initiated in $mode mode" "warning"
    
    # Step 1: Assess current state
    log "$YELLOW" "Assessing database state..."
    
    if [ "$mode" == "auto" ]; then
        # Check if primary is really down
        if check_database_health "$PRIMARY_CONTEXT" "primary"; then
            log "$YELLOW" "Primary database is still healthy. Aborting automatic failover."
            notify "Automatic failover aborted - primary is healthy" "good"
            exit 0
        fi
    fi
    
    # Step 2: Check standby health and lag
    log "$YELLOW" "Checking standby databases..."
    
    local best_standby=""
    local min_lag=999999
    
    for i in 0 1 2; do
        local pod="postgres-standby-$i"
        if kubectl --context "$target_context" get pod -n "$NAMESPACE" "$pod" &>/dev/null; then
            local lag=$(get_replication_lag "$target_context" "$pod")
            log "$YELLOW" "$pod replication lag: ${lag}s"
            
            if [ "$lag" -lt "$min_lag" ]; then
                min_lag=$lag
                best_standby=$pod
            fi
        fi
    done
    
    if [ -z "$best_standby" ]; then
        log "$RED" "No healthy standby found!"
        notify "Failover failed - no healthy standby available" "danger"
        exit 1
    fi
    
    log "$GREEN" "Selected $best_standby with lag of ${min_lag}s"
    
    # Step 3: Stop applications in primary (if accessible)
    if [ "$mode" == "manual" ]; then
        log "$YELLOW" "Stopping applications in primary region..."
        scale_applications "$PRIMARY_CONTEXT" 0 || true
    fi
    
    # Step 4: Promote standby
    if ! promote_standby "$target_context" "$best_standby"; then
        log "$RED" "Failed to promote standby!"
        notify "Failover failed during promotion" "danger"
        exit 1
    fi
    
    # Step 5: Update service endpoints
    update_service_endpoints "$target_context" "standby"
    
    # Step 6: Scale up applications in target region
    scale_applications "$target_context" 3
    
    # Step 7: Wait for applications to be ready
    log "$YELLOW" "Waiting for applications to be ready..."
    kubectl --context "$target_context" wait --for=condition=ready pods -n "$NAMESPACE" \
        -l app.kubernetes.io/part-of=mimir --timeout=300s
    
    # Step 8: Validate failover
    if validate_failover "$target_context"; then
        log "$GREEN" "Failover completed successfully!"
        notify "Database failover completed successfully" "good"
        
        # Generate summary report
        cat > "${LOG_FILE%.log}-summary.txt" <<EOF
Database Failover Summary
========================
Date: $(date)
Mode: $mode
Target Context: $target_context
New Primary: $best_standby
Replication Lag: ${min_lag}s
Status: SUCCESS

Next Steps:
1. Update DNS records if needed
2. Monitor application performance
3. Plan for restoring redundancy
4. Schedule post-mortem meeting
EOF
        
        log "$GREEN" "Summary saved to ${LOG_FILE%.log}-summary.txt"
    else
        log "$RED" "Failover validation failed!"
        notify "Database failover completed but validation failed" "danger"
        exit 1
    fi
}

# Parse arguments
case "${1:-}" in
    auto|manual)
        main "$@"
        ;;
    test)
        log "$YELLOW" "Running in test mode..."
        check_database_health "$PRIMARY_CONTEXT" "primary" && log "$GREEN" "Primary is healthy"
        get_replication_lag "$DR_CONTEXT" "postgres-standby-0"
        ;;
    *)
        echo "Usage: $0 {auto|manual|test} [target-context]"
        echo "  auto   - Automatic failover (checks primary health first)"
        echo "  manual - Manual failover (forces failover)"
        echo "  test   - Test connectivity only"
        exit 1
        ;;
esac