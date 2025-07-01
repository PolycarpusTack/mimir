#!/bin/bash
# Disaster Recovery Validation Script
# Performs comprehensive checks on DR readiness

set -euo pipefail

# Configuration
NAMESPACE="mimir"
PRIMARY_CONTEXT="${PRIMARY_CONTEXT:-primary-cluster}"
DR_CONTEXT="${DR_CONTEXT:-dr-cluster}"
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-mimir-backups-primary}"
S3_DR_BUCKET="${S3_DR_BUCKET:-mimir-backups-dr}"
REPORT_FILE="/tmp/dr-validation-$(date +%Y%m%d-%H%M%S).html"

# Test results
declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# HTML report header
init_report() {
    cat > "$REPORT_FILE" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>DR Validation Report - $(date)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .passed { color: green; }
        .failed { color: red; }
        .warning { color: orange; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric { display: inline-block; margin: 0 20px; }
    </style>
</head>
<body>
    <h1>Disaster Recovery Validation Report</h1>
    <p>Generated: $(date)</p>
    <div class="summary" id="summary">
        <h2>Summary</h2>
        <div class="metric">Total Tests: <span id="total">0</span></div>
        <div class="metric">Passed: <span id="passed" class="passed">0</span></div>
        <div class="metric">Failed: <span id="failed" class="failed">0</span></div>
        <div class="metric">Success Rate: <span id="rate">0%</span></div>
    </div>
EOF
}

# Add test result to report
add_test_result() {
    local category=$1
    local test_name=$2
    local status=$3
    local details=$4
    local duration=$5
    
    ((TOTAL_TESTS++))
    if [ "$status" == "PASSED" ]; then
        ((PASSED_TESTS++))
    fi
    
    TEST_RESULTS["${category}_${test_name}"]="$status|$details|$duration"
}

# Log function
log() {
    local level=$1
    shift
    echo -e "${level}$@${NC}"
}

# Test cluster connectivity
test_cluster_connectivity() {
    log "$BLUE" "\n=== Testing Cluster Connectivity ==="
    
    local start_time=$(date +%s)
    
    # Test primary cluster
    if kubectl --context "$PRIMARY_CONTEXT" get nodes &>/dev/null; then
        local node_count=$(kubectl --context "$PRIMARY_CONTEXT" get nodes --no-headers | wc -l)
        add_test_result "Infrastructure" "Primary Cluster Connectivity" "PASSED" \
            "$node_count nodes available" "$(($(date +%s) - start_time))s"
        log "$GREEN" "✓ Primary cluster: $node_count nodes"
    else
        add_test_result "Infrastructure" "Primary Cluster Connectivity" "FAILED" \
            "Cannot connect to cluster" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ Primary cluster: unreachable"
    fi
    
    # Test DR cluster
    start_time=$(date +%s)
    if kubectl --context "$DR_CONTEXT" get nodes &>/dev/null; then
        local node_count=$(kubectl --context "$DR_CONTEXT" get nodes --no-headers | wc -l)
        add_test_result "Infrastructure" "DR Cluster Connectivity" "PASSED" \
            "$node_count nodes available" "$(($(date +%s) - start_time))s"
        log "$GREEN" "✓ DR cluster: $node_count nodes"
    else
        add_test_result "Infrastructure" "DR Cluster Connectivity" "FAILED" \
            "Cannot connect to cluster" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ DR cluster: unreachable"
    fi
}

# Test database replication
test_database_replication() {
    log "$BLUE" "\n=== Testing Database Replication ==="
    
    local start_time=$(date +%s)
    
    # Check if standby exists in DR
    if kubectl --context "$DR_CONTEXT" get pod postgres-standby-0 -n "$NAMESPACE" &>/dev/null; then
        # Check replication lag
        local lag=$(kubectl --context "$DR_CONTEXT" exec -n "$NAMESPACE" postgres-standby-0 -- \
            psql -U postgres -t -c \
            "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int AS lag_seconds;" \
            2>/dev/null || echo "999999")
        
        if [ "$lag" -lt 60 ]; then
            add_test_result "Database" "Replication Lag" "PASSED" \
                "Lag: ${lag}s" "$(($(date +%s) - start_time))s"
            log "$GREEN" "✓ Replication lag: ${lag}s"
        else
            add_test_result "Database" "Replication Lag" "FAILED" \
                "Lag: ${lag}s (>60s)" "$(($(date +%s) - start_time))s"
            log "$RED" "✗ Replication lag: ${lag}s (exceeds threshold)"
        fi
        
        # Check if standby is in recovery mode
        local recovery=$(kubectl --context "$DR_CONTEXT" exec -n "$NAMESPACE" postgres-standby-0 -- \
            psql -U postgres -t -c "SELECT pg_is_in_recovery();" 2>/dev/null || echo "unknown")
        
        if [ "$recovery" == "t" ]; then
            add_test_result "Database" "Standby Mode" "PASSED" \
                "In recovery mode" "1s"
            log "$GREEN" "✓ Standby is in recovery mode"
        else
            add_test_result "Database" "Standby Mode" "FAILED" \
                "Not in recovery mode" "1s"
            log "$RED" "✗ Standby is not in recovery mode"
        fi
    else
        add_test_result "Database" "Standby Database" "FAILED" \
            "Standby pod not found" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ No standby database in DR region"
    fi
}

# Test backup integrity
test_backup_integrity() {
    log "$BLUE" "\n=== Testing Backup Integrity ==="
    
    local start_time=$(date +%s)
    
    # Check latest backup age
    local latest_backup=$(aws s3 ls "s3://${S3_BACKUP_BUCKET}/postgres/" --recursive \
        | grep ".sql.gz$" | sort | tail -1 | awk '{print $1" "$2}')
    
    if [ -n "$latest_backup" ]; then
        local backup_date=$(date -d "$latest_backup" +%s 2>/dev/null || date +%s)
        local current_date=$(date +%s)
        local age_hours=$(( (current_date - backup_date) / 3600 ))
        
        if [ $age_hours -lt 24 ]; then
            add_test_result "Backup" "Backup Freshness" "PASSED" \
                "Latest backup: ${age_hours}h old" "$(($(date +%s) - start_time))s"
            log "$GREEN" "✓ Latest backup: ${age_hours}h old"
        else
            add_test_result "Backup" "Backup Freshness" "FAILED" \
                "Latest backup: ${age_hours}h old (>24h)" "$(($(date +%s) - start_time))s"
            log "$RED" "✗ Latest backup: ${age_hours}h old (too old)"
        fi
    else
        add_test_result "Backup" "Backup Availability" "FAILED" \
            "No backups found" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ No backups found in S3"
    fi
    
    # Check cross-region replication
    start_time=$(date +%s)
    local dr_backup_count=$(aws s3 ls "s3://${S3_DR_BUCKET}/postgres/" --recursive \
        | grep ".sql.gz$" | wc -l)
    
    if [ $dr_backup_count -gt 0 ]; then
        add_test_result "Backup" "Cross-Region Replication" "PASSED" \
            "$dr_backup_count backups in DR region" "$(($(date +%s) - start_time))s"
        log "$GREEN" "✓ Cross-region replication: $dr_backup_count backups"
    else
        add_test_result "Backup" "Cross-Region Replication" "FAILED" \
            "No backups in DR region" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ No backups in DR region"
    fi
}

# Test application readiness
test_application_readiness() {
    log "$BLUE" "\n=== Testing Application Readiness ==="
    
    for app in api worker analytics web; do
        local start_time=$(date +%s)
        local deployment="mimir-$app"
        
        if kubectl --context "$DR_CONTEXT" get deployment "$deployment" -n "$NAMESPACE" &>/dev/null; then
            local replicas=$(kubectl --context "$DR_CONTEXT" get deployment "$deployment" -n "$NAMESPACE" \
                -o jsonpath='{.spec.replicas}')
            local ready=$(kubectl --context "$DR_CONTEXT" get deployment "$deployment" -n "$NAMESPACE" \
                -o jsonpath='{.status.readyReplicas}')
            
            if [ "${ready:-0}" -eq "${replicas:-1}" ] && [ "${replicas:-0}" -gt 0 ]; then
                add_test_result "Application" "$app readiness" "PASSED" \
                    "$ready/$replicas replicas ready" "$(($(date +%s) - start_time))s"
                log "$GREEN" "✓ $app: $ready/$replicas replicas ready"
            else
                add_test_result "Application" "$app readiness" "FAILED" \
                    "${ready:-0}/${replicas:-0} replicas ready" "$(($(date +%s) - start_time))s"
                log "$RED" "✗ $app: ${ready:-0}/${replicas:-0} replicas ready"
            fi
        else
            add_test_result "Application" "$app deployment" "FAILED" \
                "Deployment not found" "$(($(date +%s) - start_time))s"
            log "$RED" "✗ $app: deployment not found"
        fi
    done
}

# Test network connectivity
test_network_connectivity() {
    log "$BLUE" "\n=== Testing Network Connectivity ==="
    
    # Test service endpoints
    for svc in postgres-primary-service redis-master-service; do
        local start_time=$(date +%s)
        
        if kubectl --context "$DR_CONTEXT" get endpoints "$svc" -n "$NAMESPACE" &>/dev/null; then
            local endpoints=$(kubectl --context "$DR_CONTEXT" get endpoints "$svc" -n "$NAMESPACE" \
                -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
            
            if [ $endpoints -gt 0 ]; then
                add_test_result "Network" "$svc endpoints" "PASSED" \
                    "$endpoints endpoints available" "$(($(date +%s) - start_time))s"
                log "$GREEN" "✓ $svc: $endpoints endpoints"
            else
                add_test_result "Network" "$svc endpoints" "FAILED" \
                    "No endpoints available" "$(($(date +%s) - start_time))s"
                log "$RED" "✗ $svc: no endpoints"
            fi
        else
            add_test_result "Network" "$svc service" "FAILED" \
                "Service not found" "$(($(date +%s) - start_time))s"
            log "$RED" "✗ $svc: not found"
        fi
    done
}

# Test Redis Sentinel
test_redis_sentinel() {
    log "$BLUE" "\n=== Testing Redis Sentinel ==="
    
    local start_time=$(date +%s)
    local sentinel_count=0
    
    for i in 0 1 2; do
        if kubectl --context "$DR_CONTEXT" get pod "redis-sentinel-$i" -n "$NAMESPACE" &>/dev/null; then
            ((sentinel_count++))
        fi
    done
    
    if [ $sentinel_count -ge 2 ]; then
        add_test_result "Redis" "Sentinel Quorum" "PASSED" \
            "$sentinel_count/3 sentinels available" "$(($(date +%s) - start_time))s"
        log "$GREEN" "✓ Redis Sentinel: $sentinel_count/3 available (quorum met)"
    else
        add_test_result "Redis" "Sentinel Quorum" "FAILED" \
            "$sentinel_count/3 sentinels available" "$(($(date +%s) - start_time))s"
        log "$RED" "✗ Redis Sentinel: $sentinel_count/3 available (no quorum)"
    fi
}

# Generate HTML report
generate_report() {
    log "$BLUE" "\n=== Generating Report ==="
    
    # Add test results to HTML
    cat >> "$REPORT_FILE" <<EOF
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Test</th>
            <th>Status</th>
            <th>Details</th>
            <th>Duration</th>
        </tr>
EOF
    
    local current_category=""
    for key in "${!TEST_RESULTS[@]}"; do
        IFS='|' read -r status details duration <<< "${TEST_RESULTS[$key]}"
        IFS='_' read -r category test_name <<< "$key"
        
        local status_class="failed"
        if [ "$status" == "PASSED" ]; then
            status_class="passed"
        fi
        
        if [ "$category" != "$current_category" ]; then
            current_category=$category
            echo "        <tr><td colspan='5'><strong>$category</strong></td></tr>" >> "$REPORT_FILE"
        fi
        
        cat >> "$REPORT_FILE" <<EOF
        <tr>
            <td></td>
            <td>$test_name</td>
            <td class="$status_class">$status</td>
            <td>$details</td>
            <td>$duration</td>
        </tr>
EOF
    done
    
    # Calculate success rate
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    fi
    
    # Add recommendations
    cat >> "$REPORT_FILE" <<EOF
    </table>
    
    <h2>Recommendations</h2>
    <ul>
EOF
    
    if [ $success_rate -lt 50 ]; then
        echo "        <li class='failed'>CRITICAL: DR readiness is below 50%. Immediate action required.</li>" >> "$REPORT_FILE"
    elif [ $success_rate -lt 80 ]; then
        echo "        <li class='warning'>WARNING: DR readiness is below 80%. Address failing tests.</li>" >> "$REPORT_FILE"
    else
        echo "        <li class='passed'>DR readiness is acceptable at ${success_rate}%.</li>" >> "$REPORT_FILE"
    fi
    
    # Specific recommendations based on failures
    if [[ "${TEST_RESULTS[Database_Replication Lag]}" == "FAILED"* ]]; then
        echo "        <li>Investigate and resolve database replication lag</li>" >> "$REPORT_FILE"
    fi
    
    if [[ "${TEST_RESULTS[Backup_Backup Freshness]}" == "FAILED"* ]]; then
        echo "        <li>Check backup job scheduling and S3 permissions</li>" >> "$REPORT_FILE"
    fi
    
    cat >> "$REPORT_FILE" <<EOF
    </ul>
    
    <script>
        document.getElementById('total').textContent = '$TOTAL_TESTS';
        document.getElementById('passed').textContent = '$PASSED_TESTS';
        document.getElementById('failed').textContent = '$((TOTAL_TESTS - PASSED_TESTS))';
        document.getElementById('rate').textContent = '${success_rate}%';
        
        // Color code the success rate
        var rateElement = document.getElementById('rate');
        if ($success_rate >= 80) {
            rateElement.className = 'passed';
        } else if ($success_rate >= 50) {
            rateElement.className = 'warning';
        } else {
            rateElement.className = 'failed';
        }
    </script>
</body>
</html>
EOF
    
    log "$GREEN" "Report generated: $REPORT_FILE"
    
    # Display summary
    echo
    echo "========================================="
    echo "DR Validation Summary"
    echo "========================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
    echo "Success Rate: ${success_rate}%"
    echo "========================================="
    
    # Return non-zero if success rate is below threshold
    if [ $success_rate -lt 80 ]; then
        return 1
    fi
}

# Main execution
main() {
    init_report
    
    test_cluster_connectivity
    test_database_replication
    test_backup_integrity
    test_application_readiness
    test_network_connectivity
    test_redis_sentinel
    
    generate_report
}

# Run tests
main