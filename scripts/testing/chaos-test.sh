#!/bin/bash

# Mimir Chaos Engineering Test Script
# Automated chaos testing with monitoring and reporting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="mimir-production"
MONITORING_NAMESPACE="monitoring"
TEST_DURATION="600"  # 10 minutes
RECOVERY_TIMEOUT="300"  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    local required_tools=("kubectl" "jq" "curl")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check if Chaos Mesh is installed
    if ! kubectl get crd podchaos.chaos-mesh.org &> /dev/null; then
        log_error "Chaos Mesh is not installed. Please install it first:"
        log_error "helm repo add chaos-mesh https://charts.chaos-mesh.org"
        log_error "helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-system --create-namespace"
        exit 1
    fi
    
    # Check if monitoring is available
    if ! kubectl get service prometheus -n "$MONITORING_NAMESPACE" &> /dev/null; then
        log_warning "Prometheus monitoring not found. Some metrics collection will be skipped."
    fi
}

# Get baseline metrics
collect_baseline_metrics() {
    log_info "Collecting baseline metrics..."
    
    local output_dir="$PROJECT_ROOT/chaos-test-results/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"
    
    # Application health status
    kubectl get pods -n "$NAMESPACE" -o json > "$output_dir/baseline_pods.json"
    
    # Resource usage
    kubectl top pods -n "$NAMESPACE" --no-headers > "$output_dir/baseline_resources.txt" 2>/dev/null || echo "Metrics not available" > "$output_dir/baseline_resources.txt"
    
    # Service endpoints
    kubectl get services -n "$NAMESPACE" -o json > "$output_dir/baseline_services.json"
    
    # Application metrics (if Prometheus is available)
    if kubectl get service prometheus -n "$MONITORING_NAMESPACE" &> /dev/null; then
        local prometheus_url="http://localhost:9090"
        kubectl port-forward -n "$MONITORING_NAMESPACE" service/prometheus 9090:9090 &
        local port_forward_pid=$!
        sleep 5
        
        # Collect key metrics
        curl -s "${prometheus_url}/api/v1/query?query=up{namespace='${NAMESPACE}'}" > "$output_dir/baseline_up_status.json" || true
        curl -s "${prometheus_url}/api/v1/query?query=http_requests_total{namespace='${NAMESPACE}'}" > "$output_dir/baseline_request_rate.json" || true
        curl -s "${prometheus_url}/api/v1/query?query=http_request_duration_seconds{namespace='${NAMESPACE}'}" > "$output_dir/baseline_response_time.json" || true
        
        kill $port_forward_pid || true
    fi
    
    echo "$output_dir"
}

# Run chaos experiment
run_chaos_experiment() {
    local experiment_type="$1"
    local output_dir="$2"
    
    log_info "Starting chaos experiment: $experiment_type"
    
    case $experiment_type in
        "pod-failure")
            run_pod_failure_test "$output_dir"
            ;;
        "network-partition")
            run_network_partition_test "$output_dir"
            ;;
        "resource-stress")
            run_resource_stress_test "$output_dir"
            ;;
        "database-failure")
            run_database_failure_test "$output_dir"
            ;;
        "full-suite")
            run_full_chaos_suite "$output_dir"
            ;;
        *)
            log_error "Unknown experiment type: $experiment_type"
            exit 1
            ;;
    esac
}

# Pod failure test
run_pod_failure_test() {
    local output_dir="$1"
    log_info "Running pod failure test..."
    
    # Create pod chaos
    cat << EOF | kubectl apply -f -
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: test-pod-failure
  namespace: $NAMESPACE
spec:
  action: pod-kill
  mode: one
  duration: "60s"
  selector:
    namespaces:
      - $NAMESPACE
    labelSelectors:
      app: mimir-web
EOF
    
    # Monitor recovery
    monitor_experiment_impact "test-pod-failure" "$output_dir" "pod-failure"
    
    # Cleanup
    kubectl delete podchaos test-pod-failure -n "$NAMESPACE" || true
}

# Network partition test
run_network_partition_test() {
    local output_dir="$1"
    log_info "Running network partition test..."
    
    # Create network chaos
    cat << EOF | kubectl apply -f -
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: test-network-partition
  namespace: $NAMESPACE
spec:
  action: partition
  mode: one
  duration: "120s"
  selector:
    namespaces:
      - $NAMESPACE
    labelSelectors:
      app: mimir-api
  direction: to
  target:
    mode: one
    selector:
      namespaces:
        - $NAMESPACE
      labelSelectors:
        app: postgres
EOF
    
    # Monitor recovery
    monitor_experiment_impact "test-network-partition" "$output_dir" "network-partition"
    
    # Cleanup
    kubectl delete networkchaos test-network-partition -n "$NAMESPACE" || true
}

# Resource stress test
run_resource_stress_test() {
    local output_dir="$1"
    log_info "Running resource stress test..."
    
    # Create stress chaos
    cat << EOF | kubectl apply -f -
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: test-resource-stress
  namespace: $NAMESPACE
spec:
  mode: one
  duration: "180s"
  selector:
    namespaces:
      - $NAMESPACE
    labelSelectors:
      app: mimir-worker
  stressors:
    cpu:
      workers: 2
      load: 90
    memory:
      workers: 1
      size: "512MB"
EOF
    
    # Monitor recovery
    monitor_experiment_impact "test-resource-stress" "$output_dir" "resource-stress"
    
    # Cleanup
    kubectl delete stresschaos test-resource-stress -n "$NAMESPACE" || true
}

# Database failure test
run_database_failure_test() {
    local output_dir="$1"
    log_info "Running database failure test..."
    
    # Create database pod chaos
    cat << EOF | kubectl apply -f -
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: test-database-failure
  namespace: $NAMESPACE
spec:
  action: pod-kill
  mode: one
  duration: "90s"
  selector:
    namespaces:
      - $NAMESPACE
    labelSelectors:
      app: postgres
EOF
    
    # Monitor recovery
    monitor_experiment_impact "test-database-failure" "$output_dir" "database-failure"
    
    # Cleanup
    kubectl delete podchaos test-database-failure -n "$NAMESPACE" || true
}

# Monitor experiment impact
monitor_experiment_impact() {
    local experiment_name="$1"
    local output_dir="$2"
    local test_type="$3"
    
    log_info "Monitoring experiment impact for $experiment_name..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + TEST_DURATION))
    local check_interval=10
    
    # Initialize monitoring
    echo "timestamp,pods_ready,services_healthy,response_time_ms,error_rate" > "$output_dir/${test_type}_metrics.csv"
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local current_time=$(date +%s)
        
        # Check pod status
        local pods_ready=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -c "Running\|Completed" || echo "0")
        local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
        
        # Check service health
        local services_healthy=0
        local total_services=$(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)
        
        # Test application endpoints
        local response_time=0
        local error_rate=0
        
        # Try to access the web interface
        if kubectl get service mimir-web -n "$NAMESPACE" &> /dev/null; then
            kubectl port-forward -n "$NAMESPACE" service/mimir-web 8080:80 &
            local port_forward_pid=$!
            sleep 2
            
            local response_start=$(date +%s%N)
            if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
                local response_end=$(date +%s%N)
                response_time=$(( (response_end - response_start) / 1000000 ))  # Convert to ms
                services_healthy=$((services_healthy + 1))
            else
                response_time=9999
                error_rate=100
            fi
            
            kill $port_forward_pid || true
        fi
        
        # Log metrics
        echo "$current_time,$pods_ready,$services_healthy,$response_time,$error_rate" >> "$output_dir/${test_type}_metrics.csv"
        
        # Log status
        if [[ $((current_time % 30)) -eq 0 ]]; then
            log_info "Pods ready: $pods_ready/$total_pods, Services healthy: $services_healthy/$total_services"
        fi
        
        sleep $check_interval
    done
    
    # Check recovery
    log_info "Waiting for system recovery..."
    wait_for_recovery "$output_dir" "$test_type"
}

# Wait for system recovery
wait_for_recovery() {
    local output_dir="$1"
    local test_type="$2"
    
    local recovery_start=$(date +%s)
    local recovery_timeout_time=$((recovery_start + RECOVERY_TIMEOUT))
    
    while [[ $(date +%s) -lt $recovery_timeout_time ]]; do
        # Check if all pods are ready
        local not_ready_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -v "Running\|Completed" | wc -l)
        
        if [[ $not_ready_pods -eq 0 ]]; then
            local recovery_time=$(($(date +%s) - recovery_start))
            log_success "System recovered in ${recovery_time} seconds"
            echo "recovery_time=$recovery_time" >> "$output_dir/${test_type}_summary.txt"
            return 0
        fi
        
        sleep 5
    done
    
    log_warning "System did not fully recover within timeout"
    echo "recovery_time=TIMEOUT" >> "$output_dir/${test_type}_summary.txt"
    return 1
}

# Run full chaos suite
run_full_chaos_suite() {
    local output_dir="$1"
    
    log_info "Running full chaos engineering suite..."
    
    local tests=("pod-failure" "network-partition" "resource-stress" "database-failure")
    
    for test in "${tests[@]}"; do
        log_info "Starting test: $test"
        run_chaos_experiment "$test" "$output_dir"
        
        # Wait between tests
        log_info "Waiting for system stabilization..."
        sleep 60
    done
}

# Generate test report
generate_report() {
    local output_dir="$1"
    local report_file="$output_dir/chaos_test_report.md"
    
    log_info "Generating chaos test report..."
    
    cat << EOF > "$report_file"
# Mimir Chaos Engineering Test Report

**Date**: $(date)
**Environment**: $NAMESPACE
**Test Duration**: ${TEST_DURATION}s per experiment

## Test Summary

EOF
    
    # Analyze each test type
    local test_types=("pod-failure" "network-partition" "resource-stress" "database-failure")
    
    for test_type in "${test_types[@]}"; do
        if [[ -f "$output_dir/${test_type}_metrics.csv" ]]; then
            echo "### $test_type Test" >> "$report_file"
            
            # Calculate metrics
            local max_response_time=$(tail -n +2 "$output_dir/${test_type}_metrics.csv" | cut -d',' -f4 | sort -n | tail -1)
            local avg_response_time=$(tail -n +2 "$output_dir/${test_type}_metrics.csv" | cut -d',' -f4 | awk '{sum+=$1} END {print sum/NR}')
            local max_error_rate=$(tail -n +2 "$output_dir/${test_type}_metrics.csv" | cut -d',' -f5 | sort -n | tail -1)
            
            # Get recovery time
            local recovery_time="Unknown"
            if [[ -f "$output_dir/${test_type}_summary.txt" ]]; then
                recovery_time=$(grep "recovery_time=" "$output_dir/${test_type}_summary.txt" | cut -d'=' -f2)
            fi
            
            cat << EOF >> "$report_file"

- **Max Response Time**: ${max_response_time}ms
- **Avg Response Time**: ${avg_response_time}ms
- **Max Error Rate**: ${max_error_rate}%
- **Recovery Time**: ${recovery_time}s

EOF
        fi
    done
    
    # Add recommendations
    cat << EOF >> "$report_file"

## Recommendations

1. **Response Time**: Application should maintain sub-500ms response times during chaos events
2. **Error Rate**: Error rate should not exceed 5% during failures
3. **Recovery Time**: System should recover within 2 minutes of chaos event completion
4. **Monitoring**: Ensure monitoring and alerting systems function during chaos events

## Files Generated

- Baseline metrics: baseline_*.json
- Test metrics: *_metrics.csv
- Test summaries: *_summary.txt

EOF
    
    log_success "Report generated: $report_file"
}

# Usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] EXPERIMENT_TYPE

Experiment Types:
    pod-failure       Kill random pods to test resilience
    network-partition Simulate network partitions
    resource-stress   Create CPU/memory stress
    database-failure  Simulate database failures
    full-suite        Run all experiments sequentially

Options:
    -h, --help        Show this help message
    -n, --namespace   Target namespace (default: $NAMESPACE)
    -d, --duration    Test duration in seconds (default: $TEST_DURATION)
    -t, --timeout     Recovery timeout in seconds (default: $RECOVERY_TIMEOUT)

Examples:
    $0 pod-failure
    $0 full-suite
    $0 -n mimir-staging resource-stress

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            -t|--timeout)
                RECOVERY_TIMEOUT="$2"
                shift 2
                ;;
            pod-failure|network-partition|resource-stress|database-failure|full-suite)
                EXPERIMENT_TYPE="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ -z "${EXPERIMENT_TYPE:-}" ]]; then
        log_error "Experiment type is required"
        show_usage
        exit 1
    fi
}

# Main execution
main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    parse_args "$@"
    check_prerequisites
    
    log_info "Starting chaos engineering test: $EXPERIMENT_TYPE"
    log_info "Target namespace: $NAMESPACE"
    log_info "Test duration: ${TEST_DURATION}s"
    
    # Collect baseline metrics
    local output_dir
    output_dir=$(collect_baseline_metrics)
    
    # Run the experiment
    run_chaos_experiment "$EXPERIMENT_TYPE" "$output_dir"
    
    # Generate report
    generate_report "$output_dir"
    
    log_success "Chaos engineering test completed successfully"
    log_info "Results saved to: $output_dir"
}

# Run main function
main "$@"