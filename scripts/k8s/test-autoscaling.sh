#!/bin/bash

# Mimir Kubernetes Autoscaling Test Script
# Tests HPA, VPA, and cluster autoscaling functionality

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="mimir"
LOAD_TEST_DURATION="15m"
MONITORING_INTERVAL=30

# Function to log messages
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
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

log_info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl >/dev/null 2>&1; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check metrics server
    if ! kubectl get deployment metrics-server -n kube-system >/dev/null 2>&1; then
        log_warning "Metrics Server not found. Installing..."
        kubectl apply -f k8s/components/metrics-server.yaml
        sleep 30
    fi
    
    log_success "Prerequisites check passed"
}

# Function to get current scaling status
get_scaling_status() {
    local component=$1
    echo "=== Scaling Status for $component ==="
    
    # Get deployment info
    kubectl get deployment "$component" -n "$NAMESPACE" -o wide
    
    # Get HPA info
    if kubectl get hpa "$component-hpa" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo ""
        echo "HPA Status:"
        kubectl get hpa "$component-hpa" -n "$NAMESPACE"
        kubectl describe hpa "$component-hpa" -n "$NAMESPACE" | grep -E "(Current|Target)" || true
    fi
    
    # Get VPA info
    if kubectl get vpa "$component-vpa" -n "$NAMESPACE" >/dev/null 2>&1; then
        echo ""
        echo "VPA Status:"
        kubectl get vpa "$component-vpa" -n "$NAMESPACE"
    fi
    
    # Get pod metrics
    echo ""
    echo "Pod Metrics:"
    kubectl top pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$component" --containers || echo "Metrics not available yet"
    echo ""
}

# Function to monitor autoscaling
monitor_autoscaling() {
    local duration=$1
    local end_time=$(($(date +%s) + duration))
    
    log "Monitoring autoscaling for $duration seconds..."
    
    while [ $(date +%s) -lt $end_time ]; do
        clear
        echo "========================================="
        echo "Autoscaling Monitor - $(date)"
        echo "========================================="
        echo ""
        
        # Show HPA status
        echo "=== Horizontal Pod Autoscalers ==="
        kubectl get hpa -n "$NAMESPACE" || true
        echo ""
        
        # Show deployment replicas
        echo "=== Deployment Replicas ==="
        kubectl get deployments -n "$NAMESPACE" -o custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,CURRENT:.status.replicas,DESIRED:.spec.replicas
        echo ""
        
        # Show pod count by deployment
        echo "=== Pod Count by Deployment ==="
        for deployment in mimir-api mimir-web mimir-worker mimir-analytics; do
            count=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$deployment" --no-headers 2>/dev/null | wc -l)
            echo "$deployment: $count pods"
        done
        echo ""
        
        # Show node utilization
        echo "=== Node Utilization ==="
        kubectl top nodes || echo "Node metrics not available"
        echo ""
        
        # Show events related to scaling
        echo "=== Recent Scaling Events ==="
        kubectl get events -n "$NAMESPACE" --field-selector reason=SuccessfulRescale --sort-by='.lastTimestamp' | tail -5 || echo "No recent scaling events"
        
        sleep $MONITORING_INTERVAL
    done
}

# Function to generate load
generate_load() {
    log "Starting load generation..."
    
    # Check if Locust is already running
    if kubectl get deployment locust-master -n "$NAMESPACE" >/dev/null 2>&1; then
        log_warning "Locust deployment already exists. Deleting..."
        kubectl delete deployment locust-master locust-worker -n "$NAMESPACE" || true
        kubectl delete service locust-master -n "$NAMESPACE" || true
        sleep 10
    fi
    
    # Apply load test configuration
    log "Deploying load test..."
    kubectl apply -f k8s/testing/load-test-job.yaml
    
    # Wait for load test to start
    log "Waiting for load test to start..."
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=locust -n "$NAMESPACE" --timeout=120s || true
    
    # Port forward to Locust web UI
    log_info "Load test running. Locust web UI available at http://localhost:8089"
    kubectl port-forward service/locust-master 8089:8089 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    # Monitor for duration
    monitor_autoscaling $((${LOAD_TEST_DURATION%m} * 60))
    
    # Stop port forwarding
    kill $port_forward_pid 2>/dev/null || true
}

# Function to test HPA scaling
test_hpa_scaling() {
    log "Testing Horizontal Pod Autoscaler..."
    
    # Initial status
    log_info "Initial scaling status:"
    for component in mimir-api mimir-web mimir-worker mimir-analytics; do
        get_scaling_status "$component"
    done
    
    # Generate load
    generate_load
    
    # Final status
    log_info "Final scaling status:"
    for component in mimir-api mimir-web mimir-worker mimir-analytics; do
        get_scaling_status "$component"
    done
    
    # Analyze results
    log "Analyzing HPA scaling results..."
    for component in mimir-api mimir-web mimir-worker mimir-analytics; do
        if kubectl get hpa "$component-hpa" -n "$NAMESPACE" >/dev/null 2>&1; then
            local current_replicas=$(kubectl get hpa "$component-hpa" -n "$NAMESPACE" -o jsonpath='{.status.currentReplicas}')
            local min_replicas=$(kubectl get hpa "$component-hpa" -n "$NAMESPACE" -o jsonpath='{.spec.minReplicas}')
            
            if [ "$current_replicas" -gt "$min_replicas" ]; then
                log_success "$component scaled up from $min_replicas to $current_replicas replicas"
            else
                log_warning "$component did not scale up (current: $current_replicas)"
            fi
        fi
    done
}

# Function to test VPA recommendations
test_vpa_recommendations() {
    log "Testing Vertical Pod Autoscaler recommendations..."
    
    # Ensure VPAs are created
    kubectl apply -f k8s/base/vpa.yaml || true
    
    # Wait for recommendations
    log "Waiting for VPA recommendations (this may take a few minutes)..."
    sleep 120
    
    # Check VPA recommendations
    for component in mimir-api mimir-web mimir-worker mimir-analytics postgres redis; do
        if kubectl get vpa "$component-vpa" -n "$NAMESPACE" >/dev/null 2>&1; then
            log_info "VPA recommendations for $component:"
            kubectl describe vpa "$component-vpa" -n "$NAMESPACE" | grep -A 20 "Recommendation:" || echo "No recommendations yet"
            echo ""
        fi
    done
}

# Function to test cluster autoscaling
test_cluster_autoscaling() {
    log "Testing Cluster Autoscaler..."
    
    # Check if cluster autoscaler is deployed
    if ! kubectl get deployment cluster-autoscaler -n kube-system >/dev/null 2>&1; then
        log_warning "Cluster Autoscaler not deployed. Skipping test."
        return
    fi
    
    # Get initial node count
    local initial_nodes=$(kubectl get nodes --no-headers | wc -l)
    log_info "Initial node count: $initial_nodes"
    
    # Create high resource demand
    log "Creating high resource demand..."
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: cluster-autoscale-test
  namespace: $NAMESPACE
spec:
  parallelism: 10
  completions: 10
  template:
    spec:
      containers:
      - name: stress
        image: progrium/stress
        command: ["stress"]
        args: ["--cpu", "2", "--vm", "1", "--vm-bytes", "2G", "--timeout", "300s"]
        resources:
          requests:
            cpu: 2
            memory: 2Gi
          limits:
            cpu: 2
            memory: 2Gi
      restartPolicy: Never
      nodeSelector:
        workload-type: compute-optimized
      tolerations:
      - key: "workload"
        operator: "Equal"
        value: "compute"
        effect: "NoSchedule"
EOF
    
    # Monitor node scaling
    log "Monitoring node scaling..."
    local timeout=600  # 10 minutes
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        local current_nodes=$(kubectl get nodes --no-headers | wc -l)
        
        if [ $current_nodes -gt $initial_nodes ]; then
            log_success "Cluster scaled from $initial_nodes to $current_nodes nodes"
            break
        fi
        
        log_info "Current nodes: $current_nodes (waiting for scale up...)"
        sleep 30
        elapsed=$((elapsed + 30))
    done
    
    # Cleanup
    kubectl delete job cluster-autoscale-test -n "$NAMESPACE" || true
    
    # Wait for scale down
    log "Waiting for cluster to scale down..."
    sleep 300
    
    local final_nodes=$(kubectl get nodes --no-headers | wc -l)
    if [ $final_nodes -eq $initial_nodes ]; then
        log_success "Cluster scaled back down to $initial_nodes nodes"
    else
        log_warning "Cluster has $final_nodes nodes (expected $initial_nodes)"
    fi
}

# Function to generate report
generate_report() {
    local report_file="autoscaling-test-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "Mimir Autoscaling Test Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo ""
        
        echo "HPA Status:"
        kubectl get hpa -n "$NAMESPACE"
        echo ""
        
        echo "VPA Status:"
        kubectl get vpa -n "$NAMESPACE"
        echo ""
        
        echo "Current Deployments:"
        kubectl get deployments -n "$NAMESPACE"
        echo ""
        
        echo "Node Status:"
        kubectl get nodes
        echo ""
        
        echo "Recent Scaling Events:"
        kubectl get events -n "$NAMESPACE" --field-selector reason=SuccessfulRescale
        echo ""
        
    } > "$report_file"
    
    log_success "Report generated: $report_file"
}

# Function to cleanup
cleanup() {
    log "Cleaning up test resources..."
    
    # Delete load test
    kubectl delete deployment locust-master locust-worker -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete service locust-master -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete job mimir-load-test -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete job cluster-autoscale-test -n "$NAMESPACE" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log "Starting Mimir Autoscaling Tests"
    echo "================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Parse command line arguments
    case "${1:-all}" in
        hpa)
            test_hpa_scaling
            ;;
        vpa)
            test_vpa_recommendations
            ;;
        cluster)
            test_cluster_autoscaling
            ;;
        monitor)
            monitor_autoscaling "${2:-600}"
            ;;
        all)
            test_hpa_scaling
            echo ""
            test_vpa_recommendations
            echo ""
            test_cluster_autoscaling
            ;;
        *)
            echo "Usage: $0 {hpa|vpa|cluster|monitor|all} [duration]"
            exit 1
            ;;
    esac
    
    # Generate report
    generate_report
    
    # Cleanup
    cleanup
    
    log_success "Autoscaling tests completed!"
}

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"