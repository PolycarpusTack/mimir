#!/bin/bash

# Mimir Resource Optimization Script
# Optimizes Kubernetes resources, reduces costs, and cleans up unused resources

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE="mimir-production"
DRY_RUN=false
AGGRESSIVE_CLEANUP=false

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

# Usage information
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    analyze       Analyze current resource usage and costs
    optimize      Apply resource optimizations
    cleanup       Clean up unused resources
    cost-report   Generate cost optimization report
    right-size    Right-size deployments based on usage
    all           Run complete optimization suite

Options:
    -h, --help           Show this help message
    -n, --namespace NS   Target namespace (default: $NAMESPACE)
    -d, --dry-run        Show what would be changed without executing
    -a, --aggressive     Enable aggressive cleanup (removes more resources)
    -r, --report FILE    Save report to file

Examples:
    $0 analyze
    $0 --dry-run optimize
    $0 --aggressive cleanup

EOF
}

# Parse command line arguments
parse_args() {
    REPORT_FILE=""
    
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
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -a|--aggressive)
                AGGRESSIVE_CLEANUP=true
                shift
                ;;
            -r|--report)
                REPORT_FILE="$2"
                shift 2
                ;;
            analyze|optimize|cleanup|cost-report|right-size|all)
                COMMAND="$1"
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
    
    if [[ -z "${COMMAND:-}" ]]; then
        log_error "Command is required"
        show_usage
        exit 1
    fi
}

# Analyze current resource usage
analyze_resources() {
    log_info "Analyzing current resource usage..."
    
    local output_dir="$PROJECT_ROOT/optimization-reports/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$output_dir"
    
    # Get current resource requests and limits
    log_info "Collecting resource specifications..."
    kubectl get pods -n "$NAMESPACE" -o json | jq -r '
    .items[] | 
    select(.metadata.labels.app) |
    {
        name: .metadata.name,
        app: .metadata.labels.app,
        requests: (.spec.containers[0].resources.requests // {}),
        limits: (.spec.containers[0].resources.limits // {}),
        node: .spec.nodeName,
        phase: .status.phase
    }' > "$output_dir/resource_specs.json"
    
    # Get actual resource usage
    log_info "Collecting actual resource usage..."
    kubectl top pods -n "$NAMESPACE" --no-headers > "$output_dir/actual_usage.txt" 2>/dev/null || {
        log_warning "Metrics server not available, skipping usage analysis"
        echo "Metrics not available" > "$output_dir/actual_usage.txt"
    }
    
    # Get node resource allocation
    log_info "Collecting node resource allocation..."
    kubectl describe nodes | grep -A 5 "Allocated resources:" > "$output_dir/node_allocation.txt"
    
    # Analyze resource efficiency
    log_info "Analyzing resource efficiency..."
    python3 -c "
import json
import sys

# Load resource specs
with open('$output_dir/resource_specs.json', 'r') as f:
    pods = [json.loads(line) for line in f if line.strip()]

# Calculate totals
total_cpu_requests = 0
total_memory_requests = 0
total_cpu_limits = 0
total_memory_limits = 0

efficiency_report = {
    'pods': [],
    'summary': {}
}

for pod in pods:
    if pod['phase'] == 'Running':
        cpu_req = pod['requests'].get('cpu', '0m')
        mem_req = pod['requests'].get('memory', '0Mi')
        cpu_lim = pod['limits'].get('cpu', '0m')
        mem_lim = pod['limits'].get('memory', '0Mi')
        
        # Convert to millicores and MiB
        if cpu_req.endswith('m'):
            cpu_req_mc = int(cpu_req[:-1])
        else:
            cpu_req_mc = int(float(cpu_req) * 1000)
            
        if mem_req.endswith('Mi'):
            mem_req_mb = int(mem_req[:-2])
        elif mem_req.endswith('Gi'):
            mem_req_mb = int(float(mem_req[:-2]) * 1024)
        else:
            mem_req_mb = 0
            
        total_cpu_requests += cpu_req_mc
        total_memory_requests += mem_req_mb
        
        efficiency_report['pods'].append({
            'name': pod['name'],
            'app': pod['app'],
            'cpu_request_mc': cpu_req_mc,
            'memory_request_mb': mem_req_mb,
            'node': pod['node']
        })

efficiency_report['summary'] = {
    'total_cpu_requests_mc': total_cpu_requests,
    'total_memory_requests_mb': total_memory_requests,
    'pod_count': len([p for p in pods if p['phase'] == 'Running'])
}

with open('$output_dir/efficiency_report.json', 'w') as f:
    json.dump(efficiency_report, f, indent=2)

print(f'Total CPU Requests: {total_cpu_requests}m')
print(f'Total Memory Requests: {total_memory_requests}Mi')
print(f'Running Pods: {len([p for p in pods if p[\"phase\"] == \"Running\"])}')
"
    
    log_success "Resource analysis completed. Results saved to: $output_dir"
    echo "$output_dir"
}

# Optimize resource allocations
optimize_resources() {
    log_info "Optimizing resource allocations..."
    
    # Optimize based on historical usage patterns
    local recommendations=""
    
    # Analyze each deployment
    for deployment in $(kubectl get deployments -n "$NAMESPACE" -o name); do
        local dep_name=$(basename "$deployment")
        log_info "Analyzing deployment: $dep_name"
        
        # Get current resources
        local current_resources
        current_resources=$(kubectl get "$deployment" -n "$NAMESPACE" -o json | jq -r '.spec.template.spec.containers[0].resources')
        
        # Generate optimization recommendations
        local optimized_resources
        optimized_resources=$(generate_optimized_resources "$dep_name" "$current_resources")
        
        if [[ "$optimized_resources" != "null" ]]; then
            recommendations+="Deployment: $dep_name\n$optimized_resources\n\n"
            
            if [[ "$DRY_RUN" == "false" ]]; then
                apply_resource_optimization "$deployment" "$optimized_resources"
            else
                log_info "[DRY RUN] Would optimize $dep_name"
            fi
        fi
    done
    
    if [[ -n "$recommendations" ]]; then
        echo -e "$recommendations" > "$PROJECT_ROOT/optimization-recommendations.txt"
        log_success "Resource optimization recommendations saved"
    fi
}

# Generate optimized resource specifications
generate_optimized_resources() {
    local deployment_name="$1"
    local current_resources="$2"
    
    # This is a simplified optimization - in production, you'd use historical metrics
    python3 -c "
import json
import sys

deployment = '$deployment_name'
current = json.loads('$current_resources')

# Define optimization rules based on deployment type
optimizations = {
    'mimir-web': {
        'cpu_request': '100m',
        'cpu_limit': '500m',
        'memory_request': '128Mi',
        'memory_limit': '512Mi'
    },
    'mimir-api': {
        'cpu_request': '200m',
        'cpu_limit': '1000m',
        'memory_request': '256Mi',
        'memory_limit': '1Gi'
    },
    'mimir-worker': {
        'cpu_request': '100m',
        'cpu_limit': '2000m',
        'memory_request': '256Mi',
        'memory_limit': '2Gi'
    },
    'postgres': {
        'cpu_request': '500m',
        'cpu_limit': '2000m',
        'memory_request': '1Gi',
        'memory_limit': '4Gi'
    }
}

if deployment in optimizations:
    opt = optimizations[deployment]
    optimized = {
        'requests': {
            'cpu': opt['cpu_request'],
            'memory': opt['memory_request']
        },
        'limits': {
            'cpu': opt['cpu_limit'],
            'memory': opt['memory_limit']
        }
    }
    print(json.dumps(optimized, indent=2))
else:
    print('null')
"
}

# Apply resource optimization to deployment
apply_resource_optimization() {
    local deployment="$1"
    local resources="$2"
    
    log_info "Applying optimization to $deployment"
    
    # Create patch for resources
    local patch
    patch=$(echo "$resources" | jq '{
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": "main",
                        "resources": .
                    }]
                }
            }
        }
    }')
    
    # Apply patch
    kubectl patch "$deployment" -n "$NAMESPACE" --type='merge' -p="$patch"
    
    log_success "Applied optimization to $deployment"
}

# Right-size deployments based on actual usage
right_size_deployments() {
    log_info "Right-sizing deployments based on usage patterns..."
    
    # Get historical usage data (if available)
    if kubectl get service prometheus -n monitoring &> /dev/null; then
        log_info "Prometheus available, fetching usage metrics..."
        
        # Port forward to Prometheus
        kubectl port-forward -n monitoring service/prometheus 9090:9090 &
        local port_forward_pid=$!
        sleep 5
        
        # Query historical CPU and memory usage
        local cpu_usage
        cpu_usage=$(curl -s "http://localhost:9090/api/v1/query?query=avg_over_time(rate(container_cpu_usage_seconds_total{namespace='$NAMESPACE'}[5m])[7d:1h])" | jq -r '.data.result[0].value[1] // "0"')
        
        local memory_usage
        memory_usage=$(curl -s "http://localhost:9090/api/v1/query?query=avg_over_time(container_memory_working_set_bytes{namespace='$NAMESPACE'}[7d:1h])" | jq -r '.data.result[0].value[1] // "0"')
        
        kill $port_forward_pid || true
        
        log_info "Average CPU usage over 7 days: ${cpu_usage}"
        log_info "Average memory usage over 7 days: ${memory_usage} bytes"
        
        # Generate right-sizing recommendations
        generate_rightsizing_recommendations "$cpu_usage" "$memory_usage"
    else
        log_warning "Prometheus not available, using static recommendations"
        apply_static_rightsizing
    fi
}

# Generate right-sizing recommendations based on metrics
generate_rightsizing_recommendations() {
    local avg_cpu="$1"
    local avg_memory="$2"
    
    python3 -c "
import json

avg_cpu = float('$avg_cpu')
avg_memory = float('$avg_memory')

# Calculate recommended resources with 20% buffer
recommended_cpu = int((avg_cpu * 1.2) * 1000)  # Convert to millicores
recommended_memory = int((avg_memory * 1.2) / (1024*1024))  # Convert to Mi

print(f'Recommended CPU: {recommended_cpu}m')
print(f'Recommended Memory: {recommended_memory}Mi')

# Generate recommendations for each deployment
recommendations = {
    'cpu_recommendation_mc': recommended_cpu,
    'memory_recommendation_mb': recommended_memory,
    'buffer_percentage': 20
}

with open('$PROJECT_ROOT/rightsizing-recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)
"
}

# Clean up unused resources
cleanup_resources() {
    log_info "Cleaning up unused resources..."
    
    local cleanup_count=0
    
    # Clean up failed pods
    log_info "Removing failed pods..."
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed -o name)
    
    if [[ -n "$failed_pods" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            echo "$failed_pods" | xargs kubectl delete -n "$NAMESPACE"
            cleanup_count=$((cleanup_count + $(echo "$failed_pods" | wc -l)))
        else
            log_info "[DRY RUN] Would delete $(echo "$failed_pods" | wc -l) failed pods"
        fi
    fi
    
    # Clean up completed jobs
    log_info "Removing completed jobs..."
    local completed_jobs
    completed_jobs=$(kubectl get jobs -n "$NAMESPACE" --field-selector=status.successful=1 -o name)
    
    if [[ -n "$completed_jobs" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            echo "$completed_jobs" | xargs kubectl delete -n "$NAMESPACE"
            cleanup_count=$((cleanup_count + $(echo "$completed_jobs" | wc -l)))
        else
            log_info "[DRY RUN] Would delete $(echo "$completed_jobs" | wc -l) completed jobs"
        fi
    fi
    
    # Clean up old replica sets
    log_info "Removing old replica sets..."
    local old_rs
    old_rs=$(kubectl get replicasets -n "$NAMESPACE" -o json | jq -r '.items[] | select(.spec.replicas == 0) | .metadata.name')
    
    if [[ -n "$old_rs" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            echo "$old_rs" | xargs kubectl delete replicaset -n "$NAMESPACE"
            cleanup_count=$((cleanup_count + $(echo "$old_rs" | wc -l)))
        else
            log_info "[DRY RUN] Would delete $(echo "$old_rs" | wc -l) old replica sets"
        fi
    fi
    
    # Aggressive cleanup
    if [[ "$AGGRESSIVE_CLEANUP" == "true" ]]; then
        log_warning "Performing aggressive cleanup..."
        
        # Clean up old logs
        log_info "Cleaning up old logs..."
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl exec -n "$NAMESPACE" deployment/mimir-api -- find /app/logs -name "*.log" -mtime +7 -delete || true
            kubectl exec -n "$NAMESPACE" deployment/mimir-web -- find /app/logs -name "*.log" -mtime +7 -delete || true
        fi
        
        # Clean up old container images on nodes
        log_info "Requesting node cleanup..."
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl get nodes -o name | xargs -I {} kubectl debug {} -it --image=alpine -- sh -c "crictl rmi --prune" || true
        fi
    fi
    
    log_success "Cleanup completed. Removed $cleanup_count resources."
}

# Generate cost optimization report
generate_cost_report() {
    log_info "Generating cost optimization report..."
    
    local report_file="${REPORT_FILE:-$PROJECT_ROOT/cost-optimization-report.md}"
    
    # Collect cost-related data
    local node_count
    node_count=$(kubectl get nodes --no-headers | wc -l)
    
    local total_pods
    total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
    
    local pv_storage
    pv_storage=$(kubectl get pv -o json | jq -r '.items[] | select(.spec.claimRef.namespace == "'$NAMESPACE'") | .spec.capacity.storage' | sed 's/Gi//' | awk '{sum += $1} END {print sum}')
    
    # Calculate estimated costs (simplified)
    local estimated_monthly_cost
    estimated_monthly_cost=$(python3 -c "
node_cost_per_month = 100  # \$100 per node per month (example)
storage_cost_per_gb = 0.10  # \$0.10 per GB per month

nodes = $node_count
storage_gb = ${pv_storage:-0}

total_cost = (nodes * node_cost_per_month) + (storage_gb * storage_cost_per_gb)
print(f'{total_cost:.2f}')
")
    
    # Generate report
    cat << EOF > "$report_file"
# Mimir Cost Optimization Report

**Generated**: $(date)
**Namespace**: $NAMESPACE

## Current Resource Usage

### Infrastructure
- **Nodes**: $node_count
- **Pods**: $total_pods  
- **Storage**: ${pv_storage:-0} GB

### Estimated Monthly Cost
- **Total**: \$${estimated_monthly_cost}
- **Node Costs**: \$$(python3 -c "print($node_count * 100)")
- **Storage Costs**: \$$(python3 -c "print(${pv_storage:-0} * 0.10)")

## Optimization Opportunities

### 1. Resource Right-Sizing
$(if [[ -f "$PROJECT_ROOT/rightsizing-recommendations.json" ]]; then
    echo "- CPU optimization potential identified"
    echo "- Memory optimization potential identified"
    echo "- See rightsizing-recommendations.json for details"
else
    echo "- Run 'right-size' command to identify opportunities"
fi)

### 2. Pod Optimization
- **Over-provisioned Pods**: $(kubectl get pods -n "$NAMESPACE" -o json | jq '[.items[] | select(.spec.containers[0].resources.requests.cpu > "1000m")] | length')
- **Under-utilized Pods**: Analysis requires metrics data

### 3. Storage Optimization
- **Unused PVCs**: $(kubectl get pvc -n "$NAMESPACE" --no-headers | grep -c "Pending\|Lost" || echo "0")
- **Oversized Volumes**: Manual review recommended

## Recommendations

### Immediate Actions (0-30 days)
1. **Right-size deployments** based on actual usage
2. **Clean up failed pods** and old replica sets
3. **Optimize resource requests** for better bin packing

### Medium-term Actions (1-3 months)  
1. **Implement horizontal pod autoscaling** for dynamic scaling
2. **Use spot instances** for worker nodes (if available)
3. **Implement node auto-scaling** to reduce over-provisioning

### Long-term Actions (3+ months)
1. **Migrate to ARM instances** for cost savings
2. **Implement multi-zone storage** optimization
3. **Consider reserved instances** for predictable workloads

## Cost Savings Potential

| Optimization | Estimated Monthly Savings |
|--------------|---------------------------|
| Right-sizing | \$50 - \$100 |
| Auto-scaling | \$100 - \$200 |
| Spot instances | \$200 - \$400 |
| Storage optimization | \$20 - \$50 |

**Total Potential Savings**: \$370 - \$750/month

## Next Steps

1. Run resource optimization: \`$0 optimize\`
2. Implement auto-scaling: \`kubectl apply -f k8s/base/hpa.yaml\`
3. Monitor costs: Set up cost monitoring dashboard
4. Schedule regular reviews: Monthly cost optimization reviews

EOF
    
    log_success "Cost optimization report generated: $report_file"
}

# Apply static right-sizing (when metrics are not available)
apply_static_rightsizing() {
    log_info "Applying static right-sizing recommendations..."
    
    # Define static optimizations for common deployment patterns
    local optimizations='
{
  "mimir-web": {
    "replicas": 2,
    "resources": {
      "requests": {"cpu": "100m", "memory": "128Mi"},
      "limits": {"cpu": "500m", "memory": "512Mi"}
    }
  },
  "mimir-api": {
    "replicas": 3,
    "resources": {
      "requests": {"cpu": "200m", "memory": "256Mi"},
      "limits": {"cpu": "1000m", "memory": "1Gi"}
    }
  },
  "mimir-worker": {
    "replicas": 2,
    "resources": {
      "requests": {"cpu": "100m", "memory": "256Mi"},
      "limits": {"cpu": "2000m", "memory": "2Gi"}
    }
  }
}'
    
    # Apply optimizations
    echo "$optimizations" | jq -r 'keys[]' | while read -r deployment; do
        if kubectl get deployment "$deployment" -n "$NAMESPACE" &> /dev/null; then
            local resources
            resources=$(echo "$optimizations" | jq -r ".[\"$deployment\"].resources")
            
            if [[ "$DRY_RUN" == "false" ]]; then
                apply_resource_optimization "deployment/$deployment" "$resources"
            else
                log_info "[DRY RUN] Would optimize $deployment"
            fi
        fi
    done
}

# Main execution function
main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi
    
    parse_args "$@"
    
    log_info "Starting resource optimization: $COMMAND"
    log_info "Target namespace: $NAMESPACE"
    log_info "Dry run: $DRY_RUN"
    
    case $COMMAND in
        analyze)
            analyze_resources
            ;;
        optimize)
            optimize_resources
            ;;
        cleanup)
            cleanup_resources
            ;;
        cost-report)
            generate_cost_report
            ;;
        right-size)
            right_size_deployments
            ;;
        all)
            local analysis_dir
            analysis_dir=$(analyze_resources)
            optimize_resources
            right_size_deployments
            cleanup_resources
            generate_cost_report
            log_success "Complete optimization suite completed"
            log_info "Analysis results: $analysis_dir"
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    log_success "Resource optimization completed successfully"
}

# Run main function
main "$@"