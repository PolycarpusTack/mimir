#!/bin/bash

# Mimir Deployment Script
# Automated deployment with blue-green and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARGOCD_NAMESPACE="argocd"
DEFAULT_TIMEOUT="600s"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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
    deploy ENVIRONMENT [IMAGE_TAG]    Deploy to specified environment
    rollback ENVIRONMENT [REVISION]   Rollback to previous or specified revision
    status ENVIRONMENT                Show deployment status
    promote ENVIRONMENT               Promote blue-green deployment
    abort ENVIRONMENT                 Abort current deployment
    history ENVIRONMENT               Show deployment history
    sync ENVIRONMENT                  Force sync with ArgoCD

Environments:
    development, staging, production

Options:
    -h, --help                    Show this help message
    -t, --timeout DURATION        Set timeout for operations (default: ${DEFAULT_TIMEOUT})
    -w, --wait                    Wait for deployment to complete
    -f, --force                   Force deployment (skip safety checks)
    --dry-run                     Show what would be deployed without executing

Examples:
    $0 deploy staging v1.2.3
    $0 rollback production
    $0 promote production
    $0 status staging

EOF
}

# Parse command line arguments
parse_args() {
    WAIT_FOR_COMPLETION=false
    FORCE_DEPLOY=false
    DRY_RUN=false
    TIMEOUT="$DEFAULT_TIMEOUT"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -w|--wait)
                WAIT_FOR_COMPLETION=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            deploy|rollback|status|promote|abort|history|sync)
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
    
    ENVIRONMENT="${1:-}"
    if [[ -z "$ENVIRONMENT" ]]; then
        log_error "Environment is required"
        show_usage
        exit 1
    fi
    
    # Validate environment
    case $ENVIRONMENT in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    IMAGE_TAG="${2:-}"
    REVISION="${2:-}"
}

# Check prerequisites
check_prerequisites() {
    local required_tools=("kubectl" "argocd" "git" "jq")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done
    
    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context)
    log_info "Current kubectl context: $current_context"
    
    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$current_context" != *"prod"* ]]; then
        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            log_error "Production deployment requires production cluster context"
            log_error "Current context: $current_context"
            log_error "Use --force to override (not recommended)"
            exit 1
        fi
    fi
    
    # Check ArgoCD connection
    if ! argocd cluster list &> /dev/null; then
        log_error "Cannot connect to ArgoCD. Please login first:"
        log_error "argocd login <ARGOCD_SERVER>"
        exit 1
    fi
}

# Get application name for environment
get_app_name() {
    echo "mimir-$ENVIRONMENT"
}

# Deploy function
deploy() {
    local app_name
    app_name=$(get_app_name)
    
    log_info "Starting deployment to $ENVIRONMENT"
    
    if [[ -n "$IMAGE_TAG" ]]; then
        log_info "Updating image tag to: $IMAGE_TAG"
        
        # Update image tag in kustomization.yaml
        local overlay_dir="$PROJECT_ROOT/k8s/overlays/$ENVIRONMENT"
        if [[ ! -d "$overlay_dir" ]]; then
            log_error "Overlay directory not found: $overlay_dir"
            exit 1
        fi
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would update image tag in $overlay_dir/kustomization.yaml"
        else
            cd "$overlay_dir"
            kustomize edit set image "mimir=ghcr.io/your-org/mimir:$IMAGE_TAG"
            kustomize edit set image "mimir-api=ghcr.io/your-org/mimir-api:$IMAGE_TAG"
            kustomize edit set image "mimir-analytics=ghcr.io/your-org/mimir-analytics:$IMAGE_TAG"
            kustomize edit set image "mimir-worker=ghcr.io/your-org/mimir-worker:$IMAGE_TAG"
            
            # Commit changes
            cd "$PROJECT_ROOT"
            git add "$overlay_dir/kustomization.yaml"
            git commit -m "Deploy $IMAGE_TAG to $ENVIRONMENT"
            
            # Push changes
            if ! git push origin "$(git branch --show-current)"; then
                log_error "Failed to push changes to git repository"
                exit 1
            fi
            
            log_success "Updated image tags and pushed to repository"
        fi
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would trigger ArgoCD sync for $app_name"
        return 0
    fi
    
    # Trigger ArgoCD sync
    log_info "Triggering ArgoCD sync for $app_name"
    if ! argocd app sync "$app_name" --timeout "$TIMEOUT"; then
        log_error "ArgoCD sync failed for $app_name"
        exit 1
    fi
    
    if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
        log_info "Waiting for deployment to complete..."
        if ! argocd app wait "$app_name" --timeout "$TIMEOUT"; then
            log_error "Deployment did not complete within timeout"
            exit 1
        fi
        
        # Check health status
        local health_status
        health_status=$(argocd app get "$app_name" -o json | jq -r '.status.health.status')
        
        if [[ "$health_status" == "Healthy" ]]; then
            log_success "Deployment completed successfully"
            show_deployment_info "$app_name"
        else
            log_error "Deployment completed but application is not healthy: $health_status"
            exit 1
        fi
    else
        log_info "Deployment triggered. Use 'status' command to check progress"
    fi
}

# Rollback function
rollback() {
    local app_name
    app_name=$(get_app_name)
    
    log_info "Starting rollback for $ENVIRONMENT"
    
    if [[ -n "$REVISION" ]]; then
        log_info "Rolling back to revision: $REVISION"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would rollback $app_name to revision $REVISION"
            return 0
        fi
        
        if ! argocd app rollback "$app_name" "$REVISION"; then
            log_error "Rollback failed for $app_name"
            exit 1
        fi
    else
        # Get previous revision
        local previous_revision
        previous_revision=$(argocd app get "$app_name" -o json | jq -r '.status.history[-2].id // .status.history[-1].id')
        
        if [[ "$previous_revision" == "null" ]]; then
            log_error "No previous revision found for rollback"
            exit 1
        fi
        
        log_info "Rolling back to previous revision: $previous_revision"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would rollback $app_name to revision $previous_revision"
            return 0
        fi
        
        if ! argocd app rollback "$app_name" "$previous_revision"; then
            log_error "Rollback failed for $app_name"
            exit 1
        fi
    fi
    
    if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
        log_info "Waiting for rollback to complete..."
        if ! argocd app wait "$app_name" --timeout "$TIMEOUT"; then
            log_error "Rollback did not complete within timeout"
            exit 1
        fi
    fi
    
    log_success "Rollback completed successfully"
}

# Status function
show_status() {
    local app_name
    app_name=$(get_app_name)
    
    log_info "Deployment status for $ENVIRONMENT:"
    
    argocd app get "$app_name"
    
    # Show rollout status if using Argo Rollouts
    local namespace="mimir-$ENVIRONMENT"
    if kubectl get rollout -n "$namespace" &> /dev/null; then
        echo
        log_info "Rollout status:"
        kubectl argo rollouts get rollout mimir-web-rollout -n "$namespace" || true
        kubectl argo rollouts get rollout mimir-api-rollout -n "$namespace" || true
    fi
}

# Promote function (for blue-green deployments)
promote() {
    local app_name
    app_name=$(get_app_name)
    local namespace="mimir-$ENVIRONMENT"
    
    log_info "Promoting blue-green deployment for $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would promote rollouts in namespace $namespace"
        return 0
    fi
    
    # Promote web rollout
    if kubectl get rollout mimir-web-rollout -n "$namespace" &> /dev/null; then
        log_info "Promoting web rollout..."
        kubectl argo rollouts promote mimir-web-rollout -n "$namespace"
    fi
    
    # Promote API rollout
    if kubectl get rollout mimir-api-rollout -n "$namespace" &> /dev/null; then
        log_info "Promoting API rollout..."
        kubectl argo rollouts promote mimir-api-rollout -n "$namespace"
    fi
    
    log_success "Promotion completed"
}

# Abort function
abort_deployment() {
    local app_name
    app_name=$(get_app_name)
    local namespace="mimir-$ENVIRONMENT"
    
    log_warning "Aborting deployment for $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would abort rollouts in namespace $namespace"
        return 0
    fi
    
    # Abort web rollout
    if kubectl get rollout mimir-web-rollout -n "$namespace" &> /dev/null; then
        log_info "Aborting web rollout..."
        kubectl argo rollouts abort mimir-web-rollout -n "$namespace"
    fi
    
    # Abort API rollout
    if kubectl get rollout mimir-api-rollout -n "$namespace" &> /dev/null; then
        log_info "Aborting API rollout..."
        kubectl argo rollouts abort mimir-api-rollout -n "$namespace"
    fi
    
    log_success "Deployment aborted"
}

# History function
show_history() {
    local app_name
    app_name=$(get_app_name)
    
    log_info "Deployment history for $ENVIRONMENT:"
    
    argocd app history "$app_name"
}

# Sync function
sync_app() {
    local app_name
    app_name=$(get_app_name)
    
    log_info "Forcing sync for $app_name"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would sync $app_name"
        return 0
    fi
    
    if ! argocd app sync "$app_name" --force --timeout "$TIMEOUT"; then
        log_error "Sync failed for $app_name"
        exit 1
    fi
    
    log_success "Sync completed"
}

# Show deployment information
show_deployment_info() {
    local app_name="$1"
    local namespace="mimir-$ENVIRONMENT"
    
    echo
    log_info "Deployment Information:"
    echo "Application: $app_name"
    echo "Namespace: $namespace"
    echo "Environment: $ENVIRONMENT"
    
    # Show service endpoints
    if kubectl get ingress -n "$namespace" &> /dev/null; then
        echo
        log_info "Service Endpoints:"
        kubectl get ingress -n "$namespace" -o custom-columns=NAME:.metadata.name,HOSTS:.spec.rules[*].host,PORTS:.spec.rules[*].http.paths[*].backend.service.port.number
    fi
    
    # Show resource usage
    echo
    log_info "Resource Usage:"
    kubectl top pods -n "$namespace" || echo "Metrics not available"
}

# Main execution
main() {
    parse_args "$@"
    check_prerequisites
    
    case $COMMAND in
        deploy)
            deploy
            ;;
        rollback)
            rollback
            ;;
        status)
            show_status
            ;;
        promote)
            promote
            ;;
        abort)
            abort_deployment
            ;;
        history)
            show_history
            ;;
        sync)
            sync_app
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"