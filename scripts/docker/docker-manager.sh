#!/bin/bash

# Mimir Docker Environment Manager
# Comprehensive script for managing different Docker environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_NAME="mimir"

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
ACTION=""
SERVICES=""
BUILD_CACHE="${BUILD_CACHE:-true}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"

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
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Mimir Docker Environment Manager

Usage: $0 [OPTIONS] ACTION [SERVICES...]

ACTIONS:
    up              Start services
    down            Stop and remove services
    restart         Restart services
    build           Build images
    rebuild         Force rebuild images (no cache)
    logs            Show logs
    status          Show service status
    shell           Open shell in service
    exec            Execute command in service
    clean           Clean up containers, networks, and volumes
    deep-clean      Clean everything including images
    test            Run tests
    migrate         Run database migrations
    backup          Backup data
    restore         Restore data from backup

ENVIRONMENTS:
    development     Development environment (default)
    production      Production environment
    testing         Testing environment

OPTIONS:
    -e, --env ENV       Set environment (development|production|testing)
    -s, --services      Specify services to operate on
    -v, --verbose       Enable verbose output
    -n, --dry-run       Show what would be done without executing
    --no-cache          Disable build cache
    --pull              Pull latest images before building
    --force             Force operation (use with caution)
    -h, --help          Show this help message

EXAMPLES:
    $0 up                           # Start development environment
    $0 -e production up             # Start production environment
    $0 build --no-cache             # Rebuild all images without cache
    $0 logs -s mimir-api            # Show logs for API service
    $0 shell mimir-worker           # Open shell in worker container
    $0 test --profile unit-tests    # Run unit tests
    $0 clean                        # Clean up development environment

PROFILES (for testing):
    unit-tests          Run unit tests only
    integration-tests   Run integration tests
    e2e-tests          Run end-to-end tests
    load-tests         Run load tests
    security-tests     Run security tests
    migration-tests    Run migration tests

EOF
}

# Function to check if Docker is available
check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running or not accessible"
        exit 1
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
}

# Function to get compose files for environment
get_compose_files() {
    local env=$1
    local files=()
    
    case $env in
        development)
            files+=("-f" "docker-compose.yml")
            files+=("-f" "docker-compose.development.yml")
            ;;
        production)
            files+=("-f" "docker-compose.production.yml")
            ;;
        testing)
            files+=("-f" "docker-compose.yml")
            files+=("-f" "docker-compose.testing.yml")
            ;;
        *)
            log_error "Unknown environment: $env"
            exit 1
            ;;
    esac
    
    echo "${files[@]}"
}

# Function to get environment file
get_env_file() {
    local env=$1
    case $env in
        development)
            echo ".env.development"
            ;;
        production)
            if [ -f ".env.production" ]; then
                echo ".env.production"
            else
                log_warning "Production environment file not found. Using .env.production.example"
                echo ".env.production.example"
            fi
            ;;
        testing)
            echo ".env.testing"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to execute Docker Compose command
docker_compose_exec() {
    local cmd=("docker-compose")
    
    # Add compose files
    local compose_files
    read -ra compose_files <<< "$(get_compose_files "$ENVIRONMENT")"
    cmd+=("${compose_files[@]}")
    
    # Add environment file
    local env_file
    env_file=$(get_env_file "$ENVIRONMENT")
    if [ -n "$env_file" ] && [ -f "$env_file" ]; then
        cmd+=("--env-file" "$env_file")
    fi
    
    # Add project name
    cmd+=("-p" "${PROJECT_NAME}_${ENVIRONMENT}")
    
    # Add the actual command
    cmd+=("$@")
    
    if [ "$VERBOSE" = true ]; then
        log_info "Executing: ${cmd[*]}"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: ${cmd[*]}"
        return 0
    fi
    
    "${cmd[@]}"
}

# Function to start services
action_up() {
    log "Starting $ENVIRONMENT environment..."
    
    # Check if environment file exists
    local env_file
    env_file=$(get_env_file "$ENVIRONMENT")
    if [ -n "$env_file" ] && [ ! -f "$env_file" ]; then
        log_error "Environment file $env_file not found"
        if [ "$ENVIRONMENT" = "production" ]; then
            log_info "Copy .env.production.example to .env.production and configure it"
        fi
        exit 1
    fi
    
    # Build services if needed
    if [ "$BUILD_CACHE" = true ]; then
        log "Building services (if needed)..."
        docker_compose_exec build $SERVICES
    fi
    
    # Start services
    docker_compose_exec up -d $SERVICES
    
    # Show status
    action_status
    
    log_success "$ENVIRONMENT environment started successfully"
}

# Function to stop services
action_down() {
    log "Stopping $ENVIRONMENT environment..."
    
    docker_compose_exec down --remove-orphans
    
    log_success "$ENVIRONMENT environment stopped"
}

# Function to restart services
action_restart() {
    log "Restarting $ENVIRONMENT environment..."
    
    if [ -n "$SERVICES" ]; then
        docker_compose_exec restart $SERVICES
    else
        action_down
        action_up
    fi
    
    log_success "$ENVIRONMENT environment restarted"
}

# Function to build services
action_build() {
    log "Building services for $ENVIRONMENT environment..."
    
    local build_args=()
    
    if [ "$BUILD_CACHE" = false ]; then
        build_args+=("--no-cache")
    fi
    
    if [ "${PULL_IMAGES:-false}" = true ]; then
        build_args+=("--pull")
    fi
    
    docker_compose_exec build "${build_args[@]}" $SERVICES
    
    log_success "Build completed"
}

# Function to force rebuild
action_rebuild() {
    log "Force rebuilding services for $ENVIRONMENT environment..."
    
    BUILD_CACHE=false
    PULL_IMAGES=true
    action_build
}

# Function to show logs
action_logs() {
    local follow_flag=""
    local tail_lines="100"
    
    if [ "${FOLLOW_LOGS:-false}" = true ]; then
        follow_flag="-f"
    fi
    
    if [ -n "${LOG_LINES:-}" ]; then
        tail_lines="$LOG_LINES"
    fi
    
    docker_compose_exec logs $follow_flag --tail="$tail_lines" $SERVICES
}

# Function to show status
action_status() {
    log_info "Status of $ENVIRONMENT environment:"
    echo
    
    docker_compose_exec ps
    echo
    
    log_info "Service health:"
    docker_compose_exec ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"
    echo
    
    # Show resource usage
    if command -v docker >/dev/null 2>&1; then
        log_info "Resource usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
            $(docker_compose_exec ps -q 2>/dev/null | tr '\n' ' ') 2>/dev/null || true
    fi
}

# Function to open shell in service
action_shell() {
    if [ -z "$SERVICES" ]; then
        log_error "Please specify a service for shell access"
        exit 1
    fi
    
    local service="${SERVICES%% *}"  # Take first service
    log "Opening shell in $service..."
    
    # Try different shells in order of preference
    for shell in bash sh; do
        if docker_compose_exec exec "$service" which "$shell" >/dev/null 2>&1; then
            docker_compose_exec exec "$service" "$shell"
            return 0
        fi
    done
    
    log_error "No suitable shell found in $service"
    exit 1
}

# Function to execute command in service
action_exec() {
    if [ -z "$SERVICES" ]; then
        log_error "Please specify a service for command execution"
        exit 1
    fi
    
    if [ -z "${EXEC_COMMAND:-}" ]; then
        log_error "Please specify a command to execute"
        exit 1
    fi
    
    local service="${SERVICES%% *}"  # Take first service
    log "Executing command in $service: $EXEC_COMMAND"
    
    docker_compose_exec exec "$service" sh -c "$EXEC_COMMAND"
}

# Function to clean up
action_clean() {
    log "Cleaning up $ENVIRONMENT environment..."
    
    # Stop and remove containers
    docker_compose_exec down --remove-orphans --volumes
    
    # Remove networks
    docker network prune -f
    
    # Remove unused volumes (be careful with this)
    if [ "${FORCE_CLEAN:-false}" = true ]; then
        docker volume prune -f
    fi
    
    log_success "Cleanup completed"
}

# Function to deep clean
action_deep_clean() {
    log_warning "Deep cleaning will remove all containers, networks, volumes, and images!"
    
    if [ "${FORCE_CLEAN:-false}" != true ]; then
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Deep clean cancelled"
            return 0
        fi
    fi
    
    # Clean environment
    action_clean
    
    # Remove images
    docker_compose_exec down --rmi all
    
    # System prune
    docker system prune -a -f --volumes
    
    log_success "Deep clean completed"
}

# Function to run tests
action_test() {
    log "Running tests in $ENVIRONMENT environment..."
    
    # Set testing environment
    ENVIRONMENT=testing
    
    # Determine test profile
    local profile="${TEST_PROFILE:-unit-tests}"
    
    log "Running $profile..."
    
    case $profile in
        unit-tests)
            docker_compose_exec run --rm test-runner
            ;;
        integration-tests)
            docker_compose_exec --profile integration-tests up -d api-integration-test
            docker_compose_exec run --rm test-runner python -m pytest tests/integration/ -v
            ;;
        e2e-tests)
            docker_compose_exec --profile e2e-tests up -d
            docker_compose_exec run --rm e2e-test
            ;;
        load-tests)
            docker_compose_exec --profile load-tests up -d api-integration-test
            docker_compose_exec run --rm load-test
            ;;
        security-tests)
            docker_compose_exec --profile security-tests up -d api-integration-test
            docker_compose_exec run --rm security-test
            ;;
        migration-tests)
            docker_compose_exec --profile migration-tests run --rm migration-test
            ;;
        all)
            for test_type in unit-tests integration-tests migration-tests; do
                TEST_PROFILE=$test_type action_test
            done
            ;;
        *)
            log_error "Unknown test profile: $profile"
            exit 1
            ;;
    esac
    
    log_success "Tests completed"
}

# Function to run database migrations
action_migrate() {
    log "Running database migrations for $ENVIRONMENT environment..."
    
    if [ "$ENVIRONMENT" = "production" ]; then
        log_warning "Running migrations in production!"
        if [ "${FORCE_MIGRATE:-false}" != true ]; then
            read -p "Are you sure? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Migration cancelled"
                return 0
            fi
        fi
    fi
    
    # Use API service for migrations
    docker_compose_exec run --rm mimir-api alembic upgrade head
    
    log_success "Migrations completed"
}

# Function to backup data
action_backup() {
    log "Creating backup of $ENVIRONMENT environment..."
    
    local backup_dir="./backups/$(date +%Y%m%d_%H%M%S)_${ENVIRONMENT}"
    mkdir -p "$backup_dir"
    
    # Backup database
    log "Backing up database..."
    docker_compose_exec exec postgres pg_dumpall -U "$POSTGRES_USER" > "$backup_dir/database.sql"
    
    # Backup Redis (if applicable)
    if docker_compose_exec ps redis >/dev/null 2>&1; then
        log "Backing up Redis..."
        docker_compose_exec exec redis redis-cli --rdb - > "$backup_dir/redis.rdb"
    fi
    
    # Backup volumes
    log "Backing up volumes..."
    docker run --rm -v "${PROJECT_NAME}_${ENVIRONMENT}_postgres_data:/data" -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/postgres_data.tar.gz -C /data .
    
    log_success "Backup created at $backup_dir"
}

# Function to restore data
action_restore() {
    if [ -z "${BACKUP_PATH:-}" ]; then
        log_error "Please specify backup path with BACKUP_PATH environment variable"
        exit 1
    fi
    
    if [ ! -d "$BACKUP_PATH" ]; then
        log_error "Backup directory not found: $BACKUP_PATH"
        exit 1
    fi
    
    log_warning "Restoring will overwrite existing data!"
    if [ "${FORCE_RESTORE:-false}" != true ]; then
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Restore cancelled"
            return 0
        fi
    fi
    
    log "Restoring from $BACKUP_PATH..."
    
    # Stop services
    action_down
    
    # Restore database
    if [ -f "$BACKUP_PATH/database.sql" ]; then
        log "Restoring database..."
        # Start only postgres
        docker_compose_exec up -d postgres
        sleep 10  # Wait for postgres to be ready
        docker_compose_exec exec -T postgres psql -U "$POSTGRES_USER" < "$BACKUP_PATH/database.sql"
    fi
    
    # Restore volumes
    if [ -f "$BACKUP_PATH/postgres_data.tar.gz" ]; then
        log "Restoring volumes..."
        docker run --rm -v "${PROJECT_NAME}_${ENVIRONMENT}_postgres_data:/data" -v "$BACKUP_PATH:/backup" alpine tar xzf /backup/postgres_data.tar.gz -C /data
    fi
    
    # Start all services
    action_up
    
    log_success "Restore completed"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--services)
                SERVICES="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-cache)
                BUILD_CACHE=false
                shift
                ;;
            --pull)
                PULL_IMAGES=true
                shift
                ;;
            --force)
                FORCE_CLEAN=true
                FORCE_MIGRATE=true
                FORCE_RESTORE=true
                shift
                ;;
            --profile)
                TEST_PROFILE="$2"
                shift 2
                ;;
            --follow)
                FOLLOW_LOGS=true
                shift
                ;;
            --lines)
                LOG_LINES="$2"
                shift 2
                ;;
            --command)
                EXEC_COMMAND="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            up|down|restart|build|rebuild|logs|status|shell|exec|clean|deep-clean|test|migrate|backup|restore)
                if [ -z "$ACTION" ]; then
                    ACTION="$1"
                else
                    SERVICES="$SERVICES $1"
                fi
                shift
                ;;
            *)
                if [ -z "$ACTION" ]; then
                    log_error "Unknown action: $1"
                    show_usage
                    exit 1
                else
                    SERVICES="$SERVICES $1"
                    shift
                fi
                ;;
        esac
    done
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    check_docker
    
    # Parse arguments
    parse_args "$@"
    
    # Validate action
    if [ -z "$ACTION" ]; then
        log_error "No action specified"
        show_usage
        exit 1
    fi
    
    # Set up environment
    log_info "Environment: $ENVIRONMENT"
    log_info "Action: $ACTION"
    if [ -n "$SERVICES" ]; then
        log_info "Services: $SERVICES"
    fi
    
    # Execute action
    case $ACTION in
        up)
            action_up
            ;;
        down)
            action_down
            ;;
        restart)
            action_restart
            ;;
        build)
            action_build
            ;;
        rebuild)
            action_rebuild
            ;;
        logs)
            action_logs
            ;;
        status)
            action_status
            ;;
        shell)
            action_shell
            ;;
        exec)
            action_exec
            ;;
        clean)
            action_clean
            ;;
        deep-clean)
            action_deep_clean
            ;;
        test)
            action_test
            ;;
        migrate)
            action_migrate
            ;;
        backup)
            action_backup
            ;;
        restore)
            action_restore
            ;;
        *)
            log_error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi