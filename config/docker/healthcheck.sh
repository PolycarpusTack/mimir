#!/bin/sh

# Universal Health Check Script for Mimir Services
# Performs comprehensive health checks for different service types

set -e

# Configuration
SERVICE_TYPE="${SERVICE_TYPE:-web}"
HEALTH_ENDPOINT="${HEALTH_ENDPOINT:-/health}"
HEALTH_PORT="${HEALTH_PORT:-5000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-2}"

# Colors (if supported)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] $1"
}

log_success() {
    echo "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] ✓ $1${NC}"
}

log_warning() {
    echo "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] ⚠ $1${NC}"
}

log_error() {
    echo "${RED}$(date '+%Y-%m-%d %H:%M:%S') [HEALTH] ✗ $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url="$1"
    local timeout="$2"
    
    if command_exists curl; then
        curl -f -s --max-time "$timeout" "$url" >/dev/null 2>&1
    elif command_exists wget; then
        wget -q -T "$timeout" --spider "$url" >/dev/null 2>&1
    else
        log_error "Neither curl nor wget available for HTTP health check"
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if [ "${DATABASE_TYPE:-postgres}" = "postgres" ]; then
            if command_exists pg_isready; then
                if pg_isready -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" >/dev/null 2>&1; then
                    log_success "Database connection OK"
                    return 0
                fi
            else
                # Alternative check using psql
                if command_exists psql; then
                    if PGPASSWORD="$DB_PASSWORD" psql -h "${DB_HOST:-localhost}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}" -d "${DB_NAME:-postgres}" -c "SELECT 1;" >/dev/null 2>&1; then
                        log_success "Database connection OK"
                        return 0
                    fi
                fi
            fi
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            log_warning "Database check failed, retrying in ${RETRY_DELAY}s... (attempt $retries/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
        fi
    done
    
    log_error "Database health check failed after $MAX_RETRIES attempts"
    return 1
}

# Function to check Redis connectivity
check_redis() {
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if command_exists redis-cli; then
            local auth_arg=""
            if [ -n "${REDIS_PASSWORD:-}" ]; then
                auth_arg="-a $REDIS_PASSWORD"
            fi
            
            if redis-cli -h "${REDIS_HOST:-localhost}" -p "${REDIS_PORT:-6379}" $auth_arg ping >/dev/null 2>&1; then
                log_success "Redis connection OK"
                return 0
            fi
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            log_warning "Redis check failed, retrying in ${RETRY_DELAY}s... (attempt $retries/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
        fi
    done
    
    log_error "Redis health check failed after $MAX_RETRIES attempts"
    return 1
}

# Function to check Celery worker
check_celery_worker() {
    if command_exists celery; then
        local broker_url="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
        
        # Check if worker is responsive
        if celery -A celery_app inspect ping -b "$broker_url" >/dev/null 2>&1; then
            log_success "Celery worker responsive"
            return 0
        else
            log_error "Celery worker not responsive"
            return 1
        fi
    else
        log_error "Celery command not available"
        return 1
    fi
}

# Function to check file system health
check_filesystem() {
    local critical_paths="${CRITICAL_PATHS:-/app /tmp}"
    
    for path in $critical_paths; do
        if [ ! -d "$path" ]; then
            log_error "Critical path not accessible: $path"
            return 1
        fi
        
        # Check if path is writable
        if [ ! -w "$path" ]; then
            log_warning "Path not writable: $path"
        fi
    done
    
    # Check disk space
    local available_space
    available_space=$(df /app | awk 'NR==2 {print $4}')
    local min_space="${MIN_DISK_SPACE:-1000000}"  # 1GB in KB
    
    if [ "$available_space" -lt "$min_space" ]; then
        log_error "Low disk space: ${available_space}KB available, minimum ${min_space}KB required"
        return 1
    fi
    
    log_success "Filesystem health OK"
    return 0
}

# Function to check memory usage
check_memory() {
    if [ -r /proc/meminfo ]; then
        local mem_available
        mem_available=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
        local min_memory="${MIN_MEMORY:-100000}"  # 100MB in KB
        
        if [ "$mem_available" -lt "$min_memory" ]; then
            log_warning "Low memory: ${mem_available}KB available"
            # Don't fail on low memory, just warn
        else
            log_success "Memory usage OK"
        fi
    fi
    
    return 0
}

# Function to check Python application
check_python_app() {
    local url="http://localhost:${HEALTH_PORT}${HEALTH_ENDPOINT}"
    local retries=0
    
    while [ $retries -lt $MAX_RETRIES ]; do
        if check_http_endpoint "$url" "$HEALTH_TIMEOUT"; then
            log_success "HTTP endpoint health check passed"
            return 0
        fi
        
        retries=$((retries + 1))
        if [ $retries -lt $MAX_RETRIES ]; then
            log_warning "HTTP health check failed, retrying in ${RETRY_DELAY}s... (attempt $retries/$MAX_RETRIES)"
            sleep "$RETRY_DELAY"
        fi
    done
    
    log_error "HTTP health check failed after $MAX_RETRIES attempts"
    return 1
}

# Main health check function
main_health_check() {
    log "Starting health check for service type: $SERVICE_TYPE"
    
    local checks_passed=0
    local total_checks=0
    
    # Common checks for all services
    total_checks=$((total_checks + 1))
    if check_filesystem; then
        checks_passed=$((checks_passed + 1))
    fi
    
    total_checks=$((total_checks + 1))
    if check_memory; then
        checks_passed=$((checks_passed + 1))
    fi
    
    # Service-specific checks
    case "$SERVICE_TYPE" in
        web|api|analytics)
            total_checks=$((total_checks + 1))
            if check_python_app; then
                checks_passed=$((checks_passed + 1))
            fi
            
            # Check database if required
            if [ "${CHECK_DATABASE:-true}" = "true" ]; then
                total_checks=$((total_checks + 1))
                if check_database; then
                    checks_passed=$((checks_passed + 1))
                fi
            fi
            
            # Check Redis if required
            if [ "${CHECK_REDIS:-true}" = "true" ]; then
                total_checks=$((total_checks + 1))
                if check_redis; then
                    checks_passed=$((checks_passed + 1))
                fi
            fi
            ;;
        worker)
            # Celery worker specific checks
            total_checks=$((total_checks + 1))
            if check_celery_worker; then
                checks_passed=$((checks_passed + 1))
            fi
            
            total_checks=$((total_checks + 1))
            if check_database; then
                checks_passed=$((checks_passed + 1))
            fi
            
            total_checks=$((total_checks + 1))
            if check_redis; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        database)
            # Database specific checks
            total_checks=$((total_checks + 1))
            if check_database; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        cache)
            # Redis specific checks
            total_checks=$((total_checks + 1))
            if check_redis; then
                checks_passed=$((checks_passed + 1))
            fi
            ;;
        *)
            log_warning "Unknown service type: $SERVICE_TYPE, running basic checks only"
            ;;
    esac
    
    # Summary
    log "Health check completed: $checks_passed/$total_checks checks passed"
    
    if [ $checks_passed -eq $total_checks ]; then
        log_success "All health checks passed"
        return 0
    else
        log_error "Some health checks failed"
        return 1
    fi
}

# Run health check
main_health_check