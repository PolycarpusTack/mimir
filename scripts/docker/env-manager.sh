#!/bin/bash

# Mimir Environment Configuration Manager
# Manages environment variables and configurations across different deployment environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/config/environments"
TEMPLATES_DIR="${CONFIG_DIR}/templates"

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

# Function to create configuration directories
create_config_structure() {
    log "Creating configuration directory structure..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$TEMPLATES_DIR"
    mkdir -p "${CONFIG_DIR}/development"
    mkdir -p "${CONFIG_DIR}/staging"
    mkdir -p "${CONFIG_DIR}/production"
    
    log_success "Configuration directories created"
}

# Function to create environment template
create_env_template() {
    local env="$1"
    local template_file="${TEMPLATES_DIR}/${env}.env.template"
    
    log "Creating environment template for $env..."
    
    case "$env" in
        development)
            cat > "$template_file" << 'EOF'
# ===================================
# Mimir Development Environment
# ===================================

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
POSTGRES_DB=mimir_news_dev
POSTGRES_USER=mimir_dev_user
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}
REDIS_MAX_CONNECTIONS=50

# Celery Configuration
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}
CELERY_TASK_ROUTES={"analytics.*": {"queue": "analytics"}, "scraping.*": {"queue": "scraping"}}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
API_TIMEOUT=60
API_MAX_REQUESTS=100

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000
FLASK_ENV=development
FLASK_DEBUG=1

# Analytics Configuration
ANALYTICS_HOST=0.0.0.0
ANALYTICS_PORT=5001
ANALYTICS_DEBUG=true

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://localhost:5000,http://localhost:5001,http://localhost:8000

# Security (Development - NOT for production)
SECRET_KEY=dev-secret-key-change-in-production
JWT_SECRET_KEY=dev-jwt-secret-change-in-production
JWT_ACCESS_TOKEN_EXPIRES=3600
JWT_REFRESH_TOKEN_EXPIRES=2592000

# Feature Flags
ENABLE_API_DOCS=true
ENABLE_DEBUG_TOOLBAR=true
ENABLE_PROFILER=true
ENABLE_RATE_LIMITING=false

# File Upload
MAX_CONTENT_LENGTH=16777216  # 16MB
UPLOAD_FOLDER=/tmp/uploads

# Monitoring and Observability
METRICS_ENABLED=true
TRACING_ENABLED=false
HEALTH_CHECK_INTERVAL=30

# Development Tools
FLOWER_PORT=5555
ADMINER_PORT=8080
REDIS_COMMANDER_PORT=8081

# Resource Limits (Development)
WORKER_CONCURRENCY=2
WORKER_MAX_TASKS_PER_CHILD=100
WORKER_PREFETCH_MULTIPLIER=1

# Backup Settings
BACKUP_ENABLED=false
BACKUP_LOCAL_PATH=/tmp/backups

# Email Configuration (Development)
MAIL_SERVER=localhost
MAIL_PORT=1025  # MailHog
MAIL_USE_TLS=false
MAIL_USE_SSL=false

# External Services (Development)
OPENAI_API_BASE_URL=http://localhost:11434  # Ollama for local testing
WEBHOOK_BASE_URL=http://localhost:8000

# Cache Settings
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=300
CACHE_KEY_PREFIX=mimir_dev

# Session Configuration
SESSION_TYPE=redis
SESSION_PERMANENT=false
SESSION_USE_SIGNER=true
SESSION_KEY_PREFIX=session:

# Timezone
TZ=UTC
EOF
            ;;
        staging)
            cat > "$template_file" << 'EOF'
# ===================================
# Mimir Staging Environment
# ===================================

# Application Settings
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
POSTGRES_DB=mimir_news_staging
POSTGRES_USER=mimir_staging_user
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_TIMEOUT=30
REDIS_SOCKET_CONNECT_TIMEOUT=30

# Celery Configuration
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}
CELERY_TASK_ROUTES={"analytics.*": {"queue": "analytics"}, "scraping.*": {"queue": "scraping"}}
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_ACKS_LATE=true
CELERY_WORKER_MAX_TASKS_PER_CHILD=500

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=3
API_TIMEOUT=120
API_MAX_REQUESTS=500
API_MAX_REQUESTS_JITTER=50

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000

# Analytics Configuration
ANALYTICS_HOST=0.0.0.0
ANALYTICS_PORT=5001

# Security
JWT_ACCESS_TOKEN_EXPIRES=1800
JWT_REFRESH_TOKEN_EXPIRES=604800
PASSWORD_MIN_LENGTH=12
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Lax

# Feature Flags
ENABLE_API_DOCS=true
ENABLE_DEBUG_TOOLBAR=false
ENABLE_PROFILER=false
ENABLE_RATE_LIMITING=true

# Rate Limiting
RATE_LIMIT_DEFAULT=100 per hour
RATE_LIMIT_LOGIN=10 per minute
RATE_LIMIT_API=1000 per hour

# File Upload
MAX_CONTENT_LENGTH=33554432  # 32MB
UPLOAD_FOLDER=/app/uploads

# Monitoring and Observability
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=30
PROMETHEUS_PORT=9090

# Resource Limits (Staging)
WORKER_CONCURRENCY=3
WORKER_MAX_TASKS_PER_CHILD=500
WORKER_PREFETCH_MULTIPLIER=1

# Backup Settings
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=14
BACKUP_LOCAL_PATH=/var/backups/mimir

# Email Configuration
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USE_SSL=false

# Cache Settings
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=600
CACHE_KEY_PREFIX=mimir_staging

# Session Configuration
SESSION_TYPE=redis
SESSION_PERMANENT=false
SESSION_USE_SIGNER=true
SESSION_KEY_PREFIX=session:

# SSL/TLS
SSL_DISABLE=false
PREFERRED_URL_SCHEME=https

# Timezone
TZ=UTC
EOF
            ;;
        production)
            cat > "$template_file" << 'EOF'
# ===================================
# Mimir Production Environment
# ===================================

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database Configuration
POSTGRES_DB=mimir_news_prod
POSTGRES_USER=mimir_prod_user
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_POOL_TIMEOUT=60
DATABASE_POOL_RECYCLE=3600
DATABASE_ENGINE_OPTIONS={"pool_pre_ping": true, "pool_recycle": 300}

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}
REDIS_MAX_CONNECTIONS=200
REDIS_SOCKET_TIMEOUT=30
REDIS_SOCKET_CONNECT_TIMEOUT=30
REDIS_RETRY_ON_TIMEOUT=true

# Celery Configuration
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}
CELERY_TASK_ROUTES={"analytics.*": {"queue": "analytics"}, "scraping.*": {"queue": "scraping"}}
CELERY_WORKER_PREFETCH_MULTIPLIER=1
CELERY_TASK_ACKS_LATE=true
CELERY_WORKER_MAX_TASKS_PER_CHILD=1000
CELERY_TASK_COMPRESSION=gzip
CELERY_RESULT_COMPRESSION=gzip

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300
API_MAX_REQUESTS=1000
API_MAX_REQUESTS_JITTER=100
API_WORKER_CLASS=uvicorn.workers.UvicornWorker

# Web Interface Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000

# Analytics Configuration
ANALYTICS_HOST=0.0.0.0
ANALYTICS_PORT=5001

# Security (CRITICAL - Configure properly)
JWT_ACCESS_TOKEN_EXPIRES=900   # 15 minutes
JWT_REFRESH_TOKEN_EXPIRES=86400  # 24 hours
PASSWORD_MIN_LENGTH=16
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Strict
WTF_CSRF_ENABLED=true
BCRYPT_LOG_ROUNDS=15

# Feature Flags
ENABLE_API_DOCS=false
ENABLE_DEBUG_TOOLBAR=false
ENABLE_PROFILER=false
ENABLE_RATE_LIMITING=true

# Rate Limiting (Strict for production)
RATE_LIMIT_DEFAULT=50 per hour
RATE_LIMIT_LOGIN=5 per minute
RATE_LIMIT_API=500 per hour
RATE_LIMIT_STORAGE_URL=${REDIS_URL}

# File Upload
MAX_CONTENT_LENGTH=67108864  # 64MB
UPLOAD_FOLDER=/var/uploads
ALLOWED_EXTENSIONS=txt,pdf,png,jpg,jpeg,gif,csv,json

# Monitoring and Observability
METRICS_ENABLED=true
TRACING_ENABLED=true
HEALTH_CHECK_INTERVAL=60
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
JAEGER_AGENT_HOST=jaeger
JAEGER_AGENT_PORT=6831

# Resource Limits (Production)
WORKER_CONCURRENCY=4
WORKER_MAX_TASKS_PER_CHILD=1000
WORKER_PREFETCH_MULTIPLIER=1

# Backup Settings
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_S3_BUCKET=${BACKUP_S3_BUCKET}
BACKUP_S3_REGION=${BACKUP_S3_REGION}
BACKUP_ENCRYPTION_ENABLED=true

# Email Configuration
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USE_SSL=false
MAIL_TIMEOUT=30

# Cache Settings
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=1800
CACHE_KEY_PREFIX=mimir_prod
CACHE_OPTIONS={"socket_timeout": 30, "socket_connect_timeout": 30}

# Session Configuration
SESSION_TYPE=redis
SESSION_PERMANENT=false
SESSION_USE_SIGNER=true
SESSION_KEY_PREFIX=session:
SESSION_COOKIE_NAME=mimir_session

# SSL/TLS
SSL_DISABLE=false
PREFERRED_URL_SCHEME=https
SSL_REDIRECT=true

# Content Security Policy
CSP_DEFAULT_SRC='self'
CSP_SCRIPT_SRC='self' 'unsafe-inline'
CSP_STYLE_SRC='self' 'unsafe-inline'
CSP_IMG_SRC='self' data: https:
CSP_CONNECT_SRC='self'

# CORS (Restrict in production)
CORS_ORIGINS=${PRODUCTION_CORS_ORIGINS}

# Logging
LOG_FORMAT=json
LOG_FILE=/var/log/mimir/application.log
LOG_MAX_BYTES=10485760  # 10MB
LOG_BACKUP_COUNT=5

# Performance
GZIP_COMPRESSION_LEVEL=6
SEND_FILE_MAX_AGE=31536000  # 1 year

# Timezone
TZ=UTC

# Health Checks
HEALTH_CHECK_DATABASE=true
HEALTH_CHECK_REDIS=true
HEALTH_CHECK_EXTERNAL_SERVICES=true

# Feature Toggles
FEATURE_ANALYTICS_ENABLED=true
FEATURE_API_V2_ENABLED=false
FEATURE_ADVANCED_SEARCH_ENABLED=true
FEATURE_EXPORT_ENABLED=true
EOF
            ;;
        *)
            log_error "Unknown environment: $env"
            return 1
            ;;
    esac
    
    log_success "Environment template created: $template_file"
}

# Function to generate environment file from template
generate_env_file() {
    local env="$1"
    local template_file="${TEMPLATES_DIR}/${env}.env.template"
    local output_file="${PROJECT_ROOT}/.env.${env}"
    
    if [ ! -f "$template_file" ]; then
        log_error "Template file not found: $template_file"
        log_info "Create it first with: $0 create-template $env"
        return 1
    fi
    
    if [ -f "$output_file" ]; then
        log_warning "Environment file already exists: $output_file"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "File generation cancelled"
            return 0
        fi
    fi
    
    log "Generating environment file for $env..."
    
    # Copy template to environment file
    cp "$template_file" "$output_file"
    
    # Set appropriate permissions
    chmod 600 "$output_file"
    
    log_success "Environment file generated: $output_file"
    
    if [ "$env" = "production" ]; then
        log_warning "IMPORTANT: Update all placeholder values and secrets in the production environment file!"
        log_info "Use the secrets manager to handle sensitive values: ./scripts/docker/secrets-manager.sh"
    fi
}

# Function to validate environment file
validate_env_file() {
    local env="$1"
    local env_file="${PROJECT_ROOT}/.env.${env}"
    
    if [ ! -f "$env_file" ]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    log "Validating environment file for $env..."
    
    local issues=0
    
    # Check for required variables
    local required_vars=(
        "ENVIRONMENT"
        "POSTGRES_DB"
        "POSTGRES_USER"
        "REDIS_HOST"
        "SECRET_KEY"
        "JWT_SECRET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$env_file"; then
            log_error "Missing required variable: $var"
            issues=$((issues + 1))
        fi
    done
    
    # Check for insecure development values in production
    if [ "$env" = "production" ]; then
        local insecure_patterns=(
            "dev-secret-key"
            "change-in-production"
            "localhost"
            "DEBUG=true"
            "password"
        )
        
        for pattern in "${insecure_patterns[@]}"; do
            if grep -i "$pattern" "$env_file" >/dev/null; then
                log_warning "Found potentially insecure value in production config: $pattern"
                issues=$((issues + 1))
            fi
        done
    fi
    
    # Check for empty values
    local empty_values
    empty_values=$(grep -c "^[A-Z_]*=$" "$env_file" 2>/dev/null || true)
    if [ "$empty_values" -gt 0 ]; then
        log_warning "Found $empty_values empty environment variables"
    fi
    
    if [ $issues -eq 0 ]; then
        log_success "Environment file validation passed"
        return 0
    else
        log_error "Environment file validation failed with $issues issues"
        return 1
    fi
}

# Function to show environment differences
diff_environments() {
    local env1="$1"
    local env2="$2"
    local file1="${PROJECT_ROOT}/.env.${env1}"
    local file2="${PROJECT_ROOT}/.env.${env2}"
    
    if [ ! -f "$file1" ]; then
        log_error "Environment file not found: $file1"
        return 1
    fi
    
    if [ ! -f "$file2" ]; then
        log_error "Environment file not found: $file2"
        return 1
    fi
    
    log_info "Differences between $env1 and $env2 environments:"
    echo
    
    # Show side-by-side diff if available
    if command -v diff >/dev/null 2>&1; then
        diff -u "$file1" "$file2" | head -50 || true
    else
        log_warning "diff command not available"
    fi
}

# Function to merge environment files
merge_environments() {
    local base_env="$1"
    local override_env="$2"
    local output_env="$3"
    local base_file="${PROJECT_ROOT}/.env.${base_env}"
    local override_file="${PROJECT_ROOT}/.env.${override_env}"
    local output_file="${PROJECT_ROOT}/.env.${output_env}"
    
    if [ ! -f "$base_file" ]; then
        log_error "Base environment file not found: $base_file"
        return 1
    fi
    
    if [ ! -f "$override_file" ]; then
        log_error "Override environment file not found: $override_file"
        return 1
    fi
    
    log "Merging $base_env with $override_env into $output_env..."
    
    # Start with base file
    cp "$base_file" "$output_file"
    
    # Apply overrides
    while IFS= read -r line; do
        if [[ $line =~ ^[A-Z_]+=.* ]]; then
            local var_name
            var_name=$(echo "$line" | cut -d'=' -f1)
            
            # Remove existing variable from output
            sed -i "/^${var_name}=/d" "$output_file"
            
            # Add override value
            echo "$line" >> "$output_file"
        fi
    done < "$override_file"
    
    log_success "Environment files merged: $output_file"
}

# Function to list environments
list_environments() {
    log_info "Available environment configurations:"
    echo
    
    echo "Templates:"
    if [ -d "$TEMPLATES_DIR" ]; then
        find "$TEMPLATES_DIR" -name "*.env.template" -type f | sed 's|.*/||; s|\.env\.template||' | sort || echo "  None found"
    else
        echo "  None found"
    fi
    
    echo
    echo "Environment files:"
    find "$PROJECT_ROOT" -name ".env.*" -type f | sed 's|.*/\.env\.||' | sort || echo "  None found"
}

# Function to show environment variables
show_env_vars() {
    local env="$1"
    local env_file="${PROJECT_ROOT}/.env.${env}"
    
    if [ ! -f "$env_file" ]; then
        log_error "Environment file not found: $env_file"
        return 1
    fi
    
    log_info "Environment variables for $env:"
    echo
    
    # Show variables with sensitive values masked
    while IFS= read -r line; do
        if [[ $line =~ ^[A-Z_]+=.* ]]; then
            local var_name
            local var_value
            var_name=$(echo "$line" | cut -d'=' -f1)
            var_value=$(echo "$line" | cut -d'=' -f2-)
            
            # Mask sensitive variables
            if [[ $var_name =~ (PASSWORD|SECRET|KEY|TOKEN) ]]; then
                echo "$var_name=***MASKED***"
            else
                echo "$line"
            fi
        fi
    done < "$env_file"
}

# Function to show usage
show_usage() {
    cat << EOF
Mimir Environment Configuration Manager

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    create-template ENV         Create environment template
    generate ENV                Generate environment file from template
    validate ENV                Validate environment file
    diff ENV1 ENV2              Show differences between environments
    merge BASE OVERRIDE OUTPUT  Merge two environment files
    list                        List available environments
    show ENV                    Show environment variables (sensitive values masked)
    init                        Initialize configuration structure

ENVIRONMENTS:
    development, staging, production

EXAMPLES:
    $0 init                                 # Initialize configuration structure
    $0 create-template production           # Create production template
    $0 generate production                  # Generate .env.production file
    $0 validate production                  # Validate production environment
    $0 diff development production          # Compare environments
    $0 merge development staging custom     # Merge dev + staging -> custom
    $0 show development                     # Show development variables

NOTES:
    - Templates are stored in config/environments/templates/
    - Generated .env files are placed in project root
    - Sensitive values are masked when displaying
    - Use secrets manager for handling sensitive data

EOF
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    case "${1:-}" in
        init)
            create_config_structure
            ;;
        create-template)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            create_config_structure
            create_env_template "$2"
            ;;
        generate)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            generate_env_file "$2"
            ;;
        validate)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            validate_env_file "$2"
            ;;
        diff)
            if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
                log_error "Two environments must be specified"
                show_usage
                exit 1
            fi
            diff_environments "$2" "$3"
            ;;
        merge)
            if [ -z "${2:-}" ] || [ -z "${3:-}" ] || [ -z "${4:-}" ]; then
                log_error "Base, override, and output environments must be specified"
                show_usage
                exit 1
            fi
            merge_environments "$2" "$3" "$4"
            ;;
        list)
            list_environments
            ;;
        show)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            show_env_vars "$2"
            ;;
        help|--help|-h|"")
            show_usage
            ;;
        *)
            log_error "Unknown command: ${1:-}"
            show_usage
            exit 1
            ;;
    esac
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi