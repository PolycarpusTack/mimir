#!/bin/bash

# Mimir Secrets Management System
# Secure handling of secrets for Docker environments

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
SECRETS_DIR="${PROJECT_ROOT}/secrets"
VAULT_FILE="${SECRETS_DIR}/vault.enc"
VAULT_KEY_FILE="${SECRETS_DIR}/.vault_key"
ENVIRONMENT="${ENVIRONMENT:-development}"

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

# Function to check if required tools are available
check_dependencies() {
    local missing_deps=()
    
    if ! command -v openssl >/dev/null 2>&1; then
        missing_deps+=("openssl")
    fi
    
    if ! command -v base64 >/dev/null 2>&1; then
        missing_deps+=("base64")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
}

# Function to generate a secure random key
generate_vault_key() {
    if [ -f "$VAULT_KEY_FILE" ]; then
        log_warning "Vault key file already exists at $VAULT_KEY_FILE"
        read -p "Do you want to generate a new key? This will make existing secrets unreadable. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Key generation cancelled"
            return 0
        fi
    fi
    
    log "Generating new vault key..."
    
    # Create secrets directory if it doesn't exist
    mkdir -p "$SECRETS_DIR"
    
    # Generate 256-bit key
    openssl rand -base64 32 > "$VAULT_KEY_FILE"
    chmod 600 "$VAULT_KEY_FILE"
    
    log_success "Vault key generated and saved to $VAULT_KEY_FILE"
    log_warning "IMPORTANT: Backup this key file securely. Without it, you cannot decrypt your secrets!"
}

# Function to encrypt a file
encrypt_file() {
    local input_file="$1"
    local output_file="$2"
    
    if [ ! -f "$VAULT_KEY_FILE" ]; then
        log_error "Vault key file not found. Run 'generate-key' first."
        exit 1
    fi
    
    if [ ! -f "$input_file" ]; then
        log_error "Input file not found: $input_file"
        exit 1
    fi
    
    local key
    key=$(cat "$VAULT_KEY_FILE")
    
    openssl enc -aes-256-cbc -salt -in "$input_file" -out "$output_file" -k "$key"
    
    log_success "File encrypted: $input_file -> $output_file"
}

# Function to decrypt a file
decrypt_file() {
    local input_file="$1"
    local output_file="$2"
    
    if [ ! -f "$VAULT_KEY_FILE" ]; then
        log_error "Vault key file not found. Cannot decrypt secrets."
        exit 1
    fi
    
    if [ ! -f "$input_file" ]; then
        log_error "Encrypted file not found: $input_file"
        exit 1
    fi
    
    local key
    key=$(cat "$VAULT_KEY_FILE")
    
    openssl enc -aes-256-cbc -d -salt -in "$input_file" -out "$output_file" -k "$key" 2>/dev/null || {
        log_error "Failed to decrypt file. Check if the vault key is correct."
        exit 1
    }
    
    log_success "File decrypted: $input_file -> $output_file"
}

# Function to create secrets template
create_secrets_template() {
    local env="$1"
    local template_file="${SECRETS_DIR}/secrets.${env}.yml"
    
    if [ -f "$template_file" ]; then
        log_warning "Secrets template already exists: $template_file"
        return 0
    fi
    
    mkdir -p "$SECRETS_DIR"
    
    cat > "$template_file" << EOF
# Mimir Secrets Configuration for ${env} environment
# This file contains sensitive information and should be encrypted

database:
  postgres_password: "CHANGE_THIS_STRONG_PASSWORD_123"
  postgres_replication_password: "CHANGE_THIS_REPLICATION_PASSWORD_456"

cache:
  redis_password: "CHANGE_THIS_REDIS_PASSWORD_789"

application:
  secret_key: "$(openssl rand -base64 64 | tr -d '\n')"
  jwt_secret_key: "$(openssl rand -base64 64 | tr -d '\n')"
  encryption_key: "$(openssl rand -base64 32 | tr -d '\n')"

api_keys:
  admin_api_key: "$(openssl rand -hex 32)"
  monitoring_api_key: "$(openssl rand -hex 32)"
  backup_api_key: "$(openssl rand -hex 32)"

ssl:
  certificate_key_passphrase: "CHANGE_THIS_SSL_PASSPHRASE"

external_services:
  smtp_password: "CHANGE_THIS_SMTP_PASSWORD"
  s3_access_key: "CHANGE_THIS_S3_ACCESS_KEY"
  s3_secret_key: "CHANGE_THIS_S3_SECRET_KEY"

monitoring:
  grafana_admin_password: "CHANGE_THIS_GRAFANA_PASSWORD"
  prometheus_web_password: "CHANGE_THIS_PROMETHEUS_PASSWORD"

backup:
  encryption_passphrase: "$(openssl rand -base64 32 | tr -d '\n')"
  pgp_passphrase: "CHANGE_THIS_PGP_PASSPHRASE"

# OAuth/OIDC secrets (if applicable)
oauth:
  client_secret: "CHANGE_THIS_OAUTH_CLIENT_SECRET"
  signing_key: "$(openssl rand -base64 64 | tr -d '\n')"

# Webhook secrets
webhooks:
  github_webhook_secret: "$(openssl rand -hex 32)"
  slack_webhook_secret: "$(openssl rand -hex 32)"
EOF

    log_success "Secrets template created: $template_file"
    log_warning "Please update all placeholder values before encrypting!"
}

# Function to encrypt secrets
encrypt_secrets() {
    local env="$1"
    local secrets_file="${SECRETS_DIR}/secrets.${env}.yml"
    local encrypted_file="${SECRETS_DIR}/secrets.${env}.yml.enc"
    
    if [ ! -f "$secrets_file" ]; then
        log_error "Secrets file not found: $secrets_file"
        log_info "Create it first with: $0 create-template $env"
        exit 1
    fi
    
    encrypt_file "$secrets_file" "$encrypted_file"
    
    # Secure delete original file
    if command -v shred >/dev/null 2>&1; then
        shred -vfz -n 3 "$secrets_file"
    else
        rm -f "$secrets_file"
        log_warning "Original file deleted (secure deletion not available)"
    fi
    
    log_success "Secrets encrypted and original file securely deleted"
}

# Function to decrypt secrets
decrypt_secrets() {
    local env="$1"
    local encrypted_file="${SECRETS_DIR}/secrets.${env}.yml.enc"
    local secrets_file="${SECRETS_DIR}/secrets.${env}.yml"
    
    if [ ! -f "$encrypted_file" ]; then
        log_error "Encrypted secrets file not found: $encrypted_file"
        exit 1
    fi
    
    decrypt_file "$encrypted_file" "$secrets_file"
    chmod 600 "$secrets_file"
    
    log_success "Secrets decrypted to: $secrets_file"
    log_warning "Remember to encrypt and delete this file when done!"
}

# Function to generate Docker secrets
generate_docker_secrets() {
    local env="$1"
    local secrets_file="${SECRETS_DIR}/secrets.${env}.yml"
    local docker_secrets_dir="${SECRETS_DIR}/docker"
    
    if [ ! -f "$secrets_file" ]; then
        log_error "Secrets file not found. Decrypt it first with: $0 decrypt $env"
        exit 1
    fi
    
    if ! command -v yq >/dev/null 2>&1; then
        log_error "yq is required to parse YAML secrets file"
        log_info "Install yq: https://github.com/mikefarah/yq"
        exit 1
    fi
    
    mkdir -p "$docker_secrets_dir"
    
    log "Generating Docker secrets files..."
    
    # Generate individual secret files
    {
        echo "postgres_password=$(yq eval '.database.postgres_password' "$secrets_file")"
        echo "redis_password=$(yq eval '.cache.redis_password' "$secrets_file")"
        echo "secret_key=$(yq eval '.application.secret_key' "$secrets_file")"
        echo "jwt_secret_key=$(yq eval '.application.jwt_secret_key' "$secrets_file")"
    } > "${docker_secrets_dir}/${env}_secrets.env"
    
    # Generate Docker Compose secrets format
    cat > "${docker_secrets_dir}/docker-compose.secrets.yml" << EOF
# Docker Compose Secrets Configuration
# Use with: docker-compose -f docker-compose.yml -f docker-compose.secrets.yml

version: '3.8'

secrets:
  postgres_password:
    file: ${docker_secrets_dir}/postgres_password.txt
  redis_password:
    file: ${docker_secrets_dir}/redis_password.txt
  app_secret_key:
    file: ${docker_secrets_dir}/app_secret_key.txt
  jwt_secret_key:
    file: ${docker_secrets_dir}/jwt_secret_key.txt

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

  redis:
    secrets:
      - redis_password
    environment:
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password

  mimir-api:
    secrets:
      - app_secret_key
      - jwt_secret_key
    environment:
      - SECRET_KEY_FILE=/run/secrets/app_secret_key
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret_key
EOF
    
    # Generate individual secret files
    yq eval '.database.postgres_password' "$secrets_file" > "${docker_secrets_dir}/postgres_password.txt"
    yq eval '.cache.redis_password' "$secrets_file" > "${docker_secrets_dir}/redis_password.txt"
    yq eval '.application.secret_key' "$secrets_file" > "${docker_secrets_dir}/app_secret_key.txt"
    yq eval '.application.jwt_secret_key' "$secrets_file" > "${docker_secrets_dir}/jwt_secret_key.txt"
    
    # Set proper permissions
    chmod 600 "${docker_secrets_dir}"/*.txt
    chmod 600 "${docker_secrets_dir}"/*.env
    
    log_success "Docker secrets generated in: $docker_secrets_dir"
}

# Function to rotate secrets
rotate_secrets() {
    local env="$1"
    local secret_type="$2"
    
    log "Rotating secrets for $env environment..."
    
    # Decrypt current secrets
    decrypt_secrets "$env"
    
    local secrets_file="${SECRETS_DIR}/secrets.${env}.yml"
    local backup_file="${secrets_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Backup current secrets
    cp "$secrets_file" "$backup_file"
    
    case "$secret_type" in
        database)
            log "Rotating database passwords..."
            # Generate new passwords
            local new_postgres_pass
            new_postgres_pass=$(openssl rand -base64 32 | tr -d '\n')
            
            # Update secrets file
            if command -v yq >/dev/null 2>&1; then
                yq eval ".database.postgres_password = \"$new_postgres_pass\"" -i "$secrets_file"
                log_success "Database passwords rotated"
            else
                log_warning "yq not available. Please manually update the database passwords in $secrets_file"
            fi
            ;;
        application)
            log "Rotating application secrets..."
            local new_secret_key
            local new_jwt_key
            new_secret_key=$(openssl rand -base64 64 | tr -d '\n')
            new_jwt_key=$(openssl rand -base64 64 | tr -d '\n')
            
            if command -v yq >/dev/null 2>&1; then
                yq eval ".application.secret_key = \"$new_secret_key\"" -i "$secrets_file"
                yq eval ".application.jwt_secret_key = \"$new_jwt_key\"" -i "$secrets_file"
                log_success "Application secrets rotated"
            else
                log_warning "yq not available. Please manually update the application secrets in $secrets_file"
            fi
            ;;
        all)
            rotate_secrets "$env" "database"
            rotate_secrets "$env" "application"
            ;;
        *)
            log_error "Unknown secret type: $secret_type"
            exit 1
            ;;
    esac
    
    # Re-encrypt secrets
    encrypt_secrets "$env"
    
    log_success "Secret rotation completed. Backup saved as: $backup_file"
    log_warning "Update your running services with the new secrets!"
}

# Function to show usage
show_usage() {
    cat << EOF
Mimir Secrets Management System

Usage: $0 COMMAND [OPTIONS]

COMMANDS:
    generate-key                Generate new vault encryption key
    create-template ENV         Create secrets template for environment
    encrypt ENV                 Encrypt secrets file for environment
    decrypt ENV                 Decrypt secrets file for environment
    docker-secrets ENV          Generate Docker secrets format
    rotate ENV TYPE             Rotate secrets (TYPE: database|application|all)
    list                        List available secret files
    validate ENV                Validate secrets format
    
ENVIRONMENTS:
    development, staging, production

EXAMPLES:
    $0 generate-key                     # Generate encryption key
    $0 create-template production       # Create production secrets template
    $0 encrypt production               # Encrypt production secrets
    $0 decrypt production               # Decrypt for editing
    $0 docker-secrets production        # Generate Docker format
    $0 rotate production database       # Rotate database passwords

SECURITY NOTES:
    - Keep the vault key file (.vault_key) secure and backed up
    - Never commit unencrypted secrets to version control
    - Regularly rotate secrets in production
    - Use proper file permissions (600) for secret files

EOF
}

# Function to list secret files
list_secrets() {
    log_info "Available secret files in $SECRETS_DIR:"
    
    if [ ! -d "$SECRETS_DIR" ]; then
        log_warning "Secrets directory does not exist"
        return 0
    fi
    
    echo
    echo "Encrypted files:"
    find "$SECRETS_DIR" -name "*.enc" -type f | sed 's|.*/||' | sort || echo "  None found"
    
    echo
    echo "Decrypted files (should be temporary):"
    find "$SECRETS_DIR" -name "secrets.*.yml" -type f | sed 's|.*/||' | sort || echo "  None found"
    
    echo
    echo "Vault key:"
    if [ -f "$VAULT_KEY_FILE" ]; then
        echo "  ✓ Vault key exists"
    else
        echo "  ✗ Vault key missing"
    fi
}

# Function to validate secrets format
validate_secrets() {
    local env="$1"
    local secrets_file="${SECRETS_DIR}/secrets.${env}.yml"
    
    if [ ! -f "$secrets_file" ]; then
        log_error "Secrets file not found. Decrypt it first with: $0 decrypt $env"
        exit 1
    fi
    
    log "Validating secrets format for $env environment..."
    
    # Check if yq is available for validation
    if ! command -v yq >/dev/null 2>&1; then
        log_warning "yq not available. Basic file existence check only."
        log_success "Secrets file exists and is readable"
        return 0
    fi
    
    # Validate YAML format
    if ! yq eval '.' "$secrets_file" >/dev/null 2>&1; then
        log_error "Invalid YAML format in secrets file"
        exit 1
    fi
    
    # Check required sections
    local required_sections=("database" "cache" "application")
    local missing_sections=()
    
    for section in "${required_sections[@]}"; do
        if ! yq eval "has(\"$section\")" "$secrets_file" | grep -q "true"; then
            missing_sections+=("$section")
        fi
    done
    
    if [ ${#missing_sections[@]} -ne 0 ]; then
        log_error "Missing required sections: ${missing_sections[*]}"
        exit 1
    fi
    
    # Check for placeholder values
    local placeholders
    placeholders=$(grep -c "CHANGE_THIS" "$secrets_file" 2>/dev/null || true)
    
    if [ "$placeholders" -gt 0 ]; then
        log_warning "Found $placeholders placeholder values that need to be updated"
    fi
    
    log_success "Secrets validation completed"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    # Check dependencies
    check_dependencies
    
    # Parse command
    case "${1:-}" in
        generate-key)
            generate_vault_key
            ;;
        create-template)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            create_secrets_template "$2"
            ;;
        encrypt)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            encrypt_secrets "$2"
            ;;
        decrypt)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            decrypt_secrets "$2"
            ;;
        docker-secrets)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            generate_docker_secrets "$2"
            ;;
        rotate)
            if [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
                log_error "Environment or secret type not specified"
                show_usage
                exit 1
            fi
            rotate_secrets "$2" "$3"
            ;;
        list)
            list_secrets
            ;;
        validate)
            if [ -z "${2:-}" ]; then
                log_error "Environment not specified"
                show_usage
                exit 1
            fi
            validate_secrets "$2"
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