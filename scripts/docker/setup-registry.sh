#!/bin/bash

# Docker Registry Setup Script for Mimir
# Sets up a local Docker registry for development and CI/CD

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY_NAME="mimir-registry"
REGISTRY_PORT="${REGISTRY_PORT:-5000}"
REGISTRY_DATA_DIR="${REGISTRY_DATA_DIR:-./registry-data}"
REGISTRY_CONFIG_DIR="${REGISTRY_CONFIG_DIR:-./config/registry}"

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

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log_success "Docker is running"
}

# Function to create registry directories
create_directories() {
    log "Creating registry directories..."
    mkdir -p "${REGISTRY_DATA_DIR}"
    mkdir -p "${REGISTRY_CONFIG_DIR}"
    log_success "Registry directories created"
}

# Function to create registry configuration
create_registry_config() {
    log "Creating registry configuration..."
    
    cat > "${REGISTRY_CONFIG_DIR}/config.yml" << EOF
version: 0.1
log:
  accesslog:
    disabled: false
  level: info
  formatter: text
  fields:
    service: registry
    environment: development

storage:
  cache:
    blobdescriptor: inmemory
  filesystem:
    rootdirectory: /var/lib/registry
  delete:
    enabled: true

http:
  addr: :5000
  headers:
    X-Content-Type-Options: [nosniff]
    Access-Control-Allow-Origin: ['*']
    Access-Control-Allow-Methods: ['HEAD', 'GET', 'OPTIONS', 'DELETE']
    Access-Control-Allow-Headers: ['Authorization', 'Accept', 'Cache-Control']

health:
  storagedriver:
    enabled: true
    interval: 10s
    threshold: 3

# Enable debugging for development
debug:
  addr: :5001
  prometheus:
    enabled: true
    path: /metrics

# Garbage collection
storage:
  maintenance:
    uploadpurging:
      enabled: true
      age: 168h
      interval: 24h
      dryrun: false
EOF

    log_success "Registry configuration created at ${REGISTRY_CONFIG_DIR}/config.yml"
}

# Function to create htpasswd file for authentication (optional)
create_auth_config() {
    log "Creating authentication configuration..."
    
    # Create auth directory
    mkdir -p "${REGISTRY_CONFIG_DIR}/auth"
    
    # Create htpasswd file with default user (admin/admin)
    # In production, use stronger passwords and proper authentication
    docker run --rm --entrypoint htpasswd registry:2 -Bbn admin admin > "${REGISTRY_CONFIG_DIR}/auth/htpasswd"
    
    log_success "Authentication configured (admin/admin)"
    log_warning "Change default credentials in production!"
}

# Function to create TLS certificates for HTTPS (development only)
create_tls_certs() {
    log "Creating TLS certificates for development..."
    
    mkdir -p "${REGISTRY_CONFIG_DIR}/certs"
    
    # Generate self-signed certificate
    openssl req -newkey rsa:4096 -nodes -sha256 -keyout "${REGISTRY_CONFIG_DIR}/certs/domain.key" \
        -x509 -days 365 -out "${REGISTRY_CONFIG_DIR}/certs/domain.crt" \
        -subj "/C=US/ST=CA/L=San Francisco/O=Mimir/OU=Development/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:mimir-registry,IP:127.0.0.1" 2>/dev/null || {
        log_warning "Failed to create TLS certificates. Registry will run in HTTP mode."
        return 1
    }
    
    log_success "TLS certificates created for development"
    log_warning "These are self-signed certificates for development only!"
}

# Function to start the registry
start_registry() {
    log "Starting Docker registry..."
    
    # Stop existing registry if running
    if docker ps -q -f name="${REGISTRY_NAME}" | grep -q .; then
        log "Stopping existing registry..."
        docker stop "${REGISTRY_NAME}" >/dev/null 2>&1 || true
        docker rm "${REGISTRY_NAME}" >/dev/null 2>&1 || true
    fi
    
    # Start registry with configuration
    docker run -d \
        --name "${REGISTRY_NAME}" \
        --restart=always \
        -p "${REGISTRY_PORT}:5000" \
        -p 5001:5001 \
        -v "$(pwd)/${REGISTRY_DATA_DIR}:/var/lib/registry" \
        -v "$(pwd)/${REGISTRY_CONFIG_DIR}/config.yml:/etc/docker/registry/config.yml" \
        -v "$(pwd)/${REGISTRY_CONFIG_DIR}/auth:/auth" \
        -e "REGISTRY_AUTH=htpasswd" \
        -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm" \
        -e "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd" \
        registry:2
    
    # Wait for registry to be ready
    log "Waiting for registry to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:${REGISTRY_PORT}/v2/" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    
    if curl -s "http://localhost:${REGISTRY_PORT}/v2/" >/dev/null 2>&1; then
        log_success "Registry is running at http://localhost:${REGISTRY_PORT}"
        log_success "Registry metrics available at http://localhost:5001/metrics"
    else
        log_error "Registry failed to start properly"
        return 1
    fi
}

# Function to configure Docker daemon for insecure registry (development)
configure_docker_daemon() {
    log "Configuring Docker daemon for local registry..."
    
    # For development, we need to allow insecure registry
    DOCKER_CONFIG="/etc/docker/daemon.json"
    
    if [ -f "${DOCKER_CONFIG}" ]; then
        log_warning "Docker daemon configuration already exists at ${DOCKER_CONFIG}"
        log "You may need to manually add 'localhost:${REGISTRY_PORT}' to insecure-registries"
    else
        log "Creating Docker daemon configuration..."
        sudo mkdir -p /etc/docker
        sudo tee "${DOCKER_CONFIG}" > /dev/null << EOF
{
  "insecure-registries": ["localhost:${REGISTRY_PORT}"],
  "registry-mirrors": [],
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF
        log_success "Docker daemon configuration created"
        log_warning "Docker daemon needs to be restarted for changes to take effect"
    fi
}

# Function to test registry functionality
test_registry() {
    log "Testing registry functionality..."
    
    # Test image push/pull
    local test_image="hello-world"
    local registry_test_image="localhost:${REGISTRY_PORT}/test/${test_image}"
    
    # Pull test image
    docker pull "${test_image}" >/dev/null 2>&1 || {
        log_error "Failed to pull test image"
        return 1
    }
    
    # Tag for registry
    docker tag "${test_image}" "${registry_test_image}"
    
    # Push to registry (with authentication)
    echo "admin" | docker login -u admin --password-stdin "localhost:${REGISTRY_PORT}" >/dev/null 2>&1 || {
        log_error "Failed to login to registry"
        return 1
    }
    
    docker push "${registry_test_image}" >/dev/null 2>&1 || {
        log_error "Failed to push to registry"
        return 1
    }
    
    # Remove local image and pull from registry
    docker rmi "${registry_test_image}" >/dev/null 2>&1 || true
    docker pull "${registry_test_image}" >/dev/null 2>&1 || {
        log_error "Failed to pull from registry"
        return 1
    }
    
    # Cleanup
    docker rmi "${registry_test_image}" >/dev/null 2>&1 || true
    
    log_success "Registry test completed successfully"
}

# Function to create convenience scripts
create_convenience_scripts() {
    log "Creating convenience scripts..."
    
    # Build and push script
    cat > "./scripts/docker/build-and-push.sh" << 'EOF'
#!/bin/bash

# Build and Push Mimir Images to Registry

set -euo pipefail

REGISTRY="${REGISTRY:-localhost:5000}"
PROJECT_NAME="mimir"
VERSION="${VERSION:-latest}"

# Build images
echo "Building Mimir images..."
docker build -t "${REGISTRY}/${PROJECT_NAME}:${VERSION}" .
docker build -f Dockerfile.api -t "${REGISTRY}/${PROJECT_NAME}:api-${VERSION}" .
docker build -f Dockerfile.worker -t "${REGISTRY}/${PROJECT_NAME}:worker-${VERSION}" .
docker build -f Dockerfile.analytics -t "${REGISTRY}/${PROJECT_NAME}:analytics-${VERSION}" .

# Push images
echo "Pushing images to registry..."
docker push "${REGISTRY}/${PROJECT_NAME}:${VERSION}"
docker push "${REGISTRY}/${PROJECT_NAME}:api-${VERSION}"
docker push "${REGISTRY}/${PROJECT_NAME}:worker-${VERSION}"
docker push "${REGISTRY}/${PROJECT_NAME}:analytics-${VERSION}"

echo "Images pushed successfully!"
EOF

    # Registry management script
    cat > "./scripts/docker/registry-manager.sh" << 'EOF'
#!/bin/bash

# Docker Registry Management Script

set -euo pipefail

REGISTRY_NAME="mimir-registry"
REGISTRY_PORT="5000"

case "${1:-help}" in
    start)
        echo "Starting registry..."
        docker start "${REGISTRY_NAME}" 2>/dev/null || echo "Registry not found or already running"
        ;;
    stop)
        echo "Stopping registry..."
        docker stop "${REGISTRY_NAME}" 2>/dev/null || echo "Registry not running"
        ;;
    restart)
        echo "Restarting registry..."
        docker restart "${REGISTRY_NAME}" 2>/dev/null || echo "Registry not found"
        ;;
    status)
        if docker ps -q -f name="${REGISTRY_NAME}" | grep -q .; then
            echo "Registry is running"
            curl -s "http://localhost:${REGISTRY_PORT}/v2/_catalog" | jq . 2>/dev/null || echo "Registry API not responding"
        else
            echo "Registry is not running"
        fi
        ;;
    logs)
        docker logs -f "${REGISTRY_NAME}" 2>/dev/null || echo "Registry not found"
        ;;
    cleanup)
        echo "Cleaning up unused registry data..."
        docker exec "${REGISTRY_NAME}" registry garbage-collect /etc/docker/registry/config.yml
        ;;
    list)
        echo "Images in registry:"
        curl -s "http://localhost:${REGISTRY_PORT}/v2/_catalog" | jq -r '.repositories[]' 2>/dev/null || echo "No images found"
        ;;
    help|*)
        echo "Usage: $0 {start|stop|restart|status|logs|cleanup|list|help}"
        ;;
esac
EOF

    chmod +x "./scripts/docker/build-and-push.sh"
    chmod +x "./scripts/docker/registry-manager.sh"
    
    log_success "Convenience scripts created"
}

# Function to display usage information
show_usage() {
    cat << EOF
Docker Registry Setup for Mimir

Usage: $0 [OPTIONS]

Options:
    --port PORT          Registry port (default: 5000)
    --data-dir DIR       Registry data directory (default: ./registry-data)
    --config-dir DIR     Registry config directory (default: ./config/registry)
    --no-auth           Skip authentication setup
    --no-tls            Skip TLS certificate generation
    --test              Test registry after setup
    --help              Show this help message

Examples:
    $0                          # Setup with defaults
    $0 --port 5001 --test       # Setup on port 5001 and test
    $0 --no-auth --no-tls       # Setup without auth and TLS

After setup:
    - Registry URL: http://localhost:${REGISTRY_PORT}
    - Username: admin
    - Password: admin (change in production!)
    - Use ./scripts/docker/registry-manager.sh for management
    - Use ./scripts/docker/build-and-push.sh to build and push images

EOF
}

# Main execution
main() {
    local setup_auth=true
    local setup_tls=true
    local run_test=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                REGISTRY_PORT="$2"
                shift 2
                ;;
            --data-dir)
                REGISTRY_DATA_DIR="$2"
                shift 2
                ;;
            --config-dir)
                REGISTRY_CONFIG_DIR="$2"
                shift 2
                ;;
            --no-auth)
                setup_auth=false
                shift
                ;;
            --no-tls)
                setup_tls=false
                shift
                ;;
            --test)
                run_test=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log "Setting up Docker Registry for Mimir"
    echo "====================================="
    
    # Check prerequisites
    check_docker
    
    # Setup registry
    create_directories
    create_registry_config
    
    if [ "$setup_auth" = true ]; then
        create_auth_config
    fi
    
    if [ "$setup_tls" = true ]; then
        create_tls_certs || true
    fi
    
    start_registry
    configure_docker_daemon
    create_convenience_scripts
    
    if [ "$run_test" = true ]; then
        test_registry
    fi
    
    log_success "Docker Registry setup completed!"
    echo
    echo "Registry Information:"
    echo "  URL: http://localhost:${REGISTRY_PORT}"
    echo "  Data Directory: ${REGISTRY_DATA_DIR}"
    echo "  Config Directory: ${REGISTRY_CONFIG_DIR}"
    if [ "$setup_auth" = true ]; then
        echo "  Username: admin"
        echo "  Password: admin"
    fi
    echo
    echo "Management Commands:"
    echo "  ./scripts/docker/registry-manager.sh status"
    echo "  ./scripts/docker/registry-manager.sh list"
    echo "  ./scripts/docker/build-and-push.sh"
    echo
    log_warning "Remember to change default credentials in production!"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi