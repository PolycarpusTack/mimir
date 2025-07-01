#!/bin/bash

# Docker Security Scanning Script for Mimir
# Performs comprehensive security scanning of Docker images

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_REGISTRY="${IMAGE_REGISTRY:-localhost:5000}"
PROJECT_NAME="mimir"
SCAN_REPORT_DIR="./security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Ensure report directory exists
mkdir -p "${SCAN_REPORT_DIR}"

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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install security scanning tools if not present
install_security_tools() {
    log "Checking security scanning tools..."
    
    # Check for Docker Bench Security
    if ! command_exists docker-bench-security; then
        log "Installing Docker Bench Security..."
        git clone https://github.com/docker/docker-bench-security.git /tmp/docker-bench-security || true
    fi
    
    # Check for Trivy
    if ! command_exists trivy; then
        log "Installing Trivy..."
        if command_exists apt-get; then
            sudo apt-get update
            sudo apt-get install wget apt-transport-https gnupg lsb-release -y
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update
            sudo apt-get install trivy -y
        elif command_exists yum; then
            sudo yum install -y wget
            wget -O /tmp/trivy.rpm https://github.com/aquasecurity/trivy/releases/latest/download/trivy_Linux-64bit.rpm
            sudo rpm -ivh /tmp/trivy.rpm
        else
            log_warning "Cannot install Trivy automatically. Please install manually."
        fi
    fi
    
    # Check for Hadolint (Dockerfile linter)
    if ! command_exists hadolint; then
        log "Installing Hadolint..."
        wget -O hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
        chmod +x hadolint
        sudo mv hadolint /usr/local/bin/
    fi
}

# Function to scan Dockerfile with Hadolint
scan_dockerfile() {
    local dockerfile=$1
    local report_file="${SCAN_REPORT_DIR}/hadolint_${TIMESTAMP}_$(basename ${dockerfile}).json"
    
    log "Scanning Dockerfile: ${dockerfile}"
    
    if [ -f "${dockerfile}" ]; then
        hadolint "${dockerfile}" --format json > "${report_file}" 2>&1 || {
            log_warning "Hadolint found issues in ${dockerfile}"
            cat "${report_file}"
        }
        log_success "Dockerfile scan completed: ${report_file}"
    else
        log_error "Dockerfile not found: ${dockerfile}"
    fi
}

# Function to scan Docker image with Trivy
scan_image_trivy() {
    local image_name=$1
    local report_file="${SCAN_REPORT_DIR}/trivy_${TIMESTAMP}_$(echo ${image_name} | tr '/' '_' | tr ':' '_').json"
    
    log "Scanning image with Trivy: ${image_name}"
    
    # Update vulnerability database
    trivy image --download-db-only
    
    # Scan image
    trivy image --format json --output "${report_file}" "${image_name}" || {
        log_error "Trivy scan failed for ${image_name}"
        return 1
    }
    
    # Generate human-readable report
    trivy image --format table "${image_name}" > "${report_file%.json}.txt"
    
    log_success "Trivy scan completed: ${report_file}"
    
    # Check for critical vulnerabilities
    critical_count=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "${report_file}" 2>/dev/null | wc -l)
    high_count=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | .VulnerabilityID' "${report_file}" 2>/dev/null | wc -l)
    
    if [ "${critical_count}" -gt 0 ]; then
        log_error "Found ${critical_count} CRITICAL vulnerabilities in ${image_name}"
        return 1
    elif [ "${high_count}" -gt 5 ]; then
        log_warning "Found ${high_count} HIGH vulnerabilities in ${image_name}"
        return 2
    fi
    
    log_success "No critical vulnerabilities found in ${image_name}"
    return 0
}

# Function to run Docker Bench Security
run_docker_bench() {
    local report_file="${SCAN_REPORT_DIR}/docker_bench_${TIMESTAMP}.log"
    
    log "Running Docker Bench Security..."
    
    if [ -d "/tmp/docker-bench-security" ]; then
        cd /tmp/docker-bench-security
        sudo ./docker-bench-security.sh > "${report_file}" 2>&1
        cd - > /dev/null
        log_success "Docker Bench Security completed: ${report_file}"
    else
        log_error "Docker Bench Security not found"
        return 1
    fi
}

# Function to check image size and optimization
check_image_optimization() {
    local image_name=$1
    local report_file="${SCAN_REPORT_DIR}/optimization_${TIMESTAMP}_$(echo ${image_name} | tr '/' '_' | tr ':' '_').txt"
    
    log "Checking image optimization: ${image_name}"
    
    {
        echo "Image Optimization Report for: ${image_name}"
        echo "Generated at: $(date)"
        echo "=================================="
        echo
        
        # Image size
        echo "Image Size:"
        docker images "${image_name}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
        echo
        
        # Layer analysis
        echo "Layer Analysis:"
        docker history "${image_name}" --format "table {{.CreatedBy}}\t{{.Size}}"
        echo
        
        # Image inspect
        echo "Image Configuration:"
        docker inspect "${image_name}" | jq '.[0] | {
            Architecture: .Architecture,
            Os: .Os,
            Size: .Size,
            VirtualSize: .VirtualSize,
            RootFS: .RootFS,
            Config: {
                User: .Config.User,
                ExposedPorts: .Config.ExposedPorts,
                Env: .Config.Env,
                Cmd: .Config.Cmd,
                Healthcheck: .Config.Healthcheck
            }
        }'
        
    } > "${report_file}"
    
    log_success "Image optimization analysis completed: ${report_file}"
}

# Function to generate summary report
generate_summary_report() {
    local summary_file="${SCAN_REPORT_DIR}/security_summary_${TIMESTAMP}.md"
    
    log "Generating security summary report..."
    
    {
        echo "# Mimir Docker Security Scan Summary"
        echo "Generated at: $(date)"
        echo "Scan ID: ${TIMESTAMP}"
        echo
        
        echo "## Scanned Images"
        for image in "${PROJECT_NAME}" "${PROJECT_NAME}:api" "${PROJECT_NAME}:worker" "${PROJECT_NAME}:analytics"; do
            if docker images "${image}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${image}"; then
                echo "- ${image}"
            fi
        done
        echo
        
        echo "## Dockerfile Analysis (Hadolint)"
        if ls "${SCAN_REPORT_DIR}"/hadolint_${TIMESTAMP}_*.json >/dev/null 2>&1; then
            for file in "${SCAN_REPORT_DIR}"/hadolint_${TIMESTAMP}_*.json; do
                issues=$(jq length "${file}" 2>/dev/null || echo "0")
                dockerfile=$(basename "${file}" | sed 's/hadolint_.*_\(.*\)\.json/\1/')
                echo "- ${dockerfile}: ${issues} issues found"
            done
        else
            echo "- No Dockerfile scans performed"
        fi
        echo
        
        echo "## Vulnerability Scans (Trivy)"
        if ls "${SCAN_REPORT_DIR}"/trivy_${TIMESTAMP}_*.json >/dev/null 2>&1; then
            for file in "${SCAN_REPORT_DIR}"/trivy_${TIMESTAMP}_*.json; do
                image=$(basename "${file}" | sed 's/trivy_.*_\(.*\)\.json/\1/' | tr '_' '/')
                critical=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "${file}" 2>/dev/null | wc -l)
                high=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH") | .VulnerabilityID' "${file}" 2>/dev/null | wc -l)
                medium=$(jq '.Results[]?.Vulnerabilities[]? | select(.Severity == "MEDIUM") | .VulnerabilityID' "${file}" 2>/dev/null | wc -l)
                echo "- ${image}: ${critical} Critical, ${high} High, ${medium} Medium"
            done
        else
            echo "- No vulnerability scans performed"
        fi
        echo
        
        echo "## Recommendations"
        echo "1. Address all CRITICAL severity vulnerabilities immediately"
        echo "2. Review and fix HIGH severity vulnerabilities"
        echo "3. Regularly update base images and dependencies"
        echo "4. Follow Dockerfile best practices identified by Hadolint"
        echo "5. Implement regular security scanning in CI/CD pipeline"
        echo
        
        echo "## Report Files"
        find "${SCAN_REPORT_DIR}" -name "*${TIMESTAMP}*" -type f | sort | while read -r file; do
            echo "- $(basename "${file}")"
        done
        
    } > "${summary_file}"
    
    log_success "Security summary generated: ${summary_file}"
    echo
    cat "${summary_file}"
}

# Main execution
main() {
    log "Starting Docker Security Scan for Mimir"
    echo "======================================="
    
    # Install security tools
    install_security_tools
    
    # Scan Dockerfiles
    log "Phase 1: Dockerfile Analysis"
    scan_dockerfile "Dockerfile"
    scan_dockerfile "Dockerfile.api"
    scan_dockerfile "Dockerfile.worker" 
    scan_dockerfile "Dockerfile.analytics"
    
    # Scan images (if they exist)
    log "Phase 2: Image Vulnerability Scanning"
    
    # Build images if they don't exist
    if ! docker images "${PROJECT_NAME}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${PROJECT_NAME}:latest"; then
        log "Building main image for scanning..."
        docker build -t "${PROJECT_NAME}:latest" .
    fi
    
    # Scan main image
    scan_image_trivy "${PROJECT_NAME}:latest"
    check_image_optimization "${PROJECT_NAME}:latest"
    
    # Build and scan specialized images if Dockerfiles exist
    for service in "api" "worker" "analytics"; do
        if [ -f "Dockerfile.${service}" ]; then
            image_name="${PROJECT_NAME}:${service}"
            if ! docker images "${image_name}" --format "{{.Repository}}:{{.Tag}}" | grep -q "${image_name}"; then
                log "Building ${service} image for scanning..."
                docker build -f "Dockerfile.${service}" -t "${image_name}" .
            fi
            scan_image_trivy "${image_name}"
            check_image_optimization "${image_name}"
        fi
    done
    
    # Run Docker Bench Security
    log "Phase 3: Docker Environment Security"
    run_docker_bench
    
    # Generate summary
    log "Phase 4: Report Generation"
    generate_summary_report
    
    log_success "Docker security scan completed!"
    log "Report directory: ${SCAN_REPORT_DIR}"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi