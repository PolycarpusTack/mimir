# Mimir Deployment Guide

This comprehensive guide covers deploying Mimir News Scraper in different environments using modern DevOps practices.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Environment Setup](#environment-setup)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [GitOps with ArgoCD](#gitops-with-argocd)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Configuration](#security-configuration)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

```bash
# Container & Orchestration
docker >= 20.10
kubectl >= 1.24
helm >= 3.8
kustomize >= 4.5

# CI/CD & GitOps
argocd >= 2.6
git >= 2.30

# Database & Monitoring
postgresql-client >= 14
redis-tools >= 6.2

# Optional but Recommended
k9s >= 0.27          # Kubernetes UI
kubectx >= 0.9       # Context switching
stern >= 1.20        # Log streaming
```

### Infrastructure Requirements

| Environment | CPU | Memory | Storage | Network |
|-------------|-----|--------|---------|---------|
| Development | 2 cores | 4 GB | 20 GB | 1 Gbps |
| Staging | 4 cores | 8 GB | 50 GB | 1 Gbps |
| Production | 8 cores | 16 GB | 200 GB | 10 Gbps |

### Kubernetes Cluster Requirements

- Kubernetes 1.24+ with RBAC enabled
- StorageClass for persistent volumes
- LoadBalancer or Ingress controller
- Metrics Server for autoscaling
- Network policies support (recommended)

## Quick Start

### 1. Local Development with Docker Compose

```bash
# Clone repository
git clone https://github.com/your-org/mimir.git
cd mimir

# Copy environment template
cp .env.example .env

# Edit configuration
vim .env  # Set required variables

# Start services
docker-compose up -d

# Verify deployment
curl http://localhost:5000/health
curl http://localhost:8000/api/health
```

### 2. Kubernetes Quick Deploy

```bash
# Apply base manifests
kubectl apply -k k8s/overlays/development

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s \
    deployment/mimir-web deployment/mimir-api -n mimir-development

# Access application
kubectl port-forward -n mimir-development svc/mimir-web 8080:80
```

## Environment Setup

### Development Environment

```bash
# Create namespace
kubectl create namespace mimir-development

# Deploy development stack
kubectl apply -k k8s/overlays/development

# Configure development secrets
kubectl create secret generic postgres-secret \
    --from-literal=database-url="postgresql://..." \
    -n mimir-development

kubectl create secret generic redis-secret \
    --from-literal=redis-url="redis://..." \
    -n mimir-development
```

### Staging Environment

```bash
# Create namespace
kubectl create namespace mimir-staging

# Deploy TimescaleDB for analytics
kubectl apply -k k8s/overlays/staging

# Verify deployment
kubectl get pods -n mimir-staging
kubectl logs -f deployment/mimir-api -n mimir-staging
```

### Production Environment

```bash
# Create namespace with labels
kubectl create namespace mimir-production
kubectl label namespace mimir-production \
    environment=production \
    monitoring=enabled \
    backup=enabled

# Deploy production stack
kubectl apply -k k8s/overlays/production

# Enable monitoring
kubectl apply -k k8s/monitoring

# Configure autoscaling
kubectl apply -f k8s/base/hpa.yaml
```

## Kubernetes Deployment

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ingress       │    │   Web Service   │    │  API Service    │
│   Controller    │────│   (Port 80)     │    │  (Port 8000)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Web Pods      │    │   API Pods      │
                       │   (Replicas: 3) │    │   (Replicas: 3) │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                       ┌─────────────────────────────────────────┐
                       │           Worker Pods                   │
                       │         (Replicas: 2)                  │
                       └─────────────────────────────────────────┘
                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   PostgreSQL    │    │     Redis       │
                       │   (Primary +    │    │   (Sentinel)    │
                       │    Standby)     │    │                 │
                       └─────────────────┘    └─────────────────┘
```

### Deployment Steps

#### 1. Prepare Configuration

```bash
# Create kustomization overlay
cd k8s/overlays/production

# Update image tags
kustomize edit set image mimir=ghcr.io/your-org/mimir:v1.2.3
kustomize edit set image mimir-api=ghcr.io/your-org/mimir-api:v1.2.3

# Generate manifests
kustomize build . > mimir-production.yaml

# Review before applying
kubectl apply --dry-run=client -f mimir-production.yaml
```

#### 2. Deploy Database Layer

```bash
# Deploy PostgreSQL with replication
kubectl apply -f k8s/base/postgres-ha.yaml

# Wait for primary to be ready
kubectl wait --for=condition=ready pod/postgres-primary-0 -n mimir-production

# Deploy Redis with Sentinel
kubectl apply -f k8s/base/redis-ha.yaml

# Verify database connectivity
kubectl exec -it postgres-primary-0 -n mimir-production -- pg_isready
```

#### 3. Deploy Application Layer

```bash
# Apply application manifests
kubectl apply -k k8s/overlays/production

# Monitor deployment progress
kubectl rollout status deployment/mimir-web -n mimir-production
kubectl rollout status deployment/mimir-api -n mimir-production
kubectl rollout status deployment/mimir-worker -n mimir-production

# Verify all pods are running
kubectl get pods -n mimir-production
```

#### 4. Configure Ingress

```bash
# Apply ingress configuration
kubectl apply -f k8s/base/ingress.yaml

# Get external IP
kubectl get ingress -n mimir-production

# Test external access
curl -H "Host: mimir.example.com" http://<EXTERNAL-IP>/health
```

### Resource Management

#### Horizontal Pod Autoscaler

```yaml
# k8s/base/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mimir-web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mimir-web
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Vertical Pod Autoscaler

```bash
# Enable VPA
kubectl apply -f k8s/base/vpa.yaml

# Monitor recommendations
kubectl describe vpa mimir-api-vpa -n mimir-production
```

## GitOps with ArgoCD

### ArgoCD Setup

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Get admin password
kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" | base64 -d

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

### Configure Applications

```bash
# Apply ArgoCD applications
kubectl apply -f k8s/argocd/application.yaml

# Sync applications
argocd app sync mimir-production
argocd app sync mimir-staging
argocd app sync mimir-development

# Monitor sync status
argocd app get mimir-production
```

### GitOps Workflow

1. **Make Changes**: Update manifests in git repository
2. **Commit & Push**: Changes are automatically detected by ArgoCD
3. **Auto Sync**: ArgoCD syncs changes to cluster (if enabled)
4. **Health Check**: ArgoCD monitors application health
5. **Rollback**: Automatic rollback on failure (if configured)

### Blue-Green Deployments

```bash
# Deploy new version to preview environment
kubectl apply -k k8s/overlays/production-preview

# Run automated tests
./scripts/testing/integration-test.sh production-preview

# Promote to production
kubectl argo rollouts promote mimir-web-rollout -n mimir-production

# Monitor rollout
kubectl argo rollouts get rollout mimir-web-rollout -n mimir-production --watch
```

## Monitoring & Observability

### Prometheus & Grafana

```bash
# Deploy monitoring stack
kubectl apply -k k8s/monitoring

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring

# Access Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
```

### Log Aggregation

```bash
# Deploy ELK stack
kubectl apply -k k8s/logging

# Access Kibana
kubectl port-forward svc/kibana 5601:5601 -n logging

# View logs
kubectl logs -f deployment/mimir-api -n mimir-production | jq .
```

### Distributed Tracing

```bash
# Deploy Jaeger
kubectl apply -k k8s/tracing

# Access Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686 -n tracing
```

### Custom Metrics

```python
# Application metrics example
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
ARTICLES_SCRAPED = Counter('mimir_articles_scraped_total', 
                          'Total articles scraped', ['source'])
SCRAPE_DURATION = Histogram('mimir_scrape_duration_seconds',
                           'Time spent scraping articles', ['source'])

# Use in application
ARTICLES_SCRAPED.labels(source='techcrunch').inc()
SCRAPE_DURATION.labels(source='techcrunch').observe(response_time)
```

## Security Configuration

### Network Policies

```yaml
# Restrict traffic between namespaces
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mimir-network-policy
spec:
  podSelector:
    matchLabels:
      app: mimir-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mimir-web
    ports:
    - protocol: TCP
      port: 8000
```

### Pod Security Standards

```yaml
# Pod Security Context
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: mimir-api
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### Secrets Management

```bash
# Create secrets from environment
kubectl create secret generic postgres-secret \
    --from-env-file=.env.production \
    -n mimir-production

# Use External Secrets Operator
kubectl apply -f k8s/security/external-secrets.yaml

# Rotate secrets
kubectl patch secret postgres-secret -p='{"data":{"password":"'$(echo -n $NEW_PASSWORD | base64)'"}}'
```

## Troubleshooting

### Common Issues

#### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n mimir-production

# Check logs
kubectl logs <pod-name> -n mimir-production --previous

# Check resource constraints
kubectl top pod <pod-name> -n mimir-production

# Check node resources
kubectl describe node <node-name>
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    python -c "
import psycopg2
import os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
print('Database connection successful')
"

# Check PostgreSQL logs
kubectl logs statefulset/postgres-primary -n mimir-production

# Verify service discovery
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    nslookup postgres
```

#### Performance Issues

```bash
# Check resource usage
kubectl top pods -n mimir-production

# Monitor API response times
kubectl exec -it deployment/mimir-api -n mimir-production -- \
    curl localhost:8000/metrics | grep http_request_duration

# Check database performance
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
```

### Debugging Commands

```bash
# Get all resources in namespace
kubectl get all -n mimir-production

# Describe deployment
kubectl describe deployment mimir-api -n mimir-production

# Get events
kubectl get events -n mimir-production --sort-by='.lastTimestamp'

# Port forward for debugging
kubectl port-forward deployment/mimir-api 8000:8000 -n mimir-production

# Execute commands in container
kubectl exec -it deployment/mimir-api -n mimir-production -- /bin/bash

# Copy files from container
kubectl cp mimir-production/mimir-api-xxx:/app/logs/app.log ./debug.log
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health
curl http://localhost:8000/ready

# Database health
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    pg_isready -U mimir_user

# Redis health
kubectl exec -it redis-0 -n mimir-production -- \
    redis-cli ping

# Cluster health
kubectl get componentstatuses
kubectl cluster-info
```

### Backup and Recovery

```bash
# Create database backup
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    pg_dump -U mimir_user mimir_news > backup.sql

# Restore from backup
kubectl cp backup.sql postgres-primary-0:/tmp/ -n mimir-production
kubectl exec -it postgres-primary-0 -n mimir-production -- \
    psql -U mimir_user mimir_news < /tmp/backup.sql

# Test disaster recovery
./scripts/testing/dr-test.sh full-test
```

## Best Practices

### Development Workflow

1. **Local Development**: Use Docker Compose for rapid iteration
2. **Testing**: Run unit tests before committing
3. **Integration**: Deploy to development cluster for integration testing
4. **Staging**: Full production simulation with production data volume
5. **Production**: GitOps deployment with automated monitoring

### Security Checklist

- [ ] All secrets stored in Kubernetes secrets or external secret management
- [ ] Network policies restrict inter-pod communication
- [ ] Pod security contexts prevent privilege escalation
- [ ] Images scanned for vulnerabilities
- [ ] RBAC configured with least privilege principle
- [ ] TLS enabled for all external communications

### Monitoring Checklist

- [ ] Application metrics exported to Prometheus
- [ ] Structured logging with correlation IDs
- [ ] Distributed tracing enabled
- [ ] Alert rules configured for critical scenarios
- [ ] Dashboards created for key business metrics
- [ ] SLIs and SLOs defined and monitored

### Performance Optimization

- [ ] Resource requests and limits configured
- [ ] Horizontal Pod Autoscaler enabled
- [ ] Database connection pooling configured
- [ ] Redis caching implemented
- [ ] CDN configured for static assets
- [ ] Database indexes optimized

## Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [GitOps Best Practices](https://www.weave.works/technologies/gitops/)
- [Security Benchmarks](https://www.cisecurity.org/benchmark/kubernetes)