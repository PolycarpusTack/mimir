# Mimir Architecture Documentation

This document provides a comprehensive overview of the Mimir News Scraper architecture, covering system design, component interactions, data flow, and architectural decisions.

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Patterns](#architecture-patterns)
3. [Component Architecture](#component-architecture)
4. [Data Architecture](#data-architecture)
5. [Infrastructure Architecture](#infrastructure-architecture)
6. [Security Architecture](#security-architecture)
7. [Scalability & Performance](#scalability--performance)
8. [Deployment Architecture](#deployment-architecture)

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          External Interfaces                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Web UI     │    REST API     │   GraphQL API   │   Webhooks   │   Metrics   │
│ (Port 5000) │   (Port 8000)   │   (Port 8001)   │ (Port 8002) │ (Port 9090) │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Application Layer                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web App   │  │  API Server │  │   Workers   │  │ AI Services │        │
│  │             │  │             │  │             │  │             │        │
│  │ • Dashboard │  │ • REST API  │  │ • Scraping  │  │ • Analysis  │        │
│  │ • Reports   │  │ • Auth      │  │ • Analytics │  │ • NLP       │        │
│  │ • Config    │  │ • GraphQL   │  │ • Cleanup   │  │ • ML        │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Service Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Scraper   │  │ Analytics   │  │    Auth     │  │   Search    │        │
│  │   Service   │  │   Service   │  │   Service   │  │   Service   │        │
│  │             │  │             │  │             │  │             │        │
│  │ • RSS Feed  │  │ • Metrics   │  │ • JWT       │  │ • Full-text │        │
│  │ • HTML      │  │ • Reports   │  │ • RBAC      │  │ • Semantic  │        │
│  │ • Dedup     │  │ • ML        │  │ • Sessions  │  │ • Vectors   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Data Layer                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │TimescaleDB  │  │    Redis    │  │ Vector DB   │        │
│  │             │  │             │  │             │  │             │        │
│  │ • Articles  │  │ • Metrics   │  │ • Cache     │  │ • Embeddings│        │
│  │ • Users     │  │ • Analytics │  │ • Sessions  │  │ • Similarity│        │
│  │ • Config    │  │ • Time-series│ │ • Queue     │  │ • Search    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Infrastructure Layer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Kubernetes  │  │   Docker    │  │   Network   │  │   Storage   │        │
│  │             │  │             │  │             │  │             │        │
│  │ • Pods      │  │ • Images    │  │ • Ingress   │  │ • PV/PVC    │        │
│  │ • Services  │  │ • Registry  │  │ • LB        │  │ • Backup    │        │
│  │ • Config    │  │ • Security  │  │ • Firewall  │  │ • Archive   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Microservices Architecture**: Loosely coupled, independently deployable services
2. **Event-Driven Design**: Asynchronous communication via events and message queues
3. **Cloud-Native**: Built for containerized, orchestrated environments
4. **Scalability**: Horizontal scaling with load balancing and auto-scaling
5. **Observability**: Comprehensive logging, metrics, and tracing
6. **Security**: Defense in depth with multiple security layers

## Architecture Patterns

### 1. Domain-Driven Design (DDD)

```
┌─────────────────────────────────────────────────────────────────┐
│                       Domain Model                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Content    │  │  Analytics   │  │    Users     │          │
│  │   Domain     │  │   Domain     │  │   Domain     │          │
│  │              │  │              │  │              │          │
│  │ • Articles   │  │ • Metrics    │  │ • Auth       │          │
│  │ • Sources    │  │ • Reports    │  │ • RBAC       │          │
│  │ • Keywords   │  │ • ML Models  │  │ • Sessions   │          │
│  │ • Categories │  │ • Forecasts  │  │ • Orgs       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Clean Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Controllers/Adapters                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Web API   │  │   GraphQL   │  │   Workers   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     Use Cases/Services                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Article    │  │  Analytics  │  │    User     │            │
│  │  Service    │  │   Service   │  │  Service    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Domain Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Entities   │  │ Value Obs   │  │  Domain     │            │
│  │             │  │             │  │  Services   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Repositories│  │   External  │  │   Database  │            │
│  │             │  │    APIs     │  │   Adapters  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Event-Driven Architecture

```
                    ┌─────────────────┐
                    │   Event Bus     │
                    │    (Redis)      │
                    └─────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │    │   Router    │    │  Consumer   │
│             │    │             │    │             │
│ • Scraper   │    │ • Filter    │    │ • Analytics │
│ • API       │    │ • Route     │    │ • ML        │
│ • User      │    │ • Transform │    │ • Notify    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Component Architecture

### 1. Web Application Component

```python
# Flask-based web application
┌─────────────────────────────────────────────────────────────────┐
│                        Web Application                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Templates  │  │   Routes    │  │   Static    │            │
│  │             │  │             │  │             │            │
│  │ • Dashboard │  │ • /api/*    │  │ • CSS       │            │
│  │ • Reports   │  │ • /admin/*  │  │ • JS        │            │
│  │ • Config    │  │ • /auth/*   │  │ • Images    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Middleware  │  │  Security   │  │    Forms    │            │
│  │             │  │             │  │             │            │
│  │ • CORS      │  │ • CSRF      │  │ • WTForms   │            │
│  │ • Auth      │  │ • Headers   │  │ • Validate  │            │
│  │ • Logging   │  │ • Rate Limit│  │ • Render    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. API Server Component

```python
# FastAPI-based REST API
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Routers   │  │ Middleware  │  │  Dependencies│           │
│  │             │  │             │  │             │            │
│  │ • Articles  │  │ • Auth      │  │ • Database  │            │
│  │ • Users     │  │ • CORS      │  │ • Cache     │            │
│  │ • Analytics │  │ • Rate Limit│  │ • Services  │            │
│  │ • Search    │  │ • Logging   │  │ • Validation│            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Models    │  │  Services   │  │    Utils    │            │
│  │             │  │             │  │             │            │
│  │ • Pydantic  │  │ • Business  │  │ • Helpers   │            │
│  │ • Schemas   │  │ • Logic     │  │ • Validators│            │
│  │ • Responses │  │ • External  │  │ • Formatters│            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Worker Component

```python
# Celery-based background workers
┌─────────────────────────────────────────────────────────────────┐
│                          Workers                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Scraper   │  │  Analytics  │  │   Cleanup   │            │
│  │   Worker    │  │   Worker    │  │   Worker    │            │
│  │             │  │             │  │             │            │
│  │ • RSS Feeds │  │ • ML Models │  │ • Old Data  │            │
│  │ • HTML      │  │ • Reports   │  │ • Logs      │            │
│  │ • Dedup     │  │ • Metrics   │  │ • Cache     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Queue     │  │  Scheduler  │  │  Monitor    │            │
│  │             │  │             │  │             │            │
│  │ • Redis     │  │ • Periodic  │  │ • Health    │            │
│  │ • Priority  │  │ • Cron      │  │ • Metrics   │            │
│  │ • Retry     │  │ • Triggers  │  │ • Alerts    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Architecture

### 1. Data Flow Diagram

```
External Sources        Application Layer           Data Layer
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ RSS Feeds   │───────▶│   Scraper   │───────▶│ PostgreSQL  │
│ Web Sites   │        │   Service   │        │  (Primary)  │
│ APIs        │        └─────────────┘        └─────────────┘
└─────────────┘               │                       │
                              │                       ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ User Input  │───────▶│     API     │───────▶│   Redis     │
│ Admin       │        │   Server    │        │  (Cache)    │
│ Dashboard   │        └─────────────┘        └─────────────┘
└─────────────┘               │                       │
                              │                       ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│ ML Models   │◀──────│ Analytics   │───────▶│ TimescaleDB │
│ AI Services │        │   Worker    │        │ (Analytics) │
│ Reports     │        └─────────────┘        └─────────────┘
└─────────────┘                                      │
                                                     ▼
                                            ┌─────────────┐
                                            │ Vector DB   │
                                            │(Embeddings) │
                                            └─────────────┘
```

### 2. Database Schema

#### PostgreSQL (Primary Database)

```sql
-- Core entities
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'viewer',
    password_hash VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id),
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'rss', 'html', 'api'
    config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    last_scraped TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id),
    source_id UUID REFERENCES sources(id),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    author VARCHAR(255),
    published_date TIMESTAMPTZ,
    scraped_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    embedding_vector vector(768), -- For semantic search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || COALESCE(content, ''))
    ) STORED
);

-- Analytics and ML
CREATE TABLE keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id),
    keyword VARCHAR(255) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE saved_searches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    query JSONB NOT NULL,
    alert_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### TimescaleDB (Analytics Database)

```sql
-- Time-series data for analytics
CREATE TABLE article_metrics (
    time TIMESTAMPTZ NOT NULL,
    organization_id UUID NOT NULL,
    source_id UUID,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    dimensions JSONB DEFAULT '{}'
);

SELECT create_hypertable('article_metrics', 'time');

CREATE TABLE user_activity (
    time TIMESTAMPTZ NOT NULL,
    organization_id UUID NOT NULL,
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('user_activity', 'time');
```

### 3. Caching Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Redis Cache                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Cache Key Pattern          │  TTL    │  Usage                 │
│ ──────────────────────────────────────────────────────────────  │
│  articles:list:{hash}       │  5m     │  Article listings      │
│  article:{id}              │  1h     │  Individual articles   │
│  user:session:{token}       │  24h    │  User sessions         │
│  search:results:{hash}      │  15m    │  Search results        │
│  analytics:report:{id}      │  30m    │  Analytics reports     │
│  ml:predictions:{model}     │  1h     │  ML model results      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Infrastructure Architecture

### 1. Kubernetes Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Master    │  │    Node 1   │  │    Node 2   │            │
│  │             │  │             │  │             │            │
│  │ • API Server│  │ • Web Pods  │  │ • API Pods  │            │
│  │ • etcd      │  │ • Workers   │  │ • Database  │            │
│  │ • Scheduler │  │ • Monitoring│  │ • Cache     │            │
│  │ • Controller│  │             │  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    Node 3   │  │  Ingress    │  │   Storage   │            │
│  │             │  │             │  │             │            │
│  │ • Analytics │  │ • NGINX     │  │ • PVs       │            │
│  │ • ML Models │  │ • Cert Mgr  │  │ • Backup    │            │
│  │ • Logs      │  │ • Rate Limit│  │ • Archive   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Network Architecture

```
Internet
    │
    ▼
┌─────────────┐
│Load Balancer│ (Cloud Provider LB)
└─────────────┘
    │
    ▼
┌─────────────┐
│   Ingress   │ (NGINX Ingress Controller)
│ Controller  │
└─────────────┘
    │
    ▼
┌─────────────┐
│  Services   │ (ClusterIP Services)
└─────────────┘
    │
    ▼
┌─────────────┐
│    Pods     │ (Application Instances)
└─────────────┘
```

### 3. Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Security Layers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Network Security                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Firewall   │  │ Network     │  │     TLS     │            │
│  │    Rules    │  │ Policies    │  │ Termination │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  Layer 2: Application Security                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    RBAC     │  │    JWT      │  │    CSRF     │            │
│  │             │  │    Auth     │  │ Protection  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  Layer 3: Container Security                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Pod       │  │  Security   │  │   Image     │            │
│  │ Security    │  │  Context    │  │  Scanning   │            │
│  │ Policies    │  │             │  │             │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  Layer 4: Data Security                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Encryption  │  │   Access    │  │   Backup    │            │
│  │ at Rest     │  │  Control    │  │ Encryption  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Scalability & Performance

### 1. Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────┐
│                      Auto-scaling Strategy                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Component         │  Min │  Max │  Trigger              │     │
│ ────────────────────────────────────────────────────────────── │
│  Web Application   │   2  │  10  │  CPU > 70%           │     │
│  API Server        │   3  │  20  │  CPU > 70%, RPS > 1K │     │
│  Worker Pods       │   2  │  50  │  Queue Length > 100  │     │
│  Database (Read)   │   1  │   5  │  Connection Pool 80% │     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Performance Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Strategies                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Database                                                       │
│  • Read replicas for scaling reads                            │
│  • Connection pooling (PgBouncer)                             │
│  • Query optimization with indexes                            │
│  • Partitioning for large tables                              │
│                                                                 │
│  Caching                                                        │
│  • Redis for application cache                                │
│  • CDN for static assets                                      │
│  • Application-level caching                                  │
│  • Database query result caching                              │
│                                                                 │
│  Application                                                    │
│  • Async processing with Celery                               │
│  • Database query optimization                                │
│  • Memory-efficient data processing                           │
│  • Resource pooling and reuse                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

### 1. Multi-Environment Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      Environment Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Development    │    Staging      │   Production               │
│ ──────────────────────────────────────────────────────────────  │
│  • Local dev    │  • Pre-prod     │  • Live system            │
│  • Feature work │  • Integration  │  • High availability      │
│  • Unit tests   │  • E2E tests    │  • Monitoring             │
│  • Hot reload   │  • Performance  │  • Disaster recovery      │
│                 │  • Security     │  • Compliance             │
│                                                                 │
│  Auto-deploy    │  Manual deploy  │  Approval required        │
│  from feature   │  from develop   │  from main branch         │
│  branches       │  branch         │                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. GitOps Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitOps Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Developer      │   Git Repo      │   ArgoCD        │  Cluster │
│ ──────────────────────────────────────────────────────────────  │
│                 │                 │                 │          │
│  1. Code        │  2. Push to     │  3. Detect      │  4. Apply│
│     Change  ────┼────▶ Repo   ────┼────▶ Change ────┼────▶ K8s │
│                 │                 │                 │          │
│  5. Monitor ◀───┼─── 6. Status ◀──┼─── 7. Health ◀──┼───── Pod │
│     Result      │                 │     Check       │    Status│
│                 │                 │                 │          │
└─────────────────────────────────────────────────────────────────┘
```

## Observability Architecture

### 1. Three Pillars of Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                       Observability                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Metrics   │  │    Logs     │  │   Traces    │            │
│  │             │  │             │  │             │            │
│  │ • Prometheus│  │ • ELK Stack │  │ • Jaeger    │            │
│  │ • Grafana   │  │ • Fluentd   │  │ • OpenTel   │            │
│  │ • AlertMgr  │  │ • Kibana    │  │ • Zipkin    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  What is broken? │ Why is it      │ Where is the              │
│  How much?       │ broken?        │ bottleneck?               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Monitoring Stack

```
Application Pods
      │
      ▼ (metrics)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Prometheus  │────│   Grafana   │    │ AlertMgr    │
│  (Scrape)   │    │ (Dashboard) │    │(Notify)     │
└─────────────┘    └─────────────┘    └─────────────┘
      │                                      │
      ▼ (logs)                               ▼ (alerts)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Elasticsearch│────│   Kibana    │    │   Slack     │
│  (Store)    │    │ (Search)    │    │ PagerDuty   │
└─────────────┘    └─────────────┘    └─────────────┘
      │
      ▼ (traces)
┌─────────────┐    ┌─────────────┐
│   Jaeger    │────│  Jaeger UI  │
│ (Collect)   │    │  (Analyze)  │
└─────────────┘    └─────────────┘
```

This architecture documentation provides a comprehensive overview of the Mimir system design, enabling developers and operators to understand the system's structure, make informed decisions about modifications, and effectively maintain and scale the application.