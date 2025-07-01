# Mimir Disaster Recovery Procedures

This directory contains disaster recovery runbooks, procedures, and automation scripts for the Mimir application.

## Overview

Our disaster recovery strategy includes:
- **RTO (Recovery Time Objective)**: 30 minutes
- **RPO (Recovery Point Objective)**: 1 hour
- **Multi-region failover capabilities**
- **Automated backup and restore procedures**
- **Regular DR testing and validation**

## Architecture

```
Primary Region (us-east-1)          DR Region (us-west-2)
┌─────────────────┐                ┌─────────────────┐
│   Kubernetes    │                │   Kubernetes    │
│    Cluster      │                │    Cluster      │
│                 │                │   (Standby)     │
├─────────────────┤                ├─────────────────┤
│  PostgreSQL     │◄──────────────►│  PostgreSQL     │
│   Primary       │  Replication   │   Standby       │
├─────────────────┤                ├─────────────────┤
│     Redis       │                │     Redis       │
│   Sentinel      │                │   Sentinel      │
├─────────────────┤                ├─────────────────┤
│   S3 Bucket     │◄──────────────►│   S3 Bucket     │
│   (Primary)     │  Cross-Region  │     (DR)        │
└─────────────────┘  Replication   └─────────────────┘
```

## Disaster Scenarios

### 1. Database Failure
- **Single Node Failure**: Automatic failover via streaming replication
- **Complete Database Loss**: Restore from S3 backups
- **Data Corruption**: Point-in-time recovery using WAL archives

### 2. Redis Cache Failure
- **Master Failure**: Automatic failover via Redis Sentinel
- **Complete Cache Loss**: Rebuild from database (degraded performance)

### 3. Application Service Failure
- **Pod Failure**: Kubernetes automatically restarts
- **Node Failure**: Pods rescheduled to healthy nodes
- **Zone Failure**: Multi-zone deployment ensures availability

### 4. Complete Region Failure
- **Failover to DR Region**: Manual process (see runbooks/)
- **DNS Update**: Route53 health checks and failover
- **Data Sync**: Ensure replication lag is minimal

## Quick Reference

| Component | Primary Endpoint | DR Endpoint | Failover Time |
|-----------|-----------------|-------------|---------------|
| API | api.mimir.example.com | api-dr.mimir.example.com | < 5 min |
| Database | postgres-primary-service | postgres-dr-primary-service | < 10 min |
| Redis | redis-master-service | redis-dr-master-service | < 2 min |

## Directory Structure

```
disaster-recovery/
├── README.md                    # This file
├── runbooks/                    # Step-by-step procedures
│   ├── database-failover.md
│   ├── redis-failover.md
│   ├── region-failover.md
│   └── backup-restore.md
├── scripts/                     # Automation scripts
│   ├── failover-database.sh
│   ├── failover-redis.sh
│   ├── failover-region.sh
│   ├── validate-dr.sh
│   └── test-restore.sh
├── monitoring/                  # DR monitoring configs
│   ├── alerts.yaml
│   └── dashboards/
└── tests/                       # Chaos engineering tests
    ├── zone-failure.yaml
    ├── network-partition.yaml
    └── resource-exhaustion.yaml
```

## Regular Testing Schedule

- **Weekly**: Automated backup validation
- **Monthly**: Component failover testing
- **Quarterly**: Full region failover drill
- **Annually**: Complete DR simulation with stakeholders

## Contact Information

- **On-Call Engineer**: Via PagerDuty
- **DR Coordinator**: dr-team@example.com
- **Escalation**: management@example.com

## Related Documentation

- [Backup and Restore Procedures](runbooks/backup-restore.md)
- [Monitoring and Alerting](../monitoring/README.md)
- [Security Incident Response](../security/incident-response.md)