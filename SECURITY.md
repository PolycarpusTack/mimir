# Mimir Security Guide

This document outlines the security measures implemented in Mimir and provides guidelines for secure deployment and operation.

## 🔐 Security Features Implemented

### 1. Secrets Management

**Environment Variables**
- All sensitive credentials moved to environment variables
- No hardcoded passwords, API keys, or secrets in code
- `.env.example` file provided for configuration reference

**Configuration**
```bash
# Required environment variables
POSTGRES_PASSWORD=your_strong_password
JWT_SECRET_KEY=your_jwt_secret_minimum_32_characters
SESSION_SECRET_KEY=your_session_secret_key
SMTP_PASSWORD=your_smtp_app_password
```

### 2. SQL Injection Prevention

**Secure Query Builder**
- Custom `SecureQueryBuilder` class prevents SQL injection
- All queries use parameterized statements
- Input validation for table/column names
- Whitelist-based identifier validation

**Usage Example**
```python
from db_security import get_secure_executor

# Secure SELECT query
articles = secure_db.execute_select(
    table="articles",
    columns=["id", "title", "url"],
    where_conditions={"source": "TechCrunch"},
    limit=10
)
```

### 3. Web Application Security

**CSRF Protection**
- Flask-WTF CSRF protection enabled
- CSRF tokens for all state-changing operations
- Endpoint: `/api/csrf-token` for token retrieval

**Security Headers**
```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'; ...
```

**CORS Configuration**
- Specific origin allowlist (no wildcard `*`)
- Credentials support for authenticated requests
- Configurable via `CORS_ORIGINS` environment variable

### 4. Database Security

**Connection Security**
- Connection pooling with psycopg2
- SSL connections supported (configure via `POSTGRES_SSLMODE`)
- Connection string validation
- Pool size limits to prevent resource exhaustion

**Access Control**
- Database user with minimum required privileges
- Separate users for different application components
- Regular credential rotation recommended

### 5. API Security

**Authentication**
- JWT-based authentication for users
- API key authentication for service access
- Role-based access control (RBAC)
- Session management with secure tokens

**Rate Limiting**
- Per-endpoint rate limiting (to be implemented)
- IP-based throttling for abuse prevention
- Request size limits

## 🛡️ Security Best Practices

### Deployment Security

**Environment Configuration**
```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
POSTGRES_SSLMODE=require
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
```

**Docker Security**
```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Minimal base image
FROM python:3.11-slim-bullseye

# Health checks
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1
```

**Kubernetes Security**
```yaml
# Security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

### Database Security

**PostgreSQL Configuration**
```sql
-- Create dedicated user with limited privileges
CREATE USER mimir_app WITH PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE mimir_news TO mimir_app;
GRANT USAGE ON SCHEMA public TO mimir_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mimir_app;

-- Enable SSL
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
```

**Connection Security**
```python
# SSL-enabled connection string
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require
```

### Application Security

**Input Validation**
```python
from wtforms import validators

class ArticleSearchForm(FlaskForm):
    query = StringField('Search', [
        validators.Length(min=1, max=200),
        validators.Regexp(r'^[a-zA-Z0-9\s\-_]+$')
    ])
    category = SelectField('Category', choices=ALLOWED_CATEGORIES)
```

**Output Encoding**
```python
from markupsafe import escape

# Always escape user input in templates
safe_title = escape(article.title)
```

## 🔍 Security Monitoring

### Logging and Alerting

**Security Events to Log**
- Authentication failures
- API rate limit violations
- SQL injection attempts
- Unauthorized access attempts
- Configuration changes

**Log Format**
```json
{
  "timestamp": "2024-01-19T10:30:00Z",
  "level": "WARNING",
  "event": "auth_failure",
  "user_id": null,
  "ip_address": "192.168.1.100",
  "user_agent": "curl/7.68.0",
  "details": {
    "reason": "invalid_credentials",
    "email": "admin@example.com"
  }
}
```

### Security Scanning

**Regular Scans**
```bash
# Dependency vulnerability scanning
pip-audit

# Code security analysis
bandit -r . -f json -o security_report.json

# Container security scanning
docker scan mimir:latest
```

## 🚨 Incident Response

### Security Incident Checklist

1. **Immediate Response**
   - [ ] Isolate affected systems
   - [ ] Preserve evidence (logs, memory dumps)
   - [ ] Assess scope of compromise
   - [ ] Notify stakeholders

2. **Investigation**
   - [ ] Analyze logs for attack vectors
   - [ ] Check for data exfiltration
   - [ ] Document timeline of events
   - [ ] Identify root cause

3. **Recovery**
   - [ ] Patch vulnerabilities
   - [ ] Reset compromised credentials
   - [ ] Update security configurations
   - [ ] Monitor for reoccurrence

4. **Post-Incident**
   - [ ] Conduct post-mortem review
   - [ ] Update security procedures
   - [ ] Implement additional controls
   - [ ] Report to relevant authorities

### Emergency Contacts

- **Security Team**: security@company.com
- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Management**: management@company.com

## 📋 Security Compliance

### Data Protection

**GDPR Compliance**
- Data encryption at rest and in transit
- Right to erasure implementation
- Data processing consent tracking
- Privacy impact assessments

**Data Retention**
```python
# Automatic data cleanup
def cleanup_old_data():
    cutoff_date = datetime.now() - timedelta(days=365)
    delete_articles_older_than(cutoff_date)
    delete_logs_older_than(cutoff_date)
```

### Regular Security Tasks

**Daily**
- [ ] Monitor security logs
- [ ] Check system health
- [ ] Review failed login attempts

**Weekly**
- [ ] Update dependencies
- [ ] Review access logs
- [ ] Backup verification

**Monthly**
- [ ] Security scan (dependencies)
- [ ] Access review (users/permissions)
- [ ] Penetration testing

**Quarterly**
- [ ] Security policy review
- [ ] Incident response drill
- [ ] Third-party security assessment

## 🔧 Security Tools Integration

### Static Analysis
```yaml
# .github/workflows/security.yml
- name: Security Scan
  uses: github/super-linter@v4
  env:
    DEFAULT_BRANCH: main
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    LINTER_RULES_PATH: .github/linters
```

### Dynamic Analysis
```bash
# OWASP ZAP scanning
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:5000 \
  -J zap-report.json
```

### Dependency Monitoring
```bash
# Automated security updates
pip-audit --fix --dry-run
safety check --json --output safety-report.json
```

## 📞 Security Support

For security issues or questions:
- Create a security issue in the repository (for non-critical issues)
- Email security@company.com for sensitive matters
- Follow responsible disclosure for vulnerabilities

**Vulnerability Reporting**
Please include:
- Detailed description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if known)

Response time: We aim to respond to security reports within 24 hours and provide a fix timeline within 72 hours.