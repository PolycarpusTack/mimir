# Mimir Proxy System - Complete Production Deployment Guide

## 🎯 Overview

This guide provides step-by-step instructions for deploying the Mimir proxy system in production environments with enterprise-grade reliability and monitoring.

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 2GB+ RAM
- 10GB+ disk space
- Stable internet connection
- Linux/Windows/macOS

### Required Dependencies
```bash
pip install requests playwright jsonschema
```

### Recommended Setup
- Dedicated server/VM for scraping operations
- Monitoring dashboard access
- Log aggregation system
- Backup and recovery procedures

## 🚀 Quick Start (5 Minutes)

### Option A: Interactive Assistant
```bash
python proxy_production_assistant.py
```
Follow the interactive prompts to configure your proxy providers and deploy.

### Option B: Manual Setup
```bash
# 1. Validate current configuration
python proxy_config_validator.py --report

# 2. Set up proxy providers
python proxy_provider_templates.py

# 3. Run end-to-end tests
python proxy_end_to_end_test.py

# 4. Deploy configuration
python setup_complete_proxy_system.py
```

## 📊 Deployment Steps

### Step 1: Environment Preparation

#### 1.1 Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Install Playwright browsers
python setup_playwright.py
```

#### 1.2 Create Directory Structure
```bash
mkdir -p logs/playwright_screenshots
mkdir -p logs/proxy_health
mkdir -p backups/configs
```

#### 1.3 Set File Permissions
```bash
chmod +x *.py
chmod 644 *.json
chmod 755 logs/
```

### Step 2: Proxy Provider Configuration

#### 2.1 Choose Your Provider

**Bright Data (Residential/Mobile)**
```bash
# Set environment variables
export BRIGHTDATA_USERNAME="your_username"
export BRIGHTDATA_PASSWORD="your_password"  
export BRIGHTDATA_ZONE="residential"

# Test configuration
python -c "from proxy_provider_templates import ProxyProviderManager; print(ProxyProviderManager().validate_provider_credentials('brightdata'))"
```

**Oxylabs (Residential/Datacenter)**
```bash
# Set environment variables
export OXYLABS_USERNAME="your_username"
export OXYLABS_PASSWORD="your_password"

# Test configuration
python -c "from proxy_provider_templates import ProxyProviderManager; print(ProxyProviderManager().validate_provider_credentials('oxylabs'))"
```

**SmartProxy (Residential)**
```bash
# Set environment variables
export SMARTPROXY_USERNAME="your_username"
export SMARTPROXY_PASSWORD="your_password"

# Test configuration
python -c "from proxy_provider_templates import ProxyProviderManager; print(ProxyProviderManager().validate_provider_credentials('smartproxy'))"
```

#### 2.2 Generate Provider Configuration
```python
from proxy_provider_templates import ProxyProviderManager

manager = ProxyProviderManager()

# Validate all providers
results = manager.validate_all_providers()
print("Provider validation:", results)

# Generate config for working providers
working_providers = [name for name, valid in results.items() if valid]
config = manager.create_production_config(working_providers)

# Update main configuration
import json
with open('config.json', 'r') as f:
    current_config = json.load(f)

current_config.update(config)

with open('config.json', 'w') as f:
    json.dump(current_config, f, indent=4)
```

### Step 3: Configuration Validation

#### 3.1 Run Comprehensive Validation
```bash
python proxy_config_validator.py --report
```

#### 3.2 Fix Common Issues
```bash
# Auto-fix configuration issues
python proxy_config_validator.py --auto-fix

# Re-validate after fixes
python proxy_config_validator.py --report
```

#### 3.3 Verify File Structure
```bash
# Check required files exist
ls -la config.json
ls -la proxy_list.json
ls -la playwright_config.json

# Validate JSON syntax
python -m json.tool config.json > /dev/null && echo "✅ config.json valid"
python -m json.tool proxy_list.json > /dev/null && echo "✅ proxy_list.json valid"
```

### Step 4: System Testing

#### 4.1 Component Tests
```bash
# Test individual components
python test_proxy_system.py
python test_complete_proxy_integration.py
```

#### 4.2 End-to-End Testing
```bash
# Full system test
python proxy_end_to_end_test.py

# Load testing (optional)
python proxy_end_to_end_test.py --load-test --duration 120 --concurrent 10
```

#### 4.3 Integration Testing
```bash
# Test with main scraper
python scraper.py --run

# Monitor proxy usage
python proxy_dashboard.py &
# Access dashboard at http://localhost:5001/proxy
```

### Step 5: Production Deployment

#### 5.1 Enable Proxy System
Edit `config.json`:
```json
{
  "proxy_settings": {
    "enabled": true,
    "rotation_strategy": "health_based"
  }
}
```

#### 5.2 Configure Monitoring
```bash
# Start proxy dashboard
python proxy_dashboard.py &

# Set up log monitoring
tail -f logs/mimir_scraper.log | grep -i proxy
```

#### 5.3 Set Up Scheduled Scraping
```bash
# Add to crontab for regular scraping
crontab -e

# Add line for every 4 hours:
0 */4 * * * cd /path/to/mimir && python scraper.py --run >> logs/cron.log 2>&1
```

## 🔧 Configuration Templates

### Small-Scale Deployment (1-10 requests/minute)
```json
{
  "proxy_settings": {
    "enabled": true,
    "rotation_strategy": "round_robin",
    "health_check_interval": 600,
    "failure_threshold": 5,
    "proxy_providers": [
      {
        "type": "static_list",
        "name": "primary_proxies",
        "enabled": true,
        "proxies": ["http://proxy1:8080", "http://proxy2:8080"]
      }
    ]
  }
}
```

### Medium-Scale Deployment (10-100 requests/minute)
```json
{
  "proxy_settings": {
    "enabled": true,
    "rotation_strategy": "health_based",
    "health_check_interval": 300,
    "failure_threshold": 3,
    "proxy_providers": [
      {
        "type": "api",
        "name": "brightdata_residential",
        "enabled": true
      }
    ]
  }
}
```

### Enterprise-Scale Deployment (100+ requests/minute)
```json
{
  "proxy_settings": {
    "enabled": true,
    "rotation_strategy": "health_based",
    "health_check_interval": 120,
    "failure_threshold": 2,
    "domain_specific": {
      "enabled": true,
      "assignments": {
        "news.site1.com": ["proxy_pool_1"],
        "feeds.site2.com": ["proxy_pool_2"]
      }
    }
  }
}
```

## 📊 Monitoring and Maintenance

### Health Monitoring
```bash
# Check proxy system status
python -c "from scraper_proxy_integration import get_proxy_integration_status; print(get_proxy_integration_status())"

# View proxy statistics
python -c "from proxy_integration import get_proxy_statistics; from config_loader import config_loader; print(get_proxy_statistics(config_loader.load_config()))"

# Trigger health check
python -c "from proxy_integration import health_check_all_proxies; from config_loader import config_loader; health_check_all_proxies(config_loader.load_config())"
```

### Performance Monitoring
```bash
# Monitor proxy performance during scraping
python scraper.py --run & 
PID=$!
sleep 30
python -c "from scraper_proxy_integration import log_proxy_performance_summary; log_proxy_performance_summary()"
wait $PID
```

### Log Analysis
```bash
# Analyze proxy performance logs
grep "Proxy Performance Summary" logs/mimir_scraper.log | tail -5

# Check for proxy errors
grep -i "proxy.*error" logs/mimir_scraper.log | tail -10

# Monitor success rates
grep "success_rate" logs/mimir_scraper.log | tail -5
```

## 🚨 Troubleshooting

### Common Issues

#### Issue: "No proxy available"
**Cause**: All proxies blacklisted or credentials invalid
**Solution**:
```bash
# Check proxy status
python proxy_dashboard.py

# Reset blacklisted proxies
python -c "from scraper_proxy_integration import get_scraper_proxy_manager; mgr=get_scraper_proxy_manager(); [mgr.reset_proxy(url) for url in list(mgr.proxy_manager.blacklisted_proxies)]"

# Validate credentials
python proxy_provider_templates.py
```

#### Issue: High failure rate
**Cause**: Poor proxy quality or rate limiting
**Solution**:
```bash
# Increase failure threshold
# Edit config.json: "failure_threshold": 5

# Reduce request rate
# Edit config.json: "default_request_delay_seconds": 5

# Check rate limits
python proxy_dashboard.py  # Review success rates
```

#### Issue: Configuration errors
**Cause**: Invalid JSON or missing required fields
**Solution**:
```bash
# Validate and auto-fix
python proxy_config_validator.py --auto-fix

# Generate fresh configuration
python setup_complete_proxy_system.py
```

### Emergency Procedures

#### Disable Proxy System
```bash
# Quick disable via environment variable
export MIMIR_DISABLE_PROXY=true
python scraper.py --run

# Or edit config.json
sed -i 's/"enabled": true/"enabled": false/' config.json
```

#### Rollback Configuration
```bash
# Restore from backup
cp config.json.backup_* config.json

# Or regenerate default config
python setup_complete_proxy_system.py
```

#### Reset Proxy State
```bash
# Clear proxy state
rm -f proxy_state.json

# Reset all proxy metrics
python -c "from scraper_proxy_integration import reset_global_proxy_manager; reset_global_proxy_manager()"
```

## 🔒 Security Best Practices

### Credential Management
```bash
# Use environment variables for sensitive data
echo "export PROXY_USERNAME='your_username'" >> ~/.bashrc
echo "export PROXY_PASSWORD='your_password'" >> ~/.bashrc
source ~/.bashrc

# Never commit credentials to version control
echo "*.env*" >> .gitignore
echo "proxy_credentials.json" >> .gitignore
```

### Network Security
```bash
# Use HTTPS proxies when possible
# Configure firewall rules for proxy access
# Monitor proxy usage for anomalies
```

### Access Control
```bash
# Restrict dashboard access
# Set up authentication for monitoring endpoints
# Use VPN for remote access
```

## 📈 Performance Optimization

### Proxy Pool Optimization
```bash
# Use health-based rotation
"rotation_strategy": "health_based"

# Adjust health check frequency
"health_check_interval": 180  # 3 minutes

# Configure failure thresholds
"failure_threshold": 2  # Quick blacklisting
```

### Request Optimization
```bash
# Optimize request delays
"default_request_delay_seconds": 1  # Faster for good proxies

# Use domain-specific assignments
"domain_specific": {
  "enabled": true,
  "assignments": {
    "fast-site.com": ["fast_proxy_pool"],
    "slow-site.com": ["stable_proxy_pool"]
  }
}
```

### Resource Management
```bash
# Monitor memory usage
ps aux | grep python | grep scraper

# Monitor disk usage
du -sh logs/

# Clean old logs
find logs/ -name "*.log.*" -mtime +7 -delete
```

## 🔄 Backup and Recovery

### Configuration Backup
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/configs/$DATE
cp config.json backups/configs/$DATE/
cp proxy_list.json backups/configs/$DATE/
cp playwright_config.json backups/configs/$DATE/
echo "Backup created: backups/configs/$DATE"
```

### State Backup
```bash
# Backup proxy state
cp proxy_state.json backups/proxy_state_$(date +%Y%m%d_%H%M%S).json

# Backup logs
tar czf backups/logs_$(date +%Y%m%d_%H%M%S).tar.gz logs/
```

### Recovery Procedures
```bash
# Restore configuration
cp backups/configs/YYYYMMDD_HHMMSS/config.json .

# Restore proxy state
cp backups/proxy_state_YYYYMMDD_HHMMSS.json proxy_state.json

# Restart services
python scraper.py --run
```

## 📞 Support and Resources

### Documentation
- **Configuration Reference**: proxy_config_validator.py --help
- **Provider Setup**: proxy_provider_templates.py --help
- **Testing Guide**: proxy_end_to_end_test.py --help

### Debugging Tools
- **Configuration Validator**: `python proxy_config_validator.py --report`
- **End-to-End Testing**: `python proxy_end_to_end_test.py`
- **Interactive Dashboard**: `python proxy_dashboard.py`

### Community Resources
- GitHub Issues: [Report bugs and request features]
- Documentation: [Comprehensive guides and tutorials]
- Examples: [Real-world configuration examples]

## ✅ Production Checklist

Before going live, ensure all items are completed:

### Pre-Deployment
- [ ] All dependencies installed
- [ ] Proxy providers configured and tested
- [ ] Configuration validated with no critical issues
- [ ] End-to-end tests passing
- [ ] Monitoring dashboard accessible
- [ ] Backup procedures in place

### Security
- [ ] Credentials stored securely (environment variables)
- [ ] No sensitive data in version control
- [ ] Network access properly configured
- [ ] Dashboard access secured

### Performance
- [ ] Load testing completed successfully
- [ ] Resource usage within acceptable limits
- [ ] Proxy rotation working correctly
- [ ] Health monitoring active

### Monitoring
- [ ] Log aggregation configured
- [ ] Alert thresholds set
- [ ] Performance baselines established
- [ ] Escalation procedures documented

### Documentation
- [ ] Configuration documented
- [ ] Runbooks created
- [ ] Contact information updated
- [ ] Recovery procedures tested

---

## 🎉 You're Ready for Production!

Once you've completed this guide, your Mimir proxy system will be production-ready with:

- ✅ **Enterprise-grade reliability** with comprehensive error handling
- ✅ **Real-time monitoring** and health management
- ✅ **Scalable architecture** supporting multiple proxy providers
- ✅ **Complete observability** with dashboards and metrics
- ✅ **Security best practices** for credential management
- ✅ **Automated testing** and validation workflows

For ongoing support and optimization, refer to the monitoring sections and use the provided diagnostic tools.

**Happy scraping! 🕷️**