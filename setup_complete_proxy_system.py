#!/usr/bin/env python
"""
Complete Proxy System Setup and Validation for Mimir
Sets up proxy system, validates configuration, and runs integration tests
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxySystemSetup:
    """Complete proxy system setup and validation"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.config_file = self.project_root / "config.json"
        self.setup_results = []
    
    def validate_prerequisites(self) -> bool:
        """Validate that all required files and dependencies exist"""
        logger.info("🔍 Validating prerequisites...")
        
        # Check required files
        required_files = [
            "config.json",
            "proxy_manager.py",
            "proxy_integration.py", 
            "scraper_proxy_integration.py",
            "scraper.py",
            "playwright_integration.py"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ Missing required files: {missing_files}")
            self.setup_results.append(("Prerequisites", False, f"Missing files: {missing_files}"))
            return False
        
        # Check Python dependencies
        try:
            import requests
            import playwright
        except ImportError as e:
            logger.error(f"❌ Missing Python dependencies: {e}")
            self.setup_results.append(("Prerequisites", False, f"Missing dependencies: {e}"))
            return False
        
        logger.info("✅ All prerequisites validated")
        self.setup_results.append(("Prerequisites", True, "All files and dependencies present"))
        return True
    
    def backup_current_config(self) -> bool:
        """Backup current configuration"""
        logger.info("💾 Backing up current configuration...")
        
        try:
            if self.config_file.exists():
                backup_file = self.project_root / f"config.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                with open(self.config_file, 'r') as src:
                    config_data = src.read()
                
                with open(backup_file, 'w') as dst:
                    dst.write(config_data)
                
                logger.info(f"✅ Configuration backed up to: {backup_file}")
                self.setup_results.append(("Config Backup", True, f"Backup created: {backup_file}"))
                return True
            else:
                logger.warning("⚠️ No existing config.json found")
                self.setup_results.append(("Config Backup", True, "No existing config to backup"))
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to backup configuration: {e}")
            self.setup_results.append(("Config Backup", False, f"Backup failed: {e}"))
            return False
    
    def create_sample_proxy_configuration(self) -> Dict:
        """Create sample proxy configuration for testing"""
        
        # Create sample proxy list file
        proxy_list_file = self.project_root / "sample_proxy_list.json"
        sample_proxy_data = {
            "description": "Sample proxy list for testing - replace with real proxies",
            "proxies": [
                {
                    "url": "http://proxy1.example.com:8080",
                    "location": "US-East",
                    "tags": ["fast", "reliable"],
                    "note": "Replace with real proxy URL"
                },
                {
                    "url": "http://proxy2.example.com:8080",
                    "location": "EU-West", 
                    "tags": ["secure"],
                    "note": "Replace with real proxy URL"
                }
            ]
        }
        
        try:
            with open(proxy_list_file, 'w') as f:
                json.dump(sample_proxy_data, f, indent=2)
            logger.info(f"📝 Created sample proxy list: {proxy_list_file}")
        except Exception as e:
            logger.warning(f"⚠️ Could not create sample proxy list: {e}")
        
        # Return enhanced proxy configuration
        return {
            "enabled": False,  # Disabled by default for safety
            "rotation_strategy": "health_based",
            "health_check_interval": 300,
            "failure_threshold": 3,
            "health_check_timeout": 10,
            "health_check_url": "http://httpbin.org/ip",
            "save_state_file": "proxy_state.json",
            "auto_save_interval": 600,
            "proxy_providers": [
                {
                    "type": "static_list",
                    "name": "primary_proxies",
                    "enabled": False,
                    "description": "Primary proxy pool",
                    "proxies": [
                        "http://proxy1.example.com:8080",
                        "http://proxy2.example.com:8080"
                    ]
                },
                {
                    "type": "file",
                    "name": "file_proxies",
                    "enabled": False,
                    "description": "Proxies loaded from file",
                    "file_path": "sample_proxy_list.json"
                },
                {
                    "type": "api",
                    "name": "api_proxies",
                    "enabled": False,
                    "description": "Proxies from API endpoint",
                    "api_url": "https://api.example.com/proxies",
                    "headers": {
                        "Authorization": "Bearer YOUR_API_TOKEN"
                    }
                }
            ],
            "domain_specific": {
                "enabled": False,
                "description": "Assign specific proxies to specific domains",
                "assignments": {
                    "news.example.com": ["proxy1.example.com:8080"],
                    "feeds.example.com": ["proxy2.example.com:8080"]
                }
            },
            "tenant_isolation": {
                "enabled": false,
                "description": "Separate proxy pools for different tenants/customers",
                "tenant_proxy_pools": {
                    "customer1": ["proxy1.example.com:8080"],
                    "customer2": ["proxy2.example.com:8080"]
                }
            }
        }
    
    def update_configuration_file(self) -> bool:
        """Update config.json with enhanced proxy settings"""
        logger.info("⚙️ Updating configuration file...")
        
        try:
            # Load existing configuration
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Add or update proxy settings
            proxy_config = self.create_sample_proxy_configuration()
            
            if "proxy_settings" in config:
                logger.info("ℹ️ Proxy settings already exist, merging with new settings...")
                
                # Preserve existing enabled status and any custom settings
                existing_enabled = config["proxy_settings"].get("enabled", False)
                proxy_config["enabled"] = existing_enabled
                
                # Merge existing custom settings
                for key, value in config["proxy_settings"].items():
                    if key not in proxy_config:
                        proxy_config[key] = value
            
            config["proxy_settings"] = proxy_config
            
            # Write updated configuration
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info("✅ Configuration file updated with proxy settings")
            self.setup_results.append(("Config Update", True, "Proxy settings added to config.json"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update configuration: {e}")
            self.setup_results.append(("Config Update", False, f"Update failed: {e}"))
            return False
    
    def validate_configuration(self) -> bool:
        """Validate the updated configuration"""
        logger.info("🔧 Validating proxy configuration...")
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Test configuration validation
            from scraper_proxy_integration import ScraperProxyManager
            
            # Test with current configuration
            proxy_manager = ScraperProxyManager(config)
            
            if proxy_manager.initialization_error:
                logger.warning(f"⚠️ Configuration validation warning: {proxy_manager.initialization_error}")
                self.setup_results.append(("Config Validation", True, f"Warning: {proxy_manager.initialization_error}"))
            else:
                logger.info("✅ Configuration validation passed")
                self.setup_results.append(("Config Validation", True, "Configuration is valid"))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            self.setup_results.append(("Config Validation", False, f"Validation failed: {e}"))
            return False
    
    def run_integration_tests(self) -> bool:
        """Run integration tests to verify system works"""
        logger.info("🧪 Running integration tests...")
        
        try:
            from test_complete_proxy_integration import CompleteProxyIntegrationTester
            
            tester = CompleteProxyIntegrationTester()
            results = tester.run_all_integration_tests()
            
            if results["failed"] == 0:
                logger.info("✅ All integration tests passed")
                self.setup_results.append(("Integration Tests", True, f"All {results['total_tests']} tests passed"))
                return True
            else:
                logger.warning(f"⚠️ {results['failed']} integration tests failed")
                self.setup_results.append(("Integration Tests", False, f"{results['failed']}/{results['total_tests']} tests failed"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            self.setup_results.append(("Integration Tests", False, f"Test execution failed: {e}"))
            return False
    
    def create_usage_documentation(self) -> bool:
        """Create comprehensive usage documentation"""
        logger.info("📚 Creating usage documentation...")
        
        try:
            docs_content = """# Mimir Proxy System Usage Guide

## Overview
The Mimir proxy system provides enterprise-grade proxy rotation, health monitoring, and domain-specific proxy assignment for web scraping operations.

## Configuration

### 1. Enable Proxy System
Edit `config.json` and set:
```json
{
  "proxy_settings": {
    "enabled": true
  }
}
```

### 2. Configure Proxy Providers

#### Static List
```json
{
  "proxy_providers": [
    {
      "type": "static_list",
      "name": "primary_proxies",
      "enabled": true,
      "proxies": [
        "http://proxy1.example.com:8080",
        "http://proxy2.example.com:8080"
      ]
    }
  ]
}
```

#### File-based
```json
{
  "proxy_providers": [
    {
      "type": "file",
      "name": "file_proxies", 
      "enabled": true,
      "file_path": "my_proxy_list.json"
    }
  ]
}
```

#### API-based
```json
{
  "proxy_providers": [
    {
      "type": "api",
      "name": "api_proxies",
      "enabled": true,
      "api_url": "https://api.proxyservice.com/proxies",
      "headers": {
        "Authorization": "Bearer YOUR_API_TOKEN"
      }
    }
  ]
}
```

### 3. Domain-Specific Assignment
```json
{
  "domain_specific": {
    "enabled": true,
    "assignments": {
      "news.site1.com": ["proxy1.example.com:8080"],
      "feeds.site2.com": ["proxy2.example.com:8080"]
    }
  }
}
```

## Usage

### 1. Normal Scraping
```bash
python scraper.py --run
```
The scraper automatically uses proxies if enabled.

### 2. Monitor Proxy Health
```bash
python proxy_dashboard.py
```
Access dashboard at http://localhost:5001/proxy

### 3. Test Proxy System
```bash
python test_complete_proxy_integration.py
```

### 4. Check Status Programmatically
```python
from scraper_proxy_integration import get_proxy_integration_status
status = get_proxy_integration_status()
print(status)
```

## Monitoring

### Proxy Dashboard
- Real-time proxy health monitoring
- Success rates and response times
- Manual proxy reset and health checks
- Auto-refresh every 30 seconds

### Log Monitoring
The scraper logs proxy performance summaries:
```
INFO - Proxy Performance Summary: {
  'proxy_enabled': True,
  'total_requests': 150,
  'proxy_usage_rate': 95.3,
  'active_proxies': 4,
  'overall_success_rate': 87.2
}
```

## Troubleshooting

### Common Issues

1. **No proxies available**
   - Check proxy configuration in config.json
   - Verify proxy URLs are correct
   - Run health check: `python proxy_dashboard.py`

2. **High failure rate**
   - Check proxy quality
   - Adjust failure_threshold in config
   - Review blacklisted proxies in dashboard

3. **Configuration errors**
   - Run validation: `python setup_complete_proxy_system.py`
   - Check logs for specific error messages

### Debug Commands
```bash
# Test configuration
python -c "from scraper_proxy_integration import get_proxy_integration_status; print(get_proxy_integration_status())"

# Run health check
python -c "from proxy_integration import health_check_all_proxies; from config_loader import config_loader; health_check_all_proxies(config_loader.load_config())"
```

## Security Best Practices

1. **Never commit proxy credentials to git**
2. **Use environment variables for sensitive data**
3. **Regularly rotate proxy credentials**
4. **Monitor proxy usage for abuse**
5. **Use HTTPS proxies when possible**

## Performance Optimization

1. **Use health_based rotation strategy** for best performance
2. **Set appropriate failure_threshold** (recommended: 3)
3. **Configure domain-specific assignment** for different sites
4. **Monitor and adjust health_check_interval** based on proxy stability

For more information, see the proxy system implementation files:
- `proxy_manager.py` - Core proxy management
- `proxy_integration.py` - Integration with requests/Playwright
- `scraper_proxy_integration.py` - Enhanced scraper integration
"""
            
            docs_file = self.project_root / "PROXY_SYSTEM_USAGE.md"
            with open(docs_file, 'w') as f:
                f.write(docs_content)
            
            logger.info(f"📖 Usage documentation created: {docs_file}")
            self.setup_results.append(("Documentation", True, f"Created {docs_file}"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create documentation: {e}")
            self.setup_results.append(("Documentation", False, f"Documentation failed: {e}"))
            return False
    
    def run_complete_setup(self) -> Dict:
        """Run complete proxy system setup"""
        logger.info("🚀 Starting complete proxy system setup...")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Setup steps
        steps = [
            ("Validate Prerequisites", self.validate_prerequisites),
            ("Backup Configuration", self.backup_current_config),
            ("Update Configuration", self.update_configuration_file),
            ("Validate Configuration", self.validate_configuration),
            ("Run Integration Tests", self.run_integration_tests),
            ("Create Documentation", self.create_usage_documentation)
        ]
        
        passed = 0
        failed = 0
        
        for step_name, step_function in steps:
            logger.info(f"\n📋 {step_name}...")
            try:
                if step_function():
                    passed += 1
                else:
                    failed += 1
                    if step_name in ["Validate Prerequisites", "Update Configuration"]:
                        logger.error(f"❌ Critical step failed: {step_name}")
                        break
            except Exception as e:
                logger.error(f"❌ Unexpected error in {step_name}: {e}")
                failed += 1
                self.setup_results.append((step_name, False, f"Unexpected error: {e}"))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = {
            "total_steps": len(steps),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(steps)) * 100,
            "duration_seconds": duration,
            "setup_results": self.setup_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 PROXY SYSTEM SETUP RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Steps: {summary['total_steps']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info("=" * 60)
        
        # Print detailed results
        for step_name, success, message in self.setup_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status}: {step_name} - {message}")
        
        logger.info("=" * 60)
        
        if failed == 0:
            logger.info("🎉 PROXY SYSTEM SETUP COMPLETED SUCCESSFULLY!")
            logger.info("\n📋 Next Steps:")
            logger.info("1. Edit config.json proxy settings with your real proxy URLs")
            logger.info("2. Set 'enabled': true in proxy_settings when ready")
            logger.info("3. Run: python scraper.py --run")
            logger.info("4. Monitor: python proxy_dashboard.py")
        else:
            logger.warning(f"⚠️ Setup completed with {failed} issues.")
            logger.warning("Check logs above for details and resolve issues before using.")
        
        return summary


def main():
    """Main setup function"""
    print("🔄 MIMIR COMPLETE PROXY SYSTEM SETUP")
    print("=" * 60)
    print("Setting up enterprise-grade proxy rotation for Mimir")
    print("=" * 60)
    
    setup = ProxySystemSetup()
    results = setup.run_complete_setup()
    
    # Save results
    results_file = f"proxy_setup_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Setup results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save setup results: {e}")
    
    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    
    if exit_code == 0:
        print("\n🎯 SETUP COMPLETE: Proxy system is ready for configuration!")
        print("📖 See PROXY_SYSTEM_USAGE.md for detailed usage instructions.")
    else:
        print("\n⚠️ SETUP INCOMPLETE: Please resolve the issues above.")
    
    return exit_code


if __name__ == "__main__":
    exit(main())