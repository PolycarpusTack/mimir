"""
Advanced Proxy Configuration Validator and Auto-Corrector for Mimir
Validates, diagnoses, and automatically fixes proxy configuration issues
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import jsonschema
import requests

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    severity: str  # 'critical', 'warning', 'info'
    issue: str
    suggestion: str
    auto_fixable: bool
    fix_applied: bool = False


class ProxyConfigValidator:
    """Advanced proxy configuration validator with auto-correction capabilities"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.validation_results: List[ValidationResult] = []
        self.config_data: Optional[Dict] = None
        
    def load_config(self) -> bool:
        """Load configuration file"""
        try:
            if not self.config_file.exists():
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="critical",
                    issue="Configuration file does not exist",
                    suggestion=f"Create {self.config_file} with proxy settings",
                    auto_fixable=True
                ))
                return False
            
            with open(self.config_file, 'r') as f:
                self.config_data = json.load(f)
            
            return True
            
        except json.JSONDecodeError as e:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="critical",
                issue=f"Invalid JSON in configuration file: {e}",
                suggestion="Fix JSON syntax errors in config.json",
                auto_fixable=False
            ))
            return False
        except Exception as e:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="critical",
                issue=f"Failed to load configuration: {e}",
                suggestion="Check file permissions and path",
                auto_fixable=False
            ))
            return False
    
    def get_proxy_config_schema(self) -> Dict:
        """Get JSON schema for proxy configuration validation"""
        return {
            "type": "object",
            "properties": {
                "proxy_settings": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "rotation_strategy": {
                            "type": "string",
                            "enum": ["round_robin", "random", "health_based"]
                        },
                        "health_check_interval": {"type": "number", "minimum": 10},
                        "failure_threshold": {"type": "number", "minimum": 1},
                        "health_check_timeout": {"type": "number", "minimum": 1},
                        "health_check_url": {"type": "string", "format": "uri"},
                        "proxy_providers": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["static_list", "file", "api"]
                                    },
                                    "name": {"type": "string", "minLength": 1},
                                    "enabled": {"type": "boolean"}
                                },
                                "required": ["type", "name", "enabled"]
                            }
                        }
                    },
                    "required": ["enabled", "rotation_strategy", "proxy_providers"]
                }
            }
        }
    
    def validate_json_schema(self) -> bool:
        """Validate configuration against JSON schema"""
        if not self.config_data:
            return False
        
        schema = self.get_proxy_config_schema()
        
        try:
            jsonschema.validate(self.config_data, schema)
            self.validation_results.append(ValidationResult(
                is_valid=True,
                severity="info",
                issue="Configuration structure is valid",
                suggestion="",
                auto_fixable=False
            ))
            return True
        except jsonschema.ValidationError as e:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="critical",
                issue=f"Schema validation failed: {e.message}",
                suggestion=f"Fix schema issue at path: {' -> '.join(str(p) for p in e.absolute_path)}",
                auto_fixable=True
            ))
            return False
    
    def validate_proxy_settings_exist(self) -> bool:
        """Validate that proxy settings section exists"""
        if not self.config_data:
            return False
        
        if "proxy_settings" not in self.config_data:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="critical",
                issue="Missing 'proxy_settings' section in configuration",
                suggestion="Add proxy_settings section to config.json",
                auto_fixable=True
            ))
            return False
        
        return True
    
    def validate_proxy_providers(self) -> bool:
        """Validate proxy provider configurations"""
        if not self.config_data or "proxy_settings" not in self.config_data:
            return False
        
        proxy_settings = self.config_data["proxy_settings"]
        providers = proxy_settings.get("proxy_providers", [])
        
        if not providers:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="critical",
                issue="No proxy providers configured",
                suggestion="Add at least one proxy provider to proxy_providers array",
                auto_fixable=True
            ))
            return False
        
        enabled_providers = [p for p in providers if p.get("enabled", False)]
        if not enabled_providers:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                issue="No proxy providers are enabled",
                suggestion="Enable at least one proxy provider by setting 'enabled': true",
                auto_fixable=True
            ))
        
        # Validate each provider
        for i, provider in enumerate(providers):
            self._validate_single_provider(provider, i)
        
        return True
    
    def _validate_single_provider(self, provider: Dict, index: int) -> None:
        """Validate a single proxy provider configuration"""
        provider_type = provider.get("type")
        provider_name = provider.get("name", f"provider_{index}")
        
        if provider_type == "static_list":
            proxies = provider.get("proxies", [])
            if not proxies:
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="warning",
                    issue=f"Static provider '{provider_name}' has no proxies",
                    suggestion="Add proxy URLs to the 'proxies' array",
                    auto_fixable=False
                ))
            
            # Check for example URLs
            example_urls = ["proxy1.example.com", "proxy2.example.com", "example.com"]
            for proxy_url in proxies:
                if any(example in proxy_url for example in example_urls):
                    self.validation_results.append(ValidationResult(
                        is_valid=False,
                        severity="warning",
                        issue=f"Provider '{provider_name}' contains example proxy URL: {proxy_url}",
                        suggestion="Replace example URLs with real proxy endpoints",
                        auto_fixable=False
                    ))
        
        elif provider_type == "file":
            file_path = provider.get("file_path")
            if not file_path:
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="critical",
                    issue=f"File provider '{provider_name}' missing 'file_path'",
                    suggestion="Add 'file_path' pointing to proxy list JSON file",
                    auto_fixable=True
                ))
            elif not Path(file_path).exists():
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="critical",
                    issue=f"File provider '{provider_name}' references non-existent file: {file_path}",
                    suggestion=f"Create {file_path} or update file_path to correct location",
                    auto_fixable=True
                ))
        
        elif provider_type == "api":
            api_url = provider.get("api_url")
            if not api_url:
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="critical",
                    issue=f"API provider '{provider_name}' missing 'api_url'",
                    suggestion="Add 'api_url' for the API endpoint",
                    auto_fixable=False
                ))
    
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables for proxy authentication"""
        if not self.config_data:
            return False
        
        # Check for commonly used environment variables
        common_env_vars = [
            "BRIGHTDATA_USERNAME", "BRIGHTDATA_PASSWORD",
            "OXYLABS_USERNAME", "OXYLABS_PASSWORD",
            "SMARTPROXY_USERNAME", "SMARTPROXY_PASSWORD",
            "PROXY_USERNAME", "PROXY_PASSWORD"
        ]
        
        missing_vars = []
        for var in common_env_vars:
            if var.endswith("_USERNAME") and os.getenv(var):
                password_var = var.replace("_USERNAME", "_PASSWORD")
                if not os.getenv(password_var):
                    missing_vars.append(password_var)
        
        if missing_vars:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                issue=f"Missing environment variables: {', '.join(missing_vars)}",
                suggestion="Set missing environment variables for proxy authentication",
                auto_fixable=False
            ))
        
        return True
    
    def validate_referenced_files(self) -> bool:
        """Validate that all referenced files exist"""
        if not self.config_data:
            return False
        
        proxy_settings = self.config_data.get("proxy_settings", {})
        
        # Check save_state_file
        state_file = proxy_settings.get("save_state_file")
        if state_file and not Path(state_file).parent.exists():
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                issue=f"Directory for save_state_file does not exist: {Path(state_file).parent}",
                suggestion=f"Create directory {Path(state_file).parent} or update save_state_file path",
                auto_fixable=True
            ))
        
        # Check playwright config file
        playwright_settings = self.config_data.get("playwright", {})
        playwright_config_file = playwright_settings.get("config_file")
        if playwright_config_file and not Path(playwright_config_file).exists():
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                issue=f"Playwright config file does not exist: {playwright_config_file}",
                suggestion=f"Create {playwright_config_file} or run setup_playwright.py",
                auto_fixable=True
            ))
        
        return True
    
    def validate_proxy_connectivity(self) -> bool:
        """Test connectivity to configured proxies"""
        if not self.config_data:
            return False
        
        proxy_settings = self.config_data.get("proxy_settings", {})
        if not proxy_settings.get("enabled", False):
            self.validation_results.append(ValidationResult(
                is_valid=True,
                severity="info",
                issue="Proxy system is disabled - skipping connectivity tests",
                suggestion="Enable proxy system to test connectivity",
                auto_fixable=False
            ))
            return True
        
        # Test health check URL
        health_check_url = proxy_settings.get("health_check_url", "http://httpbin.org/ip")
        try:
            response = requests.get(health_check_url, timeout=10)
            if response.status_code != 200:
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="warning",
                    issue=f"Health check URL returned status {response.status_code}: {health_check_url}",
                    suggestion="Update health_check_url to a working endpoint",
                    auto_fixable=False
                ))
        except Exception as e:
            self.validation_results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                issue=f"Cannot reach health check URL: {e}",
                suggestion="Check internet connectivity or update health_check_url",
                auto_fixable=False
            ))
        
        return True
    
    def auto_fix_issues(self) -> bool:
        """Automatically fix issues that can be auto-corrected"""
        if not self.config_data:
            if not self.config_file.exists():
                # Create basic config file
                self._create_basic_config()
                return True
        
        fixes_applied = 0
        
        # Fix missing proxy_settings section
        if "proxy_settings" not in self.config_data:
            self.config_data["proxy_settings"] = self._get_default_proxy_settings()
            fixes_applied += 1
        
        # Fix missing required fields in proxy_settings
        proxy_settings = self.config_data["proxy_settings"]
        
        if "rotation_strategy" not in proxy_settings:
            proxy_settings["rotation_strategy"] = "health_based"
            fixes_applied += 1
        
        if "proxy_providers" not in proxy_settings or not proxy_settings["proxy_providers"]:
            proxy_settings["proxy_providers"] = self._get_default_proxy_providers()
            fixes_applied += 1
        
        # Create missing directories
        state_file = proxy_settings.get("save_state_file")
        if state_file:
            state_dir = Path(state_file).parent
            if not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)
                fixes_applied += 1
        
        # Create missing proxy_list.json if referenced
        for provider in proxy_settings.get("proxy_providers", []):
            if provider.get("type") == "file":
                file_path = provider.get("file_path")
                if file_path and not Path(file_path).exists():
                    self._create_proxy_list_file(file_path)
                    fixes_applied += 1
        
        # Create missing playwright config if referenced
        playwright_settings = self.config_data.get("playwright", {})
        playwright_config_file = playwright_settings.get("config_file")
        if playwright_config_file and not Path(playwright_config_file).exists():
            self._create_playwright_config(playwright_config_file)
            fixes_applied += 1
        
        # Save fixed configuration
        if fixes_applied > 0:
            self._save_config()
            logger.info(f"Applied {fixes_applied} automatic fixes to configuration")
        
        return fixes_applied > 0
    
    def _create_basic_config(self) -> None:
        """Create basic configuration file"""
        basic_config = {
            "user_agent": "Mimir-NewsScraper/1.0",
            "default_request_delay_seconds": 2,
            "database_name": "mimir_news.db",
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "notification_email": {
                "send_email": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/mimir_scraper.log",
                "max_bytes": 10485760,
                "backup_count": 5
            },
            "keywords_monitoring": {
                "enabled": True,
                "keywords": ["AI", "machine learning", "automation"],
                "alert_on_match": True
            },
            "api_sources": {
                "enabled": False
            },
            "playwright": {
                "enabled": False,
                "config_file": "playwright_config.json"
            },
            "proxy_settings": self._get_default_proxy_settings()
        }
        
        self.config_data = basic_config
        self._save_config()
    
    def _get_default_proxy_settings(self) -> Dict:
        """Get default proxy settings"""
        return {
            "enabled": False,
            "rotation_strategy": "health_based",
            "health_check_interval": 300,
            "failure_threshold": 3,
            "health_check_timeout": 10,
            "health_check_url": "http://httpbin.org/ip",
            "save_state_file": "proxy_state.json",
            "auto_save_interval": 600,
            "proxy_providers": self._get_default_proxy_providers(),
            "domain_specific": {
                "enabled": False,
                "assignments": {}
            },
            "tenant_isolation": {
                "enabled": False,
                "tenant_proxy_pools": {}
            }
        }
    
    def _get_default_proxy_providers(self) -> List[Dict]:
        """Get default proxy provider configurations"""
        return [
            {
                "type": "static_list",
                "name": "example_proxies",
                "enabled": False,
                "description": "Example proxy configuration - replace with real proxies",
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
                "file_path": "proxy_list.json"
            }
        ]
    
    def _create_proxy_list_file(self, file_path: str) -> None:
        """Create missing proxy list file"""
        proxy_list_data = {
            "description": "Proxy list configuration for Mimir News Scraper",
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "proxies": [
                {
                    "url": "http://proxy1.example.com:8080",
                    "location": "US-East",
                    "provider": "example_provider",
                    "tags": ["reliable", "fast"],
                    "notes": "Replace with actual proxy URL"
                },
                {
                    "url": "http://proxy2.example.com:8080",
                    "location": "EU-West",
                    "provider": "example_provider",
                    "tags": ["secure", "stable"],
                    "notes": "Replace with actual proxy URL"
                }
            ],
            "usage_notes": [
                "Replace example URLs with actual proxy endpoints",
                "Configure authentication if required",
                "Adjust tags and metadata as needed"
            ]
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(proxy_list_data, f, indent=2)
    
    def _create_playwright_config(self, config_file: str) -> None:
        """Create missing Playwright configuration file"""
        # This is a simplified version - the real one should be created by setup_playwright.py
        playwright_config = {
            "browsers": {
                "chromium": {
                    "enabled": True,
                    "headless": True,
                    "args": ["--disable-blink-features=AutomationControlled"]
                }
            },
            "defaults": {
                "viewport": {"width": 1920, "height": 1080},
                "user_agent": "Mozilla/5.0 (compatible; Mimir/1.0)",
                "locale": "en-US"
            },
            "note": "Run setup_playwright.py for complete configuration"
        }
        
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(playwright_config, f, indent=2)
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        if self.config_data:
            # Create backup
            backup_file = f"{self.config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.config_file.exists():
                import shutil
                shutil.copy2(self.config_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            
            # Save updated config
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=4)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks"""
        logger.info("Starting comprehensive proxy configuration validation...")
        
        validation_steps = [
            ("Loading configuration", self.load_config),
            ("Validating JSON schema", self.validate_json_schema),
            ("Checking proxy settings", self.validate_proxy_settings_exist),
            ("Validating proxy providers", self.validate_proxy_providers),
            ("Checking environment variables", self.validate_environment_variables),
            ("Validating referenced files", self.validate_referenced_files),
            ("Testing connectivity", self.validate_proxy_connectivity)
        ]
        
        steps_passed = 0
        critical_issues = 0
        warnings = 0
        
        for step_name, step_function in validation_steps:
            logger.info(f"Running: {step_name}")
            try:
                step_function()
                steps_passed += 1
            except Exception as e:
                logger.error(f"Validation step failed: {step_name} - {e}")
                self.validation_results.append(ValidationResult(
                    is_valid=False,
                    severity="critical",
                    issue=f"Validation step failed: {step_name}",
                    suggestion=f"Check logs for details: {e}",
                    auto_fixable=False
                ))
        
        # Count issues by severity
        for result in self.validation_results:
            if result.severity == "critical":
                critical_issues += 1
            elif result.severity == "warning":
                warnings += 1
        
        # Apply auto-fixes
        fixes_applied = self.auto_fix_issues()
        
        summary = {
            "validation_passed": critical_issues == 0,
            "total_steps": len(validation_steps),
            "steps_passed": steps_passed,
            "critical_issues": critical_issues,
            "warnings": warnings,
            "fixes_applied": fixes_applied,
            "results": [
                {
                    "severity": r.severity,
                    "issue": r.issue,
                    "suggestion": r.suggestion,
                    "auto_fixable": r.auto_fixable,
                    "fix_applied": r.fix_applied
                }
                for r in self.validation_results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def generate_validation_report(self) -> str:
        """Generate human-readable validation report"""
        summary = self.run_comprehensive_validation()
        
        report = f"""
# Mimir Proxy Configuration Validation Report

**Generated**: {summary['timestamp']}
**Overall Status**: {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}

## Summary
- **Total Validation Steps**: {summary['total_steps']}
- **Steps Passed**: {summary['steps_passed']}
- **Critical Issues**: {summary['critical_issues']}
- **Warnings**: {summary['warnings']}
- **Auto-fixes Applied**: {summary['fixes_applied']}

## Detailed Results

"""
        
        for result in summary['results']:
            status_icon = "✅" if result['severity'] == "info" else "⚠️" if result['severity'] == "warning" else "❌"
            report += f"### {status_icon} {result['severity'].upper()}: {result['issue']}\n"
            if result['suggestion']:
                report += f"**Suggestion**: {result['suggestion']}\n"
            if result['auto_fixable']:
                fix_status = "Applied" if result['fix_applied'] else "Available"
                report += f"**Auto-fix**: {fix_status}\n"
            report += "\n"
        
        if summary['critical_issues'] == 0:
            report += "## ✅ Configuration is ready for use!\n"
            if summary['warnings'] > 0:
                report += "Consider addressing the warnings above for optimal performance.\n"
        else:
            report += "## ❌ Critical issues must be resolved before using the proxy system.\n"
        
        return report


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate and fix Mimir proxy configuration")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--auto-fix", action="store_true", help="Automatically fix issues")
    parser.add_argument("--report", action="store_true", help="Generate validation report")
    
    args = parser.parse_args()
    
    validator = ProxyConfigValidator(args.config)
    
    if args.report:
        report = validator.generate_validation_report()
        print(report)
        
        # Save report to file
        report_file = f"proxy_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_file}")
    else:
        summary = validator.run_comprehensive_validation()
        
        if summary['validation_passed']:
            print("✅ Proxy configuration validation PASSED")
            if summary['warnings'] > 0:
                print(f"⚠️  {summary['warnings']} warnings found")
        else:
            print("❌ Proxy configuration validation FAILED")
            print(f"Critical issues: {summary['critical_issues']}")
        
        if summary['fixes_applied']:
            print(f"🔧 Applied {summary['fixes_applied']} automatic fixes")
    
    return 0 if validator.validation_results else 1


if __name__ == "__main__":
    exit(main())