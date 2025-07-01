#!/usr/bin/env python3
"""
Security Configuration and Scanning Script for Mimir Enterprise

Configures security settings and runs comprehensive security scans.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from security.security_scanner import SecurityScanner, run_security_scan

logger = logging.getLogger(__name__)


class SecurityConfigurator:
    """Security configuration manager"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config_file = self.project_root / "config" / "security.json"

    def create_default_config(self) -> Dict[str, Any]:
        """Create default security configuration"""
        return {
            "security": {
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_hour": 1000,
                    "requests_per_minute": 60,
                    "burst_limit": 10,
                },
                "input_validation": {
                    "enabled": True,
                    "max_request_size": 10485760,  # 10MB
                    "dangerous_patterns": [
                        "<script[^>]*>.*?</script>",
                        "javascript:",
                        "vbscript:",
                        "on\\w+\\s*=",
                        "<iframe[^>]*>",
                        "\\.\\./",
                        "union.*select",
                        "drop\\s+table",
                    ],
                },
                "headers": {
                    "strict_transport_security": "max-age=31536000; includeSubDomains; preload",
                    "content_security_policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; img-src 'self' data: https: blob:; font-src 'self' https://fonts.gstatic.com; connect-src 'self' https: wss:; media-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; upgrade-insecure-requests",
                    "x_content_type_options": "nosniff",
                    "x_frame_options": "DENY",
                    "x_xss_protection": "1; mode=block",
                    "referrer_policy": "strict-origin-when-cross-origin",
                    "permissions_policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
                },
                "authentication": {
                    "jwt_expiry_hours": 24,
                    "refresh_token_expiry_days": 30,
                    "require_2fa": False,
                    "password_policy": {
                        "min_length": 12,
                        "require_uppercase": True,
                        "require_lowercase": True,
                        "require_numbers": True,
                        "require_symbols": True,
                        "max_age_days": 90,
                    },
                },
                "encryption": {"algorithm": "AES-256-GCM", "key_rotation_days": 30, "backup_encryption": True},
                "logging": {
                    "security_events": True,
                    "failed_login_threshold": 5,
                    "log_retention_days": 365,
                    "sensitive_data_masking": True,
                },
                "vulnerability_scanning": {
                    "enabled": True,
                    "daily_scans": True,
                    "scan_dependencies": True,
                    "scan_code": True,
                    "scan_config": True,
                },
            },
            "compliance": {
                "gdpr": {"enabled": True, "data_retention_days": 1095, "anonymization_enabled": True},  # 3 years
                "hipaa": {"enabled": False, "encryption_at_rest": True, "audit_logging": True},
                "sox": {"enabled": False, "change_management": True, "segregation_of_duties": True},
            },
        }

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save security configuration"""
        # Create config directory if it doesn't exist
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Security configuration saved to {self.config_file}")

    def load_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        if not self.config_file.exists():
            logger.warning(f"Security config not found at {self.config_file}, creating default")
            config = self.create_default_config()
            self.save_config(config)
            return config

        with open(self.config_file, "r") as f:
            return json.load(f)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate security configuration"""
        issues = []

        # Check rate limiting
        if not config.get("security", {}).get("rate_limiting", {}).get("enabled"):
            issues.append("Rate limiting is disabled")

        # Check input validation
        if not config.get("security", {}).get("input_validation", {}).get("enabled"):
            issues.append("Input validation is disabled")

        # Check HTTPS enforcement
        hsts = config.get("security", {}).get("headers", {}).get("strict_transport_security", "")
        if "max-age" not in hsts:
            issues.append("HTTPS Strict Transport Security not properly configured")

        # Check CSP
        csp = config.get("security", {}).get("headers", {}).get("content_security_policy", "")
        if "default-src" not in csp:
            issues.append("Content Security Policy not properly configured")

        # Check password policy
        password_policy = config.get("security", {}).get("authentication", {}).get("password_policy", {})
        if password_policy.get("min_length", 0) < 12:
            issues.append("Password minimum length should be at least 12 characters")

        return issues


class SecurityAuditor:
    """Security audit manager"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scanner = SecurityScanner(str(project_root))
        self.configurator = SecurityConfigurator(str(project_root))

    def run_comprehensive_audit(self, output_dir: str = None) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        if not output_dir:
            output_dir = self.project_root / "reports" / "security"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        audit_results = {
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "project_root": str(self.project_root),
            "components": {},
        }

        logger.info("Starting comprehensive security audit...")

        # 1. Code vulnerability scan
        logger.info("Running code vulnerability scan...")
        scan_result = self.scanner.scan_project(include_tests=False)

        # Save detailed scan report
        scan_report_path = output_path / "vulnerability_scan.html"
        with open(scan_report_path, "w") as f:
            f.write(self.scanner.generate_report(scan_result, "html"))

        audit_results["components"]["vulnerability_scan"] = {
            "total_vulnerabilities": len(scan_result.vulnerabilities),
            "critical": scan_result.get_critical_count(),
            "high": scan_result.get_high_count(),
            "report_path": str(scan_report_path),
        }

        # 2. Configuration audit
        logger.info("Auditing security configuration...")
        config = self.configurator.load_config()
        config_issues = self.configurator.validate_config(config)

        audit_results["components"]["configuration"] = {
            "issues": config_issues,
            "config_path": str(self.configurator.config_file),
        }

        # 3. Dependency audit (if tools available)
        logger.info("Checking dependencies...")
        dep_issues = self._audit_dependencies()
        audit_results["components"]["dependencies"] = dep_issues

        # 4. Infrastructure security
        logger.info("Checking infrastructure security...")
        infra_issues = self._audit_infrastructure()
        audit_results["components"]["infrastructure"] = infra_issues

        # 5. Generate summary report
        summary_path = output_path / "security_audit_summary.json"
        with open(summary_path, "w") as f:
            json.dump(audit_results, f, indent=2)

        logger.info(f"Security audit completed. Results saved to {output_path}")
        return audit_results

    def _audit_dependencies(self) -> Dict[str, Any]:
        """Audit project dependencies"""
        issues = []

        # Check for requirements.txt
        req_files = list(self.project_root.glob("**/requirements*.txt"))
        if not req_files:
            issues.append("No requirements.txt files found")

        # Check for known vulnerable packages (basic check)
        vulnerable_packages = ["django<3.2.0", "flask<2.0.0", "requests<2.20.0", "pyyaml<5.4.0"]

        for req_file in req_files:
            try:
                with open(req_file, "r") as f:
                    content = f.read()
                    for vuln_pkg in vulnerable_packages:
                        if vuln_pkg.split("<")[0] in content:
                            issues.append(f"Potentially vulnerable package in {req_file}: {vuln_pkg}")
            except Exception as e:
                issues.append(f"Error reading {req_file}: {e}")

        return {"issues": issues, "requirements_files": [str(f) for f in req_files]}

    def _audit_infrastructure(self) -> Dict[str, Any]:
        """Audit infrastructure security"""
        issues = []

        # Check for Docker security
        docker_files = list(self.project_root.glob("**/Dockerfile*"))
        for dockerfile in docker_files:
            try:
                with open(dockerfile, "r") as f:
                    content = f.read()
                    if "USER root" in content:
                        issues.append(f"Dockerfile runs as root: {dockerfile}")
                    if "ADD" in content and "http" in content:
                        issues.append(f"Dockerfile uses ADD with remote URL: {dockerfile}")
            except Exception as e:
                issues.append(f"Error reading {dockerfile}: {e}")

        # Check for exposed secrets in config files
        config_files = list(self.project_root.glob("**/*.{env,conf,ini,yaml,yml}"))
        secret_patterns = ["password", "secret", "key", "token", "api_key"]

        for config_file in config_files:
            if config_file.name.startswith("."):
                continue
            try:
                with open(config_file, "r") as f:
                    content = f.read().lower()
                    for pattern in secret_patterns:
                        if f"{pattern}=" in content or f"{pattern}:" in content:
                            issues.append(f"Potential secret in config file: {config_file}")
                            break
            except Exception:
                pass  # Skip binary or unreadable files

        return {
            "issues": issues,
            "docker_files": [str(f) for f in docker_files],
            "config_files_checked": len(config_files),
        }


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Mimir Security Configuration and Auditing")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", help="Output directory for reports")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage security configuration")
    config_parser.add_argument("--create", action="store_true", help="Create default config")
    config_parser.add_argument("--validate", action="store_true", help="Validate current config")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Run security scans")
    scan_parser.add_argument("--code", action="store_true", help="Scan code for vulnerabilities")
    scan_parser.add_argument("--full", action="store_true", help="Run comprehensive audit")
    scan_parser.add_argument("--format", choices=["json", "html", "txt"], default="html", help="Output format")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.command == "config":
        configurator = SecurityConfigurator(args.project_root)

        if args.create:
            config = configurator.create_default_config()
            configurator.save_config(config)
            print(f"Default security configuration created at {configurator.config_file}")

        elif args.validate:
            config = configurator.load_config()
            issues = configurator.validate_config(config)
            if issues:
                print("Security configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("Security configuration is valid")

    elif args.command == "scan":
        if args.code:
            result = run_security_scan(
                project_root=args.project_root,
                output_file=os.path.join(args.output_dir or ".", f"security_scan.{args.format}"),
            )
            print(f"Vulnerability scan completed. Found {len(result.vulnerabilities)} issues.")

        elif args.full:
            auditor = SecurityAuditor(args.project_root)
            results = auditor.run_comprehensive_audit(args.output_dir)

            total_issues = sum(
                [
                    results["components"]["vulnerability_scan"]["total_vulnerabilities"],
                    len(results["components"]["configuration"]["issues"]),
                    len(results["components"]["dependencies"]["issues"]),
                    len(results["components"]["infrastructure"]["issues"]),
                ]
            )

            print(f"Comprehensive security audit completed. Found {total_issues} total issues.")
            print(f"Critical vulnerabilities: {results['components']['vulnerability_scan']['critical']}")
            print(f"High severity vulnerabilities: {results['components']['vulnerability_scan']['high']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
