"""
Security Scanner for Mimir Enterprise

OWASP compliance checking and vulnerability scanning.
"""

import ast
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class VulnerabilityLevel(str, Enum):
    """Vulnerability severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityCategory(str, Enum):
    """OWASP Top 10 vulnerability categories"""

    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    KNOWN_VULNERABILITIES = "known_vulnerabilities"
    INSUFFICIENT_LOGGING = "insufficient_logging"


@dataclass
class SecurityVulnerability:
    """Security vulnerability finding"""

    id: str
    title: str
    description: str
    category: VulnerabilityCategory
    level: VulnerabilityLevel
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "level": self.level.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "confidence": self.confidence,
        }


@dataclass
class SecurityScanResult:
    """Security scan result"""

    scan_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    scan_config: Dict[str, Any] = field(default_factory=dict)

    def add_vulnerability(self, vulnerability: SecurityVulnerability):
        """Add vulnerability to results"""
        self.vulnerabilities.append(vulnerability)

        # Update summary
        level = vulnerability.level.value
        self.summary[level] = self.summary.get(level, 0) + 1

    def get_critical_count(self) -> int:
        """Get count of critical vulnerabilities"""
        return self.summary.get("critical", 0)

    def get_high_count(self) -> int:
        """Get count of high vulnerabilities"""
        return self.summary.get("high", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scan_id": self.scan_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "summary": self.summary,
            "scan_config": self.scan_config,
            "total_vulnerabilities": len(self.vulnerabilities),
        }


class SecurityScanner:
    """Main security scanner class"""

    def __init__(self, project_root: str):
        """
        Initialize security scanner

        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root)
        self.scan_patterns = self._initialize_scan_patterns()
        self.excluded_paths = {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            ".pytest_cache",
            "migrations",
            "tests",
        }

    def _initialize_scan_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize vulnerability detection patterns"""
        return {
            # SQL Injection patterns
            "sql_injection": [
                {
                    "pattern": r'f["\'].*?{.*?}.*?["\'].*?(execute|query|fetchone|fetchall)',
                    "title": "Potential SQL Injection via f-string",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-89",
                    "recommendation": "Use parameterized queries instead of f-strings for SQL",
                },
                {
                    "pattern": r"\.format\(.*?\).*?(execute|query)",
                    "title": "Potential SQL Injection via string formatting",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-89",
                    "recommendation": "Use parameterized queries with placeholders",
                },
                {
                    "pattern": r"%.*?%.*?(execute|query)",
                    "title": "Potential SQL Injection via string interpolation",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-89",
                    "recommendation": "Use parameterized queries",
                },
            ],
            # Command Injection patterns
            "command_injection": [
                {
                    "pattern": r"subprocess\.(call|run|Popen).*?shell=True",
                    "title": "Command injection risk with shell=True",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-78",
                    "recommendation": "Avoid shell=True, use array of arguments instead",
                },
                {
                    "pattern": r"os\.system\(",
                    "title": "Command injection risk with os.system",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-78",
                    "recommendation": "Use subprocess with argument list instead of os.system",
                },
            ],
            # Hardcoded secrets
            "hardcoded_secrets": [
                {
                    "pattern": r'(?i)(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                    "title": "Hardcoded secret detected",
                    "category": VulnerabilityCategory.SENSITIVE_DATA,
                    "level": VulnerabilityLevel.CRITICAL,
                    "cwe": "CWE-798",
                    "recommendation": "Use environment variables or secure key management",
                },
                {
                    "pattern": r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']',
                    "title": "Potential hardcoded base64 secret",
                    "category": VulnerabilityCategory.SENSITIVE_DATA,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-798",
                    "recommendation": "Verify if this is a hardcoded secret and move to environment",
                },
            ],
            # Weak cryptography
            "weak_crypto": [
                {
                    "pattern": r"hashlib\.(md5|sha1)\(",
                    "title": "Weak cryptographic hash function",
                    "category": VulnerabilityCategory.SENSITIVE_DATA,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-327",
                    "recommendation": "Use SHA-256 or stronger hash functions",
                },
                {
                    "pattern": r"random\.random\(\)|random\.randint\(",
                    "title": "Weak random number generation",
                    "category": VulnerabilityCategory.SENSITIVE_DATA,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-338",
                    "recommendation": "Use secrets module for cryptographic purposes",
                },
            ],
            # Path traversal
            "path_traversal": [
                {
                    "pattern": r"open\(.*?\+.*?\)",
                    "title": "Potential path traversal in file operations",
                    "category": VulnerabilityCategory.BROKEN_ACCESS,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-22",
                    "recommendation": "Validate and sanitize file paths",
                }
            ],
            # Unsafe deserialization
            "unsafe_deserialization": [
                {
                    "pattern": r"pickle\.loads?\(",
                    "title": "Unsafe deserialization with pickle",
                    "category": VulnerabilityCategory.INSECURE_DESERIALIZATION,
                    "level": VulnerabilityLevel.HIGH,
                    "cwe": "CWE-502",
                    "recommendation": "Use JSON or other safe serialization formats",
                },
                {
                    "pattern": r"eval\(",
                    "title": "Code injection via eval()",
                    "category": VulnerabilityCategory.INJECTION,
                    "level": VulnerabilityLevel.CRITICAL,
                    "cwe": "CWE-95",
                    "recommendation": "Avoid eval(), use ast.literal_eval() for safe evaluation",
                },
            ],
            # Debug/development code
            "debug_code": [
                {
                    "pattern": r"print\(.*?(password|secret|token|key)",
                    "title": "Sensitive data in debug output",
                    "category": VulnerabilityCategory.INSUFFICIENT_LOGGING,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-532",
                    "recommendation": "Remove debug prints containing sensitive data",
                },
                {
                    "pattern": r"debug\s*=\s*True",
                    "title": "Debug mode enabled",
                    "category": VulnerabilityCategory.SECURITY_MISCONFIG,
                    "level": VulnerabilityLevel.MEDIUM,
                    "cwe": "CWE-489",
                    "recommendation": "Disable debug mode in production",
                },
            ],
        }

    def scan_project(self, include_tests: bool = False) -> SecurityScanResult:
        """
        Perform comprehensive security scan

        Args:
            include_tests: Whether to include test files in scan

        Returns:
            Security scan results
        """
        scan_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        result = SecurityScanResult(
            scan_id=scan_id, started_at=datetime.now(timezone.utc), scan_config={"include_tests": include_tests}
        )

        logger.info(f"Starting security scan {scan_id}")

        try:
            # Scan Python files
            python_files = self._get_python_files(include_tests)
            for file_path in python_files:
                self._scan_file(file_path, result)

            # Scan configuration files
            config_files = self._get_config_files()
            for file_path in config_files:
                self._scan_config_file(file_path, result)

            # Check dependencies
            self._scan_dependencies(result)

            # Check security headers
            self._check_security_headers(result)

            # Check authentication implementation
            self._check_authentication(result)

            result.completed_at = datetime.now(timezone.utc)

            logger.info(f"Security scan {scan_id} completed. Found {len(result.vulnerabilities)} vulnerabilities")

        except Exception as e:
            logger.error(f"Error during security scan: {e}")
            result.completed_at = datetime.now(timezone.utc)

        return result

    def _get_python_files(self, include_tests: bool = False) -> List[Path]:
        """Get list of Python files to scan"""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_paths]

            # Skip test directories if not including tests
            if not include_tests:
                dirs[:] = [d for d in dirs if not d.startswith("test")]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def _get_config_files(self) -> List[Path]:
        """Get list of configuration files to scan"""
        config_files = []
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.env*", "*.conf", "*.ini"]

        for pattern in config_patterns:
            config_files.extend(self.project_root.glob(f"**/{pattern}"))

        return config_files

    def _scan_file(self, file_path: Path, result: SecurityScanResult):
        """Scan a single Python file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Pattern-based scanning
            for category, patterns in self.scan_patterns.items():
                for pattern_config in patterns:
                    matches = re.finditer(pattern_config["pattern"], content, re.MULTILINE | re.IGNORECASE)

                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        # Extract code snippet
                        lines = content.split("\n")
                        start_line = max(0, line_number - 2)
                        end_line = min(len(lines), line_number + 1)
                        code_snippet = "\n".join(lines[start_line:end_line])

                        vulnerability = SecurityVulnerability(
                            id=f"{category}_{line_number}_{hash(str(file_path))}",
                            title=pattern_config["title"],
                            description=f"Found in {file_path}:{line_number}",
                            category=pattern_config["category"],
                            level=pattern_config["level"],
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_number,
                            code_snippet=code_snippet,
                            recommendation=pattern_config["recommendation"],
                            cwe_id=pattern_config.get("cwe"),
                            owasp_category=pattern_config["category"].value,
                        )

                        result.add_vulnerability(vulnerability)

            # AST-based analysis for more complex patterns
            self._ast_analysis(file_path, content, result)

        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")

    def _ast_analysis(self, file_path: Path, content: str, result: SecurityScanResult):
        """Perform AST-based security analysis"""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    self._check_dangerous_calls(node, file_path, result)

                # Check for hardcoded strings that might be secrets
                elif isinstance(node, ast.Str):
                    self._check_hardcoded_strings(node, file_path, result)

                # Check for unsafe imports
                elif isinstance(node, ast.Import):
                    self._check_unsafe_imports(node, file_path, result)

        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            logger.debug(f"AST analysis error for {file_path}: {e}")

    def _check_dangerous_calls(self, node: ast.Call, file_path: Path, result: SecurityScanResult):
        """Check for dangerous function calls"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check for eval/exec
            if func_name in ["eval", "exec"]:
                vulnerability = SecurityVulnerability(
                    id=f"dangerous_call_{node.lineno}_{hash(str(file_path))}",
                    title=f"Dangerous function call: {func_name}",
                    description=f"Use of {func_name}() can lead to code injection",
                    category=VulnerabilityCategory.INJECTION,
                    level=VulnerabilityLevel.CRITICAL,
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    recommendation=f"Avoid {func_name}(), use safer alternatives",
                    cwe_id="CWE-95",
                )
                result.add_vulnerability(vulnerability)

    def _check_hardcoded_strings(self, node: ast.Str, file_path: Path, result: SecurityScanResult):
        """Check for potential hardcoded secrets in strings"""
        if len(node.s) > 20 and re.match(r"^[A-Za-z0-9+/=]+$", node.s):
            # Looks like base64
            vulnerability = SecurityVulnerability(
                id=f"hardcoded_b64_{node.lineno}_{hash(str(file_path))}",
                title="Potential hardcoded base64 string",
                description="Long base64-like string found",
                category=VulnerabilityCategory.SENSITIVE_DATA,
                level=VulnerabilityLevel.LOW,
                file_path=str(file_path.relative_to(self.project_root)),
                line_number=node.lineno,
                recommendation="Verify if this is sensitive data that should be externalized",
                cwe_id="CWE-798",
            )
            result.add_vulnerability(vulnerability)

    def _check_unsafe_imports(self, node: ast.Import, file_path: Path, result: SecurityScanResult):
        """Check for unsafe module imports"""
        unsafe_modules = ["pickle", "marshal", "shelve"]

        for alias in node.names:
            if alias.name in unsafe_modules:
                vulnerability = SecurityVulnerability(
                    id=f"unsafe_import_{node.lineno}_{hash(str(file_path))}",
                    title=f"Unsafe module import: {alias.name}",
                    description=f"Import of potentially unsafe module {alias.name}",
                    category=VulnerabilityCategory.INSECURE_DESERIALIZATION,
                    level=VulnerabilityLevel.MEDIUM,
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=node.lineno,
                    recommendation=f"Use safer alternatives to {alias.name}",
                    cwe_id="CWE-502",
                )
                result.add_vulnerability(vulnerability)

    def _scan_config_file(self, file_path: Path, result: SecurityScanResult):
        """Scan configuration files for security issues"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for secrets in config files
            secret_patterns = [
                r'(?i)(password|secret|key|token)\s*[:=]\s*["\']?[^"\'\s]{8,}',
                r'(?i)api[_-]?key\s*[:=]\s*["\']?[^"\'\s]{8,}',
                r'(?i)database[_-]?url\s*[:=]\s*["\']?[^"\'\s]+',
            ]

            for pattern in secret_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_number = content[: match.start()].count("\n") + 1

                    vulnerability = SecurityVulnerability(
                        id=f"config_secret_{line_number}_{hash(str(file_path))}",
                        title="Potential secret in configuration file",
                        description=f"Potential secret found in {file_path}:{line_number}",
                        category=VulnerabilityCategory.SENSITIVE_DATA,
                        level=VulnerabilityLevel.HIGH,
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_number,
                        recommendation="Move secrets to environment variables or secure vault",
                        cwe_id="CWE-798",
                    )
                    result.add_vulnerability(vulnerability)

        except Exception as e:
            logger.debug(f"Error scanning config file {file_path}: {e}")

    def _scan_dependencies(self, result: SecurityScanResult):
        """Scan dependencies for known vulnerabilities"""
        try:
            # Check if safety is installed and run it
            requirements_files = list(self.project_root.glob("**/requirements*.txt"))

            for req_file in requirements_files:
                try:
                    # Run safety check
                    cmd = ["safety", "check", "-r", str(req_file), "--json"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                    if proc.returncode == 0:
                        continue  # No vulnerabilities found

                    # Parse safety output
                    try:
                        safety_data = json.loads(proc.stdout)
                        for vuln in safety_data:
                            vulnerability = SecurityVulnerability(
                                id=f"dep_vuln_{vuln.get('id', 'unknown')}",
                                title=f"Vulnerable dependency: {vuln.get('package_name')}",
                                description=vuln.get("advisory", "Known vulnerability in dependency"),
                                category=VulnerabilityCategory.KNOWN_VULNERABILITIES,
                                level=VulnerabilityLevel.HIGH,
                                file_path=str(req_file.relative_to(self.project_root)),
                                recommendation=f"Update {vuln.get('package_name')} to version {vuln.get('safe_versions', 'latest')}",
                                cwe_id="CWE-1035",
                            )
                            result.add_vulnerability(vulnerability)
                    except json.JSONDecodeError:
                        pass

                except subprocess.TimeoutExpired:
                    logger.warning("Safety check timed out")
                except FileNotFoundError:
                    # Safety not installed
                    vulnerability = SecurityVulnerability(
                        id="missing_safety_tool",
                        title="Missing dependency vulnerability scanner",
                        description="Safety tool not installed for dependency scanning",
                        category=VulnerabilityCategory.SECURITY_MISCONFIG,
                        level=VulnerabilityLevel.INFO,
                        file_path="requirements.txt",
                        recommendation="Install 'safety' package for dependency vulnerability scanning",
                    )
                    result.add_vulnerability(vulnerability)
                    break

        except Exception as e:
            logger.error(f"Error scanning dependencies: {e}")

    def _check_security_headers(self, result: SecurityScanResult):
        """Check for security headers implementation"""
        # Look for security middleware or header configuration
        middleware_files = list(self.project_root.glob("**/middleware*.py"))

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
        ]

        found_headers = set()

        for file_path in middleware_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for header in required_headers:
                    if header in content:
                        found_headers.add(header)

            except Exception as e:
                logger.debug(f"Error checking headers in {file_path}: {e}")

        # Report missing security headers
        for header in required_headers:
            if header not in found_headers:
                vulnerability = SecurityVulnerability(
                    id=f"missing_header_{header.lower().replace('-', '_')}",
                    title=f"Missing security header: {header}",
                    description=f"Security header {header} not implemented",
                    category=VulnerabilityCategory.SECURITY_MISCONFIG,
                    level=VulnerabilityLevel.MEDIUM,
                    file_path="api/middleware.py",
                    recommendation=f"Implement {header} security header",
                    cwe_id="CWE-693",
                )
                result.add_vulnerability(vulnerability)

    def _check_authentication(self, result: SecurityScanResult):
        """Check authentication implementation"""
        auth_files = list(self.project_root.glob("**/auth*.py"))

        security_checks = {
            "rate_limiting": {
                "patterns": ["rate_limit", "slowapi", "limiter"],
                "title": "Rate limiting implementation",
                "missing_level": VulnerabilityLevel.MEDIUM,
            },
            "session_security": {
                "patterns": ["secure=True", "httponly=True", "samesite"],
                "title": "Secure session configuration",
                "missing_level": VulnerabilityLevel.MEDIUM,
            },
            "password_hashing": {
                "patterns": ["bcrypt", "scrypt", "argon2", "pbkdf2"],
                "title": "Secure password hashing",
                "missing_level": VulnerabilityLevel.HIGH,
            },
        }

        for check_name, check_config in security_checks.items():
            found = False

            for file_path in auth_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if any(pattern in content.lower() for pattern in check_config["patterns"]):
                        found = True
                        break

                except Exception as e:
                    logger.debug(f"Error checking auth file {file_path}: {e}")

            if not found:
                vulnerability = SecurityVulnerability(
                    id=f"missing_{check_name}",
                    title=f"Missing {check_config['title']}",
                    description=f"{check_config['title']} not found in authentication code",
                    category=VulnerabilityCategory.BROKEN_AUTH,
                    level=check_config["missing_level"],
                    file_path="auth/",
                    recommendation=f"Implement {check_config['title']}",
                    cwe_id="CWE-287",
                )
                result.add_vulnerability(vulnerability)

    def generate_report(self, result: SecurityScanResult, format: str = "json") -> str:
        """
        Generate security report

        Args:
            result: Security scan result
            format: Report format (json, html, txt)

        Returns:
            Report content
        """
        if format == "json":
            return json.dumps(result.to_dict(), indent=2, default=str)

        elif format == "html":
            return self._generate_html_report(result)

        elif format == "txt":
            return self._generate_text_report(result)

        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_html_report(self, result: SecurityScanResult) -> str:
        """Generate HTML security report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Scan Report - {result.scan_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .critical {{ color: #dc3545; }}
                .high {{ color: #fd7e14; }}
                .medium {{ color: #ffc107; }}
                .low {{ color: #28a745; }}
                .info {{ color: #17a2b8; }}
                .vulnerability {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
                .summary {{ background: #f8f9fa; padding: 15px; margin-bottom: 20px; }}
                pre {{ background: #f8f9fa; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>Security Scan Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Scan ID:</strong> {result.scan_id}</p>
                <p><strong>Started:</strong> {result.started_at}</p>
                <p><strong>Completed:</strong> {result.completed_at}</p>
                <p><strong>Total Vulnerabilities:</strong> {len(result.vulnerabilities)}</p>
                <ul>
        """

        for level, count in result.summary.items():
            html += f'<li class="{level}"><strong>{level.title()}:</strong> {count}</li>'

        html += """
                </ul>
            </div>
            <h2>Vulnerabilities</h2>
        """

        for vuln in result.vulnerabilities:
            html += f"""
            <div class="vulnerability">
                <h3 class="{vuln.level.value}">{vuln.title}</h3>
                <p><strong>Level:</strong> <span class="{vuln.level.value}">{vuln.level.value.upper()}</span></p>
                <p><strong>Category:</strong> {vuln.category.value}</p>
                <p><strong>File:</strong> {vuln.file_path}:{vuln.line_number or 'N/A'}</p>
                <p><strong>Description:</strong> {vuln.description}</p>
                {f'<p><strong>CWE:</strong> {vuln.cwe_id}</p>' if vuln.cwe_id else ''}
                {f'<pre><code>{vuln.code_snippet}</code></pre>' if vuln.code_snippet else ''}
                <p><strong>Recommendation:</strong> {vuln.recommendation}</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _generate_text_report(self, result: SecurityScanResult) -> str:
        """Generate text security report"""
        report = f"""
SECURITY SCAN REPORT
====================

Scan ID: {result.scan_id}
Started: {result.started_at}
Completed: {result.completed_at}
Total Vulnerabilities: {len(result.vulnerabilities)}

SUMMARY:
"""

        for level, count in result.summary.items():
            report += f"  {level.upper()}: {count}\n"

        report += "\nVULNERABILITIES:\n" + "=" * 50 + "\n"

        for i, vuln in enumerate(result.vulnerabilities, 1):
            report += f"""
{i}. {vuln.title}
   Level: {vuln.level.value.upper()}
   Category: {vuln.category.value}
   File: {vuln.file_path}:{vuln.line_number or 'N/A'}
   Description: {vuln.description}
   {f'CWE: {vuln.cwe_id}' if vuln.cwe_id else ''}
   Recommendation: {vuln.recommendation}
   
"""

        return report


def run_security_scan(project_root: str = ".", output_file: Optional[str] = None) -> SecurityScanResult:
    """
    Run security scan and optionally save report

    Args:
        project_root: Project root directory
        output_file: Optional output file for report

    Returns:
        Security scan result
    """
    scanner = SecurityScanner(project_root)
    result = scanner.scan_project()

    if output_file:
        # Determine format from file extension
        format = "json"
        if output_file.endswith(".html"):
            format = "html"
        elif output_file.endswith(".txt"):
            format = "txt"

        report = scanner.generate_report(result, format)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Security report saved to {output_file}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mimir Security Scanner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["json", "html", "txt"], default="json", help="Report format")
    parser.add_argument("--include-tests", action="store_true", help="Include test files in scan")

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root)
    result = scanner.scan_project(include_tests=args.include_tests)

    if args.output:
        report = scanner.generate_report(result, args.format)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(scanner.generate_report(result, "txt"))
