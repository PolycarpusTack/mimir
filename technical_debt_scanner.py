"""
Technical Debt Scanner and Auto-Fix Tool
For Mimir News Intelligence Platform

This tool performs comprehensive technical debt analysis and implements
automated fixes for common issues:
- Code quality issues
- Security vulnerabilities  
- Performance bottlenecks
- Documentation gaps
- Test coverage issues
- Dependency problems

Author: Claude Code
"""

import ast
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TechnicalDebtIssue:
    """Data class for technical debt issues."""

    category: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: int
    description: str
    suggested_fix: str
    auto_fixable: bool
    code_snippet: str = ""


class TechnicalDebtScanner:
    """
    Comprehensive technical debt scanner and auto-fix tool.
    """

    def __init__(self, project_root: str = "."):
        """
        Initialize the technical debt scanner.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root).resolve()
        self.issues: List[TechnicalDebtIssue] = []
        self.stats = {
            "files_scanned": 0,
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "auto_fixable": 0,
        }

        # File patterns to scan
        self.python_files = list(self.project_root.glob("*.py"))
        self.python_files.extend(self.project_root.glob("**/*.py"))

        # Exclude certain directories
        self.exclude_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "venv",
            "env",
            ".env",
            "build",
            "dist",
            ".tox",
        }

        self.python_files = [
            f for f in self.python_files if not any(pattern in str(f) for pattern in self.exclude_patterns)
        ]

        logger.info(f"Initialized scanner for {len(self.python_files)} Python files")

    def add_issue(
        self,
        category: str,
        severity: str,
        file_path: str,
        line_number: int,
        description: str,
        suggested_fix: str,
        auto_fixable: bool = False,
        code_snippet: str = "",
    ):
        """Add a technical debt issue."""
        issue = TechnicalDebtIssue(
            category=category,
            severity=severity,
            file_path=str(file_path),
            line_number=line_number,
            description=description,
            suggested_fix=suggested_fix,
            auto_fixable=auto_fixable,
            code_snippet=code_snippet,
        )

        self.issues.append(issue)
        self.stats["total_issues"] += 1
        self.stats[f"{severity}_issues"] += 1

        if auto_fixable:
            self.stats["auto_fixable"] += 1

    def scan_code_quality(self):
        """Scan for code quality issues."""
        logger.info("Scanning for code quality issues...")

        for file_path in self.python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                self._check_line_length(file_path, lines)
                self._check_imports(file_path, content, lines)
                self._check_naming_conventions(file_path, content, lines)
                self._check_complexity(file_path, content)
                self._check_docstrings(file_path, content, lines)
                self._check_hardcoded_values(file_path, lines)

                self.stats["files_scanned"] += 1

            except Exception as e:
                logger.error(f"Error scanning {file_path}: {e}")

    def _check_line_length(self, file_path: Path, lines: List[str]):
        """Check for overly long lines."""
        max_length = 120

        for i, line in enumerate(lines, 1):
            if len(line) > max_length:
                self.add_issue(
                    category="Code Style",
                    severity="medium",
                    file_path=str(file_path),
                    line_number=i,
                    description=f"Line too long ({len(line)} characters, max {max_length})",
                    suggested_fix="Break line into multiple lines or use parentheses for continuation",
                    auto_fixable=True,
                    code_snippet=line[:100] + "..." if len(line) > 100 else line,
                )

    def _check_imports(self, file_path: Path, content: str, lines: List[str]):
        """Check import-related issues."""
        try:
            tree = ast.parse(content)
            imports_found = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_found.append((node.lineno, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports_found.append((node.lineno, f"{module}.{alias.name}"))

            # Check for unused imports
            for line_no, import_name in imports_found:
                base_name = import_name.split(".")[0]
                if base_name not in content.replace(f"import {base_name}", ""):
                    # Simple check - might have false positives
                    import_usage_count = content.count(base_name)
                    if import_usage_count <= 1:  # Only the import statement itself
                        self.add_issue(
                            category="Unused Code",
                            severity="low",
                            file_path=str(file_path),
                            line_number=line_no,
                            description=f"Potentially unused import: {import_name}",
                            suggested_fix=f"Remove unused import or add '# noqa' if intentional",
                            auto_fixable=True,
                            code_snippet=lines[line_no - 1] if line_no - 1 < len(lines) else "",
                        )

            # Check for wildcard imports
            for i, line in enumerate(lines, 1):
                if re.search(r"from\s+\w+\s+import\s+\*", line):
                    self.add_issue(
                        category="Code Style",
                        severity="medium",
                        file_path=str(file_path),
                        line_number=i,
                        description="Wildcard import found",
                        suggested_fix="Use explicit imports instead of wildcard imports",
                        auto_fixable=False,
                        code_snippet=line.strip(),
                    )

        except SyntaxError as e:
            self.add_issue(
                category="Syntax Error",
                severity="critical",
                file_path=str(file_path),
                line_number=e.lineno or 1,
                description=f"Syntax error: {e.msg}",
                suggested_fix="Fix syntax error",
                auto_fixable=False,
            )

    def _check_naming_conventions(self, file_path: Path, content: str, lines: List[str]):
        """Check naming convention issues."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check function naming (should be snake_case)
                    if not re.match(r"^[a-z_][a-z0-9_]*$", node.name) and not node.name.startswith("__"):
                        self.add_issue(
                            category="Naming Convention",
                            severity="low",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Function '{node.name}' doesn't follow snake_case convention",
                            suggested_fix="Use snake_case for function names",
                            auto_fixable=True,
                            code_snippet=f"def {node.name}(",
                        )

                elif isinstance(node, ast.ClassDef):
                    # Check class naming (should be PascalCase)
                    if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                        self.add_issue(
                            category="Naming Convention",
                            severity="low",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Class '{node.name}' doesn't follow PascalCase convention",
                            suggested_fix="Use PascalCase for class names",
                            auto_fixable=True,
                            code_snippet=f"class {node.name}:",
                        )

                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    # Check variable naming
                    if re.match(r"^[A-Z_]+$", node.id) and len(node.id) > 1:
                        # Likely a constant, should be at module level
                        pass
                    elif not re.match(r"^[a-z_][a-z0-9_]*$", node.id):
                        self.add_issue(
                            category="Naming Convention",
                            severity="low",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Variable '{node.id}' doesn't follow snake_case convention",
                            suggested_fix="Use snake_case for variable names",
                            auto_fixable=True,
                        )

        except Exception as e:
            logger.warning(f"Error checking naming conventions in {file_path}: {e}")

    def _check_complexity(self, file_path: Path, content: str):
        """Check for overly complex functions."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)

                    if complexity > 15:
                        severity = "high"
                    elif complexity > 10:
                        severity = "medium"
                    elif complexity > 7:
                        severity = "low"
                    else:
                        continue

                    self.add_issue(
                        category="Code Complexity",
                        severity=severity,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' has high cyclomatic complexity ({complexity})",
                        suggested_fix="Consider breaking function into smaller functions",
                        auto_fixable=False,
                        code_snippet=f"def {node.name}(...):",
                    )

        except Exception as e:
            logger.warning(f"Error checking complexity in {file_path}: {e}")

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1

        return complexity

    def _check_docstrings(self, file_path: Path, content: str, lines: List[str]):
        """Check for missing or inadequate docstrings."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Skip private methods and test methods
                    if node.name.startswith("_") or node.name.startswith("test_"):
                        continue

                    docstring = ast.get_docstring(node)

                    if not docstring:
                        self.add_issue(
                            category="Documentation",
                            severity="medium",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Missing docstring for {type(node).__name__.lower()} '{node.name}'",
                            suggested_fix="Add comprehensive docstring with Args, Returns, and description",
                            auto_fixable=True,
                            code_snippet=f"{'class' if isinstance(node, ast.ClassDef) else 'def'} {node.name}",
                        )
                    elif len(docstring.strip()) < 10:
                        self.add_issue(
                            category="Documentation",
                            severity="low",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Inadequate docstring for {type(node).__name__.lower()} '{node.name}'",
                            suggested_fix="Expand docstring with more detailed description",
                            auto_fixable=True,
                            code_snippet=docstring[:50] + "..." if len(docstring) > 50 else docstring,
                        )

        except Exception as e:
            logger.warning(f"Error checking docstrings in {file_path}: {e}")

    def _check_hardcoded_values(self, file_path: Path, lines: List[str]):
        """Check for hardcoded values that should be constants."""
        patterns = [
            (r"\b\d{2,}\b", "Large numeric literal"),
            (r'["\'][^"\']{20,}["\']', "Long string literal"),
            (r'https?://[^\s\'"]+', "Hardcoded URL"),
            (r"/[^/\s]+/[^/\s]+", "Hardcoded file path"),
        ]

        for i, line in enumerate(lines, 1):
            # Skip comments and docstrings
            if line.strip().startswith("#") or '"""' in line or "'''" in line:
                continue

            for pattern, description in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    self.add_issue(
                        category="Code Maintainability",
                        severity="low",
                        file_path=str(file_path),
                        line_number=i,
                        description=f"{description} found: {match.group()}",
                        suggested_fix="Move to configuration file or define as named constant",
                        auto_fixable=False,
                        code_snippet=line.strip(),
                    )

    def scan_security_issues(self):
        """Scan for security vulnerabilities."""
        logger.info("Scanning for security issues...")

        security_patterns = [
            (r"exec\s*\(", "Use of exec() function", "critical"),
            (r"eval\s*\(", "Use of eval() function", "critical"),
            (r"subprocess\.call\s*\([^)]*shell\s*=\s*True", "Shell injection risk", "high"),
            (r"os\.system\s*\(", "Use of os.system()", "high"),
            (r"pickle\.loads?\s*\(", "Unsafe pickle usage", "high"),
            (r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader", "Unsafe YAML loading", "high"),
            (r"random\.random\(\)", "Use of weak random generator", "medium"),
            (r"hashlib\.md5\(\)", "Use of weak hash algorithm", "medium"),
            (r"hashlib\.sha1\(\)", "Use of weak hash algorithm", "medium"),
            (r"requests\.get\([^)]*verify\s*=\s*False", "SSL verification disabled", "high"),
            (
                r"urllib\.request\.urlopen\([^)]*context\s*=\s*ssl\._create_unverified_context",
                "SSL verification disabled",
                "high",
            ),
        ]

        for file_path in self.python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for pattern, description, severity in security_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            self.add_issue(
                                category="Security",
                                severity=severity,
                                file_path=str(file_path),
                                line_number=i,
                                description=description,
                                suggested_fix="Review and use secure alternatives",
                                auto_fixable=False,
                                code_snippet=line.strip(),
                            )

            except Exception as e:
                logger.error(f"Error scanning security in {file_path}: {e}")

    def scan_performance_issues(self):
        """Scan for performance-related issues."""
        logger.info("Scanning for performance issues...")

        performance_patterns = [
            (r"\.append\s*\([^)]*\)\s*$", "List append in loop (consider list comprehension)", "medium"),
            (r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\([^)]+\)\s*\)", "Using range(len()) instead of enumerate", "low"),
            (r"\.keys\(\)\s*:", "Iterating over dict.keys() unnecessarily", "low"),
            (r"len\s*\([^)]+\)\s*==\s*0", "Using len() == 0 instead of 'not'", "low"),
            (r"len\s*\([^)]+\)\s*>\s*0", "Using len() > 0 instead of bool check", "low"),
        ]

        for file_path in self.python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.split("\n")

                for pattern, description, severity in performance_patterns:
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            self.add_issue(
                                category="Performance",
                                severity=severity,
                                file_path=str(file_path),
                                line_number=i,
                                description=description,
                                suggested_fix="Use more efficient alternative",
                                auto_fixable=True,
                                code_snippet=line.strip(),
                            )

            except Exception as e:
                logger.error(f"Error scanning performance in {file_path}: {e}")

    def scan_dependency_issues(self):
        """Scan for dependency-related issues."""
        logger.info("Scanning for dependency issues...")

        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            self._check_requirements_file(requirements_file)

        # Check for missing __init__.py files
        for py_file in self.python_files:
            parent_dir = py_file.parent
            if parent_dir != self.project_root:
                init_file = parent_dir / "__init__.py"
                if not init_file.exists() and not any(p.name.startswith(".") for p in parent_dir.parents):
                    self.add_issue(
                        category="Project Structure",
                        severity="medium",
                        file_path=str(init_file),
                        line_number=1,
                        description=f"Missing __init__.py in {parent_dir.name}",
                        suggested_fix="Add __init__.py file to make directory a Python package",
                        auto_fixable=True,
                    )

    def _check_requirements_file(self, requirements_file: Path):
        """Check requirements.txt for issues."""
        try:
            with open(requirements_file, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Check for unpinned versions
                if not re.search(r"[=<>!]", line):
                    self.add_issue(
                        category="Dependency Management",
                        severity="medium",
                        file_path=str(requirements_file),
                        line_number=i,
                        description=f"Unpinned dependency: {line}",
                        suggested_fix="Pin dependency versions for reproducible builds",
                        auto_fixable=False,
                        code_snippet=line,
                    )

                # Check for outdated syntax
                if "==" in line and line.count("==") > 1:
                    self.add_issue(
                        category="Dependency Management",
                        severity="low",
                        file_path=str(requirements_file),
                        line_number=i,
                        description=f"Multiple version specifiers: {line}",
                        suggested_fix="Use single version specifier",
                        auto_fixable=True,
                        code_snippet=line,
                    )

        except Exception as e:
            logger.error(f"Error checking requirements file: {e}")

    def run_external_tools(self):
        """Run external code quality tools."""
        logger.info("Running external tools...")

        tools_to_run = [
            ("flake8", ["flake8", "--max-line-length=120", "--select=E,W,F", "."]),
            ("mypy", ["mypy", "--ignore-missing-imports", "."]),
            ("bandit", ["bandit", "-r", ".", "-f", "json"]),
        ]

        for tool_name, command in tools_to_run:
            try:
                result = subprocess.run(command, capture_output=True, text=True, cwd=self.project_root, timeout=60)

                if result.returncode != 0 and result.stdout:
                    self._parse_tool_output(tool_name, result.stdout)

            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"Tool {tool_name} not available or timed out")
            except Exception as e:
                logger.error(f"Error running {tool_name}: {e}")

    def _parse_tool_output(self, tool_name: str, output: str):
        """Parse output from external tools."""
        if tool_name == "flake8":
            for line in output.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 4:
                        file_path = parts[0]
                        line_no = int(parts[1]) if parts[1].isdigit() else 1
                        error_code = parts[3].strip().split()[0]
                        description = parts[3].strip()

                        severity = "high" if error_code.startswith("E") else "medium"

                        self.add_issue(
                            category="Code Style",
                            severity=severity,
                            file_path=file_path,
                            line_number=line_no,
                            description=f"Flake8: {description}",
                            suggested_fix="Fix according to PEP 8 guidelines",
                            auto_fixable=True,
                        )

        elif tool_name == "bandit" and output.strip():
            try:
                data = json.loads(output)
                for result in data.get("results", []):
                    self.add_issue(
                        category="Security",
                        severity=result.get("issue_severity", "medium").lower(),
                        file_path=result.get("filename", ""),
                        line_number=result.get("line_number", 1),
                        description=f"Bandit: {result.get('issue_text', '')}",
                        suggested_fix=result.get("issue_text", ""),
                        auto_fixable=False,
                        code_snippet=result.get("code", ""),
                    )
            except json.JSONDecodeError:
                pass

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive technical debt report."""
        issues_by_category = defaultdict(list)
        issues_by_severity = defaultdict(list)
        issues_by_file = defaultdict(list)

        for issue in self.issues:
            issues_by_category[issue.category].append(issue)
            issues_by_severity[issue.severity].append(issue)
            issues_by_file[issue.file_path].append(issue)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_files_scanned": self.stats["files_scanned"],
                "total_issues_found": self.stats["total_issues"],
                "issues_by_severity": {
                    "critical": self.stats["critical_issues"],
                    "high": self.stats["high_issues"],
                    "medium": self.stats["medium_issues"],
                    "low": self.stats["low_issues"],
                },
                "auto_fixable_issues": self.stats["auto_fixable"],
                "technical_debt_score": self._calculate_debt_score(),
            },
            "issues_by_category": {category: len(issues) for category, issues in issues_by_category.items()},
            "top_problematic_files": self._get_top_problematic_files(issues_by_file),
            "detailed_issues": [asdict(issue) for issue in self.issues],
            "recommendations": self._generate_recommendations(issues_by_category),
        }

        return report

    def _calculate_debt_score(self) -> float:
        """Calculate technical debt score (0-100, lower is better)."""
        if self.stats["files_scanned"] == 0:
            return 0

        # Weight issues by severity
        weighted_score = (
            self.stats["critical_issues"] * 10
            + self.stats["high_issues"] * 5
            + self.stats["medium_issues"] * 2
            + self.stats["low_issues"] * 1
        )

        # Normalize by number of files
        normalized_score = weighted_score / self.stats["files_scanned"]

        # Convert to 0-100 scale
        return min(100, normalized_score * 5)

    def _get_top_problematic_files(self, issues_by_file: Dict[str, List]) -> List[Dict[str, Any]]:
        """Get top 10 most problematic files."""
        file_scores = []

        for file_path, file_issues in issues_by_file.items():
            score = sum(
                10
                if issue.severity == "critical"
                else 5
                if issue.severity == "high"
                else 2
                if issue.severity == "medium"
                else 1
                for issue in file_issues
            )

            file_scores.append({"file": file_path, "issues_count": len(file_issues), "severity_score": score})

        return sorted(file_scores, key=lambda x: x["severity_score"], reverse=True)[:10]

    def _generate_recommendations(self, issues_by_category: Dict[str, List]) -> List[str]:
        """Generate recommendations based on found issues."""
        recommendations = []

        if issues_by_category.get("Security"):
            recommendations.append("URGENT: Address security vulnerabilities immediately")

        if issues_by_category.get("Code Complexity"):
            recommendations.append("Consider refactoring complex functions to improve maintainability")

        if issues_by_category.get("Documentation"):
            recommendations.append("Improve code documentation with comprehensive docstrings")

        if issues_by_category.get("Performance"):
            recommendations.append("Optimize performance-critical code sections")

        if issues_by_category.get("Code Style"):
            recommendations.append("Set up automated code formatting (black, autopep8)")

        if self.stats["auto_fixable"] > 10:
            recommendations.append(
                f"Run auto-fix tool to resolve {self.stats['auto_fixable']} automatically fixable issues"
            )

        return recommendations

    def run_complete_scan(self) -> Dict[str, Any]:
        """Run complete technical debt scan."""
        logger.info("Starting comprehensive technical debt scan...")

        # Run all scans
        self.scan_code_quality()
        self.scan_security_issues()
        self.scan_performance_issues()
        self.scan_dependency_issues()
        self.run_external_tools()

        # Generate report
        report = self.generate_report()

        logger.info(
            f"Scan completed. Found {self.stats['total_issues']} issues across {self.stats['files_scanned']} files"
        )
        return report


class TechnicalDebtAutoFixer:
    """
    Automated fix implementation for common technical debt issues.
    """

    def __init__(self, project_root: str = "."):
        """Initialize the auto-fixer."""
        self.project_root = Path(project_root).resolve()
        self.fixes_applied = []

    def apply_auto_fixes(self, issues: List[TechnicalDebtIssue]) -> List[str]:
        """Apply automated fixes for eligible issues."""
        logger.info("Applying automated fixes...")

        # Group issues by file for efficient processing
        issues_by_file = defaultdict(list)
        for issue in issues:
            if issue.auto_fixable:
                issues_by_file[issue.file_path].append(issue)

        for file_path, file_issues in issues_by_file.items():
            self._fix_file_issues(file_path, file_issues)

        return self.fixes_applied

    def _fix_file_issues(self, file_path: str, issues: List[TechnicalDebtIssue]):
        """Fix issues in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Sort issues by line number in reverse order to maintain line numbers
            issues.sort(key=lambda x: x.line_number, reverse=True)

            modified = False

            for issue in issues:
                if self._apply_fix(lines, issue):
                    modified = True
                    self.fixes_applied.append(f"Fixed {issue.category} in {file_path}:{issue.line_number}")

            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                logger.info(f"Applied fixes to {file_path}")

        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")

    def _apply_fix(self, lines: List[str], issue: TechnicalDebtIssue) -> bool:
        """Apply a specific fix to file lines."""
        if issue.line_number > len(lines):
            return False

        line_idx = issue.line_number - 1
        original_line = lines[line_idx]

        # Apply specific fixes based on issue type
        if issue.category == "Code Style" and "Line too long" in issue.description:
            # Simple line breaking for imports and function calls
            if "import" in original_line or "from" in original_line:
                fixed_line = self._fix_long_import(original_line)
                if fixed_line != original_line:
                    lines[line_idx] = fixed_line
                    return True

        elif issue.category == "Performance":
            fixed_line = self._fix_performance_issue(original_line, issue.description)
            if fixed_line != original_line:
                lines[line_idx] = fixed_line
                return True

        elif issue.category == "Documentation" and "Missing docstring" in issue.description:
            # Add basic docstring
            indent = len(original_line) - len(original_line.lstrip())
            if "def " in original_line:
                docstring = " " * (indent + 4) + '"""TODO: Add docstring."""\n'
                lines.insert(line_idx + 1, docstring)
                return True
            elif "class " in original_line:
                docstring = " " * (indent + 4) + '"""TODO: Add class docstring."""\n'
                lines.insert(line_idx + 1, docstring)
                return True

        elif issue.category == "Project Structure" and "Missing __init__.py" in issue.description:
            # Create __init__.py file
            init_file_path = Path(issue.file_path)
            try:
                init_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(init_file_path, "w") as f:
                    f.write('"""Package initialization."""\n')
                return True
            except Exception:
                pass

        return False

    def _fix_long_import(self, line: str) -> str:
        """Fix long import lines."""
        if len(line) <= 120:
            return line

        # Simple fix for from imports
        if line.strip().startswith("from") and "import" in line:
            parts = line.split("import", 1)
            if len(parts) == 2:
                from_part = parts[0] + "import ("
                imports = parts[1].strip().split(",")

                if len(imports) > 1:
                    indent = len(line) - len(line.lstrip())
                    fixed_imports = []
                    for imp in imports:
                        fixed_imports.append(" " * (indent + 4) + imp.strip())

                    return from_part + "\n" + ",\n".join(fixed_imports) + "\n" + " " * indent + ")\n"

        return line

    def _fix_performance_issue(self, line: str, description: str) -> str:
        """Fix performance-related issues."""
        # Fix len() == 0 to not
        if "len() == 0" in description:
            line = re.sub(r"len\s*\(\s*([^)]+)\s*\)\s*==\s*0", r"not \1", line)

        # Fix len() > 0 to bool check
        elif "len() > 0" in description:
            line = re.sub(r"len\s*\(\s*([^)]+)\s*\)\s*>\s*0", r"\1", line)

        # Fix range(len()) to enumerate
        elif "range(len())" in description:
            line = re.sub(
                r"for\s+(\w+)\s+in\s+range\s*\(\s*len\s*\(\s*([^)]+)\s*\)\s*\)", r"for \1, _ in enumerate(\2)", line
            )

        return line


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Technical Debt Scanner and Auto-Fix Tool")
    parser.add_argument("command", choices=["scan", "fix", "report"], help="Command to run")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--auto-fix", action="store_true", help="Apply automatic fixes after scanning")
    parser.add_argument(
        "--severity", choices=["critical", "high", "medium", "low"], help="Minimum severity level to report"
    )

    args = parser.parse_args()

    scanner = TechnicalDebtScanner()

    if args.command == "scan":
        report = scanner.run_complete_scan()

        # Filter by severity if specified
        if args.severity:
            severity_order = ["critical", "high", "medium", "low"]
            min_level = severity_order.index(args.severity)

            filtered_issues = [issue for issue in scanner.issues if severity_order.index(issue.severity) <= min_level]

            print(f"Found {len(filtered_issues)} issues at {args.severity}+ severity")
        else:
            print(f"Found {scanner.stats['total_issues']} total issues")

        print(f"Technical Debt Score: {report['summary']['technical_debt_score']:.1f}/100")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Detailed report saved to {args.output}")

        if args.auto_fix:
            fixer = TechnicalDebtAutoFixer()
            fixes = fixer.apply_auto_fixes(scanner.issues)
            print(f"Applied {len(fixes)} automatic fixes")

    elif args.command == "fix":
        # Run scan first
        scanner.run_complete_scan()

        # Apply fixes
        fixer = TechnicalDebtAutoFixer()
        fixes = fixer.apply_auto_fixes(scanner.issues)

        print(f"Applied {len(fixes)} automatic fixes:")
        for fix in fixes:
            print(f"  - {fix}")

    elif args.command == "report":
        report = scanner.run_complete_scan()

        print("\n" + "=" * 60)
        print("TECHNICAL DEBT REPORT")
        print("=" * 60)
        print(f"Scan Date: {report['timestamp']}")
        print(f"Files Scanned: {report['summary']['total_files_scanned']}")
        print(f"Total Issues: {report['summary']['total_issues_found']}")
        print(f"Technical Debt Score: {report['summary']['technical_debt_score']:.1f}/100")
        print()

        print("Issues by Severity:")
        for severity, count in report["summary"]["issues_by_severity"].items():
            print(f"  {severity.capitalize()}: {count}")
        print()

        print("Issues by Category:")
        for category, count in report["issues_by_category"].items():
            print(f"  {category}: {count}")
        print()

        print("Top Problematic Files:")
        for file_info in report["top_problematic_files"][:5]:
            print(f"  {file_info['file']}: {file_info['issues_count']} issues (score: {file_info['severity_score']})")
        print()

        if report["recommendations"]:
            print("Recommendations:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")


if __name__ == "__main__":
    main()
