#!/usr/bin/env python3
"""
Common Issue Fix Script for Mimir Enterprise

Automatically detects and fixes common performance and reliability issues.
"""

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Analyzes code for common issues"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues_found = []

        # Common issue patterns
        self.issue_patterns = {
            "n_plus_one_query": [r"for.*in.*:.*\.query\(", r"for.*in.*:.*\.get\(", r"for.*in.*:.*\.filter\("],
            "missing_async_await": [
                r"def\s+\w+.*\):\s*\n.*client\.",
                r"def\s+\w+.*\):\s*\n.*httpx\.",
                r"def\s+\w+.*\):\s*\n.*aiohttp\.",
            ],
            "hardcoded_config": [r'host\s*=\s*["\']localhost["\']', r"port\s*=\s*8000", r"debug\s*=\s*True"],
            "inefficient_imports": [r"import \*", r"from .* import \*"],
            "missing_error_handling": [r"requests\.get\(", r"requests\.post\(", r"json\.loads\(", r"open\("],
            "sql_injection_risk": [r'f".*{.*}.*".*execute', r"format\(.*\).*execute", r"%.*%.*execute"],
        }

    def analyze_python_files(self) -> List[Dict[str, Any]]:
        """Analyze Python files for common issues"""
        issues = []

        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                file_issues = self._analyze_file_content(file_path, content)
                issues.extend(file_issues)

            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")

        return issues

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ["__pycache__", ".git", "node_modules", ".venv", "venv", "migrations", "test_", "_test.py"]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file_content(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Analyze file content for issues"""
        issues = []
        lines = content.split("\n")

        # Pattern-based analysis
        for issue_type, patterns in self.issue_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(
                            {
                                "type": issue_type,
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_num,
                                "content": line.strip(),
                                "severity": self._get_severity(issue_type),
                                "fix_suggestion": self._get_fix_suggestion(issue_type),
                            }
                        )

        # AST-based analysis
        try:
            tree = ast.parse(content)
            ast_issues = self._analyze_ast(file_path, tree)
            issues.extend(ast_issues)
        except SyntaxError:
            pass  # Skip files with syntax errors

        return issues

    def _analyze_ast(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """AST-based analysis for more complex patterns"""
        issues = []

        for node in ast.walk(tree):
            # Check for synchronous functions that should be async
            if isinstance(node, ast.FunctionDef):
                if self._should_be_async(node):
                    issues.append(
                        {
                            "type": "missing_async",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": node.lineno,
                            "content": f"def {node.name}(...)",
                            "severity": "medium",
                            "fix_suggestion": f"Consider making '{node.name}' async if it performs I/O operations",
                        }
                    )

            # Check for inefficient exception handling
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:  # bare except:
                    issues.append(
                        {
                            "type": "bare_except",
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": node.lineno,
                            "content": "except:",
                            "severity": "medium",
                            "fix_suggestion": "Use specific exception types instead of bare except",
                        }
                    )

        return issues

    def _should_be_async(self, func_node: ast.FunctionDef) -> bool:
        """Check if function should be async based on its content"""
        async_indicators = [
            "client.",
            "requests.",
            "httpx.",
            "aiohttp.",
            "database.",
            ".execute(",
            ".fetch(",
            ".query(",
        ]

        # Convert function to string and check for async indicators
        func_str = ast.unparse(func_node).lower()
        return any(indicator in func_str for indicator in async_indicators)

    def _get_severity(self, issue_type: str) -> str:
        """Get severity level for issue type"""
        severity_map = {
            "sql_injection_risk": "critical",
            "n_plus_one_query": "high",
            "missing_error_handling": "high",
            "missing_async_await": "medium",
            "hardcoded_config": "medium",
            "inefficient_imports": "low",
            "bare_except": "medium",
        }
        return severity_map.get(issue_type, "low")

    def _get_fix_suggestion(self, issue_type: str) -> str:
        """Get fix suggestion for issue type"""
        suggestions = {
            "sql_injection_risk": "Use parameterized queries with placeholders instead of string formatting",
            "n_plus_one_query": "Use bulk queries or joins to fetch related data efficiently",
            "missing_error_handling": "Add try-except blocks to handle potential exceptions",
            "missing_async_await": "Convert function to async and use await for I/O operations",
            "hardcoded_config": "Move configuration to environment variables or config files",
            "inefficient_imports": "Import specific functions/classes instead of using wildcard imports",
            "bare_except": "Catch specific exception types for better error handling",
        }
        return suggestions.get(issue_type, "Review and optimize this code")


class AutoFixer:
    """Automatically fixes common issues"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.fixes_applied = []

    def fix_imports(self) -> List[str]:
        """Fix common import issues"""
        fixes = []
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Fix unused imports (basic detection)
                content = self._remove_unused_imports(content)

                # Sort imports
                content = self._sort_imports(content)

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixes.append(f"Fixed imports in {file_path.relative_to(self.project_root)}")

            except Exception as e:
                logger.error(f"Error fixing imports in {file_path}: {e}")

        return fixes

    def fix_formatting(self) -> List[str]:
        """Fix code formatting issues"""
        fixes = []

        try:
            # Run black formatter
            result = subprocess.run(
                ["black", "--line-length", "120", str(self.project_root)], capture_output=True, text=True
            )

            if result.returncode == 0:
                fixes.append("Applied black code formatting")
            else:
                logger.warning(f"Black formatting failed: {result.stderr}")

        except FileNotFoundError:
            logger.warning("Black formatter not found, skipping formatting")

        return fixes

    def fix_security_issues(self) -> List[str]:
        """Fix basic security issues"""
        fixes = []
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Fix hardcoded debug=os.getenv("DEBUG", "false").lower() == "true"
                content = re.sub(r"debug\s*=\s*True", 'debug=os.getenv("DEBUG", "false").lower() == "true"', content)

                # Add missing imports if we added os.getenv
                if "os.getenv" in content and "import os" not in content:
                    lines = content.split("\n")
                    # Find the last import line
                    import_line = 0
                    for i, line in enumerate(lines):
                        if line.strip().startswith("import ") or line.strip().startswith("from "):
                            import_line = i

                    lines.insert(import_line + 1, "import os")
                    content = "\n".join(lines)

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixes.append(f"Fixed security issues in {file_path.relative_to(self.project_root)}")

            except Exception as e:
                logger.error(f"Error fixing security issues in {file_path}: {e}")

        return fixes

    def fix_performance_issues(self) -> List[str]:
        """Fix basic performance issues"""
        fixes = []

        # Add database connection pooling configuration
        config_fixes = self._add_connection_pooling()
        fixes.extend(config_fixes)

        # Optimize database queries
        query_fixes = self._optimize_queries()
        fixes.extend(query_fixes)

        return fixes

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ["__pycache__", ".git", "node_modules", ".venv", "venv", "migrations"]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _remove_unused_imports(self, content: str) -> str:
        """Remove unused imports (basic implementation)"""
        lines = content.split("\n")
        used_imports = set()
        import_lines = []

        # Find all imports
        for i, line in enumerate(lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                import_lines.append((i, line))

        # Simple check: if import name appears elsewhere in file
        for i, import_line in import_lines:
            import_name = self._extract_import_name(import_line)
            if import_name and import_name in content:
                used_imports.add(i)

        # Keep only used imports
        filtered_lines = []
        for i, line in enumerate(lines):
            if i in [idx for idx, _ in import_lines] and i not in used_imports:
                continue  # Skip unused import
            filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _extract_import_name(self, import_line: str) -> Optional[str]:
        """Extract the imported name from import line"""
        # Simple extraction
        if "import " in import_line:
            parts = import_line.split("import ")
            if len(parts) > 1:
                return parts[1].split()[0].split(".")[0]
        return None

    def _sort_imports(self, content: str) -> str:
        """Sort imports (basic implementation)"""
        try:
            # Use isort if available
            result = subprocess.run(["isort", "--stdout", "-"], input=content, capture_output=True, text=True)

            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            pass  # isort not available

        return content

    def _add_connection_pooling(self) -> List[str]:
        """Add database connection pooling configuration"""
        fixes = []

        # Look for database configuration files
        config_files = list(self.project_root.glob("**/config*.py"))
        config_files.extend(list(self.project_root.glob("**/settings*.py")))

        for config_file in config_files:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if "pool_size" not in content and "database" in content.lower():
                    # Add basic pooling configuration
                    pooling_config = """
# Database connection pooling
DATABASE_POOL_SIZE = int(os.getenv('DATABASE_POOL_SIZE', '20'))
DATABASE_MAX_OVERFLOW = int(os.getenv('DATABASE_MAX_OVERFLOW', '30'))
DATABASE_POOL_TIMEOUT = int(os.getenv('DATABASE_POOL_TIMEOUT', '30'))
"""

                    with open(config_file, "a", encoding="utf-8") as f:
                        f.write(pooling_config)

                    fixes.append(f"Added connection pooling config to {config_file.relative_to(self.project_root)}")

            except Exception as e:
                logger.error(f"Error adding pooling config to {config_file}: {e}")

        return fixes

    def _optimize_queries(self) -> List[str]:
        """Optimize database queries"""
        fixes = []

        # Look for files with database queries
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # Add select_related/prefetch_related hints (for Django-style ORMs)
                content = re.sub(r"\.filter\(", ".select_related().filter(", content)

                # Convert SELECT * to specific columns (basic pattern)
                content = re.sub(
                    r"SELECT \*",
                    "SELECT id, title, content, created_at",  # Example columns
                    content,
                    flags=re.IGNORECASE,
                )

                if content != original_content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    fixes.append(f"Optimized queries in {file_path.relative_to(self.project_root)}")

            except Exception as e:
                logger.error(f"Error optimizing queries in {file_path}: {e}")

        return fixes


class IssueReporter:
    """Generates issue reports"""

    def __init__(self):
        pass

    def generate_report(self, issues: List[Dict[str, Any]], fixes: List[str], output_format: str = "json") -> str:
        """Generate issue report"""

        # Categorize issues
        categorized = self._categorize_issues(issues)

        report_data = {
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "summary": {
                "total_issues": len(issues),
                "critical": len([i for i in issues if i["severity"] == "critical"]),
                "high": len([i for i in issues if i["severity"] == "high"]),
                "medium": len([i for i in issues if i["severity"] == "medium"]),
                "low": len([i for i in issues if i["severity"] == "low"]),
                "fixes_applied": len(fixes),
            },
            "issues_by_category": categorized,
            "fixes_applied": fixes,
            "recommendations": self._get_recommendations(issues),
        }

        if output_format == "json":
            return json.dumps(report_data, indent=2)
        elif output_format == "html":
            return self._generate_html_report(report_data)
        else:
            return self._generate_text_report(report_data)

    def _categorize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Categorize issues by type"""
        categorized = {}

        for issue in issues:
            category = issue["type"]
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(issue)

        return categorized

    def _get_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []

        # Count issues by type
        issue_counts = {}
        for issue in issues:
            issue_type = issue["type"]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Generate recommendations
        if issue_counts.get("n_plus_one_query", 0) > 5:
            recommendations.append("Consider implementing query optimization and caching strategies")

        if issue_counts.get("missing_async_await", 0) > 3:
            recommendations.append("Migrate I/O operations to async/await pattern for better performance")

        if issue_counts.get("sql_injection_risk", 0) > 0:
            recommendations.append("URGENT: Fix SQL injection vulnerabilities immediately")

        if issue_counts.get("hardcoded_config", 0) > 2:
            recommendations.append("Implement proper configuration management with environment variables")

        return recommendations

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; margin-bottom: 20px; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #28a745; }}
        .issue {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
        pre {{ background: #f8f9fa; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Code Quality Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Issues: {data['summary']['total_issues']}</p>
        <p><span class="critical">Critical: {data['summary']['critical']}</span></p>
        <p><span class="high">High: {data['summary']['high']}</span></p>
        <p><span class="medium">Medium: {data['summary']['medium']}</span></p>
        <p><span class="low">Low: {data['summary']['low']}</span></p>
        <p>Fixes Applied: {data['summary']['fixes_applied']}</p>
    </div>
    
    <h2>Recommendations</h2>
    <ul>
"""

        for rec in data["recommendations"]:
            html += f"<li>{rec}</li>"

        html += "</ul></body></html>"
        return html

    def _generate_text_report(self, data: Dict[str, Any]) -> str:
        """Generate text report"""
        report = f"""
CODE QUALITY REPORT
==================

Summary:
  Total Issues: {data['summary']['total_issues']}
  Critical: {data['summary']['critical']}
  High: {data['summary']['high']}
  Medium: {data['summary']['medium']}
  Low: {data['summary']['low']}
  Fixes Applied: {data['summary']['fixes_applied']}

Recommendations:
"""

        for rec in data["recommendations"]:
            report += f"  - {rec}\n"

        return report


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Mimir Common Issues Fixer")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["json", "html", "txt"], default="json", help="Report format")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code for issues")
    analyze_parser.add_argument("--no-fix", action="store_true", help="Only analyze, do not fix")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix common issues")
    fix_parser.add_argument(
        "--type",
        choices=["imports", "formatting", "security", "performance", "all"],
        default="all",
        help="Type of fixes to apply",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "analyze":
        analyzer = CodeAnalyzer(args.project_root)
        issues = analyzer.analyze_python_files()

        fixes = []
        if not args.no_fix:
            fixer = AutoFixer(args.project_root)
            fixes = fixer.fix_imports()
            fixes.extend(fixer.fix_security_issues())

        # Generate report
        reporter = IssueReporter()
        report = reporter.generate_report(issues, fixes, args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

    elif args.command == "fix":
        fixer = AutoFixer(args.project_root)
        all_fixes = []

        if args.type in ["imports", "all"]:
            fixes = fixer.fix_imports()
            all_fixes.extend(fixes)

        if args.type in ["formatting", "all"]:
            fixes = fixer.fix_formatting()
            all_fixes.extend(fixes)

        if args.type in ["security", "all"]:
            fixes = fixer.fix_security_issues()
            all_fixes.extend(fixes)

        if args.type in ["performance", "all"]:
            fixes = fixer.fix_performance_issues()
            all_fixes.extend(fixes)

        print(f"Applied {len(all_fixes)} fixes:")
        for fix in all_fixes:
            print(f"  - {fix}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
