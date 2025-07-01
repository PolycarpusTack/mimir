#!/usr/bin/env python3
"""Code quality and technical debt assessment for Mimir PostgreSQL implementation."""

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


class CodeQualityChecker:
    """Analyze code quality and identify technical debt."""

    def __init__(self):
        self.issues = {"critical": [], "high": [], "medium": [], "low": []}
        self.metrics = {
            "files_analyzed": 0,
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "docstring_coverage": 0,
            "type_hint_coverage": 0,
            "complexity_scores": [],
        }

    def analyze_file(self, filepath: Path) -> None:
        """Analyze a single Python file."""
        if not filepath.suffix == ".py":
            return

        self.metrics["files_analyzed"] += 1

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        self.metrics["total_lines"] += len(lines)

        # Count code vs comment lines
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                self.metrics["comment_lines"] += 1
            elif stripped:
                self.metrics["code_lines"] += 1

        # Parse AST for deeper analysis
        try:
            tree = ast.parse(content)
            self.analyze_ast(tree, filepath)
        except SyntaxError as e:
            self.issues["critical"].append(f"Syntax error in {filepath}: {e}")

    def analyze_ast(self, tree: ast.AST, filepath: Path) -> None:
        """Analyze Abstract Syntax Tree for code quality metrics."""
        # Check for docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self.issues["medium"].append(f"Missing docstring: {filepath}:{node.lineno} - {node.name}")

                # Check for type hints
                if isinstance(node, ast.FunctionDef):
                    if not node.returns and node.name != "__init__":
                        self.issues["low"].append(f"Missing return type hint: {filepath}:{node.lineno} - {node.name}")

                    for arg in node.args.args:
                        if not arg.annotation and arg.arg != "self":
                            self.issues["low"].append(
                                f"Missing parameter type hint: {filepath}:{node.lineno} - {node.name}({arg.arg})"
                            )

                # Calculate cyclomatic complexity
                complexity = self.calculate_complexity(node)
                if complexity > 15:
                    self.issues["high"].append(
                        f"High complexity ({complexity}): {filepath}:{node.lineno} - {node.name}"
                    )
                elif complexity > 10:
                    self.issues["medium"].append(
                        f"Medium complexity ({complexity}): {filepath}:{node.lineno} - {node.name}"
                    )
                self.metrics["complexity_scores"].append(complexity)

    def calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function/method."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def check_security_issues(self, filepath: Path) -> None:
        """Check for common security issues."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for hardcoded passwords
        password_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'PASSWORD\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
        ]

        for pattern in password_patterns:
            if re.search(pattern, content) and "os.getenv" not in content:
                self.issues["critical"].append(f"Potential hardcoded credential in {filepath}")

        # Check for SQL injection vulnerabilities
        if "cursor.execute" in content:
            # Look for string formatting in SQL queries
            sql_patterns = [
                r'cursor\.execute\s*\(\s*["\'].*%[s|d].*["\'].*%',
                r'cursor\.execute\s*\(\s*f["\']',
                r'cursor\.execute\s*\(\s*["\'].*\.format\(',
            ]

            for pattern in sql_patterns:
                if re.search(pattern, content):
                    self.issues["critical"].append(f"Potential SQL injection vulnerability in {filepath}")

    def check_error_handling(self, filepath: Path) -> None:
        """Check for proper error handling."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for bare except clauses
        if re.search(r"except\s*:", content):
            self.issues["medium"].append(f"Bare except clause found in {filepath}")

        # Check for missing error logging
        if "except" in content and "logger" not in content:
            self.issues["low"].append(f"Exception handling without logging in {filepath}")

    def check_postgresql_specific(self) -> None:
        """Check PostgreSQL-specific implementation quality."""
        pg_file = Path("db_manager_postgres.py")
        if not pg_file.exists():
            return

        with open(pg_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for connection pool usage
        if "ThreadedConnectionPool" not in content:
            self.issues["high"].append("No connection pooling implemented in PostgreSQL module")

        # Check for prepared statements
        if "%s" not in content:
            self.issues["medium"].append("Not using parameterized queries in PostgreSQL module")

        # Check for transaction management
        if "commit()" not in content or "rollback()" not in content:
            self.issues["medium"].append("Missing explicit transaction management")

        # Check for proper index usage hints
        if "CREATE INDEX" not in content and "idx_" not in content:
            self.issues["low"].append("No index creation or usage hints found")

    def check_migration_safety(self) -> None:
        """Check migration script for safety."""
        migration_file = Path("migrate_to_postgres.py")
        if not migration_file.exists():
            return

        with open(migration_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for data validation
        if "validate" not in content.lower():
            self.issues["high"].append("No data validation in migration script")

        # Check for rollback capability
        if "rollback" not in content.lower():
            self.issues["high"].append("No rollback mechanism in migration script")

        # Check for progress tracking
        if "progress" not in content.lower():
            self.issues["medium"].append("No progress tracking in migration script")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        # Calculate metrics
        if self.metrics["code_lines"] > 0:
            self.metrics["comment_ratio"] = self.metrics["comment_lines"] / self.metrics["code_lines"]
        else:
            self.metrics["comment_ratio"] = 0

        if self.metrics["complexity_scores"]:
            self.metrics["avg_complexity"] = sum(self.metrics["complexity_scores"]) / len(
                self.metrics["complexity_scores"]
            )
        else:
            self.metrics["avg_complexity"] = 0

        # Count issues by severity
        issue_counts = {
            "critical": len(self.issues["critical"]),
            "high": len(self.issues["high"]),
            "medium": len(self.issues["medium"]),
            "low": len(self.issues["low"]),
        }

        # Technical debt score (0-100, lower is better)
        debt_score = (
            issue_counts["critical"] * 25
            + issue_counts["high"] * 10
            + issue_counts["medium"] * 3
            + issue_counts["low"] * 1
        )

        return {
            "metrics": self.metrics,
            "issues": self.issues,
            "issue_counts": issue_counts,
            "technical_debt_score": min(debt_score, 100),
            "quality_grade": self._calculate_grade(debt_score),
        }

    def _calculate_grade(self, score: int) -> str:
        """Calculate quality grade based on debt score."""
        if score <= 10:
            return "A"
        elif score <= 25:
            return "B"
        elif score <= 50:
            return "C"
        elif score <= 75:
            return "D"
        else:
            return "F"


def check_test_coverage():
    """Check test coverage for PostgreSQL implementation."""
    test_file = Path("tests/test_db_postgres.py")
    db_file = Path("db_manager_postgres.py")

    if not test_file.exists():
        return {"coverage": 0, "missing_tests": ["No test file found"]}

    # Parse main module to find functions
    with open(db_file, "r") as f:
        tree = ast.parse(f.read())

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith("_"):
                functions.append(node.name)

    # Parse test file to find tested functions
    with open(test_file, "r") as f:
        test_content = f.read()

    tested = []
    for func in functions:
        if func in test_content:
            tested.append(func)

    missing = list(set(functions) - set(tested))
    coverage = len(tested) / len(functions) * 100 if functions else 0

    return {
        "coverage": coverage,
        "total_functions": len(functions),
        "tested_functions": len(tested),
        "missing_tests": missing,
    }


def check_documentation():
    """Check documentation completeness."""
    required_docs = ["docs/postgres-setup.md", "docs/postgresql-migration-guide.md", "README.md", "CLAUDE.md"]

    missing_docs = []
    outdated_docs = []

    for doc in required_docs:
        doc_path = Path(doc)
        if not doc_path.exists():
            missing_docs.append(doc)
        else:
            with open(doc_path, "r") as f:
                content = f.read().lower()

            # Check if PostgreSQL is mentioned in general docs
            if "postgresql" not in content and doc in ["README.md", "CLAUDE.md"]:
                outdated_docs.append(doc)

    return {
        "missing_docs": missing_docs,
        "outdated_docs": outdated_docs,
        "documentation_complete": not missing_docs and not outdated_docs,
    }


def main():
    """Run complete quality check."""
    print("🔍 Mimir PostgreSQL Implementation - Quality Check\n")
    print("=" * 60)

    # Initialize checker
    checker = CodeQualityChecker()

    # Analyze Python files
    files_to_check = ["db_manager_postgres.py", "migrate_to_postgres.py", "db_adapter.py", "test_postgres_migration.py"]

    for filename in files_to_check:
        filepath = Path(filename)
        if filepath.exists():
            print(f"Analyzing {filename}...")
            checker.analyze_file(filepath)
            checker.check_security_issues(filepath)
            checker.check_error_handling(filepath)

    # PostgreSQL-specific checks
    checker.check_postgresql_specific()
    checker.check_migration_safety()

    # Generate report
    report = checker.generate_report()

    # Display results
    print("\n📊 Code Metrics:")
    print(f"  Files analyzed: {report['metrics']['files_analyzed']}")
    print(f"  Total lines: {report['metrics']['total_lines']}")
    print(f"  Code lines: {report['metrics']['code_lines']}")
    print(f"  Comment lines: {report['metrics']['comment_lines']}")
    print(f"  Comment ratio: {report['metrics']['comment_ratio']:.2%}")
    print(f"  Average complexity: {report['metrics']['avg_complexity']:.1f}")

    print("\n🐛 Issues Found:")
    for severity in ["critical", "high", "medium", "low"]:
        count = report["issue_counts"][severity]
        if count > 0:
            print(f"\n  {severity.upper()} ({count}):")
            for issue in report["issues"][severity][:5]:  # Show first 5
                print(f"    - {issue}")
            if count > 5:
                print(f"    ... and {count - 5} more")

    # Test coverage
    print("\n🧪 Test Coverage:")
    coverage_report = check_test_coverage()
    print(f"  Function coverage: {coverage_report['coverage']:.1f}%")
    print(f"  Functions tested: {coverage_report['tested_functions']}/{coverage_report['total_functions']}")
    if coverage_report["missing_tests"]:
        print(f"  Missing tests for: {', '.join(coverage_report['missing_tests'][:5])}")

    # Documentation
    print("\n📚 Documentation:")
    doc_report = check_documentation()
    if doc_report["documentation_complete"]:
        print("  ✅ All documentation is complete and up-to-date")
    else:
        if doc_report["missing_docs"]:
            print(f"  ❌ Missing: {', '.join(doc_report['missing_docs'])}")
        if doc_report["outdated_docs"]:
            print(f"  ⚠️  Outdated: {', '.join(doc_report['outdated_docs'])}")

    # Final score
    print("\n🎯 Overall Assessment:")
    print(f"  Technical Debt Score: {report['technical_debt_score']}/100")
    print(f"  Quality Grade: {report['quality_grade']}")

    # Recommendations
    print("\n💡 Recommendations:")
    if report["issue_counts"]["critical"] > 0:
        print("  1. Fix critical security/stability issues immediately")
    if report["issue_counts"]["high"] > 0:
        print("  2. Address high-priority issues before production")
    if coverage_report["coverage"] < 80:
        print("  3. Increase test coverage to at least 80%")
    if report["metrics"]["avg_complexity"] > 10:
        print("  4. Refactor complex functions to reduce complexity")

    # Save detailed report
    with open("quality_report.json", "w") as f:
        json.dump(
            {
                "timestamp": str(datetime.now()),
                "report": report,
                "test_coverage": coverage_report,
                "documentation": doc_report,
            },
            f,
            indent=2,
        )

    print("\n✅ Detailed report saved to quality_report.json")

    # Exit code based on critical issues
    sys.exit(1 if report["issue_counts"]["critical"] > 0 else 0)


if __name__ == "__main__":
    from datetime import datetime

    main()
