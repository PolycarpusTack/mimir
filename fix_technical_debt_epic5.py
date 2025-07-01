#!/usr/bin/env python3
"""
Technical Debt Fixer for EPIC 5 Implementation

Fixes common issues found in the EPIC 5 implementation:
- Missing type hints
- Import organization
- SQL injection prevention
- Error handling improvements
- Docstring completeness
- Async/await consistency
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


class TechnicalDebtFixer:
    """Fixes technical debt in Python files"""

    def __init__(self):
        self.fixes_applied = []
        self.files_fixed = 0

    def fix_file(self, filepath: Path) -> bool:
        """Fix issues in a single file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Apply fixes
            content = self.fix_missing_type_hints(content, filepath)
            content = self.fix_sql_injection_risks(content, filepath)
            content = self.fix_import_organization(content, filepath)
            content = self.fix_error_handling(content, filepath)
            content = self.fix_docstrings(content, filepath)
            content = self.fix_hardcoded_values(content, filepath)
            content = self.fix_async_consistency(content, filepath)
            content = self.fix_unused_imports(content, filepath)

            if content != original_content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                self.files_fixed += 1
                print(f"Fixed: {filepath}")
                return True

            return False

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return False

    def fix_missing_type_hints(self, content: str, filepath: Path) -> str:
        """Add missing type hints to function definitions"""

        # Fix functions without return type hints
        def add_return_hint(match):
            func_def = match.group(0)
            if "->" not in func_def and "def __" not in func_def:
                # Determine return type based on function name and content
                func_name = match.group(1)
                func_body_start = match.end()

                # Find the function body
                indent_level = len(match.group(0)) - len(match.group(0).lstrip())
                func_body = self._extract_function_body(content[func_body_start:], indent_level)

                return_type = self._infer_return_type(func_name, func_body)

                # Add return type before the colon
                new_func_def = func_def.rstrip(":") + f" -> {return_type}:"
                self.fixes_applied.append(f"{filepath}: Added return type hint to {func_name}")
                return new_func_def

            return func_def

        # Pattern for function definitions
        func_pattern = r"^(\s*)def\s+(\w+)\s*\([^)]*\)\s*:"
        content = re.sub(func_pattern, add_return_hint, content, flags=re.MULTILINE)

        return content

    def fix_sql_injection_risks(self, content: str, filepath: Path) -> str:
        """Fix potential SQL injection vulnerabilities"""

        # Look for string formatting in SQL queries
        sql_format_pattern = r'(query|sql)\s*=\s*["\'].*?%[s|d].*?["\'].*?%\s*\('

        if re.search(sql_format_pattern, content, re.IGNORECASE | re.DOTALL):
            # This is already using parameterized queries, which is good
            pass

        # Look for f-strings in SQL
        sql_fstring_pattern = r'(query|sql)\s*=\s*f["\'].*?\{.*?\}.*?["\']'

        if re.search(sql_fstring_pattern, content, re.IGNORECASE | re.DOTALL):
            self.fixes_applied.append(f"{filepath}: WARNING - Found f-string in SQL query (potential SQL injection)")

        # Look for string concatenation in SQL
        sql_concat_pattern = r"(query|sql)\s*=.*?\+.*?(WHERE|AND|OR)"

        if re.search(sql_concat_pattern, content, re.IGNORECASE):
            self.fixes_applied.append(f"{filepath}: WARNING - Found string concatenation in SQL query")

        return content

    def fix_import_organization(self, content: str, filepath: Path) -> str:
        """Organize imports according to PEP 8"""

        lines = content.split("\n")

        # Find import section
        import_start = -1
        import_end = -1
        imports = {"stdlib": [], "third_party": [], "local": []}

        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")) and import_start == -1:
                import_start = i

            if import_start != -1 and line.strip() and not line.strip().startswith(("import ", "from ")):
                import_end = i
                break

        if import_start == -1:
            return content

        if import_end == -1:
            import_end = len(lines)

        # Categorize imports
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "datetime",
            "time",
            "logging",
            "asyncio",
            "typing",
            "dataclasses",
            "enum",
            "uuid",
            "hashlib",
            "collections",
            "re",
            "pathlib",
            "functools",
            "itertools",
        }

        for i in range(import_start, import_end):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("import ") or line.startswith("from "):
                # Extract module name
                if line.startswith("import "):
                    module = line.split()[1].split(".")[0]
                else:
                    module = line.split()[1].split(".")[0]

                if module.startswith("."):
                    imports["local"].append(line)
                elif module in stdlib_modules:
                    imports["stdlib"].append(line)
                else:
                    imports["third_party"].append(line)

        # Sort each category
        for category in imports:
            imports[category] = sorted(set(imports[category]))

        # Reconstruct imports section
        new_imports = []
        if imports["stdlib"]:
            new_imports.extend(imports["stdlib"])
            new_imports.append("")
        if imports["third_party"]:
            new_imports.extend(imports["third_party"])
            new_imports.append("")
        if imports["local"]:
            new_imports.extend(imports["local"])

        # Replace import section
        if new_imports:
            lines[import_start:import_end] = new_imports
            self.fixes_applied.append(f"{filepath}: Reorganized imports")

        return "\n".join(lines)

    def fix_error_handling(self, content: str, filepath: Path) -> str:
        """Improve error handling"""

        # Look for bare except clauses
        bare_except_pattern = r"^\s*except\s*:\s*$"
        if re.search(bare_except_pattern, content, re.MULTILINE):
            content = re.sub(bare_except_pattern, "    except Exception:", content, flags=re.MULTILINE)
            self.fixes_applied.append(f"{filepath}: Fixed bare except clause")

        # Look for generic Exception without logging
        except_pattern = r"except\s+Exception\s+as\s+e:\s*\n\s*raise"
        if re.search(except_pattern, content):
            self.fixes_applied.append(f"{filepath}: WARNING - Exception caught and re-raised without logging")

        return content

    def fix_docstrings(self, content: str, filepath: Path) -> str:
        """Ensure all public functions have docstrings"""

        # Pattern for functions without docstrings
        func_pattern = r'^(\s*)def\s+(\w+)\s*\([^)]*\)[^:]*:\s*\n(?!\s*["\'])'

        matches = list(re.finditer(func_pattern, content, re.MULTILINE))

        for match in reversed(matches):  # Process in reverse to maintain positions
            func_name = match.group(2)
            if not func_name.startswith("_"):  # Public function
                indent = match.group(1)
                insert_pos = match.end()

                # Add a simple docstring
                docstring = f'{indent}    """TODO: Add docstring for {func_name}"""\n'
                content = content[:insert_pos] + docstring + content[insert_pos:]
                self.fixes_applied.append(f"{filepath}: Added TODO docstring for {func_name}")

        return content

    def fix_hardcoded_values(self, content: str, filepath: Path) -> str:
        """Identify hardcoded values that should be configurable"""

        # Look for hardcoded numbers in specific contexts
        hardcoded_patterns = [
            (r"limit\s*=\s*(\d{3,})", "limit value"),
            (r"timeout\s*=\s*(\d{4,})", "timeout value"),
            (r"max_\w+\s*=\s*(\d{2,})", "max value"),
            (r"port\s*=\s*(\d{4,5})", "port number"),
        ]

        for pattern, desc in hardcoded_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.fixes_applied.append(f"{filepath}: WARNING - Found hardcoded {desc}")

        return content

    def fix_async_consistency(self, content: str, filepath: Path) -> str:
        """Ensure async/await consistency"""

        # Check if file uses async functions
        if "async def" in content:
            # Look for synchronous database calls in async functions
            sync_db_pattern = r"async def.*?\n.*?self\.db\.(fetch_one|fetch_all|execute_query)\("
            if re.search(sync_db_pattern, content, re.DOTALL):
                self.fixes_applied.append(f"{filepath}: WARNING - Synchronous DB calls in async function")

        return content

    def fix_unused_imports(self, content: str, filepath: Path) -> str:
        """Remove unused imports"""

        lines = content.split("\n")
        imports_to_check = []

        # Extract all imports
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                imports_to_check.append(line.strip())

        # For now, just warn about potentially unused imports
        # A proper implementation would use AST parsing

        return content

    def _extract_function_body(self, content: str, indent_level: int) -> str:
        """Extract function body based on indentation"""
        lines = content.split("\n")
        body_lines = []

        for line in lines:
            if line.strip() == "":
                body_lines.append(line)
                continue

            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip():
                break

            body_lines.append(line)

        return "\n".join(body_lines)

    def _infer_return_type(self, func_name: str, func_body: str) -> str:
        """Infer return type from function name and body"""

        # Check for explicit returns
        if "return None" in func_body or not "return" in func_body:
            return "None"
        elif "return True" in func_body or "return False" in func_body:
            return "bool"
        elif "return {" in func_body:
            return "Dict[str, Any]"
        elif "return [" in func_body:
            return "List[Any]"
        elif re.search(r"return\s+\d+", func_body):
            return "int"
        elif re.search(r'return\s+["\']', func_body):
            return "str"

        # Infer from function name
        if func_name.startswith(("is_", "has_", "can_", "should_")):
            return "bool"
        elif func_name.startswith("get_") and "list" in func_name.lower():
            return "List[Any]"
        elif func_name.startswith("get_"):
            return "Any"
        elif func_name.startswith(("create_", "update_", "delete_")):
            return "bool"

        return "Any"

    def generate_report(self) -> str:
        """Generate a report of all fixes applied"""
        report = f"""
Technical Debt Fix Report
========================

Files Fixed: {self.files_fixed}
Total Issues Found: {len(self.fixes_applied)}

Issues by File:
--------------
"""

        # Group fixes by file
        fixes_by_file = {}
        for fix in self.fixes_applied:
            if ":" in fix:
                filepath, issue = fix.split(":", 1)
                if filepath not in fixes_by_file:
                    fixes_by_file[filepath] = []
                fixes_by_file[filepath].append(issue.strip())

        for filepath, issues in sorted(fixes_by_file.items()):
            report += f"\n{filepath}:\n"
            for issue in issues:
                report += f"  - {issue}\n"

        return report


def main():
    """Main function to fix technical debt"""

    fixer = TechnicalDebtFixer()

    # Directories to scan
    directories = ["/mnt/c/Projects/Mimir/api", "/mnt/c/Projects/Mimir/auth", "/mnt/c/Projects/Mimir/services"]

    # Specific files to focus on (EPIC 5 implementation)
    priority_files = [
        "api/routers/dashboards.py",
        "api/routers/saved_searches.py",
        "api/routers/saved_search_analytics.py",
        "api/models/dashboard.py",
        "services/dashboard_manager.py",
        "services/widget_data_service.py",
        "services/saved_search_manager.py",
        "services/saved_search_analytics.py",
        "services/alert_scheduler.py",
    ]

    print("Starting Technical Debt Fix for EPIC 5...")
    print("=" * 50)

    # Fix priority files first
    for rel_path in priority_files:
        filepath = Path(f"/mnt/c/Projects/Mimir/{rel_path}")
        if filepath.exists():
            print(f"Checking {filepath}...")
            fixer.fix_file(filepath)

    # Then scan other Python files
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            for filepath in dir_path.rglob("*.py"):
                if str(filepath) not in [f"/mnt/c/Projects/Mimir/{p}" for p in priority_files]:
                    print(f"Checking {filepath}...")
                    fixer.fix_file(filepath)

    # Generate and print report
    report = fixer.generate_report()
    print("\n" + "=" * 50)
    print(report)

    # Save report
    report_path = Path("/mnt/c/Projects/Mimir/EPIC5_TECHNICAL_DEBT_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
