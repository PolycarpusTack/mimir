#!/usr/bin/env python3
"""
Comprehensive Test Runner for Mimir Enterprise

Orchestrates different types of testing: unit, integration, contract, performance.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


class TestRunner:
    """Main test runner orchestrator"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports" / "tests"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Test configuration
        self.test_config = {
            "unit": {"path": "tests/unit", "markers": "unit", "timeout": 300, "parallel": True},
            "integration": {"path": "tests/integration", "markers": "integration", "timeout": 600, "parallel": False},
            "contract": {"path": "tests/api/test_contract.py", "markers": "contract", "timeout": 300, "parallel": True},
            "performance": {"path": "tests/performance", "markers": "performance", "timeout": 1800, "parallel": False},
            "load": {"path": "tests/locustfile.py", "tool": "locust", "timeout": 3600},
        }

    def run_pytest_tests(
        self, test_type: str, verbose: bool = True, coverage: bool = True, html_report: bool = True
    ) -> Dict[str, Any]:
        """Run pytest-based tests"""
        config = self.test_config.get(test_type)
        if not config:
            raise ValueError(f"Unknown test type: {test_type}")

        test_path = self.project_root / config["path"]
        if not test_path.exists():
            return {"success": False, "error": f"Test path does not exist: {test_path}", "duration": 0}

        # Build pytest command
        cmd = ["python", "-m", "pytest"]

        # Add test path
        cmd.append(str(test_path))

        # Add markers
        if config.get("markers"):
            cmd.extend(["-m", config["markers"]])

        # Add verbosity
        if verbose:
            cmd.append("-v")

        # Add parallel execution
        if config.get("parallel") and test_type != "integration":
            cmd.extend(["-n", "auto"])

        # Add coverage
        if coverage:
            cmd.extend(["--cov=.", "--cov-report=html", f"--cov-report=term"])
            cmd.append(f"--cov-report=html:{self.reports_dir}/coverage_{test_type}")

        # Add HTML report
        if html_report:
            cmd.extend(["--html", str(self.reports_dir / f"{test_type}_report.html")])
            cmd.append("--self-contained-html")

        # Add JUnit XML report
        cmd.extend(["--junit-xml", str(self.reports_dir / f"{test_type}_junit.xml")])

        # Add timeout
        if config.get("timeout"):
            cmd.extend(["--timeout", str(config["timeout"])])

        print(f"Running {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd, cwd=self.project_root, capture_output=True, text=True, timeout=config.get("timeout", 3600)
            )

            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Tests timed out after {config.get('timeout')} seconds",
                "duration": time.time() - start_time,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    def run_load_tests(
        self,
        host: str = "http://localhost:8000",
        users: int = 50,
        spawn_rate: int = 5,
        duration: str = "300s",
        user_class: str = "MimirAPIUser",
    ) -> Dict[str, Any]:
        """Run Locust load tests"""
        locust_file = self.project_root / "tests" / "locustfile.py"
        if not locust_file.exists():
            return {"success": False, "error": f"Locust file not found: {locust_file}", "duration": 0}

        # Build locust command
        cmd = [
            "locust",
            "-f",
            str(locust_file),
            "--host",
            host,
            "-u",
            str(users),
            "-r",
            str(spawn_rate),
            "-t",
            duration,
            "--headless",
            "--html",
            str(self.reports_dir / "load_test_report.html"),
            "--csv",
            str(self.reports_dir / "load_test"),
            user_class,
        ]

        print(f"Running load tests...")
        print(f"Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=3600)

            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Load tests timed out", "duration": time.time() - start_time}
        except Exception as e:
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        # Run the security scanner
        security_script = self.project_root / "scripts" / "security_config.py"
        if not security_script.exists():
            return {"success": False, "error": "Security script not found", "duration": 0}

        cmd = ["python", str(security_script), "scan", "--code", "--output-dir", str(self.reports_dir / "security")]

        print("Running security tests...")
        start_time = time.time()

        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=1800)

            duration = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "duration": time.time() - start_time}

    def run_all_tests(self, include_load: bool = False, include_security: bool = True) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        print("=" * 60)
        print("MIMIR ENTERPRISE - COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "tests": {},
            "summary": {"total_duration": 0, "passed": 0, "failed": 0, "skipped": 0},
        }

        # Test execution order (dependencies matter)
        test_order = ["unit", "contract", "integration", "performance"]

        for test_type in test_order:
            print(f"\n{'='*20} {test_type.upper()} TESTS {'='*20}")

            test_result = self.run_pytest_tests(
                test_type=test_type,
                verbose=True,
                coverage=(test_type == "unit"),  # Only coverage for unit tests
                html_report=True,
            )

            results["tests"][test_type] = test_result
            results["summary"]["total_duration"] += test_result.get("duration", 0)

            if test_result["success"]:
                results["summary"]["passed"] += 1
                print(f"✅ {test_type.title()} tests PASSED ({test_result['duration']:.1f}s)")
            else:
                results["summary"]["failed"] += 1
                print(f"❌ {test_type.title()} tests FAILED ({test_result['duration']:.1f}s)")
                if "error" in test_result:
                    print(f"   Error: {test_result['error']}")

                # Print stderr if available
                if test_result.get("stderr"):
                    print(f"   stderr: {test_result['stderr']}")

        # Run load tests if requested
        if include_load:
            print(f"\n{'='*20} LOAD TESTS {'='*20}")
            load_result = self.run_load_tests()
            results["tests"]["load"] = load_result
            results["summary"]["total_duration"] += load_result.get("duration", 0)

            if load_result["success"]:
                results["summary"]["passed"] += 1
                print(f"✅ Load tests PASSED ({load_result['duration']:.1f}s)")
            else:
                results["summary"]["failed"] += 1
                print(f"❌ Load tests FAILED ({load_result['duration']:.1f}s)")

        # Run security tests if requested
        if include_security:
            print(f"\n{'='*20} SECURITY TESTS {'='*20}")
            security_result = self.run_security_tests()
            results["tests"]["security"] = security_result
            results["summary"]["total_duration"] += security_result.get("duration", 0)

            if security_result["success"]:
                results["summary"]["passed"] += 1
                print(f"✅ Security tests PASSED ({security_result['duration']:.1f}s)")
            else:
                results["summary"]["failed"] += 1
                print(f"❌ Security tests FAILED ({security_result['duration']:.1f}s)")

        # Save comprehensive report
        report_file = self.reports_dir / "comprehensive_test_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        self.print_test_summary(results)

        return results

    def print_test_summary(self, results: Dict[str, Any]):
        """Print test execution summary"""
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")

        summary = results["summary"]
        print(f"Total Duration: {summary['total_duration']:.1f}s")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['passed'] / (summary['passed'] + summary['failed']) * 100:.1f}%")

        print(f"\nReports generated in: {self.reports_dir}")

        # List generated reports
        report_files = (
            list(self.reports_dir.glob("*.html"))
            + list(self.reports_dir.glob("*.xml"))
            + list(self.reports_dir.glob("*.json"))
        )
        if report_files:
            print("\nGenerated Reports:")
            for report in sorted(report_files):
                print(f"  - {report.name}")

        # Overall result
        if summary["failed"] == 0:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            return True
        else:
            print(f"\n💥 {summary['failed']} TEST SUITE(S) FAILED!")
            return False

    def check_test_environment(self) -> bool:
        """Check if test environment is properly setup"""
        print("Checking test environment...")

        # Check required tools
        required_tools = ["python", "pytest"]
        optional_tools = ["locust", "safety", "bandit"]

        missing_required = []
        missing_optional = []

        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                print(f"✅ {tool} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_required.append(tool)
                print(f"❌ {tool} is missing (required)")

        for tool in optional_tools:
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
                print(f"✅ {tool} is available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                missing_optional.append(tool)
                print(f"⚠️  {tool} is missing (optional)")

        if missing_required:
            print(f"\n❌ Missing required tools: {', '.join(missing_required)}")
            print("Please install missing tools before running tests.")
            return False

        if missing_optional:
            print(f"\n⚠️  Missing optional tools: {', '.join(missing_optional)}")
            print("Some test types may not be available.")

        print("✅ Test environment check passed!")
        return True


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Mimir Enterprise Test Runner")
    parser.add_argument("--project-root", default=".", help="Project root directory")

    subparsers = parser.add_subparsers(dest="command", help="Test commands")

    # Check environment command
    check_parser = subparsers.add_parser("check", help="Check test environment")

    # Run specific test types
    run_parser = subparsers.add_parser("run", help="Run tests")
    run_parser.add_argument(
        "test_type", choices=["unit", "integration", "contract", "performance", "load", "security", "all"]
    )
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    run_parser.add_argument("--no-coverage", action="store_true", help="Disable coverage")
    run_parser.add_argument("--no-html", action="store_true", help="Disable HTML reports")

    # Load test specific options
    load_parser = subparsers.add_parser("load", help="Run load tests")
    load_parser.add_argument("--host", default="http://localhost:8000", help="Target host")
    load_parser.add_argument("--users", "-u", type=int, default=50, help="Number of users")
    load_parser.add_argument("--spawn-rate", "-r", type=int, default=5, help="Spawn rate")
    load_parser.add_argument("--duration", "-t", default="300s", help="Test duration")
    load_parser.add_argument("--user-class", default="MimirAPIUser", help="User class to use")

    # All tests options
    all_parser = subparsers.add_parser("all", help="Run all tests")
    all_parser.add_argument("--include-load", action="store_true", help="Include load tests")
    all_parser.add_argument("--no-security", action="store_true", help="Skip security tests")

    args = parser.parse_args()

    runner = TestRunner(args.project_root)

    if args.command == "check":
        success = runner.check_test_environment()
        sys.exit(0 if success else 1)

    elif args.command == "run":
        if args.test_type == "all":
            results = runner.run_all_tests()
            sys.exit(0 if results["summary"]["failed"] == 0 else 1)
        elif args.test_type == "load":
            result = runner.run_load_tests()
            sys.exit(0 if result["success"] else 1)
        elif args.test_type == "security":
            result = runner.run_security_tests()
            sys.exit(0 if result["success"] else 1)
        else:
            result = runner.run_pytest_tests(
                test_type=args.test_type,
                verbose=args.verbose,
                coverage=not args.no_coverage,
                html_report=not args.no_html,
            )
            print(f"Test result: {'PASSED' if result['success'] else 'FAILED'}")
            if not result["success"] and "error" in result:
                print(f"Error: {result['error']}")
            sys.exit(0 if result["success"] else 1)

    elif args.command == "load":
        result = runner.run_load_tests(
            host=args.host,
            users=args.users,
            spawn_rate=args.spawn_rate,
            duration=args.duration,
            user_class=args.user_class,
        )
        sys.exit(0 if result["success"] else 1)

    elif args.command == "all":
        results = runner.run_all_tests(include_load=args.include_load, include_security=not args.no_security)
        sys.exit(0 if results["summary"]["failed"] == 0 else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
