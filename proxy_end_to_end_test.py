"""
End-to-End Proxy System Testing Framework for Mimir
Comprehensive testing of the complete proxy system with real-world scenarios
"""

import asyncio
import json
import logging
import time
import tempfile
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import requests
import concurrent.futures
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of an end-to-end test"""
    test_name: str
    passed: bool
    duration_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class LoadTestMetrics:
    """Metrics from load testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    proxy_usage_rate: float
    unique_proxies_used: int


class ProxyEndToEndTester:
    """Comprehensive end-to-end testing framework for proxy system"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.test_results: List[TestResult] = []
        self.config_data: Optional[Dict] = None
        self.temp_files: List[str] = []
        
    def load_configuration(self) -> bool:
        """Load and validate configuration"""
        try:
            if not self.config_file.exists():
                logger.error(f"Configuration file not found: {self.config_file}")
                return False
            
            with open(self.config_file, 'r') as f:
                self.config_data = json.load(f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def test_proxy_system_initialization(self) -> TestResult:
        """Test proxy system initialization"""
        start_time = time.time()
        
        try:
            from scraper_proxy_integration import ScraperProxyManager
            
            # Initialize proxy manager
            proxy_manager = ScraperProxyManager(self.config_data)
            
            # Check initialization
            if proxy_manager.initialization_error:
                return TestResult(
                    test_name="Proxy System Initialization",
                    passed=False,
                    duration_seconds=time.time() - start_time,
                    details={"error": proxy_manager.initialization_error},
                    error_message=proxy_manager.initialization_error
                )
            
            # Get status
            status = proxy_manager.get_proxy_status()
            
            return TestResult(
                test_name="Proxy System Initialization",
                passed=True,
                duration_seconds=time.time() - start_time,
                details={
                    "enabled": status["enabled"],
                    "total_requests": status["total_requests"],
                    "proxy_system": status.get("proxy_system", {})
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Proxy System Initialization",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_configuration_validation(self) -> TestResult:
        """Test configuration validation"""
        start_time = time.time()
        
        try:
            from proxy_config_validator import ProxyConfigValidator
            
            validator = ProxyConfigValidator(str(self.config_file))
            summary = validator.run_comprehensive_validation()
            
            return TestResult(
                test_name="Configuration Validation",
                passed=summary["critical_issues"] == 0,
                duration_seconds=time.time() - start_time,
                details=summary
            )
            
        except Exception as e:
            return TestResult(
                test_name="Configuration Validation",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_proxy_provider_connectivity(self) -> TestResult:
        """Test connectivity to configured proxy providers"""
        start_time = time.time()
        
        try:
            from proxy_provider_templates import ProxyProviderManager
            
            provider_manager = ProxyProviderManager()
            validation_results = provider_manager.validate_all_providers()
            
            working_providers = [name for name, valid in validation_results.items() if valid]
            
            return TestResult(
                test_name="Proxy Provider Connectivity",
                passed=len(working_providers) > 0,
                duration_seconds=time.time() - start_time,
                details={
                    "validation_results": validation_results,
                    "working_providers": working_providers,
                    "total_providers": len(validation_results)
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Proxy Provider Connectivity",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_scraper_integration(self) -> TestResult:
        """Test integration with main scraper"""
        start_time = time.time()
        
        try:
            from scraper_proxy_integration import enhanced_fetch_url
            
            # Test with a simple HTTP endpoint
            test_url = "http://httpbin.org/user-agent"
            
            # Fetch with proxy system
            result = enhanced_fetch_url(test_url, self.config_data)
            
            success = result is not None
            
            details = {
                "test_url": test_url,
                "response_received": success,
                "response_length": len(result) if result else 0
            }
            
            if result:
                try:
                    # Parse response to check user agent
                    response_data = json.loads(result)
                    details["user_agent"] = response_data.get("user-agent", "")
                except:
                    pass
            
            return TestResult(
                test_name="Scraper Integration",
                passed=success,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Scraper Integration",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_proxy_rotation(self) -> TestResult:
        """Test proxy rotation functionality"""
        start_time = time.time()
        
        try:
            from scraper_proxy_integration import ScraperProxyManager
            
            proxy_manager = ScraperProxyManager(self.config_data)
            
            if not proxy_manager.enabled or proxy_manager.initialization_error:
                return TestResult(
                    test_name="Proxy Rotation",
                    passed=False,
                    duration_seconds=time.time() - start_time,
                    details={"error": "Proxy system not properly initialized"},
                    error_message="Proxy system not available"
                )
            
            # Test multiple requests to check rotation
            test_urls = [
                "http://httpbin.org/ip",
                "http://httpbin.org/user-agent",
                "http://httpbin.org/headers"
            ]
            
            responses = []
            unique_ips = set()
            
            for i, url in enumerate(test_urls * 2):  # Test 6 requests
                try:
                    result = proxy_manager.fetch_url_with_proxy(url)
                    if result:
                        responses.append(True)
                        # Try to extract IP if possible
                        if "ip" in url:
                            try:
                                ip_data = json.loads(result)
                                if "origin" in ip_data:
                                    unique_ips.add(ip_data["origin"])
                            except:
                                pass
                    else:
                        responses.append(False)
                except Exception as e:
                    responses.append(False)
                    logger.warning(f"Request {i} failed: {e}")
            
            success_rate = sum(responses) / len(responses) * 100
            
            return TestResult(
                test_name="Proxy Rotation",
                passed=success_rate >= 50,  # At least 50% success rate
                duration_seconds=time.time() - start_time,
                details={
                    "total_requests": len(responses),
                    "successful_requests": sum(responses),
                    "success_rate": success_rate,
                    "unique_ips": len(unique_ips),
                    "proxy_performance": proxy_manager.get_performance_summary()
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Proxy Rotation",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_error_handling_and_fallbacks(self) -> TestResult:
        """Test error handling and fallback mechanisms"""
        start_time = time.time()
        
        try:
            from scraper_proxy_integration import ScraperProxyManager
            
            # Test with broken configuration
            broken_config = self.config_data.copy()
            broken_config["proxy_settings"] = {
                "enabled": True,
                "rotation_strategy": "invalid_strategy",
                "proxy_providers": []
            }
            
            proxy_manager = ScraperProxyManager(broken_config)
            
            # Should handle gracefully
            test_url = "http://httpbin.org/ip"
            result = proxy_manager.fetch_url_with_proxy(test_url)
            
            # Should use fallback and still work
            fallback_worked = result is not None
            
            return TestResult(
                test_name="Error Handling and Fallbacks",
                passed=fallback_worked,
                duration_seconds=time.time() - start_time,
                details={
                    "initialization_error": proxy_manager.initialization_error,
                    "fallback_worked": fallback_worked,
                    "performance": proxy_manager.get_performance_summary()
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Error Handling and Fallbacks",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_playwright_proxy_integration(self) -> TestResult:
        """Test Playwright proxy integration"""
        start_time = time.time()
        
        try:
            # Test Playwright configuration loading
            from playwright_integration import load_playwright_config
            
            playwright_config = load_playwright_config()
            
            if not playwright_config:
                return TestResult(
                    test_name="Playwright Proxy Integration",
                    passed=False,
                    duration_seconds=time.time() - start_time,
                    details={"error": "Playwright configuration not available"},
                    error_message="Playwright config missing"
                )
            
            # Test proxy integration setup
            from scraper_proxy_integration import get_scraper_proxy_manager
            from proxy_integration import integrate_proxy_with_playwright_scraper
            
            scraper_proxy_manager = get_scraper_proxy_manager()
            
            if scraper_proxy_manager and scraper_proxy_manager.proxy_manager:
                # Test proxy integration
                test_site_config = {"url": "http://example.com"}
                
                enhanced_config = integrate_proxy_with_playwright_scraper(
                    playwright_config.copy(),
                    test_site_config,
                    scraper_proxy_manager.proxy_manager
                )
                
                # Check if proxy was added to config
                proxy_added = "proxy" in enhanced_config.get("defaults", {})
                
                return TestResult(
                    test_name="Playwright Proxy Integration",
                    passed=True,
                    duration_seconds=time.time() - start_time,
                    details={
                        "playwright_config_loaded": True,
                        "proxy_integration_available": True,
                        "proxy_added_to_config": proxy_added,
                        "config_keys": list(enhanced_config.keys())
                    }
                )
            else:
                return TestResult(
                    test_name="Playwright Proxy Integration",
                    passed=False,
                    duration_seconds=time.time() - start_time,
                    details={"error": "Proxy manager not available"},
                    error_message="Proxy manager not initialized"
                )
            
        except Exception as e:
            return TestResult(
                test_name="Playwright Proxy Integration",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def test_monitoring_and_dashboard(self) -> TestResult:
        """Test monitoring and dashboard functionality"""
        start_time = time.time()
        
        try:
            from proxy_integration import get_proxy_statistics
            from scraper_proxy_integration import get_proxy_integration_status
            
            # Test statistics generation
            stats = get_proxy_statistics(self.config_data)
            
            # Test integration status
            status = get_proxy_integration_status()
            
            # Test dashboard components
            from proxy_dashboard import create_proxy_dashboard_app
            
            app = create_proxy_dashboard_app()
            
            return TestResult(
                test_name="Monitoring and Dashboard",
                passed=True,
                duration_seconds=time.time() - start_time,
                details={
                    "statistics_available": bool(stats),
                    "integration_status": status,
                    "dashboard_app_created": app is not None,
                    "stats_keys": list(stats.keys()) if stats else []
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Monitoring and Dashboard",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def run_load_test(self, duration_seconds: int = 60, concurrent_requests: int = 5) -> LoadTestMetrics:
        """Run load test on proxy system"""
        logger.info(f"Starting load test: {duration_seconds}s duration, {concurrent_requests} concurrent requests")
        
        from scraper_proxy_integration import ScraperProxyManager
        
        proxy_manager = ScraperProxyManager(self.config_data)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        response_times = []
        unique_proxies = set()
        
        def make_request():
            nonlocal total_requests, successful_requests, failed_requests
            
            request_start = time.time()
            try:
                result = proxy_manager.fetch_url_with_proxy("http://httpbin.org/delay/1")
                request_time = (time.time() - request_start) * 1000  # ms
                
                if result:
                    successful_requests += 1
                    response_times.append(request_time)
                    
                    # Try to track proxy usage
                    try:
                        data = json.loads(result)
                        origin = data.get("origin", "")
                        if origin:
                            unique_proxies.add(origin)
                    except:
                        pass
                else:
                    failed_requests += 1
                    
            except Exception as e:
                failed_requests += 1
                logger.debug(f"Request failed: {e}")
            
            total_requests += 1
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit new requests up to concurrent limit
                while len(futures) < concurrent_requests and time.time() < end_time:
                    future = executor.submit(make_request)
                    futures.append(future)
                
                # Remove completed futures
                futures = [f for f in futures if not f.done()]
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
        
        # Wait for remaining requests to complete
        for future in futures:
            future.result()
        
        # Calculate metrics
        total_duration = time.time() - start_time
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        
        proxy_usage_rate = 0
        if proxy_manager.total_requests > 0:
            proxy_usage_rate = (proxy_manager.proxy_requests / proxy_manager.total_requests) * 100
        
        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=sum(response_times) / len(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            requests_per_second=requests_per_second,
            proxy_usage_rate=proxy_usage_rate,
            unique_proxies_used=len(unique_proxies)
        )
    
    def test_load_performance(self) -> TestResult:
        """Test proxy system under load"""
        start_time = time.time()
        
        try:
            # Run short load test
            metrics = self.run_load_test(duration_seconds=30, concurrent_requests=3)
            
            # Evaluate performance
            success_rate = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
            
            # Performance criteria
            acceptable_success_rate = 70  # At least 70% success rate
            acceptable_response_time = 10000  # Under 10 seconds average
            
            passed = (
                success_rate >= acceptable_success_rate and
                metrics.average_response_time <= acceptable_response_time
            )
            
            return TestResult(
                test_name="Load Performance",
                passed=passed,
                duration_seconds=time.time() - start_time,
                details={
                    "metrics": {
                        "total_requests": metrics.total_requests,
                        "successful_requests": metrics.successful_requests,
                        "success_rate": success_rate,
                        "average_response_time": metrics.average_response_time,
                        "requests_per_second": metrics.requests_per_second,
                        "proxy_usage_rate": metrics.proxy_usage_rate,
                        "unique_proxies_used": metrics.unique_proxies_used
                    },
                    "performance_criteria": {
                        "acceptable_success_rate": acceptable_success_rate,
                        "acceptable_response_time": acceptable_response_time
                    }
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Load Performance",
                passed=False,
                duration_seconds=time.time() - start_time,
                details={"exception": str(e)},
                error_message=str(e)
            )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all end-to-end tests"""
        logger.info("🧪 Starting comprehensive end-to-end proxy system tests...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load configuration
        if not self.load_configuration():
            return {
                "success": False,
                "error": "Failed to load configuration",
                "tests": []
            }
        
        # Define test suite
        test_methods = [
            ("Proxy System Initialization", self.test_proxy_system_initialization),
            ("Configuration Validation", self.test_configuration_validation),
            ("Proxy Provider Connectivity", self.test_proxy_provider_connectivity),
            ("Scraper Integration", self.test_scraper_integration),
            ("Proxy Rotation", self.test_proxy_rotation),
            ("Error Handling and Fallbacks", self.test_error_handling_and_fallbacks),
            ("Playwright Proxy Integration", self.test_playwright_proxy_integration),
            ("Monitoring and Dashboard", self.test_monitoring_and_dashboard),
            ("Load Performance", self.test_load_performance)
        ]
        
        # Run all tests
        for test_name, test_method in test_methods:
            logger.info(f"Running: {test_name}")
            try:
                result = test_method()
                self.test_results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"   {status} ({result.duration_seconds:.2f}s)")
                
                if not result.passed and result.error_message:
                    logger.warning(f"   Error: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"   ❌ FAILED - Unexpected error: {e}")
                self.test_results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    duration_seconds=0,
                    details={"unexpected_error": str(e)},
                    error_message=str(e)
                ))
        
        # Calculate summary
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in self.test_results if r.passed)
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "success": passed_tests == total_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "timestamp": datetime.now().isoformat(),
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "duration": r.duration_seconds,
                    "details": r.details,
                    "error": r.error_message
                }
                for r in self.test_results
            ]
        }
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🏁 END-TO-END TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info("=" * 80)
        
        if summary["success"]:
            logger.info("🎉 ALL TESTS PASSED - Proxy system is fully functional!")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed - system needs attention")
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


def main():
    """Main function for end-to-end testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run end-to-end tests for Mimir proxy system")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--load-test", action="store_true", help="Run extended load test")
    parser.add_argument("--duration", type=int, default=60, help="Load test duration in seconds")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent requests for load test")
    
    args = parser.parse_args()
    
    print("🔄 MIMIR PROXY SYSTEM - END-TO-END TESTING")
    print("=" * 80)
    print("Testing complete proxy system functionality with real-world scenarios")
    print("=" * 80)
    
    tester = ProxyEndToEndTester(args.config)
    
    try:
        if args.load_test:
            print(f"Running extended load test ({args.duration}s, {args.concurrent} concurrent)...")
            metrics = tester.run_load_test(args.duration, args.concurrent)
            
            print("\n📊 LOAD TEST RESULTS:")
            print(f"   Total Requests: {metrics.total_requests}")
            print(f"   Successful: {metrics.successful_requests}")
            print(f"   Failed: {metrics.failed_requests}")
            print(f"   Success Rate: {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
            print(f"   Avg Response Time: {metrics.average_response_time:.2f}ms")
            print(f"   Requests/Second: {metrics.requests_per_second:.2f}")
            print(f"   Proxy Usage Rate: {metrics.proxy_usage_rate:.1f}%")
            print(f"   Unique Proxies: {metrics.unique_proxies_used}")
        
        else:
            # Run full test suite
            results = tester.run_all_tests()
            
            # Save results
            results_file = f"proxy_e2e_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n📄 Test results saved to: {results_file}")
            
            return 0 if results["success"] else 1
    
    except KeyboardInterrupt:
        print("\n⚠️  Testing cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.exception("End-to-end testing failed")
        return 1
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit(main())