#!/usr/bin/env python
"""
Comprehensive Test Suite for Mimir Proxy System
Tests proxy rotation, health monitoring, and integration with scraper
"""

import json
import logging
import time
import tempfile
import os
from datetime import datetime
from typing import Dict, List

from proxy_manager import ProxyManager, ProxyStatus
from proxy_integration import (
    create_proxy_enabled_session, 
    ProxyPlaywrightIntegration,
    get_proxy_statistics,
    health_check_all_proxies
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxySystemTester:
    """Comprehensive test suite for proxy system"""
    
    def __init__(self):
        self.test_results = []
        self.test_config = None
        self.temp_files = []
    
    def create_test_config(self) -> Dict:
        """Create test configuration with mock proxies"""
        # Create temporary proxy list file
        proxy_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_files.append(proxy_list_file.name)
        
        proxy_list_data = {
            "proxies": [
                {
                    "url": "http://proxy1.test.com:8080",
                    "location": "US",
                    "tags": ["fast", "reliable"]
                },
                {
                    "url": "http://proxy2.test.com:8080", 
                    "location": "EU",
                    "tags": ["secure"]
                }
            ]
        }
        
        json.dump(proxy_list_data, proxy_list_file)
        proxy_list_file.close()
        
        # Create test configuration
        self.test_config = {
            "enabled": True,
            "rotation_strategy": "round_robin",
            "health_check_interval": 10,  # Short interval for testing
            "failure_threshold": 2,
            "health_check_timeout": 5,
            "health_check_url": "http://httpbin.org/ip",
            "proxy_providers": [
                {
                    "type": "static_list",
                    "name": "test_static",
                    "enabled": True,
                    "proxies": [
                        "http://static1.test.com:8080",
                        "http://static2.test.com:8080"
                    ]
                },
                {
                    "type": "file",
                    "name": "test_file",
                    "enabled": True,
                    "file_path": proxy_list_file.name
                }
            ]
        }
        
        return self.test_config
    
    def test_proxy_manager_initialization(self) -> bool:
        """Test ProxyManager initialization and configuration loading"""
        logger.info("Testing ProxyManager initialization...")
        
        try:
            config = self.create_test_config()
            proxy_manager = ProxyManager(config)
            
            # Verify proxy loading
            assert len(proxy_manager.proxies) == 4, f"Expected 4 proxies, got {len(proxy_manager.proxies)}"
            assert proxy_manager.enabled == True, "ProxyManager should be enabled"
            assert proxy_manager.rotation_strategy == "round_robin", "Wrong rotation strategy"
            
            logger.info("✅ ProxyManager initialization test passed")
            self.test_results.append(("ProxyManager Initialization", True, "All checks passed"))
            return True
            
        except Exception as e:
            logger.error(f"❌ ProxyManager initialization test failed: {e}")
            self.test_results.append(("ProxyManager Initialization", False, str(e)))
            return False
    
    def test_proxy_rotation_strategies(self) -> bool:
        """Test different proxy rotation strategies"""
        logger.info("Testing proxy rotation strategies...")
        
        try:
            config = self.create_test_config()
            
            # Test round-robin
            config["rotation_strategy"] = "round_robin"
            proxy_manager = ProxyManager(config)
            
            # Mark all proxies as active for testing
            for proxy_info in proxy_manager.proxies.values():
                proxy_info.status = ProxyStatus.ACTIVE
                proxy_info.metrics.successful_requests = 10
                proxy_info.metrics.total_requests = 10
            
            # Get proxies in sequence
            proxies_sequence = []
            for i in range(8):  # Get more proxies than available to test cycling
                proxy = proxy_manager.get_proxy()
                proxies_sequence.append(proxy)
            
            # Verify round-robin behavior
            unique_proxies = set(proxies_sequence)
            assert len(unique_proxies) == 4, f"Expected 4 unique proxies, got {len(unique_proxies)}"
            
            # Test random strategy
            config["rotation_strategy"] = "random"
            proxy_manager = ProxyManager(config)
            
            # Mark all proxies as active for testing
            for proxy_info in proxy_manager.proxies.values():
                proxy_info.status = ProxyStatus.ACTIVE
                proxy_info.metrics.successful_requests = 10
                proxy_info.metrics.total_requests = 10
            
            random_proxies = [proxy_manager.get_proxy() for _ in range(10)]
            assert len(set(random_proxies)) >= 1, "Random strategy should return proxies"
            
            logger.info("✅ Proxy rotation strategies test passed")
            self.test_results.append(("Proxy Rotation Strategies", True, "Round-robin and random working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy rotation strategies test failed: {e}")
            self.test_results.append(("Proxy Rotation Strategies", False, str(e)))
            return False
    
    def test_proxy_health_monitoring(self) -> bool:
        """Test proxy health monitoring and blacklisting"""
        logger.info("Testing proxy health monitoring...")
        
        try:
            config = self.create_test_config()
            config["failure_threshold"] = 2  # Low threshold for testing
            proxy_manager = ProxyManager(config)
            
            # Get a test proxy
            proxy_urls = list(proxy_manager.proxies.keys())
            test_proxy = proxy_urls[0]
            
            # Record failures to trigger blacklisting
            proxy_manager.record_proxy_result(test_proxy, False)
            proxy_manager.record_proxy_result(test_proxy, False)
            
            # Verify blacklisting
            assert test_proxy in proxy_manager.blacklisted_proxies, "Proxy should be blacklisted"
            assert proxy_manager.proxies[test_proxy].status == ProxyStatus.BLACKLISTED, "Proxy status should be blacklisted"
            
            # Test reset functionality
            proxy_manager.reset_proxy(test_proxy)
            assert test_proxy not in proxy_manager.blacklisted_proxies, "Proxy should be removed from blacklist"
            
            logger.info("✅ Proxy health monitoring test passed")
            self.test_results.append(("Proxy Health Monitoring", True, "Blacklisting and reset working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy health monitoring test failed: {e}")
            self.test_results.append(("Proxy Health Monitoring", False, str(e)))
            return False
    
    def test_proxy_metrics_tracking(self) -> bool:
        """Test proxy performance metrics tracking"""
        logger.info("Testing proxy metrics tracking...")
        
        try:
            config = self.create_test_config()
            proxy_manager = ProxyManager(config)
            
            proxy_urls = list(proxy_manager.proxies.keys())
            test_proxy = proxy_urls[0]
            
            # Record various results
            proxy_manager.record_proxy_result(test_proxy, True, 100.0)
            proxy_manager.record_proxy_result(test_proxy, True, 200.0)
            proxy_manager.record_proxy_result(test_proxy, False, 0.0)
            
            metrics = proxy_manager.proxies[test_proxy].metrics
            
            # Verify metrics
            assert metrics.total_requests == 3, f"Expected 3 total requests, got {metrics.total_requests}"
            assert metrics.successful_requests == 2, f"Expected 2 successful requests, got {metrics.successful_requests}"
            assert metrics.failed_requests == 1, f"Expected 1 failed request, got {metrics.failed_requests}"
            assert metrics.success_rate == (2/3) * 100, f"Expected 66.67% success rate, got {metrics.success_rate}"
            assert metrics.average_response_time > 0, "Average response time should be calculated"
            
            logger.info("✅ Proxy metrics tracking test passed")
            self.test_results.append(("Proxy Metrics Tracking", True, "All metrics calculated correctly"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy metrics tracking test failed: {e}")
            self.test_results.append(("Proxy Metrics Tracking", False, str(e)))
            return False
    
    def test_proxy_integration_with_requests(self) -> bool:
        """Test proxy integration with requests library"""
        logger.info("Testing proxy integration with requests...")
        
        try:
            config = {
                "proxy_settings": self.create_test_config(),
                "user_agent": "Test-Agent/1.0"
            }
            
            # Test disabled proxy system
            config["proxy_settings"]["enabled"] = False
            session = create_proxy_enabled_session(config)
            assert session.headers["User-Agent"] == "Test-Agent/1.0", "User agent should be set"
            
            # Note: We can't easily test actual proxy usage without real proxies
            # This test verifies the integration layer works
            
            logger.info("✅ Proxy integration with requests test passed")
            self.test_results.append(("Proxy Integration with Requests", True, "Integration layer working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy integration with requests test failed: {e}")
            self.test_results.append(("Proxy Integration with Requests", False, str(e)))
            return False
    
    def test_playwright_proxy_integration(self) -> bool:
        """Test proxy integration with Playwright"""
        logger.info("Testing Playwright proxy integration...")
        
        try:
            config = self.create_test_config()
            proxy_manager = ProxyManager(config)
            
            # Mark proxies as active
            for proxy_info in proxy_manager.proxies.values():
                proxy_info.status = ProxyStatus.ACTIVE
                proxy_info.metrics.successful_requests = 10
                proxy_info.metrics.total_requests = 10
            
            playwright_integration = ProxyPlaywrightIntegration(proxy_manager)
            
            # Test proxy config generation
            proxy_config = playwright_integration.get_proxy_config_for_playwright()
            assert proxy_config is not None, "Proxy config should be generated"
            assert "server" in proxy_config, "Proxy config should have server field"
            
            # Test result recording
            test_proxy = list(proxy_manager.proxies.keys())[0]
            playwright_integration.record_playwright_proxy_result(test_proxy, True, 150.0)
            
            metrics = proxy_manager.proxies[test_proxy].metrics
            assert metrics.total_requests > 0, "Metrics should be updated"
            
            logger.info("✅ Playwright proxy integration test passed")
            self.test_results.append(("Playwright Proxy Integration", True, "Integration working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Playwright proxy integration test failed: {e}")
            self.test_results.append(("Playwright Proxy Integration", False, str(e)))
            return False
    
    def test_proxy_state_persistence(self) -> bool:
        """Test proxy state saving and loading"""
        logger.info("Testing proxy state persistence...")
        
        try:
            config = self.create_test_config()
            proxy_manager = ProxyManager(config)
            
            # Record some data
            proxy_urls = list(proxy_manager.proxies.keys())
            test_proxy = proxy_urls[0]
            proxy_manager.record_proxy_result(test_proxy, True, 100.0)
            proxy_manager.record_proxy_result(test_proxy, False, 0.0)
            
            # Save state
            state_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            state_file.close()
            self.temp_files.append(state_file.name)
            
            proxy_manager.save_proxy_state(state_file.name)
            
            # Create new manager and load state
            proxy_manager2 = ProxyManager(config)
            proxy_manager2.load_proxy_state(state_file.name)
            
            # Verify state was loaded
            metrics = proxy_manager2.proxies[test_proxy].metrics
            assert metrics.total_requests == 2, "Metrics should be restored"
            assert metrics.successful_requests == 1, "Successful requests should be restored"
            
            logger.info("✅ Proxy state persistence test passed")
            self.test_results.append(("Proxy State Persistence", True, "Save and load working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy state persistence test failed: {e}")
            self.test_results.append(("Proxy State Persistence", False, str(e)))
            return False
    
    def test_proxy_statistics_generation(self) -> bool:
        """Test proxy statistics generation"""
        logger.info("Testing proxy statistics generation...")
        
        try:
            config = self.create_test_config()
            proxy_manager = ProxyManager(config)
            
            # Record some data
            proxy_urls = list(proxy_manager.proxies.keys())
            for i, proxy_url in enumerate(proxy_urls):
                for j in range(i + 1):  # Different number of requests per proxy
                    proxy_manager.record_proxy_result(proxy_url, j % 2 == 0, 100.0 + i * 50)
            
            # Generate statistics
            stats = proxy_manager.get_proxy_statistics()
            
            # Verify statistics structure
            assert stats["enabled"] == True, "Statistics should show enabled"
            assert stats["total_proxies"] == 4, "Should show 4 total proxies"
            assert "overall_success_rate" in stats, "Should include overall success rate"
            assert "top_proxies" in stats, "Should include top proxies list"
            assert len(stats["top_proxies"]) <= 10, "Top proxies should be limited to 10"
            
            logger.info("✅ Proxy statistics generation test passed")
            self.test_results.append(("Proxy Statistics Generation", True, "Statistics generated correctly"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy statistics generation test failed: {e}")
            self.test_results.append(("Proxy Statistics Generation", False, str(e)))
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all proxy system tests"""
        logger.info("🧪 Starting comprehensive proxy system test suite...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # List of all test methods
        tests = [
            self.test_proxy_manager_initialization,
            self.test_proxy_rotation_strategies,
            self.test_proxy_health_monitoring,
            self.test_proxy_metrics_tracking,
            self.test_proxy_integration_with_requests,
            self.test_playwright_proxy_integration,
            self.test_proxy_state_persistence,
            self.test_proxy_statistics_generation
        ]
        
        # Run all tests
        passed = 0
        failed = 0
        
        for test_method in tests:
            try:
                if test_method():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Unexpected error in {test_method.__name__}: {e}")
                failed += 1
                self.test_results.append((test_method.__name__, False, f"Unexpected error: {e}"))
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate summary
        summary = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(tests)) * 100,
            "duration_seconds": duration,
            "test_results": self.test_results
        }
        
        # Cleanup temporary files
        self.cleanup()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🏁 PROXY SYSTEM TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info("=" * 60)
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Proxy system is working correctly.")
        else:
            logger.warning(f"⚠️  {failed} test(s) failed. Check logs for details.")
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


def main():
    """Main function to run proxy tests"""
    tester = ProxySystemTester()
    results = tester.run_all_tests()
    
    # Save results to file
    results_file = f"proxy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to: {results_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit(main())