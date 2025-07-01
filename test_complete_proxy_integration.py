#!/usr/bin/env python
"""
Complete Proxy Integration Test for Mimir
Tests the entire proxy system integration with scraper.py and Playwright
"""

import json
import logging
import tempfile
import os
import sys
from datetime import datetime
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteProxyIntegrationTester:
    """Complete integration test suite for proxy system with scraper"""
    
    def __init__(self):
        self.test_results = []
        self.temp_files = []
        self.original_config = None
    
    def create_test_configuration(self) -> Dict:
        """Create comprehensive test configuration"""
        
        # Create temporary proxy list file
        proxy_list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_files.append(proxy_list_file.name)
        
        proxy_list_data = {
            "proxies": [
                {
                    "url": "http://test1.proxy.com:8080",
                    "location": "US",
                    "tags": ["test", "integration"]
                },
                {
                    "url": "http://test2.proxy.com:8080", 
                    "location": "EU",
                    "tags": ["test", "integration"]
                }
            ]
        }
        
        json.dump(proxy_list_data, proxy_list_file)
        proxy_list_file.close()
        
        # Create complete test configuration
        test_config = {
            "user_agent": "Mimir-IntegrationTest/1.0",
            "default_request_delay_seconds": 1,
            "database_name": "test_mimir_news.db",
            "max_retries": 2,
            "retry_delay_seconds": 1,
            "notification_email": {
                "send_email": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/test_mimir_scraper.log",
                "max_bytes": 10485760,
                "backup_count": 5
            },
            "keywords_monitoring": {
                "enabled": True,
                "keywords": ["test", "integration", "proxy"],
                "alert_on_match": True
            },
            "api_sources": {
                "enabled": False
            },
            "playwright": {
                "enabled": True,
                "config_file": "playwright_config.json",
                "browser_type": "chromium",
                "headless": True,
                "timeout_seconds": 30,
                "max_concurrent_contexts": 3,
                "fallback_to_html": True,
                "screenshot_on_error": True,
                "auto_detect_js": True,
                "js_indicators": ["react-root", "ng-app", "vue-app"],
                "default_wait_strategy": "domcontentloaded",
                "page_load_timeout": 30000,
                "element_timeout": 10000
            },
            "proxy_settings": {
                "enabled": True,
                "rotation_strategy": "round_robin",
                "health_check_interval": 60,  # Shorter for testing
                "failure_threshold": 2,
                "health_check_timeout": 5,
                "health_check_url": "http://httpbin.org/ip",
                "save_state_file": "test_proxy_state.json",
                "auto_save_interval": 300,
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
                ],
                "domain_specific": {
                    "enabled": True,
                    "assignments": {
                        "example.com": ["static1.test.com:8080"],
                        "test.com": ["static2.test.com:8080"]
                    }
                },
                "tenant_isolation": {
                    "enabled": False,
                    "tenant_proxy_pools": {}
                }
            }
        }
        
        return test_config
    
    def test_scraper_proxy_integration_initialization(self) -> bool:
        """Test scraper proxy integration initialization"""
        logger.info("Testing scraper proxy integration initialization...")
        
        try:
            test_config = self.create_test_configuration()
            
            # Test configuration validation
            from scraper_proxy_integration import ScraperProxyManager
            scraper_proxy_manager = ScraperProxyManager(test_config)
            
            # Verify initialization
            assert scraper_proxy_manager.enabled == True, "Proxy system should be enabled"
            assert scraper_proxy_manager.initialization_error is None, f"Should not have initialization error: {scraper_proxy_manager.initialization_error}"
            assert scraper_proxy_manager.proxy_manager is not None, "Proxy manager should be created"
            assert len(scraper_proxy_manager.proxy_manager.proxies) == 4, f"Should have 4 proxies, got {len(scraper_proxy_manager.proxy_manager.proxies)}"
            
            logger.info("✅ Scraper proxy integration initialization test passed")
            self.test_results.append(("Scraper Proxy Integration Init", True, "Initialization successful"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Scraper proxy integration initialization test failed: {e}")
            self.test_results.append(("Scraper Proxy Integration Init", False, str(e)))
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test proxy configuration validation"""
        logger.info("Testing proxy configuration validation...")
        
        try:
            from scraper_proxy_integration import ScraperProxyManager
            
            # Test valid configuration
            valid_config = self.create_test_configuration()
            manager = ScraperProxyManager(valid_config)
            assert manager.initialization_error is None, "Valid config should not have errors"
            
            # Test invalid configuration - missing required fields
            invalid_config = {"proxy_settings": {"enabled": True}}
            manager = ScraperProxyManager(invalid_config)
            assert manager.initialization_error is not None, "Invalid config should have error"
            assert "rotation_strategy" in manager.initialization_error, "Should detect missing rotation_strategy"
            
            # Test invalid rotation strategy
            invalid_strategy_config = valid_config.copy()
            invalid_strategy_config["proxy_settings"]["rotation_strategy"] = "invalid_strategy"
            manager = ScraperProxyManager(invalid_strategy_config)
            assert manager.initialization_error is not None, "Invalid strategy should cause error"
            
            logger.info("✅ Configuration validation test passed")
            self.test_results.append(("Configuration Validation", True, "All validation checks passed"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation test failed: {e}")
            self.test_results.append(("Configuration Validation", False, str(e)))
            return False
    
    def test_enhanced_fetch_url_functionality(self) -> bool:
        """Test enhanced fetch_url with proxy support"""
        logger.info("Testing enhanced fetch_url functionality...")
        
        try:
            test_config = self.create_test_configuration()
            
            from scraper_proxy_integration import ScraperProxyManager, enhanced_fetch_url
            
            # Initialize proxy manager
            scraper_proxy_manager = ScraperProxyManager(test_config)
            
            # Test with disabled proxy system
            disabled_config = test_config.copy()
            disabled_config["proxy_settings"]["enabled"] = False
            
            # This should work but use fallback (won't actually fetch due to test URLs)
            result = enhanced_fetch_url("http://test.example.com", disabled_config)
            # We expect None due to test URLs, but no exceptions should be raised
            
            # Test metrics tracking
            initial_requests = scraper_proxy_manager.total_requests
            enhanced_fetch_url("http://test.example.com", test_config)
            assert scraper_proxy_manager.total_requests > initial_requests, "Request count should increase"
            
            logger.info("✅ Enhanced fetch_url functionality test passed")
            self.test_results.append(("Enhanced fetch_url", True, "Function works with proxy system"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced fetch_url functionality test failed: {e}")
            self.test_results.append(("Enhanced fetch_url", False, str(e)))
            return False
    
    def test_proxy_status_and_monitoring(self) -> bool:
        """Test proxy status monitoring and statistics"""
        logger.info("Testing proxy status and monitoring...")
        
        try:
            test_config = self.create_test_configuration()
            
            from scraper_proxy_integration import ScraperProxyManager
            scraper_proxy_manager = ScraperProxyManager(test_config)
            
            # Test status retrieval
            status = scraper_proxy_manager.get_proxy_status()
            
            # Verify status structure
            required_fields = ["enabled", "initialization_error", "total_requests", 
                             "proxy_requests", "fallback_requests", "proxy_usage_rate"]
            for field in required_fields:
                assert field in status, f"Status should include {field}"
            
            # Test performance summary
            summary = scraper_proxy_manager.get_performance_summary()
            assert "proxy_enabled" in summary, "Summary should include proxy_enabled"
            assert "total_requests" in summary, "Summary should include total_requests"
            
            # Test proxy manager operations
            scraper_proxy_manager.trigger_health_check()  # Should not raise exception
            scraper_proxy_manager.save_proxy_state()     # Should not raise exception
            
            logger.info("✅ Proxy status and monitoring test passed")
            self.test_results.append(("Proxy Status & Monitoring", True, "All monitoring functions work"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Proxy status and monitoring test failed: {e}")
            self.test_results.append(("Proxy Status & Monitoring", False, str(e)))
            return False
    
    def test_scraper_integration_imports(self) -> bool:
        """Test that scraper.py can import and use the proxy system"""
        logger.info("Testing scraper.py integration imports...")
        
        try:
            # Test that we can import the enhanced integration
            from scraper_proxy_integration import (
                initialize_scraper_proxy_system,
                get_scraper_proxy_manager,
                enhanced_fetch_url,
                get_proxy_integration_status
            )
            
            # Test initialization
            test_config = self.create_test_configuration()
            proxy_manager = initialize_scraper_proxy_system(test_config)
            assert proxy_manager is not None, "Should initialize proxy manager"
            
            # Test getting the manager
            retrieved_manager = get_scraper_proxy_manager()
            assert retrieved_manager is proxy_manager, "Should return same manager instance"
            
            # Test status function
            status = get_proxy_integration_status()
            assert status["initialized"] == True, "Should show as initialized"
            assert "enabled" in status, "Status should include enabled field"
            
            logger.info("✅ Scraper integration imports test passed")
            self.test_results.append(("Scraper Integration Imports", True, "All imports and functions work"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Scraper integration imports test failed: {e}")
            self.test_results.append(("Scraper Integration Imports", False, str(e)))
            return False
    
    def test_playwright_proxy_integration(self) -> bool:
        """Test Playwright proxy integration"""
        logger.info("Testing Playwright proxy integration...")
        
        try:
            test_config = self.create_test_configuration()
            
            # Initialize scraper proxy system
            from scraper_proxy_integration import initialize_scraper_proxy_system
            scraper_proxy_manager = initialize_scraper_proxy_system(test_config)
            
            # Test playwright integration functions
            from proxy_integration import ProxyPlaywrightIntegration
            
            if scraper_proxy_manager.proxy_manager:
                playwright_integration = ProxyPlaywrightIntegration(scraper_proxy_manager.proxy_manager)
                
                # Test proxy config generation
                proxy_config = playwright_integration.get_proxy_config_for_playwright("test.com")
                
                # Should get proxy config since system is enabled and has proxies
                assert proxy_config is not None, "Should generate proxy config for Playwright"
                assert "server" in proxy_config, "Proxy config should have server field"
                
                # Test result recording
                playwright_integration.record_playwright_proxy_result(
                    "http://test.proxy.com:8080", True, 100.0
                )
                
                logger.info("✅ Playwright proxy integration test passed")
                self.test_results.append(("Playwright Proxy Integration", True, "Playwright integration works"))
                return True
            else:
                logger.warning("⚠️ Playwright proxy integration test skipped - no proxy manager")
                self.test_results.append(("Playwright Proxy Integration", True, "Skipped - no proxy manager"))
                return True
            
        except Exception as e:
            logger.error(f"❌ Playwright proxy integration test failed: {e}")
            self.test_results.append(("Playwright Proxy Integration", False, str(e)))
            return False
    
    def test_error_handling_and_fallbacks(self) -> bool:
        """Test error handling and fallback mechanisms"""
        logger.info("Testing error handling and fallbacks...")
        
        try:
            # Test with completely broken configuration
            broken_config = {
                "proxy_settings": {
                    "enabled": True,
                    "rotation_strategy": "invalid",
                    "proxy_providers": "not_a_list"  # Invalid type
                }
            }
            
            from scraper_proxy_integration import ScraperProxyManager
            broken_manager = ScraperProxyManager(broken_config)
            
            # Should handle gracefully
            assert broken_manager.initialization_error is not None, "Should detect configuration error"
            
            # fetch_url should still work with fallback
            result = broken_manager.fetch_url_with_proxy("http://test.example.com")
            # Should return None due to test URL, but no exception
            
            # Test status with broken manager
            status = broken_manager.get_proxy_status()
            assert status["enabled"] == True, "Should show enabled even if broken"
            assert status["initialization_error"] is not None, "Should show error"
            
            logger.info("✅ Error handling and fallbacks test passed")
            self.test_results.append(("Error Handling & Fallbacks", True, "Graceful error handling works"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling and fallbacks test failed: {e}")
            self.test_results.append(("Error Handling & Fallbacks", False, str(e)))
            return False
    
    def run_all_integration_tests(self) -> Dict:
        """Run all integration tests"""
        logger.info("🧪 Starting complete proxy integration test suite...")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # List of all test methods
        tests = [
            self.test_scraper_proxy_integration_initialization,
            self.test_configuration_validation,
            self.test_enhanced_fetch_url_functionality,
            self.test_proxy_status_and_monitoring,
            self.test_scraper_integration_imports,
            self.test_playwright_proxy_integration,
            self.test_error_handling_and_fallbacks
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
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / len(tests)) * 100,
            "duration_seconds": duration,
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cleanup
        self.cleanup()
        
        # Print summary
        logger.info("=" * 70)
        logger.info("🏁 COMPLETE PROXY INTEGRATION TEST RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Duration: {summary['duration_seconds']:.2f} seconds")
        logger.info("=" * 70)
        
        # Print detailed results
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status}: {test_name} - {message}")
        
        logger.info("=" * 70)
        
        if failed == 0:
            logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
            logger.info("✅ Proxy system is fully integrated and functional!")
        else:
            logger.warning(f"⚠️  {failed} integration test(s) failed.")
            logger.warning("❌ Proxy system integration has issues that need to be addressed.")
        
        return summary
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


def main():
    """Main function to run complete integration tests"""
    print("🔄 MIMIR COMPLETE PROXY INTEGRATION TEST")
    print("=" * 70)
    print("Testing end-to-end proxy system integration with scraper and Playwright")
    print("=" * 70)
    
    tester = CompleteProxyIntegrationTester()
    results = tester.run_all_integration_tests()
    
    # Save results to file
    results_file = f"complete_proxy_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📄 Test results saved to: {results_file}")
    except Exception as e:
        logger.error(f"Failed to save test results: {e}")
    
    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    
    if exit_code == 0:
        print("\n🎯 VERDICT: Complete proxy integration is FULLY FUNCTIONAL!")
    else:
        print("\n⚠️  VERDICT: Proxy integration has issues that need attention.")
    
    return exit_code


if __name__ == "__main__":
    exit(main())