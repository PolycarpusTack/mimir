"""
Proxy Production Deployment Assistant for Mimir
Helps users safely deploy proxy system from development to production
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import getpass
import subprocess

from proxy_provider_templates import ProxyProviderManager
from proxy_config_validator import ProxyConfigValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxyProductionAssistant:
    """Interactive assistant for production proxy deployment"""
    
    def __init__(self):
        self.config_file = Path("config.json")
        self.deployment_config = {}
        self.provider_manager = ProxyProviderManager()
        self.validator = ProxyConfigValidator()
    
    def welcome_message(self):
        """Display welcome message"""
        print("=" * 70)
        print("🚀 MIMIR PROXY SYSTEM - PRODUCTION DEPLOYMENT ASSISTANT")
        print("=" * 70)
        print("This assistant will help you configure and deploy the proxy system")
        print("for production use with real proxy providers.")
        print("=" * 70)
        print()
    
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met"""
        print("📋 Checking prerequisites...")
        
        prerequisites = [
            ("Configuration file exists", self.config_file.exists()),
            ("Proxy system components installed", self._check_proxy_components()),
            ("Internet connectivity", self._check_internet()),
        ]
        
        all_met = True
        for desc, status in prerequisites:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {desc}")
            if not status:
                all_met = False
        
        if not all_met:
            print("\n❌ Prerequisites not met. Please resolve issues above.")
            return False
        
        print("\n✅ All prerequisites met!")
        return True
    
    def _check_proxy_components(self) -> bool:
        """Check if proxy system components are available"""
        try:
            from proxy_manager import ProxyManager
            from proxy_integration import ProxyEnabledSession
            from scraper_proxy_integration import ScraperProxyManager
            return True
        except ImportError:
            return False
    
    def _check_internet(self) -> bool:
        """Check internet connectivity"""
        try:
            import requests
            response = requests.get("http://httpbin.org/ip", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def analyze_current_configuration(self) -> Dict[str, Any]:
        """Analyze current proxy configuration"""
        print("🔍 Analyzing current configuration...")
        
        summary = self.validator.run_comprehensive_validation()
        
        print(f"   📊 Validation results:")
        print(f"      - Critical issues: {summary['critical_issues']}")
        print(f"      - Warnings: {summary['warnings']}")
        print(f"      - Auto-fixes applied: {summary['fixes_applied']}")
        
        if summary['critical_issues'] > 0:
            print(f"\n⚠️  Found {summary['critical_issues']} critical issues.")
            print("   Run 'python proxy_config_validator.py --report' for details.")
        
        return summary
    
    def choose_deployment_type(self) -> str:
        """Let user choose deployment type"""
        print("\n🎯 Choose your deployment type:")
        print("   1. Small-scale (1-10 concurrent requests)")
        print("   2. Medium-scale (10-100 concurrent requests)")  
        print("   3. Enterprise-scale (100+ concurrent requests)")
        print("   4. Custom configuration")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                types = {
                    "1": "small",
                    "2": "medium", 
                    "3": "enterprise",
                    "4": "custom"
                }
                return types[choice]
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    def choose_proxy_providers(self) -> List[str]:
        """Let user choose proxy providers"""
        print("\n🌐 Available proxy providers:")
        available_providers = self.provider_manager.get_available_providers()
        
        for i, provider in enumerate(available_providers, 1):
            print(f"   {i}. {provider.title()}")
        
        print("   0. Skip provider setup (manual configuration)")
        
        selected = []
        while True:
            choice = input(f"\nSelect providers (1-{len(available_providers)}, 0 to skip, or 'done'): ").strip().lower()
            
            if choice == "done":
                break
            elif choice == "0":
                print("   Skipping provider setup - you'll need to configure manually.")
                break
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_providers):
                    provider = available_providers[idx]
                    if provider not in selected:
                        selected.append(provider)
                        print(f"   Added: {provider}")
                    else:
                        print(f"   {provider} already selected")
                else:
                    print("   Invalid selection")
            else:
                print("   Enter numbers, 'done', or '0'")
        
        return selected
    
    def configure_provider_credentials(self, providers: List[str]) -> Dict[str, bool]:
        """Configure credentials for selected providers"""
        results = {}
        
        if not providers:
            return results
        
        print("\n🔐 Configuring provider credentials...")
        print("We'll help you set up environment variables for authentication.")
        
        for provider in providers:
            print(f"\n--- {provider.upper()} SETUP ---")
            
            if provider == "brightdata":
                results[provider] = self._setup_brightdata_credentials()
            elif provider == "oxylabs":
                results[provider] = self._setup_oxylabs_credentials()
            elif provider == "smartproxy":
                results[provider] = self._setup_smartproxy_credentials()
            else:
                print(f"Manual setup required for {provider}")
                results[provider] = False
        
        return results
    
    def _setup_brightdata_credentials(self) -> bool:
        """Setup Bright Data credentials"""
        print("Bright Data requires username, password, and zone.")
        print("Get these from your Bright Data dashboard.")
        
        username = input("Enter Bright Data username: ").strip()
        if not username:
            print("Username required")
            return False
        
        password = getpass.getpass("Enter Bright Data password: ").strip()
        if not password:
            print("Password required")
            return False
        
        zone = input("Enter Bright Data zone (default: residential): ").strip() or "residential"
        
        # Set environment variables
        env_vars = [
            f"export BRIGHTDATA_USERNAME='{username}'",
            f"export BRIGHTDATA_PASSWORD='{password}'",
            f"export BRIGHTDATA_ZONE='{zone}'"
        ]
        
        self._save_env_vars("brightdata", env_vars)
        
        # Test credentials
        os.environ["BRIGHTDATA_USERNAME"] = username
        os.environ["BRIGHTDATA_PASSWORD"] = password
        os.environ["BRIGHTDATA_ZONE"] = zone
        
        print("Testing credentials...")
        return self.provider_manager.validate_provider_credentials("brightdata")
    
    def _setup_oxylabs_credentials(self) -> bool:
        """Setup Oxylabs credentials"""
        print("Oxylabs requires username and password.")
        print("Get these from your Oxylabs dashboard.")
        
        username = input("Enter Oxylabs username: ").strip()
        if not username:
            print("Username required")
            return False
        
        password = getpass.getpass("Enter Oxylabs password: ").strip()
        if not password:
            print("Password required")
            return False
        
        # Set environment variables
        env_vars = [
            f"export OXYLABS_USERNAME='{username}'",
            f"export OXYLABS_PASSWORD='{password}'"
        ]
        
        self._save_env_vars("oxylabs", env_vars)
        
        # Test credentials
        os.environ["OXYLABS_USERNAME"] = username
        os.environ["OXYLABS_PASSWORD"] = password
        
        print("Testing credentials...")
        return self.provider_manager.validate_provider_credentials("oxylabs")
    
    def _setup_smartproxy_credentials(self) -> bool:
        """Setup SmartProxy credentials"""
        print("SmartProxy requires username and password.")
        print("Get these from your SmartProxy dashboard.")
        
        username = input("Enter SmartProxy username: ").strip()
        if not username:
            print("Username required")
            return False
        
        password = getpass.getpass("Enter SmartProxy password: ").strip()
        if not password:
            print("Password required")
            return False
        
        # Set environment variables
        env_vars = [
            f"export SMARTPROXY_USERNAME='{username}'",
            f"export SMARTPROXY_PASSWORD='{password}'"
        ]
        
        self._save_env_vars("smartproxy", env_vars)
        
        # Test credentials
        os.environ["SMARTPROXY_USERNAME"] = username
        os.environ["SMARTPROXY_PASSWORD"] = password
        
        print("Testing credentials...")
        return self.provider_manager.validate_provider_credentials("smartproxy")
    
    def _save_env_vars(self, provider: str, env_vars: List[str]) -> None:
        """Save environment variables to file"""
        env_file = Path(f".env.{provider}")
        
        with open(env_file, 'w') as f:
            f.write(f"# {provider.upper()} proxy credentials\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")
            for var in env_vars:
                f.write(var + "\n")
        
        print(f"💾 Saved credentials to {env_file}")
        print(f"   To use: source {env_file}")
    
    def generate_production_config(self, deployment_type: str, working_providers: List[str]) -> Dict[str, Any]:
        """Generate production-ready configuration"""
        print(f"\n⚙️ Generating {deployment_type} production configuration...")
        
        # Base configuration templates
        base_configs = {
            "small": {
                "health_check_interval": 300,
                "failure_threshold": 3,
                "rotation_strategy": "round_robin"
            },
            "medium": {
                "health_check_interval": 180,
                "failure_threshold": 2,
                "rotation_strategy": "health_based"
            },
            "enterprise": {
                "health_check_interval": 120,
                "failure_threshold": 2,
                "rotation_strategy": "health_based"
            }
        }
        
        base_config = base_configs.get(deployment_type, base_configs["medium"])
        
        # Generate provider configurations
        production_config = self.provider_manager.create_production_config(working_providers)
        
        # Merge with deployment-specific settings
        production_config["proxy_settings"].update(base_config)
        
        return production_config
    
    def deploy_configuration(self, production_config: Dict[str, Any]) -> bool:
        """Deploy the production configuration"""
        print("\n🚀 Deploying production configuration...")
        
        try:
            # Load current config
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    current_config = json.load(f)
            else:
                current_config = {}
            
            # Create backup
            backup_file = f"config.json.backup_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if self.config_file.exists():
                import shutil
                shutil.copy2(self.config_file, backup_file)
                print(f"   📦 Created backup: {backup_file}")
            
            # Merge configurations
            current_config.update(production_config)
            
            # Save new configuration
            with open(self.config_file, 'w') as f:
                json.dump(current_config, f, indent=4)
            
            print("   ✅ Configuration deployed successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Deployment failed: {e}")
            return False
    
    def run_production_tests(self) -> bool:
        """Run tests to verify production deployment"""
        print("\n🧪 Running production validation tests...")
        
        tests = [
            ("Configuration validation", self._test_config_validation),
            ("Proxy connectivity", self._test_proxy_connectivity),
            ("Scraper integration", self._test_scraper_integration),
        ]
        
        passed = 0
        for test_name, test_func in tests:
            print(f"   Running: {test_name}")
            try:
                if test_func():
                    print(f"      ✅ {test_name} passed")
                    passed += 1
                else:
                    print(f"      ❌ {test_name} failed")
            except Exception as e:
                print(f"      ❌ {test_name} error: {e}")
        
        success_rate = (passed / len(tests)) * 100
        print(f"\n📊 Test Results: {passed}/{len(tests)} passed ({success_rate:.1f}%)")
        
        return passed == len(tests)
    
    def _test_config_validation(self) -> bool:
        """Test configuration validation"""
        summary = self.validator.run_comprehensive_validation()
        return summary['critical_issues'] == 0
    
    def _test_proxy_connectivity(self) -> bool:
        """Test proxy connectivity"""
        try:
            from scraper_proxy_integration import get_proxy_integration_status
            status = get_proxy_integration_status()
            return status.get("initialized", False) and not status.get("error")
        except Exception:
            return False
    
    def _test_scraper_integration(self) -> bool:
        """Test scraper integration"""
        try:
            from scraper_proxy_integration import ScraperProxyManager
            
            # Load current config
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Test initialization
            manager = ScraperProxyManager(config)
            return manager.initialization_error is None
        except Exception:
            return False
    
    def provide_next_steps(self, deployment_successful: bool) -> None:
        """Provide next steps guidance"""
        print("\n" + "=" * 70)
        
        if deployment_successful:
            print("🎉 PRODUCTION DEPLOYMENT SUCCESSFUL!")
            print("=" * 70)
            print("\n📋 Next Steps:")
            print("   1. Test the scraper: python scraper.py --run")
            print("   2. Monitor proxy health: python proxy_dashboard.py")
            print("   3. Check logs for any issues")
            print("   4. Set up monitoring alerts")
            print("   5. Schedule regular scraping")
            
            print("\n📊 Monitoring Commands:")
            print("   - Proxy status: python -c \"from scraper_proxy_integration import get_proxy_integration_status; print(get_proxy_integration_status())\"")
            print("   - Health check: python proxy_config_validator.py --report")
            print("   - Dashboard: python proxy_dashboard.py (http://localhost:5001/proxy)")
            
        else:
            print("❌ DEPLOYMENT NEEDS ATTENTION")
            print("=" * 70)
            print("\n🔧 Troubleshooting Steps:")
            print("   1. Review error messages above")
            print("   2. Check proxy provider credentials")
            print("   3. Validate configuration: python proxy_config_validator.py --report")
            print("   4. Test connectivity manually")
            print("   5. Contact support if issues persist")
        
        print("\n📚 Documentation:")
        print("   - Usage guide: PROXY_SYSTEM_USAGE.md")
        print("   - Configuration reference: proxy_config_validator.py --help")
        print("   - Provider templates: proxy_provider_templates.py")
        
        print("=" * 70)
    
    def run_interactive_deployment(self) -> bool:
        """Run interactive deployment process"""
        self.welcome_message()
        
        # Step 1: Prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 2: Current configuration analysis
        config_summary = self.analyze_current_configuration()
        if config_summary['critical_issues'] > 0:
            print("\n⚠️  Critical configuration issues found.")
            fix_now = input("Apply automatic fixes now? (y/n): ").strip().lower()
            if fix_now == 'y':
                self.validator.auto_fix_issues()
                print("✅ Auto-fixes applied. Re-analyzing...")
                config_summary = self.analyze_current_configuration()
        
        # Step 3: Deployment type
        deployment_type = self.choose_deployment_type()
        
        # Step 4: Provider selection
        selected_providers = self.choose_proxy_providers()
        
        # Step 5: Credential configuration
        working_providers = []
        if selected_providers:
            credential_results = self.configure_provider_credentials(selected_providers)
            working_providers = [p for p, working in credential_results.items() if working]
            
            if not working_providers:
                print("\n⚠️  No working proxy providers configured.")
                continue_anyway = input("Continue with manual configuration? (y/n): ").strip().lower()
                if continue_anyway != 'y':
                    print("Deployment cancelled.")
                    return False
        
        # Step 6: Generate production config
        production_config = self.generate_production_config(deployment_type, working_providers)
        
        # Step 7: Deploy configuration
        if not self.deploy_configuration(production_config):
            return False
        
        # Step 8: Run tests
        tests_passed = self.run_production_tests()
        
        # Step 9: Next steps
        self.provide_next_steps(tests_passed)
        
        return tests_passed


def main():
    """Main function for production deployment assistant"""
    try:
        assistant = ProxyProductionAssistant()
        success = assistant.run_interactive_deployment()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Production deployment failed")
        return 1


if __name__ == "__main__":
    exit(main())