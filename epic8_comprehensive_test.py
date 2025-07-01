#!/usr/bin/env python3
"""
Comprehensive test suite for EPIC 8 - Advanced ML & AI Features
Tests all components: Summarization, Translation, Content Generation, Advanced NLU
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class EPIC8TestSuite:
    """Comprehensive test suite for all EPIC 8 components."""
    
    def __init__(self):
        self.test_results = {
            'summarization': {},
            'translation': {},
            'content_generation': {},
            'advanced_nlu': {},
            'integration': {},
            'performance': {}
        }
        
        # Sample test data
        self.test_article = """
        Technology giants continue to reshape the global economy as artificial intelligence 
        advances accelerate. Apple Inc. reported record quarterly revenue of $123.9 billion 
        in Q4 2023, driven by strong iPhone sales and growing services revenue. The company's 
        AI initiatives, including enhanced Siri capabilities and machine learning features, 
        contributed significantly to user engagement. Meanwhile, regulatory scrutiny from the 
        SEC has intensified, with new compliance requirements affecting how tech companies 
        handle user data. Industry analysts predict that AI integration will drive the next 
        wave of innovation, potentially adding $2.3 trillion to the global economy by 2030. 
        The merger between DataCorp and AI Solutions LLC, valued at $4.5 billion, represents 
        the largest AI-focused acquisition this year. CEO Sarah Johnson stated that this 
        strategic move will accelerate product development and expand market reach.
        """
        
        self.test_questions = [
            "What was Apple's quarterly revenue in Q4 2023?",
            "Who is the CEO mentioned in the article?",
            "What is the value of the DataCorp merger?",
            "When is AI expected to add value to the global economy?",
            "Why did Apple report record revenue?"
        ]
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🧪 EPIC 8 - Advanced ML & AI Features - Comprehensive Test Suite")
        print("=" * 80)
        
        total_start = time.time()
        
        # Test all components
        print("\n🔄 Testing Summarization Engine...")
        self.test_summarization()
        
        print("\n🔄 Testing Translation Engine...")
        self.test_translation()
        
        print("\n🔄 Testing Content Generation...")
        self.test_content_generation()
        
        print("\n🔄 Testing Advanced NLU...")
        self.test_advanced_nlu()
        
        print("\n🔄 Testing Integration...")
        self.test_integration()
        
        print("\n🔄 Testing Performance...")
        self.test_performance()
        
        total_time = time.time() - total_start
        
        # Generate final report
        report = self.generate_test_report(total_time)
        
        print("\n" + "=" * 80)
        print("🎉 EPIC 8 Test Suite Completed!")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        
        return report
    
    def test_summarization(self) -> None:
        """Test summarization engine functionality."""
        try:
            from summarization_engine import SummarizationEngine
            
            engine = SummarizationEngine()
            start_time = time.time()
            
            # Test different summary formats
            formats_to_test = ['custom', 'bullet_points', 'executive', 'tweet']
            format_results = {}
            
            for format_type in formats_to_test:
                try:
                    result = engine.generate_summary(
                        text=self.test_article,
                        format_type=format_type
                    )
                    
                    format_results[format_type] = {
                        'success': result.get('success', False),
                        'quality_score': result.get('quality_score', 0),
                        'compression_ratio': result.get('compression_ratio', 0),
                        'processing_time': result.get('processing_time_seconds', 0)
                    }
                    
                    status = "✅" if result.get('success') else "❌"
                    quality = result.get('quality_score', 0)
                    print(f"  {status} {format_type}: Quality {quality:.2f}")
                    
                except Exception as e:
                    format_results[format_type] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ❌ {format_type}: Error - {e}")
            
            # Test batch summarization
            try:
                batch_articles = [
                    {'id': 'test_1', 'content': self.test_article},
                    {'id': 'test_2', 'content': self.test_article[:500]}
                ]
                
                batch_results = engine.batch_summarize(batch_articles, format_type='custom')
                batch_success = all(r.get('success', False) for r in batch_results)
                
                print(f"  {'✅' if batch_success else '❌'} Batch summarization: {len(batch_results)} processed")
                
            except Exception as e:
                batch_success = False
                print(f"  ❌ Batch summarization failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['summarization'] = {
                'engine_loaded': True,
                'format_results': format_results,
                'batch_success': batch_success,
                'total_time': total_time,
                'overall_success': all(r.get('success', False) for r in format_results.values()) and batch_success
            }
            
        except Exception as e:
            print(f"  ❌ Summarization engine failed to load: {e}")
            self.test_results['summarization'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_translation(self) -> None:
        """Test translation engine functionality."""
        try:
            from translation_engine import TranslationEngine
            
            engine = TranslationEngine()
            start_time = time.time()
            
            # Test different target languages
            languages_to_test = ['es', 'fr', 'de']
            language_results = {}
            
            test_text = "Hello, this is a test of the translation system."
            
            for target_lang in languages_to_test:
                try:
                    result = engine.translate_text(
                        text=test_text,
                        target_language=target_lang,
                        source_language='en'
                    )
                    
                    language_results[target_lang] = {
                        'success': result.get('success', False),
                        'quality_score': result.get('quality_score', 0),
                        'detected_source': result.get('source_language', ''),
                        'processing_time': result.get('processing_time_seconds', 0)
                    }
                    
                    status = "✅" if result.get('success') else "❌"
                    quality = result.get('quality_score', 0)
                    print(f"  {status} EN→{target_lang.upper()}: Quality {quality:.2f}")
                    
                except Exception as e:
                    language_results[target_lang] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ❌ EN→{target_lang.upper()}: Error - {e}")
            
            # Test language detection
            try:
                detected_lang, confidence = engine.detect_language(test_text)
                detection_success = detected_lang == 'en' and confidence > 0.5
                print(f"  {'✅' if detection_success else '❌'} Language detection: {detected_lang} ({confidence:.2f})")
                
            except Exception as e:
                detection_success = False
                print(f"  ❌ Language detection failed: {e}")
            
            # Test batch translation
            try:
                batch_texts = [test_text, "This is another test sentence."]
                batch_results = engine.batch_translate(batch_texts, target_language='es')
                batch_success = all(r.get('success', False) for r in batch_results)
                
                print(f"  {'✅' if batch_success else '❌'} Batch translation: {len(batch_results)} processed")
                
            except Exception as e:
                batch_success = False
                print(f"  ❌ Batch translation failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['translation'] = {
                'engine_loaded': True,
                'language_results': language_results,
                'detection_success': detection_success,
                'batch_success': batch_success,
                'total_time': total_time,
                'overall_success': all(r.get('success', False) for r in language_results.values()) and detection_success and batch_success
            }
            
        except Exception as e:
            print(f"  ❌ Translation engine failed to load: {e}")
            self.test_results['translation'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_content_generation(self) -> None:
        """Test content generation functionality."""
        try:
            from content_generation import ContentGenerationEngine
            
            engine = ContentGenerationEngine()
            start_time = time.time()
            
            # Test newsletter generation
            try:
                newsletter_result = engine.generate_newsletter(
                    date_range_days=1,
                    user_preferences={'interests': ['technology', 'business']}
                )
                
                newsletter_success = newsletter_result.get('success', False)
                print(f"  {'✅' if newsletter_success else '❌'} Newsletter generation")
                
            except Exception as e:
                newsletter_success = False
                print(f"  ❌ Newsletter generation failed: {e}")
            
            # Test executive briefing
            try:
                briefing_result = engine.generate_executive_briefing(
                    focus_areas=['technology', 'business'],
                    date_range_days=7,
                    executive_level='senior'
                )
                
                briefing_success = briefing_result.get('success', False)
                print(f"  {'✅' if briefing_success else '❌'} Executive briefing generation")
                
            except Exception as e:
                briefing_success = False
                print(f"  ❌ Executive briefing failed: {e}")
            
            # Test trend report
            try:
                trend_result = engine.generate_trend_report(
                    analysis_period_days=7,
                    trend_categories=['technology']
                )
                
                trend_success = trend_result.get('success', False)
                print(f"  {'✅' if trend_success else '❌'} Trend report generation")
                
            except Exception as e:
                trend_success = False
                print(f"  ❌ Trend report failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['content_generation'] = {
                'engine_loaded': True,
                'newsletter_success': newsletter_success,
                'briefing_success': briefing_success,
                'trend_success': trend_success,
                'total_time': total_time,
                'overall_success': newsletter_success and briefing_success and trend_success
            }
            
        except Exception as e:
            print(f"  ❌ Content generation engine failed to load: {e}")
            self.test_results['content_generation'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_advanced_nlu(self) -> None:
        """Test advanced NLU functionality."""
        try:
            from advanced_nlu import AdvancedNLUEngine
            
            engine = AdvancedNLUEngine()
            start_time = time.time()
            
            # Test question answering
            qa_results = []
            for question in self.test_questions:
                try:
                    result = engine.answer_question(
                        question=question,
                        context=self.test_article,
                        search_articles=False
                    )
                    
                    qa_results.append({
                        'question': question,
                        'success': result.get('success', False),
                        'confidence': result.get('confidence', 0),
                        'quality_score': result.get('quality_score', 0)
                    })
                    
                    status = "✅" if result.get('success') else "❌"
                    confidence = result.get('confidence', 0)
                    print(f"  {status} QA: {question[:40]}... (Conf: {confidence:.2f})")
                    
                except Exception as e:
                    qa_results.append({
                        'question': question,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"  ❌ QA failed for: {question[:40]}...")
            
            qa_success = all(r.get('success', False) for r in qa_results)
            
            # Test fact extraction
            try:
                fact_result = engine.extract_facts(
                    text=self.test_article,
                    fact_types=['financial', 'corporate', 'temporal'],
                    include_verification=True
                )
                
                fact_success = fact_result.get('success', False)
                total_facts = fact_result.get('total_facts', 0)
                print(f"  {'✅' if fact_success else '❌'} Fact extraction: {total_facts} facts found")
                
            except Exception as e:
                fact_success = False
                print(f"  ❌ Fact extraction failed: {e}")
            
            # Test claim detection
            try:
                claim_result = engine.detect_claims(
                    text=self.test_article,
                    verify_claims=True
                )
                
                claim_success = claim_result.get('success', False)
                total_claims = claim_result.get('total_claims', 0)
                print(f"  {'✅' if claim_success else '❌'} Claim detection: {total_claims} claims found")
                
            except Exception as e:
                claim_success = False
                print(f"  ❌ Claim detection failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['advanced_nlu'] = {
                'engine_loaded': True,
                'qa_results': qa_results,
                'qa_success': qa_success,
                'fact_success': fact_success,
                'claim_success': claim_success,
                'total_time': total_time,
                'overall_success': qa_success and fact_success and claim_success
            }
            
        except Exception as e:
            print(f"  ❌ Advanced NLU engine failed to load: {e}")
            self.test_results['advanced_nlu'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_integration(self) -> None:
        """Test integration between different EPIC 8 components."""
        start_time = time.time()
        
        try:
            # Test summarization + translation workflow
            from summarization_engine import SummarizationEngine
            from translation_engine import TranslationEngine
            
            summarizer = SummarizationEngine()
            translator = TranslationEngine()
            
            # Generate summary and translate it
            summary_result = summarizer.generate_summary(
                text=self.test_article,
                format_type='executive'
            )
            
            if summary_result.get('success'):
                translation_result = translator.translate_text(
                    text=summary_result['summary'],
                    target_language='es',
                    source_language='en'
                )
                
                workflow_success = translation_result.get('success', False)
                print(f"  {'✅' if workflow_success else '❌'} Summarization + Translation workflow")
            else:
                workflow_success = False
                print(f"  ❌ Summarization failed in integration test")
            
        except Exception as e:
            workflow_success = False
            print(f"  ❌ Integration workflow failed: {e}")
        
        try:
            # Test content generation + NLU workflow
            from content_generation import ContentGenerationEngine
            from advanced_nlu import AdvancedNLUEngine
            
            content_gen = ContentGenerationEngine()
            nlu = AdvancedNLUEngine()
            
            # Generate newsletter and extract facts from it
            newsletter_result = content_gen.generate_newsletter(date_range_days=1)
            
            if newsletter_result.get('success'):
                fact_result = nlu.extract_facts(
                    text=newsletter_result['content'],
                    fact_types=['financial', 'temporal']
                )
                
                nlu_workflow_success = fact_result.get('success', False)
                print(f"  {'✅' if nlu_workflow_success else '❌'} Content Generation + NLU workflow")
            else:
                nlu_workflow_success = False
                print(f"  ❌ Content generation failed in NLU workflow")
            
        except Exception as e:
            nlu_workflow_success = False
            print(f"  ❌ Content + NLU workflow failed: {e}")
        
        total_time = time.time() - start_time
        
        self.test_results['integration'] = {
            'summarization_translation_workflow': workflow_success,
            'content_generation_nlu_workflow': nlu_workflow_success,
            'total_time': total_time,
            'overall_success': workflow_success and nlu_workflow_success
        }
    
    def test_performance(self) -> None:
        """Test performance characteristics of EPIC 8 components."""
        start_time = time.time()
        
        performance_metrics = {}
        
        # Test summarization performance
        try:
            from summarization_engine import SummarizationEngine
            engine = SummarizationEngine()
            
            perf_start = time.time()
            result = engine.generate_summary(text=self.test_article, format_type='custom')
            summarization_time = time.time() - perf_start
            
            performance_metrics['summarization'] = {
                'time_seconds': summarization_time,
                'success': result.get('success', False),
                'characters_per_second': len(self.test_article) / summarization_time if summarization_time > 0 else 0
            }
            
            print(f"  📊 Summarization: {summarization_time:.2f}s ({performance_metrics['summarization']['characters_per_second']:.0f} chars/sec)")
            
        except Exception as e:
            performance_metrics['summarization'] = {'error': str(e)}
            print(f"  ❌ Summarization performance test failed: {e}")
        
        # Test translation performance
        try:
            from translation_engine import TranslationEngine
            engine = TranslationEngine()
            
            test_text = "This is a performance test for the translation engine."
            
            perf_start = time.time()
            result = engine.translate_text(text=test_text, target_language='es', source_language='en')
            translation_time = time.time() - perf_start
            
            performance_metrics['translation'] = {
                'time_seconds': translation_time,
                'success': result.get('success', False),
                'characters_per_second': len(test_text) / translation_time if translation_time > 0 else 0
            }
            
            print(f"  📊 Translation: {translation_time:.2f}s ({performance_metrics['translation']['characters_per_second']:.0f} chars/sec)")
            
        except Exception as e:
            performance_metrics['translation'] = {'error': str(e)}
            print(f"  ❌ Translation performance test failed: {e}")
        
        # Test NLU performance
        try:
            from advanced_nlu import AdvancedNLUEngine
            engine = AdvancedNLUEngine()
            
            perf_start = time.time()
            result = engine.answer_question(
                question="What is this article about?",
                context=self.test_article,
                search_articles=False
            )
            nlu_time = time.time() - perf_start
            
            performance_metrics['nlu'] = {
                'time_seconds': nlu_time,
                'success': result.get('success', False),
                'context_length': len(self.test_article)
            }
            
            print(f"  📊 NLU QA: {nlu_time:.2f}s")
            
        except Exception as e:
            performance_metrics['nlu'] = {'error': str(e)}
            print(f"  ❌ NLU performance test failed: {e}")
        
        total_time = time.time() - start_time
        
        # Performance thresholds (acceptable response times)
        thresholds = {
            'summarization': 10.0,  # 10 seconds
            'translation': 5.0,     # 5 seconds
            'nlu': 8.0              # 8 seconds
        }
        
        performance_passed = True
        for component, metrics in performance_metrics.items():
            if 'time_seconds' in metrics and component in thresholds:
                if metrics['time_seconds'] > thresholds[component]:
                    performance_passed = False
                    print(f"  ⚠️ {component} exceeded threshold: {metrics['time_seconds']:.2f}s > {thresholds[component]}s")
        
        self.test_results['performance'] = {
            'metrics': performance_metrics,
            'thresholds': thresholds,
            'total_time': total_time,
            'performance_passed': performance_passed,
            'overall_success': performance_passed
        }
    
    def generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        # Calculate overall success
        component_successes = []
        for component, results in self.test_results.items():
            component_successes.append(results.get('overall_success', False))
        
        overall_success = all(component_successes)
        success_rate = sum(component_successes) / len(component_successes) * 100
        
        # Component summary
        component_summary = {}
        for component, results in self.test_results.items():
            component_summary[component] = {
                'success': results.get('overall_success', False),
                'time': results.get('total_time', 0),
                'details': results
            }
        
        report = {
            'test_execution': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'overall_success': overall_success,
                'success_rate_percentage': success_rate
            },
            'component_results': component_summary,
            'epic8_status': 'PASSED' if overall_success else 'FAILED',
            'recommendations': self._generate_recommendations()
        }
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"  Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Components Tested: {len(component_summary)}")
        print(f"  Total Time: {total_time:.2f}s")
        
        print(f"\n📋 Component Status:")
        for component, summary in component_summary.items():
            status = "✅" if summary['success'] else "❌"
            time_str = f"{summary['time']:.2f}s"
            print(f"  {status} {component.replace('_', ' ').title()}: {time_str}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for component, results in self.test_results.items():
            if not results.get('overall_success', False):
                if 'error' in results:
                    recommendations.append(f"Fix {component} initialization: {results['error']}")
                else:
                    recommendations.append(f"Investigate {component} failures and improve reliability")
            
            # Performance recommendations
            if component == 'performance' and results.get('performance_passed', True) == False:
                recommendations.append("Optimize component performance to meet response time thresholds")
        
        if not recommendations:
            recommendations.append("All components passed testing - system is ready for production")
        
        return recommendations


def main():
    """Main test execution function."""
    print("🚀 Starting EPIC 8 Comprehensive Test Suite")
    print("Testing: Summarization, Translation, Content Generation, Advanced NLU")
    print()
    
    # Set environment variables
    os.environ["USE_POSTGRES"] = "true"
    
    # Initialize and run tests
    test_suite = EPIC8TestSuite()
    
    try:
        report = test_suite.run_comprehensive_test()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"epic8_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Test report saved to: {report_file}")
        
        # Exit with appropriate code
        exit_code = 0 if report['test_execution']['overall_success'] else 1
        print(f"\n🏁 Test suite completed with exit code: {exit_code}")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️ Test suite interrupted by user")
        return 2
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        logger.exception("Test suite execution failed")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)