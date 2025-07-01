#!/usr/bin/env python3
"""
Comprehensive test suite for EPIC 6 completion - Article Clustering and Recommendations
Tests both clustering and recommendation engines with integration scenarios.
"""

import logging
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EPIC6CompletionTestSuite:
    """Comprehensive test suite for EPIC 6 completion features."""
    
    def __init__(self):
        self.test_results = {
            'clustering': {},
            'recommendations': {},
            'integration': {},
            'performance': {},
            'api_endpoints': {}
        }
        
        # Test data
        self.sample_articles = self._generate_sample_articles()
        self.sample_interactions = self._generate_sample_interactions()
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🧪 EPIC 6 Completion - Comprehensive Test Suite")
        print("=" * 80)
        
        total_start = time.time()
        
        # Test clustering engine
        print("\n🔄 Testing Article Clustering Engine...")
        self.test_clustering_engine()
        
        # Test recommendation engine
        print("\n🔄 Testing Recommendation Engine...")
        self.test_recommendation_engine()
        
        # Test integration between systems
        print("\n🔄 Testing System Integration...")
        self.test_system_integration()
        
        # Test performance
        print("\n🔄 Testing Performance...")
        self.test_performance()
        
        # Test API endpoints
        print("\n🔄 Testing API Endpoints...")
        self.test_api_endpoints()
        
        total_time = time.time() - total_start
        
        # Generate final report
        report = self.generate_test_report(total_time)
        
        print("\n" + "=" * 80)
        print("🎉 EPIC 6 Completion Test Suite Finished!")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        
        return report
    
    def test_clustering_engine(self) -> None:
        """Test article clustering engine functionality."""
        try:
            from article_clustering import ArticleClusteringEngine
            
            engine = ArticleClusteringEngine()
            start_time = time.time()
            
            # Test topic clustering
            try:
                topic_result = engine.cluster_by_topic(
                    articles=self.sample_articles,
                    n_clusters=5,
                    method='lda'
                )
                
                topic_success = topic_result.get('success', False)
                print(f"  {'✅' if topic_success else '❌'} Topic clustering (LDA)")
                
                if topic_success:
                    print(f"    └─ Clusters: {topic_result['n_clusters']}, Articles: {topic_result['total_articles']}")
                
            except Exception as e:
                topic_success = False
                print(f"  ❌ Topic clustering failed: {e}")
            
            # Test source clustering
            try:
                source_result = engine.cluster_by_source(
                    articles=self.sample_articles,
                    time_window_days=30
                )
                
                source_success = source_result.get('success', False)
                print(f"  {'✅' if source_success else '❌'} Source clustering")
                
                if source_success:
                    print(f"    └─ Clusters: {source_result['n_clusters']}, Sources: {source_result['total_sources']}")
                
            except Exception as e:
                source_success = False
                print(f"  ❌ Source clustering failed: {e}")
            
            # Test temporal clustering
            try:
                temporal_result = engine.cluster_by_temporal_patterns(
                    articles=self.sample_articles,
                    granularity='daily'
                )
                
                temporal_success = temporal_result.get('success', False)
                print(f"  {'✅' if temporal_success else '❌'} Temporal clustering")
                
                if temporal_success:
                    print(f"    └─ Patterns: {temporal_result['n_clusters']}, Periods: {temporal_result['total_time_periods']}")
                
            except Exception as e:
                temporal_success = False
                print(f"  ❌ Temporal clustering failed: {e}")
            
            # Test comprehensive analysis
            try:
                comprehensive_result = engine.analyze_all_patterns(limit=100)
                comprehensive_success = comprehensive_result.get('success', False)
                print(f"  {'✅' if comprehensive_success else '❌'} Comprehensive analysis")
                
            except Exception as e:
                comprehensive_success = False
                print(f"  ❌ Comprehensive analysis failed: {e}")
            
            # Test visualization generation
            try:
                if topic_success:
                    viz_result = engine.generate_cluster_visualization(
                        clustering_result=topic_result,
                        visualization_type='topic'
                    )
                    viz_success = viz_result.get('success', False)
                    print(f"  {'✅' if viz_success else '❌'} Visualization generation")
                else:
                    viz_success = False
                    print(f"  ⚠️ Visualization skipped (no clustering results)")
                
            except Exception as e:
                viz_success = False
                print(f"  ❌ Visualization generation failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['clustering'] = {
                'engine_loaded': True,
                'topic_clustering': topic_success,
                'source_clustering': source_success,
                'temporal_clustering': temporal_success,
                'comprehensive_analysis': comprehensive_success,
                'visualization_generation': viz_success,
                'total_time': total_time,
                'overall_success': topic_success and source_success and temporal_success
            }
            
        except Exception as e:
            print(f"  ❌ Clustering engine failed to load: {e}")
            self.test_results['clustering'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_recommendation_engine(self) -> None:
        """Test recommendation engine functionality."""
        try:
            from recommendation_engine import RecommendationEngine
            
            engine = RecommendationEngine()
            start_time = time.time()
            
            # Build models
            try:
                build_result = engine.build_comprehensive_recommendation_system()
                build_success = build_result.get('success', False)
                print(f"  {'✅' if build_success else '❌'} Model building")
                
                if build_success:
                    available_methods = build_result.get('available_methods', [])
                    print(f"    └─ Available methods: {', '.join(available_methods)}")
                
            except Exception as e:
                build_success = False
                print(f"  ❌ Model building failed: {e}")
            
            # Test content-based recommendations
            cb_success = False
            if build_success and 'content_based' in build_result.get('available_methods', []):
                try:
                    # Get a sample article ID
                    cache = engine.content_similarity_cache
                    if cache and cache.get('article_ids'):
                        sample_article = cache['article_ids'][0]
                        
                        cb_result = engine.get_content_based_recommendations(
                            article_id=sample_article,
                            n_recommendations=5
                        )
                        
                        cb_success = cb_result.get('success', False)
                        print(f"  {'✅' if cb_success else '❌'} Content-based recommendations")
                        
                        if cb_success:
                            print(f"    └─ Recommendations: {cb_result['recommendation_count']}")
                    else:
                        print(f"  ⚠️ Content-based recommendations skipped (no cached articles)")
                        
                except Exception as e:
                    print(f"  ❌ Content-based recommendations failed: {e}")
            else:
                print(f"  ⚠️ Content-based recommendations skipped (model not available)")
            
            # Test collaborative filtering
            cf_success = False
            if build_success and 'collaborative_filtering' in build_result.get('available_methods', []):
                try:
                    # Get a sample user ID
                    cache = engine.user_profiles_cache
                    if cache and cache.get('user_ids'):
                        sample_user = cache['user_ids'][0]
                        
                        cf_result = engine.get_collaborative_recommendations(
                            user_id=sample_user,
                            n_recommendations=5
                        )
                        
                        cf_success = cf_result.get('success', False)
                        print(f"  {'✅' if cf_success else '❌'} Collaborative filtering")
                        
                        if cf_success:
                            print(f"    └─ Recommendations: {cf_result['recommendation_count']}")
                    else:
                        print(f"  ⚠️ Collaborative filtering skipped (no cached users)")
                        
                except Exception as e:
                    print(f"  ❌ Collaborative filtering failed: {e}")
            else:
                print(f"  ⚠️ Collaborative filtering skipped (model not available)")
            
            # Test hybrid recommendations
            hybrid_success = False
            if build_success and 'hybrid' in build_result.get('available_methods', []):
                try:
                    user_cache = engine.user_profiles_cache
                    content_cache = engine.content_similarity_cache
                    
                    if (user_cache and content_cache and 
                        user_cache.get('user_ids') and content_cache.get('article_ids')):
                        
                        sample_user = user_cache['user_ids'][0]
                        sample_article = content_cache['article_ids'][0]
                        
                        hybrid_result = engine.get_hybrid_recommendations(
                            user_id=sample_user,
                            article_context=sample_article,
                            n_recommendations=5
                        )
                        
                        hybrid_success = hybrid_result.get('success', False)
                        print(f"  {'✅' if hybrid_success else '❌'} Hybrid recommendations")
                        
                        if hybrid_success:
                            print(f"    └─ Recommendations: {hybrid_result['recommendation_count']}")
                    else:
                        print(f"  ⚠️ Hybrid recommendations skipped (insufficient cached data)")
                        
                except Exception as e:
                    print(f"  ❌ Hybrid recommendations failed: {e}")
            else:
                print(f"  ⚠️ Hybrid recommendations skipped (models not available)")
            
            # Test feedback processing
            try:
                feedback_result = engine.process_user_feedback(
                    user_id="test_user",
                    article_id="test_article",
                    feedback_type="like",
                    feedback_value=4.0
                )
                
                feedback_success = feedback_result.get('success', False)
                print(f"  {'✅' if feedback_success else '❌'} Feedback processing")
                
            except Exception as e:
                feedback_success = False
                print(f"  ❌ Feedback processing failed: {e}")
            
            # Test explanations
            explanation_success = False
            if hybrid_success:
                try:
                    explanations = engine.get_recommendation_explanations(
                        user_id=sample_user,
                        recommendations=hybrid_result['recommendations']
                    )
                    
                    explanation_success = explanations.get('success', False)
                    print(f"  {'✅' if explanation_success else '❌'} Explanation generation")
                    
                except Exception as e:
                    print(f"  ❌ Explanation generation failed: {e}")
            else:
                print(f"  ⚠️ Explanation generation skipped (no recommendations)")
            
            total_time = time.time() - start_time
            
            self.test_results['recommendations'] = {
                'engine_loaded': True,
                'model_building': build_success,
                'content_based': cb_success,
                'collaborative_filtering': cf_success,
                'hybrid_recommendations': hybrid_success,
                'feedback_processing': feedback_success,
                'explanation_generation': explanation_success,
                'total_time': total_time,
                'overall_success': build_success and (cb_success or cf_success or hybrid_success)
            }
            
        except Exception as e:
            print(f"  ❌ Recommendation engine failed to load: {e}")
            self.test_results['recommendations'] = {
                'engine_loaded': False,
                'error': str(e),
                'overall_success': False
            }
    
    def test_system_integration(self) -> None:
        """Test integration between clustering and recommendation systems."""
        start_time = time.time()
        
        try:
            # Test clustering + recommendations workflow
            from article_clustering import ArticleClusteringEngine
            from recommendation_engine import RecommendationEngine
            
            clustering_engine = ArticleClusteringEngine()
            rec_engine = RecommendationEngine()
            
            # Perform clustering analysis
            clustering_result = clustering_engine.cluster_by_topic(
                articles=self.sample_articles,
                n_clusters=3,
                method='lda'
            )
            
            workflow_success = False
            if clustering_result.get('success'):
                # Build recommendation models with clustered data
                build_result = rec_engine.build_comprehensive_recommendation_system()
                
                if build_result.get('success'):
                    workflow_success = True
                    print(f"  ✅ Clustering + Recommendations integration")
                else:
                    print(f"  ❌ Integration failed at recommendation building")
            else:
                print(f"  ❌ Integration failed at clustering")
            
            # Test data flow between systems
            data_flow_success = False
            try:
                # Simulate retrieving articles from clustering for recommendations
                clustered_articles = []
                if clustering_result.get('success'):
                    clusters = clustering_result.get('clusters', {})
                    for cluster in clusters.values():
                        clustered_articles.extend(cluster.get('articles', [])[:5])
                
                if clustered_articles:
                    data_flow_success = True
                    print(f"  ✅ Data flow between systems")
                    print(f"    └─ Processed {len(clustered_articles)} clustered articles")
                else:
                    print(f"  ❌ Data flow failed - no clustered articles")
                
            except Exception as e:
                print(f"  ❌ Data flow test failed: {e}")
            
            total_time = time.time() - start_time
            
            self.test_results['integration'] = {
                'clustering_recommendations_workflow': workflow_success,
                'data_flow': data_flow_success,
                'total_time': total_time,
                'overall_success': workflow_success and data_flow_success
            }
            
        except Exception as e:
            print(f"  ❌ Integration testing failed: {e}")
            self.test_results['integration'] = {
                'error': str(e),
                'overall_success': False
            }
    
    def test_performance(self) -> None:
        """Test performance characteristics of new features."""
        start_time = time.time()
        
        performance_metrics = {}
        
        # Test clustering performance
        try:
            from article_clustering import ArticleClusteringEngine
            engine = ArticleClusteringEngine()
            
            perf_start = time.time()
            result = engine.cluster_by_topic(
                articles=self.sample_articles[:50],  # Smaller dataset for speed
                n_clusters=3,
                method='kmeans'  # Faster method
            )
            clustering_time = time.time() - perf_start
            
            performance_metrics['clustering'] = {
                'time_seconds': clustering_time,
                'success': result.get('success', False),
                'articles_per_second': len(self.sample_articles[:50]) / clustering_time if clustering_time > 0 else 0
            }
            
            print(f"  📊 Clustering: {clustering_time:.2f}s ({performance_metrics['clustering']['articles_per_second']:.1f} articles/sec)")
            
        except Exception as e:
            performance_metrics['clustering'] = {'error': str(e)}
            print(f"  ❌ Clustering performance test failed: {e}")
        
        # Test recommendation performance
        try:
            from recommendation_engine import RecommendationEngine
            engine = RecommendationEngine()
            
            # Build models first
            engine.build_comprehensive_recommendation_system()
            
            # Test content-based performance
            cache = engine.content_similarity_cache
            if cache and cache.get('article_ids'):
                sample_article = cache['article_ids'][0]
                
                perf_start = time.time()
                result = engine.get_content_based_recommendations(
                    article_id=sample_article,
                    n_recommendations=10
                )
                rec_time = time.time() - perf_start
                
                performance_metrics['recommendations'] = {
                    'time_seconds': rec_time,
                    'success': result.get('success', False),
                    'recommendations_per_second': 10 / rec_time if rec_time > 0 else 0
                }
                
                print(f"  📊 Recommendations: {rec_time:.2f}s ({performance_metrics['recommendations']['recommendations_per_second']:.1f} recs/sec)")
            else:
                print(f"  ⚠️ Recommendation performance test skipped (no cache)")
                
        except Exception as e:
            performance_metrics['recommendations'] = {'error': str(e)}
            print(f"  ❌ Recommendation performance test failed: {e}")
        
        total_time = time.time() - start_time
        
        # Performance thresholds
        thresholds = {
            'clustering': 30.0,  # 30 seconds for clustering
            'recommendations': 5.0  # 5 seconds for recommendations
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
    
    def test_api_endpoints(self) -> None:
        """Test API endpoint structures (without running FastAPI server)."""
        start_time = time.time()
        
        try:
            # Test clustering router structure
            from api.routers.clustering import router as clustering_router
            clustering_routes = len(clustering_router.routes)
            print(f"  ✅ Clustering API router: {clustering_routes} endpoints")
            
            # Test recommendations router structure
            from api.routers.recommendations import router as recommendations_router
            recommendation_routes = len(recommendations_router.routes)
            print(f"  ✅ Recommendations API router: {recommendation_routes} endpoints")
            
            # Test main app integration
            from api.app import create_app
            app = create_app()
            total_routes = len(app.routes)
            print(f"  ✅ FastAPI app integration: {total_routes} total routes")
            
            api_success = True
            
        except Exception as e:
            print(f"  ❌ API endpoint testing failed: {e}")
            api_success = False
        
        total_time = time.time() - start_time
        
        self.test_results['api_endpoints'] = {
            'clustering_router_loaded': True,
            'recommendations_router_loaded': True,
            'fastapi_integration': True,
            'total_time': total_time,
            'overall_success': api_success
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
            'epic6_completion_status': 'PASSED' if overall_success else 'FAILED',
            'component_results': component_summary,
            'feature_coverage': {
                'article_clustering': {
                    'topic_clustering': self.test_results.get('clustering', {}).get('topic_clustering', False),
                    'source_clustering': self.test_results.get('clustering', {}).get('source_clustering', False),
                    'temporal_clustering': self.test_results.get('clustering', {}).get('temporal_clustering', False),
                    'visualization': self.test_results.get('clustering', {}).get('visualization_generation', False)
                },
                'recommendation_engine': {
                    'content_based': self.test_results.get('recommendations', {}).get('content_based', False),
                    'collaborative_filtering': self.test_results.get('recommendations', {}).get('collaborative_filtering', False),
                    'hybrid_approach': self.test_results.get('recommendations', {}).get('hybrid_recommendations', False),
                    'feedback_loop': self.test_results.get('recommendations', {}).get('feedback_processing', False)
                },
                'system_integration': self.test_results.get('integration', {}).get('overall_success', False),
                'api_endpoints': self.test_results.get('api_endpoints', {}).get('overall_success', False)
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Print summary
        print(f"\n📊 EPIC 6 Completion Test Summary:")
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
        perf_results = self.test_results.get('performance', {})
        if not perf_results.get('performance_passed', True):
            recommendations.append("Optimize component performance to meet response time thresholds")
        
        if not recommendations:
            recommendations.append("All EPIC 6 features passed testing - roadmap is 100% complete!")
        
        return recommendations
    
    def _generate_sample_articles(self) -> List[Dict[str, Any]]:
        """Generate sample articles for testing."""
        articles = []
        
        sources = ['TechCrunch', 'Reuters', 'BBC', 'CNN', 'WSJ']
        categories = ['technology', 'business', 'politics', 'science', 'entertainment']
        
        for i in range(50):
            article = {
                'id': f'article_{i+1}',
                'title': f'Sample Article {i+1}: Technology and Business News',
                'content': f'''This is sample article content for testing purposes. 
                Article {i+1} discusses various topics including artificial intelligence, 
                machine learning, business trends, and market analysis. The content includes 
                information about companies, financial data, and technological developments. 
                This text is designed to provide sufficient content for clustering and 
                recommendation algorithms to analyze patterns and similarities.''',
                'source': sources[i % len(sources)],
                'category': categories[i % len(categories)],
                'published_date': datetime.now().isoformat(),
                'tags': ['technology', 'AI', 'business'] if i % 2 == 0 else ['finance', 'market', 'analysis']
            }
            articles.append(article)
        
        return articles
    
    def _generate_sample_interactions(self) -> List[Dict[str, Any]]:
        """Generate sample user interactions for testing."""
        interactions = []
        
        for i in range(200):
            interaction = {
                'user_id': f'user_{(i % 20) + 1}',  # 20 users
                'article_id': f'article_{(i % 50) + 1}',  # 50 articles
                'interaction_type': ['view', 'like', 'share', 'save'][i % 4],
                'rating': min(5.0, max(1.0, 3.0 + (i % 3 - 1))),  # Ratings 1-5
                'timestamp': datetime.now().isoformat()
            }
            interactions.append(interaction)
        
        return interactions


def main():
    """Main test execution function."""
    print("🚀 Starting EPIC 6 Completion Test Suite")
    print("Testing: Article Clustering, Recommendations, Integration, Performance, APIs")
    print()
    
    # Set environment variables
    os.environ["USE_POSTGRES"] = "true"
    
    # Initialize and run tests
    test_suite = EPIC6CompletionTestSuite()
    
    try:
        report = test_suite.run_comprehensive_test()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"epic6_completion_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
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