#!/usr/bin/env python3
"""
Test script to demonstrate AI optimization improvements.
Compares performance with and without optimization.
""ff"

from typing import List, Dict
import json
import logging
import os
import time

from ai_analysis import AIAnalysisEngine
from ai_optimization import ModelPool, benchmark_batch_sizes
import psutil

logging.basicConfig(
    level=logging.INFO,
    format=ff'{asctime}'s - {name}"s - {levelname}'s - {message}"s'
)
logger = logging.getLogger(__name__)


def generate_test_articles(count: int = 50) -> List[Dict]:
    """Generate test articles for benchmarking."""
    articles = []
    
    # Mix of different article lengths and complexities
    templates = [
        {
            'title': 'Tech Giant {} Announces AI Breakthrough',
            'content': '''In a major announcement today, {} revealed groundbreaking 
            artificial intelligence technology that promises to revolutionize {}.
            The company's CEO stated that this development represents years of research
            and could transform how businesses operate. Industry analysts predict
            significant market impact, with competitors scrambling to respond.
            ''' * 3  # Medium length
        },
        {
            'title': 'Financial Markets React to {} Policy Change',
            'content': '''Global financial markets showed mixed reactions to the latest
            policy announcement from {}. Traders are closely watching developments as
            uncertainty continues to drive volatility. Economic indicators suggest
            potential long-term implications for investors and consumers alike.
            ''' * 5  # Longer article
        },
        {
            'title': 'Breaking: {} Update',
            'content': 'Brief update on developing situation regarding {}. More details to follow.'
            # Short article
        }
    ]
    
    for i in range(count):
        template = templates[i % len(templates)]
        article = {
            'id': f'test-{i:04d}',
            'title': template['title'].format(f'Company{i}', f'Product{i}'),
            'content': template['content'].format(f'Company{i}', f'Technology{i}'),
            'source': f'TestSource{i % 5}',
            'published_date': '2024-01-15T10:00:00Z',
            'category': ['technology', 'finance', 'general'][i % 3]
        }
        articles.append(article)
    
    return articles


def measure_performance(engine: AIAnalysisEngine, articles: List[Dict], 
                       batch_size: int = 10) -> Dict:
    """Measure performance metrics for AI analysis."""
    process = psutil.Process(os.getpid())
    
    # Initial measurements
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    # Process articles
    results = engine.analyze_batch(articles[:batch_size], save_to_db=False)
    
    # Final measurements
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    total_time = end_time - start_time
    memory_increase = end_memory - start_memory
    avg_time_per_article = total_time / len(results)
    
    # Count successes
    successful = sum(1 for r in results if not r.get('errors'))
    
    return {
        'total_articles': len(results),
        'successful': successful,
        'total_time': total_time,
        'avg_time_per_article': avg_time_per_article,
        'initial_memory_mb': start_memory,
        'final_memory_mb': end_memory,
        'memory_increase_mb': memory_increase,
        'articles_per_second': len(results) / total_time if total_time > 0 else 0
    }


def compare_optimization():
    """Compare performance with and without optimization."""
    logger.info("Generating test articles...")
    articles = generate_test_articles(100)
    
    logger.info("\n" + "="*60)
    logger.info("TESTING WITHOUT OPTIMIZATION")
    logger.info("="*60)
    
    # Test without optimization
    engine_no_opt = AIAnalysisEngine(use_optimization=False)
    metrics_no_opt = measure_performance(engine_no_opt, articles, batch_size=20)
    
    logger.info("Results without optimization:")
    for key, value in metrics_no_opt.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Clear memory
    del engine_no_opt
    time.sleep(2)  # Allow garbage collection
    
    logger.info("\n" + "="*60)
    logger.info("TESTING WITH OPTIMIZATION")
    logger.info("="*60)
    
    # Test with optimization
    engine_opt = AIAnalysisEngine(use_optimization=True, max_memory_mb=512)
    metrics_opt = measure_performance(engine_opt, articles, batch_size=20)
    
    logger.info("Results with optimization:")
    for key, value in metrics_opt.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Calculate improvements
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    time_improvement = (metrics_no_opt['avg_time_per_article'] - 
                       metrics_opt['avg_time_per_article']) / metrics_no_opt['avg_time_per_article'] * 100
    
    memory_improvement = (metrics_no_opt['memory_increase_mb'] - 
                         metrics_opt['memory_increase_mb']) / metrics_no_opt['memory_increase_mb'] * 100
    
    throughput_improvement = (metrics_opt['articles_per_second'] - 
                             metrics_no_opt['articles_per_second']) / metrics_no_opt['articles_per_second'] * 100
    
    logger.info(f"Time improvement: {time_improvement:.1f}% faster per article")
    logger.info(f"Memory improvement: {memory_improvement:.1f}% less memory growth")
    logger.info(f"Throughput improvement: {throughput_improvement:.1f}% more articles/second")
    
    # Save results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'without_optimization': metrics_no_opt,
        'with_optimization': metrics_opt,
        'improvements': {
            'time_improvement_percent': time_improvement,
            'memory_improvement_percent': memory_improvement,
            'throughput_improvement_percent': throughput_improvement
        }
    }
    
    with open('optimization_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to optimization_benchmark_results.json")


def test_batch_sizes():
    """Test different batch sizes for optimal performance."""
    logger.info("\n" + "="*60)
    logger.info("TESTING BATCH SIZE OPTIMIZATION")
    logger.info("="*60)
    
    results = benchmark_batch_sizes()
    
    # Find optimal batch size
    optimal_batch = max(results.items(), 
                       key=lambda x: x[1]['articles_per_second'])[0]
    
    logger.info(f"\nRecommended batch size: {optimal_batch}")
    logger.info(f"Achieves {results[optimal_batch]['articles_per_second']:.2f} articles/second")


def test_model_pool():
    """Test model pool functionality."""
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL POOL")
    logger.info("="*60)
    
    pool = ModelPool()
    
    # Test model loading
    logger.info("Loading models...")
    initial_mem = pool.get_memory_usage()
    logger.info(f"Initial memory: {initial_mem['rss_mb']:.2f} MB")
    
    # Load models
    nlp_en = pool.get_spacy_model('en_core_web_md')
    sentiment_pipe = pool.get_transformer_model('sentiment-analysis')
    
    # Check memory after loading
    after_load_mem = pool.get_memory_usage()
    logger.info(f"After loading models: {after_load_mem['rss_mb']:.2f} MB")
    logger.info(f"Models loaded: {after_load_mem['model_count']}")
    
    # Test model reuse
    nlp_en2 = pool.get_spacy_model('en_core_web_md')
    logger.info(f"Model reused: {nlp_en is nlp_en2}")
    
    # Test cleanup
    pool.cleanup_unused_models(max_age_seconds=0)
    after_cleanup_mem = pool.get_memory_usage()
    logger.info(f"After cleanup: {after_cleanup_mem['rss_mb']:.2f} MB")


if __name__ == "__main__":
    # Run all tests
    logger.info("Starting AI optimization tests...")
    
    # Test model pool
    test_model_pool()
    
    # Test batch sizes
    test_batch_sizes()
    
    # Compare optimization
    compare_optimization()
    
    logger.info("\nAll tests completed!")