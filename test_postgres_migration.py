#!/usr/bin/env python3
"""Test script to verify PostgreSQL migration and functionality.""f"

from datetime import datetime, timedelta
from pathlib import Path
import logging
import os
import sys

sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format=f'{asctime}'s - {message}"s')
logger = logging.getLogger(__name__)

def test_postgres_connection():
    """Test PostgreSQL connection."""
    logger.info("Testing PostgreSQL connection...")
    try:
        import db_manager_postgres as db
        db.db_pool.initialize()
        
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            logger.info(f"✅ Connected to PostgreSQL: {version}")
            return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def test_full_text_search():
    """Test full-text search functionality."""
    logger.info("\nTesting full-text search...")
    try:
        import db_manager_postgres as db
        
        # Add a test article
        test_id = db.add_article(
            url="https://test.example.com/ai-article",
            title="Artificial Intelligence Breakthrough in Natural Language Processing",
            publication_date=datetime.now(),
            source_website="Test Source",
            content_summary="A major breakthrough in AI and machine learning...",
            full_content="Researchers have discovered new methods for natural language processing using advanced neural networks...",
            keywords_found=["AI", "machine learning", "NLP"]
        )
        
        if test_id:
            logger.info(f"✅ Test article created: {test_id}")
            
            # Search for the article
            results = db.search_articles("artificial intelligence breakthrough")
            if results:
                logger.info(f"✅ Found {len(results)} articles via full-text search")
                for result in results[:3]:
                    logger.info(f"  - {result['title'][:50]}... (rank: {result['rank']:.2f})")
            else:
                logger.warning("⚠️  No search results found")
                
            # Clean up
            with db.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM articles WHERE id = %s", (test_id,))
                
        return True
    except Exception as e:
        logger.error(f"❌ Full-text search test failed: {e}")
        return False

def test_source_management():
    """Test source management features."""
    logger.info("\nTesting source management...")
    try:
        import db_manager_postgres as db
        
        # Get active sources
        sources = db.get_active_sources()
        logger.info(f"✅ Found {len(sources)} active sources")
        
        for source in sources[:3]:
            logger.info(f"  - {source['name']}: {source['type']} ({source['url'][:50]}...)")
            
        # Test source stats update
        if sources:
            db.update_source_stats(sources[0]['name'], success=True)
            logger.info(f"✅ Updated stats for {sources[0]['name']}")
            
        return True
    except Exception as e:
        logger.error(f"❌ Source management test failed: {e}")
        return False

def test_performance():
    """Test query performance."""
    logger.info("\nTesting query performance...")
    try:
        import db_manager_postgres as db
        import time
        
        # Test bulk insert performance
        start = time.time()
        articles_added = 0
        
        for i in range(100):
            article_id = db.add_article(
                url=f"https://test.example.com/perf-test-{i}",
                title=f"Performance Test Article {i}",
                publication_date=datetime.now() - timedelta(hours=i),
                source_website="Performance Test",
                content_summary=f"This is test article number {i} for performance testing.",
                keywords_found=["test", "performance"] if i % 2 == 0 else []
            )
            if article_id:
                articles_added += 1
                
        elapsed = time.time() - start
        logger.info(f"✅ Added {articles_added} articles in {elapsed:.2f} seconds ({articles_added/elapsed:.1f} articles/sec)")
        
        # Test search performance
        start = time.time()
        results = db.search_articles("performance test")
        search_time = time.time() - start
        logger.info(f"✅ Full-text search returned {len(results)} results in {search_time:.3f} seconds")
        
        # Clean up test data
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM articles WHERE source = 'Performance Test'"
            )
            logger.info(f"✅ Cleaned up {cursor.rowcount} test articles")
            
        return True
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def test_adapter_compatibility():
    """Test database adapter backward compatibility."""
    logger.info("\nTesting database adapter...")
    try:
        # Force PostgreSQL mode
        os.environ['USE_POSTGRES'] = 'true'
        
        
        logger.info(f"✅ Database type: {db.get_database_type()}")
        logger.info(f"✅ Full-text search support: {db.supports_full_text_search()}")
        logger.info(f"✅ Source management support: {db.supports_source_management()}")
        
        # Test migration status check
        status = db.check_migration_status()
        logger.info(f"✅ Migration status:")
        logger.info(f"  - Current DB: {status['current_db']}")
        logger.info(f"  - PostgreSQL available: {status['postgres_available']}")
        logger.info(f"  - PostgreSQL articles: {status['postgres_articles']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Adapter test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("=== PostgreSQL Migration Test Suite ===\n")
    
    tests = [
        ("PostgreSQL Connection", test_postgres_connection),
        ("Full-Text Search", test_full_text_search),
        ("Source Management", test_source_management),
        ("Performance", test_performance),
        ("Adapter Compatibility", test_adapter_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == '__main__':
    main()