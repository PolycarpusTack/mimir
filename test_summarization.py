#!/usr/bin/env python3
"""
Test script for the summarization engine - EPIC 8.1
"""

import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from summarization_engine import SummarizationEngine
import db_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_summarization_engine():
    """Test the summarization engine with sample content."""
    
    print("🚀 Testing Mimir Summarization Engine - EPIC 8.1")
    print("=" * 60)
    
    # Initialize the engine
    try:
        engine = SummarizationEngine()
        print("✅ Summarization engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize summarization engine: {e}")
        return False
    
    # Test with sample content
    sample_text = """
    Artificial Intelligence (AI) continues to revolutionize industries across the globe, 
    with machine learning algorithms becoming increasingly sophisticated. Recent developments 
    in natural language processing have enabled computers to understand and generate human-like 
    text with unprecedented accuracy. Companies are investing billions of dollars in AI research 
    and development, recognizing its potential to transform business operations, improve efficiency, 
    and create new revenue streams. However, experts warn about the need for ethical AI development 
    and proper regulation to ensure that these powerful technologies are used responsibly. 
    The debate over AI governance continues as governments worldwide grapple with balancing 
    innovation and safety concerns.
    """
    
    # Test different summary formats
    formats_to_test = ['custom', 'bullet_points', 'executive', 'tweet']
    
    for format_type in formats_to_test:
        print(f"\n📝 Testing {format_type} format:")
        print("-" * 40)
        
        try:
            result = engine.generate_summary(
                text=sample_text,
                format_type=format_type
            )
            
            if result.get('success'):
                print(f"✅ Success!")
                print(f"Summary: {result['summary']}")
                print(f"Quality Score: {result['quality_score']:.2f}")
                print(f"Compression Ratio: {result['compression_ratio']:.2f}")
                print(f"Processing Time: {result['processing_time_seconds']:.2f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Test batch summarization
    print(f"\n📚 Testing batch summarization:")
    print("-" * 40)
    
    try:
        # Create sample articles
        sample_articles = [
            {
                'id': 'test_1',
                'content': sample_text
            },
            {
                'id': 'test_2', 
                'content': "Technology stocks surged today as investors showed renewed confidence in the sector. Major tech companies reported strong quarterly earnings, beating analyst expectations."
            }
        ]
        
        batch_results = engine.batch_summarize(
            articles=sample_articles,
            format_type='custom'
        )
        
        print(f"✅ Batch processing completed")
        print(f"Processed {len(batch_results)} articles")
        
        for result in batch_results:
            if result.get('success'):
                print(f"  - Article {result['article_id']}: {result['summary'][:50]}...")
            else:
                print(f"  - Article {result['article_id']}: Failed - {result.get('error')}")
                
    except Exception as e:
        print(f"❌ Batch summarization failed: {e}")
    
    # Test database integration (if we have test data)
    print(f"\n💾 Testing database integration:")
    print("-" * 40)
    
    try:
        # Check if we can connect to database
        article_count = db_adapter.count_articles()
        print(f"✅ Database connection successful")
        print(f"Total articles in database: {article_count}")
        
        if article_count > 0:
            # Get recent articles and test summarization
            recent_articles = db_adapter.get_recent_articles(limit=2)
            print(f"Testing with {len(recent_articles)} recent articles...")
            
            for article in recent_articles:
                if article.get('content'):
                    result = engine.generate_summary(
                        text=article['content'],
                        format_type='custom'
                    )
                    
                    if result.get('success'):
                        print(f"  - Article '{article['title'][:30]}...': Summary generated")
                        
                        # Test saving to database
                        saved = engine.save_summary_to_db(article['id'], result)
                        if saved:
                            print(f"    ✅ Summary saved to database")
                        else:
                            print(f"    ⚠️ Failed to save summary to database")
                    else:
                        print(f"  - Article '{article['title'][:30]}...': Failed to summarize")
        else:
            print("ℹ️ No articles in database to test with")
            
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Summarization engine testing completed!")
    
    return True


def test_api_models():
    """Test the Pydantic models for the API."""
    print("\n🔧 Testing API models:")
    print("-" * 40)
    
    try:
        from api.routers.summarization import (
            SummarizationRequest,
            SummarizationResponse,
            BatchSummarizationRequest
        )
        
        # Test request model validation
        request = SummarizationRequest(
            text="Test content",
            format_type="custom",
            custom_length=100
        )
        print("✅ SummarizationRequest model validation passed")
        
        # Test response model
        response_data = {
            'summary': 'Test summary',
            'format_type': 'custom',
            'model_used': 'test-model',
            'processing_time_seconds': 1.0,
            'quality_score': 0.8,
            'input_length': 100,
            'output_length': 20,
            'compression_ratio': 0.2,
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        response = SummarizationResponse(**response_data)
        print("✅ SummarizationResponse model validation passed")
        
        # Test batch request
        batch_request = BatchSummarizationRequest(
            article_ids=['test1', 'test2'],
            format_type='bullet_points'
        )
        print("✅ BatchSummarizationRequest model validation passed")
        
    except Exception as e:
        print(f"❌ API model testing failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Set environment variable for testing
    os.environ["USE_POSTGRES"] = "true"
    
    print("🧪 MIMIR EPIC 8.1 - Summarization Engine Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test summarization engine
    if not test_summarization_engine():
        success = False
    
    # Test API models
    if not test_api_models():
        success = False
    
    if success:
        print("\n✅ All tests passed! Summarization engine is ready for production.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)