#!/usr/bin/env python3
"""
Command-line interface for Mimir Summarization Engine - EPIC 8.1
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

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


class SummarizationCLI:
    """Command-line interface for summarization operations."""
    
    def __init__(self):
        self.engine = SummarizationEngine()
    
    def summarize_text(self, text: str, format_type: str = "custom", 
                      custom_length: Optional[int] = None, 
                      model_type: str = "facebook/bart-large-cnn") -> None:
        """Summarize provided text and print result."""
        print(f"🔄 Summarizing text using {model_type} model...")
        print(f"Format: {format_type}")
        if custom_length:
            print(f"Custom length: {custom_length}")
        print("-" * 50)
        
        result = self.engine.generate_summary(
            text=text,
            format_type=format_type,
            custom_length=custom_length,
            model_type=model_type
        )
        
        if result['success']:
            print(f"✅ Summary generated successfully!")
            print(f"\n📝 Summary ({format_type}):")
            print(f"{result['summary']}")
            print(f"\n📊 Metrics:")
            print(f"  Quality Score: {result['quality_score']:.2f}/1.0")
            print(f"  Compression Ratio: {result['compression_ratio']:.2f}")
            print(f"  Processing Time: {result['processing_time_seconds']:.2f}s")
            print(f"  Input Length: {result['input_length']} chars")
            print(f"  Output Length: {result['output_length']} chars")
        else:
            print(f"❌ Summarization failed: {result.get('error', 'Unknown error')}")
    
    def summarize_article(self, article_id: str, format_type: str = "custom",
                         custom_length: Optional[int] = None,
                         save_to_db: bool = True) -> None:
        """Summarize an article from the database."""
        print(f"🔄 Fetching and summarizing article: {article_id}")
        print(f"Format: {format_type}")
        print("-" * 50)
        
        # Get article from database
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            print(f"❌ Article not found: {article_id}")
            return
        
        print(f"📰 Article: {article['title']}")
        print(f"📅 Published: {article.get('published_date', 'Unknown')}")
        print(f"🌐 Source: {article.get('source', 'Unknown')}")
        
        content = article.get('content', '')
        if not content:
            print(f"❌ Article has no content to summarize")
            return
        
        # Generate summary
        result = self.engine.generate_summary(
            text=content,
            format_type=format_type,
            custom_length=custom_length
        )
        
        if result['success']:
            print(f"\n✅ Summary generated successfully!")
            print(f"\n📝 Summary ({format_type}):")
            print(f"{result['summary']}")
            print(f"\n📊 Metrics:")
            print(f"  Quality Score: {result['quality_score']:.2f}/1.0")
            print(f"  Compression Ratio: {result['compression_ratio']:.2f}")
            print(f"  Processing Time: {result['processing_time_seconds']:.2f}s")
            
            # Save to database if requested
            if save_to_db:
                saved = self.engine.save_summary_to_db(article_id, result)
                if saved:
                    print(f"💾 Summary saved to database")
                else:
                    print(f"⚠️ Failed to save summary to database")
        else:
            print(f"❌ Summarization failed: {result.get('error', 'Unknown error')}")
    
    def batch_summarize_recent(self, limit: int = 10, format_type: str = "custom",
                              hours: int = 24, save_to_db: bool = True) -> None:
        """Batch summarize recent articles."""
        print(f"🔄 Batch summarizing {limit} recent articles from last {hours} hours")
        print(f"Format: {format_type}")
        print("-" * 50)
        
        # Get recent articles
        articles = db_adapter.get_recent_articles(limit=limit, hours=hours)
        
        if not articles:
            print(f"📭 No recent articles found")
            return
        
        print(f"📚 Found {len(articles)} recent articles to summarize")
        
        # Prepare articles for batch processing
        batch_articles = []
        for article in articles:
            if article.get('content'):
                batch_articles.append({
                    'id': article['id'],
                    'content': article['content']
                })
        
        if not batch_articles:
            print(f"📭 No articles with content to summarize")
            return
        
        # Process batch
        results = self.engine.batch_summarize(
            articles=batch_articles,
            format_type=format_type
        )
        
        successful = 0
        failed = 0
        
        print(f"\n📋 Batch Processing Results:")
        print("-" * 30)
        
        for result in results:
            article_id = result.get('article_id', 'unknown')
            if result.get('success'):
                successful += 1
                article = next((a for a in articles if a['id'] == article_id), None)
                title = article['title'][:40] if article else 'Unknown'
                
                print(f"✅ {title}... (Quality: {result['quality_score']:.2f})")
                
                # Save to database if requested
                if save_to_db:
                    saved = self.engine.save_summary_to_db(article_id, result)
                    if not saved:
                        print(f"  ⚠️ Failed to save to database")
            else:
                failed += 1
                print(f"❌ Failed to summarize article {article_id}: {result.get('error', 'Unknown')}")
        
        print(f"\n📊 Summary Statistics:")
        print(f"  Total processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        if successful > 0:
            avg_time = sum(r.get('processing_time_seconds', 0) for r in results if r.get('success')) / successful
            print(f"  Average processing time: {avg_time:.2f}s")
    
    def list_summaries(self, article_id: str) -> None:
        """List all summaries for an article."""
        print(f"📋 Summaries for article: {article_id}")
        print("-" * 50)
        
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            print(f"❌ Article not found: {article_id}")
            return
        
        print(f"📰 Article: {article['title']}")
        
        metadata = article.get('metadata', '{}')
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        summaries = metadata.get('summaries', {})
        
        if not summaries:
            print(f"📭 No summaries found for this article")
            return
        
        print(f"\n📝 Available summaries ({len(summaries)}):")
        
        for format_type, summary_data in summaries.items():
            print(f"\n🏷️ Format: {format_type}")
            print(f"   Summary: {summary_data.get('summary', 'N/A')}")
            print(f"   Quality: {summary_data.get('quality_score', 'N/A')}")
            print(f"   Model: {summary_data.get('model_used', 'N/A')}")
            print(f"   Created: {summary_data.get('timestamp', 'N/A')}")
    
    def show_stats(self) -> None:
        """Show summarization statistics."""
        print(f"📊 Summarization Statistics")
        print("-" * 50)
        
        try:
            total_articles = db_adapter.count_articles()
            articles_with_summaries = db_adapter.count_articles_with_metadata_key('summaries')
            format_distribution = db_adapter.get_summary_format_distribution()
            
            coverage = (articles_with_summaries / total_articles * 100) if total_articles > 0 else 0
            
            print(f"📚 Total articles: {total_articles}")
            print(f"📝 Articles with summaries: {articles_with_summaries}")
            print(f"📈 Coverage: {coverage:.1f}%")
            
            if format_distribution:
                print(f"\n🏷️ Summary format distribution:")
                for format_type, count in format_distribution.items():
                    print(f"   {format_type}: {count}")
            else:
                print(f"\n📭 No summaries found in database")
                
        except Exception as e:
            print(f"❌ Failed to retrieve statistics: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mimir Summarization Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize text directly
  python summarization_cli.py text "Your text here" --format bullet_points

  # Summarize article from database
  python summarization_cli.py article abc123 --format executive --save

  # Batch summarize recent articles
  python summarization_cli.py batch --limit 20 --hours 48 --format custom

  # List summaries for an article
  python summarization_cli.py list abc123

  # Show statistics
  python summarization_cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Text summarization command
    text_parser = subparsers.add_parser('text', help='Summarize provided text')
    text_parser.add_argument('text', help='Text content to summarize')
    text_parser.add_argument('--format', choices=['custom', 'bullet_points', 'executive', 'tweet'], 
                           default='custom', help='Summary format')
    text_parser.add_argument('--length', type=int, help='Custom summary length')
    text_parser.add_argument('--model', default='facebook/bart-large-cnn', 
                           help='Model to use for summarization')
    
    # Article summarization command
    article_parser = subparsers.add_parser('article', help='Summarize article from database')
    article_parser.add_argument('article_id', help='Article ID to summarize')
    article_parser.add_argument('--format', choices=['custom', 'bullet_points', 'executive', 'tweet'],
                              default='custom', help='Summary format')
    article_parser.add_argument('--length', type=int, help='Custom summary length')
    article_parser.add_argument('--save', action='store_true', help='Save summary to database')
    
    # Batch summarization command
    batch_parser = subparsers.add_parser('batch', help='Batch summarize recent articles')
    batch_parser.add_argument('--limit', type=int, default=10, help='Number of articles to process')
    batch_parser.add_argument('--hours', type=int, default=24, help='Hours back to search for articles')
    batch_parser.add_argument('--format', choices=['custom', 'bullet_points', 'executive', 'tweet'],
                            default='custom', help='Summary format')
    batch_parser.add_argument('--save', action='store_true', help='Save summaries to database')
    
    # List summaries command
    list_parser = subparsers.add_parser('list', help='List summaries for an article')
    list_parser.add_argument('article_id', help='Article ID to list summaries for')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show summarization statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set environment for database
    os.environ["USE_POSTGRES"] = "true"
    
    # Initialize CLI
    cli = SummarizationCLI()
    
    try:
        if args.command == 'text':
            cli.summarize_text(
                text=args.text,
                format_type=args.format,
                custom_length=args.length,
                model_type=args.model
            )
        
        elif args.command == 'article':
            cli.summarize_article(
                article_id=args.article_id,
                format_type=args.format,
                custom_length=args.length,
                save_to_db=args.save
            )
        
        elif args.command == 'batch':
            cli.batch_summarize_recent(
                limit=args.limit,
                format_type=args.format,
                hours=args.hours,
                save_to_db=args.save
            )
        
        elif args.command == 'list':
            cli.list_summaries(args.article_id)
        
        elif args.command == 'stats':
            cli.show_stats()
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("CLI operation failed")


if __name__ == "__main__":
    main()