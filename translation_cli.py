#!/usr/bin/env python3
"""
Command-line interface for Mimir Translation Engine - EPIC 8.2
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

from translation_engine import TranslationEngine
import db_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TranslationCLI:
    """Command-line interface for translation operations."""
    
    def __init__(self):
        self.engine = TranslationEngine()
    
    def translate_text(self, text: str, target_language: str, 
                      source_language: Optional[str] = None,
                      model_override: Optional[str] = None) -> None:
        """Translate provided text and print result."""
        print(f"🌐 Translating text to {target_language.upper()}")
        if source_language:
            print(f"Source language: {source_language.upper()}")
        else:
            print("Source language: Auto-detect")
        if model_override:
            print(f"Model: {model_override}")
        print("-" * 60)
        
        result = self.engine.translate_text(
            text=text,
            target_language=target_language,
            source_language=source_language,
            model_override=model_override
        )
        
        if result['success']:
            print(f"✅ Translation completed!")
            print(f"\n🔤 Original ({result['source_language'].upper()}):")
            print(f"{result['source_text']}")
            print(f"\n🌐 Translation ({result['target_language'].upper()}):")
            print(f"{result['translated_text']}")
            print(f"\n📊 Metrics:")
            print(f"  Quality Score: {result['quality_score']:.2f}/1.0")
            print(f"  Processing Time: {result['processing_time_seconds']:.2f}s")
            print(f"  Model Used: {result['model_used']}")
            
            if result.get('note'):
                print(f"📝 Note: {result['note']}")
        else:
            print(f"❌ Translation failed: {result.get('error', 'Unknown error')}")
    
    def translate_article(self, article_id: str, target_language: str,
                         source_language: Optional[str] = None,
                         save_to_db: bool = True) -> None:
        """Translate an article from the database."""
        print(f"🌐 Translating article: {article_id}")
        print(f"Target language: {target_language.upper()}")
        print("-" * 60)
        
        # Get article from database
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            print(f"❌ Article not found: {article_id}")
            return
        
        print(f"📰 Article: {article['title']}")
        print(f"📅 Published: {article.get('published_date', 'Unknown')}")
        print(f"🌐 Source: {article.get('source', 'Unknown')}")
        
        content = article.get('content', '') or article.get('title', '')
        if not content:
            print(f"❌ Article has no content to translate")
            return
        
        # Show content preview
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n📄 Content preview: {preview}")
        
        # Perform translation
        result = self.engine.translate_text(
            text=content,
            target_language=target_language,
            source_language=source_language
        )
        
        if result['success']:
            print(f"\n✅ Translation completed!")
            print(f"\n🌐 Translated to {result['target_language'].upper()}:")
            
            # Show translation preview
            translation_preview = result['translated_text'][:400] + "..." if len(result['translated_text']) > 400 else result['translated_text']
            print(f"{translation_preview}")
            
            print(f"\n📊 Metrics:")
            print(f"  Source Language: {result['source_language'].upper()}")
            print(f"  Quality Score: {result['quality_score']:.2f}/1.0")
            print(f"  Processing Time: {result['processing_time_seconds']:.2f}s")
            print(f"  Model Used: {result['model_used']}")
            
            # Save to database if requested
            if save_to_db:
                saved = self.engine.save_translation_to_db(article_id, result)
                if saved:
                    print(f"💾 Translation saved to database")
                else:
                    print(f"⚠️ Failed to save translation to database")
        else:
            print(f"❌ Translation failed: {result.get('error', 'Unknown error')}")
    
    def batch_translate_recent(self, target_language: str, limit: int = 10,
                              hours: int = 24, source_language: Optional[str] = None,
                              save_to_db: bool = True) -> None:
        """Batch translate recent articles."""
        print(f"🌐 Batch translating {limit} recent articles to {target_language.upper()}")
        print(f"Time range: Last {hours} hours")
        if source_language:
            print(f"Source language: {source_language.upper()}")
        print("-" * 60)
        
        # Get recent articles
        articles = db_adapter.get_recent_articles(limit=limit, hours=hours)
        
        if not articles:
            print(f"📭 No recent articles found")
            return
        
        print(f"📚 Found {len(articles)} recent articles")
        
        # Prepare texts for batch processing
        texts = []
        article_ids = []
        
        for article in articles:
            content = article.get('content', '') or article.get('title', '')
            if content:
                texts.append(content)
                article_ids.append(article['id'])
        
        if not texts:
            print(f"📭 No articles with content to translate")
            return
        
        print(f"🔄 Processing {len(texts)} articles with content...")
        
        # Process batch translation
        results = self.engine.batch_translate(
            texts=texts,
            target_language=target_language,
            source_language=source_language
        )
        
        successful = 0
        failed = 0
        
        print(f"\n📋 Batch Translation Results:")
        print("-" * 40)
        
        for i, result in enumerate(results):
            if i < len(article_ids):
                article_id = article_ids[i]
                article = next((a for a in articles if a['id'] == article_id), None)
                title = article['title'][:40] + "..." if article and len(article['title']) > 40 else article['title'] if article else 'Unknown'
                
                if result.get('success'):
                    successful += 1
                    source_lang = result.get('source_language', 'Unknown').upper()
                    quality = result.get('quality_score', 0)
                    
                    print(f"✅ {title}")
                    print(f"   {source_lang} → {target_language.upper()} (Quality: {quality:.2f})")
                    
                    # Save to database if requested
                    if save_to_db:
                        saved = self.engine.save_translation_to_db(article_id, result)
                        if not saved:
                            print(f"   ⚠️ Failed to save to database")
                else:
                    failed += 1
                    print(f"❌ {title}")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📊 Summary Statistics:")
        print(f"  Total processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {(successful/len(results)*100):.1f}%")
        
        if successful > 0:
            avg_time = sum(r.get('processing_time_seconds', 0) for r in results if r.get('success')) / successful
            avg_quality = sum(r.get('quality_score', 0) for r in results if r.get('success')) / successful
            print(f"  Average processing time: {avg_time:.2f}s")
            print(f"  Average quality score: {avg_quality:.2f}")
    
    def detect_language(self, text: str) -> None:
        """Detect language of provided text."""
        print(f"🔍 Detecting language...")
        print("-" * 40)
        
        detected_lang, confidence = self.engine.detect_language(text)
        
        supported_languages = self.engine.get_supported_languages()
        language_name = supported_languages.get(detected_lang, "Unknown")
        
        print(f"📝 Text sample: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"\n🌐 Detection Results:")
        print(f"  Language: {language_name} ({detected_lang.upper()})")
        print(f"  Confidence: {confidence:.2f}")
        
        if confidence < 0.7:
            print(f"⚠️ Low confidence detection - results may be inaccurate")
    
    def list_translations(self, article_id: str) -> None:
        """List all translations for an article."""
        print(f"📋 Translations for article: {article_id}")
        print("-" * 50)
        
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            print(f"❌ Article not found: {article_id}")
            return
        
        print(f"📰 Article: {article['title']}")
        
        metadata = article.get('metadata', '{}')
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        translations = metadata.get('translations', {})
        
        if not translations:
            print(f"📭 No translations found for this article")
            return
        
        print(f"\n🌐 Available translations ({len(translations)}):")
        
        supported_languages = self.engine.get_supported_languages()
        
        for target_lang, translation_data in translations.items():
            language_name = supported_languages.get(target_lang, "Unknown")
            translation_preview = translation_data.get('translated_text', '')[:100] + "..." if len(translation_data.get('translated_text', '')) > 100 else translation_data.get('translated_text', 'N/A')
            
            print(f"\n🏷️ {language_name} ({target_lang.upper()})")
            print(f"   Translation: {translation_preview}")
            print(f"   Quality: {translation_data.get('quality_score', 'N/A')}")
            print(f"   Source: {translation_data.get('source_language', 'N/A').upper()}")
            print(f"   Model: {translation_data.get('model_used', 'N/A')}")
            print(f"   Created: {translation_data.get('timestamp', 'N/A')}")
    
    def show_languages(self) -> None:
        """Show supported languages and models."""
        print(f"🌐 Supported Languages & Models")
        print("-" * 50)
        
        supported_languages = self.engine.get_supported_languages()
        available_models = self.engine.get_available_models()
        
        print(f"📚 Supported Languages ({len(supported_languages)}):")
        for code, name in supported_languages.items():
            print(f"  {code.upper()}: {name}")
        
        print(f"\n🤖 Available Models ({len(available_models)}):")
        for model_key, model_info in available_models.items():
            print(f"\n  {model_key}:")
            print(f"    Model: {model_info['model']}")
            print(f"    Description: {model_info['description']}")
            languages = ", ".join(model_info['languages'][:5])
            if len(model_info['languages']) > 5:
                languages += f" (+{len(model_info['languages'])-5} more)"
            print(f"    Languages: {languages}")
    
    def show_stats(self) -> None:
        """Show translation statistics."""
        print(f"📊 Translation Statistics")
        print("-" * 50)
        
        try:
            total_articles = db_adapter.count_articles()
            articles_with_translations = db_adapter.count_articles_with_metadata_key('translations')
            
            coverage = (articles_with_translations / total_articles * 100) if total_articles > 0 else 0
            
            print(f"📚 Total articles: {total_articles}")
            print(f"🌐 Articles with translations: {articles_with_translations}")
            print(f"📈 Coverage: {coverage:.1f}%")
            
            supported_languages = self.engine.get_supported_languages()
            available_models = self.engine.get_available_models()
            
            print(f"🗣️ Supported languages: {len(supported_languages)}")
            print(f"🤖 Available models: {len(available_models)}")
            
        except Exception as e:
            print(f"❌ Failed to retrieve statistics: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mimir Translation Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate text directly
  python translation_cli.py text "Hello world" --target es --source en

  # Translate article from database
  python translation_cli.py article abc123 --target fr --save

  # Batch translate recent articles
  python translation_cli.py batch --target de --limit 10 --hours 24 --save

  # Detect language of text
  python translation_cli.py detect "Bonjour le monde"

  # List translations for an article
  python translation_cli.py list abc123

  # Show supported languages and models
  python translation_cli.py languages

  # Show statistics
  python translation_cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Text translation command
    text_parser = subparsers.add_parser('text', help='Translate provided text')
    text_parser.add_argument('text', help='Text content to translate')
    text_parser.add_argument('--target', required=True, help='Target language code (e.g., es, fr, de)')
    text_parser.add_argument('--source', help='Source language code (auto-detected if not provided)')
    text_parser.add_argument('--model', help='Specific model to use for translation')
    
    # Article translation command
    article_parser = subparsers.add_parser('article', help='Translate article from database')
    article_parser.add_argument('article_id', help='Article ID to translate')
    article_parser.add_argument('--target', required=True, help='Target language code')
    article_parser.add_argument('--source', help='Source language code (auto-detected if not provided)')
    article_parser.add_argument('--save', action='store_true', help='Save translation to database')
    
    # Batch translation command
    batch_parser = subparsers.add_parser('batch', help='Batch translate recent articles')
    batch_parser.add_argument('--target', required=True, help='Target language code')
    batch_parser.add_argument('--limit', type=int, default=10, help='Number of articles to process')
    batch_parser.add_argument('--hours', type=int, default=24, help='Hours back to search for articles')
    batch_parser.add_argument('--source', help='Source language code (auto-detected if not provided)')
    batch_parser.add_argument('--save', action='store_true', help='Save translations to database')
    
    # Language detection command
    detect_parser = subparsers.add_parser('detect', help='Detect language of text')
    detect_parser.add_argument('text', help='Text to analyze for language detection')
    
    # List translations command
    list_parser = subparsers.add_parser('list', help='List translations for an article')
    list_parser.add_argument('article_id', help='Article ID to list translations for')
    
    # Languages command
    languages_parser = subparsers.add_parser('languages', help='Show supported languages and models')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show translation statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set environment for database
    os.environ["USE_POSTGRES"] = "true"
    
    # Initialize CLI
    cli = TranslationCLI()
    
    try:
        if args.command == 'text':
            cli.translate_text(
                text=args.text,
                target_language=args.target,
                source_language=args.source,
                model_override=args.model
            )
        
        elif args.command == 'article':
            cli.translate_article(
                article_id=args.article_id,
                target_language=args.target,
                source_language=args.source,
                save_to_db=args.save
            )
        
        elif args.command == 'batch':
            cli.batch_translate_recent(
                target_language=args.target,
                limit=args.limit,
                hours=args.hours,
                source_language=args.source,
                save_to_db=args.save
            )
        
        elif args.command == 'detect':
            cli.detect_language(args.text)
        
        elif args.command == 'list':
            cli.list_translations(args.article_id)
        
        elif args.command == 'languages':
            cli.show_languages()
        
        elif args.command == 'stats':
            cli.show_stats()
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("CLI operation failed")


if __name__ == "__main__":
    main()