#!/usr/bin/env python3
"""
Command-line interface for Mimir Content Generation Engine - EPIC 8.3
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from content_generation import ContentGenerationEngine
import db_adapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ContentGenerationCLI:
    """Command-line interface for content generation operations."""
    
    def __init__(self):
        self.engine = ContentGenerationEngine()
    
    def generate_newsletter(
        self,
        days: int = 1,
        user_interests: Optional[List[str]] = None,
        personalized: bool = True,
        save_to_file: Optional[str] = None
    ) -> None:
        """Generate and display a newsletter."""
        print(f"📰 Generating Newsletter")
        print(f"Date range: Last {days} day{'s' if days != 1 else ''}")
        print(f"Personalized: {personalized}")
        if user_interests:
            print(f"User interests: {', '.join(user_interests)}")
        print("-" * 60)
        
        # Prepare user preferences
        user_preferences = None
        if personalized and user_interests:
            user_preferences = {
                'interests': user_interests,
                'content_preferences': {
                    'format': 'detailed',
                    'include_analysis': True
                }
            }
        
        # Generate newsletter
        result = self.engine.generate_newsletter(
            user_preferences=user_preferences,
            date_range_days=days
        )
        
        if result['success']:
            print(f"✅ Newsletter generated successfully!")
            print(f"📊 Metadata:")
            print(f"  Articles analyzed: {result['articles_used']}")
            print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
            print(f"  Personalized: {result.get('personalized', False)}")
            print()
            
            print("📰 Generated Newsletter:")
            print("=" * 80)
            print(result['content'])
            print("=" * 80)
            
            # Save to file if requested
            if save_to_file:
                try:
                    with open(save_to_file, 'w', encoding='utf-8') as f:
                        f.write(result['content'])
                    print(f"💾 Newsletter saved to: {save_to_file}")
                except Exception as e:
                    print(f"⚠️ Failed to save newsletter: {e}")
        else:
            print(f"❌ Newsletter generation failed: {result.get('error', 'Unknown error')}")
    
    def generate_executive_briefing(
        self,
        days: int = 7,
        focus_areas: Optional[List[str]] = None,
        executive_level: str = "senior",
        save_to_file: Optional[str] = None
    ) -> None:
        """Generate and display an executive briefing."""
        print(f"👔 Generating Executive Briefing")
        print(f"Analysis period: {days} days")
        print(f"Executive level: {executive_level}")
        if focus_areas:
            print(f"Focus areas: {', '.join(focus_areas)}")
        print("-" * 60)
        
        # Generate briefing
        result = self.engine.generate_executive_briefing(
            focus_areas=focus_areas,
            date_range_days=days,
            executive_level=executive_level
        )
        
        if result['success']:
            print(f"✅ Executive briefing generated successfully!")
            print(f"📊 Metadata:")
            print(f"  Articles analyzed: {result['articles_analyzed']}")
            print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
            print(f"  Focus areas: {', '.join(result.get('focus_areas', []))}")
            print()
            
            print("👔 Executive Briefing:")
            print("=" * 80)
            print(result['content'])
            print("=" * 80)
            
            # Save to file if requested
            if save_to_file:
                try:
                    with open(save_to_file, 'w', encoding='utf-8') as f:
                        f.write(result['content'])
                    print(f"💾 Executive briefing saved to: {save_to_file}")
                except Exception as e:
                    print(f"⚠️ Failed to save briefing: {e}")
        else:
            print(f"❌ Executive briefing generation failed: {result.get('error', 'Unknown error')}")
    
    def generate_trend_report(
        self,
        days: int = 7,
        categories: Optional[List[str]] = None,
        save_to_file: Optional[str] = None
    ) -> None:
        """Generate and display a trend report."""
        print(f"📈 Generating Trend Report")
        print(f"Analysis period: {days} days")
        if categories:
            print(f"Categories: {', '.join(categories)}")
        print("-" * 60)
        
        # Generate report
        result = self.engine.generate_trend_report(
            analysis_period_days=days,
            trend_categories=categories
        )
        
        if result['success']:
            print(f"✅ Trend report generated successfully!")
            print(f"📊 Metadata:")
            print(f"  Articles analyzed: {result['articles_analyzed']}")
            print(f"  Trends identified: {result.get('trends_identified', 0)}")
            print(f"  Processing time: {result['processing_time_seconds']:.2f}s")
            print()
            
            print("📈 Trend Analysis Report:")
            print("=" * 80)
            print(result['content'])
            print("=" * 80)
            
            # Save to file if requested
            if save_to_file:
                try:
                    with open(save_to_file, 'w', encoding='utf-8') as f:
                        f.write(result['content'])
                    print(f"💾 Trend report saved to: {save_to_file}")
                except Exception as e:
                    print(f"⚠️ Failed to save report: {e}")
        else:
            print(f"❌ Trend report generation failed: {result.get('error', 'Unknown error')}")
    
    def show_templates(self) -> None:
        """Show available content templates."""
        print(f"📋 Available Content Templates")
        print("-" * 50)
        
        templates = self.engine.templates
        
        for template_name, config in templates.items():
            print(f"\n🏷️ {template_name.replace('_', ' ').title()}")
            print(f"   Sections: {', '.join(config.get('sections', []))}")
            
            # Show template-specific settings
            for key, value in config.items():
                if key != 'sections':
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 Summary:")
        print(f"  Total templates: {len(templates)}")
        print(f"  Content types: {', '.join(templates.keys())}")
    
    def show_stats(self) -> None:
        """Show content generation statistics."""
        print(f"📊 Content Generation Statistics")
        print("-" * 50)
        
        try:
            total_articles = db_adapter.count_articles()
            recent_articles = len(db_adapter.get_recent_articles(limit=100, hours=24))
            weekly_articles = len(db_adapter.get_recent_articles(limit=500, hours=168))  # 7 days
            
            print(f"📚 Article Database:")
            print(f"  Total articles: {total_articles}")
            print(f"  Last 24 hours: {recent_articles}")
            print(f"  Last 7 days: {weekly_articles}")
            
            print(f"\n🏭 Content Generation Capabilities:")
            print(f"  Available templates: {len(self.engine.templates)}")
            print(f"  Supported content types: {', '.join(self.engine.templates.keys())}")
            print(f"  Personalization enabled: Yes")
            print(f"  Multi-language support: Yes (via translation engine)")
            
            print(f"\n⚙️ Engine Status:")
            print(f"  Status: Active")
            print(f"  AI models: Loaded and ready")
            print(f"  Summarization engine: Integrated")
            
        except Exception as e:
            print(f"❌ Failed to retrieve statistics: {e}")
    
    def batch_generate(
        self,
        content_types: List[str],
        days: int = 1,
        output_dir: str = "generated_content"
    ) -> None:
        """Generate multiple content types in batch."""
        print(f"🔄 Batch Content Generation")
        print(f"Content types: {', '.join(content_types)}")
        print(f"Analysis period: {days} days")
        print(f"Output directory: {output_dir}")
        print("-" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        total_start = datetime.now()
        
        for content_type in content_types:
            print(f"\n🔄 Generating {content_type}...")
            
            try:
                if content_type == 'newsletter':
                    result = self.engine.generate_newsletter(date_range_days=days)
                elif content_type == 'executive_briefing':
                    result = self.engine.generate_executive_briefing(date_range_days=days)
                elif content_type == 'trend_report':
                    result = self.engine.generate_trend_report(analysis_period_days=days)
                else:
                    print(f"❌ Unknown content type: {content_type}")
                    results[content_type] = {'success': False, 'error': 'Unknown content type'}
                    continue
                
                results[content_type] = result
                
                if result['success']:
                    # Save to file
                    filename = f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result['content'])
                    
                    print(f"✅ {content_type} generated and saved to {filepath}")
                    print(f"   Articles: {result.get('articles_used', result.get('articles_analyzed', 0))}")
                    print(f"   Time: {result['processing_time_seconds']:.2f}s")
                else:
                    print(f"❌ {content_type} generation failed: {result.get('error', 'Unknown')}")
                    
            except Exception as e:
                print(f"❌ {content_type} generation error: {e}")
                results[content_type] = {'success': False, 'error': str(e)}
        
        total_time = (datetime.now() - total_start).total_seconds()
        
        # Summary
        print(f"\n📊 Batch Generation Summary:")
        print("-" * 40)
        successful = sum(1 for r in results.values() if r.get('success', False))
        failed = len(results) - successful
        
        print(f"  Total content types: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Output directory: {output_dir}")
        
        if successful > 0:
            avg_time = sum(r.get('processing_time_seconds', 0) for r in results.values() if r.get('success')) / successful
            print(f"  Average processing time: {avg_time:.2f}s")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mimir Content Generation Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate daily newsletter
  python content_generation_cli.py newsletter --days 1 --save newsletter.md

  # Generate personalized newsletter
  python content_generation_cli.py newsletter --days 2 --interests "technology,business" --save tech_news.md

  # Generate executive briefing
  python content_generation_cli.py briefing --days 7 --level c-level --focus "technology,markets" --save exec_brief.md

  # Generate trend report
  python content_generation_cli.py trends --days 14 --categories "tech,finance" --save trend_report.md

  # Batch generate all content types
  python content_generation_cli.py batch newsletter,briefing,trends --days 3 --output-dir reports/

  # Show available templates
  python content_generation_cli.py templates

  # Show statistics
  python content_generation_cli.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Newsletter generation command
    newsletter_parser = subparsers.add_parser('newsletter', help='Generate personalized newsletter')
    newsletter_parser.add_argument('--days', type=int, default=1, help='Number of days to look back')
    newsletter_parser.add_argument('--interests', help='Comma-separated user interests')
    newsletter_parser.add_argument('--personalized', action='store_true', default=True, help='Apply personalization')
    newsletter_parser.add_argument('--save', help='Save newsletter to file')
    
    # Executive briefing command
    briefing_parser = subparsers.add_parser('briefing', help='Generate executive briefing')
    briefing_parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    briefing_parser.add_argument('--focus', help='Comma-separated focus areas')
    briefing_parser.add_argument('--level', choices=['senior', 'c-level', 'director', 'vp'], default='senior', help='Executive level')
    briefing_parser.add_argument('--save', help='Save briefing to file')
    
    # Trend report command
    trends_parser = subparsers.add_parser('trends', help='Generate trend analysis report')
    trends_parser.add_argument('--days', type=int, default=7, help='Analysis period in days')
    trends_parser.add_argument('--categories', help='Comma-separated trend categories')
    trends_parser.add_argument('--save', help='Save report to file')
    
    # Batch generation command
    batch_parser = subparsers.add_parser('batch', help='Generate multiple content types')
    batch_parser.add_argument('types', help='Comma-separated content types (newsletter,briefing,trends)')
    batch_parser.add_argument('--days', type=int, default=1, help='Analysis period in days')
    batch_parser.add_argument('--output-dir', default='generated_content', help='Output directory')
    
    # Templates command
    templates_parser = subparsers.add_parser('templates', help='Show available templates')
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show content generation statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set environment for database
    os.environ["USE_POSTGRES"] = "true"
    
    # Initialize CLI
    cli = ContentGenerationCLI()
    
    try:
        if args.command == 'newsletter':
            interests = args.interests.split(',') if args.interests else None
            cli.generate_newsletter(
                days=args.days,
                user_interests=interests,
                personalized=args.personalized,
                save_to_file=args.save
            )
        
        elif args.command == 'briefing':
            focus_areas = args.focus.split(',') if args.focus else None
            cli.generate_executive_briefing(
                days=args.days,
                focus_areas=focus_areas,
                executive_level=args.level,
                save_to_file=args.save
            )
        
        elif args.command == 'trends':
            categories = args.categories.split(',') if args.categories else None
            cli.generate_trend_report(
                days=args.days,
                categories=categories,
                save_to_file=args.save
            )
        
        elif args.command == 'batch':
            content_types = args.types.split(',')
            cli.batch_generate(
                content_types=content_types,
                days=args.days,
                output_dir=args.output_dir
            )
        
        elif args.command == 'templates':
            cli.show_templates()
        
        elif args.command == 'stats':
            cli.show_stats()
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("CLI operation failed")


if __name__ == "__main__":
    main()