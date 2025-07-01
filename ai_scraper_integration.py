#!/usr/bin/env python3
"""
Integration script to add AI analysis to the Mimir scraper workflow.
This script can be run after scraping to analyze new articles.
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

import db_adapter
from ai_analysis import AIAnalysisEngine, analyze_new_articles
from nlp_service import NLPService, NLPTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_recent_articles(hours: int = 24, batch_size: int = 10, use_optimization: bool = True):
    """
    Analyze articles from the last N hours.

    Args:
        hours: Number of hours to look back
        batch_size: Number of articles to process in each batch
        use_optimization: Whether to use memory optimization
    """
    logger.info(f"Analyzing articles from the last {hours} hours")
    logger.info(f"Using optimization: {use_optimization}")

    # Initialize AI engine
    engine = AIAnalysisEngine(use_optimization=use_optimization)

    # Get recent articles
    db = db_adapter.get_connection()

    # For PostgreSQL, we can filter by date
    if db_adapter.USE_POSTGRES:
        since_date = datetime.utcnow() - timedelta(hours=hours)
        articles = []

        with db_adapter.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, url, title, content, source, published_date, category
                FROM articles 
                WHERE created_at >= %s
                AND NOT (metadata ? 'ai_analysis')
                AND content IS NOT NULL AND content != ''
                ORDER BY published_date DESC
            """,
                (since_date,),
            )

            for row in cursor.fetchall():
                articles.append(
                    {
                        "id": row[0],
                        "url": row[1],
                        "title": row[2],
                        "content": row[3],
                        "source": row[4],
                        "published_date": row[5],
                        "category": row[6],
                    }
                )
    else:
        # For SQLite, just get recent articles
        articles = db_adapter.get_recent_articles(limit=100)
        articles = [a for a in articles if a.get("content")]

    if not articles:
        logger.info("No new articles to analyze")
        return

    logger.info(f"Found {len(articles)} articles to analyze")

    # Process in batches
    total_processed = 0
    total_errors = 0
    start_time = time.time()

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} articles)")

        batch_start = time.time()
        results = engine.analyze_batch(batch, save_to_db=True)
        batch_time = time.time() - batch_start

        # Count successes and errors
        for result in results:
            if result.get("errors"):
                total_errors += 1
            else:
                total_processed += 1

        logger.info(f"  Batch processed in {batch_time:.2f} seconds ({batch_time/len(batch):.2f}s per article)")

    total_time = time.time() - start_time

    # Print summary
    logger.info(f"\nAnalysis complete:")
    logger.info(f"  - Total processed: {total_processed}")
    logger.info(f"  - Total errors: {total_errors}")
    logger.info(f"  - Total time: {total_time:.2f} seconds")
    if total_processed > 0:
        logger.info(f"  - Average time per article: {total_time/total_processed:.2f} seconds")

    # Get and display statistics
    stats = engine.get_analysis_stats()
    if stats:
        logger.info("\nAnalysis Statistics:")
        logger.info(f"  - Total analyzed articles: {stats.get('total_analyzed', 0)}")

        if stats.get("by_language"):
            logger.info("  - Articles by language:")
            for lang, count in stats["by_language"].items():
                logger.info(f"    - {lang}: {count}")

        if stats.get("sentiment_distribution"):
            logger.info("  - Sentiment distribution:")
            for sentiment, count in stats["sentiment_distribution"].items():
                logger.info(f"    - {sentiment}: {count}")


def analyze_by_source(source: str, limit: int = 50):
    """
    Analyze articles from a specific source.

    Args:
        source: Source name to filter by
        limit: Maximum number of articles to process
    """
    logger.info(f"Analyzing articles from source: {source}")

    engine = AIAnalysisEngine()

    # Get articles from specific source
    articles = []

    with db_adapter.get_db_connection() as conn:
        cursor = conn.cursor()

        if db_adapter.USE_POSTGRES:
            cursor.execute(
                """
                SELECT id, url, title, content, source, published_date, category
                FROM articles 
                WHERE source = %s
                AND NOT (metadata ? 'ai_analysis')
                AND content IS NOT NULL AND content != ''
                ORDER BY published_date DESC
                LIMIT %s
            """,
                (source, limit),
            )
        else:
            cursor.execute(
                """
                SELECT url, title, content, source, published_date, category
                FROM articles 
                WHERE source = ?
                AND content IS NOT NULL AND content != ''
                ORDER BY published_date DESC
                LIMIT ?
            """,
                (source, limit),
            )

        for row in cursor.fetchall():
            if db_adapter.USE_POSTGRES:
                articles.append(
                    {
                        "id": row[0],
                        "url": row[1],
                        "title": row[2],
                        "content": row[3],
                        "source": row[4],
                        "published_date": row[5],
                        "category": row[6],
                    }
                )
            else:
                articles.append(
                    {
                        "id": row[0],  # URL as ID
                        "url": row[0],
                        "title": row[1],
                        "content": row[2],
                        "source": row[3],
                        "published_date": row[4],
                        "category": row[5],
                    }
                )

    if not articles:
        logger.info(f"No unanalyzed articles found for source: {source}")
        return

    logger.info(f"Found {len(articles)} articles to analyze")

    # Analyze articles
    results = engine.analyze_batch(articles, save_to_db=True)

    # Print summary by sentiment
    sentiment_counts = {}
    for result in results:
        if "sentiment" in result and not result.get("errors"):
            sentiment = result["sentiment"].get("overall_sentiment", "unknown")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

    logger.info(f"\nSentiment summary for {source}:")
    for sentiment, count in sorted(sentiment_counts.items()):
        logger.info(f"  - {sentiment}: {count}")


def start_ai_service(num_workers: int = None):
    """
    Start the NLP service with worker processes.

    Args:
        num_workers: Number of worker processes to start
    """
    logger.info("Starting AI analysis service...")

    try:
        # Create and start service
        service = NLPService(num_workers=num_workers)
        service.start_workers()

        logger.info(f"AI service started with {service.num_workers} workers")
        logger.info("Press Ctrl+C to stop")

        # Keep service running
        import signal
        import time

        def signal_handler(signum, frame):
            """TODO: Add docstring."""
            """TODO: Add docstring."""
            logger.info("Stopping AI service...")
            service.stop_workers()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Monitor queue stats
        while True:
            time.sleep(30)  # Check every 30 seconds
            stats = service.get_queue_stats()
            logger.info(
                f"Queue stats - Pending: {stats['pending_tasks']}, "
                f"Processing: {stats['processing_tasks']}, "
                f"Failed: {stats['failed_tasks']}"
            )

    except Exception as e:
        logger.error(f"Failed to start AI service: {e}")
        sys.exit(1)


def main():
    """Main entry point for the AI integration script."""
    parser = argparse.ArgumentParser(description="AI Analysis integration for Mimir news scraper")

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze recent articles
    recent_parser = subparsers.add_parser("recent", help="Analyze recent articles")
    recent_parser.add_argument("--hours", type=int, default=24, help="Number of hours to look back (default: 24)")
    recent_parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (default: 10)")
    recent_parser.add_argument("--no-optimization", action="store_true", help="Disable memory optimization")

    # Analyze by source
    source_parser = subparsers.add_parser("source", help="Analyze articles from a specific source")
    source_parser.add_argument("source", help="Source name to analyze")
    source_parser.add_argument("--limit", type=int, default=50, help="Maximum articles to process (default: 50)")

    # Analyze all unanalyzed
    all_parser = subparsers.add_parser("all", help="Analyze all unanalyzed articles")
    all_parser.add_argument("--limit", type=int, default=100, help="Maximum articles to process (default: 100)")

    # Start service
    service_parser = subparsers.add_parser("service", help="Start the AI analysis service")
    service_parser.add_argument("--workers", type=int, help="Number of worker processes")

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == "recent":
        use_optimization = not args.no_optimization
        analyze_recent_articles(args.hours, args.batch_size, use_optimization)
    elif args.command == "source":
        analyze_by_source(args.source, args.limit)
    elif args.command == "all":
        analyze_new_articles(args.limit)
    elif args.command == "service":
        start_ai_service(args.workers)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
