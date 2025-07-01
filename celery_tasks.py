"""
Celery Tasks for Mimir Async Scraper
EPIC 4 - Task 4.2.2: Task definitions for scraping, AI processing, notifications
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from celery import chain, chord, group
from celery.utils.log import get_task_logger

import db_manager

# Import async scraper components
from async_scraper import AsyncNewsScraper
from celery_app import app, task_metrics

logger = get_task_logger(__name__)

# Load configuration
CONFIG_FILE = "config.json"
SITES_FILE = "sites_to_scrape.json"


def load_config():
    """Load configuration files."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        with open(SITES_FILE, "r", encoding="utf-8") as f:
            sites = json.load(f)
        return config, sites
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        raise


@app.task(bind=True, name="mimir_scraper.scrape_site")
def scrape_site_task(self, site_config: dict, run_id: int):
    """Scrape a single site asynchronously."""
    start_time = time.time()

    try:
        logger.info(f"Starting scrape for site: {site_config['name']}")

        # Load configuration
        config, _ = load_config()

        # Set database name
        db_manager.DATABASE_NAME = config.get("database_name", "mimir_news.db")

        # Run async scraper for single site
        async def scrape_single_site():
            async with AsyncNewsScraper(config, [site_config]) as scraper:
                return await scraper.process_site(site_config, run_id)

        # Run the async function
        new_articles = asyncio.run(scrape_single_site())

        duration = time.time() - start_time
        logger.info(f"Completed scrape for {site_config['name']}: {new_articles} new articles in {duration:.2f}s")

        return {
            "site_name": site_config["name"],
            "new_articles": new_articles,
            "duration": duration,
            "success": True,
            "run_id": run_id,
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error scraping site {site_config['name']}: {e}")

        # Log error to database
        db_manager.log_scrape_error(run_id, site_config["name"], type(e).__name__, str(e), site_config.get("url", ""))

        return {
            "site_name": site_config["name"],
            "new_articles": 0,
            "duration": duration,
            "success": False,
            "error": str(e),
            "run_id": run_id,
        }


@app.task(bind=True, name="mimir_scraper.scrape_article")
def scrape_article_task(self, article_url: str, site_config: dict, run_id: int):
    """Scrape a single article."""
    start_time = time.time()

    try:
        logger.debug(f"Scraping article: {article_url}")

        # Load configuration
        config, _ = load_config()

        # Set database name
        db_manager.DATABASE_NAME = config.get("database_name", "mimir_news.db")

        # Keywords for monitoring
        keywords_config = config.get("keywords_monitoring", {})
        monitor_keywords = keywords_config.get("enabled", False)
        keywords_list = keywords_config.get("keywords", [])

        # Run async article processing
        async def process_single_article():
            async with AsyncNewsScraper(config, [site_config]) as scraper:
                return await scraper.process_article(article_url, site_config, run_id, monitor_keywords, keywords_list)

        # Run the async function
        article_id = asyncio.run(process_single_article())

        duration = time.time() - start_time

        if article_id:
            logger.debug(f"Successfully processed article {article_url}")
            return {"article_url": article_url, "article_id": article_id, "duration": duration, "success": True}
        else:
            logger.warning(f"Article not processed (possibly duplicate): {article_url}")
            return {
                "article_url": article_url,
                "article_id": None,
                "duration": duration,
                "success": False,
                "reason": "not_processed",
            }

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error processing article {article_url}: {e}")

        return {"article_url": article_url, "article_id": None, "duration": duration, "success": False, "error": str(e)}


@app.task(bind=True, name="mimir_scraper.run_scheduled_scraping")
def run_scheduled_scraping(self):
    """Run complete scraping process for all enabled sites."""
    logger.info("Starting scheduled scraping run")

    try:
        # Load configuration
        config, sites_to_scrape = load_config()

        # Set database name
        db_manager.DATABASE_NAME = config.get("database_name", "mimir_news.db")

        # Initialize database
        db_manager.init_db()

        # Start new scrape run
        run_id = db_manager.start_scrape_run()

        # Filter enabled sites
        enabled_sites = [site for site in sites_to_scrape if site.get("enabled", False)]

        if not enabled_sites:
            logger.warning("No enabled sites found")
            return {"message": "No enabled sites found", "run_id": run_id}

        # Create scraping tasks for each site
        scraping_jobs = group(scrape_site_task.s(site_config, run_id) for site_config in enabled_sites)

        # Create a chord that processes all sites and then sends notifications
        scraping_workflow = chord(scraping_jobs)(process_scraping_results.s(run_id))

        logger.info(f"Scheduled scraping workflow started with {len(enabled_sites)} sites")

        return {
            "message": f"Scraping started for {len(enabled_sites)} sites",
            "run_id": run_id,
            "workflow_id": scraping_workflow.id,
        }

    except Exception as e:
        logger.error(f"Error starting scheduled scraping: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)


@app.task(bind=True, name="mimir_scraper.process_scraping_results")
def process_scraping_results(self, scraping_results: List[dict], run_id: int):
    """Process results from all scraping tasks and send notifications."""
    logger.info(f"Processing scraping results for run {run_id}")

    try:
        # Aggregate results
        total_new_articles = 0
        total_errors = 0
        successful_sites = []
        failed_sites = []

        for result in scraping_results:
            if result.get("success", False):
                total_new_articles += result.get("new_articles", 0)
                successful_sites.append(result["site_name"])
            else:
                total_errors += 1
                failed_sites.append({"site_name": result["site_name"], "error": result.get("error", "Unknown error")})

        # Update scrape run statistics
        db_manager.end_scrape_run(run_id, 0, total_new_articles, total_errors)

        # Log summary
        logger.info(f"Scraping run {run_id} completed: {total_new_articles} new articles, {total_errors} errors")

        # Schedule notification if there are new articles or errors
        if total_new_articles > 0 or total_errors > 0:
            send_notification.delay(run_id, total_new_articles, total_errors, failed_sites)

        # Schedule AI processing for new articles if enabled
        if total_new_articles > 0:
            schedule_ai_processing.delay(run_id)

        return {
            "run_id": run_id,
            "total_new_articles": total_new_articles,
            "total_errors": total_errors,
            "successful_sites": successful_sites,
            "failed_sites": failed_sites,
        }

    except Exception as e:
        logger.error(f"Error processing scraping results: {e}")
        raise


@app.task(bind=True, name="mimir_scraper.send_notification")
def send_notification(self, run_id: int, new_articles: int, errors: int, failed_sites: List[dict]):
    """Send email notification about scraping results."""
    logger.info(f"Sending notification for run {run_id}")

    try:
        # Load configuration
        config, _ = load_config()

        # Import email functionality from original scraper
        from scraper import send_email_summary

        # Get recent articles for email
        new_articles_details = []
        keyword_alerts_details = []

        if new_articles > 0:
            new_articles_details = db_manager.get_recent_articles(limit=new_articles, hours=1)

            # Get keyword alerts
            with db_manager.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                SELECT ka.keyword, ka.context, a.title as article_title, a.url
                FROM keyword_alerts ka
                JOIN articles a ON ka.article_id = a.id
                WHERE ka.created_at > datetime('now', '-1 hour')
                ORDER BY ka.created_at DESC
                """
                )
                keyword_alerts_details = [dict(row) for row in cursor.fetchall()]

        # Send email summary
        if (new_articles_details or keyword_alerts_details) and config.get("notification_email", {}).get("send_email"):
            send_email_summary(new_articles_details, keyword_alerts_details)
            logger.info("Email notification sent successfully")
        else:
            logger.info("Email notifications disabled or no content to send")

        return {"run_id": run_id, "notification_sent": True, "new_articles": new_articles, "errors": errors}

    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=2)


@app.task(bind=True, name="mimir_scraper.process_ai_analysis")
def process_ai_analysis(self, article_ids: List[int]):
    """Process AI analysis for a batch of articles."""
    logger.info(f"Processing AI analysis for {len(article_ids)} articles")

    try:
        # Check if AI analysis is available
        try:
            from ai_scraper_integration import analyze_articles_batch

            # Process articles in batch
            results = analyze_articles_batch(article_ids)

            logger.info(f"AI analysis completed for {len(article_ids)} articles")
            return {"processed_articles": len(article_ids), "success": True, "results": results}

        except ImportError:
            logger.warning("AI analysis module not available")
            return {"processed_articles": 0, "success": False, "error": "AI analysis module not available"}

    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        raise self.retry(exc=e, countdown=120, max_retries=2)


@app.task(bind=True, name="mimir_scraper.schedule_ai_processing")
def schedule_ai_processing(self, run_id: int):
    """Schedule AI processing for articles from a scraping run."""
    logger.info(f"Scheduling AI processing for run {run_id}")

    try:
        # Get recent articles that need AI processing
        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT id FROM articles 
            WHERE scraped_at > datetime('now', '-2 hours')
            AND id NOT IN (
                SELECT DISTINCT article_id FROM ai_analysis 
                WHERE article_id IS NOT NULL
            )
            ORDER BY scraped_at DESC
            LIMIT 100
            """
            )
            article_ids = [row[0] for row in cursor.fetchall()]

        if not article_ids:
            logger.info("No articles need AI processing")
            return {"message": "No articles need AI processing"}

        # Process in batches of 10
        batch_size = 10
        batches = [article_ids[i : i + batch_size] for i in range(0, len(article_ids), batch_size)]

        # Schedule AI processing tasks
        ai_jobs = group(process_ai_analysis.s(batch) for batch in batches)

        workflow = ai_jobs.apply_async()

        logger.info(f"Scheduled AI processing for {len(article_ids)} articles in {len(batches)} batches")

        return {"articles_scheduled": len(article_ids), "batches": len(batches), "workflow_id": workflow.id}

    except Exception as e:
        logger.error(f"Error scheduling AI processing: {e}")
        raise


@app.task(bind=True, name="mimir_scraper.cleanup_old_data")
def cleanup_old_data(self):
    """Clean up old data from the database."""
    logger.info("Starting cleanup of old data")

    try:
        # Load configuration
        config, _ = load_config()
        cleanup_config = config.get("cleanup", {})

        # Set database name
        db_manager.DATABASE_NAME = config.get("database_name", "mimir_news.db")

        # Default cleanup settings
        days_to_keep = cleanup_config.get("articles_days_to_keep", 90)
        scrape_runs_to_keep = cleanup_config.get("scrape_runs_to_keep", 100)

        cleaned_items = 0

        with db_manager.get_db_connection() as conn:
            cursor = conn.cursor()

            # Clean up old articles
            cursor.execute(
                """
            DELETE FROM articles 
            WHERE scraped_at < datetime('now', '-{} days')
            """.format(
                    days_to_keep
                )
            )
            cleaned_articles = cursor.rowcount
            cleaned_items += cleaned_articles

            # Clean up old scrape runs
            cursor.execute(
                """
            DELETE FROM scrape_runs 
            WHERE id NOT IN (
                SELECT id FROM scrape_runs 
                ORDER BY started_at DESC 
                LIMIT {}
            )
            """.format(
                    scrape_runs_to_keep
                )
            )
            cleaned_runs = cursor.rowcount
            cleaned_items += cleaned_runs

            # Clean up orphaned keyword alerts
            cursor.execute(
                """
            DELETE FROM keyword_alerts 
            WHERE article_id NOT IN (SELECT id FROM articles)
            """
            )
            cleaned_alerts = cursor.rowcount
            cleaned_items += cleaned_alerts

            conn.commit()

        logger.info(f"Cleanup completed: {cleaned_articles} articles, {cleaned_runs} runs, {cleaned_alerts} alerts")

        return {
            "cleaned_articles": cleaned_articles,
            "cleaned_runs": cleaned_runs,
            "cleaned_alerts": cleaned_alerts,
            "total_cleaned": cleaned_items,
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise


@app.task(bind=True, name="mimir_scraper.generate_daily_report")
def generate_daily_report(self):
    """Generate daily scraping report."""
    logger.info("Generating daily report")

    try:
        # Get statistics from database
        stats = db_manager.get_statistics()

        # Get task metrics from Redis
        task_names = [
            "mimir_scraper.scrape_site",
            "mimir_scraper.process_ai_analysis",
            "mimir_scraper.send_notification",
        ]

        task_stats = {}
        for task_name in task_names:
            task_stats[task_name] = task_metrics.get_task_metrics(task_name)

        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "database_stats": stats,
            "task_stats": task_stats,
            "generated_at": datetime.now().isoformat(),
        }

        # Save report to file
        report_file = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs("reports", exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Daily report generated: {report_file}")

        return {"report_file": report_file, "total_articles": stats.get("total_articles", 0), "task_stats": task_stats}

    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise


# Utility tasks for monitoring and management
@app.task(bind=True, name="mimir_scraper.get_system_status")
def get_system_status(self):
    """Get comprehensive system status."""
    from celery_app import get_active_tasks, get_queue_length, get_worker_stats

    return {
        "timestamp": datetime.now().isoformat(),
        "queue_lengths": {
            "scraping": get_queue_length("scraping"),
            "ai_processing": get_queue_length("ai_processing"),
            "notifications": get_queue_length("notifications"),
            "maintenance": get_queue_length("maintenance"),
        },
        "active_tasks": get_active_tasks(),
        "worker_stats": get_worker_stats(),
        "task_metrics": {
            task_name: task_metrics.get_task_metrics(task_name)
            for task_name in [
                "mimir_scraper.scrape_site",
                "mimir_scraper.process_ai_analysis",
                "mimir_scraper.send_notification",
            ]
        },
    }
