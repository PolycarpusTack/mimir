import json
import logging
import os
import re
import smtplib
import sys
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import feedparser
import requests
import schedule
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

import db_manager

# Import API modules if available
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False


def setup_logging(config: dict) -> logging.Logger:
    """Configureer logging met rotating file handler."""
    log_config = config.get("logging", {})

    # Maak logs directory als het niet bestaat
    log_dir = os.path.dirname(log_config.get("file", "logs/mimir_scraper.log"))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configureer logger
    logger = logging.getLogger("mimir_scraper")
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

    # File handler met rotation
    file_handler = RotatingFileHandler(
        log_config.get("file", "logs/mimir_scraper.log"),
        maxBytes=log_config.get("max_bytes", 10485760),  # 10MB
        backupCount=log_config.get("backup_count", 5),
    )
    file_handler.setFormatter(
        logging.Formatter(log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger


# --- CONFIGURATIE LADEN ---
CONFIG_FILE = "config.json"
SITES_FILE = "sites_to_scrape.json"

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
    with open(SITES_FILE, "r", encoding="utf-8") as f:
        sites_data = json.load(f)
        # Handle both old format (direct array) and new format (with "sites" key)
        if isinstance(sites_data, list):
            SITES_TO_SCRAPE = sites_data
        elif isinstance(sites_data, dict) and "sites" in sites_data:
            SITES_TO_SCRAPE = sites_data["sites"]
        else:
            raise ValueError("Invalid sites_to_scrape.json format")
except FileNotFoundError as e:
    print(f"Fout: {e.filename} niet gevonden. Zorg dat deze bestanden bestaan.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Fout bij het parsen van JSON: {e}")
    sys.exit(1)
# Setup logging
logger = setup_logging(CONFIG)

# Globale variabelen
USER_AGENT = CONFIG.get("user_agent", "Mimir-Scraper/1.0")
DEFAULT_DELAY = CONFIG.get("default_request_delay_seconds", 2)
MAX_RETRIES = CONFIG.get("max_retries", 3)
RETRY_DELAY = CONFIG.get("retry_delay_seconds", 5)
db_manager.DATABASE_NAME = CONFIG.get("database_name", "mimir_news.db")

# Enhanced proxy system integration
try:
    from scraper_proxy_integration import initialize_scraper_proxy_system, get_scraper_proxy_manager
    proxy_manager = initialize_scraper_proxy_system(CONFIG)
    
    # Log proxy system status
    status = proxy_manager.get_proxy_status()
    if status["enabled"] and not status["initialization_error"]:
        logger.info(f"✅ Proxy system active with {status.get('proxy_system', {}).get('total_proxies', 0)} proxies")
    elif status["enabled"] and status["initialization_error"]:
        logger.warning(f"⚠️  Proxy system enabled but failed: {status['initialization_error']}")
    else:
        logger.info("ℹ️  Proxy system disabled")
    
    # Create fallback session for any remaining direct session usage
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    
except ImportError as e:
    logger.warning(f"Enhanced proxy integration not available: {e}")
    proxy_manager = None
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
except Exception as e:
    logger.error(f"Failed to initialize proxy system: {e}")
    proxy_manager = None
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})


# --- HELPER FUNCTIES ---
def check_robots_txt(base_url: str, path: str = "/") -> bool:
    """Controleer of we een pad mogen scrapen volgens robots.txt."""
    try:
        rp = RobotFileParser()
        rp.set_url(urljoin(base_url, "/robots.txt"))
        rp.read()
        can_fetch = rp.can_fetch(USER_AGENT, urljoin(base_url, path))
        if not can_fetch:
            logger.warning(f"Robots.txt verbiedt toegang tot {path} op {base_url}")
        return can_fetch
    except Exception as e:
        logger.warning(f"Kon robots.txt niet lezen voor {base_url}: {e}")
        return True  # Bij twijfel, ga door (maar wees voorzichtig)


def fetch_url(url: str, retries: int = 0) -> Optional[str]:
    """Haalt de content van een URL op met enhanced proxy support."""
    # Use enhanced proxy system if available
    if proxy_manager is not None:
        try:
            from scraper_proxy_integration import enhanced_fetch_url
            return enhanced_fetch_url(url, CONFIG, retries)
        except ImportError:
            logger.debug("Enhanced fetch_url not available, using fallback")
    
    # Fallback to original implementation
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        logger.debug(f"Succesvol opgehaald: {url}")
        time.sleep(DEFAULT_DELAY)
        return response.text
    except requests.exceptions.RequestException as e:
        if retries < MAX_RETRIES:
            logger.warning(f"Fout bij ophalen {url}, poging {retries + 1}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY)
            return fetch_url(url, retries + 1)
        logger.error(f"Definitief gefaald bij ophalen {url}: {e}")
        return None


def extract_keywords(text: str, keywords_list: List[str]) -> List[Tuple[str, str]]:
    """Zoek keywords in tekst en return (keyword, context) tuples."""
    found_keywords = []
    text_lower = text.lower()

    for keyword in keywords_list:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Vind context rond het keyword (50 karakters voor en na)
            pattern = re.compile(f".{{0,50}}{re.escape(keyword_lower)}.{{0,50}}", re.IGNORECASE)
            matches = pattern.findall(text)
            for match in matches[:3]:  # Maximaal 3 contexten per keyword
                found_keywords.append((keyword, match.strip()))

    return found_keywords


def parse_date(date_string: str, date_format: Optional[str] = None) -> str:
    """Parse een datum string naar ISO formaat."""
    if not date_string:
        return datetime.now().isoformat()

    try:
        if date_format:
            # Probeer eerst het opgegeven formaat
            dt = datetime.strptime(date_string, date_format)
        else:
            # Gebruik dateutil parser voor flexibele parsing
            dt = date_parser.parse(date_string)
        return dt.isoformat()
    except Exception as e:
        logger.warning(f"Kon datum '{date_string}' niet parsen: {e}")
        return date_string  # Return originele string als fallback


# --- PARSING FUNCTIES ---
def parse_rss_feed(feed_content: str, site_config: Dict) -> int:
    """Parset een RSS feed en voegt artikelen toe aan de database."""
    feed = feedparser.parse(feed_content)
    new_articles_count = 0
    source_name = site_config["name"]
    category = site_config.get("category", "general")

    # Keywords voor monitoring
    keywords_config = CONFIG.get("keywords_monitoring", {})
    monitor_keywords = keywords_config.get("enabled", False)
    keywords_list = keywords_config.get("keywords", [])
    for entry in feed.entries:
        title = entry.get("title", "Geen titel")
        link = entry.get("link", None)

        if not link:
            continue

        if db_manager.article_exists(link):
            continue

        # Parse publicatiedatum
        pub_date_parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if pub_date_parsed:
            publication_date = datetime(*pub_date_parsed[:6]).isoformat()
        else:
            publication_date = parse_date(entry.get("published") or entry.get("updated", ""))

        # Haal content
        content_summary = entry.get("summary", "")
        full_content = entry.get("content", [{}])[0].get("value", "") if "content" in entry else ""
        author = entry.get("author", "")

        # Zoek keywords
        keywords_found = []
        if monitor_keywords:
            combined_text = f"{title} {content_summary} {full_content}"
            keyword_matches = extract_keywords(combined_text, keywords_list)
            keywords_found = [kw[0] for kw in keyword_matches]

        # Voeg artikel toe
        article_id = db_manager.add_article(
            url=link,
            title=title,
            publication_date=publication_date,
            source_website=source_name,
            category=category,
            content_summary=content_summary[:1000],  # Limit summary
            full_content=full_content[:5000],  # Limit content
            author=author,
            keywords_found=keywords_found,
        )
        if article_id:
            new_articles_count += 1

            # Log keyword alerts
            if monitor_keywords and keywords_found:
                for kw, context in keyword_matches:
                    db_manager.add_keyword_alert(article_id, kw, context)

    return new_articles_count


def parse_html_site(site_config: Dict, run_id: int) -> int:
    """Parset een HTML site met pagination support."""
    base_url = site_config["url"]
    new_articles_count = 0
    source_name = site_config["name"]
    category = site_config.get("category", "general")

    # Keywords voor monitoring
    keywords_config = CONFIG.get("keywords_monitoring", {})
    monitor_keywords = keywords_config.get("enabled", False)
    keywords_list = keywords_config.get("keywords", [])

    # Pagination settings
    pagination = site_config.get("pagination", {})
    max_pages = pagination.get("max_pages", 1) if pagination.get("enabled", False) else 1

    current_url = base_url
    pages_processed = 0

    while current_url and pages_processed < max_pages:
        logger.info(f"Processing pagina {pages_processed + 1} van {source_name}: {current_url}")

        # Check robots.txt
        if not check_robots_txt(base_url, urlparse(current_url).path):
            logger.warning(f"Robots.txt verbiedt toegang tot {current_url}")
            break
        overview_html = fetch_url(current_url)
        if not overview_html:
            db_manager.log_scrape_error(
                run_id, source_name, "FETCH_ERROR", f"Kon overzichtspagina niet ophalen", current_url
            )
            break

        soup_overview = BeautifulSoup(overview_html, "lxml")

        # Zoek artikel links
        selectors = site_config.get("selectors", {})
        article_links = soup_overview.select(selectors.get("overview_article_link", ""))

        logger.info(f"Gevonden: {len(article_links)} artikel links op pagina {pages_processed + 1}")

        for link_element in article_links:
            article_url = link_element.get("href")
            if not article_url:
                continue

            # Maak absolute URL
            article_url = urljoin(base_url, article_url)

            if db_manager.article_exists(article_url):
                continue

            # Haal artikel details op
            logger.debug(f"Artikel details ophalen: {article_url}")
            detail_html = fetch_url(article_url)

            if not detail_html:
                db_manager.log_scrape_error(
                    run_id, source_name, "FETCH_ERROR", f"Kon artikel niet ophalen", article_url
                )
                continue
            soup_detail = BeautifulSoup(detail_html, "lxml")

            # Extract artikel informatie
            try:
                title_elem = soup_detail.select_one(selectors.get("detail_title", ""))
                title = title_elem.get_text(strip=True) if title_elem else "Geen titel"
            except Exception as e:
                logger.error(f"Fout bij titel extractie: {e}")
                title = "Geen titel"

            # Datum
            publication_date = datetime.now().isoformat()
            try:
                date_elem = soup_detail.select_one(selectors.get("detail_date", ""))
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    publication_date = parse_date(date_text, site_config.get("date_format"))
            except Exception as e:
                logger.warning(f"Fout bij datum extractie: {e}")

            # Content
            content_summary = ""
            full_content = ""
            try:
                content_elem = soup_detail.select_one(selectors.get("detail_content", ""))
                if content_elem:
                    full_content = content_elem.get_text(strip=True, separator="\n")
                    content_summary = full_content[:500] + "..." if len(full_content) > 500 else full_content
            except Exception as e:
                logger.warning(f"Fout bij content extractie: {e}")
            # Author
            author = ""
            try:
                author_elem = soup_detail.select_one(selectors.get("detail_author", ""))
                if author_elem:
                    author = author_elem.get_text(strip=True)
            except Exception:
                pass

            # Keywords monitoring
            keywords_found = []
            keyword_matches = []
            if monitor_keywords:
                combined_text = f"{title} {full_content}"
                keyword_matches = extract_keywords(combined_text, keywords_list)
                keywords_found = [kw[0] for kw in keyword_matches]

            # Voeg artikel toe
            article_id = db_manager.add_article(
                url=article_url,
                title=title,
                publication_date=publication_date,
                source_website=source_name,
                category=category,
                content_summary=content_summary,
                full_content=full_content[:5000],
                author=author,
                keywords_found=keywords_found,
            )

            if article_id:
                new_articles_count += 1

                # Log keyword alerts
                if keyword_matches:
                    for kw, context in keyword_matches:
                        db_manager.add_keyword_alert(article_id, kw, context)
        # Zoek volgende pagina
        pages_processed += 1
        if pagination.get("enabled", False) and pages_processed < max_pages:
            next_page_elem = soup_overview.select_one(pagination.get("next_page_selector", ""))
            if next_page_elem:
                next_url = next_page_elem.get("href")
                if next_url:
                    current_url = urljoin(base_url, next_url)
                    logger.info(f"Ga naar volgende pagina: {current_url}")
                else:
                    current_url = None
            else:
                current_url = None
        else:
            current_url = None

    return new_articles_count


# --- EMAIL FUNCTIONALITEIT ---
def send_email_summary(new_articles: List[dict], keyword_alerts: List[dict]):
    """Verstuurt een email samenvatting van nieuwe artikelen en keyword alerts."""
    email_config = CONFIG.get("notification_email", {})

    if not email_config.get("send_email"):
        logger.info("Email notificaties zijn uitgeschakeld")
        return

    if not new_articles and not keyword_alerts:
        logger.info("Geen nieuwe content om te mailen")
        return

    # Maak email content
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Mimir Nieuws Update - {len(new_articles)} nieuwe artikelen"
    msg["From"] = email_config["smtp_user"]
    msg["To"] = email_config["recipient_email"]
    # Plain text versie
    text_parts = ["=== MIMIR NIEUWS UPDATE ===\n"]
    text_parts.append(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if new_articles:
        text_parts.append(f"\n📰 NIEUWE ARTIKELEN ({len(new_articles)}):\n")
        for article in new_articles:
            text_parts.append(f"\n• {article['title']}")
            text_parts.append(f"  Bron: {article['source_website']}")
            text_parts.append(f"  Categorie: {article.get('category', 'Algemeen')}")
            text_parts.append(f"  Datum: {article['publication_date']}")
            text_parts.append(f"  Link: {article['url']}")
            if article.get("content_summary"):
                text_parts.append(f"  Samenvatting: {article['content_summary'][:200]}...")

    if keyword_alerts:
        text_parts.append(f"\n\n🔍 KEYWORD ALERTS ({len(keyword_alerts)}):\n")
        for alert in keyword_alerts:
            text_parts.append(f"\n• Keyword: {alert['keyword']}")
            text_parts.append(f"  Artikel: {alert['article_title']}")
            text_parts.append(f"  Context: ...{alert['context']}...")

    text_content = "\n".join(text_parts)

    # HTML versie
    html_parts = ['<html><body style="font-family: Arial, sans-serif;">']
    html_parts.append("<h2>🚀 Mimir Nieuws Update</h2>")
    html_parts.append(f'<p>Datum: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
    if new_articles:
        html_parts.append(f"<h3>📰 Nieuwe Artikelen ({len(new_articles)})</h3>")
        for article in new_articles:
            html_parts.append('<div style="margin-bottom: 20px; padding: 10px; border-left: 3px solid #3498db;">')
            html_parts.append(f'<h4 style="margin: 0;">{article["title"]}</h4>')
            html_parts.append(f'<p style="color: #666; margin: 5px 0;">Bron: {article["source_website"]} | ')
            html_parts.append(f'Categorie: {article.get("category", "Algemeen")} | ')
            html_parts.append(f'Datum: {article["publication_date"]}</p>')
            html_parts.append(f'<p><a href="{article["url"]}">Lees meer →</a></p>')
            if article.get("content_summary"):
                html_parts.append(f'<p style="color: #333;">{article["content_summary"][:200]}...</p>')
            html_parts.append("</div>")

    if keyword_alerts:
        html_parts.append(f"<h3>🔍 Keyword Alerts ({len(keyword_alerts)})</h3>")
        for alert in keyword_alerts:
            html_parts.append('<div style="margin-bottom: 15px; padding: 10px; background: #f0f0f0;">')
            html_parts.append(f'<strong>Keyword:</strong> <span style="color: #e74c3c;">{alert["keyword"]}</span><br>')
            html_parts.append(f'<strong>Artikel:</strong> {alert["article_title"]}<br>')
            html_parts.append(f'<strong>Context:</strong> <em>...{alert["context"]}...</em>')
            html_parts.append("</div>")

    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)

    # Voeg beide versies toe
    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))
    # Verstuur email
    try:
        with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
            server.starttls()
            server.login(email_config["smtp_user"], email_config["smtp_password"])
            server.send_message(msg)
        logger.info("Email samenvatting succesvol verzonden")
    except Exception as e:
        logger.error(f"Fout bij verzenden email: {e}")


# --- HOOFD SCRAPER FUNCTIE ---
def run_scraper():
    """Hoofdfunctie die alle sites scraped."""
    logger.info(f"{'='*50}")
    logger.info(f"Scraper gestart op {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialiseer database
    db_manager.init_db()

    # Start nieuwe scrape run
    run_id = db_manager.start_scrape_run()

    # Statistieken
    total_articles_found = 0
    total_new_articles = 0
    total_errors = 0
    new_articles_details = []
    keyword_alerts_details = []

    # Verwerk elke site
    for site_config in SITES_TO_SCRAPE:
        if not site_config.get("enabled", False):
            logger.info(f"Site '{site_config['name']}' is uitgeschakeld, overslaan")
            continue
        logger.info(f"\nVerwerken: {site_config['name']} ({site_config['type']})")

        try:
            site_new_articles = 0

            if site_config["type"] == "rss":
                feed_content = fetch_url(site_config["url"])
                if feed_content:
                    site_new_articles = parse_rss_feed(feed_content, site_config)
                else:
                    raise Exception("Kon RSS feed niet ophalen")

            elif site_config["type"] == "html":
                site_new_articles = parse_html_site(site_config, run_id)

            elif site_config["type"] == "playwright":
                from playwright_integration import parse_playwright_site
                site_new_articles = parse_playwright_site(site_config, run_id)

            else:
                logger.warning(f"Onbekend site type: {site_config['type']}")
                continue

            total_new_articles += site_new_articles
            logger.info(f"✓ {site_config['name']}: {site_new_articles} nieuwe artikelen")

        except Exception as e:
            logger.error(f"Fout bij verwerken van {site_config['name']}: {e}")
            db_manager.log_scrape_error(run_id, site_config["name"], type(e).__name__, str(e), site_config["url"])
            total_errors += 1

    # Scrape API sources if enabled
    api_results = {}
    if CONFIG.get("api_sources", {}).get("enabled", False):
        try:
            from api_scraper import scrape_api_sources
            logger.info("Starting API sources scraping...")
            api_results = scrape_api_sources()
            total_new_articles += api_results.get('total', 0)
            logger.info(f"API sources: {api_results.get('total', 0)} nieuwe artikelen")
        except ImportError:
            logger.warning("API scraper module not available - install newsapi-python and praw")
        except Exception as e:
            logger.error(f"Error scraping API sources: {e}")
            total_errors += 1
    
    # Scrape broadcast industry sources if enabled
    broadcast_results = {}
    if os.path.exists('broadcast_sources.json'):
        try:
            from broadcast_scraper import scrape_broadcast_sources
            logger.info("Starting broadcast industry scraping...")
            # Get limit from config
            limit = CONFIG.get("api_sources", {}).get("articles_per_run", 100)
            broadcast_results = scrape_broadcast_sources(limit_per_source=limit // 10)  # Divide by number of sources
            total_new_articles += broadcast_results.get('total', 0)
            logger.info(f"Broadcast sources: {broadcast_results.get('total', 0)} nieuwe artikelen")
        except Exception as e:
            logger.error(f"Error scraping broadcast sources: {e}")
            total_errors += 1
    
    # Update scrape run statistieken
    db_manager.end_scrape_run(run_id, total_articles_found, total_new_articles, total_errors)
    # Haal nieuwe artikelen op voor email
    if total_new_articles > 0:
        new_articles_details = db_manager.get_recent_articles(limit=total_new_articles, hours=1)

        # Haal keyword alerts op
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

    # Log samenvatting inclusief proxy statistieken
    logger.info(f"\n{'='*50}")
    logger.info(f"Scraper run voltooid")
    logger.info(f"Nieuwe artikelen: {total_new_articles}")
    logger.info(f"Keyword alerts: {len(keyword_alerts_details)}")
    logger.info(f"Fouten: {total_errors}")
    
    # Log proxy performance if available
    if proxy_manager is not None:
        try:
            from scraper_proxy_integration import log_proxy_performance_summary
            log_proxy_performance_summary()
        except ImportError:
            pass
    
    logger.info(f"{'='*50}\n")

    # Verstuur email samenvatting
    if (new_articles_details or keyword_alerts_details) and CONFIG.get("notification_email", {}).get("send_email"):
        send_email_summary(new_articles_details, keyword_alerts_details)


def show_statistics():
    """Toon database statistieken."""
    stats = db_manager.get_statistics()

    print("\n" + "=" * 60)
    print("MIMIR SCRAPER STATISTIEKEN")
    print("=" * 60)
    print(f"Totaal aantal artikelen: {stats['total_articles']}")
    print(f"\nArtikelen per bron:")
    for source, count in stats["articles_per_source"].items():
        print(f"  - {source}: {count}")

    print(f"\nKeyword statistieken:")
    for keyword, count in stats["keyword_counts"].items():
        print(f"  - {keyword}: {count} keer gevonden")

    print(f"\nRecente scrape runs:")
    for run in stats["recent_runs"][:5]:
        status = "✓" if run["status"] == "completed" else "✗"
        print(f"  {status} {run['started_at']} - {run['articles_new']} nieuwe artikelen")
    print("=" * 60 + "\n")


# --- MAIN ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mimir News Scraper")
    parser.add_argument("--run", action="store_true", help="Voer scraper uit")
    parser.add_argument("--schedule", action="store_true", help="Start geplande scraper")
    parser.add_argument("--stats", action="store_true", help="Toon statistieken")
    parser.add_argument("--test-email", action="store_true", help="Test email configuratie")

    args = parser.parse_args()

    if args.stats:
        show_statistics()
    elif args.test_email:
        logger.info("Test email verzenden...")
        test_articles = [
            {
                "title": "Test Artikel",
                "url": "https://example.com",
                "source_website": "Test Bron",
                "publication_date": datetime.now().isoformat(),
                "content_summary": "Dit is een test artikel voor email configuratie.",
            }
        ]
        send_email_summary(test_articles, [])
    elif args.schedule:
        # Schedule de scraper
        schedule.every(4).hours.do(run_scraper)
        # schedule.every().day.at("08:00").do(run_scraper)

        logger.info("Scraper is gepland (elke 4 uur). Druk Ctrl+C om te stoppen.")
        run_scraper()  # Voer direct uit bij start

        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        # Standaard: voer scraper eenmalig uit
        run_scraper()
