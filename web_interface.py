#!/usr/bin/env python3
"""
Mimir News Scraper - Netflix-Style Web Interface
A modern, user-friendly interface for browsing scraped news articles
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta
import os
import sys
import logging
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG_FILE = 'config.json'
SITES_FILE = 'sites_to_scrape.json'

# Load database name from config
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        DATABASE = config.get('database', 'mimir_news.db')
except:
    DATABASE = 'mimir_news.db'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def load_config():
    """Load configuration from JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def load_sites():
    """Load sites configuration"""
    try:
        with open(SITES_FILE, 'r') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, list):
                return {"sites": data}
            return data
    except Exception as e:
        logger.error(f"Error loading sites: {e}")
        return {"sites": []}

def get_swimlanes():
    """Get swimlane configuration with keywords"""
    # For MVP, we'll use hardcoded swimlanes based on keywords from config
    config = load_config()
    keywords = config.get('keywords', [])
    
    # Default swimlanes
    swimlanes = [
        {
            'id': 'latest',
            'title': 'Latest Articles',
            'icon': '🔥',
            'keywords': [],
            'type': 'latest',
            'editable': False
        }
    ]
    
    # Create swimlanes from keyword groups
    # Group similar keywords together
    keyword_groups = {
        'AI & Machine Learning': ['AI', 'artificial intelligence', 'machine learning', 'ML', 'GPT', 'neural', 'deep learning'],
        'Cybersecurity': ['security', 'breach', 'vulnerability', 'ransomware', 'cyber', 'hack'],
        'Technology Companies': ['Apple', 'Google', 'Microsoft', 'Amazon', 'Meta', 'Tesla'],
        'Cloud & Infrastructure': ['AWS', 'Azure', 'cloud', 'kubernetes', 'docker'],
        'Blockchain & Crypto': ['blockchain', 'crypto', 'bitcoin', 'ethereum', 'NFT']
    }
    
    for group_name, group_keywords in keyword_groups.items():
        # Check if any of the group keywords are in user's keywords
        matching_keywords = [kw for kw in keywords if any(gk.lower() in kw.lower() or kw.lower() in gk.lower() for gk in group_keywords)]
        if matching_keywords:
            swimlanes.append({
                'id': group_name.lower().replace(' & ', '_').replace(' ', '_'),
                'title': group_name,
                'icon': '🤖' if 'AI' in group_name else '🔐' if 'security' in group_name else '🏢' if 'Companies' in group_name else '☁️' if 'Cloud' in group_name else '🔗',
                'keywords': matching_keywords,
                'type': 'keyword',
                'editable': True
            })
    
    # Add a custom watchlist
    swimlanes.append({
        'id': 'watchlist',
        'title': 'Your Watchlist',
        'icon': '⭐',
        'keywords': keywords[:5] if len(keywords) > 5 else keywords,  # First 5 keywords
        'type': 'custom',
        'editable': True
    })
    
    return swimlanes

@app.route('/')
def index():
    """Main page with Netflix-style interface"""
    return render_template('index_netflix.html')

@app.route('/api/articles')
def get_articles():
    """Get articles with optional filtering"""
    conn = get_db_connection()
    
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    keywords = request.args.getlist('keywords')
    swimlane_type = request.args.get('type', 'all')
    
    # Base query - fixed to match actual schema
    query = '''
        SELECT DISTINCT a.*, 
               GROUP_CONCAT(DISTINCT ka.keyword) as matched_keywords
        FROM articles a
        LEFT JOIN keyword_alerts ka ON a.id = ka.article_id
    '''
    
    conditions = []
    params = []
    
    # Filter by keywords if provided
    if keywords and swimlane_type == 'keyword':
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append("(a.title LIKE ? OR a.content_summary LIKE ? OR a.full_content LIKE ?)")
            params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
        conditions.append(f"({' OR '.join(keyword_conditions)})")
    
    # Add WHERE clause if conditions exist
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    # Group by article
    query += " GROUP BY a.id"
    
    # Order by date - using correct column name
    query += " ORDER BY a.publication_date DESC"
    
    # Add limit and offset
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor = conn.execute(query, params)
    articles = []
    
    for row in cursor:
        article = dict(row)
        # Parse matched keywords
        if article['matched_keywords']:
            article['keywords'] = article['matched_keywords'].split(',')
        else:
            article['keywords'] = []
        
        # Format date - using correct column name
        if article['publication_date']:
            try:
                # Try parsing the date
                pub_date = datetime.fromisoformat(article['publication_date'].replace('Z', '+00:00'))
                # Calculate relative time
                now = datetime.now()
                diff = now - pub_date.replace(tzinfo=None)
                
                if diff.days == 0:
                    if diff.seconds < 3600:
                        article['relative_time'] = f"{diff.seconds // 60} min ago"
                    else:
                        article['relative_time'] = f"{diff.seconds // 3600} hours ago"
                elif diff.days == 1:
                    article['relative_time'] = "Yesterday"
                else:
                    article['relative_time'] = f"{diff.days} days ago"
            except:
                article['relative_time'] = article['publication_date']
        else:
            article['relative_time'] = "Unknown"
        
        # Map content fields
        article['content'] = article.get('full_content') or article.get('content_summary', '')
        article['summary'] = article.get('content_summary', '')
        article['source'] = article.get('source_website', 'Unknown')
        
        articles.append(article)
    
    conn.close()
    return jsonify(articles)

@app.route('/api/swimlanes')
def get_swimlanes_api():
    """Get swimlane configuration"""
    swimlanes = get_swimlanes()
    return jsonify(swimlanes)

@app.route('/api/sources')
def get_sources():
    """Get all configured sources"""
    sites_config = load_sites()
    sources = []
    
    for site in sites_config.get('sites', []):
        # Determine type based on site configuration
        site_type = site.get('type', 'html')
        
        # Auto-detect RSS feeds
        if 'rss' in site.get('name', '').lower():
            site_type = 'rss'
        elif site.get('url', '').endswith(('.xml', '/feed/', '/rss', '/feed')):
            site_type = 'rss'
        elif 'feed' in site.get('url', '').lower():
            site_type = 'rss'
        
        sources.append({
            'name': site.get('name', 'Unnamed Source'),
            'url': site.get('url', ''),
            'type': site_type,
            'enabled': site.get('enabled', True)
        })
    
    return jsonify(sources)

@app.route('/api/sources', methods=['POST'])
def add_source():
    """Add a new source"""
    data = request.json
    sites_config = load_sites()
    
    new_site = {
        'name': data.get('name'),
        'url': data.get('url'),
        'type': data.get('type', 'html'),
        'enabled': True
    }
    
    # Add additional fields for HTML scraping
    if data.get('type') == 'html':
        new_site['article_selector'] = data.get('article_selector', 'article')
        new_site['title_selector'] = data.get('title_selector', 'h1')
        new_site['content_selector'] = data.get('content_selector', 'p')
    
    sites_config['sites'].append(new_site)
    
    # Save updated configuration
    try:
        with open(SITES_FILE, 'w') as f:
            json.dump(sites_config, f, indent=2)
        return jsonify({'success': True, 'message': 'Source added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/sources/<int:index>', methods=['PUT'])
def toggle_source(index):
    """Toggle source enabled/disabled"""
    sites_config = load_sites()
    
    if 0 <= index < len(sites_config['sites']):
        sites_config['sites'][index]['enabled'] = not sites_config['sites'][index].get('enabled', True)
        
        try:
            with open(SITES_FILE, 'w') as f:
                json.dump(sites_config, f, indent=2)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    return jsonify({'success': False, 'message': 'Invalid source index'}), 400

@app.route('/api/stats')
def get_stats():
    """Get statistics for the header"""
    conn = get_db_connection()
    
    # Total articles
    total_articles = conn.execute('SELECT COUNT(*) as count FROM articles').fetchone()['count']
    
    # Today's articles
    today = datetime.now().strftime('%Y-%m-%d')
    todays_articles = conn.execute(
        'SELECT COUNT(*) as count FROM articles WHERE DATE(scraped_at) = ?',
        (today,)
    ).fetchone()['count']
    
    # Active sources
    sites_config = load_sites()
    active_sources = sum(1 for site in sites_config.get('sites', []) if site.get('enabled', True))
    
    # Keyword alerts
    keyword_alerts = conn.execute('SELECT COUNT(*) as count FROM keyword_alerts').fetchone()['count']
    
    conn.close()
    
    return jsonify({
        'total_articles': total_articles,
        'todays_articles': todays_articles,
        'active_sources': active_sources,
        'keyword_alerts': keyword_alerts
    })

@app.route('/api/hero')
def get_hero_article():
    """Get the most recent trending article for hero section"""
    conn = get_db_connection()
    
    # Get the most recent article with keyword matches (trending) - fixed schema
    query = '''
        SELECT a.*, COUNT(ka.id) as keyword_count,
               GROUP_CONCAT(DISTINCT ka.keyword) as matched_keywords
        FROM articles a
        LEFT JOIN keyword_alerts ka ON a.id = ka.article_id
        WHERE a.publication_date > datetime('now', '-24 hours')
        GROUP BY a.id
        ORDER BY keyword_count DESC, a.publication_date DESC
        LIMIT 1
    '''
    
    cursor = conn.execute(query)
    row = cursor.fetchone()
    
    if row:
        article = dict(row)
        if article['matched_keywords']:
            article['keywords'] = article['matched_keywords'].split(',')
        else:
            article['keywords'] = []
        
        # Format date
        if article['publication_date']:
            try:
                pub_date = datetime.fromisoformat(article['publication_date'].replace('Z', '+00:00'))
                now = datetime.now()
                diff = now - pub_date.replace(tzinfo=None)
                
                if diff.seconds < 3600:
                    article['relative_time'] = f"{diff.seconds // 60} min ago"
                else:
                    article['relative_time'] = f"{diff.seconds // 3600} hours ago"
            except:
                article['relative_time'] = article['publication_date']
        
        # Map content fields
        article['content'] = article.get('full_content') or article.get('content_summary', '')
        article['summary'] = article.get('content_summary', '')
        article['source'] = article.get('source_website', 'Unknown')
    else:
        # Fallback to most recent article
        article = conn.execute(
            'SELECT * FROM articles ORDER BY publication_date DESC LIMIT 1'
        ).fetchone()
        if article:
            article = dict(article)
            article['keywords'] = []
            article['relative_time'] = "Recently"
            # Map content fields
            article['content'] = article.get('full_content') or article.get('content_summary', '')
            article['summary'] = article.get('content_summary', '')
            article['source'] = article.get('source_website', 'Unknown')
    
    conn.close()
    return jsonify(article)

@app.route('/api/sources/api-config', methods=['GET', 'POST'])
def api_config():
    """Get or update API sources configuration"""
    api_config_file = 'api_sources_config.json'
    
    if request.method == 'GET':
        try:
            if os.path.exists(api_config_file):
                with open(api_config_file, 'r') as f:
                    config = json.load(f)
                return jsonify({'success': True, 'config': config})
            else:
                return jsonify({'success': False, 'error': 'Configuration file not found'})
        except Exception as e:
            logger.error(f"Error reading API config: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            
            # Validate configuration
            if not isinstance(new_config, dict):
                return jsonify({'success': False, 'error': 'Invalid configuration format'}), 400
            
            # Save configuration
            with open(api_config_file, 'w') as f:
                json.dump(new_config, f, indent=2)
            
            return jsonify({'success': True, 'message': 'Configuration saved successfully'})
        except Exception as e:
            logger.error(f"Error saving API config: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/sources/api-status', methods=['GET'])
def api_status():
    """Get API sources status and availability"""
    try:
        # Check which APIs are available
        status = {
            'newsapi': {
                'available': False,
                'configured': False,
                'enabled': False
            },
            'reddit': {
                'available': False,
                'configured': False,
                'enabled': False
            }
        }
        
        # Check if modules are installed
        try:
            import newsapi
            status['newsapi']['available'] = True
        except ImportError:
            pass
        
        try:
            import praw
            status['reddit']['available'] = True
        except ImportError:
            pass
        
        # Check configuration
        if os.path.exists('api_sources_config.json'):
            with open('api_sources_config.json', 'r') as f:
                config = json.load(f)
                
                # NewsAPI
                newsapi_config = config.get('newsapi', {})
                if newsapi_config.get('api_key') and newsapi_config.get('api_key') != 'YOUR_NEWSAPI_KEY_HERE':
                    status['newsapi']['configured'] = True
                status['newsapi']['enabled'] = newsapi_config.get('enabled', False)
                
                # Reddit
                reddit_config = config.get('reddit', {})
                if reddit_config.get('client_id') != 'YOUR_CLIENT_ID':
                    status['reddit']['configured'] = True
                status['reddit']['enabled'] = reddit_config.get('enabled', False)
        
        return jsonify({'success': True, 'status': status})
    except Exception as e:
        logger.error(f"Error checking API status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scraper/run', methods=['POST'])
def run_scraper():
    """Run the scraper on demand"""
    import subprocess
    import threading
    
    def run_scraper_async():
        try:
            # Run the scraper in a subprocess
            result = subprocess.run(
                [sys.executable, 'scraper.py', '--run'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Log the result
            if result.returncode == 0:
                logger.info("Scraper completed successfully")
                logger.info(result.stdout)
            else:
                logger.error(f"Scraper failed with code {result.returncode}")
                logger.error(result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error("Scraper timed out after 5 minutes")
        except Exception as e:
            logger.error(f"Error running scraper: {e}")
    
    # Start scraper in background thread
    thread = threading.Thread(target=run_scraper_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Scraper started in background. Check back in a few moments for new articles.'
    })

@app.route('/api/scraper/status')
def scraper_status():
    """Get the last scraper run status"""
    conn = get_db_connection()
    
    try:
        # Get the most recent scrape run
        last_run = conn.execute('''
            SELECT * FROM scrape_runs 
            ORDER BY started_at DESC 
            LIMIT 1
        ''').fetchone()
        
        if last_run:
            last_run_dict = dict(last_run)
            # Check if scraper is currently running (last run started but not ended)
            if last_run_dict['ended_at'] is None:
                last_run_dict['status'] = 'running'
            else:
                last_run_dict['status'] = 'completed'
            
            return jsonify({
                'last_run': last_run_dict,
                'has_runs': True
            })
        else:
            return jsonify({
                'last_run': None,
                'has_runs': False,
                'message': 'No scraper runs found. Click "Refresh News" to start.'
            })
            
    except Exception as e:
        logger.error(f"Error getting scraper status: {e}")
        return jsonify({
            'error': str(e),
            'has_runs': False
        })
    finally:
        conn.close()

@app.route('/api/search')
def search_articles():
    """Search articles"""
    query = request.args.get('q', '')
    limit = request.args.get('limit', 20, type=int)
    
    if not query:
        return jsonify([])
    
    conn = get_db_connection()
    
    search_query = '''
        SELECT DISTINCT a.*, 
               GROUP_CONCAT(DISTINCT ka.keyword) as matched_keywords
        FROM articles a
        LEFT JOIN keyword_alerts ka ON a.id = ka.article_id
        WHERE a.title LIKE ? OR a.content_summary LIKE ? OR a.full_content LIKE ?
        GROUP BY a.id
        ORDER BY a.publication_date DESC
        LIMIT ?
    '''
    
    search_term = f'%{query}%'
    cursor = conn.execute(search_query, (search_term, search_term, search_term, limit))
    
    articles = []
    for row in cursor:
        article = dict(row)
        if article['matched_keywords']:
            article['keywords'] = article['matched_keywords'].split(',')
        else:
            article['keywords'] = []
        # Map content fields
        article['content'] = article.get('full_content') or article.get('content_summary', '')
        article['summary'] = article.get('content_summary', '')
        article['source'] = article.get('source_website', 'Unknown')
        articles.append(article)
    
    conn.close()
    return jsonify(articles)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create the Netflix-style template
netflix_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir - News Intelligence Platform</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #141414;
            color: #fff;
            overflow-x: hidden;
        }
        
        /* Header */
        header {
            position: fixed;
            top: 0;
            width: 100%;
            background: linear-gradient(to bottom, rgba(0,0,0,0.7) 10%, rgba(0,0,0,0));
            padding: 1rem 4%;
            z-index: 1000;
            transition: background 0.3s;
        }
        
        header.scrolled {
            background: #141414;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: bold;
            color: #e50914;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        nav {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        nav a {
            color: #e5e5e5;
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.3s;
        }
        
        nav a:hover {
            color: #b3b3b3;
        }
        
        nav a.active {
            color: #fff;
            font-weight: 500;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }
        
        .search-icon, .settings-icon {
            cursor: pointer;
            font-size: 1.2rem;
            transition: transform 0.2s;
        }
        
        .search-icon:hover, .settings-icon:hover {
            transform: scale(1.2);
        }
        
        /* Hero Section */
        .hero {
            height: 70vh;
            background: linear-gradient(to top, #141414 0%, transparent 50%), 
                        linear-gradient(to right, rgba(20,20,20,0.8) 40%, transparent 70%);
            background-size: cover;
            background-position: center;
            display: flex;
            align-items: center;
            padding: 0 4%;
            margin-top: 4rem;
            position: relative;
        }
        
        .hero-gradient {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            opacity: 0.3;
            z-index: -1;
        }
        
        .hero-content {
            max-width: 600px;
            z-index: 1;
        }
        
        .hero-title {
            font-size: 3rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        
        .hero-meta {
            display: flex;
            gap: 2rem;
            margin-bottom: 1.5rem;
            font-size: 1.1rem;
            color: #e5e5e5;
        }
        
        .hero-badge {
            background: #e50914;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .hero-description {
            font-size: 1.2rem;
            line-height: 1.5;
            margin-bottom: 2rem;
            color: #e5e5e5;
        }
        
        .hero-buttons {
            display: flex;
            gap: 1rem;
        }
        
        .btn {
            padding: 0.8rem 2rem;
            border: none;
            border-radius: 4px;
            font-size: 1.1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            color: white;
        }
        
        .btn-primary {
            background: #e50914;
        }
        
        .btn-primary:hover {
            background: #f40612;
        }
        
        .btn-secondary {
            background: rgba(109, 109, 110, 0.7);
        }
        
        .btn-secondary:hover {
            background: rgba(109, 109, 110, 0.4);
        }
        
        /* Swimlanes */
        .swimlanes-container {
            padding: 0 4%;
            margin-top: -5rem;
            position: relative;
            z-index: 100;
        }
        
        .swimlane {
            margin-bottom: 3rem;
            position: relative;
        }
        
        .swimlane-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .swimlane-title {
            font-size: 1.4rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .swimlane-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .swimlane-edit {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s;
        }
        
        .swimlane-edit:hover {
            background: rgba(255,255,255,0.2);
        }
        
        /* Article Cards */
        .articles-row {
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            scroll-behavior: smooth;
            padding: 0.5rem 0;
            scrollbar-width: none;
            position: relative;
        }
        
        .articles-row::-webkit-scrollbar {
            display: none;
        }
        
        .article-card {
            flex: 0 0 300px;
            background: #181818;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
        }
        
        .article-card:hover {
            transform: scale(1.05);
            z-index: 10;
        }
        
        .article-card:hover .article-info {
            opacity: 1;
        }
        
        .article-thumbnail {
            width: 100%;
            height: 170px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            position: relative;
            overflow: hidden;
        }
        
        .article-source {
            position: absolute;
            top: 0.5rem;
            left: 0.5rem;
            background: rgba(0,0,0,0.8);
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }
        
        .article-time {
            position: absolute;
            bottom: 0.5rem;
            right: 0.5rem;
            background: rgba(0,0,0,0.8);
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        
        .article-content {
            padding: 1rem;
        }
        
        .article-title {
            font-size: 0.95rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        
        .article-tags {
            display: flex;
            gap: 0.3rem;
            flex-wrap: wrap;
        }
        
        .article-tag {
            background: #e50914;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-size: 0.7rem;
            font-weight: 500;
        }
        
        .article-info {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.9);
            padding: 1rem;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        /* Scroll buttons */
        .scroll-button {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(0,0,0,0.7);
            border: none;
            color: white;
            width: 50px;
            height: 100px;
            cursor: pointer;
            font-size: 2rem;
            transition: all 0.3s;
            z-index: 100;
            display: none;
        }
        
        .swimlane:hover .scroll-button {
            display: block;
        }
        
        .scroll-button:hover {
            background: rgba(0,0,0,0.9);
        }
        
        .scroll-left {
            left: -25px;
        }
        
        .scroll-right {
            right: -25px;
        }
        
        /* Settings Modal */
        .settings-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 2000;
            overflow-y: auto;
        }
        
        .settings-content {
            background: #1a1a1a;
            margin: 5% auto;
            padding: 2rem;
            max-width: 800px;
            border-radius: 8px;
            position: relative;
        }
        
        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        .settings-title {
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .close-btn {
            background: none;
            border: none;
            color: white;
            font-size: 2rem;
            cursor: pointer;
        }
        
        .settings-section {
            margin-bottom: 2rem;
        }
        
        .settings-section-title {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #e5e5e5;
        }
        
        .source-item {
            background: #2a2a2a;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .source-info {
            flex: 1;
        }
        
        .source-name {
            font-weight: 500;
            margin-bottom: 0.3rem;
        }
        
        .source-url {
            font-size: 0.85rem;
            color: #999;
        }
        
        .source-toggle {
            background: #4CAF50;
            border: none;
            width: 50px;
            height: 26px;
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .source-toggle.off {
            background: #666;
        }
        
        .source-toggle::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 26px;
            transition: left 0.3s;
        }
        
        .source-toggle.off::after {
            left: 2px;
        }
        
        .add-source-form {
            background: #2a2a2a;
            padding: 1.5rem;
            border-radius: 6px;
            margin-top: 1rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            color: #e5e5e5;
        }
        
        .form-input {
            width: 100%;
            padding: 0.7rem;
            background: #1a1a1a;
            border: 1px solid #444;
            border-radius: 4px;
            color: white;
            font-size: 0.95rem;
        }
        
        .form-input:focus {
            outline: none;
            border-color: #e50914;
        }
        
        .source-type-buttons {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .type-btn {
            padding: 0.5rem 1rem;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #e5e5e5;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .type-btn.active {
            background: #e50914;
            border-color: #e50914;
            color: white;
        }
        
        /* Stats display */
        .stats-container {
            position: absolute;
            bottom: 2rem;
            right: 4%;
            display: flex;
            gap: 2rem;
            background: rgba(0,0,0,0.7);
            padding: 1rem 2rem;
            border-radius: 8px;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 1.5rem;
            font-weight: bold;
            color: #e50914;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #999;
        }
        
        /* Loading spinner */
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
            color: #999;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #e50914;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #999;
        }
        
        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Article Reader Styles */
        .article-reader {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(20, 20, 20, 0.98);
            backdrop-filter: blur(10px);
            z-index: 100;
            overflow-y: auto;
            padding: 2rem 4%;
        }
        
        .reader-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        .reader-close {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            font-size: 1.5rem;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .reader-close:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: rotate(90deg);
        }
        
        .reader-meta {
            display: flex;
            gap: 1.5rem;
            color: #999;
            font-size: 0.9rem;
        }
        
        .reader-content {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.05);
            padding: 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        
        .reader-title {
            font-size: 2.5rem;
            margin-bottom: 2rem;
            line-height: 1.3;
        }
        
        .reader-body {
            font-size: 1.1rem;
            line-height: 1.8;
            color: #e5e5e5;
            margin-bottom: 2rem;
        }
        
        .reader-body p {
            margin-bottom: 1.2rem;
        }
        
        .reader-keywords {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }
        
        .reader-keywords span {
            background: rgba(229, 9, 20, 0.2);
            border: 1px solid #e50914;
            padding: 0.3rem 0.8rem;
            border-radius: 16px;
            font-size: 0.85rem;
            color: #ff6b6b;
        }
        
        .reader-footer {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 1.5rem;
            margin-top: 2rem;
        }
        
        .reader-actions {
            display: flex;
            gap: 1rem;
            justify-content: center;
        }
        
        /* Floating action button */
        .fab {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 60px;
            height: 60px;
            background: #e50914;
            border-radius: 50%;
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: all 0.3s;
            z-index: 1000;
        }
        
        .fab:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        
        /* Search overlay */
        .search-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.95);
            z-index: 3000;
            padding: 5rem 4%;
        }
        
        .search-container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .search-input {
            width: 100%;
            padding: 1.5rem;
            font-size: 2rem;
            background: transparent;
            border: none;
            border-bottom: 2px solid #e50914;
            color: white;
            outline: none;
        }
        
        .search-results {
            margin-top: 2rem;
            max-height: 70vh;
            overflow-y: auto;
        }
        
        .search-result {
            padding: 1rem;
            border-bottom: 1px solid #333;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .search-result:hover {
            background: #222;
        }
        
        /* Notification animation */
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2rem;
            }
            
            .article-card {
                flex: 0 0 250px;
            }
            
            .stats-container {
                display: none;
            }
        }
    </style>
</head>
<body>
    <header id="header">
        <div class="header-content">
            <a href="/" class="logo">
                <span>📰</span>
                <span>MIMIR</span>
            </a>
            
            <nav>
                <a href="#" class="active">Home</a>
                <a href="#">Trending</a>
                <a href="#">Categories</a>
                <a href="#">Analytics</a>
            </nav>
            
            <div class="header-right">
                <span class="search-icon" onclick="openSearch()">🔍</span>
                <span class="settings-icon" onclick="openSettings()">⚙️</span>
            </div>
        </div>
    </header>

    <!-- Hero Section with Article Reader -->
    <section class="hero" id="hero">
        <div class="hero-gradient"></div>
        
        <!-- Article Reader View (Initially Hidden) -->
        <div class="article-reader" id="articleReader" style="display: none;">
            <div class="reader-header">
                <button class="reader-close" onclick="closeReader()">✕</button>
                <div class="reader-meta">
                    <span class="reader-source" id="readerSource">Source</span>
                    <span class="reader-date" id="readerDate">Date</span>
                    <span class="reader-time" id="readerTime">5 min read</span>
                </div>
            </div>
            <div class="reader-content">
                <h1 class="reader-title" id="readerTitle">Article Title</h1>
                <div class="reader-body" id="readerBody">
                    <!-- Article content will be loaded here -->
                </div>
                <div class="reader-footer">
                    <div class="reader-keywords" id="readerKeywords">
                        <!-- Keywords will be shown here -->
                    </div>
                    <div class="reader-actions">
                        <a href="#" class="btn btn-secondary" id="readerSourceLink" target="_blank">
                            View Original
                        </a>
                        <button class="btn btn-secondary" onclick="shareArticle()">
                            Share
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Hero Content -->
        <div class="hero-content" id="heroContent">
            <h1 class="hero-title" id="heroTitle">Loading featured article...</h1>
            <div class="hero-meta" id="heroMeta">
                <span class="hero-badge">TRENDING</span>
                <span id="heroSource">Source</span>
                <span id="heroTime">Time</span>
            </div>
            <p class="hero-description" id="heroDescription">
                Loading article summary...
            </p>
            <div class="hero-buttons">
                <button class="btn btn-primary" onclick="openReader(currentHeroArticle)">
                    <span>📖</span>
                    <span>Read Here</span>
                </button>
                <a href="#" class="btn btn-secondary" id="heroReadBtn" target="_blank">
                    <span>🔗</span>
                    <span>View Original</span>
                </a>
                <button class="btn btn-secondary" onclick="refreshHero()">
                    <span>🔄</span>
                    <span>Next Article</span>
                </button>
            </div>
        </div>
        
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-number" id="statTotal">-</div>
                <div class="stat-label">Total Articles</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="statToday">-</div>
                <div class="stat-label">Today</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" id="statSources">-</div>
                <div class="stat-label">Sources</div>
            </div>
        </div>
    </section>

    <!-- Swimlanes Container -->
    <div class="swimlanes-container" id="swimlanesContainer">
        <!-- Swimlanes will be dynamically loaded here -->
    </div>

    <!-- Settings Modal -->
    <div id="settingsModal" class="settings-modal">
        <div class="settings-content">
            <div class="settings-header">
                <h2 class="settings-title">⚙️ Settings</h2>
                <button class="close-btn" onclick="closeSettings()">&times;</button>
            </div>

            <!-- Scraper Control Section -->
            <div class="settings-section">
                <h3 class="settings-section-title">🔄 Scraper Control</h3>
                <div style="background: #2a2a2a; padding: 1.5rem; border-radius: 6px;">
                    <div id="scraperStatus" style="margin-bottom: 1rem;">
                        <p style="color: #999;">Loading scraper status...</p>
                    </div>
                    <button class="btn btn-primary" onclick="runScraper()" id="runScraperBtn">
                        🚀 Refresh News Now
                    </button>
                    <p style="margin-top: 1rem; color: #999; font-size: 0.85rem;">
                        This will fetch new articles from all enabled sources. It may take a few minutes.
                    </p>
                </div>
            </div>

            <!-- News Sources Section -->
            <div class="settings-section">
                <h3 class="settings-section-title">📰 News Sources</h3>
                <div id="sourcesList">
                    <!-- Sources will be loaded here -->
                </div>

                <!-- Add New Source -->
                <div class="add-source-form">
                    <h4 style="margin-bottom: 1rem;">Add New Source</h4>
                    <div class="source-type-buttons">
                        <button class="type-btn active" onclick="setSourceType('rss')">RSS Feed</button>
                        <button class="type-btn" onclick="setSourceType('html')">Website</button>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Source Name</label>
                        <input type="text" class="form-input" id="sourceName" placeholder="e.g., Ars Technica">
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label" id="sourceUrlLabel">Feed URL</label>
                        <input type="text" class="form-input" id="sourceUrl" placeholder="https://example.com/rss">
                    </div>
                    
                    <button class="btn btn-primary" style="width: 100%; margin-top: 1rem;" onclick="addSource()">
                        Add Source
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Search Overlay -->
    <div class="search-overlay" id="searchOverlay">
        <div class="search-container">
            <input type="text" class="search-input" id="searchInput" placeholder="Search articles..." autofocus>
            <div class="search-results" id="searchResults"></div>
        </div>
    </div>

    <!-- Floating Action Buttons -->
    <button class="fab" title="Refresh View" onclick="refreshAll()">🔄</button>
    <button class="fab" style="bottom: 6rem; background: #10b981;" title="Run Scraper" onclick="quickRunScraper()">🚀</button>

    <script>
        // Global variables
        let currentSourceType = 'rss';
        let swimlanes = [];
        let articles = {};
        let currentHeroArticle = null;

        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadHeroArticle();
            loadSwimlanes();
            loadSources();
            
            // Set up auto-refresh
            setInterval(loadStats, 60000); // Update stats every minute
            
            // Header scroll effect
            window.addEventListener('scroll', () => {
                const header = document.getElementById('header');
                if (window.scrollY > 50) {
                    header.classList.add('scrolled');
                } else {
                    header.classList.remove('scrolled');
                }
            });
            
            // Search functionality
            document.getElementById('searchInput').addEventListener('input', debounce(performSearch, 300));
            
            // Close search on escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    closeSearch();
                }
            });
        });

        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                
                document.getElementById('statTotal').textContent = stats.total_articles.toLocaleString();
                document.getElementById('statToday').textContent = stats.todays_articles;
                document.getElementById('statSources').textContent = stats.active_sources;
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        // Load hero article
        async function loadHeroArticle() {
            try {
                const response = await fetch('/api/hero');
                const article = await response.json();
                
                if (article) {
                    currentHeroArticle = article;
                    document.getElementById('heroTitle').textContent = article.title || 'No featured article';
                    document.getElementById('heroSource').textContent = article.source || 'Unknown';
                    document.getElementById('heroTime').textContent = article.relative_time || 'Recently';
                    document.getElementById('heroDescription').textContent = 
                        article.summary || article.content_summary || article.content?.substring(0, 200) + '...' || 'No description available';
                    
                    const readBtn = document.getElementById('heroReadBtn');
                    if (article.url) {
                        readBtn.href = article.url;
                        readBtn.target = '_blank';
                    }
                }
            } catch (error) {
                console.error('Error loading hero article:', error);
            }
        }

        // Load swimlanes
        async function loadSwimlanes() {
            try {
                const response = await fetch('/api/swimlanes');
                swimlanes = await response.json();
                
                const container = document.getElementById('swimlanesContainer');
                container.innerHTML = '';
                
                for (const swimlane of swimlanes) {
                    const swimlaneEl = createSwimlane(swimlane);
                    container.appendChild(swimlaneEl);
                    loadSwimlaneArticles(swimlane);
                }
            } catch (error) {
                console.error('Error loading swimlanes:', error);
            }
        }

        // Create swimlane element
        function createSwimlane(swimlane) {
            const div = document.createElement('div');
            div.className = 'swimlane';
            div.id = `swimlane-${swimlane.id}`;
            
            div.innerHTML = `
                <div class="swimlane-header">
                    <h2 class="swimlane-title">
                        <span>${swimlane.icon}</span>
                        <span>${swimlane.title}</span>
                    </h2>
                    ${swimlane.editable ? `
                    <div class="swimlane-controls">
                        <button class="swimlane-edit" onclick="editSwimlane('${swimlane.id}')">Edit</button>
                        <button class="swimlane-edit" onclick="hideSwimlane('${swimlane.id}')">Hide</button>
                    </div>
                    ` : ''}
                </div>
                <button class="scroll-button scroll-left" onclick="scrollSwimlane('${swimlane.id}', -300)">‹</button>
                <button class="scroll-button scroll-right" onclick="scrollSwimlane('${swimlane.id}', 300)">›</button>
                <div class="articles-row" id="articles-${swimlane.id}">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Loading articles...</p>
                    </div>
                </div>
            `;
            
            return div;
        }

        // Load articles for a swimlane
        async function loadSwimlaneArticles(swimlane) {
            try {
                const params = new URLSearchParams({
                    type: swimlane.type,
                    limit: 20
                });
                
                if (swimlane.keywords && swimlane.keywords.length > 0) {
                    swimlane.keywords.forEach(keyword => {
                        params.append('keywords', keyword);
                    });
                }
                
                const response = await fetch(`/api/articles?${params}`);
                const articlesData = await response.json();
                
                const articlesRow = document.getElementById(`articles-${swimlane.id}`);
                articlesRow.innerHTML = '';
                
                if (articlesData.length === 0) {
                    articlesRow.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <p>No articles found</p>
                        </div>
                    `;
                    return;
                }
                
                articlesData.forEach(article => {
                    const card = createArticleCard(article);
                    articlesRow.appendChild(card);
                });
                
                // Store articles for later use
                articles[swimlane.id] = articlesData;
            } catch (error) {
                console.error(`Error loading articles for ${swimlane.id}:`, error);
                const articlesRow = document.getElementById(`articles-${swimlane.id}`);
                articlesRow.innerHTML = '<div class="empty-state">Error loading articles</div>';
            }
        }

        // Create article card
        function createArticleCard(article) {
            const card = document.createElement('article');
            card.className = 'article-card';
            
            // Generate a color based on the source
            const colors = [
                'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)'
            ];
            const colorIndex = article.source.charCodeAt(0) % colors.length;
            
            // Choose an icon based on keywords or category
            let icon = '📰';
            const title = article.title.toLowerCase();
            if (title.includes('ai') || title.includes('artificial')) icon = '🤖';
            else if (title.includes('security') || title.includes('breach')) icon = '🔐';
            else if (title.includes('apple') || title.includes('iphone')) icon = '🍎';
            else if (title.includes('google') || title.includes('android')) icon = '🔍';
            else if (title.includes('microsoft') || title.includes('windows')) icon = '🪟';
            else if (title.includes('tesla') || title.includes('electric')) icon = '🚗';
            else if (title.includes('space') || title.includes('nasa')) icon = '🚀';
            else if (title.includes('health') || title.includes('medical')) icon = '🏥';
            else if (title.includes('finance') || title.includes('money')) icon = '💰';
            
            card.innerHTML = `
                <div class="article-thumbnail" style="background: ${colors[colorIndex]};">
                    <span>${icon}</span>
                    <span class="article-source">${article.source}</span>
                    <span class="article-time">${article.relative_time}</span>
                </div>
                <div class="article-content">
                    <h3 class="article-title">${article.title}</h3>
                    <div class="article-tags">
                        ${article.keywords.slice(0, 3).map(keyword => 
                            `<span class="article-tag">${keyword}</span>`
                        ).join('')}
                    </div>
                </div>
                <div class="article-info">
                    <p>${article.summary || article.content_summary || article.content?.substring(0, 150) + '...' || 'No summary available'}</p>
                </div>
            `;
            
            card.addEventListener('click', () => {
                if (article.url) {
                    window.open(article.url, '_blank');
                }
            });
            
            return card;
        }

        // Scroll swimlane
        function scrollSwimlane(swimlaneId, amount) {
            const articlesRow = document.getElementById(`articles-${swimlaneId}`);
            articlesRow.scrollBy({ left: amount, behavior: 'smooth' });
        }

        // Load sources
        async function loadSources() {
            try {
                const response = await fetch('/api/sources');
                const sources = await response.json();
                
                const sourcesList = document.getElementById('sourcesList');
                sourcesList.innerHTML = '';
                
                sources.forEach((source, index) => {
                    const sourceEl = document.createElement('div');
                    sourceEl.className = 'source-item';
                    sourceEl.innerHTML = `
                        <div class="source-info">
                            <div class="source-name">${source.name} <span style="background: ${source.type === 'rss' ? '#10b981' : '#3b82f6'}; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-size: 0.7rem; margin-left: 0.5rem;">${source.type.toUpperCase()}</span></div>
                            <div class="source-url">${source.url}</div>
                        </div>
                        <button class="source-toggle ${source.enabled ? '' : 'off'}" 
                                onclick="toggleSource(${index}, this)"></button>
                    `;
                    sourcesList.appendChild(sourceEl);
                });
            } catch (error) {
                console.error('Error loading sources:', error);
            }
        }

        // Toggle source
        async function toggleSource(index, button) {
            try {
                const response = await fetch(`/api/sources/${index}`, { method: 'PUT' });
                if (response.ok) {
                    button.classList.toggle('off');
                    // Reload stats to reflect change
                    loadStats();
                }
            } catch (error) {
                console.error('Error toggling source:', error);
            }
        }

        // Set source type
        function setSourceType(type) {
            currentSourceType = type;
            document.querySelectorAll('.type-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            const urlLabel = document.getElementById('sourceUrlLabel');
            const urlInput = document.getElementById('sourceUrl');
            
            if (type === 'rss') {
                urlLabel.textContent = 'Feed URL';
                urlInput.placeholder = 'https://example.com/rss';
            } else {
                urlLabel.textContent = 'Website URL';
                urlInput.placeholder = 'https://example.com';
            }
        }

        // Add source
        async function addSource() {
            const name = document.getElementById('sourceName').value;
            const url = document.getElementById('sourceUrl').value;
            
            if (!name || !url) {
                alert('Please fill in all fields');
                return;
            }
            
            try {
                const response = await fetch('/api/sources', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        url: url,
                        type: currentSourceType
                    })
                });
                
                if (response.ok) {
                    // Clear form
                    document.getElementById('sourceName').value = '';
                    document.getElementById('sourceUrl').value = '';
                    
                    // Reload sources
                    loadSources();
                    loadStats();
                    
                    alert('Source added successfully!');
                } else {
                    const error = await response.json();
                    alert('Error adding source: ' + error.message);
                }
            } catch (error) {
                console.error('Error adding source:', error);
                alert('Error adding source');
            }
        }

        // Scraper functions
        async function loadScraperStatus() {
            try {
                const response = await fetch('/api/scraper/status');
                const data = await response.json();
                
                const statusDiv = document.getElementById('scraperStatus');
                
                if (data.last_run) {
                    const lastRun = data.last_run;
                    const status = lastRun.status === 'running' ? '🔄 Running...' : '✅ Completed';
                    const articlesFound = lastRun.articles_found || 0;
                    const newArticles = lastRun.articles_new || 0;
                    
                    statusDiv.innerHTML = `
                        <p><strong>Last Run:</strong> ${lastRun.started_at || 'Unknown'}</p>
                        <p><strong>Status:</strong> ${status}</p>
                        <p><strong>Articles:</strong> ${articlesFound} found, ${newArticles} new</p>
                    `;
                    
                    // Disable button if scraper is running
                    const btn = document.getElementById('runScraperBtn');
                    if (lastRun.status === 'running') {
                        btn.disabled = true;
                        btn.textContent = '⏳ Scraper Running...';
                    } else {
                        btn.disabled = false;
                        btn.textContent = '🚀 Refresh News Now';
                    }
                } else {
                    statusDiv.innerHTML = '<p style="color: #fbbf24;">No scraper runs yet. Click below to fetch news!</p>';
                }
            } catch (error) {
                console.error('Error loading scraper status:', error);
                document.getElementById('scraperStatus').innerHTML = '<p style="color: #ef4444;">Error loading status</p>';
            }
        }

        async function runScraper() {
            const btn = document.getElementById('runScraperBtn');
            const originalText = btn.textContent;
            
            btn.disabled = true;
            btn.textContent = '⏳ Starting scraper...';
            
            try {
                const response = await fetch('/api/scraper/run', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    btn.textContent = '✅ Scraper started!';
                    
                    // Show notification
                    const notification = document.createElement('div');
                    notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #10b981; color: white; padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 5000; animation: slideIn 0.3s ease-out;';
                    notification.textContent = 'Scraper started! New articles will appear shortly.';
                    document.body.appendChild(notification);
                    
                    setTimeout(() => {
                        document.body.removeChild(notification);
                    }, 5000);
                    
                    // Start polling for status
                    let pollCount = 0;
                    const pollInterval = setInterval(async () => {
                        await loadScraperStatus();
                        pollCount++;
                        
                        // Stop polling after 2 minutes or when scraper is done
                        if (pollCount > 24) {
                            clearInterval(pollInterval);
                        }
                        
                        // Check if scraper finished
                        const statusResponse = await fetch('/api/scraper/status');
                        const statusData = await statusResponse.json();
                        if (statusData.last_run && statusData.last_run.status !== 'running') {
                            clearInterval(pollInterval);
                            // Refresh articles
                            loadStats();
                            loadHeroArticle();
                            loadSwimlanes();
                        }
                    }, 5000); // Poll every 5 seconds
                    
                } else {
                    btn.textContent = '❌ Failed to start';
                    setTimeout(() => {
                        btn.disabled = false;
                        btn.textContent = originalText;
                    }, 2000);
                }
            } catch (error) {
                console.error('Error running scraper:', error);
                btn.textContent = '❌ Error';
                setTimeout(() => {
                    btn.disabled = false;
                    btn.textContent = originalText;
                }, 2000);
            }
        }

        // Settings modal functions
        function openSettings() {
            document.getElementById('settingsModal').style.display = 'block';
            document.body.style.overflow = 'hidden';
            loadScraperStatus(); // Load scraper status when opening settings
        }

        function closeSettings() {
            document.getElementById('settingsModal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }

        // Search functions
        function openSearch() {
            document.getElementById('searchOverlay').style.display = 'block';
            document.getElementById('searchInput').focus();
            document.body.style.overflow = 'hidden';
        }

        function closeSearch() {
            document.getElementById('searchOverlay').style.display = 'none';
            document.getElementById('searchInput').value = '';
            document.getElementById('searchResults').innerHTML = '';
            document.body.style.overflow = 'auto';
        }

        async function performSearch() {
            const query = document.getElementById('searchInput').value;
            if (!query) {
                document.getElementById('searchResults').innerHTML = '';
                return;
            }
            
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const results = await response.json();
                
                const resultsContainer = document.getElementById('searchResults');
                resultsContainer.innerHTML = '';
                
                if (results.length === 0) {
                    resultsContainer.innerHTML = '<div class="empty-state">No results found</div>';
                    return;
                }
                
                results.forEach(article => {
                    const resultEl = document.createElement('div');
                    resultEl.className = 'search-result';
                    resultEl.innerHTML = `
                        <h3>${article.title}</h3>
                        <p style="color: #999; font-size: 0.9rem;">${article.source || article.source_website} • ${article.relative_time || article.publication_date}</p>
                        <p style="margin-top: 0.5rem;">${article.summary || article.content_summary || article.content?.substring(0, 150) + '...' || ''}</p>
                    `;
                    resultEl.addEventListener('click', () => {
                        if (article.url) {
                            window.open(article.url, '_blank');
                        }
                    });
                    resultsContainer.appendChild(resultEl);
                });
            } catch (error) {
                console.error('Error searching:', error);
            }
        }

        // Refresh functions
        function refreshHero() {
            loadHeroArticle();
        }

        function refreshAll() {
            loadStats();
            loadHeroArticle();
            loadSwimlanes();
        }

        // Article Reader Functions
        function openReader(article) {
            if (!article) return;
            
            // Populate reader content
            document.getElementById('readerTitle').textContent = article.title || 'Untitled';
            document.getElementById('readerSource').textContent = article.source || 'Unknown Source';
            document.getElementById('readerDate').textContent = article.publication_date ? 
                new Date(article.publication_date).toLocaleDateString() : 'Unknown Date';
            
            // Calculate reading time (rough estimate: 200 words per minute)
            const content = article.full_content || article.content || article.content_summary || '';
            const wordCount = content.split(' ').length;
            const readingTime = Math.max(1, Math.ceil(wordCount / 200));
            document.getElementById('readerTime').textContent = `${readingTime} min read`;
            
            // Set article body
            const readerBody = document.getElementById('readerBody');
            if (content) {
                // Convert plain text to paragraphs if needed
                const paragraphs = content.split('\n\n').filter(p => p.trim());
                readerBody.innerHTML = paragraphs.map(p => `<p>${p}</p>`).join('');
            } else {
                readerBody.innerHTML = '<p><em>No content available. Click "View Original" to read on the source website.</em></p>';
            }
            
            // Set keywords
            const keywordsDiv = document.getElementById('readerKeywords');
            if (article.keywords && article.keywords.length > 0) {
                keywordsDiv.innerHTML = article.keywords.map(kw => 
                    `<span>${kw.keyword || kw}</span>`
                ).join('');
            } else {
                keywordsDiv.innerHTML = '';
            }
            
            // Set original link
            const sourceLink = document.getElementById('readerSourceLink');
            if (article.url) {
                sourceLink.href = article.url;
                sourceLink.style.display = 'inline-flex';
            } else {
                sourceLink.style.display = 'none';
            }
            
            // Show reader, hide hero content
            document.getElementById('articleReader').style.display = 'block';
            document.getElementById('heroContent').style.display = 'none';
            document.getElementById('hero').style.minHeight = 'auto';
        }
        
        function closeReader() {
            document.getElementById('articleReader').style.display = 'none';
            document.getElementById('heroContent').style.display = 'block';
            document.getElementById('hero').style.minHeight = '85vh';
        }
        
        function shareArticle() {
            const article = currentHeroArticle;
            if (!article) return;
            
            if (navigator.share) {
                navigator.share({
                    title: article.title,
                    text: article.summary || article.content_summary,
                    url: article.url
                }).catch(err => console.log('Error sharing:', err));
            } else {
                // Fallback: copy URL to clipboard
                navigator.clipboard.writeText(article.url || window.location.href)
                    .then(() => {
                        alert('Link copied to clipboard!');
                    })
                    .catch(err => console.error('Could not copy text:', err));
            }
        }
        
        // Quick scraper function (no settings modal)
        async function quickRunScraper() {
            const notification = document.createElement('div');
            notification.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #3b82f6; color: white; padding: 1rem 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 5000; animation: slideIn 0.3s ease-out;';
            notification.textContent = '⏳ Starting news refresh...';
            document.body.appendChild(notification);
            
            try {
                const response = await fetch('/api/scraper/run', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    notification.style.background = '#10b981';
                    notification.textContent = '✅ Scraper started! New articles coming...';
                    
                    // Auto-refresh after 10 seconds
                    setTimeout(() => {
                        refreshAll();
                    }, 10000);
                } else {
                    notification.style.background = '#ef4444';
                    notification.textContent = '❌ Failed to start scraper';
                }
            } catch (error) {
                notification.style.background = '#ef4444';
                notification.textContent = '❌ Error starting scraper';
            }
            
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 5000);
        }

        // Utility functions
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        // Close modals on click outside
        window.onclick = function(event) {
            if (event.target.className === 'settings-modal') {
                closeSettings();
            }
            if (event.target.className === 'search-overlay') {
                closeSearch();
            }
        }

        // Placeholder functions for future features
        function editSwimlane(id) {
            alert('Edit swimlane feature coming soon! You will be able to customize keywords for: ' + id);
        }

        function hideSwimlane(id) {
            const swimlane = document.getElementById(`swimlane-${id}`);
            if (swimlane) {
                swimlane.style.display = 'none';
            }
        }
    </script>
</body>
</html>'''

# Write the template
with open('templates/index_netflix.html', 'w', encoding='utf-8') as f:
    f.write(netflix_template)

if __name__ == '__main__':
    print("Starting Mimir Netflix-Style Web Interface...")
    print("Access the interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)