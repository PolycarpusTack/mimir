#!/usr/bin/env python3
"""
Mimir News Scraper - Netflix-Style Web Interface (FIXED)
This version fixes all JavaScript template literal issues
"""

from flask import Flask, render_template_string, jsonify, request, redirect, url_for
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

# API Routes
@app.route('/api/stats')
def get_stats():
    """Get statistics about the scraped articles"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total articles
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    
    # Today's articles
    today = datetime.now().date()
    cursor.execute("SELECT COUNT(*) FROM articles WHERE DATE(scraped_at) = ?", (today,))
    todays_articles = cursor.fetchone()[0]
    
    # Active sources
    cursor.execute("SELECT COUNT(DISTINCT source_website) FROM articles WHERE scraped_at > datetime('now', '-7 day')")
    active_sources = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_articles': total_articles,
        'todays_articles': todays_articles,
        'active_sources': active_sources
    })

@app.route('/api/sources')
def get_sources():
    """Get list of configured sources"""
    sites_data = load_sites()
    sources = sites_data.get('sites', [])
    
    # Add status information
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for source in sources:
        # Check last successful scrape
        cursor.execute("""
            SELECT MAX(scraped_at) as last_scrape, COUNT(*) as article_count 
            FROM articles 
            WHERE source_website = ?
        """, (source['name'],))
        
        result = cursor.fetchone()
        source['last_scrape'] = result['last_scrape'] if result else None
        source['article_count'] = result['article_count'] if result else 0
        source['status'] = 'active' if source.get('enabled', True) else 'disabled'
    
    conn.close()
    
    return jsonify(sources)

# Fixed HTML Template with properly formatted JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir News Intelligence</title>
    <style>
        /* CSS styles here - keeping it simple for now */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: #141414; color: #fff; }
        header { position: fixed; top: 0; width: 100%; background: rgba(0,0,0,0.7); padding: 1rem 4%; z-index: 1000; }
        .header-content { display: flex; justify-content: space-between; align-items: center; }
        .logo { font-size: 1.8rem; font-weight: bold; color: #e50914; text-decoration: none; }
        .hero { position: relative; min-height: 85vh; display: flex; align-items: center; padding: 0 4%; margin-top: 60px; }
        .hero-gradient { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(to bottom, rgba(0,0,0,0) 0%, rgba(20,20,20,1) 100%); }
        .hero-content { position: relative; z-index: 2; max-width: 800px; }
        .hero-title { font-size: 3rem; margin-bottom: 1rem; }
        .btn { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1.5rem; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; text-decoration: none; transition: all 0.3s; }
        .btn-primary { background: #e50914; color: white; }
        .btn-primary:hover { background: #f40612; }
        .swimlanes-container { padding: 2rem 4%; }
        .swimlane { margin-bottom: 3rem; }
        .swimlane-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .swimlane-title { font-size: 1.4rem; font-weight: 500; }
        .articles-row { display: flex; gap: 0.5rem; overflow-x: auto; padding: 0.5rem 0; }
        .article-card { flex: 0 0 300px; background: #181818; border-radius: 6px; overflow: hidden; cursor: pointer; transition: all 0.3s; }
        .article-card:hover { transform: scale(1.05); }
        .article-thumb { width: 100%; height: 160px; background: #2a2a2a; display: flex; align-items: center; justify-content: center; font-size: 3rem; }
        .article-info { padding: 1rem; }
        .article-title { font-size: 1rem; margin-bottom: 0.5rem; line-height: 1.3; }
        .article-meta { font-size: 0.8rem; color: #999; }
        .fab { position: fixed; bottom: 2rem; right: 2rem; width: 60px; height: 60px; background: #e50914; border-radius: 50%; border: none; color: white; font-size: 1.5rem; cursor: pointer; transition: all 0.3s; }
        .fab:hover { transform: scale(1.1); }
        
        /* Article Reader Styles */
        .article-reader { position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(20, 20, 20, 0.98); z-index: 100; overflow-y: auto; padding: 2rem 4%; display: none; }
        .reader-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
        .reader-close { background: rgba(255, 255, 255, 0.1); border: none; color: white; font-size: 1.5rem; width: 40px; height: 40px; border-radius: 50%; cursor: pointer; }
        .reader-content { max-width: 800px; margin: 0 auto; background: rgba(255, 255, 255, 0.05); padding: 3rem; border-radius: 12px; }
        .reader-title { font-size: 2.5rem; margin-bottom: 2rem; }
        .reader-body { font-size: 1.1rem; line-height: 1.8; color: #e5e5e5; }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <a href="/" class="logo">📺 Mimir News</a>
            <nav>
                <button onclick="openSettings()" style="background: none; border: none; color: white; cursor: pointer; font-size: 1.2rem;">⚙️</button>
            </nav>
        </div>
    </header>

    <section class="hero" id="hero">
        <div class="hero-gradient"></div>
        
        <!-- Article Reader -->
        <div class="article-reader" id="articleReader">
            <div class="reader-header">
                <button class="reader-close" onclick="closeReader()">✕</button>
            </div>
            <div class="reader-content">
                <h1 class="reader-title" id="readerTitle">Article Title</h1>
                <div class="reader-body" id="readerBody"></div>
            </div>
        </div>
        
        <div class="hero-content" id="heroContent">
            <h1 class="hero-title" id="heroTitle">Loading...</h1>
            <p id="heroDescription">Loading article summary...</p>
            <div style="margin-top: 2rem;">
                <button class="btn btn-primary" onclick="openReader()">📖 Read Here</button>
                <a href="#" class="btn btn-primary" id="heroLink" target="_blank">🔗 View Original</a>
            </div>
        </div>
    </section>

    <div class="swimlanes-container" id="swimlanesContainer">
        <!-- Swimlanes will be loaded here -->
    </div>

    <button class="fab" onclick="refreshAll()">🔄</button>
    <button class="fab" style="bottom: 6rem; background: #10b981;" onclick="runScraper()">🚀</button>

    <script>
        // Global variables
        let currentHeroArticle = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadHeroArticle();
            loadSwimlanes();
        });
        
        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                console.log('Stats loaded:', stats);
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
                    document.getElementById('heroDescription').textContent = article.summary || article.content_summary || 'No description available';
                    
                    const heroLink = document.getElementById('heroLink');
                    if (article.url) {
                        heroLink.href = article.url;
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
                const swimlanes = await response.json();
                
                const container = document.getElementById('swimlanesContainer');
                container.innerHTML = '';
                
                for (const swimlane of swimlanes) {
                    const div = document.createElement('div');
                    div.className = 'swimlane';
                    div.innerHTML = '<div class="swimlane-header"><h2 class="swimlane-title">' + swimlane.icon + ' ' + swimlane.title + '</h2></div><div class="articles-row" id="articles-' + swimlane.id + '">Loading...</div>';
                    container.appendChild(div);
                    
                    // Load articles for this swimlane
                    loadSwimlaneArticles(swimlane);
                }
            } catch (error) {
                console.error('Error loading swimlanes:', error);
            }
        }
        
        // Load articles for a swimlane
        async function loadSwimlaneArticles(swimlane) {
            try {
                const params = new URLSearchParams({
                    type: swimlane.type,
                    keywords: swimlane.keywords.join(','),
                    limit: 20
                });
                
                const response = await fetch('/api/articles?' + params);
                const articles = await response.json();
                
                const articlesRow = document.getElementById('articles-' + swimlane.id);
                if (articles.length === 0) {
                    articlesRow.innerHTML = '<p style="color: #999;">No articles found</p>';
                    return;
                }
                
                articlesRow.innerHTML = articles.map(article => {
                    return '<div class="article-card" onclick="window.open(\'' + article.url + '\', \'_blank\')">' +
                           '<div class="article-thumb">📰</div>' +
                           '<div class="article-info">' +
                           '<h3 class="article-title">' + (article.title || 'Untitled') + '</h3>' +
                           '<div class="article-meta">' + (article.source || 'Unknown') + ' • ' + (article.relative_time || 'Recently') + '</div>' +
                           '</div></div>';
                }).join('');
            } catch (error) {
                console.error('Error loading articles:', error);
            }
        }
        
        // Article reader functions
        function openReader() {
            if (!currentHeroArticle) return;
            
            document.getElementById('readerTitle').textContent = currentHeroArticle.title || 'Untitled';
            document.getElementById('readerBody').innerHTML = '<p>' + (currentHeroArticle.content || currentHeroArticle.content_summary || 'No content available') + '</p>';
            
            document.getElementById('articleReader').style.display = 'block';
            document.getElementById('heroContent').style.display = 'none';
        }
        
        function closeReader() {
            document.getElementById('articleReader').style.display = 'none';
            document.getElementById('heroContent').style.display = 'block';
        }
        
        // Refresh all
        function refreshAll() {
            loadStats();
            loadHeroArticle();
            loadSwimlanes();
        }
        
        // Run scraper
        async function runScraper() {
            if (confirm('Run the scraper now?')) {
                try {
                    const response = await fetch('/api/scraper/run', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.success) {
                        alert('Scraper started! New articles will appear shortly.');
                        setTimeout(refreshAll, 10000);
                    } else {
                        alert('Failed to start scraper: ' + result.error);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
        }
        
        // Settings placeholder
        function openSettings() {
            alert('Settings panel coming soon!');
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/hero')
def get_hero_article():
    """Get the featured article for the hero section"""
    conn = get_db_connection()
    
    # Try to get a recent article with keywords
    article = conn.execute('''
        SELECT DISTINCT a.* 
        FROM articles a
        LEFT JOIN keyword_alerts ka ON a.id = ka.article_id
        WHERE a.publication_date > datetime('now', '-24 hours')
        ORDER BY 
            CASE WHEN ka.keyword IS NOT NULL THEN 1 ELSE 0 END DESC,
            a.publication_date DESC
        LIMIT 1
    ''').fetchone()
    
    if article:
        article = dict(article)
        # Add relative time
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
            article['relative_time'] = "Recently"
            article['content'] = article.get('full_content') or article.get('content_summary', '')
            article['summary'] = article.get('content_summary', '')
            article['source'] = article.get('source_website', 'Unknown')
    
    conn.close()
    return jsonify(article)

@app.route('/api/swimlanes')
def get_swimlanes():
    """Get swimlane configuration"""
    # For now, return static swimlanes
    config = load_config()
    keywords = config.get('keywords_monitoring', {}).get('keywords', [])
    
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
    
    # Add keyword-based swimlanes
    if keywords:
        swimlanes.append({
            'id': 'monitored',
            'title': 'Monitored Keywords',
            'icon': '🎯',
            'keywords': keywords,
            'type': 'keywords',
            'editable': True
        })
    
    return jsonify(swimlanes)

@app.route('/api/articles')
def get_articles():
    """Get articles based on filters"""
    article_type = request.args.get('type', 'latest')
    keywords = request.args.get('keywords', '').split(',')
    limit = int(request.args.get('limit', 20))
    
    conn = get_db_connection()
    
    if article_type == 'latest':
        articles = conn.execute(
            'SELECT * FROM articles ORDER BY publication_date DESC LIMIT ?',
            (limit,)
        ).fetchall()
    elif article_type == 'keywords' and keywords:
        # Build query for keyword matching
        keyword_conditions = ' OR '.join(['title LIKE ? OR content_summary LIKE ?' for _ in keywords])
        params = []
        for kw in keywords:
            if kw.strip():
                params.extend([f'%{kw.strip()}%', f'%{kw.strip()}%'])
        
        if params:
            query = f'SELECT * FROM articles WHERE {keyword_conditions} ORDER BY publication_date DESC LIMIT ?'
            params.append(limit)
            articles = conn.execute(query, params).fetchall()
        else:
            articles = []
    else:
        articles = []
    
    # Convert to list of dicts and add relative time
    articles_list = []
    for article in articles:
        article_dict = dict(article)
        
        # Add relative time
        try:
            pub_date = datetime.fromisoformat(article_dict['publication_date'].replace('Z', '+00:00'))
            now = datetime.now()
            diff = now - pub_date.replace(tzinfo=None)
            
            if diff.days > 0:
                article_dict['relative_time'] = f"{diff.days}d ago"
            elif diff.seconds < 3600:
                article_dict['relative_time'] = f"{diff.seconds // 60}m ago"
            else:
                article_dict['relative_time'] = f"{diff.seconds // 3600}h ago"
        except:
            article_dict['relative_time'] = "Recently"
        
        article_dict['source'] = article_dict.get('source_website', 'Unknown')
        articles_list.append(article_dict)
    
    conn.close()
    return jsonify(articles_list)

@app.route('/api/scraper/run', methods=['POST'])
def run_scraper():
    """Run the scraper on demand"""
    import subprocess
    import threading
    
    def run_scraper_async():
        try:
            result = subprocess.run(
                [sys.executable, 'scraper.py', '--run'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("Scraper completed successfully")
            else:
                logger.error(f"Scraper failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running scraper: {e}")
    
    # Start scraper in background thread
    thread = threading.Thread(target=run_scraper_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'message': 'Scraper started'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)