#!/usr/bin/env python3
"""
API Scraper Module for Mimir
Integrates NewsAPI, Reddit, and other API sources with smart filtering
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import hashlib

# Core imports
import db_manager

# API imports (optional)
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

logger = logging.getLogger(__name__)

class APIScraperModule:
    """Handles API-based news scraping with smart filtering"""
    
    def __init__(self, config_file='api_sources_config.json'):
        """Initialize API scraper with configuration"""
        self.config = self._load_config(config_file)
        self.api_clients = self._initialize_clients()
        self.article_hashes = set()  # For deduplication
        self.daily_count = 0
        self.last_reset = datetime.now().date()
        
    def _load_config(self, config_file):
        """Load API configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"API config file {config_file} not found")
            return {}
    
    def _initialize_clients(self):
        """Initialize API clients based on configuration"""
        clients = {}
        
        # Initialize NewsAPI
        if NEWSAPI_AVAILABLE and self.config.get('newsapi', {}).get('enabled'):
            api_key = self.config['newsapi'].get('api_key')
            if api_key and api_key != 'YOUR_NEWSAPI_KEY_HERE':
                try:
                    clients['newsapi'] = NewsApiClient(api_key=api_key)
                    logger.info("NewsAPI client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize NewsAPI: {e}")
        
        # Initialize Reddit
        if REDDIT_AVAILABLE and self.config.get('reddit', {}).get('enabled'):
            reddit_config = self.config['reddit']
            if reddit_config.get('client_id') != 'YOUR_CLIENT_ID':
                try:
                    clients['reddit'] = praw.Reddit(
                        client_id=reddit_config['client_id'],
                        client_secret=reddit_config['client_secret'],
                        user_agent=reddit_config['user_agent']
                    )
                    logger.info("Reddit client initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Reddit: {e}")
        
        return clients
    
    def _check_daily_limit(self):
        """Check if daily limit is reached"""
        # Reset counter if new day
        if datetime.now().date() > self.last_reset:
            self.daily_count = 0
            self.last_reset = datetime.now().date()
        
        daily_limit = self.config.get('api_settings', {}).get('daily_article_limit', 500)
        return self.daily_count < daily_limit
    
    def _is_relevant(self, text: str, title: str = "") -> bool:
        """Check if article is relevant based on keywords"""
        if not self.config.get('api_settings', {}).get('filter_settings', {}).get('require_keywords', True):
            return True
        
        # Combine text and title for checking
        full_text = f"{title} {text}".lower()
        
        # Check primary keywords
        primary_keywords = self.config.get('keywords', {}).get('primary', [])
        for keyword in primary_keywords:
            if keyword.lower() in full_text:
                return True
        
        # Check secondary keywords (less weight)
        secondary_keywords = self.config.get('keywords', {}).get('secondary', [])
        secondary_matches = sum(1 for kw in secondary_keywords if kw.lower() in full_text)
        if secondary_matches >= 2:  # Need at least 2 secondary matches
            return True
        
        # Check exclude keywords
        exclude_keywords = self.config.get('keywords', {}).get('exclude', [])
        for keyword in exclude_keywords:
            if keyword.lower() in full_text:
                return False
        
        return False
    
    def _deduplicate(self, url: str, title: str) -> bool:
        """Check if article is duplicate using hash"""
        article_hash = hashlib.md5(f"{url}{title}".encode()).hexdigest()
        if article_hash in self.article_hashes:
            return True
        self.article_hashes.add(article_hash)
        return False
    
    def fetch_from_newsapi(self) -> List[Dict]:
        """Fetch articles from NewsAPI with filtering"""
        if 'newsapi' not in self.api_clients:
            return []
        
        articles = []
        newsapi_config = self.config.get('newsapi', {})
        
        try:
            # Get keywords for search
            keywords = self.config.get('keywords', {}).get('primary', [])[:3]  # Top 3 keywords
            
            for keyword in keywords:
                if not self._check_daily_limit():
                    break
                
                # Search with specific sources if configured
                sources = ','.join(newsapi_config.get('sources_filter', [])) if newsapi_config.get('sources_filter') else None
                
                response = self.api_clients['newsapi'].get_everything(
                    q=keyword,
                    sources=sources,
                    language=newsapi_config.get('language', 'en'),
                    sort_by=newsapi_config.get('sort_by', 'relevancy'),
                    page_size=newsapi_config.get('page_size', 20),
                    from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                )
                
                for article in response.get('articles', []):
                    # Apply filters
                    if not self._is_relevant(article.get('description', ''), article.get('title', '')):
                        continue
                    
                    if self._deduplicate(article['url'], article['title']):
                        continue
                    
                    # Check if already in database
                    if db_manager.article_exists(article['url']):
                        continue
                    
                    # Add to results
                    articles.append({
                        'url': article['url'],
                        'title': article['title'],
                        'content_summary': article.get('description', ''),
                        'full_content': article.get('content', ''),
                        'publication_date': article.get('publishedAt', datetime.now().isoformat()),
                        'source': f"NewsAPI: {article['source']['name']}",
                        'author': article.get('author', ''),
                        'category': 'api_technology'
                    })
                    
                    self.daily_count += 1
                    
                    if len(articles) >= newsapi_config.get('page_size', 20):
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    def fetch_from_reddit(self) -> List[Dict]:
        """Fetch posts from Reddit with filtering"""
        if 'reddit' not in self.api_clients:
            return []
        
        articles = []
        reddit_config = self.config.get('reddit', {})
        
        try:
            subreddits = reddit_config.get('subreddits', ['technology'])
            post_limit = reddit_config.get('post_limit', 10)
            min_score = reddit_config.get('min_score', 10)
            
            for subreddit_name in subreddits:
                if not self._check_daily_limit():
                    break
                
                subreddit = self.api_clients['reddit'].subreddit(subreddit_name)
                
                # Get hot posts
                for submission in subreddit.hot(limit=post_limit):
                    # Filter by score
                    if submission.score < min_score:
                        continue
                    
                    # Skip self posts without URLs
                    if submission.is_self and not submission.url.startswith('http'):
                        continue
                    
                    # Apply keyword filter
                    if not self._is_relevant(submission.selftext or '', submission.title):
                        continue
                    
                    # Deduplicate
                    if self._deduplicate(submission.url, submission.title):
                        continue
                    
                    # Check database
                    if db_manager.article_exists(submission.url):
                        continue
                    
                    # Add to results
                    articles.append({
                        'url': submission.url,
                        'title': submission.title,
                        'content_summary': submission.selftext[:500] if submission.selftext else f"Reddit post with {submission.num_comments} comments",
                        'full_content': submission.selftext or '',
                        'publication_date': datetime.fromtimestamp(submission.created_utc).isoformat(),
                        'source': f"Reddit: r/{subreddit_name}",
                        'author': str(submission.author),
                        'category': 'api_community'
                    })
                    
                    self.daily_count += 1
                    
        except Exception as e:
            logger.error(f"Error fetching from Reddit: {e}")
        
        return articles
    
    def scrape_all_apis(self) -> Dict[str, int]:
        """Scrape all configured APIs and save to database"""
        if not self.config.get('api_settings', {}).get('enabled', True):
            logger.info("API scraping is disabled")
            return {}
        
        results = {
            'newsapi': 0,
            'reddit': 0,
            'total': 0,
            'filtered_out': 0
        }
        
        logger.info("Starting API scraping...")
        
        # Fetch from NewsAPI
        if self.config.get('newsapi', {}).get('enabled'):
            newsapi_articles = self.fetch_from_newsapi()
            for article in newsapi_articles:
                try:
                    db_manager.add_article(
                        url=article['url'],
                        title=article['title'],
                        content_summary=article['content_summary'],
                        full_content=article.get('full_content', ''),
                        publication_date=article['publication_date'],
                        source=article['source'],
                        author=article.get('author', ''),
                        category=article.get('category', 'general')
                    )
                    results['newsapi'] += 1
                except Exception as e:
                    logger.error(f"Error saving NewsAPI article: {e}")
        
        # Fetch from Reddit
        if self.config.get('reddit', {}).get('enabled'):
            reddit_articles = self.fetch_from_reddit()
            for article in reddit_articles:
                try:
                    db_manager.add_article(
                        url=article['url'],
                        title=article['title'],
                        content_summary=article['content_summary'],
                        full_content=article.get('full_content', ''),
                        publication_date=article['publication_date'],
                        source=article['source'],
                        author=article.get('author', ''),
                        category=article.get('category', 'general')
                    )
                    results['reddit'] += 1
                except Exception as e:
                    logger.error(f"Error saving Reddit article: {e}")
        
        results['total'] = results['newsapi'] + results['reddit']
        
        logger.info(f"API scraping complete: {results['total']} new articles "
                   f"(NewsAPI: {results['newsapi']}, Reddit: {results['reddit']})")
        
        return results

# Integration function for main scraper
def scrape_api_sources():
    """Function to be called from main scraper"""
    try:
        scraper = APIScraperModule()
        return scraper.scrape_all_apis()
    except Exception as e:
        logger.error(f"API scraping failed: {e}")
        return {}

if __name__ == "__main__":
    # Test the API scraper
    logging.basicConfig(level=logging.INFO)
    results = scrape_api_sources()
    print(f"API Scraping Results: {results}")