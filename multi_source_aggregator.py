#!/usr/bin/env python3
"""
Multi-Source News Aggregator for Mimir
Combines multiple legal APIs and RSS feeds for comprehensive coverage
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

# For RSS feeds
import feedparser

# For Reddit
try:
    import praw
except ImportError:
    praw = None

# For NewsAPI
try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None

# Standard libraries
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)

class MultiSourceAggregator:
    """Aggregates news from multiple legal sources"""
    
    def __init__(self, config_file='api_config.json'):
        """Initialize with API keys from config"""
        self.config = self._load_config(config_file)
        self.sources = self._initialize_sources()
        
    def _load_config(self, config_file):
        """Load API keys and configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create template config
            template = {
                "newsapi": {
                    "api_key": "YOUR_NEWSAPI_KEY",
                    "enabled": False
                },
                "reddit": {
                    "client_id": "YOUR_CLIENT_ID",
                    "client_secret": "YOUR_CLIENT_SECRET",
                    "user_agent": "Mimir News Aggregator",
                    "enabled": False
                },
                "rss_feeds": [
                    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index"},
                    {"name": "Hacker News", "url": "https://hnrss.org/frontpage"},
                    {"name": "The Hollywood Reporter", "url": "https://www.hollywoodreporter.com/feed"},
                    {"name": "Variety", "url": "https://variety.com/feed/"},
                    {"name": "Radio World", "url": "https://www.radioworld.com/feed"}
                ]
            }
            with open(config_file, 'w') as f:
                json.dump(template, f, indent=2)
            logger.info(f"Created template config at {config_file}")
            return template
    
    def _initialize_sources(self):
        """Initialize available news sources"""
        sources = {}
        
        # Initialize NewsAPI
        if NewsApiClient and self.config.get('newsapi', {}).get('enabled'):
            try:
                sources['newsapi'] = NewsApiClient(api_key=self.config['newsapi']['api_key'])
                logger.info("NewsAPI initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NewsAPI: {e}")
        
        # Initialize Reddit
        if praw and self.config.get('reddit', {}).get('enabled'):
            try:
                sources['reddit'] = praw.Reddit(
                    client_id=self.config['reddit']['client_id'],
                    client_secret=self.config['reddit']['client_secret'],
                    user_agent=self.config['reddit']['user_agent']
                )
                logger.info("Reddit API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit: {e}")
        
        return sources
    
    def fetch_from_newsapi(self, keywords: List[str], days_back: int = 7) -> List[Dict]:
        """Fetch articles from NewsAPI"""
        if 'newsapi' not in self.sources:
            return []
        
        articles = []
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        try:
            # Search for each keyword
            for keyword in keywords:
                response = self.sources['newsapi'].get_everything(
                    q=keyword,
                    from_param=from_date,
                    sort_by='relevancy',
                    language='en',
                    page_size=20
                )
                
                for article in response.get('articles', []):
                    articles.append({
                        'title': article['title'],
                        'url': article['url'],
                        'content': article['description'],
                        'source': article['source']['name'],
                        'published_date': article['publishedAt'],
                        'author': article.get('author', 'Unknown'),
                        'api_source': 'NewsAPI',
                        'keywords': [keyword]
                    })
                    
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            
        return articles
    
    def fetch_from_reddit(self, subreddits: List[str], limit: int = 25) -> List[Dict]:
        """Fetch posts from Reddit"""
        if 'reddit' not in self.sources:
            return []
        
        articles = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.sources['reddit'].subreddit(subreddit_name)
                
                for submission in subreddit.hot(limit=limit):
                    # Skip self posts without external links
                    if not submission.is_self:
                        articles.append({
                            'title': submission.title,
                            'url': submission.url,
                            'content': submission.selftext or f"Score: {submission.score}, Comments: {submission.num_comments}",
                            'source': f"r/{subreddit_name}",
                            'published_date': datetime.fromtimestamp(submission.created_utc).isoformat(),
                            'author': str(submission.author),
                            'api_source': 'Reddit',
                            'keywords': []
                        })
                        
        except Exception as e:
            logger.error(f"Error fetching from Reddit: {e}")
            
        return articles
    
    def fetch_from_rss(self) -> List[Dict]:
        """Fetch articles from configured RSS feeds"""
        articles = []
        
        for feed in self.config.get('rss_feeds', []):
            try:
                parsed = feedparser.parse(feed['url'])
                
                for entry in parsed.entries[:10]:  # Limit to 10 per feed
                    articles.append({
                        'title': entry.get('title', 'No title'),
                        'url': entry.get('link', ''),
                        'content': entry.get('summary', entry.get('description', '')),
                        'source': feed['name'],
                        'published_date': entry.get('published', datetime.now().isoformat()),
                        'author': entry.get('author', 'Unknown'),
                        'api_source': 'RSS',
                        'keywords': []
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed['name']}: {e}")
                
        return articles
    
    def fetch_from_arxiv(self, search_terms: List[str], max_results: int = 10) -> List[Dict]:
        """Fetch research papers from arXiv"""
        articles = []
        
        try:
            import arxiv
            
            for term in search_terms:
                search = arxiv.Search(
                    query=term,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for paper in search.results():
                    articles.append({
                        'title': paper.title,
                        'url': paper.pdf_url,
                        'content': paper.summary,
                        'source': 'arXiv',
                        'published_date': paper.published.isoformat(),
                        'author': ', '.join([author.name for author in paper.authors]),
                        'api_source': 'arXiv',
                        'keywords': [term]
                    })
                    
        except ImportError:
            logger.warning("arxiv package not installed")
        except Exception as e:
            logger.error(f"Error fetching from arXiv: {e}")
            
        return articles
    
    def aggregate_all(self, keywords: List[str] = None, 
                     subreddits: List[str] = None,
                     include_research: bool = False) -> List[Dict]:
        """Aggregate news from all configured sources"""
        
        all_articles = []
        
        # Default keywords for broadcasting industry
        if not keywords:
            keywords = ['broadcast technology', 'streaming media', 'OTT platform', 
                       'television industry', 'media technology']
        
        # Default subreddits
        if not subreddits:
            subreddits = ['technology', 'television', 'cordcutters', 'broadcasting']
        
        # Fetch from each source
        logger.info("Fetching from RSS feeds...")
        all_articles.extend(self.fetch_from_rss())
        
        if self.config.get('newsapi', {}).get('enabled'):
            logger.info("Fetching from NewsAPI...")
            all_articles.extend(self.fetch_from_newsapi(keywords))
        
        if self.config.get('reddit', {}).get('enabled'):
            logger.info("Fetching from Reddit...")
            all_articles.extend(self.fetch_from_reddit(subreddits))
        
        if include_research:
            logger.info("Fetching from arXiv...")
            all_articles.extend(self.fetch_from_arxiv(keywords, max_results=5))
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        logger.info(f"Aggregated {len(unique_articles)} unique articles from {len(set(a['api_source'] for a in unique_articles))} sources")
        
        return unique_articles

def main():
    """Example usage"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize aggregator
    aggregator = MultiSourceAggregator()
    
    # Fetch articles
    articles = aggregator.aggregate_all(
        keywords=['broadcast', 'streaming', 'OTT', 'television'],
        subreddits=['technology', 'television', 'cordcutters'],
        include_research=True
    )
    
    # Display results
    print(f"\n📰 Found {len(articles)} articles:\n")
    
    # Group by source
    by_source = {}
    for article in articles:
        source = article['api_source']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(article)
    
    for source, source_articles in by_source.items():
        print(f"\n{source} ({len(source_articles)} articles):")
        for article in source_articles[:3]:  # Show first 3 from each source
            print(f"  - {article['title'][:80]}...")
            print(f"    Source: {article['source']}")
            print(f"    URL: {article['url']}")

if __name__ == "__main__":
    main()