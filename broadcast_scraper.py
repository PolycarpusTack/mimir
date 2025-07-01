#!/usr/bin/env python3
"""
Broadcast Industry Scraper for Mimir
Specialized scraping for broadcast/media industry sources
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import feedparser

import db_manager
from scraper import fetch_url, check_robots_txt, extract_keywords

logger = logging.getLogger(__name__)

class BroadcastScraper:
    """Specialized scraper for broadcast industry sources"""
    
    def __init__(self, config_file='broadcast_sources.json'):
        """Initialize with broadcast sources configuration"""
        self.config = self._load_config(config_file)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.get('settings', {}).get('user_agent', 
                         'Mimir Broadcast Intelligence Bot (+https://mimir.ai/bot)')
        })
        
    def _load_config(self, config_file):
        """Load broadcast sources configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Broadcast config file {config_file} not found")
            return {"sources": [], "settings": {}}
    
    def _extract_press_release(self, html: str, selectors: Dict) -> Optional[Dict]:
        """Extract press release content using selectors"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            title = soup.select_one(selectors.get('title', 'h1'))
            title_text = title.get_text(strip=True) if title else None
            
            date = soup.select_one(selectors.get('date', 'time'))
            date_text = date.get_text(strip=True) if date else datetime.now().isoformat()
            
            content = soup.select_one(selectors.get('content', 'div.content'))
            content_text = content.get_text(strip=True) if content else ""
            
            # Extract executive quotes if present
            quotes = []
            quote_elements = soup.find_all('blockquote') or soup.find_all('p', class_=re.compile('quote'))
            for quote in quote_elements[:3]:  # Max 3 quotes
                quotes.append(quote.get_text(strip=True))
            
            return {
                'title': title_text,
                'date': date_text,
                'content': content_text,
                'quotes': quotes
            }
        except Exception as e:
            logger.error(f"Error extracting press release: {e}")
            return None
    
    def _smart_content_extraction(self, url: str) -> Optional[Dict]:
        """Smart extraction using common patterns when selectors fail"""
        try:
            html = fetch_url(url)
            if not html:
                return None
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to find JSON-LD structured data
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    data = json.loads(json_ld.string)
                    if data.get('@type') in ['NewsArticle', 'Article', 'PressRelease']:
                        return {
                            'title': data.get('headline'),
                            'content': data.get('articleBody', data.get('description')),
                            'date': data.get('datePublished'),
                            'author': data.get('author', {}).get('name')
                        }
                except:
                    pass
            
            # Fallback to meta tags
            title = soup.find('meta', property='og:title') or soup.find('title')
            title_text = title.get('content', '') if title and hasattr(title, 'get') else title.get_text()
            
            description = soup.find('meta', property='og:description')
            content = description.get('content', '') if description else ""
            
            # Try to find article body
            article_body = soup.find('article') or soup.find('div', class_=re.compile('content|body|article'))
            if article_body:
                content = article_body.get_text(strip=True)
            
            return {
                'title': title_text,
                'content': content,
                'date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Smart extraction failed for {url}: {e}")
            return None
    
    def _is_broadcast_relevant(self, text: str, title: str = "") -> bool:
        """Check if content is relevant to broadcast industry"""
        keywords = self.config.get('filters', {}).get('keywords', {})
        must_have = keywords.get('must_have_one', [])
        exclude = keywords.get('exclude', [])
        boost = keywords.get('boost', [])
        
        full_text = f"{title} {text}".lower()
        
        # Check exclusions first
        for exc in exclude:
            if exc.lower() in full_text:
                return False
        
        # Check must-have keywords
        has_required = any(kw.lower() in full_text for kw in must_have)
        if not has_required:
            return False
        
        # Boost score for important keywords
        boost_score = sum(1 for kw in boost if kw.lower() in full_text)
        
        return boost_score >= 0  # Can adjust threshold
    
    def scrape_broadcast_sources(self, limit_per_source: Optional[int] = None) -> Dict[str, int]:
        """Scrape all configured broadcast sources"""
        results = {
            'broadcasters': 0,
            'standards': 0,
            'trade_publications': 0,
            'technology_vendors': 0,
            'total': 0,
            'errors': 0
        }
        
        settings = self.config.get('settings', {})
        default_limit = limit_per_source or settings.get('article_limit_per_source', 10)
        
        for source in self.config.get('sources', []):
            if not source.get('enabled', False):
                continue
                
            category = source.get('category', 'unknown')
            
            try:
                logger.info(f"Scraping {source['name']}...")
                
                if source['type'] == 'rss':
                    articles = self._scrape_rss_source(source, default_limit)
                elif source['type'] == 'html':
                    articles = self._scrape_html_source(source, default_limit)
                else:
                    logger.warning(f"Unknown source type: {source['type']}")
                    continue
                
                # Save articles to database
                for article in articles:
                    if self._is_broadcast_relevant(article.get('content', ''), article.get('title', '')):
                        try:
                            db_manager.add_article(
                                url=article['url'],
                                title=article['title'],
                                content_summary=article.get('summary', article['content'][:500]),
                                full_content=article.get('content', ''),
                                publication_date=article.get('date', datetime.now().isoformat()),
                                source=f"{source['name']} ({category})",
                                author=article.get('author', ''),
                                category=f"broadcast_{category}"
                            )
                            results[category] += 1
                            results['total'] += 1
                        except Exception as e:
                            logger.error(f"Error saving article: {e}")
                            
            except Exception as e:
                logger.error(f"Error scraping {source['name']}: {e}")
                results['errors'] += 1
        
        logger.info(f"Broadcast scraping complete: {results['total']} relevant articles found")
        return results
    
    def _scrape_rss_source(self, source: Dict, limit: int) -> List[Dict]:
        """Scrape RSS feed source"""
        articles = []
        rss_url = source.get('rss_url', source.get('url'))
        
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                article = {
                    'url': entry.get('link'),
                    'title': entry.get('title'),
                    'content': entry.get('summary', entry.get('description', '')),
                    'date': entry.get('published', entry.get('updated', '')),
                    'author': entry.get('author', '')
                }
                
                # Apply category filters if specified
                if 'filters' in source:
                    categories = source['filters'].get('categories', [])
                    entry_categories = [tag.term for tag in entry.get('tags', [])]
                    if categories and not any(cat in entry_categories for cat in categories):
                        continue
                
                articles.append(article)
                
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            
        return articles
    
    def _scrape_html_source(self, source: Dict, limit: int) -> List[Dict]:
        """Scrape HTML source (press releases, etc.)"""
        articles = []
        
        # First, get the article list page
        html = fetch_url(source['url'])
        if not html:
            return articles
            
        soup = BeautifulSoup(html, 'html.parser')
        selectors = source.get('selectors', {})
        
        # Find article links
        article_links = []
        if 'article_list' in selectors and 'article_link' in selectors:
            containers = soup.select(selectors['article_list'])
            for container in containers[:limit]:
                link = container.select_one(selectors['article_link'])
                if link and link.get('href'):
                    article_links.append(link.get('href'))
        
        # Scrape individual articles
        for link in article_links[:limit]:
            full_url = link if link.startswith('http') else f"{source['url'].rstrip('/')}/{link.lstrip('/')}"
            
            # Check robots.txt
            if not check_robots_txt(full_url):
                continue
                
            # Try smart extraction
            article_data = self._smart_content_extraction(full_url)
            if article_data:
                article_data['url'] = full_url
                articles.append(article_data)
        
        return articles

# Integration function for main scraper
def scrape_broadcast_sources(limit_per_source: Optional[int] = None):
    """Function to be called from main scraper"""
    try:
        scraper = BroadcastScraper()
        return scraper.scrape_broadcast_sources(limit_per_source)
    except Exception as e:
        logger.error(f"Broadcast scraping failed: {e}")
        return {'total': 0, 'errors': 1}

if __name__ == "__main__":
    # Test the broadcast scraper
    logging.basicConfig(level=logging.INFO)
    results = scrape_broadcast_sources(limit_per_source=5)
    print(f"Broadcast Scraping Results: {results}")