# 📡 Mimir API Sources Guide

## Overview

Mimir now supports multiple API sources for comprehensive news aggregation while respecting robots.txt and staying 100% legal. With smart filtering, you won't be overwhelmed by thousands of articles.

## 🚀 Quick Start

1. **Install API modules** (optional but recommended):
   ```bash
   python install_api_modules.py
   ```

2. **Get FREE API keys**:
   - **NewsAPI**: Sign up at https://newsapi.org/register (100 requests/day free)
   - **Reddit**: Create app at https://www.reddit.com/prefs/apps (choose "script" type)

3. **Configure in Web Interface**:
   - Start Mimir: `python start_netflix_ui.py`
   - Go to Settings > API Sources
   - Enter your API keys

## 🎯 Smart Filtering System

### How It Works

Instead of pulling 100,000+ articles, Mimir uses intelligent filtering:

1. **Keyword Filtering**: Only fetches articles matching your keywords
2. **Source Selection**: Choose specific trusted sources
3. **Daily Limits**: Default 500 articles/day (configurable)
4. **Deduplication**: No duplicate articles
5. **Relevance Scoring**: Prioritizes most relevant content

### Configuration Example

```json
{
  "api_settings": {
    "daily_article_limit": 500,      // Max articles per day
    "articles_per_source": 20,       // Max per API source
    "filter_settings": {
      "require_keywords": true,      // Only fetch keyword matches
      "min_keyword_relevance": 0.3,  // Relevance threshold
      "language": "en"               // Language filter
    }
  }
}
```

## 📰 Available API Sources

### 1. NewsAPI (80,000+ sources)
- **Coverage**: TechCrunch, BBC, CNN, Reuters, Bloomberg, etc.
- **Free Tier**: 100 requests/day
- **What You Get**: ~20-100 articles per request (filtered)

### 2. Reddit API
- **Coverage**: Community discussions, breaking news
- **Free Tier**: 60 requests/minute
- **Subreddits**: r/technology, r/broadcasting, r/television

### 3. RSS Feeds (No API needed)
- **Coverage**: Direct from publishers
- **Free**: Unlimited
- **Sources**: 14+ broadcasting industry feeds included

## 🔧 Keyword Configuration

Edit keywords in `api_sources_config.json`:

```json
{
  "keywords": {
    "primary": [
      "broadcast",
      "streaming",
      "OTT",
      "television"
    ],
    "secondary": [
      "5G broadcast",
      "ATSC 3.0"
    ],
    "exclude": [
      "sports",
      "celebrity"
    ]
  }
}
```

## 📊 Daily Article Estimates

With default settings:
- **RSS Feeds Only**: ~50-100 articles/day
- **RSS + NewsAPI**: ~200-300 articles/day (filtered)
- **All Sources**: ~300-500 articles/day (with limits)

## 🛡️ Benefits Over Web Scraping

1. **100% Legal**: No robots.txt violations
2. **More Reliable**: APIs don't break
3. **Richer Data**: Metadata, dates, authors
4. **Faster**: Pre-structured data
5. **Real-time**: Social signals from Reddit

## ⚙️ Advanced Settings

### Adjusting Article Limits

In `api_sources_config.json`:
```json
{
  "api_settings": {
    "daily_article_limit": 1000,  // Increase daily limit
    "articles_per_source": 50     // More per source
  }
}
```

### Source-Specific Filtering

For NewsAPI:
```json
{
  "newsapi": {
    "sources_filter": ["techcrunch", "wired", "ars-technica"],
    "categories": ["technology", "business"]
  }
}
```

### Time-Based Filtering

Only fetch recent articles:
- NewsAPI: Last 24 hours by default
- Reddit: Hot posts from last day
- RSS: Latest entries

## 🚨 Troubleshooting

1. **"API module not available"**
   - Run: `python install_api_modules.py`

2. **"Invalid API key"**
   - Check key in web interface Settings
   - Ensure no extra spaces

3. **"Too many articles"**
   - Reduce `daily_article_limit`
   - Add more specific keywords
   - Enable `require_keywords: true`

4. **"No articles found"**
   - Check keywords aren't too specific
   - Verify API keys are correct
   - Check sources are enabled

## 📈 Future Enhancements

- Google Trends integration
- arXiv research papers
- Sentiment analysis
- Predictive trend modeling
- Custom ML-based filtering

---

Remember: Quality over quantity! Better to have 500 relevant articles than 10,000 random ones.