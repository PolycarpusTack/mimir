# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mimir is a B2B News Scraper built in Python that:
- Scrapes news from RSS feeds and HTML websites
- Stores articles in SQLite database (PostgreSQL support now available!)
- Sends email notifications with summaries
- Provides a Flask web interface for viewing articles
- Includes keyword monitoring and alerts

## Common Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database (SQLite)
python db_manager.py

# PostgreSQL setup (new)
docker-compose up -d  # Start PostgreSQL, pgBouncer, Redis
python db_manager_postgres.py  # Initialize PostgreSQL
alembic upgrade head  # Apply database migrations

# AI/NLP setup (EPIC 2)
python setup_nlp_models.py  # Download language models

# Analytics & Business Intelligence setup (EPIC 6)
docker-compose -f docker-compose.timescale.yml up -d  # Start TimescaleDB
python analytics_manager.py setup  # Initialize analytics infrastructure
```

### Running the Scraper
```bash
# Run scraper once
python scraper.py --run

# Run with scheduler (every 4 hours)
python scraper.py --schedule

# View statistics
python scraper.py --stats

# Test email configuration
python scraper.py --test-email

# Use PostgreSQL instead of SQLite
export USE_POSTGRES=true
python scraper.py --run
```

### API Sources Setup (NEW!)
```bash
# Install API modules (NewsAPI, Reddit, etc.)
python install_api_modules.py

# Configure API sources (or use web interface)
# Edit api_sources_config.json with your API keys

# API sources are automatically included in regular scraper runs
python scraper.py --run

# Check API source status
python api_scraper.py  # Test API scraping independently
```

### PostgreSQL Migration
```bash
# Migrate from SQLite to PostgreSQL
python migrate_to_postgres.py

# Test PostgreSQL functionality
python test_postgres_migration.py

# Check migration status
python -c "import db_adapter; print(db_adapter.check_migration_status())"
```

### AI Analysis (EPIC 2 - NEW!)
```bash
# Analyze recent articles
python ai_scraper_integration.py recent

# Analyze articles from specific source
python ai_scraper_integration.py source "TechCrunch"

# Analyze all unanalyzed articles
python ai_scraper_integration.py all --limit 100

# Start AI service with workers (for high volume)
python ai_scraper_integration.py service --workers 4
```

### Summarization Engine (EPIC 8 - NEW!)
```bash
# Test summarization functionality
python test_summarization.py

# Command-line summarization interface

# Summarize text directly
python summarization_cli.py text "Your article text here" --format bullet_points

# Summarize article from database  
python summarization_cli.py article <article_id> --format executive --save

# Batch summarize recent articles
python summarization_cli.py batch --limit 10 --hours 24 --format custom --save

# List existing summaries for an article
python summarization_cli.py list <article_id>

# Show summarization statistics
python summarization_cli.py stats

# Available summary formats:
# - custom: Flexible length summary (default)
# - bullet_points: Key points in bullet format
# - executive: Professional executive summary  
# - tweet: Social media ready summary (280 chars)

# Supported models:
# - facebook/bart-large-cnn (default, optimized for news)
# - google/pegasus-xsum (good for abstractive summaries)
# - t5-small (faster, smaller model)
# - t5-base (balanced performance)
```

### Translation Engine (EPIC 8 - NEW!)
```bash
# Command-line translation interface

# Translate text directly
python translation_cli.py text "Hello world" --target es --source en

# Translate article from database
python translation_cli.py article <article_id> --target fr --save

# Batch translate recent articles
python translation_cli.py batch --target de --limit 10 --hours 24 --save

# Detect language of text
python translation_cli.py detect "Bonjour le monde"

# List existing translations for an article
python translation_cli.py list <article_id>

# Show supported languages and models
python translation_cli.py languages

# Show translation statistics
python translation_cli.py stats

# Supported languages (12 languages):
# - en: English    - es: Spanish     - fr: French      - de: German
# - it: Italian    - pt: Portuguese  - nl: Dutch       - ru: Russian
# - zh: Chinese    - ja: Japanese    - ko: Korean      - ar: Arabic

# Available models:
# - facebook/mbart-large-50-many-to-many-mmt (multilingual, all languages)
# - Helsinki-NLP/opus-mt-* (specialized bilateral models)
# - Quality estimation and confidence scoring included
```

### Content Generation (EPIC 8 - NEW!)
```bash
# Command-line content generation interface

# Generate daily newsletter
python content_generation_cli.py newsletter --days 1 --save newsletter.md

# Generate personalized newsletter with interests
python content_generation_cli.py newsletter --days 2 --interests "technology,business,finance" --save tech_news.md

# Generate executive briefing
python content_generation_cli.py briefing --days 7 --level c-level --focus "technology,markets,regulation" --save exec_brief.md

# Generate trend analysis report
python content_generation_cli.py trends --days 14 --categories "tech,finance,healthcare" --save trend_report.md

# Batch generate all content types
python content_generation_cli.py batch newsletter,briefing,trends --days 3 --output-dir reports/

# Show available templates and configurations
python content_generation_cli.py templates

# Show content generation statistics
python content_generation_cli.py stats

# Available content types:
# - newsletter: Personalized news digest with highlights and trending topics
# - executive_briefing: Strategic briefing for executive decision-making  
# - trend_report: Comprehensive analysis of trending topics and patterns
# - competitive_analysis: Market intelligence focused on competitive landscape

# Executive levels supported: senior, c-level, director, vp
# Personalization based on user interests and reading history
# Multi-format output with section breakdown
```

### Advanced NLU (EPIC 8 - NEW!)
```bash
# Advanced Natural Language Understanding capabilities

# Question Answering API endpoints:
# POST /api/v1/advanced-nlu/question-answering
# POST /api/v1/advanced-nlu/batch-qa

# Fact Extraction API endpoints:
# POST /api/v1/advanced-nlu/fact-extraction

# Claim Detection API endpoints:
# POST /api/v1/advanced-nlu/claim-detection

# Article Analysis API endpoint:
# GET /api/v1/advanced-nlu/article/{article_id}/qa-summary

# Supported question types:
# - factual: What is, Who is, When did, Where is, How many
# - analytical: Why did, How does, What caused, What impact
# - temporal: When, How long, Since when, Until when
# - numerical: How much, How many, What percentage, What amount

# Supported fact types:
# - financial: Money amounts, percentages, revenue, profit
# - temporal: Dates, years, quarters, time periods
# - corporate: Mergers, acquisitions, leadership, IPOs
# - regulatory: SEC, FDA, regulations, approvals

# Features:
# - Context-aware question answering with confidence scoring
# - Multi-strategy fact extraction with verification
# - Claim detection and verification
# - Batch processing for multiple questions
# - Article database integration for context retrieval
# - Quality validation and confidence thresholds
```

### EPIC 8 Testing & Optimization (NEW!)
```bash
# Comprehensive test suite for all EPIC 8 components
python epic8_comprehensive_test.py

# Performance optimization for EPIC 8 components
python epic8_optimization.py

# Individual component testing
python test_summarization.py
# (Additional individual test scripts can be created as needed)

# Test results and reports are saved with timestamps
# - epic8_test_report_YYYYMMDD_HHMMSS.json
# - epic8_optimization_report_YYYYMMDD_HHMMSS.json

# Performance benchmarks:
# - Summarization: < 10 seconds per article
# - Translation: < 5 seconds per text
# - Content Generation: < 30 seconds per report  
# - Advanced NLU: < 8 seconds per question

# Memory optimization includes:
# - Model pool management and pre-warming
# - GPU acceleration when available
# - Intelligent caching strategies
# - Memory usage monitoring and cleanup
```

### Analytics & Business Intelligence (EPIC 6 - NEW!)
```bash
# Setup analytics infrastructure (TimescaleDB)
python setup_analytics.py

# Check analytics system status
python analytics_manager.py status

# Collect metrics from main database
python analytics_manager.py collect

# Run trend analysis
python analytics_manager.py analyze --days 30

# Analyze only keyword trends
python analytics_manager.py analyze --keywords-only --days 7

# Analyze only source performance trends  
python analytics_manager.py analyze --sources-only --days 14

# Start continuous metrics collection (every 30 minutes)
python analytics_manager.py continuous --interval 30

# Clean up old analytics data (keep 90 days)
python analytics_manager.py cleanup --days 90

# Generate predictive forecasts
python analytics_manager.py forecast --type all --days 7

# Generate specific forecast types
python analytics_manager.py forecast --type volume --days 14   # Article volume forecasting
python analytics_manager.py forecast --type topics --days 7    # Topic trend prediction  
python analytics_manager.py forecast --type sentiment --days 7 # Sentiment forecasting
python analytics_manager.py forecast --type alerts --days 30   # Adaptive alert thresholds

# Run analytics dashboard (web interface)
python analytics_manager.py dashboard --port 5001
# Access dashboard at http://localhost:5001

# Run dashboard in debug mode
python analytics_manager.py dashboard --port 5001 --debug

# Report Generation & Management
python analytics_manager.py report templates  # List available report templates

# Generate reports on demand
python analytics_manager.py report generate executive_summary --format pdf --time-period-days 30
python analytics_manager.py report generate source_performance --format html --output /path/to/report.html
python analytics_manager.py report generate keyword_trends --format json --time-period-days 14

# Scheduled reports management
python analytics_manager.py report scheduled list  # List all scheduled reports
python analytics_manager.py report scheduled create "Weekly Executive Summary" executive_summary --schedule-type weekly --schedule-time 09:00 --day-of-week monday
python analytics_manager.py report scheduled create "Daily Source Report" source_performance --schedule-type daily --schedule-time 08:30
python analytics_manager.py report scheduled run <report_id>  # Run scheduled report immediately
python analytics_manager.py report scheduled delete <report_id>  # Delete scheduled report

# Report scheduler control
python analytics_manager.py report scheduled start-scheduler  # Start automatic scheduler
python analytics_manager.py report scheduled stop-scheduler   # Stop automatic scheduler

# Data Export
python analytics_manager.py export article_metrics --format csv
python analytics_manager.py export source_metrics --format parquet --compress
python analytics_manager.py export keyword_metrics --format jsonl --stream
python analytics_manager.py export trend_analysis --format excel --output trends.xlsx

# ML Infrastructure Commands
# Model Registry
python analytics_manager.py ml models list  # List all registered models
python analytics_manager.py ml models promote --model-name article_forecast --version 1.0.0 --stage production
python analytics_manager.py ml models experiment  # View MLflow experiments

# Feature Store
python analytics_manager.py ml features list  # List registered features
python analytics_manager.py ml features register --feature-name sentiment_features --description "Article sentiment features"
python analytics_manager.py ml features compute  # Compute and store features

# A/B Testing
python analytics_manager.py ml abtest list  # List active A/B tests
python analytics_manager.py ml abtest evaluate --test-id test_001  # Evaluate test results

# TimescaleDB access
docker-compose -f docker-compose.timescale.yml up -d  # Start TimescaleDB
# Access Adminer at http://localhost:8080 (TimescaleDB on port 5433)
```

### Playwright Integration (NEW!)
```bash
# Setup Playwright for JavaScript-heavy sites
python setup_playwright.py

# Check Playwright status
python scraper_enhanced.py --check-playwright

# Run scraper with Playwright support
python scraper_enhanced.py --run

# Run database migration for Playwright columns
alembic upgrade head

# Test Playwright integration
python test_playwright_integration.py

# Configure sites for Playwright
# Add to sites_to_scrape.json:
# "js_required": true,
# "playwright_settings": {
#     "wait_for_selector": ".article-list",
#     "wait_for_load_state": "networkidle"
# }
```

### Development Tools
```bash
# Quick start interactive menu
python quick_start.py

# Run web interface
python web_interface.py
# Access at http://localhost:5000
```

### Testing
```bash
# Currently no test suite exists
# When implementing tests, use pytest:
pip install pytest pytest-cov
pytest tests/ -v --cov=.
```

### Code Quality
```bash
# Install linting tools (not yet configured)
pip install flake8 black mypy

# Run linters
flake8 . --max-line-length=120
black . --line-length=120
mypy . --ignore-missing-imports
```

## Architecture Overview

### Core Components

1. **scraper.py** - Main scraping engine
   - `NewsScraper` class handles both RSS and HTML scraping
   - Respects robots.txt and implements rate limiting
   - Supports pagination for HTML sites
   - Includes retry logic and error handling
   - **playwright_scraper.py** - JavaScript rendering support (NEW!)
     - Handles React, Vue, Angular sites
     - Automatic fallback to BeautifulSoup
     - Screenshot capture capabilities
     - Performance metrics tracking

2. **Database Management**
   - **db_manager.py** - Original SQLite implementation
   - **db_manager_postgres.py** - New PostgreSQL implementation with:
     - Full-text search with ranking
     - Connection pooling via psycopg2-pool
     - JSONB support for flexible metadata
     - Source management capabilities
   - **db_adapter.py** - Compatibility layer for seamless switching
   - **migrate_to_postgres.py** - Migration script from SQLite to PostgreSQL

3. **web_interface.py** - Flask web dashboard
   - View scraped articles
   - Search functionality
   - Statistics dashboard
   - JSON API endpoints

4. **Configuration Files**
   - `config.json` - Main configuration (email, logging, keywords)
   - `sites_to_scrape.json` - List of news sources with selectors

### Data Flow
1. Scraper reads sites from `sites_to_scrape.json`
2. For each enabled site, fetches content (RSS/HTML)
3. Parses articles using BeautifulSoup/feedparser
4. Stores unique articles in SQLite database
5. Checks for keyword matches
6. Sends email notifications if configured

### Key Design Patterns
- Modular architecture with clear separation of concerns
- Configuration-driven behavior
- Comprehensive error handling and logging
- Database abstraction layer for future PostgreSQL migration

## Development Roadmap

The project has an ambitious roadmap (`DEVELOPMENT_ROADMAP.md`) with 8 EPICs:
1. PostgreSQL Migration & Data Layer Enhancement ✅ COMPLETED
2. AI-Powered Content Analysis Engine ✅ COMPLETED (Core Features)
   - Multi-language NLP (spaCy)
   - Named Entity Recognition
   - Keyword Extraction (TF-IDF, YAKE, TextRank)
   - Sentiment Analysis with transformers
   - Queue-based processing with Redis
3. Semantic Search & Similarity Engine
4. High-Performance Async Architecture
5. Enterprise Features & Multi-tenancy
6. Analytics & Business Intelligence (IN PROGRESS)
   - ✅ TimescaleDB data warehouse
   - ✅ Trend detection & predictive analytics
   - ✅ Dashboard framework with real-time updates
   - ✅ Custom reports & scheduled delivery
   - ✅ Data export in multiple formats
   - ✅ ML infrastructure (MLflow, model registry, A/B testing, feature store)
   - 🔄 Article clustering (next)
   - 🔄 Recommendation engine (next)
7. Scalable Infrastructure & DevOps
8. Advanced ML & AI Features

### Recent Additions (Outside Roadmap)
- ✅ **Playwright Integration** - JavaScript rendering for modern news sites
  - Browser automation for React/Vue/Angular sites
  - Automatic fallback to BeautifulSoup
  - Screenshot capture and performance metrics
  - Database tracking of rendering methods

## Important Implementation Notes

### Error Handling
- All scraping operations wrapped in try-except blocks
- Retry logic with exponential backoff for network failures
- Errors logged to database for debugging
- Graceful degradation when sites are unavailable

### Security Considerations
- Never store API keys or passwords in code
- Use environment variables for sensitive data
- Validate and sanitize all user inputs
- Respect robots.txt for ethical scraping

### Performance
- Connection pooling for HTTP requests
- Rate limiting per domain
- Batch database operations
- Pagination support for large result sets

### Database Schema

SQLite (Original):
```sql
-- Articles table
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    published_date DATETIME,
    source TEXT,
    author TEXT,
    category TEXT,
    scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

PostgreSQL (New):
```sql
-- Enhanced articles table with full-text search
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    published_date TIMESTAMPTZ,
    source VARCHAR(255) NOT NULL,
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    search_vector tsvector GENERATED ALWAYS AS (...) STORED
);

-- Plus: sources, keywords, users tables for advanced features
-- See scripts/postgres/init/01_schema.sql for complete schema
```

## Future Enhancements

When implementing new features:
1. Follow existing patterns for configuration and error handling
2. Add comprehensive logging for debugging
3. Update `sites_to_scrape.json` schema documentation
4. Consider backward compatibility with SQLite during PostgreSQL migration
5. Implement proper async patterns when adding aiohttp support

## AI Coding Guidelines Reference

For detailed coding standards, refer to `/docs/ai-coding-guidelines.md` which includes:
- 18 comprehensive sections on development best practices
- Enterprise-grade coding standards
- Security, testing, and accessibility requirements
- Git workflow and code review processes

When in doubt about implementation details, always ask for clarification rather than making assumptions.