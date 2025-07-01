# 📺 Mimir Broadcast Industry Intelligence Design Document

## 1. Overview

This document outlines the enhancement of Mimir to become a specialized broadcast industry intelligence platform, building on the existing MVP while preparing for future AI-driven analytics.

## 2. Industry-Specific Requirements

### 2.1 Target Sources

#### Broadcaster Company Pages
- **NBCUniversal**: https://www.nbcuniversal.com/press-releases
- **Disney**: https://press.disneyplus.com/, https://thewaltdisneycompany.com/news/
- **Warner Bros Discovery**: https://press.wbd.com/
- **Paramount**: https://www.paramount.com/press
- **BBC**: https://www.bbc.com/mediacentre/
- **Sky**: https://www.skygroup.sky/media-centre
- **RTL Group**: https://www.rtlgroup.com/en/press_releases/
- **ProSiebenSat.1**: https://www.prosiebensat1.com/en/press

#### Industry Standards Organizations
- **ATSC**: https://www.atsc.org/news/
- **DVB**: https://dvb.org/news/
- **SMPTE**: https://www.smpte.org/news-publications
- **EBU**: https://www.ebu.ch/news (robots.txt aware)
- **NAB**: https://www.nab.org/news/

#### Trade Publications
- **Broadcasting & Cable**: https://www.nexttv.com/
- **TV Technology**: https://www.tvtechnology.com/
- **Broadcast Magazine**: https://www.broadcastnow.co.uk/
- **Digital TV Europe**: https://www.digitaltveurope.com/
- **Variety**: https://variety.com/ (entertainment/media)
- **The Hollywood Reporter**: https://www.hollywoodreporter.com/
- **Deadline**: https://deadline.com/
- **C21Media**: https://www.c21media.net/
- **TBI (Television Business International)**: https://tbivision.com/

#### Technology Vendors
- **Harmonic**: https://www.harmonicinc.com/company/press-releases/
- **MediaKind**: https://www.mediakind.com/news/
- **Grass Valley**: https://www.grassvalley.com/news/
- **Imagine Communications**: https://www.imaginecommunications.com/press-releases

## 3. Architecture Design

### 3.1 Modular Approach

```
Mimir/
├── core/
│   ├── scraper.py (existing RSS/HTML scraper)
│   ├── api_scraper.py (API sources)
│   └── broadcast_scraper.py (NEW - industry-specific)
├── intelligence/
│   ├── content_extractor.py (NEW - smart extraction)
│   ├── source_monitor.py (NEW - source health)
│   └── data_enricher.py (NEW - metadata enhancement)
├── ui/
│   ├── web_interface.py (Netflix-style)
│   ├── cli_monitor.py (NEW - CLI view)
│   └── reader_view.py (NEW - article reader)
└── config/
    ├── broadcast_sources.json (NEW - industry sources)
    └── extraction_rules.json (NEW - per-site rules)
```

### 3.2 Separate Tool Integration

The broadcast intelligence module will be:
- **Integrated** into the main scraper flow but with its own configuration
- **Modular** so it can be disabled/enabled independently
- **Configurable** with separate settings for article limits, frequency, etc.

## 4. Feature Specifications

### 4.1 Smart Source Management

```json
{
  "broadcast_sources": {
    "article_limit_per_run": "user_configurable",
    "default_limit": 100,
    "categories": {
      "broadcasters": {
        "limit_multiplier": 2.0,
        "priority": "high"
      },
      "standards": {
        "limit_multiplier": 1.5,
        "priority": "medium"
      },
      "trade_publications": {
        "limit_multiplier": 1.0,
        "priority": "normal"
      }
    }
  }
}
```

### 4.2 Filter Options

#### Content Filters
- **Company**: Filter by specific broadcasters (NBC, Disney, etc.)
- **Region**: North America, Europe, Asia-Pacific, Global
- **Topic**: Technology, Business, Programming, Standards, M&A
- **Content Type**: Press Release, Article, Analysis, Interview
- **Date Range**: Today, This Week, This Month, Custom
- **Announcement Type**: Product Launch, Partnership, Financial, Executive Change

#### Advanced Filters
- **Sentiment**: Positive, Negative, Neutral
- **Relevance Score**: High, Medium, Low (based on keyword density)
- **Source Credibility**: Official, Trade, Blog, Forum
- **Language**: English (primary), + other languages
- **Duplicate Detection**: Aggressive, Normal, Minimal

### 4.3 CLI Monitor View

```
╔══════════════════════════════════════════════════════════════════╗
║                    Mimir Broadcast Scraper Monitor                ║
╠══════════════════════════════════════════════════════════════════╣
║ [2025-06-28 10:23:45] Starting broadcast intelligence scan...     ║
║                                                                   ║
║ Sources Status:                                                   ║
║ ✅ NBCUniversal Press     [5/5 articles]   200 OK   0.3s        ║
║ ✅ Disney Press           [3/5 articles]   200 OK   0.5s        ║
║ ⏳ Warner Bros Discovery  [scanning...]                          ║
║ ❌ ATSC News             [0/5 articles]   403 Forbidden         ║
║ ⚠️  DVB News              [2/5 articles]   Partial (timeout)     ║
║                                                                   ║
║ Statistics:                                                       ║
║ Articles Found: 127 | New: 43 | Duplicates: 84                   ║
║ Keywords Matched: streaming (23), ATSC 3.0 (12), OTT (19)        ║
║                                                                   ║
║ [Press Q to quit, P to pause, R to refresh]                      ║
╚══════════════════════════════════════════════════════════════════╝
```

### 4.4 Article Reader View

```
┌─────────────────────────────────────────────────────────────────┐
│ Disney Announces New Streaming Technology Partnership            │
│ Source: Disney Press | Date: June 28, 2025 | 5 min read        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ The Walt Disney Company today announced a strategic partnership │
│ with MediaKind to enhance its streaming infrastructure across   │
│ Disney+, Hulu, and ESPN+. The collaboration will focus on      │
│ next-generation video compression and low-latency delivery...   │
│                                                                 │
│ [Full article content in clean, readable format]                │
│                                                                 │
│ Keywords: streaming, video compression, Disney+, MediaKind      │
│ Related: 5 similar articles found                              │
└─────────────────────────────────────────────────────────────────┘
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Current)
- ✅ Basic scraper with RSS/HTML
- ✅ Netflix-style UI
- ✅ API integration framework
- 🔄 Broadcast-specific sources

### Phase 2: Intelligence Layer (Next)
- Smart content extraction
- Source health monitoring
- CLI monitoring view
- Article reader view

### Phase 3: Advanced Features (Future)
- ML-based relevance scoring
- Trend prediction
- Competitive intelligence
- Executive briefings

## 6. Direct Scraping Strategy

### 6.1 Ethical Scraping Approach
1. **Respect robots.txt** but check for:
   - RSS feeds not listed in robots.txt
   - API endpoints (often not restricted)
   - Sitemap.xml files
   
2. **Smart Detection**:
   - Auto-detect press release patterns
   - Find JSON-LD structured data
   - Extract from meta tags

### 6.2 Circumvention Techniques (Legal)
1. **RSS Discovery**: Many sites have hidden RSS feeds
2. **API Endpoints**: Reverse-engineer public APIs
3. **Structured Data**: Use schema.org markup
4. **Social Media**: Official Twitter/LinkedIn feeds
5. **Google News**: Aggregate from Google News RSS

## 7. Configuration Management

### 7.1 User Settings in Web UI
```
Settings > Broadcast Intelligence
├── Article Limits
│   ├── Total per run: [100] (adjustable)
│   ├── Per source: [10] (adjustable)
│   └── Per category: [customizable multipliers]
├── Update Frequency
│   ├── Broadcasters: Every 2 hours
│   ├── Trade pubs: Every 4 hours
│   └── Standards orgs: Daily
└── Filters
    ├── Companies: [x] NBC [x] Disney [ ] Warner...
    ├── Topics: [x] Technology [x] M&A [ ] Sports...
    └── Regions: [x] North America [x] Europe...
```

## 8. Data Schema Extensions

### 8.1 Enhanced Article Model
```python
{
    # Existing fields
    "id", "url", "title", "content", "publication_date",
    
    # New broadcast-specific fields
    "company": "Disney",
    "announcement_type": "partnership",
    "region": "global",
    "relevance_score": 0.89,
    "entities_mentioned": ["MediaKind", "Disney+", "Hulu"],
    "executive_quotes": [...],
    "financial_impact": "Not disclosed",
    "competitive_implications": {...}
}
```

## 9. Integration with Future Roadmap

This design prepares for:
- **EPIC 2**: AI analysis of announcements
- **EPIC 3**: Semantic search across companies
- **EPIC 6**: Analytics on industry trends
- **EPIC 8**: Executive briefing generation

## 10. Success Metrics

- Cover 90% of major broadcast industry announcements
- Process 500-1000 relevant articles daily
- Achieve <5% duplicate rate
- Maintain 95% uptime for critical sources
- Generate actionable insights within 2 hours of announcement