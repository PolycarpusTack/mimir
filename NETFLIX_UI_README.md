# 🎬 Mimir Netflix-Style Interface

A modern, Netflix-inspired interface for the Mimir news scraper that makes browsing news articles intuitive and enjoyable.

## 🚀 Quick Start

```bash
# Easy launch
python start_netflix_ui.py

# Or manually
python web_interface.py
```

Then open http://localhost:5000 in your browser.

## ✨ Features

### 📺 Netflix-Style Design
- **Dark Theme**: Easy on the eyes with a professional look
- **Hero Section**: Featured article with large display
- **Horizontal Swimlanes**: Browse articles by category
- **Smooth Animations**: Hover effects and transitions
- **Responsive Layout**: Works on desktop and mobile

### 🏊 Smart Swimlanes
- **Latest Articles**: Always shows the newest content
- **Keyword-Based**: Automatic grouping by topics (AI, Security, etc.)
- **Your Watchlist**: Custom selection of keywords
- **Hide/Show**: Toggle swimlanes visibility
- **Future**: Drag-and-drop reordering

### ⚙️ Settings Panel
Access via the gear icon (⚙️) in the header:

#### Scraper Control (NEW!)
- **Run Scraper**: Click "🚀 Refresh News Now" to fetch new articles
- **Status Display**: See when the scraper last ran and how many articles were found
- **Auto-refresh**: The interface updates automatically when scraping completes
- **Background Operation**: Scraper runs in the background, you can continue browsing

#### Add News Sources
1. Click the settings icon
2. Choose source type (RSS Feed or Website)
3. Enter source name and URL
4. Click "Add Source"

#### Manage Sources
- Toggle sources on/off with the switch
- Currently configured sources are displayed
- Changes take effect immediately

### 🔍 Search
- Click the search icon (🔍)
- Full-screen search overlay
- Real-time results as you type
- Click any result to read the full article

### 📊 Live Statistics
- Total articles collected
- Today's new articles  
- Active news sources
- Auto-updates every minute

## 🎨 User Interface Guide

### Quick Actions
- **Green Rocket Button (🚀)**: Quick-run the scraper without opening settings
- **Blue Refresh Button (🔄)**: Refresh the current view

### Navigation
- **Home**: Main swimlane view
- **Trending**: (Coming soon) Popular articles
- **Categories**: (Coming soon) Browse by topic
- **Analytics**: (Coming soon) Detailed insights

### Article Cards
- **Colored backgrounds**: Unique gradient per source
- **Smart icons**: Based on article content (🤖 for AI, 🔐 for security, etc.)
- **Hover to preview**: See article summary
- **Click to read**: Opens original article in new tab
- **Keyword badges**: Red tags show matched keywords

### Keyboard Shortcuts
- `Esc`: Close search or settings
- `Ctrl+K`: (Coming soon) Quick search

## 🔧 Configuration

### Keywords Setup
Edit `config.json` to define your keywords:

```json
{
  "keywords": [
    "AI", "machine learning", "security", 
    "Apple", "Google", "Microsoft"
  ]
}
```

These keywords automatically create swimlanes and highlight matching articles.

### Adding Sources

#### RSS Feeds (Recommended)
Most news sites provide RSS feeds:
- TechCrunch: `https://techcrunch.com/feed/`
- The Verge: `https://www.theverge.com/rss/index.xml`
- Ars Technica: `https://feeds.arstechnica.com/arstechnica/index`

#### HTML Websites
For sites without RSS, you can scrape HTML directly (requires CSS selectors).

## 🎯 Tips

1. **First Time Setup**: Run the scraper first to populate articles
   ```bash
   python scraper.py --run
   ```

2. **Schedule Updates**: Keep content fresh
   ```bash
   python scraper.py --schedule
   ```

3. **Customize Swimlanes**: Edit keywords in `config.json` to create topic-based swimlanes

4. **Performance**: The interface loads 20 articles per swimlane for smooth scrolling

## 🚧 Coming Soon

- Drag-and-drop swimlane ordering
- Custom swimlane creation
- Advanced filtering options
- Export articles to PDF/Excel
- Team collaboration features
- Mobile app

## 🐛 Troubleshooting

**No articles showing?**
- Run the scraper first: `python scraper.py --run`
- Check if sources are enabled in settings

**Settings not saving?**
- Ensure `sites_to_scrape.json` is writable
- Check console for errors

**Slow loading?**
- Reduce number of articles per swimlane
- Enable fewer sources
- Check database performance

## 📝 Notes

- The interface is fully responsive but optimized for desktop
- All data stays local - no external tracking
- Articles open in original source websites
- Swimlanes update on page refresh

Enjoy your Netflix-style news browsing experience! 🎬📰