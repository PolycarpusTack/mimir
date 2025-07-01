# 🚀 Mimir News Scraper

Een krachtige, modulaire B2B nieuws scraper gebouwd in Python. Mimir haalt automatisch nieuws op van RSS feeds en HTML websites, slaat ze op in een SQLite database (met PostgreSQL ondersteuning!), en stuurt email notificaties met samenvattingen.

## ✨ Features

- **Dual Parsing**: Ondersteunt zowel RSS feeds als HTML websites
- **Keyword Monitoring**: Automatische detectie van belangrijke keywords in artikelen
- **Email Notificaties**: HTML/plain-text samenvattingen van nieuwe artikelen
- **Respecteert Robots.txt**: Ethisch scrapen met rate limiting
- **Database Opslag**: SQLite met duplicate detectie
- **Scheduler**: Automatisch draaien op gezette tijden
- **Error Handling**: Robuuste retry mechanismes en error logging
- **Pagination Support**: Verwerk meerdere pagina's van nieuwssites

## 📋 Vereisten

- Python 3.8+
- pip (Python package manager)

## 🛠️ Installatie

1. Clone of download dit project naar `C:/Projects/Mimir`

2. Installeer de vereiste packages:
```bash
cd C:/Projects/Mimir
pip install -r requirements.txt
```

3. Kies je database:
   - **SQLite** (standaard): Geen extra setup nodig
   - **PostgreSQL** (aanbevolen voor productie):
     ```bash
     docker-compose up -d
     python db_manager_postgres.py
     alembic upgrade head
     ```

4. Configureer de scraper door `config.json` aan te passen:
   - Stel je email instellingen in
   - Pas de keywords aan voor monitoring
   - Configureer logging preferences
## 🚀 Gebruik

### Eenmalig draaien:
```bash
python scraper.py --run
```

### Statistieken bekijken:
```bash
python scraper.py --stats
```

### Email configuratie testen:
```bash
python scraper.py --test-email
```

### Geplande uitvoering (elke 4 uur):
```bash
python scraper.py --schedule
```

### Database initialiseren:
```bash
# SQLite
python db_manager.py

# PostgreSQL
export USE_POSTGRES=true
python db_manager_postgres.py
```

### Migreren van SQLite naar PostgreSQL:
```bash
python migrate_to_postgres.py
```

## ⚙️ Configuratie

### sites_to_scrape.json

Voeg nieuwe websites toe door het volgende formaat te gebruiken:

**Voor RSS feeds:**
```json
{
    "name": "Site Naam",
    "url": "https://example.com/rss.xml",
    "type": "rss",
    "enabled": true,
    "category": "technology"
}```

**Voor HTML websites:**
```json
{
    "name": "Site Naam",
    "url": "https://example.com/nieuws",
    "type": "html",
    "enabled": true,
    "category": "business",
    "selectors": {
        "overview_article_link": "article a.title",
        "detail_title": "h1.article-title",
        "detail_date": "time.published",
        "detail_content": "div.article-body",
        "detail_author": "span.author-name"
    },
    "date_format": "%B %d, %Y",
    "pagination": {
        "enabled": true,
        "next_page_selector": "a.next-page",
        "max_pages": 5
    }
}
```

### CSS Selectors Vinden

1. Open de doelwebsite in Chrome/Firefox
2. Rechtsklik op het element → "Inspect"
3. Zoek unieke classes of IDs
4. Test in browser console: `document.querySelector('jouw-selector')`

## 📊 Database Schema

- **articles**: Opslag van alle artikelen
- **keyword_alerts**: Gevonden keywords in artikelen
- **scrape_runs**: Statistieken per scrape sessie
- **scrape_errors**: Error logging voor debugging

## 🔧 Troubleshooting

### Email werkt niet:
- Controleer SMTP instellingen in `config.json`
- Voor Gmail: gebruik een App Password (niet je normale wachtwoord)
- Zet `send_email` op `true`

### Geen artikelen gevonden:
- Controleer of de CSS selectors kloppen
- Check `logs/mimir_scraper.log` voor errors
- Verifieer dat de site niet geblokkeerd is door robots.txt

### Database errors:
```bash
# Reset de database
rm mimir_news.db
python db_manager.py
```

## 🚀 Geavanceerde Features

### Keyword Monitoring
Pas de keywords aan in `config.json`:
```json
"keywords_monitoring": {
    "enabled": true,
    "keywords": ["AI", "blockchain", "jouw-keyword"],
    "alert_on_match": true
}
```

### Rate Limiting
Respecteer websites door delays aan te passen:
```json
"default_request_delay_seconds": 2,
"max_retries": 3,
"retry_delay_seconds": 5
```

## 📝 Logging

Logs worden opgeslagen in `logs/mimir_scraper.log` met automatische rotation na 10MB.

## 🤝 Ethisch Scrapen

- Respecteer altijd robots.txt
- Gebruik redelijke delays tussen requests
- Identificeer jezelf met een duidelijke User-Agent
- Scrape alleen publiek beschikbare content

## 📄 Licentie

Dit project is vrij te gebruiken voor persoonlijke en commerciële doeleinden.

---

**Mimir** - Genoemd naar de Norse god van wijsheid en kennis 🧙‍♂️# mimir
