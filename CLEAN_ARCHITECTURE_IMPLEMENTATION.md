# Clean Architecture Implementation for Mimir

## Overview

This document describes the complete clean architecture refactoring of Mimir, transforming it from a simple scraper into a professional, maintainable broadcasting industry intelligence platform.

## Architecture Structure

```
mimir/
├── __init__.py
├── __main__.py              # Entry point
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py          # Pydantic settings
├── core/                    # Core business logic
│   ├── __init__.py
│   ├── database/            # Database layer
│   │   ├── __init__.py
│   │   └── manager.py       # Async database manager
│   ├── models/              # Domain models
│   │   ├── __init__.py
│   │   ├── base.py          # Base model classes
│   │   ├── article.py       # Article entity
│   │   ├── source.py        # Source entity
│   │   ├── keyword_alert.py # Keyword alert entity
│   │   └── scrape_run.py    # Scrape run entity
│   ├── scrapers/            # Scraping implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Base scraper interface
│   │   ├── rss_scraper.py   # RSS scraper implementation
│   │   ├── html_scraper.py  # HTML scraper implementation
│   │   └── factory.py       # Scraper factory
│   └── services/            # Business logic services
│       ├── __init__.py
│       ├── article_service.py
│       ├── source_service.py
│       └── scraping_service.py
├── api/                     # FastAPI application
│   ├── __init__.py
│   ├── app.py               # FastAPI app factory
│   ├── dependencies.py     # Dependency injection
│   ├── schemas.py           # Pydantic schemas
│   └── routers/             # API route handlers
│       ├── __init__.py
│       ├── health.py
│       ├── articles.py
│       ├── sources.py
│       └── scraping.py
└── cli/                     # Command line interface
    ├── __init__.py
    └── main.py              # Click-based CLI
```

## Key Principles Implemented

### 1. **Separation of Concerns**
- **Models**: Pure domain entities with business logic
- **Services**: Orchestrate business operations
- **Scrapers**: Handle external data acquisition
- **API**: HTTP interface layer
- **CLI**: Command-line interface

### 2. **Dependency Injection**
- Services receive dependencies through constructors
- FastAPI dependencies provide database sessions
- Clean testable interfaces

### 3. **Async/Await Throughout**
- Full async support for database operations
- Non-blocking I/O for web scraping
- Scalable concurrent processing

### 4. **Type Safety**
- Comprehensive type hints throughout
- Pydantic models for validation
- SQLAlchemy 2.0 with typed mappings

### 5. **Configuration Management**
- Hierarchical Pydantic settings
- Environment variable support
- Validation and defaults

## Database Models

### Article
- **Purpose**: Represents scraped news articles
- **Key Fields**: title, content, relevance_score, broadcast_relevance
- **Relationships**: belongs to Source, has many KeywordAlerts

### Source
- **Purpose**: Represents news sources to scrape
- **Key Fields**: name, url, source_type, category, health metrics
- **Relationships**: has many Articles, has many ScrapeRuns

### KeywordAlert
- **Purpose**: Tracks keyword matches in articles
- **Key Fields**: keyword, context, confidence_score
- **Relationships**: belongs to Article

### ScrapeRun
- **Purpose**: Tracks scraping operations for monitoring
- **Key Fields**: started_at, status, article counts, errors
- **Relationships**: may belong to Source

## Service Layer

### ArticleService
- **Purpose**: Article management and search operations
- **Key Methods**:
  - `get_recent_articles()`: Fetch recent articles with filtering
  - `get_featured_article()`: Get hero article for UI
  - `search_articles()`: Text-based article search
  - `get_statistics()`: Article analytics

### SourceService
- **Purpose**: Source management and health monitoring
- **Key Methods**:
  - `create_source()`: Add new news sources
  - `get_sources_health_report()`: Monitor source performance
  - `test_source_connection()`: Validate source accessibility
  - `get_source_statistics()`: Source-specific analytics

### ScrapingService
- **Purpose**: Orchestrates scraping operations
- **Key Methods**:
  - `scrape_all_sources()`: Bulk scraping operation
  - `scrape_single_source()`: Target specific source
  - Handles duplicate detection, keyword processing, error management

## API Endpoints

### Health & Info
- `GET /api/v1/health`: System health check
- `GET /api/v1/info`: API information

### Articles
- `GET /api/v1/articles/`: List articles with pagination
- `GET /api/v1/articles/featured`: Get featured article
- `GET /api/v1/articles/statistics`: Get article statistics
- `GET /api/v1/articles/search`: Search articles
- `GET /api/v1/articles/{id}`: Get specific article
- `PUT /api/v1/articles/{id}`: Update article
- `POST /api/v1/articles/{id}/mark-duplicate`: Mark as duplicate

### Sources
- `GET /api/v1/sources/`: List sources
- `GET /api/v1/sources/health`: Get health report
- `GET /api/v1/sources/{id}`: Get specific source
- `GET /api/v1/sources/{id}/statistics`: Get source statistics
- `GET /api/v1/sources/{id}/articles`: Get articles from source
- `POST /api/v1/sources/`: Create new source
- `PUT /api/v1/sources/{id}`: Update source
- `DELETE /api/v1/sources/{id}`: Delete source
- `POST /api/v1/sources/{id}/enable`: Enable source
- `POST /api/v1/sources/{id}/disable`: Disable source
- `POST /api/v1/sources/{id}/test`: Test source connection

### Scraping
- `POST /api/v1/scraping/run`: Run scraping on all sources
- `POST /api/v1/scraping/sources/{id}/scrape`: Scrape single source
- `GET /api/v1/scraping/status`: Get scraping status
- `POST /api/v1/scraping/test`: Test scraping system

## CLI Commands

```bash
# Start API server in development mode
python run_mimir.py

# Run scraping operation
python run_mimir.py scrape

# Initialize database
python run_mimir.py init

# Check system health
python run_mimir.py health

# Show configuration
python run_mimir.py config

# Start production server
python run_mimir.py production
```

Or using the module directly:
```bash
python -m mimir.cli.main serve --reload
python -m mimir.cli.main scrape --enabled-only
python -m mimir.cli.main init-db
```

## Usage Examples

### Starting the API Server
```bash
# Development mode with auto-reload
python run_mimir.py

# Production mode with multiple workers
python run_mimir.py production
```

### Running Scraping Operations
```bash
# Scrape all enabled sources
python run_mimir.py scrape

# Use the API
curl -X POST "http://localhost:8000/api/v1/scraping/run?enabled_only=true"
```

### Managing Sources via API
```bash
# List all sources
curl "http://localhost:8000/api/v1/sources/"

# Create a new source
curl -X POST "http://localhost:8000/api/v1/sources/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Broadcasting & Cable",
    "url": "https://www.broadcastingcable.com",
    "source_type": "RSS",
    "category": "BROADCAST",
    "rss_url": "https://www.broadcastingcable.com/feed"
  }'

# Test source connection
curl -X POST "http://localhost:8000/api/v1/sources/{source_id}/test"
```

### Searching Articles
```bash
# Search articles
curl "http://localhost:8000/api/v1/articles/search?q=broadcasting&limit=10"

# Get featured article
curl "http://localhost:8000/api/v1/articles/featured"

# Get articles by keyword
curl "http://localhost:8000/api/v1/articles/keywords/technology?days=7"
```

## Database Setup

### Initialize Database
```bash
# Initialize the database schema
python run_mimir.py init

# Or create migration
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

### Configuration
Database configuration is managed through environment variables or the settings system:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/mimir"
export DEBUG=true
export SCRAPING__ARTICLES_PER_SOURCE=50
```

## Error Handling

### Service Level
- Services use structured logging (structlog)
- Comprehensive error handling with rollback
- Validation using Pydantic

### API Level
- HTTP status codes follow REST conventions
- Structured error responses
- Global exception handlers

### CLI Level
- User-friendly error messages
- Proper exit codes
- Graceful interrupt handling

## Testing Strategy

The architecture supports comprehensive testing:

```python
# Unit tests for services
async def test_article_service():
    session = create_test_session()
    service = ArticleService(session)
    article = await service.get_by_id(test_id)
    assert article is not None

# Integration tests for API
def test_list_articles_endpoint():
    response = client.get("/api/v1/articles/")
    assert response.status_code == 200

# End-to-end tests for scraping
async def test_scraping_workflow():
    result = await scraping_service.scrape_all_sources()
    assert result["new_articles"] > 0
```

## Performance Considerations

### Database
- Async SQLAlchemy with connection pooling
- Proper indexes on frequently queried fields
- UUID primary keys for distributed scaling
- Optimized queries with eager loading

### API
- Pagination for large result sets
- Rate limiting protection
- Async request handling
- Structured responses

### Scraping
- Configurable rate limiting
- Retry logic with exponential backoff
- Concurrent source processing
- Duplicate detection

## Security

### API Security
- Mock authentication system (replace with real auth)
- Permission-based access control
- Input validation with Pydantic
- SQL injection protection via SQLAlchemy

### Scraping Security
- Respects robots.txt
- Configurable user agents
- Rate limiting to prevent blocking
- Error handling to prevent crashes

## Deployment

### Development
```bash
python run_mimir.py
```

### Production
```bash
python run_mimir.py production
```

### Docker (Future Enhancement)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "run_mimir.py", "production"]
```

## Future Enhancements

1. **Authentication & Authorization**
   - JWT-based authentication
   - Role-based access control
   - API key management

2. **Advanced Features**
   - Real-time scraping with WebSockets
   - Machine learning content analysis
   - Advanced search with Elasticsearch

3. **Monitoring**
   - Prometheus metrics
   - Health check improvements
   - Performance monitoring

4. **Scalability**
   - Message queue for scraping jobs
   - Microservices decomposition
   - Caching layers

## Migration from Legacy Code

The legacy Mimir components are preserved but should be gradually replaced:

1. Replace direct database calls with service layer
2. Migrate from Flask to FastAPI
3. Update configuration to use Pydantic settings
4. Replace raw SQL with SQLAlchemy operations

## Conclusion

This clean architecture implementation transforms Mimir into a professional, scalable broadcasting industry intelligence platform. The architecture follows SOLID principles, implements proper separation of concerns, and provides a foundation for future enhancements while maintaining the core scraping functionality.

The system is now ready for production use with proper error handling, monitoring, and a clean API interface suitable for building sophisticated broadcast industry analysis tools.