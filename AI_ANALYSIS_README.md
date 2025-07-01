# AI-Powered Content Analysis for Mimir

This document describes the AI analysis capabilities added to Mimir as part of EPIC 2: AI-Powered Content Analysis Engine.

## Overview

The AI analysis engine provides intelligent content analysis for scraped news articles, including:

- **Multi-language support** (English, Dutch, German, French)
- **Named Entity Recognition** (companies, people, locations, products)
- **Entity Linking** to Wikipedia and Wikidata knowledge bases
- **Keyword extraction** using multiple algorithms (TF-IDF, YAKE, TextRank)
- **Topic Modeling** with LDA for discovering themes across articles
- **Industry Classification** using zero-shot learning
- **Sentiment analysis** with market impact assessment
- **Text preprocessing** with language detection
- **Scalable processing** via Redis queue and worker processes

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Language Models

```bash
python setup_nlp_models.py
```

This will download spaCy language models for all supported languages.

### 3. Verify PostgreSQL Setup

The AI analysis features work best with PostgreSQL (includes JSONB support for metadata):

```bash
export USE_POSTGRES=true
```

## Usage

### Command Line Interface

The `ai_scraper_integration.py` script provides several commands:

#### Analyze Recent Articles

```bash
# Analyze articles from the last 24 hours
python ai_scraper_integration.py recent

# Analyze articles from the last 48 hours
python ai_scraper_integration.py recent --hours 48

# Use larger batch size for faster processing
python ai_scraper_integration.py recent --batch-size 20
```

#### Analyze Articles by Source

```bash
# Analyze articles from a specific news source
python ai_scraper_integration.py source "TechCrunch"

# Limit the number of articles
python ai_scraper_integration.py source "Reuters" --limit 100
```

#### Analyze All Unanalyzed Articles

```bash
# Process up to 100 unanalyzed articles
python ai_scraper_integration.py all

# Process more articles
python ai_scraper_integration.py all --limit 500
```

#### Start AI Service (Advanced)

For high-volume processing, start the AI service with worker processes:

```bash
# Start with default number of workers
python ai_scraper_integration.py service

# Start with specific number of workers
python ai_scraper_integration.py service --workers 4
```

### Integration with Scraper

Add AI analysis to your scraping workflow:

```python
from ai_analysis import analyze_new_articles

# After running the scraper
analyze_new_articles(limit=50)
```

Or in your scraper script:

```bash
# Run scraper
python scraper.py --run

# Then analyze new articles
python ai_scraper_integration.py recent
```

### Programmatic Usage

```python
from ai_analysis import AIAnalysisEngine

# Initialize engine
engine = AIAnalysisEngine()

# Analyze a single article
article = {
    'id': 'article-123',
    'title': 'Breaking: Major Tech Acquisition',
    'content': 'Full article content here...',
    'source': 'TechNews',
    'category': 'technology'
}

result = engine.analyze_article(article)

# Access results
print(f"Language: {result['preprocessing']['language']}")
print(f"Sentiment: {result['sentiment']['overall_sentiment']}")
print(f"Top entities: {result['entities']['entities']}")
print(f"Keywords: {result['keywords']['keywords']['ensemble'][:5]}")
```

## Analysis Components

### 1. Text Preprocessing (`nlp_preprocessing.py`)

- HTML cleaning and text extraction
- Character encoding fixes
- Language detection
- Sentence segmentation
- Text normalization

### 2. Entity Extraction (`entity_extraction.py`)

Extracts and normalizes:
- **Organizations**: Companies, institutions, agencies
- **People**: Names with role detection
- **Locations**: Cities, countries, regions
- **Products**: Product and technology names
- **Financial**: Stock tickers, monetary amounts
- **Dates/Times**: Temporal references

### 3. Keyword Extraction (`keyword_extraction.py`)

Multiple algorithms for comprehensive keyword extraction:
- **TF-IDF**: Statistical importance of terms
- **YAKE**: Yet Another Keyword Extractor
- **TextRank**: Graph-based ranking
- **Ensemble**: Combined approach for best results

### 4. Sentiment Analysis (`sentiment_analysis.py`)

- Overall article sentiment (very negative to very positive)
- Confidence scores
- Aspect-based sentiment (product, company, performance)
- Market impact assessment (for financial news)
- Urgency detection

### 5. NLP Service Architecture (`nlp_service.py`)

- Redis-based task queue
- Multi-process worker architecture
- Automatic retry with exponential backoff
- Result caching
- Performance monitoring

### 6. Topic Modeling (`topic_modeling.py`)

- LDA (Latent Dirichlet Allocation) implementation
- Automatic topic number optimization
- Topic coherence calculation
- Topic trend analysis over time
- Interactive visualization support

### 7. Industry Classification (`industry_classification.py`)

- Zero-shot classification (no training data required)
- 20 predefined industry categories
- Keyword-based validation
- Multi-label classification support
- Confidence scoring

### 8. Entity Linking (`entity_linking.py`)

- Wikipedia article matching
- Wikidata property extraction
- Disambiguation handling
- Knowledge graph creation
- Confidence-based linking

## Database Schema

AI analysis results are stored in the `metadata` JSONB column (PostgreSQL):

```json
{
  "ai_analysis": {
    "timestamp": "2024-01-15T10:30:00Z",
    "components": ["preprocess", "entities", "keywords", "sentiment"],
    "processing_time": 2.34
  },
  "language": "en",
  "word_count": 456,
  "entities": {
    "organization": ["TechCorp", "Microsoft"],
    "person": ["John Smith", "Sarah Johnson"],
    "location": ["San Francisco", "United States"]
  },
  "keywords": ["artificial intelligence", "machine learning", "innovation"],
  "sentiment": {
    "label": "positive",
    "score": 0.82,
    "confidence": 0.91
  },
  "insights": {
    "summary": "Key topics: AI, acquisition. Main entities: TechCorp, Microsoft. Positive sentiment.",
    "key_topics": ["artificial intelligence", "acquisition", "technology"],
    "main_entities": ["TechCorp", "Microsoft", "John Smith"]
  }
}
```

## Monitoring and Statistics

View AI analysis statistics:

```python
from ai_analysis import AIAnalysisEngine

engine = AIAnalysisEngine()
stats = engine.get_analysis_stats()

print(f"Total analyzed: {stats['total_analyzed']}")
print(f"By language: {stats['by_language']}")
print(f"Sentiment distribution: {stats['sentiment_distribution']}")
print(f"Top entities: {stats['top_entities']}")
print(f"Top keywords: {stats['top_keywords']}")
```

## Performance Considerations

1. **Batch Processing**: Process articles in batches for efficiency
2. **Model Loading**: Models are loaded once and reused
3. **Caching**: Results are cached in Redis (if using service mode)
4. **Parallel Processing**: Use multiple workers for large volumes
5. **Memory Optimization**: New optimization features for production use

### Memory Optimization Features (NEW)

The AI analysis engine now includes advanced memory optimization:

#### Optimized Mode (Default)
```bash
# Run with optimization (default)
python ai_scraper_integration.py recent

# Explicitly disable optimization
python ai_scraper_integration.py recent --no-optimization
```

#### Features:
- **Model Pool**: Singleton pattern for sharing models across processes
- **Dynamic Batching**: Automatically adjusts batch size based on text length
- **Memory Monitoring**: Tracks memory usage and triggers cleanup when needed
- **Efficient Context**: Disables gradient computation and manages GPU memory
- **Model Cleanup**: Removes unused models after configurable timeout

#### Performance Improvements:
- ~30-40% faster processing for batches
- ~50% less memory growth during processing
- Better GPU utilization (if available)
- Automatic memory cleanup prevents OOM errors

#### Configuration:
```python
# Initialize with custom memory limit
engine = AIAnalysisEngine(use_optimization=True, max_memory_mb=1024)

# Test optimization
python test_ai_optimization.py
```

## Troubleshooting

### Models Not Loading

If language models fail to load:

```bash
# Re-download models
python -m spacy download en_core_web_md
python -m spacy download nl_core_news_md
# etc.
```

### Memory Issues

For large-scale processing:

1. Reduce batch size
2. Use the service mode with workers
3. Process by source or time range

### PostgreSQL Connection Issues

Ensure PostgreSQL is running:

```bash
docker-compose up -d
export USE_POSTGRES=true
```

## Future Enhancements

Planned features (EPIC 2 remaining tasks):

1. **Custom Entity Training**: Train models for industry-specific entities
2. **Entity Linking**: Wikipedia/Wikidata integration
3. **Topic Modeling**: LDA implementation for topic discovery
4. **Zero-shot Classification**: Industry classification without training
5. **Market Impact Detection**: Advanced financial sentiment analysis

## API Endpoints (when using web_interface.py)

The web interface will be extended to include:

- `GET /api/articles/{id}/analysis` - Get analysis for specific article
- `POST /api/analyze` - Trigger analysis for new articles
- `GET /api/stats/sentiment` - Get sentiment distribution
- `GET /api/stats/entities` - Get top entities
- `GET /api/stats/keywords` - Get trending keywords

## Contributing

When adding new AI capabilities:

1. Follow the existing module structure
2. Add comprehensive error handling
3. Include language support where applicable
4. Update this documentation
5. Add tests for new functionality