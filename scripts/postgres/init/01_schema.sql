-- Mimir News PostgreSQL Schema
-- Version: 1.0.0
-- Description: Enhanced schema with full-text search, JSONB metadata, and optimized indexing

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create custom types
CREATE TYPE article_status AS ENUM ('new', 'processed', 'archived', 'error');
CREATE TYPE scrape_type AS ENUM ('rss', 'html', 'api');

-- Main articles table with enhanced features
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT NOT NULL UNIQUE,
    url_hash VARCHAR(64) GENERATED ALWAYS AS (encode(sha256(url::bytea), 'hex')) STORED,
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    published_date TIMESTAMPTZ,
    scraped_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255) NOT NULL,
    author TEXT,
    category VARCHAR(100),
    language VARCHAR(10) DEFAULT 'en',
    status article_status DEFAULT 'new',
    
    -- JSONB fields for flexible metadata
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    entities JSONB DEFAULT '{}',
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', COALESCE(title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(summary, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(content, '')), 'C') ||
        setweight(to_tsvector('english', COALESCE(author, '')), 'D')
    ) STORED,
    
    -- Constraints
    CONSTRAINT valid_published_date CHECK (published_date <= CURRENT_TIMESTAMP),
    CONSTRAINT valid_url CHECK (url ~ '^https?://')
);

-- Indexes for articles
CREATE INDEX idx_articles_published_date ON articles(published_date DESC);
CREATE INDEX idx_articles_source ON articles(source);
CREATE INDEX idx_articles_category ON articles(category);
CREATE INDEX idx_articles_status ON articles(status);
CREATE INDEX idx_articles_language ON articles(language);
CREATE INDEX idx_articles_search ON articles USING gin(search_vector);
CREATE INDEX idx_articles_metadata ON articles USING gin(metadata);
CREATE INDEX idx_articles_tags ON articles USING gin(tags);
CREATE INDEX idx_articles_url_hash ON articles(url_hash);
CREATE INDEX idx_articles_scraped_at ON articles(scraped_at DESC);

-- Table for managing news sources
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    url TEXT NOT NULL,
    type scrape_type NOT NULL,
    enabled BOOLEAN DEFAULT true,
    category VARCHAR(100),
    
    -- Configuration for scraping
    config JSONB DEFAULT '{}',
    selectors JSONB DEFAULT '{}',
    
    -- Rate limiting and scheduling
    rate_limit_seconds INTEGER DEFAULT 2,
    last_scraped_at TIMESTAMPTZ,
    next_scrape_at TIMESTAMPTZ,
    
    -- Statistics
    total_articles INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    last_error TEXT,
    last_error_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sources_enabled ON sources(enabled);
CREATE INDEX idx_sources_next_scrape ON sources(next_scrape_at) WHERE enabled = true;

-- Keyword monitoring and alerts
CREATE TABLE keywords (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    keyword VARCHAR(255) NOT NULL,
    is_regex BOOLEAN DEFAULT false,
    case_sensitive BOOLEAN DEFAULT false,
    enabled BOOLEAN DEFAULT true,
    alert_email BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 5,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_keywords_enabled ON keywords(enabled);
CREATE UNIQUE INDEX idx_keywords_unique ON keywords(lower(keyword)) WHERE is_regex = false;

-- Keyword alerts (matches found in articles)
CREATE TABLE keyword_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    keyword_id UUID NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    matched_text TEXT,
    context TEXT,
    position INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    notified BOOLEAN DEFAULT false,
    notified_at TIMESTAMPTZ,
    
    CONSTRAINT unique_alert UNIQUE(article_id, keyword_id, position)
);

CREATE INDEX idx_keyword_alerts_article ON keyword_alerts(article_id);
CREATE INDEX idx_keyword_alerts_keyword ON keyword_alerts(keyword_id);
CREATE INDEX idx_keyword_alerts_notified ON keyword_alerts(notified) WHERE notified = false;

-- Scrape runs for tracking and statistics
CREATE TABLE scrape_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    started_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMPTZ,
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    
    -- Statistics
    articles_found INTEGER DEFAULT 0,
    articles_new INTEGER DEFAULT 0,
    articles_updated INTEGER DEFAULT 0,
    articles_error INTEGER DEFAULT 0,
    
    -- Performance metrics
    duration_seconds NUMERIC,
    memory_usage_mb NUMERIC,
    
    -- Status and errors
    status VARCHAR(50),
    error_message TEXT,
    
    -- Detailed metrics in JSONB
    metrics JSONB DEFAULT '{}'
);

CREATE INDEX idx_scrape_runs_started ON scrape_runs(started_at DESC);
CREATE INDEX idx_scrape_runs_source ON scrape_runs(source_id);
CREATE INDEX idx_scrape_runs_status ON scrape_runs(status);

-- Detailed scrape errors for debugging
CREATE TABLE scrape_errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scrape_run_id UUID REFERENCES scrape_runs(id) ON DELETE CASCADE,
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    url TEXT,
    error_type VARCHAR(100),
    error_message TEXT,
    error_details JSONB DEFAULT '{}',
    occurred_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMPTZ
);

CREATE INDEX idx_scrape_errors_run ON scrape_errors(scrape_run_id);
CREATE INDEX idx_scrape_errors_source ON scrape_errors(source_id);
CREATE INDEX idx_scrape_errors_type ON scrape_errors(error_type);
CREATE INDEX idx_scrape_errors_unresolved ON scrape_errors(resolved) WHERE resolved = false;

-- User preferences and notifications (for future multi-tenancy)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    active BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}',
    api_key VARCHAR(64) UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_api_key ON users(api_key) WHERE api_key IS NOT NULL;

-- Email notifications log
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    subject TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    sent BOOLEAN DEFAULT false,
    sent_at TIMESTAMPTZ,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_pending ON notifications(sent) WHERE sent = false;
CREATE INDEX idx_notifications_created ON notifications(created_at DESC);

-- Article embeddings for similarity search and semantic capabilities
CREATE TABLE article_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    embedding vector(384),  -- Default dimension for all-MiniLM-L6-v2
    model_name VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    model_version VARCHAR(50) NOT NULL DEFAULT '1.0',
    embedding_type VARCHAR(50) NOT NULL DEFAULT 'title_content',  -- title_content, title_only, content_only
    quality_score FLOAT DEFAULT 1.0,  -- Quality score for the embedding
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure one embedding per article per type
    CONSTRAINT unique_article_embedding UNIQUE(article_id, embedding_type, model_name)
);

-- Indexes for vector similarity search
CREATE INDEX idx_article_embeddings_article ON article_embeddings(article_id);
CREATE INDEX idx_article_embeddings_model ON article_embeddings(model_name, model_version);
CREATE INDEX idx_article_embeddings_type ON article_embeddings(embedding_type);

-- HNSW index for efficient similarity search (requires sufficient data)
-- This will be created after data is populated
-- CREATE INDEX idx_article_embeddings_vector ON article_embeddings 
--   USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- IVFFlat index as alternative (better for smaller datasets)
CREATE INDEX idx_article_embeddings_vector_ivf ON article_embeddings 
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sources_updated_at BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for full-text search
CREATE OR REPLACE FUNCTION search_articles(
    query_text TEXT,
    limit_count INTEGER DEFAULT 100,
    offset_count INTEGER DEFAULT 0
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    url TEXT,
    published_date TIMESTAMPTZ,
    source VARCHAR(255),
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id,
        a.title,
        a.url,
        a.published_date,
        a.source,
        ts_rank(a.search_vector, websearch_to_tsquery('english', query_text)) AS rank
    FROM articles a
    WHERE a.search_vector @@ websearch_to_tsquery('english', query_text)
    ORDER BY rank DESC, a.published_date DESC
    LIMIT limit_count
    OFFSET offset_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for semantic similarity search
CREATE OR REPLACE FUNCTION search_articles_semantic(
    query_embedding vector(384),
    similarity_threshold FLOAT DEFAULT 0.1,
    limit_count INTEGER DEFAULT 20,
    model_name TEXT DEFAULT 'all-MiniLM-L6-v2',
    embedding_type TEXT DEFAULT 'title_content'
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    url TEXT,
    published_date TIMESTAMPTZ,
    source VARCHAR(255),
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id,
        a.title,
        a.url,
        a.published_date,
        a.source,
        1 - (ae.embedding <=> query_embedding) AS similarity_score
    FROM articles a
    INNER JOIN article_embeddings ae ON ae.article_id = a.id
    WHERE ae.model_name = search_articles_semantic.model_name
      AND ae.embedding_type = search_articles_semantic.embedding_type
      AND (1 - (ae.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY ae.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Create function for hybrid search (combines full-text and semantic)
CREATE OR REPLACE FUNCTION search_articles_hybrid(
    query_text TEXT,
    query_embedding vector(384),
    semantic_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3,
    limit_count INTEGER DEFAULT 20,
    model_name TEXT DEFAULT 'all-MiniLM-L6-v2'
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    url TEXT,
    published_date TIMESTAMPTZ,
    source VARCHAR(255),
    combined_score FLOAT,
    semantic_score FLOAT,
    keyword_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_results AS (
        SELECT 
            a.id,
            a.title,
            a.url,
            a.published_date,
            a.source,
            1 - (ae.embedding <=> query_embedding) AS sem_score
        FROM articles a
        INNER JOIN article_embeddings ae ON ae.article_id = a.id
        WHERE ae.model_name = search_articles_hybrid.model_name
          AND ae.embedding_type = 'title_content'
    ),
    keyword_results AS (
        SELECT 
            a.id,
            ts_rank(a.search_vector, websearch_to_tsquery('english', query_text)) AS kw_score
        FROM articles a
        WHERE a.search_vector @@ websearch_to_tsquery('english', query_text)
    )
    SELECT 
        sr.id,
        sr.title,
        sr.url,
        sr.published_date,
        sr.source,
        (semantic_weight * COALESCE(sr.sem_score, 0) + 
         keyword_weight * COALESCE(kr.kw_score, 0)) AS combined_score,
        sr.sem_score AS semantic_score,
        COALESCE(kr.kw_score, 0.0) AS keyword_score
    FROM semantic_results sr
    LEFT JOIN keyword_results kr ON kr.id = sr.id
    WHERE (sr.sem_score IS NOT NULL OR kr.kw_score IS NOT NULL)
    ORDER BY combined_score DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar articles to a given article
CREATE OR REPLACE FUNCTION find_similar_articles(
    target_article_id UUID,
    similarity_threshold FLOAT DEFAULT 0.3,
    limit_count INTEGER DEFAULT 10,
    model_name TEXT DEFAULT 'all-MiniLM-L6-v2'
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    url TEXT,
    published_date TIMESTAMPTZ,
    source VARCHAR(255),
    similarity_score FLOAT
) AS $$
DECLARE
    target_embedding vector(384);
BEGIN
    -- Get the embedding for the target article
    SELECT ae.embedding INTO target_embedding
    FROM article_embeddings ae
    WHERE ae.article_id = target_article_id 
      AND ae.model_name = find_similar_articles.model_name
      AND ae.embedding_type = 'title_content'
    LIMIT 1;
    
    -- If no embedding found, return empty result
    IF target_embedding IS NULL THEN
        RETURN;
    END IF;
    
    -- Find similar articles
    RETURN QUERY
    SELECT 
        a.id,
        a.title,
        a.url,
        a.published_date,
        a.source,
        1 - (ae.embedding <=> target_embedding) AS similarity_score
    FROM articles a
    INNER JOIN article_embeddings ae ON ae.article_id = a.id
    WHERE ae.model_name = find_similar_articles.model_name
      AND ae.embedding_type = 'title_content'
      AND a.id != target_article_id  -- Exclude the target article itself
      AND (1 - (ae.embedding <=> target_embedding)) >= similarity_threshold
    ORDER BY ae.embedding <=> target_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for article statistics
CREATE MATERIALIZED VIEW article_stats AS
SELECT 
    DATE(scraped_at) as date,
    source,
    category,
    status,
    COUNT(*) as article_count,
    COUNT(DISTINCT author) as author_count,
    AVG(LENGTH(content)) as avg_content_length
FROM articles
GROUP BY DATE(scraped_at), source, category, status;

CREATE UNIQUE INDEX idx_article_stats_unique ON article_stats(date, source, category, status);

-- Create view for recent articles with keyword matches
CREATE VIEW recent_articles_with_alerts AS
SELECT 
    a.id,
    a.title,
    a.url,
    a.published_date,
    a.source,
    a.category,
    COALESCE(
        json_agg(
            json_build_object(
                'keyword', k.keyword,
                'matched_text', ka.matched_text,
                'context', ka.context
            ) 
            ORDER BY k.priority DESC
        ) FILTER (WHERE k.id IS NOT NULL), 
        '[]'::json
    ) as keyword_matches
FROM articles a
LEFT JOIN keyword_alerts ka ON ka.article_id = a.id
LEFT JOIN keywords k ON k.id = ka.keyword_id
WHERE a.scraped_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY a.id, a.title, a.url, a.published_date, a.source, a.category
ORDER BY a.published_date DESC;

-- Partitioning setup for articles table (by month)
-- This will be implemented in migration scripts for existing data

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO mimir_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO mimir_user;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO mimir_user;