-- Mimir News PostgreSQL Seed Data
-- Version: 1.0.0
-- Description: Initial seed data for development and testing

-- Insert default user for development
INSERT INTO users (email, name, preferences) VALUES 
('admin@mimir.local', 'Mimir Admin', 
 '{"email_notifications": true, "digest_frequency": "daily", "timezone": "Europe/Amsterdam"}'::jsonb)
ON CONFLICT (email) DO NOTHING;

-- Insert default keywords for monitoring
INSERT INTO keywords (keyword, is_regex, case_sensitive, priority, metadata) VALUES 
('artificial intelligence', false, false, 8, '{"category": "technology"}'::jsonb),
('AI', false, true, 9, '{"category": "technology", "synonyms": ["artificial intelligence", "machine learning"]}'::jsonb),
('machine learning', false, false, 8, '{"category": "technology"}'::jsonb),
('blockchain', false, false, 7, '{"category": "technology"}'::jsonb),
('cryptocurrency', false, false, 7, '{"category": "finance"}'::jsonb),
('bitcoin', false, false, 8, '{"category": "finance"}'::jsonb),
('cybersecurity', false, false, 9, '{"category": "security"}'::jsonb),
('data breach', false, false, 10, '{"category": "security", "alert_priority": "high"}'::jsonb),
('climate change', false, false, 7, '{"category": "environment"}'::jsonb),
('renewable energy', false, false, 7, '{"category": "environment"}'::jsonb),
('\b(merger|acquisition|M&A)\b', true, false, 8, '{"category": "business", "type": "pattern"}'::jsonb),
('\b(IPO|initial public offering)\b', true, false, 9, '{"category": "business", "type": "pattern"}'::jsonb)
ON CONFLICT DO NOTHING;

-- Import sources from sites_to_scrape.json (this will be done by migration script)
-- For now, insert some example sources
INSERT INTO sources (name, url, type, category, enabled, config, selectors) VALUES 
-- RSS Feeds
('TechCrunch', 'https://techcrunch.com/feed/', 'rss', 'technology', true, 
 '{"fetch_full_content": true, "max_articles": 50}'::jsonb, '{}'::jsonb),
 
('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/technology-lab', 'rss', 'technology', true,
 '{"fetch_full_content": true}'::jsonb, '{}'::jsonb),
 
('Hacker News', 'https://news.ycombinator.com/rss', 'rss', 'technology', true,
 '{"max_articles": 30}'::jsonb, '{}'::jsonb),

-- HTML Sources (examples - selectors need to be verified)
('The Verge', 'https://www.theverge.com/tech', 'html', 'technology', false,
 '{"pagination": {"enabled": true, "max_pages": 3}}'::jsonb,
 '{
   "overview_article_link": "h2 a",
   "detail_title": "h1",
   "detail_date": "time",
   "detail_content": "div.c-entry-content",
   "detail_author": "span.c-byline__author-name"
 }'::jsonb),

('Reuters Technology', 'https://www.reuters.com/technology', 'html', 'technology', false,
 '{"rate_limit_seconds": 3}'::jsonb,
 '{
   "overview_article_link": "a[data-testid=\"Heading\"]",
   "detail_title": "h1",
   "detail_date": "time",
   "detail_content": "div[data-testid=\"ArticleBody\"]",
   "detail_author": "span.author-name"
 }'::jsonb)
ON CONFLICT (name) DO UPDATE SET
    url = EXCLUDED.url,
    type = EXCLUDED.type,
    category = EXCLUDED.category,
    config = EXCLUDED.config,
    selectors = EXCLUDED.selectors,
    updated_at = CURRENT_TIMESTAMP;

-- Create indexes for JSON queries
CREATE INDEX IF NOT EXISTS idx_keywords_metadata_category ON keywords USING gin((metadata->'category'));
CREATE INDEX IF NOT EXISTS idx_sources_config ON sources USING gin(config);
CREATE INDEX IF NOT EXISTS idx_sources_selectors ON sources USING gin(selectors);

-- Create helper function to migrate keywords from config.json
CREATE OR REPLACE FUNCTION import_keywords_from_config(keywords_array TEXT[])
RETURNS void AS $$
DECLARE
    keyword_item TEXT;
BEGIN
    FOREACH keyword_item IN ARRAY keywords_array
    LOOP
        INSERT INTO keywords (keyword, priority)
        VALUES (keyword_item, 5)
        ON CONFLICT (lower(keyword)) WHERE is_regex = false
        DO UPDATE SET updated_at = CURRENT_TIMESTAMP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate next scrape time
CREATE OR REPLACE FUNCTION calculate_next_scrape_time(
    last_scraped TIMESTAMPTZ,
    rate_limit_seconds INTEGER,
    base_interval INTERVAL DEFAULT '4 hours'
) RETURNS TIMESTAMPTZ AS $$
BEGIN
    -- If never scraped, scrape immediately
    IF last_scraped IS NULL THEN
        RETURN CURRENT_TIMESTAMP;
    END IF;
    
    -- Calculate next scrape based on base interval and rate limit
    RETURN GREATEST(
        last_scraped + base_interval,
        CURRENT_TIMESTAMP + (rate_limit_seconds || ' seconds')::INTERVAL
    );
END;
$$ LANGUAGE plpgsql;

-- Update next_scrape_at for all sources
UPDATE sources 
SET next_scrape_at = calculate_next_scrape_time(last_scraped_at, rate_limit_seconds)
WHERE enabled = true;

-- Create sample data for development (only in dev environment)
DO $$
BEGIN
    -- Check if we're in development mode (no articles exist)
    IF NOT EXISTS (SELECT 1 FROM articles LIMIT 1) THEN
        -- Insert sample articles
        INSERT INTO articles (
            url, title, content, summary, published_date, source, 
            author, category, language, status, metadata, tags
        ) VALUES 
        (
            'https://example.com/article1',
            'Breaking: AI Breakthrough in Natural Language Processing',
            'Researchers at Example University have announced a major breakthrough in natural language processing...',
            'A new AI model shows unprecedented understanding of context and nuance in human language.',
            NOW() - INTERVAL '2 hours',
            'Example Tech News',
            'Jane Doe',
            'technology',
            'en',
            'processed',
            '{"word_count": 500, "read_time_minutes": 3}'::jsonb,
            '["AI", "NLP", "research", "breakthrough"]'::jsonb
        ),
        (
            'https://example.com/article2',
            'Cryptocurrency Market Sees Major Volatility',
            'Bitcoin and other major cryptocurrencies experienced significant price swings today...',
            'Crypto markets react to regulatory news from multiple countries.',
            NOW() - INTERVAL '5 hours',
            'Example Finance News',
            'John Smith',
            'finance',
            'en',
            'processed',
            '{"word_count": 750, "read_time_minutes": 4}'::jsonb,
            '["cryptocurrency", "bitcoin", "finance", "regulation"]'::jsonb
        ),
        (
            'https://example.com/article3',
            'New Cybersecurity Threat Targets Enterprise Systems',
            'Security researchers have identified a new strain of malware targeting enterprise networks...',
            'A sophisticated attack campaign has been discovered affecting major corporations.',
            NOW() - INTERVAL '1 day',
            'Example Security News',
            'Security Team',
            'security',
            'en',
            'processed',
            '{"severity": "high", "affected_systems": ["Windows", "Linux"]}'::jsonb,
            '["cybersecurity", "malware", "enterprise", "threat"]'::jsonb
        );
        
        -- Create keyword alerts for sample articles
        INSERT INTO keyword_alerts (article_id, keyword_id, matched_text, context)
        SELECT 
            a.id,
            k.id,
            k.keyword,
            'Sample context for ' || k.keyword
        FROM articles a
        CROSS JOIN keywords k
        WHERE 
            (a.title ILIKE '%' || k.keyword || '%' OR a.content ILIKE '%' || k.keyword || '%')
            AND NOT k.is_regex
        LIMIT 10;
    END IF;
END $$;