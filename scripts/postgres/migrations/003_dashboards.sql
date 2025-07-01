-- Migration: Add Dashboard and Widget System
-- Description: Create tables for custom dashboards with configurable widgets

-- Create dashboards table
CREATE TABLE IF NOT EXISTS dashboards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    layout VARCHAR(50) NOT NULL DEFAULT 'grid',
    theme VARCHAR(50) NOT NULL DEFAULT 'light',
    is_default BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    tags JSONB DEFAULT '[]'::jsonb,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT dashboards_layout_check CHECK (layout IN ('grid', 'single_column', 'two_column', 'sidebar_left', 'sidebar_right')),
    CONSTRAINT dashboards_theme_check CHECK (theme IN ('light', 'dark', 'auto', 'custom'))
);

-- Create widgets table
CREATE TABLE IF NOT EXISTS widgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dashboard_id UUID NOT NULL REFERENCES dashboards(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    size VARCHAR(50) NOT NULL DEFAULT 'medium',
    refresh_interval INTEGER NOT NULL DEFAULT 300, -- seconds
    data_source JSONB NOT NULL,
    display_options JSONB DEFAULT '{}'::jsonb,
    position JSONB DEFAULT '{"x": 0, "y": 0}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT widgets_type_check CHECK (type IN (
        'article_count', 'article_list', 'search_results', 
        'chart_line', 'chart_bar', 'chart_pie', 'metric_card',
        'source_stats', 'keyword_cloud', 'alert_feed', 
        'activity_timeline', 'custom_html'
    )),
    CONSTRAINT widgets_size_check CHECK (size IN ('small', 'medium', 'large', 'wide', 'tall', 'extra_large'))
);

-- Create dashboard_shares table for sharing dashboards
CREATE TABLE IF NOT EXISTS dashboard_shares (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dashboard_id UUID NOT NULL REFERENCES dashboards(id) ON DELETE CASCADE,
    shared_with_user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions JSONB DEFAULT '["view"]'::jsonb,
    shared_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    shared_by UUID NOT NULL REFERENCES users(id),
    
    -- Unique constraint to prevent duplicate shares
    CONSTRAINT unique_dashboard_share UNIQUE (dashboard_id, shared_with_user_id)
);

-- Create dashboard_templates table
CREATE TABLE IF NOT EXISTS dashboard_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(100) NOT NULL,
    preview_image TEXT,
    widgets JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create widget_cache table for caching widget data
CREATE TABLE IF NOT EXISTS widget_cache (
    widget_id UUID NOT NULL REFERENCES widgets(id) ON DELETE CASCADE,
    cache_key VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (widget_id, cache_key)
);

-- Indexes for performance
CREATE INDEX idx_dashboards_org_user ON dashboards(organization_id, user_id);
CREATE INDEX idx_dashboards_is_default ON dashboards(is_default) WHERE is_default = TRUE;
CREATE INDEX idx_dashboards_is_public ON dashboards(is_public) WHERE is_public = TRUE;
CREATE INDEX idx_dashboards_tags ON dashboards USING gin(tags);

CREATE INDEX idx_widgets_dashboard ON widgets(dashboard_id);
CREATE INDEX idx_widgets_type ON widgets(type);

CREATE INDEX idx_dashboard_shares_dashboard ON dashboard_shares(dashboard_id);
CREATE INDEX idx_dashboard_shares_user ON dashboard_shares(shared_with_user_id);

CREATE INDEX idx_widget_cache_expires ON widget_cache(expires_at);

-- Row Level Security for dashboards
ALTER TABLE dashboards ENABLE ROW LEVEL SECURITY;

CREATE POLICY dashboards_isolation_policy ON dashboards
    USING (
        -- Users can see their own dashboards
        user_id = current_setting('app.current_user_id')::uuid
        OR
        -- Users can see dashboards shared with them
        EXISTS (
            SELECT 1 FROM dashboard_shares
            WHERE dashboard_id = dashboards.id
            AND shared_with_user_id = current_setting('app.current_user_id')::uuid
        )
        OR
        -- Users can see public dashboards in their organization
        (is_public = TRUE AND organization_id = current_setting('app.current_organization_id')::uuid)
    );

CREATE POLICY dashboards_insert_policy ON dashboards
    FOR INSERT
    WITH CHECK (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND user_id = current_setting('app.current_user_id')::uuid
    );

CREATE POLICY dashboards_update_policy ON dashboards
    FOR UPDATE
    USING (
        user_id = current_setting('app.current_user_id')::uuid
        OR
        -- Users with edit permission can update
        EXISTS (
            SELECT 1 FROM dashboard_shares
            WHERE dashboard_id = dashboards.id
            AND shared_with_user_id = current_setting('app.current_user_id')::uuid
            AND permissions @> '["edit"]'::jsonb
        )
    );

CREATE POLICY dashboards_delete_policy ON dashboards
    FOR DELETE
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Row Level Security for widgets
ALTER TABLE widgets ENABLE ROW LEVEL SECURITY;

CREATE POLICY widgets_isolation_policy ON widgets
    USING (
        EXISTS (
            SELECT 1 FROM dashboards
            WHERE dashboards.id = widgets.dashboard_id
            AND (
                dashboards.user_id = current_setting('app.current_user_id')::uuid
                OR EXISTS (
                    SELECT 1 FROM dashboard_shares
                    WHERE dashboard_shares.dashboard_id = dashboards.id
                    AND dashboard_shares.shared_with_user_id = current_setting('app.current_user_id')::uuid
                )
                OR (dashboards.is_public = TRUE AND dashboards.organization_id = current_setting('app.current_organization_id')::uuid)
            )
        )
    );

-- Functions for dashboard management
CREATE OR REPLACE FUNCTION update_dashboard_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER dashboards_updated_at_trigger
    BEFORE UPDATE ON dashboards
    FOR EACH ROW
    EXECUTE FUNCTION update_dashboard_updated_at();

CREATE TRIGGER widgets_updated_at_trigger
    BEFORE UPDATE ON widgets
    FOR EACH ROW
    EXECUTE FUNCTION update_dashboard_updated_at();

-- Function to ensure only one default dashboard per user
CREATE OR REPLACE FUNCTION ensure_single_default_dashboard()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.is_default = TRUE THEN
        UPDATE dashboards
        SET is_default = FALSE
        WHERE user_id = NEW.user_id
        AND id != NEW.id
        AND is_default = TRUE;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ensure_single_default_dashboard_trigger
    BEFORE INSERT OR UPDATE OF is_default ON dashboards
    FOR EACH ROW
    WHEN (NEW.is_default = TRUE)
    EXECUTE FUNCTION ensure_single_default_dashboard();

-- Insert default dashboard templates
INSERT INTO dashboard_templates (name, description, category, widgets, tags) VALUES
(
    'Executive Overview',
    'High-level metrics and KPIs for executives',
    'Business',
    '[
        {
            "type": "metric_card",
            "title": "Total Articles Today",
            "size": "small",
            "data_source": {"metric": "article_count", "time_range": "today", "aggregation": "count"}
        },
        {
            "type": "chart_line",
            "title": "Articles Over Time",
            "size": "wide",
            "data_source": {"metric": "article_count", "time_range": "7d", "group_by": "day"}
        },
        {
            "type": "source_stats",
            "title": "Top Sources",
            "size": "medium",
            "data_source": {"limit": 10, "time_range": "7d"}
        }
    ]'::jsonb,
    '["executive", "overview", "metrics"]'::jsonb
),
(
    'Content Analysis',
    'Detailed content analysis and insights',
    'Analytics',
    '[
        {
            "type": "keyword_cloud",
            "title": "Trending Keywords",
            "size": "large",
            "data_source": {"time_range": "24h", "limit": 50}
        },
        {
            "type": "article_list",
            "title": "Recent Articles",
            "size": "wide",
            "data_source": {"query": {}, "limit": 20}
        }
    ]'::jsonb,
    '["content", "analysis", "keywords"]'::jsonb
),
(
    'Alert Monitor',
    'Monitor saved search alerts and notifications',
    'Monitoring',
    '[
        {
            "type": "alert_feed",
            "title": "Recent Alerts",
            "size": "tall",
            "data_source": {"limit": 10}
        },
        {
            "type": "chart_bar",
            "title": "Alerts by Category",
            "size": "medium",
            "data_source": {"metric": "alert_count", "group_by": "category"}
        }
    ]'::jsonb,
    '["alerts", "monitoring", "notifications"]'::jsonb
);