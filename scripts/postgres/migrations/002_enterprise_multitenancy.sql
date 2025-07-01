-- EPIC 5: Enterprise Features & Multi-tenancy Schema
-- Version: 1.0.0
-- Description: Multi-tenant architecture with organizational structure, RBAC, and row-level security

-- Begin transaction for atomic migration
BEGIN;

-- Create enum types for enterprise features
CREATE TYPE organization_tier AS ENUM ('starter', 'professional', 'enterprise');
CREATE TYPE subscription_status AS ENUM ('active', 'inactive', 'suspended', 'cancelled');
CREATE TYPE user_role AS ENUM ('admin', 'editor', 'viewer', 'analyst');
CREATE TYPE api_key_type AS ENUM ('read_only', 'read_write', 'admin');
CREATE TYPE webhook_event AS ENUM ('article.created', 'article.updated', 'keyword.matched', 'alert.triggered');

-- Organizations table (top-level tenant isolation)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL UNIQUE,
    domain VARCHAR(255), -- Optional domain for SSO
    
    -- Subscription details
    tier organization_tier NOT NULL DEFAULT 'starter',
    subscription_status subscription_status NOT NULL DEFAULT 'active',
    subscription_starts_at TIMESTAMPTZ,
    subscription_ends_at TIMESTAMPTZ,
    
    -- Usage limits based on tier
    max_sources INTEGER DEFAULT 10,
    max_keywords INTEGER DEFAULT 50,
    max_users INTEGER DEFAULT 5,
    max_api_calls_per_hour INTEGER DEFAULT 1000,
    max_storage_gb NUMERIC DEFAULT 1.0,
    
    -- Current usage tracking
    current_sources INTEGER DEFAULT 0,
    current_keywords INTEGER DEFAULT 0,
    current_users INTEGER DEFAULT 0,
    current_storage_gb NUMERIC DEFAULT 0.0,
    
    -- Configuration
    settings JSONB DEFAULT '{}',
    features JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    
    -- Constraints
    CONSTRAINT valid_slug CHECK (slug ~ '^[a-z0-9][a-z0-9-]*[a-z0-9]$'),
    CONSTRAINT valid_subscription_dates CHECK (
        subscription_starts_at IS NULL OR 
        subscription_ends_at IS NULL OR 
        subscription_starts_at < subscription_ends_at
    ),
    CONSTRAINT valid_usage_limits CHECK (
        max_sources >= current_sources AND
        max_keywords >= current_keywords AND
        max_users >= current_users AND
        current_storage_gb >= 0
    )
);

-- Indexes for organizations
CREATE INDEX idx_organizations_slug ON organizations(slug);
CREATE INDEX idx_organizations_tier ON organizations(tier);
CREATE INDEX idx_organizations_status ON organizations(subscription_status);
CREATE INDEX idx_organizations_domain ON organizations(domain) WHERE domain IS NOT NULL;

-- Add organization_id to users table
ALTER TABLE users ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE users ADD COLUMN role user_role NOT NULL DEFAULT 'viewer';
ALTER TABLE users ADD COLUMN permissions JSONB DEFAULT '{}';
ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN invited_by UUID REFERENCES users(id);
ALTER TABLE users ADD COLUMN invited_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN accepted_at TIMESTAMPTZ;

-- Update users indexes
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;

-- Add organization_id to existing tables for tenant isolation
ALTER TABLE articles ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE sources ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE keywords ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE scrape_runs ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;
ALTER TABLE notifications ADD COLUMN organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE;

-- Add indexes for tenant isolation
CREATE INDEX idx_articles_organization ON articles(organization_id);
CREATE INDEX idx_sources_organization ON sources(organization_id);
CREATE INDEX idx_keywords_organization ON keywords(organization_id);
CREATE INDEX idx_scrape_runs_organization ON scrape_runs(organization_id);
CREATE INDEX idx_notifications_organization ON notifications(organization_id);

-- API Keys table for programmatic access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Key details
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(128) NOT NULL UNIQUE, -- SHA-256 hash of the actual key
    key_prefix VARCHAR(8) NOT NULL, -- First 8 chars for identification
    type api_key_type NOT NULL DEFAULT 'read_only',
    
    -- Permissions and limits
    permissions JSONB DEFAULT '{}',
    rate_limit_per_hour INTEGER,
    allowed_ips JSONB DEFAULT '[]', -- Array of allowed IP addresses/ranges
    
    -- Usage tracking
    last_used_at TIMESTAMPTZ,
    usage_count INTEGER DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    
    -- Audit
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id),
    revoked_at TIMESTAMPTZ,
    revoked_by UUID REFERENCES users(id)
);

CREATE INDEX idx_api_keys_organization ON api_keys(organization_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

-- Saved Searches table
CREATE TABLE saved_searches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Search details
    name VARCHAR(255) NOT NULL,
    description TEXT,
    query JSONB NOT NULL, -- Store complex search parameters
    
    -- Alert configuration
    is_alert BOOLEAN DEFAULT false,
    alert_frequency INTEGER DEFAULT 3600, -- seconds between checks
    last_checked_at TIMESTAMPTZ,
    next_check_at TIMESTAMPTZ,
    
    -- Notification settings
    notification_channels JSONB DEFAULT '[]', -- email, slack, webhook, etc.
    notification_settings JSONB DEFAULT '{}',
    
    -- Metadata
    is_shared BOOLEAN DEFAULT false,
    shared_with JSONB DEFAULT '[]', -- Array of user IDs or roles
    tags JSONB DEFAULT '[]',
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_saved_searches_organization ON saved_searches(organization_id);
CREATE INDEX idx_saved_searches_user ON saved_searches(user_id);
CREATE INDEX idx_saved_searches_alerts ON saved_searches(is_alert, next_check_at) WHERE is_alert = true;

-- Webhooks table for external integrations
CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Webhook details
    name VARCHAR(255) NOT NULL,
    url TEXT NOT NULL,
    secret VARCHAR(128), -- For signature verification
    events webhook_event[] NOT NULL,
    
    -- Configuration
    headers JSONB DEFAULT '{}', -- Custom headers to send
    timeout_seconds INTEGER DEFAULT 30,
    retry_attempts INTEGER DEFAULT 3,
    retry_backoff_seconds INTEGER DEFAULT 60,
    
    -- Status and stats
    is_active BOOLEAN DEFAULT true,
    last_triggered_at TIMESTAMPTZ,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    last_error TEXT,
    last_error_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_url CHECK (url ~ '^https?://')
);

CREATE INDEX idx_webhooks_organization ON webhooks(organization_id);
CREATE INDEX idx_webhooks_active ON webhooks(is_active) WHERE is_active = true;

-- Usage tracking table for billing and analytics
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    
    -- Metric details
    metric_type VARCHAR(100) NOT NULL, -- api_calls, storage_gb, articles_scraped, etc.
    metric_value NUMERIC NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Time bucketing for aggregation
    date DATE GENERATED ALWAYS AS (recorded_at::date) STORED,
    hour INTEGER GENERATED ALWAYS AS (EXTRACT(HOUR FROM recorded_at)) STORED,
    
    -- Additional context
    metadata JSONB DEFAULT '{}'
);

-- Partitioning by date for performance
CREATE INDEX idx_usage_metrics_organization_date ON usage_metrics(organization_id, date);
CREATE INDEX idx_usage_metrics_type_date ON usage_metrics(metric_type, date);

-- Activity log for audit trail
CREATE TABLE activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Activity details
    action VARCHAR(100) NOT NULL, -- login, create_article, delete_source, etc.
    resource_type VARCHAR(50), -- article, source, user, etc.
    resource_id UUID,
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    details JSONB DEFAULT '{}',
    
    -- Results
    status VARCHAR(20) DEFAULT 'success', -- success, failure, partial
    error_message TEXT,
    
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Partitioning by date for performance (keep 1 year)
CREATE INDEX idx_activity_log_organization_date ON activity_log(organization_id, created_at);
CREATE INDEX idx_activity_log_user ON activity_log(user_id);
CREATE INDEX idx_activity_log_action ON activity_log(action);

-- Organization invitations table
CREATE TABLE organization_invitations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'viewer',
    
    -- Invitation details
    token VARCHAR(128) NOT NULL UNIQUE,
    invited_by UUID NOT NULL REFERENCES users(id),
    invited_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    
    -- Status
    accepted_at TIMESTAMPTZ,
    declined_at TIMESTAMPTZ,
    
    -- Additional context
    message TEXT,
    permissions JSONB DEFAULT '{}',
    
    CONSTRAINT valid_expiry CHECK (expires_at > invited_at)
);

CREATE INDEX idx_org_invitations_org ON organization_invitations(organization_id);
CREATE INDEX idx_org_invitations_email ON organization_invitations(email);
CREATE INDEX idx_org_invitations_token ON organization_invitations(token);
CREATE INDEX idx_org_invitations_pending ON organization_invitations(expires_at) 
    WHERE accepted_at IS NULL AND declined_at IS NULL;

-- Enable Row Level Security on tenant-isolated tables
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE keywords ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrape_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_searches ENABLE ROW LEVEL SECURITY;
ALTER TABLE webhooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for tenant isolation
-- Articles policies
CREATE POLICY articles_tenant_isolation ON articles
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Sources policies
CREATE POLICY sources_tenant_isolation ON sources
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Keywords policies  
CREATE POLICY keywords_tenant_isolation ON keywords
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Scrape runs policies
CREATE POLICY scrape_runs_tenant_isolation ON scrape_runs
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Notifications policies
CREATE POLICY notifications_tenant_isolation ON notifications
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Saved searches policies
CREATE POLICY saved_searches_tenant_isolation ON saved_searches
    FOR ALL
    TO mimir_user
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND (
            user_id = current_setting('app.current_user_id')::uuid
            OR is_shared = true
            OR current_setting('app.current_user_role')::text IN ('admin', 'editor')
        )
    );

-- Webhooks policies
CREATE POLICY webhooks_tenant_isolation ON webhooks
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Usage metrics policies
CREATE POLICY usage_metrics_tenant_isolation ON usage_metrics
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Activity log policies
CREATE POLICY activity_log_tenant_isolation ON activity_log
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- API keys policies
CREATE POLICY api_keys_tenant_isolation ON api_keys
    FOR ALL
    TO mimir_user
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

-- Create helper functions for multi-tenancy

-- Function to set tenant context
CREATE OR REPLACE FUNCTION set_tenant_context(
    org_id UUID,
    user_id UUID DEFAULT NULL,
    user_role TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_organization_id', org_id::text, true);
    IF user_id IS NOT NULL THEN
        PERFORM set_config('app.current_user_id', user_id::text, true);
    END IF;
    IF user_role IS NOT NULL THEN
        PERFORM set_config('app.current_user_role', user_role, true);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get current tenant context
CREATE OR REPLACE FUNCTION get_tenant_context()
RETURNS TABLE (
    organization_id UUID,
    user_id UUID,
    user_role TEXT
) AS $$
BEGIN
    RETURN QUERY SELECT
        COALESCE(current_setting('app.current_organization_id', true), '')::uuid,
        COALESCE(current_setting('app.current_user_id', true), '')::uuid,
        COALESCE(current_setting('app.current_user_role', true), '');
END;
$$ LANGUAGE plpgsql;

-- Function to check organization limits
CREATE OR REPLACE FUNCTION check_organization_limits(
    org_id UUID,
    limit_type TEXT
)
RETURNS BOOLEAN AS $$
DECLARE
    org_record organizations%ROWTYPE;
    current_count INTEGER;
BEGIN
    SELECT * INTO org_record FROM organizations WHERE id = org_id;
    
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    CASE limit_type
        WHEN 'sources' THEN
            RETURN org_record.current_sources < org_record.max_sources;
        WHEN 'keywords' THEN
            RETURN org_record.current_keywords < org_record.max_keywords;
        WHEN 'users' THEN
            RETURN org_record.current_users < org_record.max_users;
        ELSE
            RETURN FALSE;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Function to update organization usage counters
CREATE OR REPLACE FUNCTION update_organization_usage()
RETURNS TRIGGER AS $$
DECLARE
    org_id UUID;
    delta INTEGER;
BEGIN
    -- Determine organization ID and delta
    IF TG_OP = 'INSERT' THEN
        org_id := NEW.organization_id;
        delta := 1;
    ELSIF TG_OP = 'DELETE' THEN
        org_id := OLD.organization_id;
        delta := -1;
    ELSE
        RETURN COALESCE(NEW, OLD);
    END IF;
    
    -- Update appropriate counter based on table
    CASE TG_TABLE_NAME
        WHEN 'sources' THEN
            UPDATE organizations 
            SET current_sources = current_sources + delta
            WHERE id = org_id;
        WHEN 'keywords' THEN
            UPDATE organizations 
            SET current_keywords = current_keywords + delta
            WHERE id = org_id;
        WHEN 'users' THEN
            UPDATE organizations 
            SET current_users = current_users + delta
            WHERE id = org_id;
    END CASE;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers to maintain usage counters
CREATE TRIGGER sources_usage_counter
    AFTER INSERT OR DELETE ON sources
    FOR EACH ROW EXECUTE FUNCTION update_organization_usage();

CREATE TRIGGER keywords_usage_counter
    AFTER INSERT OR DELETE ON keywords
    FOR EACH ROW EXECUTE FUNCTION update_organization_usage();

CREATE TRIGGER users_usage_counter
    AFTER INSERT OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION update_organization_usage();

-- Update the update_updated_at trigger for new tables
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_saved_searches_updated_at BEFORE UPDATE ON saved_searches
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_webhooks_updated_at BEFORE UPDATE ON webhooks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create default organization for existing data migration
DO $$
DECLARE
    default_org_id UUID;
BEGIN
    -- Create default organization
    INSERT INTO organizations (
        name, 
        slug, 
        tier, 
        subscription_status,
        max_sources,
        max_keywords, 
        max_users,
        max_api_calls_per_hour,
        max_storage_gb
    ) VALUES (
        'Default Organization',
        'default',
        'enterprise',
        'active',
        1000,
        1000,
        100,
        10000,
        100.0
    ) RETURNING id INTO default_org_id;
    
    -- Update existing records to belong to default organization
    UPDATE articles SET organization_id = default_org_id WHERE organization_id IS NULL;
    UPDATE sources SET organization_id = default_org_id WHERE organization_id IS NULL;
    UPDATE keywords SET organization_id = default_org_id WHERE organization_id IS NULL;
    UPDATE scrape_runs SET organization_id = default_org_id WHERE organization_id IS NULL;
    UPDATE notifications SET organization_id = default_org_id WHERE organization_id IS NULL;
    UPDATE users SET organization_id = default_org_id WHERE organization_id IS NULL;
    
    -- Update usage counters for default organization
    UPDATE organizations SET
        current_sources = (SELECT COUNT(*) FROM sources WHERE organization_id = default_org_id),
        current_keywords = (SELECT COUNT(*) FROM keywords WHERE organization_id = default_org_id),
        current_users = (SELECT COUNT(*) FROM users WHERE organization_id = default_org_id)
    WHERE id = default_org_id;
END $$;

-- Make organization_id required for existing tables
ALTER TABLE articles ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE sources ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE keywords ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE scrape_runs ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE notifications ALTER COLUMN organization_id SET NOT NULL;
ALTER TABLE users ALTER COLUMN organization_id SET NOT NULL;

-- Grant permissions on new tables
GRANT ALL ON organizations TO mimir_user;
GRANT ALL ON api_keys TO mimir_user;
GRANT ALL ON saved_searches TO mimir_user;
GRANT ALL ON webhooks TO mimir_user;
GRANT ALL ON usage_metrics TO mimir_user;
GRANT ALL ON activity_log TO mimir_user;
GRANT ALL ON organization_invitations TO mimir_user;

-- Grant sequence permissions
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO mimir_user;

COMMIT;