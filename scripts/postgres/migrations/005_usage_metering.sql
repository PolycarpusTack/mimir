-- Migration: Add Usage Metering System
-- Description: Create tables for tracking API usage, storage, and feature utilization

-- Create usage records table
CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    metric VARCHAR(50) NOT NULL,
    value DECIMAL(20, 4) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    resource_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT usage_records_metric_check CHECK (metric IN (
        'api_calls', 'storage_bytes', 'articles_processed', 'ai_analysis',
        'searches', 'saved_searches', 'dashboard_views', 'notifications_sent',
        'webhooks', 'data_exports', 'users_active', 'custom_models'
    ))
);

-- Create usage summaries table for performance
CREATE TABLE IF NOT EXISTS usage_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    period_type VARCHAR(20) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',
    total_cost DECIMAL(10, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT usage_summaries_period_check CHECK (period_type IN ('hourly', 'daily', 'weekly', 'monthly', 'yearly')),
    CONSTRAINT unique_org_period UNIQUE (organization_id, period_type, period_start)
);

-- Create billing history table
CREATE TABLE IF NOT EXISTS billing_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    tier VARCHAR(50) NOT NULL,
    base_cost DECIMAL(10, 2) NOT NULL,
    overage_cost DECIMAL(10, 2) NOT NULL DEFAULT 0,
    total_cost DECIMAL(10, 2) NOT NULL,
    usage_summary JSONB NOT NULL,
    invoice_number VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    paid_at TIMESTAMPTZ,
    payment_method VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT billing_history_status_check CHECK (status IN ('pending', 'paid', 'overdue', 'cancelled', 'refunded'))
);

-- Create usage alerts table
CREATE TABLE IF NOT EXISTS usage_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    metric VARCHAR(50) NOT NULL,
    threshold_value DECIMAL(20, 4) NOT NULL,
    threshold_type VARCHAR(20) NOT NULL,
    alert_channels JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    last_triggered_at TIMESTAMPTZ,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT usage_alerts_threshold_check CHECK (threshold_type IN ('absolute', 'percentage'))
);

-- Create feature usage table for detailed tracking
CREATE TABLE IF NOT EXISTS feature_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    feature_name VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    duration_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Index for common queries
    INDEX idx_feature_usage_org_feature (organization_id, feature_name, created_at),
    INDEX idx_feature_usage_user (user_id, created_at)
);

-- Indexes for performance
CREATE INDEX idx_usage_records_org_metric ON usage_records(organization_id, metric, timestamp);
CREATE INDEX idx_usage_records_user ON usage_records(user_id, timestamp) WHERE user_id IS NOT NULL;
CREATE INDEX idx_usage_records_timestamp ON usage_records(timestamp);
CREATE INDEX idx_usage_records_resource ON usage_records(resource_id) WHERE resource_id IS NOT NULL;

CREATE INDEX idx_usage_summaries_org_period ON usage_summaries(organization_id, period_type, period_start);

CREATE INDEX idx_billing_history_org ON billing_history(organization_id, period_start);
CREATE INDEX idx_billing_history_status ON billing_history(status) WHERE status != 'paid';

CREATE INDEX idx_usage_alerts_org ON usage_alerts(organization_id) WHERE is_active = TRUE;

-- Row Level Security
ALTER TABLE usage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE billing_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_usage ENABLE ROW LEVEL SECURITY;

-- RLS Policies for usage records (admins only)
CREATE POLICY usage_records_admin_policy ON usage_records
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

-- RLS Policies for usage summaries
CREATE POLICY usage_summaries_read_policy ON usage_summaries
    FOR SELECT
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') IN ('admin', 'editor')
    );

-- RLS Policies for billing history
CREATE POLICY billing_history_admin_policy ON billing_history
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

-- RLS Policies for usage alerts
CREATE POLICY usage_alerts_admin_policy ON usage_alerts
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

-- RLS Policies for feature usage (own usage visible)
CREATE POLICY feature_usage_own_policy ON feature_usage
    FOR SELECT
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND (
            user_id = current_setting('app.current_user_id')::uuid
            OR current_setting('app.current_user_role') IN ('admin', 'editor')
        )
    );

-- Functions
CREATE OR REPLACE FUNCTION calculate_usage_cost(
    p_organization_id UUID,
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
) RETURNS TABLE (
    metric VARCHAR,
    usage_value DECIMAL,
    included_value DECIMAL,
    overage_value DECIMAL,
    unit_cost DECIMAL,
    total_cost DECIMAL
) AS $$
DECLARE
    v_tier VARCHAR;
    v_tier_config JSONB;
BEGIN
    -- Get organization tier
    SELECT tier INTO v_tier
    FROM organizations
    WHERE id = p_organization_id;
    
    -- Get tier configuration (would be from a config table in production)
    -- For now, using hardcoded values
    v_tier_config := CASE v_tier
        WHEN 'starter' THEN '{"api_calls": {"included": 100000, "rate": 0.001}}'::jsonb
        WHEN 'professional' THEN '{"api_calls": {"included": 500000, "rate": 0.0008}}'::jsonb
        WHEN 'enterprise' THEN '{"api_calls": {"included": 2000000, "rate": 0.0005}}'::jsonb
        ELSE '{"api_calls": {"included": 10000, "rate": 0}}'::jsonb
    END;
    
    -- Calculate costs for each metric
    RETURN QUERY
    SELECT 
        ur.metric,
        SUM(ur.value) as usage_value,
        COALESCE((v_tier_config->ur.metric->>'included')::decimal, 0) as included_value,
        GREATEST(SUM(ur.value) - COALESCE((v_tier_config->ur.metric->>'included')::decimal, 0), 0) as overage_value,
        COALESCE((v_tier_config->ur.metric->>'rate')::decimal, 0) as unit_cost,
        GREATEST(SUM(ur.value) - COALESCE((v_tier_config->ur.metric->>'included')::decimal, 0), 0) * 
            COALESCE((v_tier_config->ur.metric->>'rate')::decimal, 0) as total_cost
    FROM usage_records ur
    WHERE ur.organization_id = p_organization_id
        AND ur.timestamp >= p_start_date
        AND ur.timestamp < p_end_date
    GROUP BY ur.metric;
END;
$$ LANGUAGE plpgsql;

-- Function to aggregate usage data
CREATE OR REPLACE FUNCTION aggregate_usage_data(
    p_period_type VARCHAR DEFAULT 'daily'
) RETURNS VOID AS $$
DECLARE
    v_period_start TIMESTAMPTZ;
    v_period_end TIMESTAMPTZ;
    v_trunc_func TEXT;
BEGIN
    -- Determine truncation function based on period type
    v_trunc_func := CASE p_period_type
        WHEN 'hourly' THEN 'hour'
        WHEN 'daily' THEN 'day'
        WHEN 'weekly' THEN 'week'
        WHEN 'monthly' THEN 'month'
        WHEN 'yearly' THEN 'year'
        ELSE 'day'
    END;
    
    -- Get the period to aggregate
    v_period_end := DATE_TRUNC(v_trunc_func, CURRENT_TIMESTAMP);
    v_period_start := v_period_end - INTERVAL '1 ' || v_trunc_func;
    
    -- Insert aggregated data
    INSERT INTO usage_summaries (organization_id, period_start, period_end, period_type, metrics)
    SELECT 
        organization_id,
        v_period_start,
        v_period_end,
        p_period_type,
        jsonb_object_agg(metric, total_value) as metrics
    FROM (
        SELECT 
            organization_id,
            metric,
            SUM(value) as total_value
        FROM usage_records
        WHERE timestamp >= v_period_start
            AND timestamp < v_period_end
        GROUP BY organization_id, metric
    ) t
    GROUP BY organization_id
    ON CONFLICT (organization_id, period_type, period_start)
    DO UPDATE SET 
        metrics = EXCLUDED.metrics,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updating timestamps
CREATE TRIGGER usage_summaries_updated_at_trigger
    BEFORE UPDATE ON usage_summaries
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER usage_alerts_updated_at_trigger
    BEFORE UPDATE ON usage_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

-- Add usage tracking columns to organizations table
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS usage_limits JSONB DEFAULT '{}';
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS billing_email VARCHAR(255);
ALTER TABLE organizations ADD COLUMN IF NOT EXISTS billing_cycle_day INTEGER DEFAULT 1 CHECK (billing_cycle_day >= 1 AND billing_cycle_day <= 28);

-- Sample data for testing (commented out for production)
-- INSERT INTO usage_records (organization_id, metric, value, user_id) VALUES
-- ((SELECT id FROM organizations LIMIT 1), 'api_calls', 100, (SELECT id FROM users LIMIT 1)),
-- ((SELECT id FROM organizations LIMIT 1), 'articles_processed', 50, (SELECT id FROM users LIMIT 1)),
-- ((SELECT id FROM organizations LIMIT 1), 'searches', 25, (SELECT id FROM users LIMIT 1));