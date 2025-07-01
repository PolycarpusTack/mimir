-- Migration: Add Subscription Management
-- Description: Create tables for subscription plans, feature flags, and billing

-- Create subscription plans table (for custom plans)
CREATE TABLE IF NOT EXISTS subscription_plans (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    tier VARCHAR(50) NOT NULL,
    price_monthly DECIMAL(10, 2) NOT NULL,
    price_yearly DECIMAL(10, 2) NOT NULL,
    trial_days INTEGER DEFAULT 0,
    features JSONB NOT NULL DEFAULT '{}',
    usage_limits JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT subscription_plans_tier_check CHECK (tier IN ('free', 'starter', 'professional', 'enterprise', 'custom'))
);

-- Create subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    plan_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    current_period_start TIMESTAMPTZ NOT NULL,
    current_period_end TIMESTAMPTZ NOT NULL,
    trial_end TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    custom_features JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT subscriptions_status_check CHECK (status IN ('active', 'trial', 'suspended', 'cancelled', 'expired', 'pending')),
    CONSTRAINT one_active_subscription UNIQUE (organization_id, status) WHERE status IN ('active', 'trial')
);

-- Create feature definitions table
CREATE TABLE IF NOT EXISTS feature_definitions (
    name VARCHAR(100) PRIMARY KEY,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    feature_type VARCHAR(20) NOT NULL,
    default_value JSONB,
    available_tiers TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT feature_definitions_type_check CHECK (feature_type IN ('boolean', 'numeric', 'list', 'custom'))
);

-- Create feature overrides table (for custom feature values per organization)
CREATE TABLE IF NOT EXISTS feature_overrides (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    feature_name VARCHAR(100) NOT NULL REFERENCES feature_definitions(name),
    value JSONB NOT NULL,
    reason TEXT,
    expires_at TIMESTAMPTZ,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT unique_org_feature UNIQUE (organization_id, feature_name)
);

-- Create subscription history table
CREATE TABLE IF NOT EXISTS subscription_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    subscription_id UUID REFERENCES subscriptions(id),
    action VARCHAR(50) NOT NULL,
    old_plan_id VARCHAR(50),
    new_plan_id VARCHAR(50),
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    metadata JSONB DEFAULT '{}',
    performed_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT subscription_history_action_check CHECK (action IN ('created', 'upgraded', 'downgraded', 'renewed', 'cancelled', 'suspended', 'reactivated', 'expired'))
);

-- Create usage limit enforcements table
CREATE TABLE IF NOT EXISTS usage_limit_enforcements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    metric VARCHAR(50) NOT NULL,
    limit_value DECIMAL(20, 4) NOT NULL,
    enforcement_type VARCHAR(20) NOT NULL,
    grace_period_days INTEGER DEFAULT 0,
    notifications_sent INTEGER DEFAULT 0,
    last_notified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT usage_limit_enforcements_type_check CHECK (enforcement_type IN ('hard', 'soft', 'notify_only')),
    CONSTRAINT unique_org_metric_limit UNIQUE (organization_id, metric)
);

-- Indexes for performance
CREATE INDEX idx_subscriptions_org ON subscriptions(organization_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status) WHERE status IN ('active', 'trial');
CREATE INDEX idx_subscriptions_trial_end ON subscriptions(trial_end) WHERE trial_end IS NOT NULL;
CREATE INDEX idx_subscriptions_period_end ON subscriptions(current_period_end);

CREATE INDEX idx_feature_overrides_org ON feature_overrides(organization_id);
CREATE INDEX idx_feature_overrides_expires ON feature_overrides(expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX idx_subscription_history_org ON subscription_history(organization_id, created_at);

-- Row Level Security
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_overrides ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscription_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_limit_enforcements ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY subscriptions_admin_policy ON subscriptions
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

CREATE POLICY feature_overrides_read_policy ON feature_overrides
    FOR SELECT
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

CREATE POLICY feature_overrides_admin_policy ON feature_overrides
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

CREATE POLICY subscription_history_read_policy ON subscription_history
    FOR SELECT
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') IN ('admin', 'editor')
    );

CREATE POLICY usage_limit_enforcements_admin_policy ON usage_limit_enforcements
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

-- Functions
CREATE OR REPLACE FUNCTION check_feature_access(
    p_organization_id UUID,
    p_feature_name VARCHAR
) RETURNS TABLE (
    has_access BOOLEAN,
    feature_value JSONB,
    source VARCHAR
) AS $$
DECLARE
    v_subscription RECORD;
    v_plan JSONB;
    v_feature RECORD;
    v_override RECORD;
BEGIN
    -- Get active subscription
    SELECT * INTO v_subscription
    FROM subscriptions
    WHERE organization_id = p_organization_id
      AND status IN ('active', 'trial')
    LIMIT 1;
    
    -- Get feature definition
    SELECT * INTO v_feature
    FROM feature_definitions
    WHERE name = p_feature_name;
    
    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, NULL::jsonb, 'not_found'::varchar;
        RETURN;
    END IF;
    
    -- Check for overrides first
    SELECT * INTO v_override
    FROM feature_overrides
    WHERE organization_id = p_organization_id
      AND feature_name = p_feature_name
      AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP);
    
    IF FOUND THEN
        RETURN QUERY SELECT TRUE, v_override.value, 'override'::varchar;
        RETURN;
    END IF;
    
    -- Check subscription features
    IF v_subscription.id IS NOT NULL THEN
        -- Check custom features
        IF v_subscription.custom_features ? p_feature_name THEN
            RETURN QUERY SELECT TRUE, v_subscription.custom_features->p_feature_name, 'subscription_custom'::varchar;
            RETURN;
        END IF;
        
        -- Check plan features (would need to load from plan configuration)
        -- For now, return default based on tier
        IF v_subscription.plan_id = ANY(v_feature.available_tiers) THEN
            RETURN QUERY SELECT TRUE, v_feature.default_value, 'plan'::varchar;
            RETURN;
        END IF;
    END IF;
    
    -- No access
    RETURN QUERY SELECT FALSE, v_feature.default_value, 'no_access'::varchar;
END;
$$ LANGUAGE plpgsql;

-- Function to enforce usage limits
CREATE OR REPLACE FUNCTION enforce_usage_limit(
    p_organization_id UUID,
    p_metric VARCHAR,
    p_current_value DECIMAL,
    p_additional_value DECIMAL DEFAULT 1
) RETURNS TABLE (
    allowed BOOLEAN,
    limit_value DECIMAL,
    enforcement_type VARCHAR,
    message TEXT
) AS $$
DECLARE
    v_subscription RECORD;
    v_limit RECORD;
    v_projected_value DECIMAL;
BEGIN
    -- Get active subscription
    SELECT * INTO v_subscription
    FROM subscriptions
    WHERE organization_id = p_organization_id
      AND status IN ('active', 'trial')
    LIMIT 1;
    
    -- Get limit enforcement
    SELECT * INTO v_limit
    FROM usage_limit_enforcements
    WHERE organization_id = p_organization_id
      AND metric = p_metric;
    
    IF NOT FOUND THEN
        -- No specific enforcement, allow
        RETURN QUERY SELECT TRUE, NULL::decimal, 'none'::varchar, NULL::text;
        RETURN;
    END IF;
    
    v_projected_value := p_current_value + p_additional_value;
    
    -- Check against limit
    IF v_projected_value <= v_limit.limit_value THEN
        RETURN QUERY SELECT TRUE, v_limit.limit_value, v_limit.enforcement_type, NULL::text;
    ELSE
        -- Over limit
        IF v_limit.enforcement_type = 'hard' THEN
            RETURN QUERY SELECT FALSE, v_limit.limit_value, v_limit.enforcement_type, 
                'Usage limit exceeded for ' || p_metric;
        ELSIF v_limit.enforcement_type = 'soft' THEN
            -- Allow but notify
            RETURN QUERY SELECT TRUE, v_limit.limit_value, v_limit.enforcement_type,
                'Usage limit exceeded for ' || p_metric || ' (grace period)';
        ELSE
            -- notify_only
            RETURN QUERY SELECT TRUE, v_limit.limit_value, v_limit.enforcement_type,
                'Usage limit exceeded for ' || p_metric || ' (notification only)';
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER subscriptions_updated_at_trigger
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER subscription_plans_updated_at_trigger
    BEFORE UPDATE ON subscription_plans
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER feature_definitions_updated_at_trigger
    BEFORE UPDATE ON feature_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER feature_overrides_updated_at_trigger
    BEFORE UPDATE ON feature_overrides
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

-- Insert default feature definitions
INSERT INTO feature_definitions (name, display_name, description, feature_type, default_value, available_tiers) VALUES
-- Boolean features
('advanced_search', 'Advanced Search', 'Advanced search with filters and operators', 'boolean', 'false', ARRAY['starter', 'professional', 'enterprise']),
('semantic_search', 'Semantic Search', 'AI-powered semantic search', 'boolean', 'false', ARRAY['professional', 'enterprise']),
('saved_searches', 'Saved Searches', 'Save and manage search queries', 'boolean', 'false', ARRAY['starter', 'professional', 'enterprise']),
('search_alerts', 'Search Alerts', 'Get notified when new results match saved searches', 'boolean', 'false', ARRAY['professional', 'enterprise']),
('ai_analysis', 'AI Analysis', 'AI-powered content analysis', 'boolean', 'false', ARRAY['starter', 'professional', 'enterprise']),
('dashboards', 'Custom Dashboards', 'Create custom dashboards', 'boolean', 'false', ARRAY['professional', 'enterprise']),
('api_access', 'API Access', 'REST API access', 'boolean', 'false', ARRAY['starter', 'professional', 'enterprise']),
('webhooks', 'Webhooks', 'Webhook notifications', 'boolean', 'false', ARRAY['professional', 'enterprise']),
('sso', 'Single Sign-On', 'SSO authentication', 'boolean', 'false', ARRAY['enterprise']),

-- Numeric features
('max_users', 'Maximum Users', 'Maximum number of users', 'numeric', '3', ARRAY['all']),
('max_saved_searches', 'Maximum Saved Searches', 'Maximum number of saved searches', 'numeric', '5', ARRAY['all']),
('max_dashboards', 'Maximum Dashboards', 'Maximum number of dashboards', 'numeric', '1', ARRAY['all']),
('data_retention_days', 'Data Retention', 'Days of data retention', 'numeric', '30', ARRAY['all'])

ON CONFLICT (name) DO NOTHING;