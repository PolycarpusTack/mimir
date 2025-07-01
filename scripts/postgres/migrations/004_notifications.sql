-- Migration: Add Notification System
-- Description: Create tables for multi-channel notification system

-- Create notification templates table
CREATE TABLE IF NOT EXISTS notification_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    notification_type VARCHAR(100) NOT NULL,
    subject_template TEXT,
    body_template TEXT NOT NULL,
    variables JSONB DEFAULT '[]'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notification_templates_channel_check CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'push')),
    CONSTRAINT unique_template_name UNIQUE (organization_id, name)
);

-- Create notification log table
CREATE TABLE IF NOT EXISTS notification_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    channel VARCHAR(50) NOT NULL,
    recipient VARCHAR(500) NOT NULL,
    notification_type VARCHAR(100) NOT NULL,
    subject TEXT,
    body TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority VARCHAR(50) NOT NULL DEFAULT 'medium',
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notification_log_channel_check CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'push')),
    CONSTRAINT notification_log_status_check CHECK (status IN ('pending', 'sent', 'failed', 'retry')),
    CONSTRAINT notification_log_priority_check CHECK (priority IN ('low', 'medium', 'high', 'critical'))
);

-- Create notification preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    channel VARCHAR(50) NOT NULL,
    notification_type VARCHAR(100) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notification_preferences_channel_check CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'push')),
    CONSTRAINT unique_user_preference UNIQUE (user_id, channel, notification_type)
);

-- Create notification channel configurations table
CREATE TABLE IF NOT EXISTS notification_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    channel VARCHAR(50) NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    configuration JSONB NOT NULL,
    rate_limit INTEGER, -- Messages per hour
    retry_count INTEGER DEFAULT 3,
    retry_delay INTEGER DEFAULT 60, -- seconds
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notification_channels_channel_check CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'push')),
    CONSTRAINT unique_org_channel UNIQUE (organization_id, channel)
);

-- Create notification queue table for async processing
CREATE TABLE IF NOT EXISTS notification_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    channel VARCHAR(50) NOT NULL,
    recipient VARCHAR(500) NOT NULL,
    notification_type VARCHAR(100) NOT NULL,
    template_id UUID REFERENCES notification_templates(id),
    data JSONB NOT NULL,
    priority VARCHAR(50) NOT NULL DEFAULT 'medium',
    scheduled_for TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    attempts INTEGER DEFAULT 0,
    last_attempt_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT notification_queue_channel_check CHECK (channel IN ('email', 'slack', 'teams', 'webhook', 'sms', 'push')),
    CONSTRAINT notification_queue_priority_check CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT notification_queue_status_check CHECK (status IN ('queued', 'processing', 'sent', 'failed', 'cancelled'))
);

-- Indexes for performance
CREATE INDEX idx_notification_templates_org ON notification_templates(organization_id);
CREATE INDEX idx_notification_templates_type ON notification_templates(notification_type);

CREATE INDEX idx_notification_log_org ON notification_log(organization_id);
CREATE INDEX idx_notification_log_recipient ON notification_log(recipient);
CREATE INDEX idx_notification_log_created ON notification_log(created_at);
CREATE INDEX idx_notification_log_status ON notification_log(status);

CREATE INDEX idx_notification_preferences_user ON notification_preferences(user_id);
CREATE INDEX idx_notification_preferences_type ON notification_preferences(notification_type);

CREATE INDEX idx_notification_queue_status ON notification_queue(status, scheduled_for);
CREATE INDEX idx_notification_queue_priority ON notification_queue(priority, scheduled_for);

-- Row Level Security
ALTER TABLE notification_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_channels ENABLE ROW LEVEL SECURITY;

-- RLS Policies for notification templates
CREATE POLICY notification_templates_isolation_policy ON notification_templates
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

CREATE POLICY notification_templates_insert_policy ON notification_templates
    FOR INSERT
    WITH CHECK (organization_id = current_setting('app.current_organization_id')::uuid);

-- RLS Policies for notification preferences
CREATE POLICY notification_preferences_isolation_policy ON notification_preferences
    USING (
        user_id = current_setting('app.current_user_id')::uuid
        OR EXISTS (
            SELECT 1 FROM users
            WHERE users.id = notification_preferences.user_id
            AND users.organization_id = current_setting('app.current_organization_id')::uuid
            AND current_setting('app.current_user_role') IN ('admin', 'editor')
        )
    );

CREATE POLICY notification_preferences_insert_policy ON notification_preferences
    FOR INSERT
    WITH CHECK (
        user_id = current_setting('app.current_user_id')::uuid
        OR current_setting('app.current_user_role') = 'admin'
    );

-- RLS Policies for notification channels
CREATE POLICY notification_channels_isolation_policy ON notification_channels
    USING (organization_id = current_setting('app.current_organization_id')::uuid);

CREATE POLICY notification_channels_modify_policy ON notification_channels
    FOR ALL
    USING (
        organization_id = current_setting('app.current_organization_id')::uuid
        AND current_setting('app.current_user_role') = 'admin'
    );

-- Functions and triggers
CREATE OR REPLACE FUNCTION update_notification_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER notification_templates_updated_at_trigger
    BEFORE UPDATE ON notification_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER notification_preferences_updated_at_trigger
    BEFORE UPDATE ON notification_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

CREATE TRIGGER notification_channels_updated_at_trigger
    BEFORE UPDATE ON notification_channels
    FOR EACH ROW
    EXECUTE FUNCTION update_notification_updated_at();

-- Function to process notification queue
CREATE OR REPLACE FUNCTION process_notification_queue()
RETURNS TABLE (
    notification_id UUID,
    processed BOOLEAN
) AS $$
DECLARE
    notification RECORD;
BEGIN
    -- Get notifications ready to be sent
    FOR notification IN
        SELECT * FROM notification_queue
        WHERE status = 'queued'
          AND scheduled_for <= CURRENT_TIMESTAMP
        ORDER BY priority DESC, scheduled_for ASC
        LIMIT 100
        FOR UPDATE SKIP LOCKED
    LOOP
        -- Mark as processing
        UPDATE notification_queue
        SET status = 'processing',
            attempts = attempts + 1,
            last_attempt_at = CURRENT_TIMESTAMP
        WHERE id = notification.id;
        
        -- Return for processing (actual sending done in application)
        notification_id := notification.id;
        processed := TRUE;
        RETURN NEXT;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Insert default notification templates
INSERT INTO notification_templates (organization_id, name, channel, notification_type, subject_template, body_template) VALUES
-- Email templates
((SELECT id FROM organizations LIMIT 1), 'saved_search_alert_email', 'email', 'saved_search_alert',
 'Alert: {{ alert_name }} - {{ result_count }} new results',
 '<h2>Your saved search alert has new results</h2>
  <p><strong>Alert:</strong> {{ alert_name }}</p>
  <p><strong>New Results:</strong> {{ result_count }}</p>
  <p><strong>Time:</strong> {{ timestamp }}</p>
  <h3>Top Results:</h3>
  <ul>
  {% for result in results[:5] %}
    <li>
      <strong>{{ result.title }}</strong><br>
      Source: {{ result.source }}<br>
      Date: {{ result.published_date }}
    </li>
  {% endfor %}
  </ul>'),

-- Slack templates  
((SELECT id FROM organizations LIMIT 1), 'saved_search_alert_slack', 'slack', 'saved_search_alert',
 NULL,
 ':bell: *Alert: {{ alert_name }}*\n\n{{ result_count }} new results found\n\n*Top Results:*\n{% for result in results[:3] %}\n• {{ result.title }} ({{ result.source }})\n{% endfor %}'),

-- Teams templates
((SELECT id FROM organizations LIMIT 1), 'saved_search_alert_teams', 'teams', 'saved_search_alert',
 NULL,
 '## Alert: {{ alert_name }}\n\n**{{ result_count }}** new results found\n\n### Top Results:\n{% for result in results[:3] %}\n* **{{ result.title }}**\n  * Source: {{ result.source }}\n  * Date: {{ result.published_date }}\n{% endfor %}')

ON CONFLICT (organization_id, name) DO NOTHING;