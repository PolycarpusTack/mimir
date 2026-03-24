-- V013: Trade show events table with seed data (Phase 5, §4.4)

CREATE TABLE IF NOT EXISTS airpulse.trade_show_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    location        TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    domains         TEXT[] NOT NULL DEFAULT '{}',
    significance    TEXT NOT NULL DEFAULT 'Standard'
                    CHECK (significance IN ('Major', 'Standard', 'Minor')),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trade_show_dates
    ON airpulse.trade_show_events(start_date, end_date);

CREATE INDEX IF NOT EXISTS idx_trade_show_domains
    ON airpulse.trade_show_events USING GIN(domains);

-- Seed data: 8 key broadcast industry events (Phase 4 Appendix B)
INSERT INTO airpulse.trade_show_events (name, location, start_date, end_date, domains, significance, notes)
VALUES
    ('IBC',
     'Amsterdam, Netherlands',
     '2026-09-11', '2026-09-14',
     ARRAY['Broadcast', 'OTT', 'Cloud', 'AI'],
     'Major',
     'International Broadcasting Convention — largest European broadcast technology event'),

    ('NAB Show',
     'Las Vegas, USA',
     '2026-04-18', '2026-04-22',
     ARRAY['Broadcast', 'OTT', 'Cloud', 'AI', 'Sports'],
     'Major',
     'National Association of Broadcasters annual convention — largest US broadcast event'),

    ('Broadcast Asia',
     'Singapore',
     '2026-06-02', '2026-06-04',
     ARRAY['Broadcast', 'OTT', 'Cloud'],
     'Standard',
     'Leading Asia-Pacific broadcasting and media technology event'),

    ('SMPTE',
     'Los Angeles, USA',
     '2026-10-19', '2026-10-22',
     ARRAY['Broadcast', 'Cloud', 'AI'],
     'Standard',
     'Society of Motion Picture and Television Engineers annual conference'),

    ('SVG Summit',
     'New York, USA',
     '2026-11-16', '2026-11-18',
     ARRAY['Sports', 'Broadcast', 'OTT'],
     'Standard',
     'Sports Video Group summit — premier sports production technology event'),

    ('SportsPro OTT Summit',
     'Madrid, Spain',
     '2026-11-23', '2026-11-25',
     ARRAY['Sports', 'OTT', 'Adtech'],
     'Standard',
     'Leading sports OTT and streaming business conference'),

    ('Streaming Media Europe',
     'London, UK',
     '2026-10-05', '2026-10-06',
     ARRAY['OTT', 'Cloud', 'Adtech'],
     'Minor',
     'European streaming and online video conference'),

    ('DVB World',
     'Brussels, Belgium',
     '2026-03-10', '2026-03-12',
     ARRAY['Broadcast'],
     'Minor',
     'Digital Video Broadcasting standards and technology conference');
