import type { Signal, SignalPage, EnrichedAnnotation, ShiftAlert, FeedSource, HealthResponse, EnrichmentQueueStatus, StatsResponse, DigestSummary, DigestDetail, ApprovalQueueItem } from '@/types';

const now = new Date().toISOString();
const recent = new Date(Date.now() - 60_000).toISOString();

export const FIXTURE_SIGNALS: Signal[] = [
  {
    id: 'sig-001', source_id: 'src-001', url: 'https://example.com/1',
    title: 'Harmonic launches cloud-native playout platform for FAST channels',
    summary: 'Harmonic unveiled its new VOS360 cloud playout platform optimized for FAST channel delivery.',
    published_at: recent, fetched_at: recent, content_hash: 'hash001',
    domains: ['Broadcast', 'Cloud'], signal_type: 'ProductLaunch',
    keyword_hits: [{ keyword: 'cloud playout', category: 'technology', weight: 0.95, position: 12 }],
    confidence_score: 0.92, relevance_score: 88, enriched: true, archived: false, created_at: recent,
  },
  {
    id: 'sig-002', source_id: 'src-002', url: 'https://example.com/2',
    title: 'Comcast acquires streaming analytics firm in $2.1B deal',
    summary: 'Comcast has agreed to acquire DataStream Analytics in a move to bolster its streaming capabilities.',
    published_at: now, fetched_at: now, content_hash: 'hash002',
    domains: ['OTT'], signal_type: 'MaSignal',
    keyword_hits: [{ keyword: 'acquisition', category: 'corporate', weight: 0.88, position: 5 }],
    confidence_score: 0.95, relevance_score: 92, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-003', source_id: 'src-003', url: 'https://example.com/3',
    title: 'FIFA partners with AWS on AI-driven broadcast analytics',
    summary: 'FIFA and AWS announce a multi-year partnership to deliver AI-powered match analytics.',
    published_at: now, fetched_at: now, content_hash: 'hash003',
    domains: ['Sports', 'AI'], signal_type: 'Partnership',
    keyword_hits: [{ keyword: 'AI analytics', category: 'technology', weight: 0.91, position: 20 }],
    confidence_score: 0.89, relevance_score: 78, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-004', source_id: 'src-001', url: 'https://example.com/4',
    title: 'ATSC 3.0 adoption reaches 60% of US broadcast stations',
    summary: 'The rollout of ATSC 3.0 NextGen TV has crossed a critical milestone with 60% coverage.',
    published_at: now, fetched_at: now, content_hash: 'hash004',
    domains: ['Broadcast'], signal_type: 'StandardsShift',
    keyword_hits: [{ keyword: 'ATSC 3.0', category: 'standards', weight: 0.97, position: 0 }],
    confidence_score: 0.94, relevance_score: 85, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-005', source_id: 'src-004', url: 'https://example.com/5',
    title: 'Imagine Communications releases SaaS ad insertion platform',
    summary: 'Imagine Communications launches a fully managed SSAI service targeting Tier 1 broadcasters.',
    published_at: now, fetched_at: now, content_hash: 'hash005',
    domains: ['Adtech'], signal_type: 'CompetitorMove',
    keyword_hits: [{ keyword: 'SSAI', category: 'adtech', weight: 0.90, position: 35 }],
    confidence_score: 0.87, relevance_score: 75, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-006', source_id: 'src-005', url: 'https://example.com/6',
    title: 'Disney demands real-time content availability SLAs from all vendors',
    summary: 'Disney is renegotiating vendor contracts with strict real-time availability requirements.',
    published_at: now, fetched_at: now, content_hash: 'hash006',
    domains: ['OTT', 'Cloud'], signal_type: 'ClientPressure',
    keyword_hits: [{ keyword: 'SLA', category: 'business', weight: 0.82, position: 50 }],
    confidence_score: 0.78, relevance_score: 65, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-007', source_id: 'src-006', url: 'https://example.com/7',
    title: 'GenAI transforms automated metadata tagging in broadcast workflows',
    summary: 'Broadcasters are rapidly adopting GenAI models for automated content tagging.',
    published_at: now, fetched_at: now, content_hash: 'hash007',
    domains: ['AI'], signal_type: 'TechnologyAdoption',
    keyword_hits: [{ keyword: 'GenAI', category: 'technology', weight: 0.96, position: 0 }],
    confidence_score: 0.91, relevance_score: 82, enriched: true, archived: false, created_at: now,
  },
  {
    id: 'sig-008', source_id: 'src-007', url: 'https://example.com/8',
    title: 'EU Digital Services Act mandates new content moderation standards for broadcasters',
    summary: null,
    published_at: now, fetched_at: now, content_hash: 'hash008',
    domains: ['Broadcast'], signal_type: 'Regulatory',
    keyword_hits: [{ keyword: 'regulation', category: 'regulatory', weight: 0.85, position: 3 }],
    confidence_score: 0.80, relevance_score: null, enriched: false, archived: false, created_at: now,
  },
  {
    id: 'sig-009', source_id: 'src-008', url: 'https://example.com/9',
    title: 'Vizrt CTO departs for Netflix streaming infrastructure role',
    summary: 'Vizrt loses its Chief Technology Officer to Netflix in a high-profile talent move.',
    published_at: now, fetched_at: now, content_hash: 'hash009',
    domains: ['Sports'], signal_type: 'TalentMove',
    keyword_hits: [{ keyword: 'Vizrt', category: 'vendor', weight: 0.88, position: 0 }],
    confidence_score: 0.75, relevance_score: null, enriched: false, archived: false, created_at: now,
  },
  {
    id: 'sig-010', source_id: 'src-002', url: 'https://example.com/10',
    title: 'Global FAST channel market projected to reach $12B by 2028',
    summary: 'New market sizing report projects exponential growth in FAST channel revenues.',
    published_at: now, fetched_at: now, content_hash: 'hash010',
    domains: ['Cloud', 'OTT'], signal_type: 'MarketSizing',
    keyword_hits: [{ keyword: 'FAST', category: 'distribution', weight: 0.93, position: 7 }],
    confidence_score: 0.86, relevance_score: null, enriched: false, archived: false, created_at: now,
  },
];

export const FIXTURE_SIGNAL_PAGE: SignalPage = {
  items: FIXTURE_SIGNALS,
  total: 10,
  page: 1,
  page_size: 50,
};

export const FIXTURE_ENRICHMENTS: Record<string, EnrichedAnnotation> = {
  'sig-001': {
    why_it_matters: 'Harmonic\'s cloud-native playout directly competes with WHATS\'ON AirCast scheduling. Their FAST optimization may attract Tier 1 broadcasters currently evaluating AirCast.',
    relevance_score: 88, affected_spokes: ['AirCast', 'AirOps'],
    vendor_mentions: ['Harmonic'], technology_terms: ['Cloud Playout', 'FAST'],
    client_mentions: [], action_recommended: 'Schedule competitive analysis of VOS360 feature set vs AirCast',
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 420, output_tokens: 280,
  },
  'sig-002': {
    why_it_matters: 'Comcast acquiring streaming analytics strengthens their vertical integration. This could reduce demand for third-party analytics and affect AirBase data partnerships.',
    relevance_score: 92, affected_spokes: ['AirBase', 'AirPlan'],
    vendor_mentions: ['Comcast', 'DataStream Analytics'], technology_terms: ['Streaming Analytics'],
    client_mentions: ['Comcast'], action_recommended: 'Assess impact on AirBase analytics roadmap',
    confidence_override: 0.96, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 380, output_tokens: 310,
  },
  'sig-003': {
    why_it_matters: 'FIFA-AWS partnership in AI analytics could set new expectations for sports broadcast intelligence. AirPulse should monitor for feature parity with AWS offerings.',
    relevance_score: 78, affected_spokes: ['AirPulse', 'AirGuard'],
    vendor_mentions: ['AWS Elemental', 'FIFA'], technology_terms: ['AI', 'Match Analytics'],
    client_mentions: ['FIFA'], action_recommended: null,
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 400, output_tokens: 260,
  },
  'sig-004': {
    why_it_matters: 'ATSC 3.0 reaching 60% coverage is a tipping point. WHATS\'ON must ensure full ATSC 3.0 scheduling support in AirCast before the next NAB cycle.',
    relevance_score: 85, affected_spokes: ['AirCast', 'AirBase', 'AirPlan'],
    vendor_mentions: [], technology_terms: ['ATSC 3.0', 'NextGen TV'],
    client_mentions: [], action_recommended: 'Prioritize ATSC 3.0 metadata support in AirCast Q2 sprint',
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 350, output_tokens: 240,
  },
  'sig-005': {
    why_it_matters: 'Imagine Communications entering managed SSAI creates direct competition with AirForge ad insertion. Their Tier 1 broadcaster targeting overlaps with WHATS\'ON key accounts.',
    relevance_score: 75, affected_spokes: ['AirForge', 'AirOps'],
    vendor_mentions: ['Imagine Communications'], technology_terms: ['SSAI', 'Ad Insertion'],
    client_mentions: [], action_recommended: 'Brief sales team on competitive positioning',
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 390, output_tokens: 270,
  },
  'sig-006': {
    why_it_matters: 'Disney demanding real-time SLAs signals market shift toward zero-downtime expectations. AirOps monitoring capabilities should be front-and-center in renewal pitches.',
    relevance_score: 65, affected_spokes: ['AirOps', 'AirGuard', 'AirQC'],
    vendor_mentions: ['Disney'], technology_terms: ['SLA', 'Real-time Monitoring'],
    client_mentions: ['Disney'], action_recommended: null,
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 360, output_tokens: 250,
  },
  'sig-007': {
    why_it_matters: 'GenAI metadata tagging is becoming table stakes. AirBase content graph should integrate LLM-powered tagging before competitors claim this as a differentiator.',
    relevance_score: 82, affected_spokes: ['AirBase', 'AirPulse', 'AirForge'],
    vendor_mentions: [], technology_terms: ['GenAI', 'Metadata Tagging', 'LLM', 'Content Graph'],
    client_mentions: [], action_recommended: 'Evaluate GenAI tagging for AirBase content graph',
    confidence_override: null, signal_type_override: null,
    enriched_at: now, prompt_version: 'v2', model: 'claude-sonnet-4-6',
    input_tokens: 410, output_tokens: 290,
  },
};

export const FIXTURE_SHIFT_ALERTS: ShiftAlert[] = [
  {
    id: 'shift-001', domain: 'AI', signal_type: 'ProductLaunch',
    z_score: 3.7, severity: 'Surge', observed: 18, baseline_mean: 5.2, baseline_std: 3.4,
    signal_count: 18, label: 'AI ProductLaunch surge (z=3.7\u03C3)',
    fired_at: now, cooldown_until: new Date(Date.now() + 8 * 3600_000).toISOString(), resolved_at: null,
  },
  {
    id: 'shift-002', domain: 'OTT', signal_type: 'MaSignal',
    z_score: 2.8, severity: 'Spike', observed: 12, baseline_mean: 4.1, baseline_std: 2.8,
    signal_count: 12, label: 'OTT M&A spike (z=2.8\u03C3)',
    fired_at: now, cooldown_until: new Date(Date.now() + 4 * 3600_000).toISOString(), resolved_at: null,
  },
  {
    id: 'shift-003', domain: 'Broadcast', signal_type: 'StandardsShift',
    z_score: 1.8, severity: 'Elevated', observed: 8, baseline_mean: 3.5, baseline_std: 2.5,
    signal_count: 8, label: 'Broadcast StandardsShift elevated (z=1.8\u03C3)',
    fired_at: now, cooldown_until: new Date(Date.now() + 2 * 3600_000).toISOString(), resolved_at: null,
  },
  {
    id: 'shift-004', domain: 'Cloud', signal_type: 'Partnership',
    z_score: 2.1, severity: 'Elevated', observed: 6, baseline_mean: 2.1, baseline_std: 1.9,
    signal_count: 6, label: 'Cloud Partnership elevated (z=2.1\u03C3)',
    fired_at: new Date(Date.now() - 86400_000).toISOString(),
    cooldown_until: new Date(Date.now() - 80000_000).toISOString(),
    resolved_at: new Date(Date.now() - 43200_000).toISOString(),
  },
];

export const FIXTURE_SOURCES: FeedSource[] = [
  { id: 'src-001', name: 'TechCrunch', tier: 'Tier1', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-002', name: 'Broadcasting & Cable', tier: 'Tier1', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-003', name: 'Rapid TV News', tier: 'Tier2', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-004', name: 'Advanced Television', tier: 'Tier2', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-005', name: 'Variety', tier: 'Tier1', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-006', name: 'The Verge', tier: 'Tier2', circuit_state: 'HalfOpen', consecutive_failures: 2, last_successful_poll: new Date(Date.now() - 3600_000).toISOString() },
  { id: 'src-007', name: 'SportsPro Media', tier: 'Tier2', circuit_state: 'Closed', consecutive_failures: 0, last_successful_poll: now },
  { id: 'src-008', name: 'AdExchanger', tier: 'Tier2', circuit_state: 'Open', consecutive_failures: 5, last_successful_poll: new Date(Date.now() - 7200_000).toISOString() },
];

export const FIXTURE_HEALTH: HealthResponse = {
  status: 'ok',
  sources_live: 6,
  sources_open: 1,
  queue_depth: 12,
  signals_today: 47,
  surges_active: 3,
};

export const FIXTURE_HEALTH_DEGRADED: HealthResponse = {
  status: 'degraded',
  sources_live: 4,
  sources_open: 3,
  queue_depth: 45,
  signals_today: 31,
  surges_active: 1,
};

export const FIXTURE_QUEUE: EnrichmentQueueStatus = {
  depth: 12,
  inflight: 3,
  rate_per_min: 8,
};

export const FIXTURE_STATS: StatsResponse = {
  total_signals: 1524,
  signals_per_type: [
    ['ProductLaunch', 342], ['MaSignal', 198], ['Partnership', 215],
    ['StandardsShift', 156], ['CompetitorMove', 134], ['ClientPressure', 89],
    ['TechnologyAdoption', 167], ['Regulatory', 78], ['TalentMove', 56], ['MarketSizing', 89],
  ],
  dedup_rate: 0.18,
  polls_last_hour: 42,
  errors_last_hour: 2,
};

// ---------------------------------------------------------------------------
// Phase 4 fixtures
// ---------------------------------------------------------------------------

export const FIXTURE_DIGESTS: DigestSummary[] = [
  {
    id: 'digest-001',
    week_starting: '2026-03-16',
    generated_at: now,
    signal_count: 47,
    total_tokens: 3200,
  },
  {
    id: 'digest-002',
    week_starting: '2026-03-09',
    generated_at: new Date(Date.now() - 7 * 86400_000).toISOString(),
    signal_count: 38,
    total_tokens: 2800,
  },
  {
    id: 'digest-003',
    week_starting: '2026-03-02',
    generated_at: new Date(Date.now() - 14 * 86400_000).toISOString(),
    signal_count: 42,
    total_tokens: 3100,
  },
];

export const FIXTURE_DIGEST_DETAIL: DigestDetail = {
  ...FIXTURE_DIGESTS[0],
  markdown: `# AirPulse Weekly Intelligence Digest\n**Week of 2026-03-16** | Generated ${now} | 47 signals analysed\n\n---\n\n## This Week's Shift Signals\nAI product launches surged this week with z=3.7, driven by...\n\n## Competitor Moves\nComcast's $2.1B acquisition signals consolidation...\n\n## Technology on the Rise\nGenAI metadata tagging adoption accelerated...\n\n## Roadmap Implications\nATSC 3.0 reaching 60% coverage requires AirCast prioritization...\n\n## Watch Next Week\nNAB Show preparations intensifying...`,
};

export const FIXTURE_APPROVAL_QUEUE: ApprovalQueueItem[] = [
  {
    id: 'aq-001',
    signal_id: 'sig-001',
    status: 'Pending',
    jira_project: 'AIRFORGE',
    jira_issue_key: null,
    ticket_title: '[AirPulse] PRODUCT: Harmonic launches cloud-native playout',
    ticket_body: 'Signal body',
    spoke_label: 'AirCast',
    domain_label: 'Broadcast',
    submitted_by: 'hive-dev-01',
    submitted_at: now,
    pushed_at: null,
    attempts: 0,
    last_error: null,
  },
  {
    id: 'aq-002',
    signal_id: 'sig-002',
    status: 'Pushed',
    jira_project: 'AIRFORGE',
    jira_issue_key: 'AIRFORGE-142',
    ticket_title: '[AirPulse] M&A: Comcast acquires streaming analytics firm',
    ticket_body: 'Signal body',
    spoke_label: 'AirBase',
    domain_label: 'OTT',
    submitted_by: 'hive-dev-01',
    submitted_at: new Date(Date.now() - 3600_000).toISOString(),
    pushed_at: new Date(Date.now() - 3500_000).toISOString(),
    attempts: 1,
    last_error: null,
  },
  {
    id: 'aq-003',
    signal_id: 'sig-003',
    status: 'Failed',
    jira_project: 'AIRFORGE',
    jira_issue_key: null,
    ticket_title: '[AirPulse] PARTNER: FIFA partners with AWS on AI analytics',
    ticket_body: 'Signal body',
    spoke_label: 'AirPulse',
    domain_label: 'Sports',
    submitted_by: 'hive-dev-01',
    submitted_at: new Date(Date.now() - 7200_000).toISOString(),
    pushed_at: null,
    attempts: 3,
    last_error: 'Atlassian API error: HTTP 500 — Internal Server Error',
  },
];
