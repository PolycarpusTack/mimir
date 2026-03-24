export type CircuitState = 'Closed' | 'Open' | 'HalfOpen';

export interface FeedSource {
  id: string;
  name: string;
  tier: 'Tier1' | 'Tier2' | 'Tier3' | 'Tier4' | 'Tier5';
  circuit_state: CircuitState;
  consecutive_failures: number;
  last_successful_poll: string | null;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'critical';
  sources_live: number;
  sources_open: number;
  queue_depth: number;
  signals_today: number;
  surges_active: number;
}

export interface StatsResponse {
  total_signals: number;
  signals_per_type: [string, number][];
  dedup_rate: number;
  polls_last_hour: number;
  errors_last_hour: number;
}

export interface EnrichmentQueueStatus {
  depth: number;
  inflight: number;
  rate_per_min: number;
}
