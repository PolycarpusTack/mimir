export type Domain = 'Broadcast' | 'OTT' | 'Sports' | 'AI' | 'Cloud' | 'Adtech';

export type SignalType =
  | 'ProductLaunch'
  | 'MaSignal'
  | 'Partnership'
  | 'StandardsShift'
  | 'CompetitorMove'
  | 'ClientPressure'
  | 'TechnologyAdoption'
  | 'Regulatory'
  | 'TalentMove'
  | 'MarketSizing';

export type AirSpoke =
  | 'AirBase'
  | 'AirCast'
  | 'AirOps'
  | 'AirGuard'
  | 'AirPlan'
  | 'AirForge'
  | 'AirPulse'
  | 'AirQC';

export interface KeywordHit {
  keyword: string;
  category: string;
  weight: number;
  position: number;
}

export interface Signal {
  id: string;
  source_id: string;
  url: string;
  title: string;
  summary: string | null;
  published_at: string;
  fetched_at: string;
  content_hash: string;
  domains: Domain[];
  signal_type: SignalType;
  keyword_hits: KeywordHit[];
  confidence_score: number;
  relevance_score: number | null;
  enriched: boolean;
  archived: boolean;
  created_at: string;
}

export interface SignalPage {
  items: Signal[];
  total: number;
  page: number;
  page_size: number;
}
