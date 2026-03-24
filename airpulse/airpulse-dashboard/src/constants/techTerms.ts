export interface TechTermEntry {
  term: string;
  cluster: string;
}

export const TECH_TERM_WATCHLIST: TechTermEntry[] = [
  { term: 'ATSC 3.0',          cluster: 'Standards' },
  { term: 'DVB-I',             cluster: 'Standards' },
  { term: 'FAST',              cluster: 'Distribution' },
  { term: 'AVOD',              cluster: 'Distribution' },
  { term: 'GenAI',             cluster: 'AI/ML' },
  { term: 'LLM',              cluster: 'AI/ML' },
  { term: 'Computer Vision',   cluster: 'AI/ML' },
  { term: 'Content Graph',     cluster: 'AI/ML' },
  { term: 'SSAI',             cluster: 'Adtech' },
  { term: 'Clean Room',        cluster: 'Adtech' },
  { term: 'SRT',              cluster: 'Transport' },
  { term: 'RIST',             cluster: 'Transport' },
  { term: 'CDN',              cluster: 'Infrastructure' },
  { term: 'Edge Computing',    cluster: 'Infrastructure' },
  { term: 'Kubernetes',        cluster: 'Infrastructure' },
  { term: 'Cloud Playout',     cluster: 'Operations' },
  { term: 'Remote Production', cluster: 'Operations' },
  { term: 'NDI',              cluster: 'Production' },
  { term: 'Virtual LED',      cluster: 'Production' },
  { term: 'Immersive Audio',   cluster: 'Experience' },
];
