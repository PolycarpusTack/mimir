import type { AirSpoke, SignalType } from './signal';

export interface EnrichedAnnotation {
  why_it_matters: string;
  relevance_score: number;
  affected_spokes: AirSpoke[];
  vendor_mentions: string[];
  technology_terms: string[];
  client_mentions: string[];
  action_recommended: string | null;
  confidence_override: number | null;
  signal_type_override: SignalType | null;
  enriched_at: string;
  prompt_version: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
}
