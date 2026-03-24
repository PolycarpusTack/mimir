// src/types/digest.ts — Phase 4 digest types
export interface DigestSummary {
  id: string;
  week_starting: string;
  generated_at: string;
  signal_count: number;
  total_tokens: number;
}

export interface DigestDetail extends DigestSummary {
  markdown: string | null;
}
