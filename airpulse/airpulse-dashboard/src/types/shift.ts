import type { Domain, SignalType } from './signal';

export type ShiftSeverity = 'Elevated' | 'Spike' | 'Surge';

export interface ShiftAlert {
  id: string;
  domain: Domain;
  signal_type: SignalType;
  z_score: number;
  severity: ShiftSeverity;
  observed: number;
  baseline_mean: number;
  baseline_std: number;
  signal_count: number;
  label: string;
  fired_at: string;
  cooldown_until: string;
  resolved_at: string | null;
}
