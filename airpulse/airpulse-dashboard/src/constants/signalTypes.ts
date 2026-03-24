import type { SignalType } from '@/types';

export interface SignalTypeConfig {
  label: string;
  color: string;
}

export const SIGNAL_TYPE_CONFIG: Record<SignalType, SignalTypeConfig> = {
  ProductLaunch:      { label: 'PRODUCT',     color: '#5dde5d' },
  MaSignal:           { label: 'M&A',         color: '#ff7b7b' },
  Partnership:        { label: 'PARTNER',     color: '#5dc8de' },
  StandardsShift:     { label: 'STANDARD',    color: '#ffc95d' },
  CompetitorMove:     { label: 'COMPETITOR',  color: '#c85dde' },
  ClientPressure:     { label: 'CLIENT',      color: '#ff9f5d' },
  TechnologyAdoption: { label: 'TECH SHIFT',  color: '#7b7bff' },
  Regulatory:         { label: 'REGULATORY',  color: '#ff5d7b' },
  TalentMove:         { label: 'TALENT',      color: '#5dde9f' },
  MarketSizing:       { label: 'MARKET DATA', color: '#ffe05d' },
};

export const SIGNAL_TYPES: SignalType[] = [
  'ProductLaunch', 'MaSignal', 'Partnership', 'StandardsShift',
  'CompetitorMove', 'ClientPressure', 'TechnologyAdoption',
  'Regulatory', 'TalentMove', 'MarketSizing',
];
