import type { Domain } from '@/types';

export interface DomainColorSet {
  bg: string;
  border: string;
  text: string;
  dot: string;
}

export const DOMAIN_COLORS: Record<Domain, DomainColorSet> = {
  Broadcast: { bg: '#1a2e1a', border: '#2d5a2d', text: '#5dde5d', dot: '#3dba3d' },
  OTT:       { bg: '#1a1a2e', border: '#2d2d5a', text: '#7b7bff', dot: '#5555dd' },
  Sports:    { bg: '#2e1a1a', border: '#5a2d2d', text: '#ff7b7b', dot: '#dd5555' },
  AI:        { bg: '#2e2a1a', border: '#5a502d', text: '#ffc95d', dot: '#e8a830' },
  Cloud:     { bg: '#1a2a2e', border: '#2d4f5a', text: '#5dc8de', dot: '#3aabcc' },
  Adtech:    { bg: '#2a1a2e', border: '#502d5a', text: '#c85dde', dot: '#a83acc' },
};

export const DOMAINS: Domain[] = ['Broadcast', 'OTT', 'Sports', 'AI', 'Cloud', 'Adtech'];
