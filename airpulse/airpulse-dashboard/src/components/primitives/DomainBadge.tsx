import React from 'react';
import type { Domain } from '@/types';
import { DOMAIN_COLORS } from '@/constants';

interface DomainBadgeProps {
  domain: Domain;
  size?: 'sm' | 'md';
}

export const DomainBadge: React.FC<DomainBadgeProps> = ({ domain, size = 'sm' }) => {
  const colors = DOMAIN_COLORS[domain];
  const fontSize = size === 'sm' ? '8px' : '10px';
  const padding = size === 'sm' ? '2px 5px' : '3px 8px';

  return (
    <span
      style={{
        display: 'inline-block',
        fontFamily: 'Syne Mono, monospace',
        fontSize,
        padding,
        borderRadius: '3px',
        backgroundColor: colors.bg,
        border: `1px solid ${colors.border}`,
        color: colors.text,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        lineHeight: 1,
      }}
    >
      {domain}
    </span>
  );
};
