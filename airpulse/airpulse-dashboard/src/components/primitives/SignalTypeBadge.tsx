import React from 'react';
import type { SignalType } from '@/types';
import { SIGNAL_TYPE_CONFIG } from '@/constants';

interface SignalTypeBadgeProps {
  signalType: SignalType;
  size?: 'sm' | 'md';
}

export const SignalTypeBadge: React.FC<SignalTypeBadgeProps> = ({ signalType, size = 'sm' }) => {
  const config = SIGNAL_TYPE_CONFIG[signalType];
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
        backgroundColor: `${config.color}15`,
        border: `1px solid ${config.color}40`,
        color: config.color,
        letterSpacing: '0.5px',
        lineHeight: 1,
      }}
    >
      {config.label}
    </span>
  );
};
