import React from 'react';
import type { Signal } from '@/types';
import { DomainBadge } from '@/components/primitives/DomainBadge';
import { SignalTypeBadge } from '@/components/primitives/SignalTypeBadge';
import { DOMAIN_COLORS } from '@/constants';
import { formatDate, relevanceColor, isFresh } from '@/utils';

interface SignalItemProps {
  signal: Signal;
  selected: boolean;
  onClick: () => void;
}

export const SignalItem: React.FC<SignalItemProps> = ({ signal, selected, onClick }) => {
  const fresh = isFresh(signal.fetched_at);
  const primaryDomain = signal.domains[0];
  const domainColors = primaryDomain ? DOMAIN_COLORS[primaryDomain] : null;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-selected={selected}
      aria-label={`${fresh ? 'New: ' : ''}${signal.title}`}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
      style={{
        borderLeft: selected ? '2px solid #e8a830' : `2px solid ${domainColors?.border ?? '#1e2530'}`,
        backgroundColor: selected
          ? 'rgba(232,168,48,0.06)'
          : fresh
            ? domainColors?.bg ?? 'transparent'
            : 'transparent',
        padding: '11px 14px',
        borderBottom: '1px solid #111820',
        cursor: 'pointer',
        minHeight: '80px',
      }}
    >
      {/* Row 1: badges + timestamp */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '4px', flexWrap: 'wrap' }}>
        {signal.domains.map((d) => <DomainBadge key={d} domain={d} />)}
        <SignalTypeBadge signalType={signal.signal_type} />
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginLeft: 'auto' }}>
          {formatDate(signal.published_at)}
        </span>
        {fresh && (
          <span
            data-testid="fresh-indicator"
            style={{
              width: '6px', height: '6px', borderRadius: '50%',
              backgroundColor: '#5dde9f', boxShadow: '0 0 4px #5dde9f',
              display: 'inline-block',
            }}
          />
        )}
      </div>

      {/* Row 2: title */}
      <div
        style={{
          fontFamily: 'Syne, sans-serif',
          fontWeight: 600,
          fontSize: '12px',
          color: '#d0d0d0',
          lineHeight: 1.4,
          overflow: 'hidden',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          marginBottom: '4px',
        }}
      >
        {signal.title}
      </div>

      {/* Row 3: source + relevance */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#3d4555' }}>
          {signal.source_id}
        </span>
        {signal.relevance_score !== null && (
          <span
            style={{
              fontFamily: 'Syne Mono, monospace',
              fontSize: '9px',
              fontWeight: 700,
              color: relevanceColor(signal.relevance_score),
              marginLeft: 'auto',
            }}
          >
            {signal.relevance_score}
          </span>
        )}
      </div>
    </div>
  );
};
