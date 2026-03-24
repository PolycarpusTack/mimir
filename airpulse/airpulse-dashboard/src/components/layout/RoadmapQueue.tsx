import React from 'react';
import { useSignals } from '@/api/signals';
import { useUIStore } from '@/store/ui';
import { DomainBadge } from '@/components/primitives/DomainBadge';
import { SignalTypeBadge } from '@/components/primitives/SignalTypeBadge';
import { EmptyState } from '@/components/shared/EmptyState';
import { FIXTURE_ENRICHMENTS } from '../../../tests/mocks/fixtures';

export const RoadmapQueue: React.FC = () => {
  const queueExpanded = useUIStore((s) => s.queueExpanded);
  const toggleQueue = useUIStore((s) => s.toggleQueue);
  const { data } = useSignals({ enriched_only: true, page_size: 20 });

  const highRelevance = (data?.items ?? [])
    .filter((s) => s.relevance_score !== null && s.relevance_score >= 70)
    .sort((a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0));

  const handlePush = (signalId: string) => {
    const signal = data?.items.find((s) => s.id === signalId);
    if (signal) {
      window.dispatchEvent(new CustomEvent('airpulse:roadmap-push', { detail: signal }));
    }
  };

  return (
    <div
      style={{
        height: queueExpanded ? '240px' : '40px',
        transition: 'height 0.25s ease',
        backgroundColor: '#0a0e14',
        borderTop: '1px solid #111820',
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div
        role="button"
        tabIndex={0}
        aria-expanded={queueExpanded}
        aria-label="Roadmap signal queue"
        onClick={toggleQueue}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleQueue(); } }}
        style={{
          height: '40px',
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#e8a830', letterSpacing: '1px' }}>
          {queueExpanded ? '\u25BC' : '\u25B2'} ROADMAP SIGNAL QUEUE
        </span>
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginLeft: '12px' }}>
          {highRelevance.length} items
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#555' }}>
          {queueExpanded ? '\u25BC COLLAPSE' : '\u25B2 EXPAND'}
        </span>
      </div>

      {/* Card lane */}
      <div
        aria-label="Roadmap signal queue"
        style={{
          display: 'flex',
          gap: '12px',
          overflowX: 'auto',
          padding: '12px 24px',
          height: '200px',
        }}
      >
        {highRelevance.length === 0 ? (
          <EmptyState message="No high-relevance signals pending" />
        ) : (
          highRelevance.map((signal) => {
            const enrichment = FIXTURE_ENRICHMENTS[signal.id];
            return (
              <div
                key={signal.id}
                style={{
                  minWidth: '260px',
                  maxWidth: '260px',
                  backgroundColor: '#111820',
                  border: '1px solid #1e2530',
                  borderRadius: '6px',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                  {signal.domains.map((d) => <DomainBadge key={d} domain={d} />)}
                  <SignalTypeBadge signalType={signal.signal_type} />
                </div>
                <div style={{
                  fontFamily: 'Syne, sans-serif',
                  fontWeight: 600,
                  fontSize: '10px',
                  color: '#d0d0d0',
                  overflow: 'hidden',
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                }}>
                  {signal.title}
                </div>
                {enrichment?.action_recommended && (
                  <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '10px', color: '#5070a0' }}>
                    {enrichment.action_recommended}
                  </div>
                )}
                {enrichment?.affected_spokes?.[0] && (
                  <span style={{
                    fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#e8a830',
                    backgroundColor: 'rgba(232,168,48,0.1)', padding: '2px 5px', borderRadius: '3px',
                    alignSelf: 'flex-start',
                  }}>
                    {enrichment.affected_spokes[0]}
                  </span>
                )}
                <div style={{ flex: 1 }} />
                <button
                  onClick={(e) => { e.stopPropagation(); handlePush(signal.id); }}
                  style={{
                    fontFamily: 'Syne Mono, monospace',
                    fontSize: '9px',
                    padding: '5px 12px',
                    background: 'linear-gradient(135deg, #e8a830, #ffc95d)',
                    border: 'none',
                    borderRadius: '3px',
                    color: '#0a0e14',
                    cursor: 'pointer',
                    fontWeight: 700,
                    alignSelf: 'flex-end',
                  }}
                >
                  Push &rarr;
                </button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
