import React from 'react';
import { useSignal, useSignalEnrichment } from '@/api/signals';
import { DomainBadge } from '@/components/primitives/DomainBadge';
import { SignalTypeBadge } from '@/components/primitives/SignalTypeBadge';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { EmptyState } from '@/components/shared/EmptyState';
import { formatDate, relevanceColor } from '@/utils';
import { ApiError } from '@/api/client';

interface SignalDetailProps {
  signalId: string | null;
}

export const SignalDetail: React.FC<SignalDetailProps> = ({ signalId }) => {
  const { data: signal } = useSignal(signalId);
  const { data: enrichment, isLoading: enrichLoading, error: enrichError } = useSignalEnrichment(signalId);

  if (!signalId) {
    return (
      <div style={{ width: '360px', borderRight: '1px solid #111820', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <EmptyState message="Select a signal" />
      </div>
    );
  }

  const isEnrichmentPending = enrichError instanceof ApiError && enrichError.status === 404;

  const handlePush = () => {
    if (signal) {
      window.dispatchEvent(new CustomEvent('airpulse:roadmap-push', { detail: signal }));
    }
  };

  const handleArchive = () => {
    if (signal) {
      fetch(`/api/v1/signals/${signal.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ archived: true }),
      });
    }
  };

  return (
    <div style={{ width: '360px', borderRight: '1px solid #111820', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        height: '40px', display: 'flex', alignItems: 'center', padding: '0 14px',
        borderBottom: '1px solid #111820', flexShrink: 0,
      }}>
        <span style={{ fontFamily: 'DM Serif Display, serif', fontSize: '14px', color: '#d0d0d0' }}>Enrichment</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
        {!signal ? (
          <LoadingSkeleton height={200} />
        ) : (
          <>
            {/* Badges + timestamp */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '6px', flexWrap: 'wrap' }}>
              {signal.domains.map((d) => <DomainBadge key={d} domain={d} size="md" />)}
              <SignalTypeBadge signalType={signal.signal_type} size="md" />
            </div>
            <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginBottom: '12px' }}>
              {formatDate(signal.published_at)} &middot; {signal.source_id}
            </div>

            {/* Title */}
            <h2 style={{
              fontFamily: 'DM Serif Display, serif', fontSize: '20px', color: '#e8e8e8',
              lineHeight: 1.35, margin: '0 0 16px 0',
              overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical',
            }}>
              {signal.title}
            </h2>

            {/* Summary */}
            {signal.summary && (
              <p style={{ fontFamily: 'Syne, sans-serif', fontSize: '12px', color: '#8899aa', lineHeight: 1.5, marginBottom: '16px' }}>
                {signal.summary}
              </p>
            )}

            {/* Enrichment section */}
            {enrichLoading ? (
              <LoadingSkeleton height={120} />
            ) : isEnrichmentPending ? (
              <div style={{
                padding: '16px', backgroundColor: 'rgba(255,255,255,0.03)', border: '1px solid #1e2530',
                borderRadius: '6px', textAlign: 'center',
              }}>
                <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#555' }}>
                  Enrichment pending...
                </div>
              </div>
            ) : enrichment ? (
              <>
                {/* AI Enrichment callout */}
                <div style={{
                  borderLeft: '3px solid #e8a830', backgroundColor: 'rgba(232,168,48,0.05)',
                  padding: '12px', borderRadius: '0 6px 6px 0', marginBottom: '16px',
                }}>
                  <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#e8a830', marginBottom: '8px', letterSpacing: '0.5px' }}>
                    AI ENRICHMENT &mdash; WHY IT MATTERS FOR WHATS&apos;ON
                  </div>
                  <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '13px', color: '#c0c8d4', lineHeight: 1.5 }}>
                    {enrichment.why_it_matters}
                  </div>
                </div>

                {/* Stats */}
                <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
                  <div style={{
                    flex: 1, backgroundColor: '#111820', borderRadius: '6px', padding: '12px', textAlign: 'center',
                  }}>
                    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', textTransform: 'uppercase' }}>
                      Relevance Score
                    </div>
                    <div style={{
                      fontFamily: 'Syne Mono, monospace', fontSize: '28px', fontWeight: 'bold',
                      color: relevanceColor(enrichment.relevance_score),
                    }}>
                      {enrichment.relevance_score}
                    </div>
                  </div>
                  <div style={{
                    flex: 1, backgroundColor: '#111820', borderRadius: '6px', padding: '12px',
                  }}>
                    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', textTransform: 'uppercase', marginBottom: '6px' }}>
                      Affected Spokes
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      {enrichment.affected_spokes.map((spoke) => (
                        <span key={spoke} style={{
                          fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#e8a830',
                          backgroundColor: 'rgba(232,168,48,0.1)', padding: '2px 5px', borderRadius: '3px',
                        }}>
                          {spoke}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Action recommended */}
                {enrichment.action_recommended && (
                  <div style={{
                    padding: '10px', backgroundColor: 'rgba(93,200,222,0.06)', border: '1px solid rgba(93,200,222,0.2)',
                    borderRadius: '6px', marginBottom: '16px',
                  }}>
                    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#5dc8de', marginBottom: '4px' }}>
                      RECOMMENDED ACTION
                    </div>
                    <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '11px', color: '#8899aa' }}>
                      {enrichment.action_recommended}
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                  <button
                    onClick={handlePush}
                    style={{
                      fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '7px 14px',
                      background: 'linear-gradient(135deg, #e8a830, #ffc95d)', border: 'none',
                      borderRadius: '4px', color: '#0a0e14', cursor: 'pointer', fontWeight: 700,
                    }}
                  >
                    Push to JIRA / AirForge
                  </button>
                  <button
                    style={{
                      fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '7px 14px',
                      backgroundColor: 'transparent', border: '1px solid #5dc8de',
                      borderRadius: '4px', color: '#5dc8de', cursor: 'pointer',
                    }}
                  >
                    Add to Digest
                  </button>
                  <button
                    onClick={handleArchive}
                    style={{
                      fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '7px 14px',
                      backgroundColor: 'transparent', border: '1px solid #3d4555',
                      borderRadius: '4px', color: '#888', cursor: 'pointer',
                    }}
                  >
                    Archive
                  </button>
                </div>

                {/* Tag clouds */}
                {enrichment.vendor_mentions.length > 0 && (
                  <TagCloud label="VENDOR MENTIONS" tags={enrichment.vendor_mentions} color="#c85dde" />
                )}
                {enrichment.technology_terms.length > 0 && (
                  <TagCloud label="TECHNOLOGY TERMS" tags={enrichment.technology_terms} color="#5dc8de" />
                )}
                {enrichment.client_mentions.length > 0 && (
                  <TagCloud label="CLIENT MENTIONS" tags={enrichment.client_mentions} color="#ffc95d" />
                )}
              </>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
};

const TagCloud: React.FC<{ label: string; tags: string[]; color: string }> = ({ label, tags, color }) => (
  <div style={{ marginBottom: '12px' }}>
    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', marginBottom: '6px', letterSpacing: '0.5px' }}>
      {label}
    </div>
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
      {tags.map((tag) => (
        <span key={tag} style={{
          fontFamily: 'Syne Mono, monospace', fontSize: '9px', color,
          backgroundColor: `${color}15`, padding: '2px 6px', borderRadius: '3px',
          border: `1px solid ${color}30`,
        }}>
          {tag}
        </span>
      ))}
    </div>
  </div>
);
