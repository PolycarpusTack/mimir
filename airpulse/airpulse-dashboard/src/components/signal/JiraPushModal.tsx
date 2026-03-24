import React, { useState, useEffect, useCallback } from 'react';
import type { Signal, EnrichedAnnotation } from '@/types';
import { useJiraPush } from '@/api/jira';
import { DomainBadge } from '@/components/primitives/DomainBadge';
import { SignalTypeBadge } from '@/components/primitives/SignalTypeBadge';
import { relevanceColor } from '@/utils';

interface JiraPushModalProps {
  signal: Signal;
  enrichment: EnrichedAnnotation;
  onClose: () => void;
  onSuccess: (issueKey: string) => void;
}

export const JiraPushModal: React.FC<JiraPushModalProps> = ({
  signal,
  enrichment,
  onClose,
  onSuccess,
}) => {
  const [title, setTitle] = useState(
    `[AirPulse] ${signal.signal_type}: ${signal.title.slice(0, 80)}`,
  );
  const [body, setBody] = useState(
    `**Why it matters for WHATS'ON:**\n\n${enrichment.why_it_matters}\n\n---\n\n` +
      (enrichment.action_recommended
        ? `**Recommended action:** ${enrichment.action_recommended}\n\n`
        : '') +
      `**Relevance:** ${enrichment.relevance_score}/100\n` +
      `**Affected spokes:** ${enrichment.affected_spokes.join(', ')}`,
  );
  const [error, setError] = useState<string | null>(null);

  const jiraPush = useJiraPush();

  const spokeLabel = enrichment.affected_spokes[0] ?? 'AirForge';
  const priority =
    enrichment.relevance_score >= 85
      ? 'Critical'
      : enrichment.relevance_score >= 70
        ? 'High'
        : 'Medium';

  const handleConfirm = async () => {
    setError(null);
    try {
      const result = await jiraPush.mutateAsync({
        signal_id: signal.id,
        ticket_title: title,
        ticket_body: body,
        spoke_label: spokeLabel,
        submitted_by: window.location.hostname,
      });
      onSuccess(result.jira_issue_key ?? result.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Push failed');
    }
  };

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Push to JIRA"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0,0,0,0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          width: '600px',
          maxHeight: '80vh',
          backgroundColor: '#0d1117',
          borderLeft: '4px solid #ff7b7b',
          borderRadius: '8px',
          overflow: 'auto',
          padding: '24px',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontFamily: 'DM Serif Display, serif', fontSize: '20px', color: '#e8e8e8', margin: 0 }}>
            Push to JIRA / AirForge
          </h2>
          <button
            onClick={onClose}
            aria-label="Close modal"
            style={{
              background: 'none', border: 'none', color: '#555', fontSize: '20px', cursor: 'pointer',
            }}
          >
            &times;
          </button>
        </div>

        {/* Metadata strip */}
        <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', flexWrap: 'wrap' }}>
          {signal.domains.map((d) => <DomainBadge key={d} domain={d} size="md" />)}
          <SignalTypeBadge signalType={signal.signal_type} size="md" />
          {signal.relevance_score !== null && (
            <span style={{
              fontFamily: 'Syne Mono, monospace', fontSize: '10px', fontWeight: 'bold',
              color: relevanceColor(signal.relevance_score), padding: '3px 8px',
            }}>
              {signal.relevance_score}
            </span>
          )}
          <span style={{
            fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#555',
            padding: '3px 8px', marginLeft: 'auto',
          }}>
            Priority: {priority}
          </span>
        </div>

        {/* Why it matters callout */}
        <div style={{
          borderLeft: '3px solid #e8a830', backgroundColor: 'rgba(232,168,48,0.05)',
          padding: '12px', borderRadius: '0 6px 6px 0', marginBottom: '16px',
        }}>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#e8a830', marginBottom: '6px' }}>
            WHY IT MATTERS
          </div>
          <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '12px', color: '#c0c8d4', lineHeight: 1.5 }}>
            {enrichment.why_it_matters}
          </div>
        </div>

        {/* Editable title */}
        <div style={{ marginBottom: '12px' }}>
          <label style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', display: 'block', marginBottom: '4px' }}>
            TICKET TITLE
          </label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            style={{
              width: '100%', fontFamily: 'Syne, sans-serif', fontSize: '12px',
              backgroundColor: '#111820', border: '1px solid #1e2530', borderRadius: '4px',
              padding: '8px', color: '#d0d0d0',
            }}
          />
        </div>

        {/* Editable body */}
        <div style={{ marginBottom: '16px' }}>
          <label style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', display: 'block', marginBottom: '4px' }}>
            TICKET BODY (MARKDOWN)
          </label>
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            rows={8}
            style={{
              width: '100%', fontFamily: 'Syne Mono, monospace', fontSize: '11px',
              backgroundColor: '#111820', border: '1px solid #1e2530', borderRadius: '4px',
              padding: '8px', color: '#d0d0d0', resize: 'vertical',
            }}
          />
        </div>

        {/* Error message */}
        {error && (
          <div style={{
            fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#ff7b7b',
            padding: '8px', backgroundColor: 'rgba(255,96,96,0.08)', borderRadius: '4px',
            marginBottom: '12px',
          }}>
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
          <button
            onClick={onClose}
            style={{
              fontFamily: 'Syne Mono, monospace', fontSize: '10px', padding: '8px 16px',
              backgroundColor: 'transparent', border: '1px solid #3d4555',
              borderRadius: '4px', color: '#888', cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={jiraPush.isPending}
            style={{
              fontFamily: 'Syne Mono, monospace', fontSize: '10px', padding: '8px 16px',
              background: 'linear-gradient(135deg, #ff7b7b, #ff9f5d)', border: 'none',
              borderRadius: '4px', color: '#0a0e14', cursor: 'pointer', fontWeight: 700,
              opacity: jiraPush.isPending ? 0.6 : 1,
            }}
          >
            {jiraPush.isPending ? 'Pushing...' : 'Confirm & Push to JIRA'}
          </button>
        </div>
      </div>
    </div>
  );
};
