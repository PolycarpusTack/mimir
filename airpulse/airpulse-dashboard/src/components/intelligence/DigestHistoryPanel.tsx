import React, { useState } from 'react';
import { useDigestLatest, useDigestList, useDigest, useGenerateDigest } from '@/api/digest';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { EmptyState } from '@/components/shared/EmptyState';

export const DigestHistoryPanel: React.FC = () => {
  const { data: latest, isLoading: latestLoading } = useDigestLatest();
  const { data: digests } = useDigestList(10);
  const generateDigest = useGenerateDigest();
  const [viewingId, setViewingId] = useState<string | null>(null);
  const [showMarkdown, setShowMarkdown] = useState(false);
  const { data: detail } = useDigest(viewingId);

  if (latestLoading) {
    return <div style={{ padding: '16px' }}><LoadingSkeleton height={300} /></div>;
  }

  return (
    <div id="panel-digest" role="tabpanel" style={{ padding: '16px', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <h2 style={{ fontFamily: 'DM Serif Display, serif', fontSize: '16px', color: '#d0d0d0', margin: 0 }}>
          Weekly Digest
        </h2>
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444' }}>
          Next: Monday 07:00 CET
        </span>
      </div>

      {/* Latest digest callout */}
      {latest ? (
        <div style={{
          backgroundColor: '#111820', border: '1px solid #1e2530', borderLeft: '3px solid #ff7b7b',
          borderRadius: '0 6px 6px 0', padding: '14px', marginBottom: '16px',
        }}>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#ff7b7b', marginBottom: '6px' }}>
            LATEST DIGEST
          </div>
          <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '13px', color: '#d0d0d0', marginBottom: '4px' }}>
            Week of {latest.week_starting}
          </div>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#555', marginBottom: '12px' }}>
            {latest.signal_count} signals analysed &middot; {latest.total_tokens} tokens &middot; {new Date(latest.generated_at).toLocaleDateString()}
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <a
              href={`/api/v1/digest/${latest.id}/download`}
              style={{
                fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '5px 12px',
                backgroundColor: 'transparent', border: '1px solid #ff7b7b',
                borderRadius: '3px', color: '#ff7b7b', textDecoration: 'none', cursor: 'pointer',
              }}
            >
              Download DOCX
            </a>
            <button
              onClick={() => {
                setViewingId(latest.id);
                setShowMarkdown(!showMarkdown);
              }}
              style={{
                fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '5px 12px',
                backgroundColor: 'transparent', border: '1px solid #5dc8de',
                borderRadius: '3px', color: '#5dc8de', cursor: 'pointer',
              }}
            >
              {showMarkdown ? 'Hide Markdown' : 'View Markdown'}
            </button>
          </div>
        </div>
      ) : (
        <EmptyState message="No digests generated yet" />
      )}

      {/* Markdown viewer */}
      {showMarkdown && detail?.markdown && (
        <div style={{
          backgroundColor: '#0a0e14', border: '1px solid #1e2530', borderRadius: '6px',
          padding: '16px', marginBottom: '16px', maxHeight: '400px', overflowY: 'auto',
          fontFamily: 'Syne Mono, monospace', fontSize: '11px', color: '#c0c8d4',
          lineHeight: 1.6, whiteSpace: 'pre-wrap',
        }}>
          {detail.markdown}
        </div>
      )}

      {/* Generate button */}
      <button
        onClick={() => generateDigest.mutate()}
        disabled={generateDigest.isPending}
        style={{
          fontFamily: 'Syne Mono, monospace', fontSize: '9px', padding: '6px 14px',
          background: 'linear-gradient(135deg, #ff7b7b, #ff9f5d)', border: 'none',
          borderRadius: '3px', color: '#0a0e14', cursor: 'pointer', fontWeight: 700,
          marginBottom: '16px', opacity: generateDigest.isPending ? 0.6 : 1,
        }}
      >
        {generateDigest.isPending ? 'Generating...' : 'Generate Now'}
      </button>

      {/* History list */}
      {digests && digests.length > 0 && (
        <div>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginBottom: '8px' }}>
            PREVIOUS DIGESTS
          </div>
          {digests.map((d) => (
            <div
              key={d.id}
              style={{
                display: 'flex', alignItems: 'center', padding: '8px 0',
                borderBottom: '1px solid #111820', gap: '12px',
              }}
            >
              <span style={{ fontFamily: 'Syne, sans-serif', fontSize: '11px', color: '#d0d0d0' }}>
                Week of {d.week_starting}
              </span>
              <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#555' }}>
                {d.signal_count} signals
              </span>
              <a
                href={`/api/v1/digest/${d.id}/download`}
                style={{ marginLeft: 'auto', fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#5dc8de', textDecoration: 'none' }}
              >
                &#8681;
              </a>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
