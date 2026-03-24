import React, { useState } from 'react';
import { useApprovalQueue } from '@/api/jira';

export const ApprovalQueueStatus: React.FC = () => {
  const { data: pendingItems } = useApprovalQueue('Pending');
  const { data: failedItems } = useApprovalQueue('Failed');
  const { data: pushedItems } = useApprovalQueue('Pushed');
  const [popoverOpen, setPopoverOpen] = useState(false);

  const pending = pendingItems?.length ?? 0;
  const failed = failedItems?.length ?? 0;
  const pushed = pushedItems?.length ?? 0;

  if (pending === 0 && failed === 0 && pushed === 0) return null;

  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        onClick={() => setPopoverOpen(!popoverOpen)}
        aria-label={`Approval queue: ${pending} pending, ${pushed} pushed, ${failed} failed`}
        style={{
          fontFamily: 'Syne Mono, monospace', fontSize: '9px', background: 'none',
          border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px',
          padding: '4px 8px', color: '#888',
        }}
      >
        {pending > 0 && (
          <span style={{ color: '#ffc95d' }}>
            &#9651; {pending} pending
          </span>
        )}
        {pushed > 0 && (
          <span style={{ color: '#5dde9f' }}>
            &#10004; {pushed} pushed
          </span>
        )}
        {failed > 0 && (
          <span style={{ color: '#ff7b7b', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{
              width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#ff7b7b',
              boxShadow: '0 0 4px #ff7b7b', display: 'inline-block',
            }} />
            &#10007; {failed} failed
          </span>
        )}
      </button>

      {/* Popover */}
      {popoverOpen && (
        <div style={{
          position: 'absolute', top: '100%', right: 0, zIndex: 100,
          backgroundColor: '#111820', border: '1px solid #1e2530', borderRadius: '6px',
          padding: '12px', minWidth: '300px', maxHeight: '300px', overflowY: 'auto',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
        }}>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginBottom: '8px' }}>
            APPROVAL QUEUE
          </div>
          {[...(pendingItems ?? []), ...(failedItems ?? []), ...(pushedItems ?? [])].map((item) => (
            <div
              key={item.id}
              style={{
                padding: '6px 0', borderBottom: '1px solid #1e2530',
                display: 'flex', alignItems: 'center', gap: '8px',
              }}
            >
              <span style={{
                fontFamily: 'Syne Mono, monospace', fontSize: '8px', padding: '2px 5px',
                borderRadius: '3px',
                color: item.status === 'Pushed' ? '#5dde9f'
                  : item.status === 'Failed' ? '#ff7b7b'
                    : item.status === 'Pending' ? '#ffc95d'
                      : '#888',
                backgroundColor: item.status === 'Pushed' ? 'rgba(93,222,157,0.1)'
                  : item.status === 'Failed' ? 'rgba(255,96,96,0.1)'
                    : 'rgba(255,201,93,0.1)',
              }}>
                {item.status}
              </span>
              <span style={{ fontFamily: 'Syne, sans-serif', fontSize: '10px', color: '#d0d0d0', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {item.ticket_title}
              </span>
              {item.jira_issue_key && (
                <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#5dc8de' }}>
                  {item.jira_issue_key}
                </span>
              )}
              {item.last_error && (
                <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#ff7b7b' }} title={item.last_error}>
                  &#9888;
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
