import React, { useEffect } from 'react';
import { useSignals } from '@/api/signals';
import { useUIStore } from '@/store/ui';
import { useFilterStore } from '@/store/filters';
import { SignalItem } from './SignalItem';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { EmptyState } from '@/components/shared/EmptyState';

export const SignalFeed: React.FC = () => {
  const filterDomain = useFilterStore((s) => s.filterDomain);
  const setFilterDomain = useFilterStore((s) => s.setFilterDomain);
  const selectedSignalId = useUIStore((s) => s.selectedSignalId);
  const setSelectedSignalId = useUIStore((s) => s.setSelectedSignalId);

  const { data, isLoading, isError, refetch } = useSignals({
    domain: filterDomain,
    page: 1,
    page_size: 50,
  });

  // Pre-select from URL param
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sig = params.get('signal');
    if (sig && !selectedSignalId) {
      setSelectedSignalId(sig);
    }
  }, [selectedSignalId, setSelectedSignalId]);

  const handleSelect = (id: string) => {
    setSelectedSignalId(id);
    const params = new URLSearchParams(window.location.search);
    params.set('signal', id);
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
  };

  if (isError) {
    return (
      <div style={{ width: '320px', borderRight: '1px solid #111820' }}>
        <div role="alert" style={{ padding: '24px', textAlign: 'center' }}>
          <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: '14px', color: '#ff7b7b', marginBottom: '8px' }}>
            Failed to load signals
          </div>
          <button
            onClick={() => refetch()}
            style={{
              fontFamily: 'Syne Mono, monospace', fontSize: '10px', padding: '6px 16px',
              backgroundColor: 'transparent', border: '1px solid #ff7b7b', color: '#ff7b7b',
              borderRadius: '4px', cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: '320px', borderRight: '1px solid #111820', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        height: '40px', display: 'flex', alignItems: 'center', padding: '0 14px',
        borderBottom: '1px solid #111820', flexShrink: 0,
      }}>
        <span style={{ fontFamily: 'DM Serif Display, serif', fontSize: '14px', color: '#d0d0d0' }}>
          Signal Feed
        </span>
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginLeft: '8px' }}>
          {data?.total ?? 0}
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444' }}>
          RELEVANCE &#9662;
        </span>
      </div>

      {/* Signal list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {isLoading ? (
          <div style={{ padding: '12px' }}>
            <LoadingSkeleton height={80} count={3} />
          </div>
        ) : data?.items.length === 0 ? (
          <EmptyState
            message="No signals match this filter"
            action={filterDomain ? { label: 'Clear filter', onClick: () => setFilterDomain(null) } : undefined}
          />
        ) : (
          data?.items.map((signal) => (
            <SignalItem
              key={signal.id}
              signal={signal}
              selected={selectedSignalId === signal.id}
              onClick={() => handleSelect(signal.id)}
            />
          ))
        )}
      </div>
    </div>
  );
};
