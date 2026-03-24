import React, { useState, useEffect, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/shared/ErrorBoundary';
import { TopBar } from '@/components/layout/TopBar';
import { FilterStrip } from '@/components/layout/FilterStrip';
import { SignalFeed } from '@/components/signal/SignalFeed';
import { SignalDetail } from '@/components/signal/SignalDetail';
import { ShiftRadar } from '@/components/intelligence/ShiftRadar';
import { VendorTracker } from '@/components/intelligence/VendorTracker';
import { TechTrends } from '@/components/intelligence/TechTrends';
import { DigestHistoryPanel } from '@/components/intelligence/DigestHistoryPanel';
import { RoadmapQueue } from '@/components/layout/RoadmapQueue';
import { JiraPushModal } from '@/components/signal/JiraPushModal';
import { useUIStore } from '@/store/ui';
import { useSignalEnrichment } from '@/api/signals';
import type { Signal } from '@/types';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

const RightPanel: React.FC = () => {
  const activePanel = useUIStore((s) => s.activePanel);

  return (
    <div style={{ flex: 1, overflow: 'hidden' }}>
      {activePanel === 'shift' && <ShiftRadar />}
      {activePanel === 'vendors' && <VendorTracker />}
      {activePanel === 'tech' && <TechTrends />}
      {activePanel === 'digest' && <DigestHistoryPanel />}
    </div>
  );
};

const JiraPushListener: React.FC = () => {
  const [pushSignal, setPushSignal] = useState<Signal | null>(null);
  const { data: enrichment } = useSignalEnrichment(pushSignal?.id ?? null);

  const handlePush = useCallback((e: Event) => {
    const signal = (e as CustomEvent).detail as Signal;
    if (signal?.id) {
      setPushSignal(signal);
    }
  }, []);

  useEffect(() => {
    window.addEventListener('airpulse:roadmap-push', handlePush);
    return () => window.removeEventListener('airpulse:roadmap-push', handlePush);
  }, [handlePush]);

  if (!pushSignal || !enrichment) return null;

  return (
    <JiraPushModal
      signal={pushSignal}
      enrichment={enrichment}
      onClose={() => setPushSignal(null)}
      onSuccess={() => setPushSignal(null)}
    />
  );
};

const MainContent: React.FC = () => {
  const selectedSignalId = useUIStore((s) => s.selectedSignalId);

  return (
    <main style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      <ErrorBoundary>
        <SignalFeed />
      </ErrorBoundary>
      <ErrorBoundary>
        <SignalDetail signalId={selectedSignalId} />
      </ErrorBoundary>
      <ErrorBoundary>
        <RightPanel />
      </ErrorBoundary>
    </main>
  );
};

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            height: '100vh',
            backgroundColor: '#0d1117',
            color: '#d0d0d0',
          }}
        >
          <TopBar />
          <FilterStrip />
          <MainContent />
          <RoadmapQueue />
          <JiraPushListener />
        </div>
      </ErrorBoundary>
    </QueryClientProvider>
  );
};
