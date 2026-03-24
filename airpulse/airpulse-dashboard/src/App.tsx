import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/shared/ErrorBoundary';
import { TopBar } from '@/components/layout/TopBar';
import { FilterStrip } from '@/components/layout/FilterStrip';
import { SignalFeed } from '@/components/signal/SignalFeed';
import { SignalDetail } from '@/components/signal/SignalDetail';
import { ShiftRadar } from '@/components/intelligence/ShiftRadar';
import { VendorTracker } from '@/components/intelligence/VendorTracker';
import { TechTrends } from '@/components/intelligence/TechTrends';
import { RoadmapQueue } from '@/components/layout/RoadmapQueue';
import { useUIStore } from '@/store/ui';

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
    </div>
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
        </div>
      </ErrorBoundary>
    </QueryClientProvider>
  );
};
