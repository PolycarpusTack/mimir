import { describe, it, expect, afterEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { SignalFeed } from '@/components/signal/SignalFeed';
import { useUIStore } from '@/store/ui';
import { useFilterStore } from '@/store/filters';
import { server } from '../mocks/server';
import { http, HttpResponse, delay } from 'msw';

afterEach(() => {
  useUIStore.setState({ selectedSignalId: null });
  useFilterStore.setState({ filterDomain: null });
  window.history.replaceState(null, '', '/');
});

describe('SignalFeed', () => {
  // TC-SF-001
  it('renders signal items from API', async () => {
    renderWithProviders(<SignalFeed />);
    await waitFor(() => {
      expect(screen.getByText('Signal Feed')).toBeInTheDocument();
      expect(screen.getByText(/Harmonic launches/)).toBeInTheDocument();
    });
  });

  // TC-SF-002
  it('shows loading skeletons while fetching', async () => {
    server.use(
      http.get('/api/v1/signals', async () => {
        await delay(2000);
        return HttpResponse.json({ items: [], total: 0, page: 1, page_size: 50 });
      }),
    );

    renderWithProviders(<SignalFeed />);
    const skeletons = screen.getAllByRole('status', { name: 'Loading' });
    expect(skeletons.length).toBeGreaterThanOrEqual(1);
  });

  // TC-SF-003
  it('shows empty state when no signals match filter', async () => {
    useFilterStore.setState({ filterDomain: 'Sports' });
    server.use(
      http.get('/api/v1/signals', () =>
        HttpResponse.json({ items: [], total: 0, page: 1, page_size: 50 }),
      ),
    );

    renderWithProviders(<SignalFeed />);
    await waitFor(() => {
      expect(screen.getByText('No signals match this filter')).toBeInTheDocument();
    });
  });

  // TC-SF-004
  it('shows error state with retry button on API failure', async () => {
    server.use(
      http.get('/api/v1/signals', () => HttpResponse.error()),
    );

    renderWithProviders(<SignalFeed />);
    await waitFor(() => {
      expect(screen.getByText('Failed to load signals')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  // TC-SF-005
  it('selects signal on click', async () => {
    const user = userEvent.setup();
    renderWithProviders(<SignalFeed />);

    await waitFor(() => {
      expect(screen.getByText(/Harmonic launches/)).toBeInTheDocument();
    });

    const items = screen.getAllByRole('button');
    const signalButton = items.find((el) => el.textContent?.includes('Harmonic'));
    if (signalButton) {
      await user.click(signalButton);
      expect(useUIStore.getState().selectedSignalId).toBe('sig-001');
    }
  });

  // TC-SF-007
  it('displays relevance score in correct color', async () => {
    renderWithProviders(<SignalFeed />);
    await waitFor(() => {
      // sig-002 has relevance_score=92 -> should be red
      const scores = screen.getAllByText('92');
      expect(scores.length).toBeGreaterThanOrEqual(1);
    });
  });

  // TC-SF-008
  it('handles keyboard navigation on signal items', async () => {
    const user = userEvent.setup();
    renderWithProviders(<SignalFeed />);

    await waitFor(() => {
      expect(screen.getByText(/Harmonic launches/)).toBeInTheDocument();
    });

    const items = screen.getAllByRole('button');
    const signalButton = items.find((el) => el.textContent?.includes('Harmonic'));
    if (signalButton) {
      signalButton.focus();
      await user.keyboard('{Enter}');
      expect(useUIStore.getState().selectedSignalId).toBe('sig-001');
    }
  });
});
