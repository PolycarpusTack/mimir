import { describe, it, expect, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { SignalDetail } from '@/components/signal/SignalDetail';

describe('SignalDetail', () => {
  // TC-SD-001
  it('shows empty state when no signal selected', () => {
    renderWithProviders(<SignalDetail signalId={null} />);
    expect(screen.getByText('Select a signal')).toBeInTheDocument();
  });

  // TC-SD-002
  it('shows enrichment data for enriched signal', async () => {
    renderWithProviders(<SignalDetail signalId="sig-001" />);
    await waitFor(() => {
      expect(screen.getByText(/Harmonic's cloud-native playout/)).toBeInTheDocument();
      expect(screen.getByText('88')).toBeInTheDocument();
    });
  });

  // TC-SD-003
  it('shows enrichment pending for unenriched signal', async () => {
    renderWithProviders(<SignalDetail signalId="sig-008" />);
    await waitFor(() => {
      expect(screen.getByText('Enrichment pending...')).toBeInTheDocument();
    });
  });

  // TC-SD-004
  it('dispatches roadmap-push event on Push to JIRA click', async () => {
    const user = userEvent.setup();
    const handler = vi.fn();
    window.addEventListener('airpulse:roadmap-push', handler);

    renderWithProviders(<SignalDetail signalId="sig-001" />);
    await waitFor(() => {
      expect(screen.getByText('Push to JIRA / AirForge')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Push to JIRA / AirForge'));
    expect(handler).toHaveBeenCalledTimes(1);

    window.removeEventListener('airpulse:roadmap-push', handler);
  });

  // TC-SD-005
  it('sends PATCH request on Archive click', async () => {
    const user = userEvent.setup();
    const fetchSpy = vi.spyOn(globalThis, 'fetch');

    renderWithProviders(<SignalDetail signalId="sig-001" />);
    await waitFor(() => {
      expect(screen.getByText('Archive')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Archive'));

    await waitFor(() => {
      const patchCall = fetchSpy.mock.calls.find(
        ([url, opts]) =>
          typeof url === 'string' &&
          url.includes('/api/v1/signals/sig-001') &&
          (opts as RequestInit)?.method === 'PATCH',
      );
      expect(patchCall).toBeTruthy();
    });

    fetchSpy.mockRestore();
  });

  // TC-SD-006
  it('shows action_recommended when present', async () => {
    renderWithProviders(<SignalDetail signalId="sig-001" />);
    await waitFor(() => {
      expect(screen.getByText(/Schedule competitive analysis/)).toBeInTheDocument();
    });
  });

  // TC-SD-007
  it('shows vendor mentions in tag cloud', async () => {
    renderWithProviders(<SignalDetail signalId="sig-001" />);
    await waitFor(() => {
      expect(screen.getByText('Harmonic')).toBeInTheDocument();
    });
  });
});
