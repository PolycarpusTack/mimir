import { describe, it, expect, vi, afterEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { RoadmapQueue } from '@/components/layout/RoadmapQueue';
import { useUIStore } from '@/store/ui';

afterEach(() => {
  useUIStore.setState({ queueExpanded: false });
});

describe('RoadmapQueue', () => {
  // TC-RQ-001
  it('renders collapsed by default', () => {
    renderWithProviders(<RoadmapQueue />);
    const header = screen.getByRole('button', { name: /roadmap signal queue/i });
    expect(header).toHaveAttribute('aria-expanded', 'false');
  });

  // TC-RQ-002
  it('expands when header is clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoadmapQueue />);

    const header = screen.getByRole('button', { name: /roadmap signal queue/i });
    await user.click(header);

    expect(header).toHaveAttribute('aria-expanded', 'true');
    expect(useUIStore.getState().queueExpanded).toBe(true);
  });

  // TC-RQ-003
  it('shows cards for high-relevance signals when expanded', async () => {
    useUIStore.setState({ queueExpanded: true });
    renderWithProviders(<RoadmapQueue />);

    await waitFor(() => {
      // Signals with relevance >= 70: sig-001 (88), sig-002 (92), sig-003 (78), sig-004 (85), sig-005 (75), sig-007 (82)
      expect(screen.getByText(/Harmonic launches/)).toBeInTheDocument();
    });
  });

  // TC-RQ-004
  it('dispatches roadmap-push event on Push button click', async () => {
    const user = userEvent.setup();
    useUIStore.setState({ queueExpanded: true });
    const handler = vi.fn();
    window.addEventListener('airpulse:roadmap-push', handler);

    renderWithProviders(<RoadmapQueue />);

    await waitFor(() => {
      expect(screen.getAllByText(/Push/).length).toBeGreaterThanOrEqual(1);
    });

    const pushButtons = screen.getAllByText(/Push/);
    await user.click(pushButtons[0]);
    expect(handler).toHaveBeenCalledTimes(1);

    window.removeEventListener('airpulse:roadmap-push', handler);
  });

  // TC-RQ-005
  it('toggles with keyboard Enter', async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoadmapQueue />);

    const header = screen.getByRole('button', { name: /roadmap signal queue/i });
    header.focus();
    await user.keyboard('{Enter}');

    expect(useUIStore.getState().queueExpanded).toBe(true);
  });
});
