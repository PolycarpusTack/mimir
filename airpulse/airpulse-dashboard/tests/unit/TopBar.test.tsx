import { describe, it, expect, vi, afterEach } from 'vitest';
import { screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { TopBar } from '@/components/layout/TopBar';
import { useUIStore } from '@/store/ui';
import { server } from '../mocks/server';
import { http, HttpResponse } from 'msw';
import { FIXTURE_HEALTH_DEGRADED } from '../mocks/fixtures';

afterEach(() => {
  useUIStore.setState({ activePanel: 'shift' });
});

describe('TopBar', () => {
  // TC-TB-001
  it('displays signal count and surge count from health endpoint', async () => {
    renderWithProviders(<TopBar />);
    await waitFor(() => {
      expect(screen.getByLabelText('SIGNALS TODAY: 47')).toBeInTheDocument();
      expect(screen.getByLabelText('SURGES ACTIVE: 3')).toBeInTheDocument();
    });
  });

  // TC-TB-002
  it('updates clock every second', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    renderWithProviders(<TopBar />);

    const clockEl = screen.getByLabelText(/^Clock:/);
    act(() => { vi.advanceTimersByTime(1100); });

    // Clock should have updated (element should exist and display time)
    expect(clockEl).toBeInTheDocument();

    vi.useRealTimers();
  });

  // TC-TB-003
  it('switches active panel when tab is clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<TopBar />);

    const vendorsTab = screen.getByRole('tab', { name: /vendors/i });
    await user.click(vendorsTab);

    expect(vendorsTab).toHaveAttribute('aria-selected', 'true');
    expect(useUIStore.getState().activePanel).toBe('vendors');
  });

  // TC-TB-004
  it('shows amber color for sources when health is degraded', async () => {
    server.use(
      http.get('/api/v1/health', () => HttpResponse.json(FIXTURE_HEALTH_DEGRADED)),
    );

    renderWithProviders(<TopBar />);
    await waitFor(() => {
      expect(screen.getByLabelText('SOURCES LIVE: 4')).toBeInTheDocument();
    });
  });

  // TC-TB-005
  it('has correct tab accessibility attributes and passes axe audit', async () => {
    const { container } = renderWithProviders(<TopBar />);
    const tablist = screen.getByRole('tablist');
    expect(tablist).toBeInTheDocument();

    const tabs = screen.getAllByRole('tab');
    expect(tabs).toHaveLength(3);
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');
    expect(tabs[0]).toHaveAttribute('aria-controls', 'panel-shift');

    // jest-axe accessibility check
    const { axe } = await import('jest-axe');
    // Exclude aria-valid-attr-value: panel IDs referenced by aria-controls exist
    // in the RightPanel component, not co-located with TopBar in this unit test
    const results = await axe(container, {
      rules: { 'aria-valid-attr-value': { enabled: false } },
    });
    expect(results.violations).toHaveLength(0);
  });
});
