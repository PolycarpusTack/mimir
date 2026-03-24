import { describe, it, expect } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { renderWithProviders } from './test-utils';
import { ShiftRadar } from '@/components/intelligence/ShiftRadar';
import { server } from '../mocks/server';
import { http, HttpResponse } from 'msw';

describe('ShiftRadar', () => {
  // TC-SR-001
  it('renders surge alert cell correctly', async () => {
    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      expect(screen.getByText('Shift Radar')).toBeInTheDocument();
      // AI ProductLaunch surge (z=3.7)
      const surgeCell = screen.getByLabelText(/AI ProductLaunch: z-score 3.7, severity SURGE/);
      expect(surgeCell).toBeInTheDocument();
    });
  });

  // TC-SR-002
  it('renders all cells as NORMAL when no alerts', async () => {
    server.use(
      http.get('/api/v1/shifts', () =>
        HttpResponse.json({ alerts: [], total: 0, as_of: new Date().toISOString() }),
      ),
    );

    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      const normalCells = screen.getAllByLabelText(/severity NORMAL/);
      expect(normalCells.length).toBe(36); // 6 domains × 6 signal types
    });
  });

  // TC-SR-003
  it('renders spike with correct label', async () => {
    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      const spikeCell = screen.getByLabelText(/OTT MaSignal: z-score 2.8, severity SPIKE/);
      expect(spikeCell).toBeInTheDocument();
    });
  });

  // TC-SR-004
  it('has correct aria-label on z-score cells', async () => {
    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      const cell = screen.getByLabelText(/Broadcast StandardsShift: z-score 1.8, severity ELEVATED/);
      expect(cell).toBeInTheDocument();
    });
  });

  // TC-SR-005
  it('shows error state on API failure', async () => {
    server.use(
      http.get('/api/v1/shifts', () => HttpResponse.error()),
    );

    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      expect(screen.getByText('Failed to load shift data')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  // TC-SR-006 (bonus): active surges callout displays
  it('shows active surges callout for high z-scores', async () => {
    renderWithProviders(<ShiftRadar />);
    await waitFor(() => {
      expect(screen.getByText('ACTIVE SURGES & SPIKES')).toBeInTheDocument();
    });
  });
});
