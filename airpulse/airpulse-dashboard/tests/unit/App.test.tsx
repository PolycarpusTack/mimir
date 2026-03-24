import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { App } from '@/App';

describe('App', () => {
  it('renders the full dashboard layout with ShiftRadar', async () => {
    render(<App />);

    // TopBar
    await waitFor(() => {
      expect(screen.getByText('AirPulse')).toBeInTheDocument();
      expect(screen.getByText('INTELLIGENCE')).toBeInTheDocument();
    });

    // FilterStrip
    expect(screen.getByRole('toolbar')).toBeInTheDocument();

    // SignalFeed
    expect(screen.getByText('Signal Feed')).toBeInTheDocument();

    // ShiftRadar (default active panel)
    await waitFor(() => {
      expect(screen.getByText('Shift Radar')).toBeInTheDocument();
    }, { timeout: 5000 });

    // RoadmapQueue
    expect(screen.getByRole('button', { name: /roadmap signal queue/i })).toBeInTheDocument();
  });
});
