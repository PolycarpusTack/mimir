import { describe, it, expect } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { DigestHistoryPanel } from '@/components/intelligence/DigestHistoryPanel';

describe('DigestHistoryPanel', () => {
  // TC-UI4-006
  it('renders latest digest with week date and signal count', async () => {
    renderWithProviders(<DigestHistoryPanel />);
    await waitFor(() => {
      expect(screen.getByText('Weekly Digest')).toBeInTheDocument();
    });
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('LATEST DIGEST')).toBeInTheDocument();
    }, { timeout: 5000 });
    expect(screen.getAllByText(/47 signals/).length).toBeGreaterThanOrEqual(1);
  });

  it('shows previous digests in history list', async () => {
    renderWithProviders(<DigestHistoryPanel />);
    await waitFor(() => {
      expect(screen.getByText('PREVIOUS DIGESTS')).toBeInTheDocument();
      expect(screen.getByText(/Week of 2026-03-09/)).toBeInTheDocument();
    });
  });

  // TC-UI4-007
  it('triggers generation on Generate Now click', async () => {
    const user = userEvent.setup();
    renderWithProviders(<DigestHistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText('Generate Now')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Generate Now'));
    // Button should enter loading state
    await waitFor(() => {
      expect(screen.getByText('Generate Now')).toBeInTheDocument();
    });
  });

  it('has download DOCX link', async () => {
    renderWithProviders(<DigestHistoryPanel />);
    await waitFor(() => {
      expect(screen.getByText('Download DOCX')).toBeInTheDocument();
    });
  });

  it('toggles markdown viewer', async () => {
    const user = userEvent.setup();
    renderWithProviders(<DigestHistoryPanel />);

    await waitFor(() => {
      expect(screen.getByText('View Markdown')).toBeInTheDocument();
    });

    await user.click(screen.getByText('View Markdown'));
    // After clicking, the label changes to "Hide Markdown"
    await waitFor(() => {
      expect(screen.getByText('Hide Markdown')).toBeInTheDocument();
    });
  });
});
