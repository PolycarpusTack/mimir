import { describe, it, expect, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { JiraPushModal } from '@/components/signal/JiraPushModal';
import { FIXTURE_SIGNALS, FIXTURE_ENRICHMENTS } from '../mocks/fixtures';
import type { Signal, EnrichedAnnotation } from '@/types';

const signal = FIXTURE_SIGNALS[0] as Signal;
const enrichment = FIXTURE_ENRICHMENTS['sig-001'] as EnrichedAnnotation;

describe('JiraPushModal', () => {
  // TC-UI4-001
  it('opens with pre-filled title, body, and metadata', () => {
    const onClose = vi.fn();
    const onSuccess = vi.fn();
    renderWithProviders(
      <JiraPushModal signal={signal} enrichment={enrichment} onClose={onClose} onSuccess={onSuccess} />,
    );

    expect(screen.getByText('Push to JIRA / AirForge')).toBeInTheDocument();
    expect(screen.getByDisplayValue(/AirPulse/)).toBeInTheDocument();
    expect(screen.getByText('WHY IT MATTERS')).toBeInTheDocument();
  });

  // TC-UI4-002
  it('allows editing ticket title', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onSuccess = vi.fn();
    renderWithProviders(
      <JiraPushModal signal={signal} enrichment={enrichment} onClose={onClose} onSuccess={onSuccess} />,
    );

    const titleInput = screen.getByDisplayValue(/AirPulse/);
    await user.clear(titleInput);
    await user.type(titleInput, 'Custom title');
    expect(titleInput).toHaveValue('Custom title');
    expect(screen.getByText('Confirm & Push to JIRA')).toBeEnabled();
  });

  // TC-UI4-003
  it('shows success on push confirm', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onSuccess = vi.fn();
    renderWithProviders(
      <JiraPushModal signal={signal} enrichment={enrichment} onClose={onClose} onSuccess={onSuccess} />,
    );

    await user.click(screen.getByText('Confirm & Push to JIRA'));
    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalled();
    });
  });

  // TC-UI4-005
  it('closes on Escape key', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onSuccess = vi.fn();
    renderWithProviders(
      <JiraPushModal signal={signal} enrichment={enrichment} onClose={onClose} onSuccess={onSuccess} />,
    );

    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalled();
  });

  it('has correct dialog accessibility attributes', () => {
    const onClose = vi.fn();
    const onSuccess = vi.fn();
    renderWithProviders(
      <JiraPushModal signal={signal} enrichment={enrichment} onClose={onClose} onSuccess={onSuccess} />,
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-label', 'Push to JIRA');
  });
});
