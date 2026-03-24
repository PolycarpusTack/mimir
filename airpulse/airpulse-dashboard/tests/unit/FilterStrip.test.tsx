import { describe, it, expect, afterEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { FilterStrip } from '@/components/layout/FilterStrip';
import { useFilterStore } from '@/store/filters';

afterEach(() => {
  useFilterStore.setState({ filterDomain: null });
  window.history.replaceState(null, '', '/');
});

describe('FilterStrip', () => {
  // TC-FS-001
  it('shows ALL pill active when no filter set', () => {
    renderWithProviders(<FilterStrip />);
    const allPill = screen.getByLabelText('Show all domains');
    expect(allPill).toHaveAttribute('aria-pressed', 'true');
  });

  // TC-FS-002
  it('activates domain filter and updates URL on click', async () => {
    const user = userEvent.setup();
    renderWithProviders(<FilterStrip />);

    const aiPill = screen.getByLabelText('Filter by AI');
    await user.click(aiPill);

    expect(aiPill).toHaveAttribute('aria-pressed', 'true');
    expect(useFilterStore.getState().filterDomain).toBe('AI');
    expect(window.location.search).toContain('domain=AI');
  });

  // TC-FS-003
  it('clears filter when clicking active domain again', async () => {
    const user = userEvent.setup();
    useFilterStore.setState({ filterDomain: 'AI' });
    renderWithProviders(<FilterStrip />);

    const aiPill = screen.getByLabelText('Filter by AI');
    await user.click(aiPill);

    expect(useFilterStore.getState().filterDomain).toBeNull();
    const allPill = screen.getByLabelText('Show all domains');
    expect(allPill).toHaveAttribute('aria-pressed', 'true');
  });

  // TC-FS-004
  it('initialises filter from URL param on mount', () => {
    window.history.replaceState(null, '', '/?domain=Cloud');
    renderWithProviders(<FilterStrip />);

    expect(useFilterStore.getState().filterDomain).toBe('Cloud');
  });

  // TC-FS-005
  it('has correct accessibility attributes on all pills', () => {
    renderWithProviders(<FilterStrip />);
    const toolbar = screen.getByRole('toolbar');
    expect(toolbar).toBeInTheDocument();

    const buttons = screen.getAllByRole('button');
    buttons.forEach((btn) => {
      expect(btn).toHaveAttribute('aria-pressed');
    });
  });
});
