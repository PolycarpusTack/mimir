import { describe, it, expect } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from './test-utils';
import { ApprovalQueueStatus } from '@/components/layout/ApprovalQueueStatus';

describe('ApprovalQueueStatus', () => {
  // TC-UI4-008
  it('shows pending and failed counts', async () => {
    renderWithProviders(<ApprovalQueueStatus />);
    await waitFor(() => {
      expect(screen.getByText(/1 pending/)).toBeInTheDocument();
      expect(screen.getByText(/1 failed/)).toBeInTheDocument();
    });
  });

  it('shows pushed count', async () => {
    renderWithProviders(<ApprovalQueueStatus />);
    await waitFor(() => {
      expect(screen.getByText(/1 pushed/)).toBeInTheDocument();
    });
  });

  it('opens popover on click', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ApprovalQueueStatus />);

    await waitFor(() => {
      expect(screen.getByText(/1 pending/)).toBeInTheDocument();
    });

    await user.click(screen.getByLabelText(/Approval queue/));
    await waitFor(() => {
      expect(screen.getByText('APPROVAL QUEUE')).toBeInTheDocument();
      expect(screen.getByText('AIRFORGE-142')).toBeInTheDocument();
    });
  });
});
