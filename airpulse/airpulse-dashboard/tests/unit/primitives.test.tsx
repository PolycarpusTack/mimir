import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SparkLine } from '@/components/primitives/SparkLine';
import { DomainBadge } from '@/components/primitives/DomainBadge';
import { SignalTypeBadge } from '@/components/primitives/SignalTypeBadge';
import { ZScoreCell } from '@/components/primitives/ZScoreCell';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { EmptyState } from '@/components/shared/EmptyState';
import { ErrorBoundary } from '@/components/shared/ErrorBoundary';
import { vi } from 'vitest';

describe('SparkLine', () => {
  it('renders SVG with correct aria-label', () => {
    render(<SparkLine data={[1, 3, 2, 5, 4, 6, 7]} color="#ff0000" />);
    const svg = screen.getByRole('img');
    expect(svg).toHaveAttribute('aria-label', 'Trend chart: 1 to 7 over 7 periods');
  });

  it('renders nothing for empty data', () => {
    const { container } = render(<SparkLine data={[]} color="#ff0000" />);
    expect(container.innerHTML).toBe('');
  });

  it('handles single data point', () => {
    render(<SparkLine data={[5]} color="#00ff00" />);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });
});

describe('DomainBadge', () => {
  it('renders domain name', () => {
    render(<DomainBadge domain="AI" />);
    expect(screen.getByText('AI')).toBeInTheDocument();
  });

  it('renders with md size', () => {
    render(<DomainBadge domain="Broadcast" size="md" />);
    expect(screen.getByText('Broadcast')).toBeInTheDocument();
  });
});

describe('SignalTypeBadge', () => {
  it('renders display label', () => {
    render(<SignalTypeBadge signalType="ProductLaunch" />);
    expect(screen.getByText('PRODUCT')).toBeInTheDocument();
  });

  it('renders M&A label for MaSignal', () => {
    render(<SignalTypeBadge signalType="MaSignal" />);
    expect(screen.getByText('M&A')).toBeInTheDocument();
  });
});

describe('ZScoreCell', () => {
  it('renders z-score value and count', () => {
    render(
      <table><tbody><tr>
        <ZScoreCell z={3.7} count={18} label="SURGE" domain="AI" signalType="ProductLaunch" />
      </tr></tbody></table>,
    );
    expect(screen.getByText('3.7')).toBeInTheDocument();
    expect(screen.getByText('18')).toBeInTheDocument();
  });

  it('has correct aria-label', () => {
    render(
      <table><tbody><tr>
        <ZScoreCell z={2.5} count={10} label="SPIKE" domain="OTT" signalType="MaSignal" />
      </tr></tbody></table>,
    );
    expect(screen.getByLabelText('OTT MaSignal: z-score 2.5, severity SPIKE')).toBeInTheDocument();
  });
});

describe('LoadingSkeleton', () => {
  it('renders correct number of skeleton rows', () => {
    render(<LoadingSkeleton height={80} count={3} />);
    const skeletons = screen.getAllByRole('status');
    expect(skeletons).toHaveLength(3);
  });
});

describe('EmptyState', () => {
  it('renders message text', () => {
    render(<EmptyState message="No signals found" />);
    expect(screen.getByText('No signals found')).toBeInTheDocument();
  });

  it('renders action button when provided', () => {
    const onClick = () => {};
    render(<EmptyState message="No data" action={{ label: 'Retry', onClick }} />);
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });
});

describe('ErrorBoundary', () => {
  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>OK content</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText('OK content')).toBeInTheDocument();
  });

  it('renders error card when child throws', () => {
    const ThrowingComponent = () => {
      throw new Error('Test error');
    };

    // Suppress React error boundary console output
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>,
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();

    spy.mockRestore();
  });
});
