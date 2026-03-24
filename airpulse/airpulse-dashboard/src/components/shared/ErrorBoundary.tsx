import React from 'react';

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div
          role="alert"
          style={{
            padding: '24px',
            margin: '12px',
            backgroundColor: 'rgba(255,96,96,0.08)',
            border: '1px solid rgba(255,96,96,0.3)',
            borderRadius: '6px',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              fontFamily: 'DM Serif Display, serif',
              fontSize: '16px',
              color: '#ff7b7b',
              marginBottom: '8px',
            }}
          >
            Something went wrong
          </div>
          {import.meta.env.DEV && this.state.error && (
            <div
              style={{
                fontFamily: 'Syne Mono, monospace',
                fontSize: '10px',
                color: '#888',
                marginBottom: '12px',
              }}
            >
              {this.state.error.message}
            </div>
          )}
          <button
            onClick={this.handleRetry}
            style={{
              fontFamily: 'Syne Mono, monospace',
              fontSize: '10px',
              padding: '6px 16px',
              backgroundColor: 'transparent',
              border: '1px solid #ff7b7b',
              color: '#ff7b7b',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
