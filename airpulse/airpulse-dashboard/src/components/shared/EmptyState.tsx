import React from 'react';

interface EmptyStateProps {
  message: string;
  action?: { label: string; onClick: () => void };
}

export const EmptyState: React.FC<EmptyStateProps> = ({ message, action }) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 24px',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontFamily: 'DM Serif Display, serif',
          fontSize: '24px',
          color: '#2a3040',
          marginBottom: '8px',
        }}
      >
        {message}
      </div>
      {action && (
        <button
          onClick={action.onClick}
          style={{
            marginTop: '16px',
            fontFamily: 'Syne Mono, monospace',
            fontSize: '10px',
            padding: '6px 16px',
            backgroundColor: 'transparent',
            border: '1px solid #3d4555',
            color: '#888',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
