import React from 'react';

interface LoadingSkeletonProps {
  height: number;
  width?: string;
  count?: number;
}

const shimmerStyle = `
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
`;

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  height,
  width = '100%',
  count = 1,
}) => {
  return (
    <>
      <style>{shimmerStyle}</style>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          role="status"
          aria-label="Loading"
          style={{
            height: `${height}px`,
            width,
            background: 'linear-gradient(90deg, #111820 25%, #1a2535 50%, #111820 75%)',
            backgroundSize: '200% 100%',
            animation: 'shimmer 1.5s infinite linear',
            borderRadius: '4px',
            marginBottom: i < count - 1 ? '8px' : 0,
          }}
        />
      ))}
    </>
  );
};
