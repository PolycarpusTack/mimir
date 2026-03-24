import React from 'react';

interface SparkLineProps {
  data: number[];
  color: string;
  width?: number;
  height?: number;
}

export const SparkLine: React.FC<SparkLineProps> = ({
  data,
  color,
  width = 80,
  height = 28,
}) => {
  if (data.length === 0) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pad = 2;

  const points = data
    .map((v, i) => {
      const x = pad + (i / (data.length - 1 || 1)) * (width - 2 * pad);
      const y = height - pad - ((v - min) / range) * (height - 2 * pad);
      return `${x},${y}`;
    })
    .join(' ');

  const lastX = pad + ((data.length - 1) / (data.length - 1 || 1)) * (width - 2 * pad);
  const lastY = height - pad - ((data[data.length - 1] - min) / range) * (height - 2 * pad);

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label={`Trend chart: ${min} to ${max} over ${data.length} periods`}
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={lastX} cy={lastY} r={2.5} fill={color} />
    </svg>
  );
};
