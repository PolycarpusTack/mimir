import React from 'react';
import { zToColor } from '@/utils';

interface ZScoreCellProps {
  z: number;
  count: number;
  label: string;
  domain: string;
  signalType: string;
}

export const ZScoreCell: React.FC<ZScoreCellProps> = ({ z, count, label, domain, signalType }) => {
  const colors = zToColor(z);

  return (
    <td
      role="cell"
      aria-label={`${domain} ${signalType}: z-score ${z.toFixed(1)}, severity ${label}`}
      style={{
        backgroundColor: colors.bg,
        border: `1px solid ${colors.border}`,
        padding: '6px 4px',
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'filter 0.15s',
        minWidth: '56px',
      }}
      title={`${domain} ${signalType}: z=${z.toFixed(1)} (${label}), ${count} signals`}
      onMouseEnter={(e) => { (e.target as HTMLElement).style.filter = 'brightness(1.15)'; }}
      onMouseLeave={(e) => { (e.target as HTMLElement).style.filter = 'none'; }}
    >
      <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '11px', fontWeight: 'bold', color: colors.text }}>
        {z.toFixed(1)}
      </div>
      <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '7px', color: '#444' }}>
        {count}
      </div>
    </td>
  );
};
