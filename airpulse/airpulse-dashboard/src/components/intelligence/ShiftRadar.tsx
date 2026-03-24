import React from 'react';
import { useShiftAlerts } from '@/api/shifts';
import { ZScoreCell } from '@/components/primitives/ZScoreCell';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { DOMAINS, DOMAIN_COLORS } from '@/constants';
import { SIGNAL_TYPES, SIGNAL_TYPE_CONFIG } from '@/constants';
import { Z_THRESHOLD_COLORS } from '@/constants';
import { zToColor } from '@/utils';
import type { Domain, SignalType } from '@/types';

export const ShiftRadar: React.FC = () => {
  const { data: alerts, isLoading, isError, refetch } = useShiftAlerts();

  // Build lookup: domain+signalType -> alert
  const alertLookup = new Map<string, { z: number; count: number; label: string }>();
  if (alerts) {
    for (const alert of alerts) {
      const key = `${alert.domain}:${alert.signal_type}`;
      alertLookup.set(key, {
        z: alert.z_score,
        count: alert.signal_count,
        label: alert.severity.toUpperCase(),
      });
    }
  }

  const activeSurges = (alerts ?? []).filter((a) => a.z_score >= 2.5 && !a.resolved_at);

  if (isError) {
    return (
      <div role="alert" style={{ padding: '24px', textAlign: 'center' }}>
        <div style={{ fontFamily: 'DM Serif Display, serif', fontSize: '14px', color: '#ff7b7b', marginBottom: '8px' }}>
          Failed to load shift data
        </div>
        <button onClick={() => refetch()} style={{
          fontFamily: 'Syne Mono, monospace', fontSize: '10px', padding: '6px 16px',
          backgroundColor: 'transparent', border: '1px solid #ff7b7b', color: '#ff7b7b',
          borderRadius: '4px', cursor: 'pointer',
        }}>Retry</button>
      </div>
    );
  }

  if (isLoading) {
    return <div style={{ padding: '16px' }}><LoadingSkeleton height={300} /></div>;
  }

  // Show only 6 signal types in the grid to fit 6x6
  const gridSignalTypes: SignalType[] = SIGNAL_TYPES.slice(0, 6);

  return (
    <div id="panel-shift" role="tabpanel" style={{ padding: '16px', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <h2 style={{ fontFamily: 'DM Serif Display, serif', fontSize: '18px', color: '#d0d0d0', margin: 0 }}>
          Shift Radar
        </h2>
        <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', marginTop: '4px' }}>
          Z-SCORE VS 90-DAY BASELINE &middot; 7-DAY WINDOW
        </div>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
        {Z_THRESHOLD_COLORS.map((t) => (
          <div key={t.label} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '10px', height: '10px', backgroundColor: t.bg, border: `1px solid ${t.border}`, borderRadius: '2px' }} />
            <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: t.text }}>{t.label}</span>
          </div>
        ))}
      </div>

      {/* Heatmap grid */}
      <table style={{ borderCollapse: 'separate', borderSpacing: '3px', width: '100%', marginBottom: '16px' }}>
        <thead>
          <tr>
            <th />
            {gridSignalTypes.map((st) => (
              <th key={st} style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', textAlign: 'center', padding: '4px' }}>
                {SIGNAL_TYPE_CONFIG[st].label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {DOMAINS.map((domain: Domain) => (
            <tr key={domain}>
              <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: DOMAIN_COLORS[domain].text, padding: '4px 8px 4px 0', textAlign: 'right', whiteSpace: 'nowrap' }}>
                {domain}
              </td>
              {gridSignalTypes.map((st) => {
                const key = `${domain}:${st}`;
                const cell = alertLookup.get(key);
                const z = cell?.z ?? 0;
                const count = cell?.count ?? 0;
                const label = cell?.label ?? zToColor(z).label;
                return (
                  <ZScoreCell
                    key={key}
                    z={z}
                    count={count}
                    label={label}
                    domain={domain}
                    signalType={st}
                  />
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Active surges callout */}
      {activeSurges.length > 0 && (
        <div style={{
          backgroundColor: 'rgba(255,96,96,0.08)', border: '1px solid rgba(255,96,96,0.25)',
          borderRadius: '6px', padding: '12px',
        }}>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#ff6060', marginBottom: '8px', letterSpacing: '0.5px' }}>
            ACTIVE SURGES & SPIKES
          </div>
          {activeSurges.map((a) => (
            <div key={a.id} style={{ marginBottom: '6px' }}>
              <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: a.severity === 'Surge' ? '#ff6060' : '#ffa032', fontWeight: 'bold' }}>
                {a.domain} &times; {SIGNAL_TYPE_CONFIG[a.signal_type].label}
              </span>
              <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#888', marginLeft: '8px' }}>
                z={a.z_score.toFixed(1)} &middot; {a.signal_count} signals
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
