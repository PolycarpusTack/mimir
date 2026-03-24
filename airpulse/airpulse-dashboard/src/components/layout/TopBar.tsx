import React, { useEffect, useState } from 'react';
import { useHealth } from '@/api/health';
import { useUIStore, type ActivePanel } from '@/store/ui';
import { formatClock } from '@/utils';

const TABS: { key: ActivePanel; label: string }[] = [
  { key: 'shift', label: 'Shift Radar' },
  { key: 'vendors', label: 'Vendors' },
  { key: 'tech', label: 'Tech Trends' },
];

export const TopBar: React.FC = () => {
  const { data: health } = useHealth();
  const activePanel = useUIStore((s) => s.activePanel);
  const setActivePanel = useUIStore((s) => s.setActivePanel);
  const [clock, setClock] = useState(formatClock());

  useEffect(() => {
    const timer = setInterval(() => setClock(formatClock()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header
      style={{
        height: '56px',
        backgroundColor: '#0a0e14',
        borderBottom: '1px solid #1e2530',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: '20px',
      }}
    >
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div
          style={{
            width: '28px',
            height: '28px',
            background: 'linear-gradient(135deg, #e8a830, #ffc95d)',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'Syne, sans-serif',
            fontWeight: 700,
            fontSize: '12px',
            color: '#0a0e14',
          }}
        >
          AP
        </div>
        <div>
          <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: '15px', color: '#d0d0d0' }}>
            AirPulse
          </div>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: '#444', letterSpacing: '2px' }}>
            INTELLIGENCE
          </div>
        </div>
      </div>

      {/* Nav tabs */}
      <nav role="tablist" style={{ display: 'flex', gap: '2px', marginLeft: '20px' }}>
        {TABS.map((tab) => {
          const isActive = activePanel === tab.key;
          return (
            <button
              key={tab.key}
              role="tab"
              aria-selected={isActive}
              aria-controls={`panel-${tab.key}`}
              onClick={() => setActivePanel(tab.key)}
              style={{
                fontFamily: 'Syne Mono, monospace',
                fontSize: '10px',
                padding: '8px 14px',
                backgroundColor: isActive ? 'rgba(232,168,48,0.1)' : 'transparent',
                border: 'none',
                borderBottom: isActive ? '1px solid #e8a830' : '1px solid transparent',
                color: isActive ? '#e8a830' : '#555',
                cursor: 'pointer',
                textTransform: 'uppercase',
                letterSpacing: '1px',
              }}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Metric tiles */}
      <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
        <MetricTile label="SIGNALS TODAY" value={health?.signals_today ?? 0} color="#5dde9f" />
        <MetricTile label="SURGES ACTIVE" value={health?.surges_active ?? 0} color={(health?.surges_active ?? 0) > 0 ? '#ff7b7b' : '#5dde9f'} />
        <MetricTile label="SOURCES LIVE" value={health?.sources_live ?? 0} color={health?.status === 'degraded' ? '#ffc95d' : '#5dc8de'} />
        <MetricTile label="QUEUE DEPTH" value={health?.queue_depth ?? 0} color="#e8a830" />
        <div role="status" aria-label={`Clock: ${clock}`} aria-live="off">
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', textTransform: 'uppercase' }}>UTC</div>
          <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '13px', color: '#888' }}>{clock}</div>
        </div>
      </div>
    </header>
  );
};

const MetricTile: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div role="status" aria-label={`${label}: ${value}`}>
    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '9px', color: '#444', textTransform: 'uppercase' }}>{label}</div>
    <div style={{ fontFamily: 'Syne Mono, monospace', fontSize: '13px', color }}>{value}</div>
  </div>
);
