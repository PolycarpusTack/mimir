import React, { useEffect, useState } from 'react';
import type { Domain } from '@/types';
import { DOMAINS, DOMAIN_COLORS } from '@/constants';
import { useFilterStore } from '@/store/filters';

export const FilterStrip: React.FC = () => {
  const filterDomain = useFilterStore((s) => s.filterDomain);
  const setFilterDomain = useFilterStore((s) => s.setFilterDomain);
  const [isLive, setIsLive] = useState(true);

  // Sync filter state with URL params
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const domain = params.get('domain') as Domain | null;
    if (domain && DOMAINS.includes(domain)) {
      setFilterDomain(domain);
    }
  }, [setFilterDomain]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (filterDomain) {
      params.set('domain', filterDomain);
    } else {
      params.delete('domain');
    }
    const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`;
    window.history.replaceState(null, '', newUrl);
  }, [filterDomain]);

  // Pulse indicator
  useEffect(() => {
    const timer = setInterval(() => setIsLive((v) => !v), 3000);
    return () => clearInterval(timer);
  }, []);

  const handleDomainClick = (domain: Domain | null) => {
    if (domain === filterDomain) {
      setFilterDomain(null);
    } else {
      setFilterDomain(domain);
    }
  };

  return (
    <div
      role="toolbar"
      aria-label="Domain filter"
      style={{
        height: '36px',
        backgroundColor: '#0a0e14',
        borderBottom: '1px solid #111820',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        gap: '6px',
      }}
    >
      <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#333', marginRight: '6px', textTransform: 'uppercase' }}>
        FILTER
      </span>

      <FilterPill
        label="ALL"
        active={filterDomain === null}
        onClick={() => handleDomainClick(null)}
        ariaLabel="Show all domains"
      />

      {DOMAINS.map((domain) => {
        const colors = DOMAIN_COLORS[domain];
        const active = filterDomain === domain;
        return (
          <FilterPill
            key={domain}
            label={domain.toUpperCase()}
            active={active}
            textColor={active ? colors.text : '#3d4555'}
            borderColor={active ? colors.border : '#1e2530'}
            bgColor={active ? colors.bg : 'transparent'}
            dotColor={active ? colors.dot : '#2a3040'}
            onClick={() => handleDomainClick(domain)}
            ariaLabel={`Filter by ${domain}`}
          />
        );
      })}

      <div style={{ flex: 1 }} />

      {/* Live indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <div
          style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            backgroundColor: isLive ? '#5dde9f' : '#2d6040',
            boxShadow: isLive ? '0 0 6px #5dde9f' : 'none',
            transition: 'all 0.5s',
          }}
        />
        <span style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#5dde9f' }}>LIVE</span>
      </div>
    </div>
  );
};

interface FilterPillProps {
  label: string;
  active: boolean;
  textColor?: string;
  borderColor?: string;
  bgColor?: string;
  dotColor?: string;
  onClick: () => void;
  ariaLabel: string;
}

const FilterPill: React.FC<FilterPillProps> = ({
  label,
  active,
  textColor = active ? '#e8a830' : '#3d4555',
  borderColor = active ? '#e8a83050' : '#1e2530',
  bgColor = active ? 'rgba(232,168,48,0.08)' : 'transparent',
  dotColor,
  onClick,
  ariaLabel,
}) => (
  <button
    role="button"
    aria-pressed={active}
    aria-label={ariaLabel}
    onClick={onClick}
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: '4px',
      fontFamily: 'Syne Mono, monospace',
      fontSize: '9px',
      padding: '3px 8px',
      backgroundColor: bgColor,
      border: `1px solid ${borderColor}`,
      color: textColor,
      borderRadius: '3px',
      cursor: 'pointer',
      letterSpacing: '0.5px',
    }}
  >
    {dotColor && (
      <span
        style={{
          width: '4px',
          height: '4px',
          borderRadius: '50%',
          backgroundColor: dotColor,
          display: 'inline-block',
        }}
      />
    )}
    {label}
  </button>
);
