import React, { useMemo } from 'react';
import { useSignals } from '@/api/signals';
import { SparkLine } from '@/components/primitives/SparkLine';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { VENDOR_WATCHLIST } from '@/constants';

export const VendorTracker: React.FC = () => {
  const thirtyDaysAgo = useMemo(() => new Date(Date.now() - 30 * 86400_000).toISOString(), []);
  const { data, isLoading } = useSignals({ page_size: 200, after: thirtyDaysAgo });

  const vendorStats = useMemo(() => {
    if (!data?.items) return [];
    const sevenDaysAgo = Date.now() - 7 * 86400_000;

    return VENDOR_WATCHLIST.map((vendor) => {
      const mentions = data.items.filter((s) => {
        const text = `${s.title} ${s.summary ?? ''}`.toLowerCase();
        return text.includes(vendor.name.toLowerCase());
      });

      const mentions7d = mentions.filter((s) => new Date(s.published_at).getTime() >= sevenDaysAgo);
      const mentions30d = mentions;

      const dailyRate7d = mentions7d.length / 7;
      const dailyRate30d = mentions30d.length / 30;
      const delta = dailyRate30d > 0 ? ((dailyRate7d - dailyRate30d) / dailyRate30d) * 100 : 0;

      // Build 7-day trend
      const trend: number[] = [];
      for (let i = 6; i >= 0; i--) {
        const dayStart = Date.now() - (i + 1) * 86400_000;
        const dayEnd = Date.now() - i * 86400_000;
        trend.push(
          mentions.filter((s) => {
            const t = new Date(s.published_at).getTime();
            return t >= dayStart && t < dayEnd;
          }).length,
        );
      }

      return {
        ...vendor,
        mentions_7d: mentions7d.length,
        mentions_30d: mentions30d.length,
        delta,
        trend,
      };
    })
      .filter((v) => v.mentions_30d > 0)
      .sort((a, b) => b.mentions_7d - a.mentions_7d);
  }, [data]);

  if (isLoading) {
    return <div style={{ padding: '16px' }}><LoadingSkeleton height={400} /></div>;
  }

  return (
    <div id="panel-vendors" role="tabpanel" style={{ padding: '16px', overflowY: 'auto' }}>
      <h2 style={{ fontFamily: 'DM Serif Display, serif', fontSize: '18px', color: '#d0d0d0', margin: '0 0 12px 0' }}>
        Vendor Tracker
      </h2>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', textTransform: 'uppercase' }}>
            <th style={{ textAlign: 'left', padding: '6px 8px' }}>Vendor</th>
            <th style={{ textAlign: 'right', padding: '6px 4px' }}>7D</th>
            <th style={{ textAlign: 'right', padding: '6px 4px' }}>30D</th>
            <th style={{ textAlign: 'right', padding: '6px 4px' }}>Delta</th>
            <th style={{ textAlign: 'center', padding: '6px 4px' }}>Trend</th>
          </tr>
        </thead>
        <tbody>
          {vendorStats.map((v) => {
            const deltaColor = v.delta > 10 ? '#ff7b7b' : v.delta > 0 ? '#5dde9f' : v.delta < -10 ? '#888' : '#5dc8de';
            const tierColor = v.tier === 'competitor' ? '#ff7b7b' : v.tier === 'partner' ? '#5dde9f' : '#5dc8de';
            return (
              <tr key={v.name} style={{ borderBottom: '1px solid #111820' }}>
                <td style={{ padding: '8px' }}>
                  <div style={{ fontFamily: 'Syne, sans-serif', fontSize: '11px', color: '#d0d0d0' }}>{v.name}</div>
                  <span style={{
                    fontFamily: 'Syne Mono, monospace', fontSize: '7px', color: tierColor,
                    textTransform: 'uppercase',
                  }}>
                    {v.tier}
                  </span>
                </td>
                <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '11px', color: '#d0d0d0', textAlign: 'right', padding: '8px 4px' }}>
                  {v.mentions_7d}
                </td>
                <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '11px', color: '#888', textAlign: 'right', padding: '8px 4px' }}>
                  {v.mentions_30d}
                </td>
                <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: deltaColor, textAlign: 'right', padding: '8px 4px' }}>
                  {v.delta >= 0 ? '+' : ''}{v.delta.toFixed(0)}%
                </td>
                <td style={{ textAlign: 'center', padding: '8px 4px' }}>
                  <SparkLine data={v.trend} color={deltaColor} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
