import React, { useMemo } from 'react';
import { useSignals } from '@/api/signals';
import { SparkLine } from '@/components/primitives/SparkLine';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import { TECH_TERM_WATCHLIST } from '@/constants';

type Velocity = 'Surging' | 'Emerging' | 'Rising' | 'Stable' | 'Declining';

function classifyVelocity(delta: number, mentions7d: number): { label: Velocity; color: string } {
  if (delta >= 30 && mentions7d >= 10) return { label: 'Surging', color: '#ff7b7b' };
  if (delta >= 50 && mentions7d < 10) return { label: 'Emerging', color: '#ffc95d' };
  if (delta >= 10 && delta < 30) return { label: 'Rising', color: '#5dde9f' };
  if (delta > -10 && delta < 10) return { label: 'Stable', color: '#5dc8de' };
  return { label: 'Declining', color: '#888' };
}

export const TechTrends: React.FC = () => {
  const eightWeeksAgo = useMemo(() => new Date(Date.now() - 56 * 86400_000).toISOString(), []);
  const { data, isLoading } = useSignals({ page_size: 200, after: eightWeeksAgo });

  const techStats = useMemo(() => {
    if (!data?.items) return [];
    const sevenDaysAgo = Date.now() - 7 * 86400_000;

    return TECH_TERM_WATCHLIST.map((entry) => {
      const mentions = data.items.filter((s) => {
        const text = `${s.title} ${s.summary ?? ''}`.toLowerCase();
        return text.includes(entry.term.toLowerCase());
      });

      const mentions7d = mentions.filter((s) => new Date(s.published_at).getTime() >= sevenDaysAgo).length;
      const mentionsOlder = mentions.length - mentions7d;
      const olderWeeks = 7;
      const weeklyRatePrior = mentionsOlder / olderWeeks;
      const delta = weeklyRatePrior > 0 ? ((mentions7d - weeklyRatePrior) / weeklyRatePrior) * 100 : 0;
      const velocity = classifyVelocity(delta, mentions7d);

      // 7-week sparkline
      const weeklyTrend: number[] = [];
      for (let w = 6; w >= 0; w--) {
        const weekStart = Date.now() - (w + 1) * 7 * 86400_000;
        const weekEnd = Date.now() - w * 7 * 86400_000;
        weeklyTrend.push(
          mentions.filter((s) => {
            const t = new Date(s.published_at).getTime();
            return t >= weekStart && t < weekEnd;
          }).length,
        );
      }

      return {
        ...entry,
        mentions_7d: mentions7d,
        delta,
        velocity,
        weeklyTrend,
      };
    })
      .sort((a, b) => b.mentions_7d - a.mentions_7d);
  }, [data]);

  if (isLoading) {
    return <div style={{ padding: '16px' }}><LoadingSkeleton height={400} /></div>;
  }

  return (
    <div id="panel-tech" role="tabpanel" style={{ padding: '16px', overflowY: 'auto' }}>
      <h2 style={{ fontFamily: 'DM Serif Display, serif', fontSize: '18px', color: '#d0d0d0', margin: '0 0 12px 0' }}>
        Tech Trends
      </h2>

      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: '#444', textTransform: 'uppercase' }}>
            <th style={{ textAlign: 'left', padding: '6px 8px' }}>Term</th>
            <th style={{ textAlign: 'left', padding: '6px 4px' }}>Cluster</th>
            <th style={{ textAlign: 'right', padding: '6px 4px' }}>7D</th>
            <th style={{ textAlign: 'right', padding: '6px 4px' }}>Delta</th>
            <th style={{ textAlign: 'center', padding: '6px 4px' }}>Trend</th>
            <th style={{ textAlign: 'center', padding: '6px 4px' }}>Velocity</th>
          </tr>
        </thead>
        <tbody>
          {techStats.map((t) => (
            <tr key={t.term} style={{ borderBottom: '1px solid #111820' }}>
              <td style={{ fontFamily: 'Syne, sans-serif', fontSize: '11px', color: '#d0d0d0', padding: '8px' }}>
                {t.term}
              </td>
              <td>
                <span style={{
                  fontFamily: 'Syne Mono, monospace', fontSize: '8px',
                  color: '#5dc8de', backgroundColor: 'rgba(93,200,222,0.1)',
                  padding: '2px 5px', borderRadius: '3px',
                }}>
                  {t.cluster}
                </span>
              </td>
              <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '11px', color: '#d0d0d0', textAlign: 'right', padding: '8px 4px' }}>
                {t.mentions_7d}
              </td>
              <td style={{ fontFamily: 'Syne Mono, monospace', fontSize: '10px', color: t.velocity.color, textAlign: 'right', padding: '8px 4px' }}>
                {t.delta >= 0 ? '+' : ''}{t.delta.toFixed(0)}%
              </td>
              <td style={{ textAlign: 'center', padding: '8px 4px' }}>
                <SparkLine data={t.weeklyTrend} color={t.velocity.color} />
              </td>
              <td style={{ textAlign: 'center', padding: '8px 4px' }}>
                <span style={{
                  fontFamily: 'Syne Mono, monospace', fontSize: '8px', color: t.velocity.color,
                  backgroundColor: `${t.velocity.color}15`, padding: '2px 6px', borderRadius: '3px',
                  textTransform: 'uppercase',
                }}>
                  {t.velocity.label}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
