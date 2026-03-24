import { Z_THRESHOLD_COLORS } from '@/constants';

export function formatDate(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMs / 3600000);

  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;

  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export function formatClock(): string {
  return new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

export function zToColor(z: number): { bg: string; border: string; text: string; label: string } {
  for (const threshold of Z_THRESHOLD_COLORS) {
    if (z >= threshold.minZ) {
      return {
        bg: threshold.bg,
        border: threshold.border,
        text: threshold.text,
        label: threshold.label,
      };
    }
  }
  return { bg: Z_THRESHOLD_COLORS[4].bg, border: Z_THRESHOLD_COLORS[4].border, text: Z_THRESHOLD_COLORS[4].text, label: 'NORMAL' };
}

export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '\u2026';
}

export function relevanceColor(score: number): string {
  if (score >= 85) return '#ff7b7b';
  if (score >= 70) return '#ffc95d';
  return '#5dde9f';
}

export function isFresh(fetchedAt: string): boolean {
  const diff = Date.now() - new Date(fetchedAt).getTime();
  return diff < 2 * 60 * 1000;
}
