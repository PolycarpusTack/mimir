import { describe, it, expect } from 'vitest';
import { formatDate, formatClock, zToColor, truncate, relevanceColor, isFresh } from '@/utils';

describe('formatDate', () => {
  it('returns "Just now" for recent dates', () => {
    expect(formatDate(new Date().toISOString())).toBe('Just now');
  });

  it('returns minutes ago', () => {
    const tenMinAgo = new Date(Date.now() - 10 * 60000).toISOString();
    expect(formatDate(tenMinAgo)).toBe('10m ago');
  });

  it('returns hours ago', () => {
    const threeHrAgo = new Date(Date.now() - 3 * 3600000).toISOString();
    expect(formatDate(threeHrAgo)).toBe('3h ago');
  });

  it('returns date for old items', () => {
    const result = formatDate('2025-01-15T00:00:00Z');
    expect(result).toMatch(/Jan 15/);
  });
});

describe('formatClock', () => {
  it('returns time string', () => {
    const clock = formatClock();
    expect(clock).toMatch(/\d{2}:\d{2}:\d{2}/);
  });
});

describe('zToColor', () => {
  it('returns SURGE for z >= 3.5', () => {
    expect(zToColor(3.7).label).toBe('SURGE');
  });

  it('returns SPIKE for z >= 2.5', () => {
    expect(zToColor(2.8).label).toBe('SPIKE');
  });

  it('returns ELEVATED for z >= 1.5', () => {
    expect(zToColor(1.8).label).toBe('ELEVATED');
  });

  it('returns ABOVE for z >= 0.8', () => {
    expect(zToColor(0.9).label).toBe('ABOVE');
  });

  it('returns NORMAL for z < 0.8', () => {
    expect(zToColor(0.3).label).toBe('NORMAL');
  });
});

describe('truncate', () => {
  it('does not truncate short text', () => {
    expect(truncate('hello', 10)).toBe('hello');
  });

  it('truncates long text with ellipsis', () => {
    expect(truncate('hello world!', 8)).toBe('hello w\u2026');
  });
});

describe('relevanceColor', () => {
  it('returns red for >= 85', () => {
    expect(relevanceColor(92)).toBe('#ff7b7b');
  });

  it('returns amber for >= 70', () => {
    expect(relevanceColor(75)).toBe('#ffc95d');
  });

  it('returns green for < 70', () => {
    expect(relevanceColor(50)).toBe('#5dde9f');
  });
});

describe('isFresh', () => {
  it('returns true for recently fetched items', () => {
    expect(isFresh(new Date().toISOString())).toBe(true);
  });

  it('returns false for old items', () => {
    expect(isFresh('2024-01-01T00:00:00Z')).toBe(false);
  });
});
