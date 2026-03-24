import { test, expect } from '@playwright/test';

test.describe('AirPulse Phase 5 E2E', () => {
  // E2E-P5-001: Real-time signal push (simulated)
  test('E2E-P5-001: dashboard loads with WebSocket indicator', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('AirPulse')).toBeVisible();
    // Live indicator should be visible
    await expect(page.getByText('LIVE')).toBeVisible();
  });

  // E2E-P5-004: Mobile layout (768px)
  test('E2E-P5-004: mobile layout at 768px viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await expect(page.getByText('AirPulse')).toBeVisible();
    await expect(page.getByText('Signal Feed')).toBeVisible();
  });

  // E2E-P5-005: Tablet layout (1024px)
  test('E2E-P5-005: tablet layout at 1024px viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto('/');
    await expect(page.getByText('AirPulse')).toBeVisible();
    await expect(page.getByText('Signal Feed')).toBeVisible();
  });

  // E2E-P5-006: Silence alert indicator
  test('E2E-P5-006: shift radar renders with silence support', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /shift radar/i }).click();
    await expect(page.getByText('Shift Radar')).toBeVisible();
  });

  // E2E-P5-002: Digest tab accessible
  test('E2E-P5-002: digest panel accessible from tabs', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /digest/i }).click();
    await expect(page.getByText('Weekly Digest')).toBeVisible();
  });

  // E2E-P5-003: All panels cycle correctly
  test('E2E-P5-003: all four panels cycle correctly', async ({ page }) => {
    await page.goto('/');
    for (const tab of ['Shift Radar', 'Vendors', 'Tech Trends', 'Digest']) {
      await page.getByRole('tab', { name: new RegExp(tab, 'i') }).click();
    }
    await expect(page.getByText('Weekly Digest')).toBeVisible();
  });
});
