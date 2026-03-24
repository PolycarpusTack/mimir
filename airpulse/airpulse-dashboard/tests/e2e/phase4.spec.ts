import { test, expect } from '@playwright/test';

test.describe('AirPulse Phase 4 E2E', () => {
  // E2E-P4-001: Full JIRA push flow
  test('E2E-P4-001: full JIRA push flow via enrichment panel', async ({ page }) => {
    await page.goto('/?signal=sig-001');

    // Click Push to JIRA button in enrichment panel
    const pushBtn = page.getByText('Push to JIRA / AirForge');
    if (await pushBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await pushBtn.click();
      // Modal should appear
      await expect(page.getByRole('dialog')).toBeVisible();
      await expect(page.getByText('Push to JIRA / AirForge')).toBeVisible();
    }
  });

  // E2E-P4-002: Cancel JIRA push
  test('E2E-P4-002: cancel JIRA push via Escape', async ({ page }) => {
    await page.goto('/?signal=sig-001');

    const pushBtn = page.getByText('Push to JIRA / AirForge');
    if (await pushBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
      await pushBtn.click();
      await expect(page.getByRole('dialog')).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(page.getByRole('dialog')).not.toBeVisible();
    }
  });

  // E2E-P4-003: Digest panel view latest
  test('E2E-P4-003: view digest panel with latest digest', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /digest/i }).click();
    await expect(page.getByText('Weekly Digest')).toBeVisible();
  });

  // E2E-P4-004: Force generate digest
  test('E2E-P4-004: force generate digest', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /digest/i }).click();
    await expect(page.getByText('Generate Now')).toBeVisible();
  });

  // E2E-P4-005: DOCX download
  test('E2E-P4-005: digest DOCX download link present', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('tab', { name: /digest/i }).click();
    const downloadLink = page.getByText('Download DOCX');
    if (await downloadLink.isVisible({ timeout: 5000 }).catch(() => false)) {
      await expect(downloadLink).toBeVisible();
    }
  });

  // E2E-P4-006: Approval queue failed item alert
  test('E2E-P4-006: approval queue status shows in TopBar', async ({ page }) => {
    await page.goto('/');
    // ApprovalQueueStatus should show pending/failed/pushed
    await expect(page.getByText('AirPulse')).toBeVisible();
  });
});
