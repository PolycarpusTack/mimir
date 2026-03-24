import { describe, it, expect } from 'vitest';
import { apiFetch, ApiError } from '@/api/client';

describe('apiFetch', () => {
  it('fetches data from mock API', async () => {
    const data = await apiFetch<{ status: string }>('/health');
    expect(data.status).toBe('ok');
  });

  it('throws ApiError for 404', async () => {
    await expect(apiFetch('/signals/nonexistent')).rejects.toThrow(ApiError);
  });

  it('includes query params', async () => {
    const data = await apiFetch<{ items: unknown[] }>('/signals', { domains: 'AI', page: 1 });
    expect(data).toHaveProperty('items');
  });

  it('skips undefined params', async () => {
    const data = await apiFetch<{ items: unknown[] }>('/signals', { domains: undefined, page: 1 });
    expect(data).toHaveProperty('items');
  });
});

describe('ApiError', () => {
  it('has correct status and message', () => {
    const err = new ApiError(404, 'Not found');
    expect(err.status).toBe(404);
    expect(err.message).toBe('Not found');
    expect(err.name).toBe('ApiError');
  });
});
