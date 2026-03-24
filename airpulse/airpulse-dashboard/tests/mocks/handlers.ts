import { http, HttpResponse, delay } from 'msw';
import {
  FIXTURE_SIGNALS,
  FIXTURE_SIGNAL_PAGE,
  FIXTURE_ENRICHMENTS,
  FIXTURE_SHIFT_ALERTS,
  FIXTURE_SOURCES,
  FIXTURE_HEALTH,
  FIXTURE_QUEUE,
  FIXTURE_STATS,
  FIXTURE_DIGESTS,
  FIXTURE_DIGEST_DETAIL,
  FIXTURE_APPROVAL_QUEUE,
} from './fixtures';
import type { Domain } from '@/types';

export const handlers = [
  // MSW-001: GET /api/v1/signals
  http.get('/api/v1/signals', ({ request }) => {
    const url = new URL(request.url);
    const domainFilter = url.searchParams.get('domains') as Domain | null;
    const enrichedOnly = url.searchParams.get('enriched_only') === 'true';

    let filtered = [...FIXTURE_SIGNALS];

    if (domainFilter) {
      filtered = filtered.filter((s) => s.domains.includes(domainFilter));
    }
    if (enrichedOnly) {
      filtered = filtered.filter((s) => s.enriched);
    }

    return HttpResponse.json({
      ...FIXTURE_SIGNAL_PAGE,
      items: filtered,
      total: filtered.length,
    });
  }),

  // MSW-002: GET /api/v1/signals/:id
  http.get('/api/v1/signals/:id', ({ params }) => {
    const signal = FIXTURE_SIGNALS.find((s) => s.id === params.id);
    if (!signal) {
      return new HttpResponse(JSON.stringify({ error: 'not_found', message: 'Signal not found' }), { status: 404 });
    }
    return HttpResponse.json(signal);
  }),

  // MSW-003: GET /api/v1/signals/:id/enrichment
  http.get('/api/v1/signals/:id/enrichment', async ({ params }) => {
    await delay(100);
    const enrichment = FIXTURE_ENRICHMENTS[params.id as string];
    if (!enrichment) {
      return new HttpResponse(JSON.stringify({ error: 'not_found', message: 'Enrichment not found' }), { status: 404 });
    }
    return HttpResponse.json(enrichment);
  }),

  // MSW-004: PATCH /api/v1/signals/:id
  http.patch('/api/v1/signals/:id', async ({ params }) => {
    const signal = FIXTURE_SIGNALS.find((s) => s.id === params.id);
    if (!signal) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json({ ...signal, archived: true });
  }),

  // MSW-005: GET /api/v1/shifts
  http.get('/api/v1/shifts', ({ request }) => {
    const url = new URL(request.url);
    const domainFilter = url.searchParams.get('domain');
    const severityFilter = url.searchParams.get('severity');

    let filtered = [...FIXTURE_SHIFT_ALERTS];
    if (domainFilter) {
      filtered = filtered.filter((a) => a.domain.toLowerCase() === domainFilter.toLowerCase());
    }
    if (severityFilter) {
      filtered = filtered.filter((a) => a.severity === severityFilter);
    }

    return HttpResponse.json({
      alerts: filtered,
      total: filtered.length,
      as_of: new Date().toISOString(),
    });
  }),

  // MSW-006: GET /api/v1/sources
  http.get('/api/v1/sources', () => {
    return HttpResponse.json(FIXTURE_SOURCES);
  }),

  // MSW-007: POST /api/v1/sources/:id/poll
  http.post('/api/v1/sources/:id/poll', ({ params }) => {
    return HttpResponse.json(
      { message: 'Poll scheduled', source_id: params.id },
      { status: 202 },
    );
  }),

  // MSW-008: GET /api/v1/health
  http.get('/api/v1/health', () => {
    return HttpResponse.json(FIXTURE_HEALTH);
  }),

  // MSW-009: GET /api/v1/enrichment/queue
  http.get('/api/v1/enrichment/queue', () => {
    return HttpResponse.json(FIXTURE_QUEUE);
  }),

  // MSW-010: GET /api/v1/stats
  http.get('/api/v1/stats', () => {
    return HttpResponse.json(FIXTURE_STATS);
  }),

  // --- Phase 4 handlers ---

  // MSW-P4-001: PATCH /api/v1/signals/:id
  http.patch('/api/v1/signals/:id', async ({ params, request }) => {
    const signal = FIXTURE_SIGNALS.find((s) => s.id === params.id);
    if (!signal) return new HttpResponse(null, { status: 404 });
    const body = await request.json() as Record<string, unknown>;
    if (body.title || body.url || body.content_hash) {
      return HttpResponse.json(
        { error: 'bad_request', message: 'title is immutable after creation' },
        { status: 400 },
      );
    }
    return HttpResponse.json({ ...signal, ...body });
  }),

  // MSW-P4-002: GET /api/v1/digest/latest
  http.get('/api/v1/digest/latest', () => {
    return HttpResponse.json(FIXTURE_DIGESTS[0]);
  }),

  // MSW-P4-003: GET /api/v1/digest
  http.get('/api/v1/digest', () => {
    return HttpResponse.json(FIXTURE_DIGESTS);
  }),

  // MSW-P4-004: GET /api/v1/digest/:id
  http.get('/api/v1/digest/:id', ({ params }) => {
    if (params.id === FIXTURE_DIGEST_DETAIL.id) {
      return HttpResponse.json(FIXTURE_DIGEST_DETAIL);
    }
    return new HttpResponse(null, { status: 404 });
  }),

  // MSW-P4-005: GET /api/v1/digest/:id/download
  http.get('/api/v1/digest/:id/download', () => {
    return new HttpResponse(new Uint8Array([0x50, 0x4B, 0x03, 0x04]), {
      headers: { 'Content-Type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' },
    });
  }),

  // MSW-P4-006: POST /api/v1/digest/generate
  http.post('/api/v1/digest/generate', () => {
    return HttpResponse.json(
      { message: 'Digest generation started', job_id: 'job-001' },
      { status: 202 },
    );
  }),

  // MSW-P4-007: POST /api/v1/jira/push
  http.post('/api/v1/jira/push', async () => {
    return HttpResponse.json(
      {
        ...FIXTURE_APPROVAL_QUEUE[0],
        id: 'aq-new-001',
        status: 'Pending',
        jira_issue_key: null,
      },
      { status: 201 },
    );
  }),

  // MSW-P4-008: GET /api/v1/jira/queue
  http.get('/api/v1/jira/queue', ({ request }) => {
    const url = new URL(request.url);
    const status = url.searchParams.get('status');
    let items = [...FIXTURE_APPROVAL_QUEUE];
    if (status) {
      items = items.filter((i) => i.status === status);
    }
    return HttpResponse.json(items);
  }),

  // MSW-P4-009: GET /api/v1/jira/queue/:id
  http.get('/api/v1/jira/queue/:id', ({ params }) => {
    const item = FIXTURE_APPROVAL_QUEUE.find((i) => i.id === params.id);
    if (!item) return new HttpResponse(null, { status: 404 });
    return HttpResponse.json(item);
  }),

  // MSW-P4-010: DELETE /api/v1/jira/queue/:id
  http.delete('/api/v1/jira/queue/:id', ({ params }) => {
    const item = FIXTURE_APPROVAL_QUEUE.find((i) => i.id === params.id);
    if (!item) return new HttpResponse(null, { status: 404 });
    if (item.status !== 'Pending') {
      return HttpResponse.json(
        { error: 'conflict', message: `Cannot cancel item in ${item.status} state` },
        { status: 409 },
      );
    }
    return new HttpResponse(null, { status: 204 });
  }),
];
