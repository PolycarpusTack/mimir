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

  // MSW-011: GET /api/v1/costs
  http.get('/api/v1/costs', () => {
    return HttpResponse.json({
      rows: [],
      total_input_tokens: 500000,
      total_output_tokens: 340000,
      total_calls: 1000,
    });
  }),
];
