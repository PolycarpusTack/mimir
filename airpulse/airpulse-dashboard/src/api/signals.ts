import { useQuery } from '@tanstack/react-query';
import { apiFetch, ApiError } from './client';
import type { Signal, SignalPage, Domain } from '@/types';
import type { EnrichedAnnotation } from '@/types';

interface SignalQueryParams {
  domain?: Domain | null;
  page?: number;
  page_size?: number;
  min_relevance?: number;
  enriched_only?: boolean;
  after?: string;
}

export function useSignals(params: SignalQueryParams = {}) {
  return useQuery({
    queryKey: ['signals', params],
    queryFn: () =>
      apiFetch<SignalPage>('/signals', {
        domains: params.domain ?? undefined,
        page: params.page ?? 1,
        page_size: params.page_size ?? 50,
        min_confidence: params.min_relevance ? params.min_relevance / 100 : undefined,
        enriched_only: params.enriched_only ?? undefined,
        after: params.after ?? undefined,
      }),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useSignal(id: string | null) {
  return useQuery({
    queryKey: ['signal', id],
    queryFn: () => apiFetch<Signal>(`/signals/${id}`),
    enabled: !!id,
    staleTime: 120_000,
  });
}

export function useSignalEnrichment(id: string | null) {
  return useQuery({
    queryKey: ['enrichment', id],
    queryFn: () => apiFetch<EnrichedAnnotation>(`/signals/${id}/enrichment`),
    enabled: !!id,
    retry: (failureCount, error) => {
      // Don't retry 404s — enrichment not yet available is an expected state
      if (error instanceof ApiError && error.status === 404) return false;
      return failureCount < 5;
    },
    refetchInterval: (query) => {
      // Poll every 30s if enrichment not found, up to implicit retry via refetch
      if (query.state.error instanceof ApiError && (query.state.error as ApiError).status === 404) {
        return 30_000;
      }
      return false;
    },
    staleTime: 120_000,
  });
}
