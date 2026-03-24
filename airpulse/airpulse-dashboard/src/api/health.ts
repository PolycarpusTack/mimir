import { useQuery } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { HealthResponse, StatsResponse, EnrichmentQueueStatus } from '@/types';

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: () => apiFetch<HealthResponse>('/health'),
    refetchInterval: 120_000,
    staleTime: 60_000,
  });
}

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => apiFetch<StatsResponse>('/stats'),
    refetchInterval: 300_000,
    staleTime: 120_000,
  });
}

export function useEnrichmentQueue() {
  return useQuery({
    queryKey: ['enrichment-queue'],
    queryFn: () => apiFetch<EnrichmentQueueStatus>('/enrichment/queue'),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}
