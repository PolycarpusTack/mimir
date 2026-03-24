import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { DigestSummary, DigestDetail } from '@/types';

export function useDigestLatest() {
  return useQuery({
    queryKey: ['digest-latest'],
    queryFn: () => apiFetch<DigestSummary | null>('/digest/latest'),
    refetchInterval: 300_000,
    staleTime: 120_000,
  });
}

export function useDigestList(limit = 10) {
  return useQuery({
    queryKey: ['digest-list', limit],
    queryFn: () => apiFetch<DigestSummary[]>('/digest', { limit }),
    refetchInterval: 300_000,
    staleTime: 120_000,
  });
}

export function useDigest(id: string | null) {
  return useQuery({
    queryKey: ['digest', id],
    queryFn: () => apiFetch<DigestDetail>(`/digest/${id}`),
    enabled: !!id,
  });
}

export function useGenerateDigest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      fetch('/api/v1/digest/generate', { method: 'POST' }).then((r) => r.json()),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['digest-latest'] });
      qc.invalidateQueries({ queryKey: ['digest-list'] });
    },
  });
}
