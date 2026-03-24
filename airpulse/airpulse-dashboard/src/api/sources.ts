import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { FeedSource } from '@/types';

export function useFeedSources() {
  return useQuery({
    queryKey: ['sources'],
    queryFn: () => apiFetch<FeedSource[]>('/sources'),
    refetchInterval: 120_000,
    staleTime: 60_000,
  });
}

export function useForcePoll() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sourceId: string) =>
      fetch(`/api/v1/sources/${sourceId}/poll`, { method: 'POST' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}
