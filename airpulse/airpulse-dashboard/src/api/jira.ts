import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { ApprovalQueueItem, JiraPushRequest } from '@/types';

export function useApprovalQueue(status?: string) {
  return useQuery({
    queryKey: ['jira-queue', status],
    queryFn: () =>
      apiFetch<ApprovalQueueItem[]>('/jira/queue', { status }),
    refetchInterval: status === 'Pending' ? 15_000 : 30_000,
    staleTime: 10_000,
  });
}

export function useApprovalQueueItem(id: string | null) {
  return useQuery({
    queryKey: ['jira-queue-item', id],
    queryFn: () => apiFetch<ApprovalQueueItem>(`/jira/queue/${id}`),
    enabled: !!id,
  });
}

export function useJiraPush() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: JiraPushRequest) =>
      fetch('/api/v1/jira/push', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      }).then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({ message: r.statusText }));
          throw new Error(body.message || `HTTP ${r.status}`);
        }
        return r.json() as Promise<ApprovalQueueItem>;
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jira-queue'] });
    },
  });
}

export function useCancelJiraPush() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      fetch(`/api/v1/jira/queue/${id}`, { method: 'DELETE' }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['jira-queue'] });
    },
  });
}
