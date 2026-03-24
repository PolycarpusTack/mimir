import { useQuery } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { ShiftAlert, Domain } from '@/types';
import type { ShiftSeverity } from '@/types';

interface ShiftsResponse {
  alerts: ShiftAlert[];
  total: number;
  as_of: string;
}

interface ShiftQueryParams {
  domain?: Domain;
  severity?: ShiftSeverity;
  after?: string;
}

export function useShiftAlerts(params: ShiftQueryParams = {}) {
  return useQuery({
    queryKey: ['shifts', params],
    queryFn: async () => {
      const res = await apiFetch<ShiftsResponse>('/shifts', {
        domain: params.domain,
        severity: params.severity,
        after: params.after,
      });
      return res.alerts;
    },
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useShiftAlert(id: string | null) {
  return useQuery({
    queryKey: ['shift', id],
    queryFn: () => apiFetch<ShiftAlert>(`/shifts/${id}`),
    enabled: !!id,
  });
}
