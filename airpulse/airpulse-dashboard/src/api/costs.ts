import { useQuery } from '@tanstack/react-query';
import { apiFetch } from './client';

interface CostRow {
  model: string;
  total_input_tokens: number;
  total_output_tokens: number;
  call_count: number;
  created_at: string;
}

interface CostReportResponse {
  rows: CostRow[];
  total_input_tokens: number;
  total_output_tokens: number;
  total_calls: number;
}

export function useCostReport(params?: { after?: string; before?: string; domain?: string }) {
  return useQuery({
    queryKey: ['costs', params],
    queryFn: () => apiFetch<CostReportResponse>('/costs', params),
    staleTime: 300_000,
  });
}
