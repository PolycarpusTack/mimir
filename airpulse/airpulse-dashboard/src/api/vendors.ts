import { useQuery } from '@tanstack/react-query';
import { apiFetch } from './client';
import type { VendorStats } from '@/types';

export function useVendorStats(vendor: string, window = '30d') {
  return useQuery({
    queryKey: ['vendor-stats', vendor, window],
    queryFn: () => apiFetch<VendorStats>('/vendors/stats', { vendor, window }),
    staleTime: 300_000,
    enabled: !!vendor,
  });
}
