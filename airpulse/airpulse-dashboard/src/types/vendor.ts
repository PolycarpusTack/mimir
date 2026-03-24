// src/types/vendor.ts — Phase 5 vendor stats API type
export interface VendorStats {
  vendor: string;
  mentions_7d: number;
  mentions_30d: number;
  delta: number;
  daily_trend: number[];
}
