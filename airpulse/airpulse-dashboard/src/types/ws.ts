// src/types/ws.ts — Phase 5 WebSocket message types
import type { Signal, ShiftAlert, EnrichedAnnotation, DigestSummary } from './index';

export type WsMessage =
  | { type: 'signal_created'; payload: Signal }
  | { type: 'signal_updated'; payload: Partial<Signal> & { id: string } }
  | { type: 'shift_alert'; payload: ShiftAlert }
  | { type: 'shift_resolved'; payload: { id: string } }
  | { type: 'enrich_complete'; payload: { signal_id: string; annotation: EnrichedAnnotation } }
  | { type: 'jira_pushed'; payload: { queue_item_id: string; issue_key: string } }
  | { type: 'digest_ready'; payload: DigestSummary }
  | { type: 'ping' }
  | { type: 'pong' };
