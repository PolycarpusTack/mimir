import { useEffect, useRef, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { WsMessage, SignalPage, ShiftAlert } from '@/types';

export function useWebSocket() {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(1000);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        switch (msg.type) {
          case 'signal_created':
            queryClient.setQueryData(['signals'], (old: SignalPage | undefined) => {
              if (!old) return old;
              return { ...old, items: [msg.payload, ...old.items], total: old.total + 1 };
            });
            break;
          case 'signal_updated':
            queryClient.invalidateQueries({ queryKey: ['signal', msg.payload.id] });
            queryClient.invalidateQueries({ queryKey: ['signals'] });
            break;
          case 'shift_alert':
            queryClient.setQueryData(['shifts'], (old: ShiftAlert[] | undefined) =>
              [msg.payload, ...(old ?? [])],
            );
            break;
          case 'shift_resolved':
            queryClient.invalidateQueries({ queryKey: ['shifts'] });
            break;
          case 'enrich_complete':
            queryClient.setQueryData(
              ['enrichment', msg.payload.signal_id],
              msg.payload.annotation,
            );
            queryClient.invalidateQueries({ queryKey: ['signals'] });
            break;
          case 'jira_pushed':
            queryClient.invalidateQueries({ queryKey: ['jira-queue'] });
            break;
          case 'digest_ready':
            queryClient.invalidateQueries({ queryKey: ['digest-latest'] });
            queryClient.invalidateQueries({ queryKey: ['digest-list'] });
            break;
          case 'ping':
            ws.send(JSON.stringify({ type: 'pong' }));
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      const delay = reconnectDelay.current;
      reconnectDelay.current = Math.min(delay * 2, 30_000);
      setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [queryClient]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { ws: wsRef };
}
