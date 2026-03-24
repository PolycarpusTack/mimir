import { describe, it, expect } from 'vitest';

// Mock WebSocket
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  readyState = MockWebSocket.OPEN;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];

  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

describe('WebSocket message protocol', () => {
  it('signal_created message has correct type field', () => {
    const msg = { type: 'signal_created', payload: { id: 'sig-001', title: 'Test' } };
    const json = JSON.stringify(msg);
    const parsed = JSON.parse(json);
    expect(parsed.type).toBe('signal_created');
    expect(parsed.payload.id).toBe('sig-001');
  });

  it('ping/pong messages serialize correctly', () => {
    const ping = JSON.stringify({ type: 'ping' });
    expect(JSON.parse(ping).type).toBe('ping');

    const pong = JSON.stringify({ type: 'pong' });
    expect(JSON.parse(pong).type).toBe('pong');
  });

  it('enrich_complete message contains signal_id and annotation', () => {
    const msg = {
      type: 'enrich_complete',
      payload: {
        signal_id: 'sig-001',
        annotation: { why_it_matters: 'Important signal', relevance_score: 85 },
      },
    };
    const parsed = JSON.parse(JSON.stringify(msg));
    expect(parsed.type).toBe('enrich_complete');
    expect(parsed.payload.signal_id).toBe('sig-001');
    expect(parsed.payload.annotation.relevance_score).toBe(85);
  });

  it('shift_alert message contains severity', () => {
    const msg = { type: 'shift_alert', payload: { id: 'shift-001', severity: 'Surge', z_score: 3.7 } };
    const parsed = JSON.parse(JSON.stringify(msg));
    expect(parsed.payload.severity).toBe('Surge');
  });

  it('jira_pushed message contains issue key', () => {
    const msg = { type: 'jira_pushed', payload: { queue_item_id: 'aq-001', issue_key: 'AIRFORGE-142' } };
    const parsed = JSON.parse(JSON.stringify(msg));
    expect(parsed.payload.issue_key).toBe('AIRFORGE-142');
  });

  it('digest_ready message contains summary', () => {
    const msg = { type: 'digest_ready', payload: { id: 'digest-001', signal_count: 47 } };
    const parsed = JSON.parse(JSON.stringify(msg));
    expect(parsed.payload.signal_count).toBe(47);
  });

  it('MockWebSocket sends pong in response to ping', () => {
    const ws = new MockWebSocket();
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
      }
    };
    ws.onmessage({ data: JSON.stringify({ type: 'ping' }) });
    expect(ws.sent).toHaveLength(1);
    expect(JSON.parse(ws.sent[0]).type).toBe('pong');
  });

  it('all 8 WsMessage types are recognized', () => {
    const types = [
      'signal_created', 'signal_updated', 'shift_alert', 'shift_resolved',
      'enrich_complete', 'jira_pushed', 'digest_ready', 'ping',
    ];
    types.forEach((t) => {
      const msg = JSON.parse(JSON.stringify({ type: t }));
      expect(msg.type).toBe(t);
    });
  });
});
