//! Integration test stubs: TC-WS-001 through TC-WS-006.

use airpulse_ws::registry::*;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

/// TC-WS-001: Register and deregister a connection.
#[tokio::test]
async fn tc_ws_001_register_deregister() {
    let registry = WsConnectionRegistry::new();
    let (tx, _rx) = mpsc::channel(16);
    let id = registry.register(tx).await;
    assert_eq!(registry.connection_count().await, 1);
    assert!(registry.deregister(id).await);
    assert_eq!(registry.connection_count().await, 0);
}

/// TC-WS-002: Broadcast reaches all registered connections.
#[tokio::test]
async fn tc_ws_002_broadcast_reaches_all() {
    let registry = WsConnectionRegistry::new();
    let (tx1, mut rx1) = mpsc::channel(16);
    let (tx2, mut rx2) = mpsc::channel(16);
    let (tx3, mut rx3) = mpsc::channel(16);

    registry.register(tx1).await;
    registry.register(tx2).await;
    registry.register(tx3).await;

    let msg = WsMessage::SignalIngested {
        signal_id: Uuid::new_v4(),
        title: "Test signal".to_string(),
    };
    let sent = registry.broadcast(&msg).await;
    assert_eq!(sent, 3);

    assert!(rx1.try_recv().is_ok());
    assert!(rx2.try_recv().is_ok());
    assert!(rx3.try_recv().is_ok());
}

/// TC-WS-003: Connection count is accurate.
#[tokio::test]
async fn tc_ws_003_connection_count() {
    let registry = WsConnectionRegistry::new();
    assert_eq!(registry.connection_count().await, 0);

    let mut ids = Vec::new();
    for _ in 0..5 {
        let (tx, _rx) = mpsc::channel(16);
        ids.push(registry.register(tx).await);
    }
    assert_eq!(registry.connection_count().await, 5);

    registry.deregister(ids[0]).await;
    registry.deregister(ids[2]).await;
    assert_eq!(registry.connection_count().await, 3);
}

/// TC-WS-004: Snapshot returns correct connection info.
#[tokio::test]
async fn tc_ws_004_snapshot() {
    let registry = WsConnectionRegistry::new();
    let (tx, _rx) = mpsc::channel(16);
    let id = registry.register(tx).await;

    let snap = registry.snapshot().await;
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].id, id);
    assert!(snap[0].connected_at <= chrono::Utc::now());
}

/// TC-WS-005: Deregistering a nonexistent connection returns false.
#[tokio::test]
async fn tc_ws_005_deregister_nonexistent() {
    let registry = WsConnectionRegistry::new();
    assert!(!registry.deregister(Uuid::new_v4()).await);
}

/// TC-WS-006: Concurrent registrations are safe.
#[tokio::test]
async fn tc_ws_006_concurrent_register() {
    let registry = Arc::new(WsConnectionRegistry::new());
    let mut handles = Vec::new();

    for _ in 0..20 {
        let reg = Arc::clone(&registry);
        handles.push(tokio::spawn(async move {
            let (tx, _rx) = mpsc::channel(16);
            reg.register(tx).await
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert_eq!(registry.connection_count().await, 20);
}
