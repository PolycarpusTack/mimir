//! WebSocket connection registry — tracks active connections and supports broadcast.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, warn};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum WsError {
    #[error("Connection not found: {0}")]
    ConnectionNotFound(Uuid),

    #[error("Send failed for connection {id}: {reason}")]
    SendFailed { id: Uuid, reason: String },

    #[error("Serialization error: {0}")]
    Serialization(String),
}

// ---------------------------------------------------------------------------
// WsMessage — messages sent to WebSocket clients
// ---------------------------------------------------------------------------

/// A message that can be broadcast or sent to individual connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsMessage {
    /// A new signal was ingested.
    SignalIngested { signal_id: Uuid, title: String },
    /// A shift alert was fired.
    ShiftAlert { domain: String, severity: String, z_score: f64 },
    /// Digest generation completed.
    DigestReady { digest_id: Uuid, week: String },
    /// JIRA ticket was pushed.
    JiraPushed { issue_key: String, signal_id: Uuid },
    /// Server heartbeat ping.
    Ping { timestamp: i64 },
    /// Client heartbeat pong.
    Pong { timestamp: i64 },
    /// Generic JSON payload.
    Custom(serde_json::Value),
}

// ---------------------------------------------------------------------------
// WsConnection — a single tracked connection
// ---------------------------------------------------------------------------

/// Internal state for a single WebSocket connection.
struct WsConnection {
    id: Uuid,
    connected_at: DateTime<Utc>,
    last_ping_at: DateTime<Utc>,
    sender: mpsc::Sender<String>,
}

/// Public snapshot info about a connection (no sender exposed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConnectionInfo {
    pub id: Uuid,
    pub connected_at: DateTime<Utc>,
    pub last_ping_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// WsConnectionRegistry
// ---------------------------------------------------------------------------

/// Thread-safe registry of active WebSocket connections.
pub struct WsConnectionRegistry {
    connections: Arc<RwLock<HashMap<Uuid, WsConnection>>>,
}

impl WsConnectionRegistry {
    pub fn new() -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new connection with its outbound sender. Returns the assigned Uuid.
    pub async fn register(&self, sender: mpsc::Sender<String>) -> Uuid {
        let id = Uuid::new_v4();
        let now = Utc::now();
        let conn = WsConnection {
            id,
            connected_at: now,
            last_ping_at: now,
            sender,
        };
        let mut conns = self.connections.write().await;
        conns.insert(id, conn);
        debug!(connection_id = %id, "WebSocket connection registered");
        id
    }

    /// Deregister a connection. Returns true if the connection existed.
    pub async fn deregister(&self, id: Uuid) -> bool {
        let mut conns = self.connections.write().await;
        let removed = conns.remove(&id).is_some();
        if removed {
            debug!(connection_id = %id, "WebSocket connection deregistered");
        } else {
            warn!(connection_id = %id, "Attempted to deregister nonexistent connection");
        }
        removed
    }

    /// Broadcast a message to all connected clients. Spawns sends concurrently.
    /// Returns the number of successful sends.
    pub async fn broadcast(&self, msg: &WsMessage) -> u32 {
        let json = match serde_json::to_string(msg) {
            Ok(j) => j,
            Err(e) => {
                error!(error = %e, "Failed to serialize WsMessage for broadcast");
                return 0;
            }
        };

        let conns = self.connections.read().await;
        let mut success_count = 0u32;
        let mut failed_ids = Vec::new();

        for (id, conn) in conns.iter() {
            match conn.sender.try_send(json.clone()) {
                Ok(()) => success_count += 1,
                Err(e) => {
                    warn!(connection_id = %id, error = %e, "Broadcast send failed");
                    failed_ids.push(*id);
                }
            }
        }

        drop(conns);

        // Clean up failed connections
        if !failed_ids.is_empty() {
            let mut conns = self.connections.write().await;
            for id in failed_ids {
                conns.remove(&id);
                debug!(connection_id = %id, "Removed failed connection after broadcast");
            }
        }

        success_count
    }

    /// Send a message to a specific connection.
    pub async fn send_to(&self, id: Uuid, msg: &WsMessage) -> Result<(), WsError> {
        let json = serde_json::to_string(msg).map_err(|e| WsError::Serialization(e.to_string()))?;

        let conns = self.connections.read().await;
        let conn = conns.get(&id).ok_or(WsError::ConnectionNotFound(id))?;

        conn.sender
            .try_send(json)
            .map_err(|e| WsError::SendFailed {
                id,
                reason: e.to_string(),
            })
    }

    /// Return the current number of active connections.
    pub async fn connection_count(&self) -> usize {
        let conns = self.connections.read().await;
        conns.len()
    }

    /// Return a snapshot of all active connections (without senders).
    pub async fn snapshot(&self) -> Vec<WsConnectionInfo> {
        let conns = self.connections.read().await;
        conns
            .values()
            .map(|c| WsConnectionInfo {
                id: c.id,
                connected_at: c.connected_at,
                last_ping_at: c.last_ping_at,
            })
            .collect()
    }

    /// Update the last_ping_at timestamp for a connection.
    pub async fn update_ping(&self, id: Uuid) -> bool {
        let mut conns = self.connections.write().await;
        if let Some(conn) = conns.get_mut(&id) {
            conn.last_ping_at = Utc::now();
            true
        } else {
            false
        }
    }
}

impl Default for WsConnectionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_deregister() {
        let registry = WsConnectionRegistry::new();
        let (tx, _rx) = mpsc::channel(16);
        let id = registry.register(tx).await;
        assert_eq!(registry.connection_count().await, 1);

        let removed = registry.deregister(id).await;
        assert!(removed);
        assert_eq!(registry.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_broadcast_reaches_all() {
        let registry = WsConnectionRegistry::new();
        let (tx1, mut rx1) = mpsc::channel(16);
        let (tx2, mut rx2) = mpsc::channel(16);

        registry.register(tx1).await;
        registry.register(tx2).await;

        let msg = WsMessage::Ping {
            timestamp: Utc::now().timestamp(),
        };
        let sent = registry.broadcast(&msg).await;
        assert_eq!(sent, 2);

        // Both receivers should have the message
        let m1 = rx1.try_recv().unwrap();
        let m2 = rx2.try_recv().unwrap();
        assert!(m1.contains("Ping"));
        assert!(m2.contains("Ping"));
    }

    #[tokio::test]
    async fn test_connection_count() {
        let registry = WsConnectionRegistry::new();
        assert_eq!(registry.connection_count().await, 0);

        let (tx1, _rx1) = mpsc::channel(16);
        let (tx2, _rx2) = mpsc::channel(16);
        registry.register(tx1).await;
        assert_eq!(registry.connection_count().await, 1);
        registry.register(tx2).await;
        assert_eq!(registry.connection_count().await, 2);
    }

    #[tokio::test]
    async fn test_snapshot() {
        let registry = WsConnectionRegistry::new();
        let (tx, _rx) = mpsc::channel(16);
        let id = registry.register(tx).await;

        let snap = registry.snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].id, id);
    }

    #[tokio::test]
    async fn test_deregister_nonexistent() {
        let registry = WsConnectionRegistry::new();
        let removed = registry.deregister(Uuid::new_v4()).await;
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_concurrent_register() {
        let registry = Arc::new(WsConnectionRegistry::new());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let reg = Arc::clone(&registry);
            handles.push(tokio::spawn(async move {
                let (tx, _rx) = mpsc::channel(16);
                reg.register(tx).await
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(registry.connection_count().await, 10);
    }

    #[tokio::test]
    async fn test_send_to_specific() {
        let registry = WsConnectionRegistry::new();
        let (tx1, mut rx1) = mpsc::channel(16);
        let (tx2, mut rx2) = mpsc::channel(16);

        let id1 = registry.register(tx1).await;
        let _id2 = registry.register(tx2).await;

        let msg = WsMessage::DigestReady {
            digest_id: Uuid::new_v4(),
            week: "2026-W13".to_string(),
        };
        registry.send_to(id1, &msg).await.unwrap();

        // Only rx1 should have the message
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_send_to_nonexistent() {
        let registry = WsConnectionRegistry::new();
        let msg = WsMessage::Ping { timestamp: 0 };
        let result = registry.send_to(Uuid::new_v4(), &msg).await;
        assert!(matches!(result, Err(WsError::ConnectionNotFound(_))));
    }

    #[tokio::test]
    async fn test_update_ping() {
        let registry = WsConnectionRegistry::new();
        let (tx, _rx) = mpsc::channel(16);
        let id = registry.register(tx).await;
        assert!(registry.update_ping(id).await);
        assert!(!registry.update_ping(Uuid::new_v4()).await);
    }
}
