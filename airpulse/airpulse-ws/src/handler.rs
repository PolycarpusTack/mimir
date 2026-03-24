//! WebSocket handler — connection lifecycle, ping/pong heartbeat, cleanup.

use crate::registry::{WsConnectionRegistry, WsMessage};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};


/// Default ping interval (30 seconds).
const PING_INTERVAL_SECS: u64 = 30;
/// Default pong timeout (10 seconds).
const PONG_TIMEOUT_SECS: u64 = 10;

/// Handle a WebSocket connection lifecycle.
///
/// - Registers the connection in the registry.
/// - Sends pings every 30s, expects pong within 10s.
/// - Cleans up on disconnect.
///
/// `socket_sender` is used to send outbound messages to the actual WebSocket.
/// `socket_receiver` receives inbound messages from the actual WebSocket.
pub async fn handle_socket(
    socket_sender: mpsc::Sender<String>,
    socket_receiver: mpsc::Receiver<String>,
    registry: Arc<WsConnectionRegistry>,
) {
    handle_socket_with_config(
        socket_sender,
        socket_receiver,
        registry,
        PING_INTERVAL_SECS,
        PONG_TIMEOUT_SECS,
    )
    .await
}

/// Handle socket with configurable intervals (for testing).
pub async fn handle_socket_with_config(
    socket_sender: mpsc::Sender<String>,
    mut socket_receiver: mpsc::Receiver<String>,
    registry: Arc<WsConnectionRegistry>,
    ping_interval_secs: u64,
    pong_timeout_secs: u64,
) {
    // Register connection
    let conn_id = registry.register(socket_sender.clone()).await;
    info!(connection_id = %conn_id, "WebSocket connected");

    let ping_interval = Duration::from_secs(ping_interval_secs);
    let pong_timeout = Duration::from_secs(pong_timeout_secs);

    let mut ping_timer = tokio::time::interval(ping_interval);
    ping_timer.tick().await; // consume the immediate first tick

    let mut awaiting_pong = false;
    let mut pong_deadline: Option<tokio::time::Instant> = None;

    loop {
        tokio::select! {
            // Ping timer fires
            _ = ping_timer.tick() => {
                let ping_msg = WsMessage::Ping {
                    timestamp: chrono::Utc::now().timestamp(),
                };
                let json = match serde_json::to_string(&ping_msg) {
                    Ok(j) => j,
                    Err(_) => continue,
                };
                if socket_sender.send(json).await.is_err() {
                    debug!(connection_id = %conn_id, "Sender closed, disconnecting");
                    break;
                }
                awaiting_pong = true;
                pong_deadline = Some(tokio::time::Instant::now() + pong_timeout);
            }

            // Check pong deadline
            _ = async {
                if let Some(deadline) = pong_deadline {
                    tokio::time::sleep_until(deadline).await;
                } else {
                    // No deadline, wait forever (will be pre-empted by other branches)
                    std::future::pending::<()>().await;
                }
            }, if awaiting_pong => {
                warn!(connection_id = %conn_id, "Pong timeout, disconnecting");
                break;
            }

            // Inbound message from client
            msg = socket_receiver.recv() => {
                match msg {
                    Some(text) => {
                        // Try to parse as WsMessage
                        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                            match ws_msg {
                                WsMessage::Pong { .. } => {
                                    debug!(connection_id = %conn_id, "Pong received");
                                    awaiting_pong = false;
                                    pong_deadline = None;
                                    registry.update_ping(conn_id).await;
                                }
                                _ => {
                                    debug!(connection_id = %conn_id, "Received client message");
                                }
                            }
                        }
                    }
                    None => {
                        // Client disconnected
                        debug!(connection_id = %conn_id, "Client disconnected (receiver closed)");
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    registry.deregister(conn_id).await;
    info!(connection_id = %conn_id, "WebSocket disconnected and cleaned up");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_socket_disconnect_cleanup() {
        let registry = Arc::new(WsConnectionRegistry::new());
        let (out_tx, _out_rx) = mpsc::channel::<String>(16);
        let (in_tx, in_rx) = mpsc::channel::<String>(16);

        // Spawn handler
        let reg_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            handle_socket_with_config(out_tx, in_rx, reg_clone, 300, 10).await;
        });

        // Give time for registration
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(registry.connection_count().await, 1);

        // Drop the inbound sender to simulate disconnect
        drop(in_tx);
        handle.await.unwrap();

        // Connection should be cleaned up
        assert_eq!(registry.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_ping_sent_on_interval() {
        let registry = Arc::new(WsConnectionRegistry::new());
        let (out_tx, mut out_rx) = mpsc::channel::<String>(16);
        let (in_tx, in_rx) = mpsc::channel::<String>(16);

        let reg_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            // Use very short ping interval for testing
            handle_socket_with_config(out_tx, in_rx, reg_clone, 1, 100).await;
        });

        // Wait for a ping to arrive
        let msg = tokio::time::timeout(Duration::from_secs(3), out_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(msg.contains("Ping"));

        // Disconnect
        drop(in_tx);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn test_pong_response_resets_timer() {
        let registry = Arc::new(WsConnectionRegistry::new());
        let (out_tx, mut out_rx) = mpsc::channel::<String>(16);
        let (in_tx, in_rx) = mpsc::channel::<String>(16);

        let reg_clone = Arc::clone(&registry);
        let handle = tokio::spawn(async move {
            handle_socket_with_config(out_tx, in_rx, reg_clone, 1, 5).await;
        });

        // Wait for ping
        let ping = tokio::time::timeout(Duration::from_secs(3), out_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(ping.contains("Ping"));

        // Send pong
        let pong = serde_json::to_string(&WsMessage::Pong {
            timestamp: chrono::Utc::now().timestamp(),
        })
        .unwrap();
        in_tx.send(pong).await.unwrap();

        // Connection should stay alive (wait for next ping)
        let next_ping = tokio::time::timeout(Duration::from_secs(3), out_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(next_ping.contains("Ping"));

        drop(in_tx);
        let _ = handle.await;
    }
}
