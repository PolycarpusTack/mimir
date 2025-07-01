"""
Proxy Dashboard for Mimir News Scraper
Provides web interface for monitoring proxy health and performance
"""

import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from typing import Dict, Any

from proxy_integration import get_global_proxy_manager, get_proxy_statistics, health_check_all_proxies
from config_loader import config_loader

logger = logging.getLogger(__name__)

def create_proxy_dashboard_app() -> Flask:
    """Create Flask app for proxy dashboard"""
    app = Flask(__name__)
    
    @app.route('/proxy')
    def proxy_dashboard():
        """Main proxy dashboard page"""
        try:
            config = config_loader.load_config()
            stats = get_proxy_statistics(config)
            
            return render_template('proxy_dashboard.html', 
                                 stats=stats,
                                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            logger.error(f"Error loading proxy dashboard: {e}")
            return f"Error loading dashboard: {e}", 500
    
    @app.route('/api/proxy/stats')
    def api_proxy_stats():
        """API endpoint for proxy statistics"""
        try:
            config = config_loader.load_config()
            stats = get_proxy_statistics(config)
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting proxy stats: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/proxy/health-check', methods=['POST'])
    def api_health_check():
        """API endpoint to trigger health check"""
        try:
            config = config_loader.load_config()
            health_check_all_proxies(config)
            return jsonify({"status": "success", "message": "Health check initiated"})
        except Exception as e:
            logger.error(f"Error triggering health check: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/proxy/reset/<proxy_url>', methods=['POST'])
    def api_reset_proxy(proxy_url: str):
        """API endpoint to reset a specific proxy"""
        try:
            config = config_loader.load_config()
            proxy_manager = get_global_proxy_manager(config)
            
            if proxy_manager is None:
                return jsonify({"error": "Proxy system is disabled"}), 400
            
            # URL decode the proxy URL
            import urllib.parse
            decoded_proxy_url = urllib.parse.unquote(proxy_url)
            
            proxy_manager.reset_proxy(decoded_proxy_url)
            return jsonify({
                "status": "success", 
                "message": f"Reset proxy {decoded_proxy_url}"
            })
        except Exception as e:
            logger.error(f"Error resetting proxy: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/proxy/save-state', methods=['POST'])
    def api_save_state():
        """API endpoint to save proxy state"""
        try:
            config = config_loader.load_config()
            proxy_manager = get_global_proxy_manager(config)
            
            if proxy_manager is None:
                return jsonify({"error": "Proxy system is disabled"}), 400
            
            proxy_manager.save_proxy_state()
            return jsonify({"status": "success", "message": "Proxy state saved"})
        except Exception as e:
            logger.error(f"Error saving proxy state: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app


# HTML template for proxy dashboard (embedded for simplicity)
PROXY_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir Proxy Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .proxy-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .proxy-table th,
        .proxy-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .proxy-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .status-active { color: #28a745; }
        .status-inactive { color: #ffc107; }
        .status-blacklisted { color: #dc3545; }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .btn-primary { background: #007bff; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .btn-success { background: #28a745; color: white; }
        .btn:hover { opacity: 0.8; }
        .controls {
            margin-bottom: 20px;
            text-align: center;
        }
        .disabled-message {
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.2em;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔄 Mimir Proxy Dashboard</h1>
            <p>Monitor proxy health, performance, and rotation status</p>
        </div>

        {% if stats.enabled %}
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ stats.total_proxies }}</div>
                <div class="stat-label">Total Proxies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.active_proxies }}</div>
                <div class="stat-label">Active Proxies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ stats.blacklisted_proxies }}</div>
                <div class="stat-label">Blacklisted</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(stats.overall_success_rate) }}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="triggerHealthCheck()">🔍 Health Check All</button>
            <button class="btn btn-success" onclick="saveProxyState()">💾 Save State</button>
            <button class="btn btn-warning" onclick="refreshDashboard()">🔄 Refresh</button>
        </div>

        <h3>Proxy Performance</h3>
        <table class="proxy-table">
            <thead>
                <tr>
                    <th>Proxy URL</th>
                    <th>Status</th>
                    <th>Provider</th>
                    <th>Success Rate</th>
                    <th>Avg Response Time</th>
                    <th>Total Requests</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for proxy in stats.top_proxies %}
                <tr>
                    <td>{{ proxy.url }}</td>
                    <td class="status-{{ proxy.status }}">{{ proxy.status.title() }}</td>
                    <td>{{ proxy.provider }}</td>
                    <td>{{ "%.1f"|format(proxy.success_rate) }}%</td>
                    <td>{{ "%.0f"|format(proxy.avg_response_time) }}ms</td>
                    <td>{{ proxy.total_requests }}</td>
                    <td>
                        <button class="btn btn-warning" onclick="resetProxy('{{ proxy.url|urlencode }}')">Reset</button>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% else %}
        <div class="disabled-message">
            <h3>Proxy System Disabled</h3>
            <p>Enable proxy settings in config.json to use the proxy dashboard.</p>
        </div>
        {% endif %}

        <div class="timestamp">
            Last updated: {{ timestamp }}
        </div>
    </div>

    <script>
        function triggerHealthCheck() {
            fetch('/api/proxy/health-check', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Health check initiated');
                    setTimeout(refreshDashboard, 5000); // Refresh after 5 seconds
                })
                .catch(error => alert('Error: ' + error));
        }

        function saveProxyState() {
            fetch('/api/proxy/save-state', { method: 'POST' })
                .then(response => response.json())
                .then(data => alert(data.message || 'State saved'))
                .catch(error => alert('Error: ' + error));
        }

        function resetProxy(proxyUrl) {
            if (confirm('Reset proxy metrics for ' + decodeURIComponent(proxyUrl) + '?')) {
                fetch('/api/proxy/reset/' + proxyUrl, { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message || 'Proxy reset');
                        refreshDashboard();
                    })
                    .catch(error => alert('Error: ' + error));
            }
        }

        function refreshDashboard() {
            location.reload();
        }

        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
    </script>
</body>
</html>
"""

def save_dashboard_template():
    """Save the HTML template to templates directory"""
    import os
    
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    template_path = os.path.join(templates_dir, "proxy_dashboard.html")
    with open(template_path, 'w') as f:
        f.write(PROXY_DASHBOARD_HTML)
    
    logger.info(f"Saved proxy dashboard template to {template_path}")


if __name__ == "__main__":
    # Save template and run dashboard
    save_dashboard_template()
    
    app = create_proxy_dashboard_app()
    app.run(host='0.0.0.0', port=5001, debug=True)