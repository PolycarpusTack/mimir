import json
import logging
from datetime import datetime, timedelta

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

import db_manager
from advanced_deduplication import AdvancedDeduplicator
from db_manager_semantic import SemanticDatabaseManager
from embedding_pipeline import EmbeddingPipeline
from semantic_search import SemanticSearchEngine
from db_security import get_secure_executor
from config_loader import get_security_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Security configuration
security_config = get_security_config()
app.config['SECRET_KEY'] = security_config.get('session_secret', 'development-secret-key')
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Configure CORS with specific origins
cors_origins = security_config.get('cors_origins', ['http://localhost:3000'])
CORS(app, origins=cors_origins, supports_credentials=True)

# Security headers
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self'"
    )
    return response

# Initialize secure database executor
secure_db = get_secure_executor(db_manager.get_db_connection)

# Initialize semantic search components
try:
    semantic_engine = SemanticSearchEngine()
    semantic_db = SemanticDatabaseManager()
    embedding_pipeline = EmbeddingPipeline()
    deduplicator = AdvancedDeduplicator(semantic_engine, semantic_db)
    logger.info("Semantic search components initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize semantic search: {e}")
    semantic_engine = None
    semantic_db = None
    embedding_pipeline = None
    deduplicator = None


@app.route("/")
def index():
    """Hoofdpagina met dashboard."""
    return render_template("index.html")


@app.route("/api/csrf-token")
def get_csrf_token():
    """Get CSRF token for API requests."""
    from flask_wtf.csrf import generate_csrf
    return jsonify({"csrf_token": generate_csrf()})


@app.route("/api/statistics")
def get_statistics():
    """API endpoint voor statistieken."""
    stats = db_manager.get_statistics()
    return jsonify(stats)


@app.route("/api/articles")
def get_articles():
    """API endpoint voor artikelen met paginatie."""
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 20, type=int), 100)  # Limit max per_page
    search = request.args.get("search", "")
    category = request.args.get("category", "")

    # Build secure where conditions
    where_conditions = {}
    
    if search:
        # Use secure LIKE search
        where_conditions["title"] = {"operator": "ILIKE", "value": f"%{search}%"}
        
    if category:
        where_conditions["category"] = category

    try:
        # Get total count using secure query
        count_result = secure_db.execute_raw_query(
            "SELECT COUNT(*) as total FROM articles WHERE " + 
            ("title ILIKE %s" if search else "1=1") +
            (" AND category = %s" if category else ""),
            [f"%{search}%", category] if search and category else 
            [f"%{search}%"] if search else 
            [category] if category else []
        )
        total = count_result[0]["total"] if count_result else 0

        # Get paginated results using secure query
        articles = secure_db.execute_select(
            table="articles",
            where_conditions=where_conditions,
            order_by="scraped_at",
            limit=per_page,
            offset=(page - 1) * per_page
        )

        return jsonify({
            "articles": articles,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        })
        
    except Exception as e:
        logger.error(f"Error fetching articles: {e}")
        return jsonify({"error": "Failed to fetch articles"}), 500


@app.route("/api/keywords")
def get_keyword_alerts():
    """API endpoint voor keyword alerts."""
    limit = request.args.get("limit", 50, type=int)

    with db_manager.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        SELECT ka.*, a.title as article_title, a.url as article_url
        FROM keyword_alerts ka
        JOIN articles a ON ka.article_id = a.id
        ORDER BY ka.created_at DESC
        LIMIT ?
        """,
            (limit,),
        )
        alerts = [dict(row) for row in cursor.fetchall()]

    return jsonify(alerts)


@app.route("/api/scrape-runs")
def get_scrape_runs():
    """API endpoint voor scrape run geschiedenis."""
    with db_manager.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
        SELECT * FROM scrape_runs
        ORDER BY started_at DESC
        LIMIT 50
        """
        )
        runs = [dict(row) for row in cursor.fetchall()]

    return jsonify(runs)


@app.route("/api/mark-read", methods=["POST"])
@csrf.exempt  # Will implement proper CSRF token handling
def mark_articles_read():
    """Markeer alle artikelen als gelezen."""
    try:
        count = db_manager.mark_articles_as_read()
        return jsonify({"success": True, "marked": count})
    except Exception as e:
        logger.error(f"Error marking articles as read: {e}")
        return jsonify({"error": "Failed to mark articles as read"}), 500


# EPIC 3: Semantic Search API Endpoints


@app.route("/api/search/semantic", methods=["POST"])
def semantic_search():
    """API endpoint for semantic search."""
    if not semantic_engine:
        return jsonify({"error": "Semantic search not available"}), 503

    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    threshold = data.get("threshold", 0.1)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        results = semantic_engine.semantic_search(query, top_k, threshold)
        return jsonify({"success": True, "query": query, "results": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search/hybrid", methods=["POST"])
def hybrid_search():
    """API endpoint for hybrid (semantic + keyword) search."""
    if not semantic_engine:
        return jsonify({"error": "Semantic search not available"}), 503

    data = request.get_json()
    query = data.get("query", "")
    top_k = data.get("top_k", 10)
    semantic_weight = data.get("semantic_weight", 0.7)
    keyword_weight = data.get("keyword_weight", 0.3)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        results = semantic_engine.hybrid_search(query, top_k, semantic_weight, keyword_weight)
        return jsonify(
            {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "weights": {"semantic": semantic_weight, "keyword": keyword_weight},
            }
        )
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/articles/<article_id>/similar")
def find_similar_articles(article_id):
    """Find articles similar to a given article."""
    if not semantic_engine:
        return jsonify({"error": "Semantic search not available"}), 503

    top_k = request.args.get("top_k", 5, type=int)
    threshold = request.args.get("threshold", 0.3, type=float)

    try:
        similar_articles = semantic_engine.find_similar_articles(article_id, top_k, threshold)
        return jsonify(
            {
                "success": True,
                "article_id": article_id,
                "similar_articles": similar_articles,
                "count": len(similar_articles),
            }
        )
    except Exception as e:
        logger.error(f"Similar articles error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings/status")
def embedding_status():
    """Get embedding coverage and statistics."""
    if not semantic_db:
        return jsonify({"error": "Semantic database not available"}), 503

    try:
        stats = semantic_db.get_embedding_statistics()
        return jsonify({"success": True, "statistics": stats})
    except Exception as e:
        logger.error(f"Embedding status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/embeddings/generate", methods=["POST"])
def generate_embeddings():
    """Generate embeddings for articles without them."""
    if not embedding_pipeline:
        return jsonify({"error": "Embedding pipeline not available"}), 503

    data = request.get_json() or {}
    limit = data.get("limit", 100)
    force = data.get("force", False)

    try:
        # Queue articles for embedding
        queued = embedding_pipeline.queue_articles_for_embedding(limit=limit, force_reprocess=force)

        if queued == 0:
            return jsonify({"success": True, "message": "No articles to process", "queued": 0})

        # Process embeddings
        stats = embedding_pipeline.process_all_pending(show_progress=False)

        return jsonify(
            {
                "success": True,
                "message": f'Generated embeddings for {stats["successful"]} articles',
                "statistics": stats,
            }
        )
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/duplicates/detect", methods=["POST"])
def detect_duplicates():
    """Detect duplicate articles."""
    if not deduplicator:
        return jsonify({"error": "Deduplication not available"}), 503

    data = request.get_json() or {}
    thresholds = data.get(
        "thresholds", {"exact": 1.0, "near": 0.8, "semantic": 0.85, "cross_language": 0.7, "title": 0.9}
    )

    try:
        # Run comprehensive deduplication
        results = deduplicator.comprehensive_deduplication(similarity_thresholds=thresholds)

        # Generate report
        report = deduplicator.get_deduplication_report(results)

        return jsonify(
            {
                "success": True,
                "report": report,
                "duplicates_by_method": {method: len(duplicates) for method, duplicates in results.items()},
            }
        )
    except Exception as e:
        logger.error(f"Duplicate detection error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search/build-index", methods=["POST"])
def build_search_index():
    """Build search index for semantic search."""
    if not semantic_engine:
        return jsonify({"error": "Semantic search not available"}), 503

    try:
        count = semantic_engine.build_article_index(force_rebuild=True)
        return jsonify(
            {"success": True, "message": f"Built search index with {count} articles", "indexed_articles": count}
        )
    except Exception as e:
        logger.error(f"Index building error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/semantic/health")
def semantic_health():
    """Health check for semantic search components."""
    health = {
        "semantic_engine": semantic_engine is not None,
        "semantic_db": semantic_db is not None,
        "embedding_pipeline": embedding_pipeline is not None,
        "deduplicator": deduplicator is not None,
    }

    # Test database connection
    if semantic_db:
        try:
            stats = semantic_db.get_embedding_statistics()
            health["database_connection"] = True
            health["embeddings_available"] = stats.get("total_embeddings", 0) > 0
        except Exception as e:
            health["database_connection"] = False
            health["database_error"] = str(e)

    # Test model loading
    if semantic_engine:
        try:
            # Try a simple embedding
            test_embedding = semantic_engine.generate_embeddings(["test"], show_progress=False)
            health["model_loaded"] = test_embedding.shape[0] > 0
        except Exception as e:
            health["model_loaded"] = False
            health["model_error"] = str(e)

    overall_health = all([health["semantic_engine"], health["semantic_db"], health.get("database_connection", False)])

    return jsonify({"healthy": overall_health, "components": health, "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    # Maak templates directory
    import os

    if not os.path.exists("templates"):
        os.makedirs("templates")

    app.run(debug=os.getenv("DEBUG", "false").lower() == "true", port=5000)
