"""
PostgreSQL Vector Database Manager for Semantic Search
EPIC 3: Semantic Search & Similarity Engine

This module extends the PostgreSQL database manager with vector operations
for semantic search, similarity calculations, and embedding management.

Author: Claude Code
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg2
import psycopg2.extras

from db_manager_postgres import PostgreSQLManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticDatabaseManager(PostgreSQLManager):
    """
    Extended PostgreSQL database manager with vector operations for semantic search.

    Inherits from PostgreSQLManager and adds:
    - Vector embedding storage and retrieval
    - Similarity search functions
    - Hybrid search capabilities
    - Batch embedding operations
    """

    def __init__(self, connection_string: str = None):
        """Initialize the semantic database manager."""
        super().__init__(connection_string)
        self.default_model = "all-MiniLM-L6-v2"
        self.default_embedding_dim = 384

        # Verify pgvector extension
        self._verify_pgvector()

    def _verify_pgvector(self):
        """Verify that pgvector extension is available."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM pg_extension WHERE extname = 'vector'
                        )
                    """
                    )

                    if not cursor.fetchone()[0]:
                        logger.warning("pgvector extension not found. Please install it first.")
                        raise Exception("pgvector extension not available")

            logger.info("pgvector extension verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify pgvector extension: {e}")
            raise

    def store_embedding(
        self,
        article_id: str,
        embedding: np.ndarray,
        model_name: str = None,
        embedding_type: str = "title_content",
        quality_score: float = 1.0,
    ) -> bool:
        """
        Store an embedding for an article.

        Args:
            article_id: UUID of the article
            embedding: Numpy array of the embedding vector
            model_name: Name of the model used to generate embedding
            embedding_type: Type of embedding (title_content, title_only, content_only)
            quality_score: Quality score for the embedding

        Returns:
            True if successful, False otherwise
        """
        if model_name is None:
            model_name = self.default_model

        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Use ON CONFLICT to handle duplicates
                    cursor.execute(
                        """
                        INSERT INTO article_embeddings 
                        (article_id, embedding, model_name, embedding_type, quality_score)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (article_id, embedding_type, model_name)
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            quality_score = EXCLUDED.quality_score,
                            updated_at = CURRENT_TIMESTAMP
                    """,
                        (article_id, embedding_list, model_name, embedding_type, quality_score),
                    )

                    conn.commit()

            logger.debug(f"Stored embedding for article {article_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store embedding for article {article_id}: {e}")
            return False

    def store_embeddings_batch(self, embeddings_data: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Store multiple embeddings in batch for better performance.

        Args:
            embeddings_data: List of dicts with keys: article_id, embedding, model_name, embedding_type, quality_score
            batch_size: Number of embeddings to process in each batch

        Returns:
            Number of embeddings successfully stored
        """
        if not embeddings_data:
            return 0

        stored_count = 0

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Process in batches
                    for i in range(0, len(embeddings_data), batch_size):
                        batch = embeddings_data[i : i + batch_size]

                        # Prepare data for batch insert
                        values = []
                        for item in batch:
                            embedding_list = (
                                item["embedding"].tolist()
                                if isinstance(item["embedding"], np.ndarray)
                                else item["embedding"]
                            )
                            values.append(
                                (
                                    item["article_id"],
                                    embedding_list,
                                    item.get("model_name", self.default_model),
                                    item.get("embedding_type", "title_content"),
                                    item.get("quality_score", 1.0),
                                )
                            )

                        # Batch insert with ON CONFLICT handling
                        cursor.executemany(
                            """
                            INSERT INTO article_embeddings 
                            (article_id, embedding, model_name, embedding_type, quality_score)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (article_id, embedding_type, model_name)
                            DO UPDATE SET 
                                embedding = EXCLUDED.embedding,
                                quality_score = EXCLUDED.quality_score,
                                updated_at = CURRENT_TIMESTAMP
                        """,
                            values,
                        )

                        stored_count += len(batch)

                    conn.commit()

            logger.info(f"Successfully stored {stored_count} embeddings in batch")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store embeddings batch: {e}")
            return 0

    def get_embedding(
        self, article_id: str, model_name: str = None, embedding_type: str = "title_content"
    ) -> Optional[np.ndarray]:
        """
        Retrieve an embedding for an article.

        Args:
            article_id: UUID of the article
            model_name: Name of the model (default: self.default_model)
            embedding_type: Type of embedding

        Returns:
            Numpy array of the embedding or None if not found
        """
        if model_name is None:
            model_name = self.default_model

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT embedding FROM article_embeddings
                        WHERE article_id = %s 
                          AND model_name = %s 
                          AND embedding_type = %s
                    """,
                        (article_id, model_name, embedding_type),
                    )

                    result = cursor.fetchone()
                    if result:
                        return np.array(result[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get embedding for article {article_id}: {e}")
            return None

    def get_articles_without_embeddings(
        self, model_name: str = None, embedding_type: str = "title_content", limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get articles that don't have embeddings yet.

        Args:
            model_name: Name of the model
            embedding_type: Type of embedding
            limit: Maximum number of articles to return

        Returns:
            List of article dictionaries
        """
        if model_name is None:
            model_name = self.default_model

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT a.id, a.url, a.title, a.content, a.source, a.published_date
                        FROM articles a
                        LEFT JOIN article_embeddings ae ON (
                            ae.article_id = a.id 
                            AND ae.model_name = %s 
                            AND ae.embedding_type = %s
                        )
                        WHERE ae.id IS NULL
                          AND a.title IS NOT NULL
                          AND a.title != ''
                        ORDER BY a.scraped_at DESC
                        LIMIT %s
                    """,
                        (model_name, embedding_type, limit),
                    )

                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get articles without embeddings: {e}")
            return []

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float = 0.1,
        limit: int = 20,
        model_name: str = None,
        embedding_type: str = "title_content",
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search using PostgreSQL vector functions.

        Args:
            query_embedding: Query embedding vector
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results
            model_name: Model name for filtering
            embedding_type: Type of embedding

        Returns:
            List of similar articles with similarity scores
        """
        if model_name is None:
            model_name = self.default_model

        try:
            query_vector = query_embedding.tolist()

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM search_articles_semantic(%s, %s, %s, %s, %s)
                    """,
                        (query_vector, similarity_threshold, limit, model_name, embedding_type),
                    )

                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        limit: int = 20,
        model_name: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining full-text and semantic search.

        Args:
            query_text: Text query for full-text search
            query_embedding: Query embedding for semantic search
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            limit: Maximum number of results
            model_name: Model name for filtering

        Returns:
            List of articles with combined scores
        """
        if model_name is None:
            model_name = self.default_model

        try:
            query_vector = query_embedding.tolist()

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM search_articles_hybrid(%s, %s, %s, %s, %s, %s)
                    """,
                        (query_text, query_vector, semantic_weight, keyword_weight, limit, model_name),
                    )

                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return []

    def find_similar_articles(
        self, article_id: str, similarity_threshold: float = 0.3, limit: int = 10, model_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Find articles similar to a given article.

        Args:
            article_id: UUID of the reference article
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results
            model_name: Model name for filtering

        Returns:
            List of similar articles
        """
        if model_name is None:
            model_name = self.default_model

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute(
                        """
                        SELECT * FROM find_similar_articles(%s, %s, %s, %s)
                    """,
                        (article_id, similarity_threshold, limit, model_name),
                    )

                    results = cursor.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to find similar articles: {e}")
            return []

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    # Get total counts
                    cursor.execute(
                        """
                        SELECT 
                            COUNT(*) as total_embeddings,
                            COUNT(DISTINCT article_id) as articles_with_embeddings,
                            COUNT(DISTINCT model_name) as unique_models,
                            COUNT(DISTINCT embedding_type) as embedding_types
                        FROM article_embeddings
                    """
                    )

                    stats = dict(cursor.fetchone())

                    # Get counts by model
                    cursor.execute(
                        """
                        SELECT model_name, COUNT(*) as count
                        FROM article_embeddings
                        GROUP BY model_name
                        ORDER BY count DESC
                    """
                    )

                    stats["by_model"] = [dict(row) for row in cursor.fetchall()]

                    # Get counts by type
                    cursor.execute(
                        """
                        SELECT embedding_type, COUNT(*) as count
                        FROM article_embeddings
                        GROUP BY embedding_type
                        ORDER BY count DESC
                    """
                    )

                    stats["by_type"] = [dict(row) for row in cursor.fetchall()]

                    # Get total articles for coverage calculation
                    cursor.execute("SELECT COUNT(*) FROM articles")
                    total_articles = cursor.fetchone()["count"]
                    stats["total_articles"] = total_articles

                    if total_articles > 0:
                        stats["coverage_percentage"] = (stats["articles_with_embeddings"] / total_articles) * 100
                    else:
                        stats["coverage_percentage"] = 0

                    return stats

        except Exception as e:
            logger.error(f"Failed to get embedding statistics: {e}")
            return {}

    def cleanup_old_embeddings(self, model_name: str, keep_latest: int = 1) -> int:
        """
        Clean up old embeddings, keeping only the latest version for each article.

        Args:
            model_name: Model name to clean up
            keep_latest: Number of latest embeddings to keep per article

        Returns:
            Number of embeddings deleted
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM article_embeddings
                        WHERE id NOT IN (
                            SELECT id FROM (
                                SELECT id, 
                                       ROW_NUMBER() OVER (
                                           PARTITION BY article_id, embedding_type 
                                           ORDER BY updated_at DESC
                                       ) as rn
                                FROM article_embeddings
                                WHERE model_name = %s
                            ) ranked
                            WHERE rn <= %s
                        )
                        AND model_name = %s
                    """,
                        (model_name, keep_latest, model_name),
                    )

                    deleted_count = cursor.rowcount
                    conn.commit()

                    logger.info(f"Cleaned up {deleted_count} old embeddings for model {model_name}")
                    return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            return 0

    def update_vector_index(self, recreate: bool = False):
        """
        Update or recreate vector indexes for better performance.

        Args:
            recreate: Whether to drop and recreate indexes
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if recreate:
                        # Drop existing vector indexes
                        cursor.execute(
                            """
                            DROP INDEX IF EXISTS idx_article_embeddings_vector_ivf;
                            DROP INDEX IF EXISTS idx_article_embeddings_vector_hnsw;
                        """
                        )

                    # Check if we have enough data for HNSW
                    cursor.execute("SELECT COUNT(*) FROM article_embeddings")
                    embedding_count = cursor.fetchone()[0]

                    if embedding_count >= 1000:
                        # Create HNSW index for large datasets
                        logger.info("Creating HNSW index for large dataset")
                        cursor.execute(
                            """
                            CREATE INDEX IF NOT EXISTS idx_article_embeddings_vector_hnsw 
                            ON article_embeddings 
                            USING hnsw (embedding vector_cosine_ops) 
                            WITH (m = 16, ef_construction = 64);
                        """
                        )
                    else:
                        # Create IVFFlat index for smaller datasets
                        logger.info("Creating IVFFlat index for smaller dataset")
                        lists = max(10, embedding_count // 100)  # Adjust based on data size
                        cursor.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_article_embeddings_vector_ivf 
                            ON article_embeddings 
                            USING ivfflat (embedding vector_cosine_ops) 
                            WITH (lists = {lists});
                        """
                        )

                    conn.commit()
                    logger.info("Vector indexes updated successfully")

        except Exception as e:
            logger.error(f"Failed to update vector indexes: {e}")

    def get_duplicate_candidates(
        self, similarity_threshold: float = 0.85, limit: int = 100
    ) -> List[Tuple[str, str, float]]:
        """
        Find potential duplicate articles using vector similarity.

        Args:
            similarity_threshold: Minimum similarity to consider as duplicate candidate
            limit: Maximum number of pairs to return

        Returns:
            List of (article_id1, article_id2, similarity_score) tuples
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Find pairs of articles with high similarity
                    cursor.execute(
                        """
                        SELECT 
                            ae1.article_id as article_id1,
                            ae2.article_id as article_id2,
                            1 - (ae1.embedding <=> ae2.embedding) as similarity_score
                        FROM article_embeddings ae1
                        JOIN article_embeddings ae2 ON (
                            ae1.article_id < ae2.article_id
                            AND ae1.model_name = ae2.model_name
                            AND ae1.embedding_type = ae2.embedding_type
                        )
                        WHERE (1 - (ae1.embedding <=> ae2.embedding)) >= %s
                        ORDER BY similarity_score DESC
                        LIMIT %s
                    """,
                        (similarity_threshold, limit),
                    )

                    results = cursor.fetchall()
                    return [(row[0], row[1], row[2]) for row in results]

        except Exception as e:
            logger.error(f"Failed to find duplicate candidates: {e}")
            return []


# Utility functions
def create_semantic_db_manager(connection_string: str = None) -> SemanticDatabaseManager:
    """Create and return a semantic database manager instance."""
    return SemanticDatabaseManager(connection_string)


def verify_semantic_setup(db_manager: SemanticDatabaseManager) -> bool:
    """Verify that semantic search setup is working correctly."""
    try:
        # Check if pgvector extension is available
        db_manager._verify_pgvector()

        # Check if tables exist
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = 'article_embeddings'
                    )
                """
                )

                if not cursor.fetchone()[0]:
                    logger.error("article_embeddings table not found")
                    return False

        # Check if functions exist
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.routines 
                        WHERE routine_name = 'search_articles_semantic'
                    )
                """
                )

                if not cursor.fetchone()[0]:
                    logger.error("search_articles_semantic function not found")
                    return False

        logger.info("Semantic search setup verified successfully")
        return True

    except Exception as e:
        logger.error(f"Semantic setup verification failed: {e}")
        return False


if __name__ == "__main__":
    # CLI interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Database Manager CLI")
    parser.add_argument("command", choices=["verify", "stats", "cleanup", "reindex"])
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")

    args = parser.parse_args()

    # Initialize database manager
    db_manager = create_semantic_db_manager()

    if args.command == "verify":
        success = verify_semantic_setup(db_manager)
        print(f"Semantic setup verification: {'SUCCESS' if success else 'FAILED'}")

    elif args.command == "stats":
        stats = db_manager.get_embedding_statistics()
        print("Embedding Statistics:")
        print(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"  Articles with embeddings: {stats.get('articles_with_embeddings', 0)}")
        print(f"  Coverage: {stats.get('coverage_percentage', 0):.1f}%")
        print(f"  Unique models: {stats.get('unique_models', 0)}")

    elif args.command == "cleanup":
        deleted = db_manager.cleanup_old_embeddings(args.model)
        print(f"Cleaned up {deleted} old embeddings")

    elif args.command == "reindex":
        db_manager.update_vector_index(recreate=True)
        print("Vector indexes updated")
