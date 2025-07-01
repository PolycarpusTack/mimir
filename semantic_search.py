"""
Semantic Search & Similarity Engine for Mimir
EPIC 3 Implementation

This module provides:
- Sentence transformer-based embeddings
- Vector similarity search with FAISS
- Hybrid search (keyword + semantic)
- Advanced deduplication using MinHash/LSH
- Cross-language semantic search

Author: Claude Code
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import redis
import torch
from datasketch import LSHForest, MinHash
from fuzzywuzzy import fuzz, process
from langdetect import DetectorFactory, detect
from sentence_transformers import SentenceTransformer, util

from db_adapter import DatabaseAdapter

DetectorFactory.seed = 0  # Consistent results

# Preprocessing
import re

# Progress tracking
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Main semantic search engine with embedding generation,
    similarity search, and deduplication capabilities.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        cache_embeddings: bool = True,
        use_redis: bool = True,
    ):
        """
        Initialize the semantic search engine.

        Args:
            model_name: Sentence transformer model name
            device: Device to run model on (auto/cpu/cuda)
            cache_embeddings: Whether to cache computed embeddings
            use_redis: Whether to use Redis for caching
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_embeddings = cache_embeddings
        self.use_redis = use_redis

        # Initialize components
        self.model = None
        self.faiss_index = None
        self.embedding_cache = {}
        self.redis_client = None
        self.db = DatabaseAdapter()

        # Deduplication components
        self.lsh_forest = None
        self.minhash_cache = {}

        # Load or initialize
        self._initialize()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing Semantic Search Engine with model: {self.model_name}")

        # Load sentence transformer model
        self._load_model()

        # Initialize Redis if enabled
        if self.use_redis:
            self._initialize_redis()

        # Initialize FAISS index
        self._initialize_faiss()

        # Initialize LSH for deduplication
        self._initialize_lsh()

        logger.info("Semantic Search Engine initialized successfully")

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _initialize_redis(self):
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
            self.redis_client.ping()  # Test connection
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Falling back to in-memory cache")
            self.use_redis = False

    def _initialize_faiss(self):
        """Initialize FAISS index for vector similarity search."""
        try:
            # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"FAISS index initialized with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise

    def _initialize_lsh(self):
        """Initialize LSH Forest for deduplication."""
        try:
            self.lsh_forest = LSHForest(num_perm=128)
            logger.info("LSH Forest initialized for deduplication")
        except Exception as e:
            logger.error(f"Failed to initialize LSH Forest: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Computing {len(uncached_texts)} new embeddings")

            if show_progress:
                iterator = tqdm(range(0, len(uncached_texts), batch_size), desc="Generating embeddings")
            else:
                iterator = range(0, len(uncached_texts), batch_size)

            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, len(uncached_texts))
                batch_texts = uncached_texts[start_idx:end_idx]

                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts, convert_to_numpy=True, normalize_embeddings=True  # For cosine similarity
                )

                # Store in cache and results
                for j, embedding in enumerate(batch_embeddings):
                    global_idx = uncached_indices[start_idx + j]
                    embeddings[global_idx] = embedding

                    # Cache the embedding
                    if self.cache_embeddings:
                        self._cache_embedding(batch_texts[j], embedding)

        # Convert to numpy array
        result = np.array([emb for emb in embeddings if emb is not None])
        logger.info(f"Generated embeddings shape: {result.shape}")

        return result

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        if not self.cache_embeddings:
            return None

        # Create cache key
        cache_key = f"embedding:{hash(text)}"

        # Try Redis first
        if self.use_redis and self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")

        # Try in-memory cache
        return self.embedding_cache.get(cache_key)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for text."""
        if not self.cache_embeddings:
            return

        cache_key = f"embedding:{hash(text)}"

        # Cache in Redis
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(cache_key, 86400, pickle.dumps(embedding))  # 24 hours TTL
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")

        # Cache in memory (with size limit)
        if len(self.embedding_cache) < 10000:  # Limit memory usage
            self.embedding_cache[cache_key] = embedding

    def build_article_index(self, force_rebuild: bool = False) -> int:
        """
        Build FAISS index from all articles in database.

        Args:
            force_rebuild: Whether to rebuild index even if it exists

        Returns:
            Number of articles indexed
        """
        logger.info("Building article index for semantic search")

        # Get all articles
        try:
            articles = self.db.get_all_articles_with_content()
            logger.info(f"Retrieved {len(articles)} articles from database")
        except Exception as e:
            logger.error(f"Failed to retrieve articles: {e}")
            return 0

        if not articles:
            logger.warning("No articles found in database")
            return 0

        # Prepare texts for embedding
        texts = []
        article_ids = []

        for article in articles:
            # Combine title and content for better semantic representation
            text = f"{article.get('title', '')} {article.get('content', '')}"
            text = self._preprocess_text(text)

            if text.strip():  # Only add non-empty texts
                texts.append(text)
                article_ids.append(article["id"])

        logger.info(f"Prepared {len(texts)} texts for embedding")

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        if embeddings.shape[0] == 0:
            logger.warning("No embeddings generated")
            return 0

        # Build FAISS index
        self.faiss_index.reset()  # Clear existing index
        self.faiss_index.add(embeddings)

        # Store article IDs for retrieval
        self.article_ids = article_ids

        # Cache index to disk
        self._save_index()

        logger.info(f"Successfully built index with {len(article_ids)} articles")
        return len(article_ids)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality."""
        if not text:
            return ""

        # Basic cleaning
        text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        text = re.sub(r"http[s]?://\S+", "", text)  # Remove URLs
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = text.strip()

        # Limit length to avoid memory issues
        if len(text) > 2000:
            text = text[:2000] + "..."

        return text

    def semantic_search(self, query: str, top_k: int = 10, threshold: float = 0.1) -> List[Dict]:
        """
        Perform semantic search for similar articles.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar articles with similarity scores
        """
        if not hasattr(self, "article_ids") or not self.article_ids:
            # Try to load index
            if not self._load_index():
                logger.error("No search index available. Please build index first.")
                return []

        # Preprocess and embed query
        processed_query = self._preprocess_text(query)
        query_embedding = self.generate_embeddings([processed_query], show_progress=False)

        if query_embedding.shape[0] == 0:
            logger.error("Failed to generate query embedding")
            return []

        # Search FAISS index
        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        # Filter by threshold and prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if similarity < threshold:
                continue

            if idx < len(self.article_ids):
                article_id = self.article_ids[idx]
                # Get full article details
                article = self.db.get_article_by_id(article_id)
                if article:
                    results.append({"article": article, "similarity_score": float(similarity), "rank": i + 1})

        logger.info(f"Semantic search returned {len(results)} results for query: '{query[:50]}...'")
        return results

    def hybrid_search(
        self, query: str, top_k: int = 10, semantic_weight: float = 0.7, keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching

        Returns:
            List of articles with combined similarity scores
        """
        # Get semantic search results
        semantic_results = self.semantic_search(query, top_k * 2)  # Get more for reranking

        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k * 2)

        # Combine and rerank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, semantic_weight, keyword_weight
        )

        # Return top_k results
        return combined_results[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform traditional keyword-based search."""
        try:
            # Use database's full-text search if available
            results = self.db.search_articles_fulltext(query, limit=top_k)

            # Format results consistently
            formatted_results = []
            for article in results:
                # Calculate simple keyword similarity score
                keyword_score = self._calculate_keyword_similarity(query, article)
                formatted_results.append(
                    {"article": article, "similarity_score": keyword_score, "rank": len(formatted_results) + 1}
                )

            return formatted_results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _calculate_keyword_similarity(self, query: str, article: Dict) -> float:
        """Calculate keyword-based similarity score."""
        article_text = f"{article.get('title', '')} {article.get('content', '')}"

        # Use fuzzy matching for similarity
        return fuzz.partial_ratio(query.lower(), article_text.lower()) / 100.0

    def _combine_search_results(
        self, semantic_results: List[Dict], keyword_results: List[Dict], semantic_weight: float, keyword_weight: float
    ) -> List[Dict]:
        """Combine and rerank search results."""
        combined_scores = {}

        # Add semantic scores
        for result in semantic_results:
            article_id = result["article"]["id"]
            combined_scores[article_id] = {
                "article": result["article"],
                "semantic_score": result["similarity_score"],
                "keyword_score": 0.0,
            }

        # Add keyword scores
        for result in keyword_results:
            article_id = result["article"]["id"]
            if article_id in combined_scores:
                combined_scores[article_id]["keyword_score"] = result["similarity_score"]
            else:
                combined_scores[article_id] = {
                    "article": result["article"],
                    "semantic_score": 0.0,
                    "keyword_score": result["similarity_score"],
                }

        # Calculate combined scores
        final_results = []
        for article_id, scores in combined_scores.items():
            combined_score = semantic_weight * scores["semantic_score"] + keyword_weight * scores["keyword_score"]

            final_results.append(
                {
                    "article": scores["article"],
                    "similarity_score": combined_score,
                    "semantic_score": scores["semantic_score"],
                    "keyword_score": scores["keyword_score"],
                }
            )

        # Sort by combined score
        final_results.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Add ranks
        for i, result in enumerate(final_results):
            result["rank"] = i + 1

        return final_results

    def find_similar_articles(self, article_id: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Find articles similar to a given article.

        Args:
            article_id: ID of the reference article
            top_k: Number of similar articles to return
            threshold: Minimum similarity threshold

        Returns:
            List of similar articles
        """
        # Get the reference article
        article = self.db.get_article_by_id(article_id)
        if not article:
            logger.error(f"Article not found: {article_id}")
            return []

        # Create query from article title and content
        query = f"{article.get('title', '')} {article.get('content', '')}"

        # Perform semantic search
        results = self.semantic_search(query, top_k + 1, threshold)  # +1 to exclude self

        # Remove the reference article from results
        filtered_results = [r for r in results if r["article"]["id"] != article_id]

        return filtered_results[:top_k]

    def detect_duplicates(
        self, similarity_threshold: float = 0.85, batch_size: int = 100
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Detect duplicate or near-duplicate articles using MinHash and semantic similarity.

        Args:
            similarity_threshold: Threshold for considering articles as duplicates
            batch_size: Batch size for processing

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        logger.info("Starting duplicate detection")

        # Get all articles
        articles = self.db.get_all_articles_with_content()
        if len(articles) < 2:
            logger.info("Not enough articles for duplicate detection")
            return []

        duplicates = []

        # Build MinHash signatures for all articles
        minhash_signatures = {}
        for article in tqdm(articles, desc="Building MinHash signatures"):
            text = f"{article.get('title', '')} {article.get('content', '')}"
            text = self._preprocess_text(text)

            if text.strip():
                signature = self._create_minhash(text)
                minhash_signatures[article["id"]] = {"signature": signature, "article": article}

        # Find potential duplicates using LSH
        logger.info("Building LSH Forest for duplicate detection")
        lsh_forest = LSHForest(num_perm=128)

        article_ids = list(minhash_signatures.keys())
        for article_id in article_ids:
            lsh_forest.add(article_id, minhash_signatures[article_id]["signature"])

        lsh_forest.index()

        # Check for duplicates
        checked_pairs = set()

        for article_id in tqdm(article_ids, desc="Detecting duplicates"):
            # Get candidates from LSH
            candidates = lsh_forest.query(minhash_signatures[article_id]["signature"], 10)

            for candidate_id in candidates:
                if candidate_id == article_id:
                    continue

                # Avoid checking the same pair twice
                pair_key = tuple(sorted([article_id, candidate_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Calculate Jaccard similarity
                jaccard_sim = minhash_signatures[article_id]["signature"].jaccard(
                    minhash_signatures[candidate_id]["signature"]
                )

                if jaccard_sim >= similarity_threshold:
                    # Also check semantic similarity for confirmation
                    article1 = minhash_signatures[article_id]["article"]
                    article2 = minhash_signatures[candidate_id]["article"]

                    text1 = f"{article1.get('title', '')} {article1.get('content', '')}"
                    text2 = f"{article2.get('title', '')} {article2.get('content', '')}"

                    semantic_sim = self._calculate_semantic_similarity(text1, text2)

                    # Use average of Jaccard and semantic similarity
                    combined_similarity = (jaccard_sim + semantic_sim) / 2

                    if combined_similarity >= similarity_threshold:
                        duplicates.append((article1, article2, combined_similarity))

        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates

    def _create_minhash(self, text: str, num_perm: int = 128) -> MinHash:
        """Create MinHash signature for text."""
        # Tokenize text into shingles
        shingles = self._create_shingles(text)

        # Create MinHash
        minhash = MinHash(num_perm=num_perm)
        for shingle in shingles:
            minhash.update(shingle.encode("utf8"))

        return minhash

    def _create_shingles(self, text: str, k: int = 3) -> set:
        """Create k-shingles from text."""
        text = text.lower()
        words = text.split()

        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i : i + k])
            shingles.add(shingle)

        return shingles

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            embeddings = self.generate_embeddings([text1, text2], show_progress=False)
            if embeddings.shape[0] == 2:
                # Cosine similarity (embeddings are already normalized)
                similarity = np.dot(embeddings[0], embeddings[1])
                return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to calculate semantic similarity: {e}")

        return 0.0

    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            index_dir = Path("semantic_search_cache")
            index_dir.mkdir(exist_ok=True)

            # Save FAISS index
            faiss.write_index(self.faiss_index, str(index_dir / "faiss_index.bin"))

            # Save article IDs
            with open(index_dir / "article_ids.json", "w") as f:
                json.dump(self.article_ids, f)

            logger.info("Search index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _load_index(self) -> bool:
        """Load FAISS index from disk."""
        try:
            index_dir = Path("semantic_search_cache")
            index_file = index_dir / "faiss_index.bin"
            ids_file = index_dir / "article_ids.json"

            if not (index_file.exists() and ids_file.exists()):
                return False

            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))

            # Load article IDs
            with open(ids_file, "r") as f:
                self.article_ids = json.load(f)

            logger.info(f"Search index loaded with {len(self.article_ids)} articles")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# Utility functions for external use
def initialize_semantic_search(model_name: str = "all-MiniLM-L6-v2") -> SemanticSearchEngine:
    """Initialize and return a semantic search engine."""
    return SemanticSearchEngine(model_name=model_name)


def build_search_index(engine: SemanticSearchEngine) -> int:
    """Build search index for all articles."""
    return engine.build_article_index()


def search_articles_semantic(engine: SemanticSearchEngine, query: str, top_k: int = 10) -> List[Dict]:
    """Perform semantic search for articles."""
    return engine.semantic_search(query, top_k)


def search_articles_hybrid(engine: SemanticSearchEngine, query: str, top_k: int = 10) -> List[Dict]:
    """Perform hybrid semantic + keyword search."""
    return engine.hybrid_search(query, top_k)


def find_duplicate_articles(engine: SemanticSearchEngine, threshold: float = 0.85) -> List[Tuple[Dict, Dict, float]]:
    """Find duplicate articles."""
    return engine.detect_duplicates(threshold)


if __name__ == "__main__":
    # CLI interface for testing
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Search Engine CLI")
    parser.add_argument("command", choices=["build", "search", "duplicates", "similar"])
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--article-id", help="Article ID for similarity search")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")

    args = parser.parse_args()

    # Initialize engine
    engine = initialize_semantic_search(args.model)

    if args.command == "build":
        count = build_search_index(engine)
        print(f"Built search index with {count} articles")

    elif args.command == "search":
        if not args.query:
            print("Error: --query required for search")
            exit(1)

        results = search_articles_hybrid(engine, args.query, args.top_k)
        print(f"Found {len(results)} results for '{args.query}':")
        for result in results:
            print(f"  [{result['rank']}] {result['article']['title'][:80]}...")
            print(f"      Score: {result['similarity_score']:.3f}")

    elif args.command == "duplicates":
        duplicates = find_duplicate_articles(engine, args.threshold)
        print(f"Found {len(duplicates)} duplicate pairs:")
        for art1, art2, score in duplicates:
            print(f"  Similarity: {score:.3f}")
            print(f"    1: {art1['title'][:60]}...")
            print(f"    2: {art2['title'][:60]}...")

    elif args.command == "similar":
        if not args.article_id:
            print("Error: --article-id required for similar command")
            exit(1)

        similar = engine.find_similar_articles(args.article_id, args.top_k)
        print(f"Found {len(similar)} similar articles:")
        for result in similar:
            print(f"  [{result['rank']}] {result['article']['title'][:80]}...")
            print(f"      Score: {result['similarity_score']:.3f}")
