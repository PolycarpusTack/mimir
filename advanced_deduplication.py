"""
Advanced Deduplication Engine
EPIC 3: Semantic Search & Similarity Engine

This module provides advanced deduplication capabilities using:
- MinHash and LSH for efficient similarity detection
- Semantic similarity using embeddings
- Cross-language deduplication
- Fuzzy title matching
- Content fingerprinting

Author: Claude Code
"""

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from datasketch import LSH, LSHForest, MinHash
from fuzzywuzzy import fuzz, process
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

# Text processing
from nltk.corpus import stopwords

# Progress tracking
from tqdm import tqdm

# Database and semantic search
from db_manager_semantic import SemanticDatabaseManager
from semantic_search import SemanticSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentFingerprinter:
    """
    Generate content fingerprints for duplicate detection.
    """

    def __init__(self, shingle_size: int = 3, num_perm: int = 128):
        """
        Initialize the content fingerprinter.

        Args:
            shingle_size: Size of text shingles (n-grams)
            num_perm: Number of permutations for MinHash
        """
        self.shingle_size = shingle_size
        self.num_perm = num_perm

        # Load stopwords for multiple languages
        self.stopwords = set()
        try:
            self.stopwords.update(stopwords.words("english"))
            self.stopwords.update(stopwords.words("dutch"))
            self.stopwords.update(stopwords.words("german"))
            self.stopwords.update(stopwords.words("french"))
            self.stopwords.update(stopwords.words("spanish"))
        except LookupError:
            logger.warning("Some NLTK stopwords not available. Using basic English only.")
            self.stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for fingerprinting.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stopwords and len(word) > 2]

        return " ".join(words)

    def create_shingles(self, text: str) -> Set[str]:
        """
        Create shingles (n-grams) from text.

        Args:
            text: Input text

        Returns:
            Set of shingles
        """
        text = self.preprocess_text(text)
        words = text.split()

        if len(words) < self.shingle_size:
            return {text}  # Return the whole text if too short

        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i : i + self.shingle_size])
            shingles.add(shingle)

        return shingles

    def create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for text.

        Args:
            text: Input text

        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self.create_shingles(text)

        for shingle in shingles:
            minhash.update(shingle.encode("utf-8"))

        return minhash

    def create_content_hash(self, title: str, content: str) -> str:
        """
        Create a simple content hash for exact duplicate detection.

        Args:
            title: Article title
            content: Article content

        Returns:
            SHA256 hash of normalized content
        """
        # Combine and normalize content
        combined = f"{title} {content}"
        normalized = self.preprocess_text(combined)

        # Create hash
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class CrossLanguageDeduplicator:
    """
    Handle cross-language duplicate detection.
    """

    def __init__(self):
        """Initialize cross-language deduplicator."""
        self.fingerprinter = ContentFingerprinter()

        # Language-specific preprocessing
        self.lang_stopwords = {}
        try:
            self.lang_stopwords["en"] = set(stopwords.words("english"))
            self.lang_stopwords["nl"] = set(stopwords.words("dutch"))
            self.lang_stopwords["de"] = set(stopwords.words("german"))
            self.lang_stopwords["fr"] = set(stopwords.words("french"))
            self.lang_stopwords["es"] = set(stopwords.words("spanish"))
        except LookupError:
            logger.warning("Limited language support for stopwords")
            self.lang_stopwords["en"] = {"the", "a", "an", "and", "or", "but"}

    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.

        Args:
            text: Input text

        Returns:
            Language code (e.g., 'en', 'de', 'fr')
        """
        try:
            return detect(text)
        except:
            return "en"  # Default to English

    def normalize_cross_language(self, text: str, language: str = None) -> str:
        """
        Normalize text for cross-language comparison.

        Args:
            text: Input text
            language: Language code (auto-detected if None)

        Returns:
            Normalized text
        """
        if language is None:
            language = self.detect_language(text)

        # Basic normalization
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # Remove language-specific stopwords
        if language in self.lang_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.lang_stopwords[language]]
            text = " ".join(words)

        return text.strip()

    def calculate_cross_language_similarity(
        self, text1: str, text2: str, lang1: str = None, lang2: str = None
    ) -> float:
        """
        Calculate similarity between texts in potentially different languages.

        Args:
            text1: First text
            text2: Second text
            lang1: Language of first text
            lang2: Language of second text

        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both texts
        norm1 = self.normalize_cross_language(text1, lang1)
        norm2 = self.normalize_cross_language(text2, lang2)

        # If texts are very short, use exact matching
        if len(norm1.split()) < 5 or len(norm2.split()) < 5:
            return 1.0 if norm1 == norm2 else 0.0

        # Use fuzzy matching for cross-language similarity
        ratio = fuzz.ratio(norm1, norm2) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        token_set_ratio = fuzz.token_set_ratio(norm1, norm2) / 100.0

        # Return the best score
        return max(ratio, token_sort_ratio, token_set_ratio)


class AdvancedDeduplicator:
    """
    Advanced deduplication engine with multiple detection strategies.
    """

    def __init__(self, semantic_engine: SemanticSearchEngine = None, db_manager: SemanticDatabaseManager = None):
        """
        Initialize the advanced deduplicator.

        Args:
            semantic_engine: Semantic search engine for embedding-based similarity
            db_manager: Database manager for storing results
        """
        self.semantic_engine = semantic_engine
        self.db_manager = db_manager or SemanticDatabaseManager()

        # Initialize components
        self.fingerprinter = ContentFingerprinter()
        self.cross_lang_dedup = CrossLanguageDeduplicator()

        # LSH for efficient similarity search
        self.lsh_forest = None
        self.article_signatures = {}

        # Statistics
        self.stats = {
            "total_articles": 0,
            "duplicate_pairs": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "cross_language_duplicates": 0,
            "semantic_duplicates": 0,
        }

    def build_lsh_index(self, articles: List[Dict[str, Any]]):
        """
        Build LSH index for efficient similarity search.

        Args:
            articles: List of article dictionaries
        """
        logger.info(f"Building LSH index for {len(articles)} articles")

        self.lsh_forest = LSHForest(num_perm=128)
        self.article_signatures = {}

        for article in tqdm(articles, desc="Building LSH index"):
            article_id = article["id"]

            # Create content signature
            combined_text = f"{article.get('title', '')} {article.get('content', '')}"
            signature = self.fingerprinter.create_minhash(combined_text)

            # Store signature
            self.article_signatures[article_id] = {"signature": signature, "article": article}

            # Add to LSH forest
            self.lsh_forest.add(article_id, signature)

        # Index the forest for querying
        self.lsh_forest.index()
        logger.info("LSH index built successfully")

    def find_exact_duplicates(self, articles: List[Dict[str, Any]]) -> List[Tuple[Dict, Dict, float]]:
        """
        Find exact duplicates using content hashing.

        Args:
            articles: List of article dictionaries

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        logger.info("Finding exact duplicates...")

        content_hashes = {}
        duplicates = []

        for article in tqdm(articles, desc="Computing content hashes"):
            title = article.get("title", "")
            content = article.get("content", "")

            content_hash = self.fingerprinter.create_content_hash(title, content)

            if content_hash in content_hashes:
                # Found exact duplicate
                original_article = content_hashes[content_hash]
                duplicates.append((original_article, article, 1.0))
                self.stats["exact_duplicates"] += 1
            else:
                content_hashes[content_hash] = article

        logger.info(f"Found {len(duplicates)} exact duplicate pairs")
        return duplicates

    def find_near_duplicates(
        self, articles: List[Dict[str, Any]], similarity_threshold: float = 0.8
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find near duplicates using MinHash and LSH.

        Args:
            articles: List of article dictionaries
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        logger.info("Finding near duplicates using LSH...")

        if not self.lsh_forest:
            self.build_lsh_index(articles)

        duplicates = []
        checked_pairs = set()

        for article_id, data in tqdm(self.article_signatures.items(), desc="Finding near duplicates"):
            # Query LSH for similar articles
            candidates = self.lsh_forest.query(data["signature"], 10)

            for candidate_id in candidates:
                if candidate_id == article_id:
                    continue

                # Avoid checking the same pair twice
                pair_key = tuple(sorted([article_id, candidate_id]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Calculate Jaccard similarity
                candidate_signature = self.article_signatures[candidate_id]["signature"]
                jaccard_sim = data["signature"].jaccard(candidate_signature)

                if jaccard_sim >= similarity_threshold:
                    article1 = data["article"]
                    article2 = self.article_signatures[candidate_id]["article"]

                    duplicates.append((article1, article2, jaccard_sim))
                    self.stats["near_duplicates"] += 1

        logger.info(f"Found {len(duplicates)} near duplicate pairs")
        return duplicates

    def find_semantic_duplicates(
        self, articles: List[Dict[str, Any]], similarity_threshold: float = 0.85
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find semantic duplicates using embeddings.

        Args:
            articles: List of article dictionaries
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        if not self.semantic_engine:
            logger.warning("No semantic engine available, skipping semantic deduplication")
            return []

        logger.info("Finding semantic duplicates...")

        duplicates = []

        # Get embeddings for all articles
        embeddings = {}
        for article in tqdm(articles, desc="Getting embeddings"):
            embedding = self.db_manager.get_embedding(article["id"])
            if embedding is not None:
                embeddings[article["id"]] = {"embedding": embedding, "article": article}

        if len(embeddings) < 2:
            logger.warning("Not enough embeddings for semantic deduplication")
            return duplicates

        # Compare all pairs
        article_ids = list(embeddings.keys())
        checked_pairs = set()

        for i, id1 in enumerate(tqdm(article_ids, desc="Computing semantic similarity")):
            for j, id2 in enumerate(article_ids[i + 1 :], i + 1):
                pair_key = tuple(sorted([id1, id2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Calculate cosine similarity
                emb1 = embeddings[id1]["embedding"]
                emb2 = embeddings[id2]["embedding"]

                # Cosine similarity (embeddings are already normalized)
                similarity = np.dot(emb1, emb2)

                if similarity >= similarity_threshold:
                    article1 = embeddings[id1]["article"]
                    article2 = embeddings[id2]["article"]

                    duplicates.append((article1, article2, float(similarity)))
                    self.stats["semantic_duplicates"] += 1

        logger.info(f"Found {len(duplicates)} semantic duplicate pairs")
        return duplicates

    def find_cross_language_duplicates(
        self, articles: List[Dict[str, Any]], similarity_threshold: float = 0.7
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find cross-language duplicates.

        Args:
            articles: List of article dictionaries
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        logger.info("Finding cross-language duplicates...")

        # Group articles by detected language
        articles_by_lang = defaultdict(list)

        for article in articles:
            title_content = f"{article.get('title', '')} {article.get('content', '')}"
            language = self.cross_lang_dedup.detect_language(title_content)
            articles_by_lang[language].append(article)

        duplicates = []
        languages = list(articles_by_lang.keys())

        # Compare articles across different languages
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i + 1 :]:
                logger.info(f"Comparing {lang1} vs {lang2}")

                articles1 = articles_by_lang[lang1]
                articles2 = articles_by_lang[lang2]

                for article1 in tqdm(articles1, desc=f"Cross-lang {lang1}-{lang2}"):
                    text1 = f"{article1.get('title', '')} {article1.get('content', '')}"

                    for article2 in articles2:
                        text2 = f"{article2.get('title', '')} {article2.get('content', '')}"

                        similarity = self.cross_lang_dedup.calculate_cross_language_similarity(
                            text1, text2, lang1, lang2
                        )

                        if similarity >= similarity_threshold:
                            duplicates.append((article1, article2, similarity))
                            self.stats["cross_language_duplicates"] += 1

        logger.info(f"Found {len(duplicates)} cross-language duplicate pairs")
        return duplicates

    def find_title_duplicates(
        self, articles: List[Dict[str, Any]], similarity_threshold: float = 0.9
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Find duplicates based on title similarity.

        Args:
            articles: List of article dictionaries
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of (article1, article2, similarity_score) tuples
        """
        logger.info("Finding title-based duplicates...")

        duplicates = []
        titles = [(article, article.get("title", "")) for article in articles if article.get("title")]

        for i, (article1, title1) in enumerate(tqdm(titles, desc="Comparing titles")):
            for article2, title2 in titles[i + 1 :]:
                # Calculate title similarity
                similarity = fuzz.ratio(title1.lower(), title2.lower()) / 100.0

                if similarity >= similarity_threshold:
                    duplicates.append((article1, article2, similarity))

        logger.info(f"Found {len(duplicates)} title-based duplicate pairs")
        return duplicates

    def comprehensive_deduplication(
        self, articles: List[Dict[str, Any]] = None, similarity_thresholds: Dict[str, float] = None
    ) -> Dict[str, List]:
        """
        Perform comprehensive deduplication using all available methods.

        Args:
            articles: List of articles to deduplicate (gets from DB if None)
            similarity_thresholds: Custom thresholds for different methods

        Returns:
            Dictionary with different types of duplicates found
        """
        if similarity_thresholds is None:
            similarity_thresholds = {"exact": 1.0, "near": 0.8, "semantic": 0.85, "cross_language": 0.7, "title": 0.9}

        if articles is None:
            logger.info("Retrieving articles from database...")
            articles = self.db_manager.get_all_articles_with_content()

        self.stats["total_articles"] = len(articles)
        logger.info(f"Starting comprehensive deduplication for {len(articles)} articles")

        results = {}

        # 1. Exact duplicates
        results["exact"] = self.find_exact_duplicates(articles)

        # 2. Near duplicates (MinHash/LSH)
        results["near"] = self.find_near_duplicates(articles, similarity_thresholds["near"])

        # 3. Title-based duplicates
        results["title"] = self.find_title_duplicates(articles, similarity_thresholds["title"])

        # 4. Semantic duplicates (if semantic engine available)
        if self.semantic_engine:
            results["semantic"] = self.find_semantic_duplicates(articles, similarity_thresholds["semantic"])
        else:
            results["semantic"] = []

        # 5. Cross-language duplicates
        results["cross_language"] = self.find_cross_language_duplicates(
            articles, similarity_thresholds["cross_language"]
        )

        # Calculate total unique duplicate pairs
        all_pairs = set()
        for duplicate_list in results.values():
            for art1, art2, score in duplicate_list:
                pair_key = tuple(sorted([art1["id"], art2["id"]]))
                all_pairs.add(pair_key)

        self.stats["duplicate_pairs"] = len(all_pairs)

        logger.info(f"Deduplication complete. Found {len(all_pairs)} unique duplicate pairs")
        return results

    def get_deduplication_report(self, results: Dict[str, List]) -> Dict[str, Any]:
        """
        Generate a comprehensive deduplication report.

        Args:
            results: Results from comprehensive_deduplication

        Returns:
            Detailed report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats.copy(),
            "summary": {},
            "details": {},
            "recommendations": [],
        }

        # Summary statistics
        total_duplicates = sum(len(dup_list) for dup_list in results.values())
        report["summary"] = {
            "total_articles_analyzed": self.stats["total_articles"],
            "total_duplicate_pairs_found": total_duplicates,
            "duplicate_percentage": (total_duplicates / max(1, self.stats["total_articles"])) * 100,
            "methods_used": list(results.keys()),
        }

        # Details by method
        for method, duplicates in results.items():
            if duplicates:
                similarities = [score for _, _, score in duplicates]
                report["details"][method] = {
                    "count": len(duplicates),
                    "avg_similarity": np.mean(similarities),
                    "min_similarity": np.min(similarities),
                    "max_similarity": np.max(similarities),
                }

        # Recommendations
        if total_duplicates > 0:
            if self.stats["exact_duplicates"] > 0:
                report["recommendations"].append("Remove exact duplicates immediately to save storage space")

            if self.stats["near_duplicates"] > 10:
                report["recommendations"].append("Consider implementing automatic near-duplicate filtering")

            if self.stats["cross_language_duplicates"] > 0:
                report["recommendations"].append("Review cross-language duplicates - they might be translations")

            high_semantic_dups = len(results.get("semantic", []))
            if high_semantic_dups > 5:
                report["recommendations"].append(
                    f"Found {high_semantic_dups} semantic duplicates - consider content source review"
                )
        else:
            report["recommendations"].append("No duplicates found - content sources are well-curated")

        return report

    def save_duplicates_to_database(self, results: Dict[str, List]):
        """
        Save duplicate findings to database for future reference.

        Args:
            results: Results from comprehensive_deduplication
        """
        # This could be implemented to store duplicate pairs in a dedicated table
        # For now, we'll just log the results
        logger.info("Saving duplicate findings to database...")

        # TODO: Implement database storage for duplicate pairs
        # Could create a 'duplicate_pairs' table with:
        # - article_id_1, article_id_2
        # - detection_method, similarity_score
        # - created_at, verified (for manual verification)

        logger.info("Duplicate findings saved (implementation pending)")


# Utility functions
def create_deduplicator(semantic_engine=None) -> AdvancedDeduplicator:
    """Create and return an advanced deduplicator instance."""
    return AdvancedDeduplicator(semantic_engine=semantic_engine)


def run_deduplication_analysis() -> Dict[str, Any]:
    """Run a complete deduplication analysis."""
    deduplicator = create_deduplicator()
    results = deduplicator.comprehensive_deduplication()
    report = deduplicator.get_deduplication_report(results)
    return report


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Deduplication CLI")
    parser.add_argument("command", choices=["analyze", "report", "exact", "near", "semantic"])
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")
    parser.add_argument(
        "--method",
        choices=["all", "exact", "near", "semantic", "cross_lang", "title"],
        default="all",
        help="Deduplication method",
    )
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    deduplicator = create_deduplicator()

    if args.command == "analyze":
        # Run comprehensive analysis
        results = deduplicator.comprehensive_deduplication()
        report = deduplicator.get_deduplication_report(results)

        print("Deduplication Analysis Results:")
        print(f"  Total articles: {report['summary']['total_articles_analyzed']}")
        print(f"  Duplicate pairs: {report['summary']['total_duplicate_pairs_found']}")
        print(f"  Duplicate percentage: {report['summary']['duplicate_percentage']:.2f}%")

        for method, details in report["details"].items():
            print(f"  {method}: {details['count']} pairs (avg sim: {details['avg_similarity']:.3f})")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Full report saved to {args.output}")

    elif args.command == "exact":
        # Find only exact duplicates
        articles = deduplicator.db_manager.get_all_articles_with_content()
        duplicates = deduplicator.find_exact_duplicates(articles)
        print(f"Found {len(duplicates)} exact duplicate pairs")

    elif args.command == "near":
        # Find near duplicates
        articles = deduplicator.db_manager.get_all_articles_with_content()
        duplicates = deduplicator.find_near_duplicates(articles, args.threshold)
        print(f"Found {len(duplicates)} near duplicate pairs")

    elif args.command == "semantic":
        # Find semantic duplicates
        articles = deduplicator.db_manager.get_all_articles_with_content()
        duplicates = deduplicator.find_semantic_duplicates(articles, args.threshold)
        print(f"Found {len(duplicates)} semantic duplicate pairs")
