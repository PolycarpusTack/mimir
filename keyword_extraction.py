"""
Keyword extraction module for Mimir.
Implements multiple algorithms: TF-IDF, YAKE, and TextRank.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import spacy
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from summa import keywords as textrank_keywords

logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_KEYWORDS = 10
MIN_TEXT_LENGTH = 50
MAX_FEATURES = 1000
MIN_KEYWORD_LENGTH = 2
MAX_KEYWORD_LENGTH = 4  # Max words in multi-word keywords
YAKE_DEDUPLICATION_THRESHOLD = 0.9
YAKE_MAX_NGRAM_SIZE = 3
TEXTRANK_RATIO = 0.2
ENSEMBLE_WEIGHT_TFIDF = 0.4
ENSEMBLE_WEIGHT_YAKE = 0.3
ENSEMBLE_WEIGHT_TEXTRANK = 0.3
DEFAULT_LANGUAGE = "en"
DOMAIN_TERM_BOOST = 0.3  # 30% score boost for domain-specific terms


class KeywordExtractor:
    """Multi-algorithm keyword extraction system."""

    # Domain-specific terms (in production, load from config)
    DOMAIN_TERMS = {
        "technology": {
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "neural network",
            "algorithm",
            "data",
            "software",
            "hardware",
            "cloud",
            "api",
            "blockchain",
            "cryptocurrency",
            "cybersecurity",
            "iot",
            "automation",
            "robotics",
            "quantum",
            "5g",
            "edge computing",
        },
        "finance": {
            "investment",
            "portfolio",
            "stock",
            "bond",
            "equity",
            "debt",
            "revenue",
            "profit",
            "loss",
            "earnings",
            "dividend",
            "ipo",
            "merger",
            "acquisition",
            "valuation",
            "market cap",
            "trading",
            "hedge fund",
            "mutual fund",
            "etf",
            "forex",
            "commodity",
        },
        "healthcare": {
            "patient",
            "treatment",
            "diagnosis",
            "therapy",
            "medication",
            "clinical",
            "trial",
            "fda",
            "approval",
            "disease",
            "vaccine",
            "hospital",
            "healthcare",
            "medical",
            "pharmaceutical",
            "biotech",
            "drug",
            "surgery",
            "prevention",
            "epidemic",
            "pandemic",
        },
    }

    def __init__(self, stopwords_dir: Optional[Path] = None):
        """Initialize keyword extractor with stopwords and models."""
        self.stopwords_dir = stopwords_dir
        self.stopwords = self._load_stopwords()
        self.models: Dict[str, Language] = {}
        self._load_models()

        # Configure extractors
        self.tfidf_vectorizer = None
        self.yake_extractor = None

    def _load_stopwords(self) -> Dict[str, Set[str]]:
        """Load stopwords for multiple languages."""
        # Try to load from files first
        if self.stopwords_dir and self.stopwords_dir.exists():
            return self._load_stopwords_from_files()

        # Fallback to basic stopwords
        stopwords = {
            "en": {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "as",
                "is",
                "was",
                "are",
                "were",
                "been",
                "be",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "shall",
                "can",
                "need",
                "i",
                "you",
                "he",
                "she",
                "it",
                "we",
                "they",
                "them",
                "their",
                "this",
                "that",
                "these",
                "those",
                "which",
                "who",
                "whom",
                "whose",
                "what",
                "where",
                "when",
                "why",
                "how",
                "all",
                "each",
                "every",
                "some",
                "any",
                "few",
                "more",
                "most",
                "other",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "up",
                "down",
                "out",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            },
            "nl": {
                "de",
                "het",
                "een",
                "en",
                "van",
                "in",
                "op",
                "te",
                "dat",
                "die",
                "is",
                "was",
                "zijn",
                "er",
                "aan",
                "met",
                "als",
                "voor",
                "had",
                "maar",
                "om",
                "dan",
                "zou",
                "of",
                "kunnen",
                "bij",
                "ook",
                "uit",
            },
            "de": {
                "der",
                "die",
                "das",
                "den",
                "dem",
                "des",
                "und",
                "in",
                "von",
                "zu",
                "mit",
                "auf",
                "ist",
                "im",
                "für",
                "sich",
                "ein",
                "eine",
                "einer",
                "als",
                "aus",
                "nach",
                "bei",
                "um",
                "es",
                "werden",
                "wurde",
                "kann",
            },
            "fr": {
                "le",
                "la",
                "les",
                "un",
                "une",
                "de",
                "du",
                "des",
                "et",
                "à",
                "au",
                "aux",
                "en",
                "dans",
                "sur",
                "pour",
                "par",
                "avec",
                "sans",
                "que",
                "qui",
                "est",
                "sont",
                "avoir",
                "être",
                "faire",
                "ce",
                "ces",
                "son",
            },
        }
        return stopwords

    def _load_stopwords_from_files(self) -> Dict[str, Set[str]]:
        """Load stopwords from JSON files."""
        stopwords = {}
        for lang_file in self.stopwords_dir.glob("*.json"):
            lang = lang_file.stem
            try:
                with open(lang_file, "r", encoding="utf-8") as f:
                    stopwords[lang] = set(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load stopwords for {lang}: {e}")
        return stopwords

    def _load_models(self) -> None:
        """Load language models for lemmatization."""
        models = {"en": "en_core_web_md", "nl": "nl_core_news_md", "de": "de_core_news_md", "fr": "fr_core_news_md"}

        for lang, model_name in models.items():
            try:
                self.models[lang] = spacy.load(model_name)
                logger.info(f"Loaded {lang} model for keyword extraction")
            except Exception as e:
                logger.warning(f"Could not load {lang} model: {e}")

    def extract_keywords(
        self, text: str, language: str = DEFAULT_LANGUAGE, method: str = "all", num_keywords: int = DEFAULT_NUM_KEYWORDS
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords using specified method(s).

        Args:
            text: Input text
            language: Language code
            method: Extraction method ('tfidf', 'yake', 'textrank', 'all')
            num_keywords: Number of keywords to extract

        Returns:
            Dictionary mapping method names to list of (keyword, score) tuples
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return {}

        results = {}

        if method in ["tfidf", "all"]:
            results["tfidf"] = self._extract_tfidf(text, language, num_keywords)

        if method in ["yake", "all"]:
            results["yake"] = self._extract_yake(text, language, num_keywords)

        if method in ["textrank", "all"]:
            results["textrank"] = self._extract_textrank(text, language, num_keywords)

        if method == "all":
            # Also compute ensemble keywords
            results["ensemble"] = self._ensemble_keywords(results, num_keywords)

        return results

    def _extract_tfidf(self, text: str, language: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF algorithm."""
        try:
            # Get stopwords for language
            stop_words = self.stopwords.get(language, self.stopwords["en"])

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                stop_words=list(stop_words),
                ngram_range=(1, MAX_KEYWORD_LENGTH),
                max_features=MAX_FEATURES,
                min_df=1,
                use_idf=True,
                smooth_idf=True,
            )

            # For single document TF-IDF, we use fit_transform with the document itself
            # This gives us term frequencies weighted by their uniqueness
            tfidf_matrix = vectorizer.fit_transform([text])

            # Get feature names and scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix[0].toarray().flatten()

            # Create keyword-score pairs
            keyword_scores = [(feature_names[i], scores[i]) for i, _ in enumerate(feature_names) if scores[i] > 0]

            # Sort by score and return top keywords
            keyword_scores.sort(key=lambda x: x[1], reverse=True)

            # Filter out short keywords and numbers
            filtered_keywords = [
                (kw, score) for kw, score in keyword_scores if len(kw) >= MIN_KEYWORD_LENGTH and not kw.isdigit()
            ]

            return filtered_keywords[:num_keywords]

        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return []

    def _extract_yake(self, text: str, language: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE algorithm."""
        try:
            # Configure YAKE
            yake_extractor = yake.KeywordExtractor(
                lan=language,
                n=YAKE_MAX_NGRAM_SIZE,
                dedupLim=YAKE_DEDUPLICATION_THRESHOLD,
                dedupFunc="seqm",
                windowsSize=1,
                top=num_keywords,
            )

            # Extract keywords
            keywords = yake_extractor.extract_keywords(text)

            # YAKE returns lower scores for better keywords, so we invert
            max_score = max([score for _, score in keywords]) if keywords else 1.0
            normalized_keywords = [(kw, 1.0 - (score / max_score)) for kw, score in keywords]

            return normalized_keywords

        except Exception as e:
            logger.error(f"YAKE extraction failed: {e}")
            return []

    def _extract_textrank(self, text: str, language: str, num_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using TextRank algorithm."""
        try:
            # Get language model for lemmatization
            nlp = self.models.get(language, self.models.get("en"))

            if nlp:
                # Process text for lemmatization
                doc = nlp(text)
                processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
            else:
                processed_text = text

            # Extract keywords using TextRank
            keywords_text = textrank_keywords(
                processed_text, words=num_keywords, split=True, scores=True, deaccent=True
            )

            # Convert to list of tuples
            keywords = []
            for kw_score in keywords_text:
                if isinstance(kw_score, tuple) and len(kw_score) == 2:
                    keywords.append(kw_score)

            return keywords[:num_keywords]

        except Exception as e:
            logger.error(f"TextRank extraction failed: {e}")
            return []

    def _ensemble_keywords(
        self, results: Dict[str, List[Tuple[str, float]]], num_keywords: int
    ) -> List[Tuple[str, float]]:
        """Combine keywords from multiple methods using ensemble approach."""
        # Aggregate scores from all methods
        keyword_scores = defaultdict(list)

        for method, keywords in results.items():
            if method == "ensemble":
                continue

            # Normalize scores to [0, 1] range
            if keywords:
                max_score = max(score for _, score in keywords)
                min_score = min(score for _, score in keywords)
                score_range = max_score - min_score if max_score != min_score else 1

                for keyword, score in keywords:
                    normalized_score = (score - min_score) / score_range
                    keyword_scores[keyword.lower()].append(normalized_score)

        # Calculate ensemble scores
        ensemble_keywords = []
        for keyword, scores in keyword_scores.items():
            # Use weighted average with preference for keywords found by multiple methods
            frequency_weight = len(scores) / len(results)
            avg_score = sum(scores) / len(scores)
            ensemble_score = avg_score * (1 + frequency_weight) / 2
            ensemble_keywords.append((keyword, ensemble_score))

        # Sort by ensemble score
        ensemble_keywords.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword, score in ensemble_keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append((keyword, score))

        return unique_keywords[:num_keywords]

    def extract_domain_specific_keywords(
        self,
        text: str,
        domain: str = "technology",
        language: str = DEFAULT_LANGUAGE,
        num_keywords: int = DEFAULT_NUM_KEYWORDS,
    ) -> List[Tuple[str, float]]:
        """
        Extract domain-specific keywords.

        Args:
            text: Input text
            domain: Domain/industry (technology, finance, healthcare, etc.)
            language: Language code
            num_keywords: Number of keywords to extract

        Returns:
            List of (keyword, relevance_score) tuples
        """
        # Get domain terms
        domain_terms_set = self.DOMAIN_TERMS.get(domain.lower(), set())

        # Get general keywords first
        all_keywords = self.extract_keywords(text, language, "all", num_keywords * 2)

        # Score keywords based on domain relevance
        domain_keywords = []

        for method_keywords in all_keywords.values():
            for keyword, score in method_keywords:
                # Check if keyword contains domain terms
                keyword_lower = keyword.lower()
                domain_boost = 0

                for domain_term in domain_terms_set:
                    if domain_term in keyword_lower:
                        domain_boost = DOMAIN_TERM_BOOST
                        break

                # Calculate domain-adjusted score
                domain_score = score * (1 + domain_boost)
                domain_keywords.append((keyword, domain_score))

        # Remove duplicates and sort
        seen = set()
        unique_domain_keywords = []
        for keyword, score in sorted(domain_keywords, key=lambda x: x[1], reverse=True):
            if keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_domain_keywords.append((keyword, score))

        return unique_domain_keywords[:num_keywords]


def extract_keywords_from_article(
    article: Dict[str, Any], methods: Optional[List[str]] = None, num_keywords: int = DEFAULT_NUM_KEYWORDS
) -> Dict[str, Any]:
    """
    Extract keywords from a news article using multiple methods.

    Args:
        article: Article dictionary with content
        methods: List of methods to use (defaults to all)
        num_keywords: Number of keywords to extract

    Returns:
        Dictionary with keywords and metadata
    """
    extractor = KeywordExtractor()

    # Get text content
    content = article.get("content", "")
    title = article.get("title", "")
    full_text = f"{title}\n\n{content}"

    # Get language
    language = "en"
    if "preprocessed" in article:
        language = article["preprocessed"].get("language", "en")

    # Determine methods to use
    if not methods:
        methods = ["tfidf", "yake", "textrank"]

    # Extract keywords
    method = "all" if len(methods) > 1 else methods[0]
    keywords = extractor.extract_keywords(full_text, language, method, num_keywords)

    # Also extract domain-specific keywords if we can detect the domain
    domain_keywords = {}
    if "category" in article or "domain" in article:
        domain = article.get("domain", article.get("category", "general"))
        domain_keywords = extractor.extract_domain_specific_keywords(full_text, domain, language, num_keywords // 2)

    # Prepare response
    result = {
        "keywords": keywords,
        "domain_keywords": domain_keywords,
        "language": language,
        "methods_used": methods,
        "total_keywords": sum(len(kw_list) for kw_list in keywords.values()),
    }

    return result


if __name__ == "__main__":
    # Test keyword extraction
    logging.basicConfig(level=logging.INFO)

    test_text = """
    Artificial Intelligence and Machine Learning Continue to Transform Industries
    
    The rapid advancement of artificial intelligence (AI) and machine learning (ML) 
    technologies is revolutionizing how businesses operate across all sectors. 
    From healthcare to finance, companies are leveraging deep learning algorithms 
    and neural networks to automate processes, enhance decision-making, and create 
    new products and services.
    
    In the healthcare industry, AI-powered diagnostic tools are helping doctors 
    detect diseases earlier and with greater accuracy. Machine learning models 
    trained on vast datasets of medical images can identify patterns that might 
    be missed by human eyes.
    
    The financial sector is using AI for fraud detection, risk assessment, and 
    algorithmic trading. Natural language processing enables banks to analyze 
    customer sentiment and provide better service through chatbots and virtual 
    assistants.
    
    As we move forward, the integration of AI and ML will only accelerate, with 
    experts predicting that these technologies will be fundamental to maintaining 
    competitive advantage in the digital economy.
    """

    extractor = KeywordExtractor()

    # Test all methods
    print("Extracting keywords using all methods:")
    all_keywords = extractor.extract_keywords(test_text, "en", "all", 15)

    for method, keywords in all_keywords.items():
        print(f"\n{method.upper()} Keywords:")
        for keyword, score in keywords[:10]:
            print(f"  - {keyword}: {score:.3f}")

    # Test domain-specific extraction
    print("\n\nDomain-specific keywords (Technology):")
    tech_keywords = extractor.extract_domain_specific_keywords(test_text, "technology", "en", 10)
    for keyword, score in tech_keywords:
        print(f"  - {keyword}: {score:.3f}")
