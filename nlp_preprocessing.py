"""
Text preprocessing pipeline for Mimir NLP analysis.
Handles HTML cleaning, text normalization, and language detection.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import chardet
import ftfy
import spacy
from bs4 import BeautifulSoup
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
from spacy.language import Language
from unidecode import unidecode

polyglot_logger.setLevel(logging.ERROR)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 1000000
DEFAULT_ENCODING = "utf-8"
LANGUAGE_CONFIDENCE_THRESHOLD = 0.9
CONFIDENCE_NORMALIZATION_FACTOR = 100.0
DEFAULT_MAX_SENTENCES = 1000
DEFAULT_BATCH_SIZE = 10


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline for news articles."""

    # Class-level compiled regex patterns for efficiency
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
    HASHTAG_PATTERN = re.compile(r"#\S+")
    MENTION_PATTERN = re.compile(r"@\S+")
    WHITESPACE_PATTERN = re.compile(r"\s+")
    NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")

    def __init__(self):
        """Initialize the text preprocessor with language models."""
        self.supported_languages = ["en", "nl", "de", "fr"]
        self.language_models: Dict[str, Language] = {}
        self._load_language_models()

    def _load_language_models(self) -> None:
        """Load spaCy language models for supported languages."""
        model_mapping = {
            "en": "en_core_web_md",
            "nl": "nl_core_news_md",
            "de": "de_core_news_md",
            "fr": "fr_core_news_md",
        }

        for lang, model_name in model_mapping.items():
            try:
                self.language_models[lang] = spacy.load(model_name)
                logger.info(f"Loaded {lang} language model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load {lang} model {model_name}: {e}")

    def clean_html(self, html_content: str) -> str:
        """
        Remove HTML tags and extract clean text content.

        Args:
            html_content: Raw HTML content

        Returns:
            Clean text without HTML tags
        """
        if not html_content:
            return ""

        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text and normalize whitespace
            text = soup.get_text(separator=" ")
            text = self.whitespace_pattern.sub(" ", text).strip()

            return text
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html_content

    def fix_encoding(self, text: Union[str, bytes]) -> str:
        """
        Fix text encoding issues and normalize unicode.

        Args:
            text: Input text with potential encoding issues

        Returns:
            Text with fixed encoding
        """
        if not text:
            return ""

        try:
            # If text is bytes, detect encoding first
            if isinstance(text, bytes):
                detected = chardet.detect(text)
                encoding = detected.get("encoding", DEFAULT_ENCODING)
                text = text.decode(encoding, errors="ignore")

            # Fix any remaining encoding issues
            text = ftfy.fix_text(text)

            return text
        except Exception as e:
            logger.error(f"Error fixing encoding: {e}")
            return str(text)

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the text.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return "unknown", 0.0

        try:
            # Use polyglot for language detection
            detector = Detector(text)
            lang = detector.language

            # Map polyglot codes to our standard codes
            lang_code = lang.code
            if lang_code not in self.supported_languages:
                lang_code = "unknown"

            return lang_code, lang.confidence / CONFIDENCE_NORMALIZATION_FACTOR
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

            # Fallback: simple heuristic based on common words
            text_lower = text.lower()
            if any(word in text_lower for word in ["the", "and", "of", "to", "in"]):
                return "en", 0.5
            elif any(word in text_lower for word in ["de", "het", "een", "van", "en"]):
                return "nl", 0.5
            elif any(word in text_lower for word in ["der", "die", "das", "und", "ist"]):
                return "de", 0.5
            elif any(word in text_lower for word in ["le", "la", "les", "de", "et"]):
                return "fr", 0.5

            return "unknown", 0.0

    def normalize_text(self, text: str, preserve_case: bool = True) -> str:
        """
        Normalize text for consistent processing.

        Args:
            text: Input text
            preserve_case: Whether to preserve original case

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Fix encoding issues first
        text = self.fix_encoding(text)

        # Replace URLs with placeholder
        text = self.URL_PATTERN.sub("[URL]", text)

        # Replace emails with placeholder
        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()

        # Optionally convert to lowercase
        if not preserve_case:
            text = text.lower()

        return text

    def extract_sentences(self, text: str, language: str = "en") -> List[str]:
        """
        Extract sentences from text using language-specific models.

        Args:
            text: Input text
            language: Language code

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Use appropriate language model
        if language in self.language_models:
            nlp = self.language_models[language]
        else:
            nlp = self.language_models.get("en")  # Fallback to English

        if not nlp:
            # Simple fallback sentence splitting
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

        try:
            # Use spaCy for sentence segmentation
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        except Exception as e:
            logger.error(f"Error extracting sentences: {e}")
            # Fallback to simple splitting
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

    def preprocess_article(self, article: Dict) -> Dict:
        """
        Complete preprocessing pipeline for a news article.

        Args:
            article: Dictionary containing article data with 'content' and optionally 'title'

        Returns:
            Preprocessed article with additional fields
        """
        try:
            # Get content and title
            content = article.get("content", "")
            title = article.get("title", "")

            # Clean HTML if present
            if "<" in content and ">" in content:
                content = self.clean_html(content)
            if "<" in title and ">" in title:
                title = self.clean_html(title)

            # Fix encoding
            content = self.fix_encoding(content)
            title = self.fix_encoding(title)

            # Combine title and content for full analysis
            full_text = f"{title}\n\n{content}" if title else content

            # Detect language
            language, lang_confidence = self.detect_language(full_text)

            # Normalize text
            normalized_content = self.normalize_text(content)
            normalized_title = self.normalize_text(title)

            # Extract sentences
            sentences = self.extract_sentences(normalized_content, language)

            # Add preprocessing results to article
            article["preprocessed"] = {
                "content": normalized_content,
                "title": normalized_title,
                "language": language,
                "language_confidence": lang_confidence,
                "sentences": sentences,
                "sentence_count": len(sentences),
                "word_count": len(normalized_content.split()),
                "char_count": len(normalized_content),
            }

            return article

        except Exception as e:
            logger.error(f"Error preprocessing article: {e}")
            # Return article with minimal preprocessing
            article["preprocessed"] = {
                "content": content,
                "title": title,
                "language": "unknown",
                "language_confidence": 0.0,
                "error": str(e),
            }
            return article


def batch_preprocess_articles(
    articles: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Dict[str, Any]]:
    """
    Preprocess multiple articles in batches for efficiency.

    Args:
        articles: List of article dictionaries
        batch_size: Number of articles to process in each batch

    Returns:
        List of preprocessed articles
    """
    preprocessor = TextPreprocessor()
    processed_articles = []

    total = len(articles)
    logger.info(f"Starting preprocessing of {total} articles")

    for i in range(0, total, batch_size):
        batch = articles[i : i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({i+1}-{min(i+batch_size, total)} of {total})")

        for article in batch:
            processed_article = preprocessor.preprocess_article(article)
            processed_articles.append(processed_article)

    logger.info(f"Completed preprocessing of {total} articles")
    return processed_articles


if __name__ == "__main__":
    # Test the preprocessing pipeline
    logging.basicConfig(level=logging.INFO)

    # Test article
    test_article = {
        "title": "Breaking: Major Tech Company Announces AI Breakthrough",
        "content": """<p>In a surprising announcement today, TechCorp revealed their latest 
        artificial intelligence system that can understand and generate human-like text 
        with unprecedented accuracy.</p>
        
        <p>The CEO stated: "This is a game-changer for the industry. Our AI can now 
        process multiple languages including English, Dutch, German, and French."</p>
        
        <p>Contact: info@techcorp.com or visit https://techcorp.com for more details.</p>""",
    }

    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess_article(test_article)

    print("Preprocessing Results:")
    print(
        f"Language: {result['preprocessed']['language']} "
        f"(confidence: {result['preprocessed']['language_confidence']:.2f})"
    )
    print(f"Word count: {result['preprocessed']['word_count']}")
    print(f"Sentence count: {result['preprocessed']['sentence_count']}")
    print(
        f"\nFirst sentence: {result['preprocessed']['sentences'][0] if result['preprocessed'].get('sentences') else 'N/A'}"
    )
