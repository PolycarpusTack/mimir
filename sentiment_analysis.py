"""
Sentiment analysis module for Mimir.
Uses transformer models for advanced sentiment and emotion detection.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 512
TEXT_TO_TOKEN_RATIO = 4
DEFAULT_BATCH_SIZE = 16
MIN_TEXT_LENGTH = 1
DEFAULT_CONFIDENCE = 0.0
DEFAULT_SCORE = 0.0

# Confidence thresholds
VERY_HIGH_CONFIDENCE_THRESHOLD = 0.9
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.6
MIN_CONFIDENCE_THRESHOLD = 0.5

# Confidence scores
VERY_HIGH_CONFIDENCE_SCORE = 0.95
HIGH_CONFIDENCE_SCORE = 0.85
MEDIUM_CONFIDENCE_SCORE = 0.75
LOW_CONFIDENCE_SCORE = 0.65
MIN_CONFIDENCE_SCORE = 0.5

# Star rating normalization
NEUTRAL_STAR_RATING = 3
STAR_RATING_SCALE = 2.0
POSITIVE_STAR_THRESHOLD = 4
NEGATIVE_STAR_THRESHOLD = 2
VERY_POSITIVE_STAR_RATING = 5
VERY_NEGATIVE_STAR_RATING = 1

# Sentiment score thresholds
VERY_POSITIVE_THRESHOLD = 0.9
HIGH_IMPACT_SCORE_THRESHOLD = 0.8
MEDIUM_IMPACT_SCORE_THRESHOLD = 0.6
LOW_CONFIDENCE_INTERPRETATION_THRESHOLD = 0.6
HIGH_CONFIDENCE_INTERPRETATION_THRESHOLD = 0.85

# Urgency thresholds
HIGH_URGENCY_THRESHOLD = 2
MEDIUM_URGENCY_THRESHOLD = 1

# Device configuration
CUDA_DEVICE_ID = 0
CPU_DEVICE_ID = -1


class SentimentLabel(Enum):
    """Standard sentiment labels."""

    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    label: str
    score: float
    confidence: float
    raw_scores: Optional[Dict[str, float]] = None
    aspects: Optional[Dict[str, Any]] = None
    market_impact: Optional[str] = None
    urgency: Optional[str] = None


class SentimentAnalyzer:
    """Advanced sentiment analysis using transformer models."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize sentiment analyzer.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._setup_device(device)
        self.models = {}
        self.tokenizers = {}
        self._load_models()

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup compute device.

        Args:
            device: Device specification ('cuda', 'cpu', or None for auto-detect)

        Returns:
            Device string for PyTorch
        """
        if device:
            if device not in ["cuda", "cpu"]:
                logger.warning(f"Invalid device '{device}', falling back to auto-detect")
                device = None
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            else:
                return device

        # Auto-detect device
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"

    def _load_models(self):
        """Load pre-trained sentiment models."""
        # Model configurations
        model_configs = {
            "general": {"model_name": "nlptown/bert-base-multilingual-uncased-sentiment", "type": "standard"},
            "financial": {"model_name": "ProsusAI/finbert", "type": "financial"},
            "news": {"model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest", "type": "news"},
        }

        for model_key, config in model_configs.items():
            try:
                logger.info(f"Loading {model_key} sentiment model: {config['model_name']}")

                # Create pipeline for easier use
                self.models[model_key] = pipeline(
                    "sentiment-analysis",
                    model=config["model_name"],
                    device=CUDA_DEVICE_ID if self.device == "cuda" else CPU_DEVICE_ID,
                    truncation=True,
                    max_length=MAX_TEXT_LENGTH,
                )

                logger.info(f"Successfully loaded {model_key} model")

            except Exception as e:
                logger.error(f"Failed to load {model_key} model: {e}")

                # Fallback to a simple model
                try:
                    self.models[model_key] = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=CUDA_DEVICE_ID if self.device == "cuda" else CPU_DEVICE_ID,
                    )
                    logger.info(f"Loaded fallback model for {model_key}")
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")

    def analyze(self, text: str, model_type: str = "general", include_aspects: bool = False) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Input text
            model_type: Type of model to use ('general', 'financial', 'news')
            include_aspects: Whether to include aspect-based sentiment

        Returns:
            SentimentResult object
        """
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return SentimentResult(label="neutral", score=DEFAULT_SCORE, confidence=DEFAULT_CONFIDENCE)

        try:
            # Get appropriate model
            model = self.models.get(model_type, self.models.get("general"))
            if not model:
                raise ValueError(f"No model available for type: {model_type}")

            # Truncate text if too long
            if len(text) > MAX_TEXT_LENGTH * TEXT_TO_TOKEN_RATIO:
                text = text[: MAX_TEXT_LENGTH * TEXT_TO_TOKEN_RATIO]

            # Get sentiment prediction
            results = model(text)

            # Process results
            if results and results:
                result = results[0]

                # Normalize label and score
                label, score = self._normalize_sentiment(result["label"], result["score"], model_type)

                # Calculate confidence (how certain the model is)
                confidence = self._calculate_confidence(result["score"])

                # Create base result
                sentiment_result = SentimentResult(
                    label=label, score=score, confidence=confidence, raw_scores={result["label"]: result["score"]}
                )

                # Add aspect-based sentiment if requested
                if include_aspects:
                    aspects = self._extract_aspect_sentiment(text)
                    sentiment_result.aspects = aspects

                # Add market impact for financial texts
                if model_type == "financial":
                    sentiment_result.market_impact = self._assess_market_impact(label, score, text)
                    sentiment_result.urgency = self._assess_urgency(text)

                return sentiment_result

            else:
                # No results, return neutral
                return SentimentResult(label="neutral", score=DEFAULT_SCORE, confidence=DEFAULT_CONFIDENCE)

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return SentimentResult(
                label="neutral", score=DEFAULT_SCORE, confidence=DEFAULT_CONFIDENCE, raw_scores={"error": str(e)}
            )

    def _normalize_sentiment(self, label: str, score: float, model_type: str) -> Tuple[str, float]:
        """Normalize sentiment labels and scores across different models."""
        label_lower = label.lower()

        # FinBERT specific normalization
        if model_type == "financial":
            if "positive" in label_lower:
                return SentimentLabel.POSITIVE.value, score
            elif "negative" in label_lower:
                return SentimentLabel.NEGATIVE.value, -score
            else:
                return SentimentLabel.NEUTRAL.value, DEFAULT_SCORE

        # 5-star rating model (nlptown)
        if "star" in label_lower:
            star_rating = int(label_lower.split()[0])
            if star_rating >= POSITIVE_STAR_THRESHOLD:
                sentiment = SentimentLabel.POSITIVE.value
                if star_rating == VERY_POSITIVE_STAR_RATING:
                    sentiment = SentimentLabel.VERY_POSITIVE.value
            elif star_rating <= NEGATIVE_STAR_THRESHOLD:
                sentiment = SentimentLabel.NEGATIVE.value
                if star_rating == VERY_NEGATIVE_STAR_RATING:
                    sentiment = SentimentLabel.VERY_NEGATIVE.value
            else:
                sentiment = SentimentLabel.NEUTRAL.value

            # Normalize score to [-1, 1]
            normalized_score = (star_rating - NEUTRAL_STAR_RATING) / STAR_RATING_SCALE
            return sentiment, normalized_score

        # Standard positive/negative/neutral
        if "positive" in label_lower:
            if "very" in label_lower or score > VERY_POSITIVE_THRESHOLD:
                return SentimentLabel.VERY_POSITIVE.value, score
            return SentimentLabel.POSITIVE.value, score
        elif "negative" in label_lower:
            if "very" in label_lower or score > VERY_POSITIVE_THRESHOLD:
                return SentimentLabel.VERY_NEGATIVE.value, -score
            return SentimentLabel.NEGATIVE.value, -score
        else:
            return SentimentLabel.NEUTRAL.value, DEFAULT_SCORE

    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence based on prediction score."""
        # Higher scores indicate higher confidence
        # Scores close to 0.5 indicate low confidence
        if score > VERY_HIGH_CONFIDENCE_THRESHOLD:
            return VERY_HIGH_CONFIDENCE_SCORE
        elif score > HIGH_CONFIDENCE_THRESHOLD:
            return HIGH_CONFIDENCE_SCORE
        elif score > MEDIUM_CONFIDENCE_THRESHOLD:
            return MEDIUM_CONFIDENCE_SCORE
        elif score > LOW_CONFIDENCE_THRESHOLD:
            return LOW_CONFIDENCE_SCORE
        else:
            return MIN_CONFIDENCE_SCORE

    def _extract_aspect_sentiment(self, text: str) -> Dict[str, Any]:
        """Extract sentiment for specific aspects mentioned in text."""
        # Define aspects to look for
        aspects = {
            "product": ["product", "service", "offering", "solution"],
            "company": ["company", "organization", "firm", "corporation"],
            "performance": ["performance", "results", "earnings", "revenue"],
            "market": ["market", "industry", "sector", "competition"],
            "technology": ["technology", "innovation", "ai", "digital"],
        }

        aspect_sentiments = {}

        # Simple approach: check if aspect keywords appear in text
        # and analyze sentiment of sentences containing them
        sentences = text.split(".")

        for aspect, keywords in aspects.items():
            aspect_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    aspect_sentences.append(sentence)

            if aspect_sentences:
                # Analyze combined aspect sentences
                aspect_text = " ".join(aspect_sentences)
                aspect_result = self.analyze(aspect_text, model_type="general")

                aspect_sentiments[aspect] = {
                    "sentiment": aspect_result.label,
                    "score": aspect_result.score,
                    "mentions": len(aspect_sentences),
                }

        return aspect_sentiments

    def _assess_market_impact(self, sentiment: str, score: float, text: str) -> str:
        """Assess potential market impact of news."""
        # Keywords indicating high impact
        high_impact_keywords = {
            "breakthrough",
            "revolutionary",
            "disrupt",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "regulation",
            "earnings beat",
            "earnings miss",
            "guidance",
            "forecast",
            "unexpected",
        }

        # Check for high impact keywords
        text_lower = text.lower()
        has_high_impact = any(keyword in text_lower for keyword in high_impact_keywords)

        # Determine impact based on sentiment and keywords
        if abs(score) > HIGH_IMPACT_SCORE_THRESHOLD and has_high_impact:
            return "high"
        elif abs(score) > MEDIUM_IMPACT_SCORE_THRESHOLD or has_high_impact:
            return "medium"
        else:
            return "low"

    def _assess_urgency(self, text: str) -> str:
        """Assess urgency of news."""
        # Urgency indicators
        urgent_keywords = {"breaking", "just in", "urgent", "immediate", "now", "today", "alert", "warning", "critical"}

        text_lower = text.lower()
        urgency_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)

        if urgency_count >= HIGH_URGENCY_THRESHOLD:
            return "high"
        elif urgency_count >= MEDIUM_URGENCY_THRESHOLD:
            return "medium"
        else:
            return "low"

    def analyze_batch(
        self, texts: List[str], model_type: str = "general", batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[SentimentResult]:
        """
        Analyze sentiment for multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            model_type: Model type to use
            batch_size: Batch size for processing

        Returns:
            List of SentimentResult objects
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = []

            for text in batch:
                result = self.analyze(text, model_type)
                batch_results.append(result)

            results.extend(batch_results)

        return results


def analyze_article_sentiment(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze sentiment of a news article.

    Args:
        article: Article dictionary with content

    Returns:
        Dictionary with sentiment analysis results
    """
    analyzer = SentimentAnalyzer()

    # Get text content
    content = article.get("content", "")
    title = article.get("title", "")

    # Determine model type based on article category
    category = article.get("category", "").lower()
    if "finance" in category or "business" in category:
        model_type = "financial"
    elif "news" in category:
        model_type = "news"
    else:
        model_type = "general"

    # Analyze title and content separately
    title_sentiment = analyzer.analyze(title, model_type) if title else None
    content_sentiment = analyzer.analyze(content, model_type, include_aspects=True)

    # Combine results
    results = {
        "overall_sentiment": content_sentiment.label,
        "overall_score": content_sentiment.score,
        "confidence": content_sentiment.confidence,
        "title_sentiment": {"label": title_sentiment.label, "score": title_sentiment.score}
        if title_sentiment
        else None,
        "content_sentiment": {
            "label": content_sentiment.label,
            "score": content_sentiment.score,
            "aspects": content_sentiment.aspects,
        },
        "market_impact": content_sentiment.market_impact,
        "urgency": content_sentiment.urgency,
        "model_used": model_type,
    }

    # Add interpretation
    results["interpretation"] = _interpret_sentiment(results)

    return results


def _interpret_sentiment(sentiment_results: Dict[str, Any]) -> str:
    """Generate human-readable interpretation of sentiment results."""
    overall = sentiment_results["overall_sentiment"]
    score = sentiment_results["overall_score"]
    confidence = sentiment_results["confidence"]

    # Base interpretation
    if overall == SentimentLabel.VERY_POSITIVE.value:
        interpretation = "Very positive sentiment detected"
    elif overall == SentimentLabel.POSITIVE.value:
        interpretation = "Positive sentiment detected"
    elif overall == SentimentLabel.NEGATIVE.value:
        interpretation = "Negative sentiment detected"
    elif overall == SentimentLabel.VERY_NEGATIVE.value:
        interpretation = "Very negative sentiment detected"
    else:
        interpretation = "Neutral sentiment detected"

    # Add confidence qualifier
    if confidence < LOW_CONFIDENCE_INTERPRETATION_THRESHOLD:
        interpretation += " (low confidence)"
    elif confidence > HIGH_CONFIDENCE_INTERPRETATION_THRESHOLD:
        interpretation += " (high confidence)"

    # Add market impact for financial
    if sentiment_results.get("market_impact"):
        impact = sentiment_results["market_impact"]
        interpretation += f" with {impact} potential market impact"

    return interpretation


if __name__ == "__main__":
    # Test sentiment analysis
    logging.basicConfig(level=logging.INFO)

    # Test texts
    test_texts = [
        "The company reported record-breaking profits and exceeded all analyst expectations.",
        "The product launch was a complete disaster, with multiple critical bugs reported.",
        "The merger talks are progressing as planned, according to sources.",
        "Breaking: Major data breach affects millions of customers, stock plummets.",
        "The new AI technology shows promising results in early trials.",
    ]

    analyzer = SentimentAnalyzer()

    print("Sentiment Analysis Results:")
    print("-" * 50)

    for text in test_texts:
        result = analyzer.analyze(text, model_type="financial", include_aspects=True)
        print(f"\nText: {text[:80]}...")
        print(f"Sentiment: {result.label}")
        print(f"Score: {result.score:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        if result.market_impact:
            print(f"Market Impact: {result.market_impact}")
        if result.urgency:
            print(f"Urgency: {result.urgency}")
        print("-" * 50)
