"""
Zero-shot industry classification module for Mimir.
Classifies articles into industries without requiring training data.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "facebook/bart-large-mnli"
FALLBACK_MODEL = "typeform/distilbert-base-uncased-mnli"
MAX_TEXT_LENGTH = 1024
DEFAULT_BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.1
SECONDARY_INDUSTRIES_LIMIT = 3
KEYWORD_BOOST_FACTOR = 0.1
MAX_KEYWORD_BOOST = 0.5
MAX_BOOST_MULTIPLIER = 1.0
DOMINANT_INDUSTRY_THRESHOLD = 10.0  # 10% threshold
MAX_KEYWORDS_TO_SHOW = 5
DEFAULT_HYPOTHESIS_TEMPLATE = "This text is about the {} industry."
MAX_SECONDARY_TO_SHOW = 2
CUDA_DEVICE_ID = 0
CPU_DEVICE_ID = -1


class Industry(Enum):
    """Standard industry categories."""

    TECHNOLOGY = "Technology"
    FINANCE = "Finance & Banking"
    HEALTHCARE = "Healthcare & Pharmaceuticals"
    RETAIL = "Retail & E-commerce"
    ENERGY = "Energy & Utilities"
    MANUFACTURING = "Manufacturing & Industrial"
    REAL_ESTATE = "Real Estate & Construction"
    TRANSPORTATION = "Transportation & Logistics"
    TELECOMMUNICATIONS = "Telecommunications"
    ENTERTAINMENT = "Media & Entertainment"
    EDUCATION = "Education"
    GOVERNMENT = "Government & Public Sector"
    AGRICULTURE = "Agriculture & Food"
    AUTOMOTIVE = "Automotive"
    AEROSPACE = "Aerospace & Defense"
    HOSPITALITY = "Hospitality & Tourism"
    LEGAL = "Legal Services"
    CONSULTING = "Consulting & Professional Services"
    NONPROFIT = "Non-Profit & NGO"
    OTHER = "Other"


@dataclass
class ClassificationResult:
    """Result of industry classification."""

    primary_industry: str
    confidence: float
    all_scores: Dict[str, float]
    secondary_industries: List[Tuple[str, float]]
    keywords_matched: List[str]


class IndustryClassifier:
    """Zero-shot industry classification using transformers.

    TODO: Move industry keywords to external configuration file
    to allow updates without code changes.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the industry classifier.

        Args:
            model_name: Pre-trained model for zero-shot classification
        """
        self.model_name = model_name
        self.classifier = None
        self.industries = [industry.value for industry in Industry]
        self._load_model()

        # Industry-specific keywords for validation
        # In production, load from config file
        self.industry_keywords = self._load_industry_keywords()

    def _load_model(self):
        """Load the zero-shot classification model."""
        try:
            logger.info(f"Loading zero-shot classifier: {self.model_name}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=CUDA_DEVICE_ID if torch.cuda.is_available() else CPU_DEVICE_ID,
            )
            logger.info("Zero-shot classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            # Fallback to a simpler model
            try:
                self.classifier = pipeline("zero-shot-classification", model=FALLBACK_MODEL, device=CPU_DEVICE_ID)
                logger.info("Loaded fallback classifier")
            except Exception as e2:
                logger.error(f"Failed to load fallback classifier: {e2}")
                self.classifier = None

    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords for validation."""
        return {
            Industry.TECHNOLOGY.value: [
                "software",
                "hardware",
                "ai",
                "artificial intelligence",
                "machine learning",
                "cloud",
                "saas",
                "api",
                "blockchain",
                "cryptocurrency",
                "tech",
                "digital",
                "cyber",
                "data",
                "algorithm",
                "platform",
                "app",
                "startup",
                "innovation",
            ],
            Industry.FINANCE.value: [
                "bank",
                "finance",
                "investment",
                "stock",
                "trading",
                "market",
                "fund",
                "capital",
                "equity",
                "debt",
                "loan",
                "credit",
                "fintech",
                "payment",
                "insurance",
                "portfolio",
                "asset",
                "revenue",
                "profit",
                "earnings",
            ],
            Industry.HEALTHCARE.value: [
                "health",
                "medical",
                "hospital",
                "clinic",
                "patient",
                "doctor",
                "pharma",
                "drug",
                "treatment",
                "therapy",
                "disease",
                "diagnosis",
                "healthcare",
                "biotech",
                "vaccine",
                "clinical trial",
                "fda",
                "medicine",
                "wellness",
            ],
            Industry.RETAIL.value: [
                "retail",
                "store",
                "shop",
                "ecommerce",
                "e-commerce",
                "consumer",
                "product",
                "brand",
                "merchandise",
                "sale",
                "customer",
                "shopping",
                "marketplace",
                "inventory",
                "supply chain",
                "omnichannel",
                "fashion",
                "apparel",
            ],
            Industry.ENERGY.value: [
                "energy",
                "oil",
                "gas",
                "renewable",
                "solar",
                "wind",
                "power",
                "electricity",
                "utility",
                "grid",
                "battery",
                "nuclear",
                "coal",
                "petroleum",
                "drilling",
                "sustainable",
                "carbon",
                "emission",
                "climate",
                "green energy",
            ],
            Industry.MANUFACTURING.value: [
                "manufacturing",
                "factory",
                "production",
                "industrial",
                "assembly",
                "machinery",
                "equipment",
                "automation",
                "robotics",
                "supply chain",
                "quality control",
                "logistics",
                "warehouse",
                "distribution",
                "lean",
            ],
            Industry.REAL_ESTATE.value: [
                "real estate",
                "property",
                "housing",
                "construction",
                "building",
                "development",
                "commercial",
                "residential",
                "mortgage",
                "rent",
                "lease",
                "architect",
                "contractor",
                "zoning",
                "investment property",
            ],
            Industry.TRANSPORTATION.value: [
                "transport",
                "logistics",
                "shipping",
                "freight",
                "delivery",
                "airline",
                "aviation",
                "rail",
                "trucking",
                "fleet",
                "vehicle",
                "cargo",
                "port",
                "infrastructure",
                "mobility",
                "autonomous",
                "electric vehicle",
            ],
            Industry.TELECOMMUNICATIONS.value: [
                "telecom",
                "telecommunications",
                "network",
                "5g",
                "4g",
                "broadband",
                "internet",
                "isp",
                "wireless",
                "fiber",
                "satellite",
                "mobile",
                "carrier",
                "bandwidth",
                "connectivity",
                "spectrum",
                "tower",
            ],
            Industry.ENTERTAINMENT.value: [
                "media",
                "entertainment",
                "film",
                "movie",
                "television",
                "streaming",
                "content",
                "production",
                "studio",
                "music",
                "gaming",
                "sports",
                "broadcast",
                "publishing",
                "advertising",
                "social media",
                "digital media",
            ],
            Industry.EDUCATION.value: [
                "education",
                "school",
                "university",
                "college",
                "student",
                "teacher",
                "learning",
                "training",
                "curriculum",
                "degree",
                "online learning",
                "edtech",
                "course",
                "academic",
                "research",
                "scholarship",
                "e-learning",
            ],
            Industry.AUTOMOTIVE.value: [
                "automotive",
                "car",
                "vehicle",
                "automobile",
                "electric vehicle",
                "ev",
                "autonomous",
                "self-driving",
                "manufacturer",
                "dealership",
                "parts",
                "engine",
                "tesla",
                "ford",
                "gm",
                "toyota",
                "volkswagen",
            ],
        }

    def _load_keywords_from_config(self, config_path: Optional[str] = None) -> Dict[str, List[str]]:
        """Load industry keywords from configuration file.

        Args:
            config_path: Path to keywords configuration file

        Returns:
            Dictionary of industry keywords
        """
        if not config_path:
            # Fall back to hardcoded keywords
            return self._load_industry_keywords()

        try:
            import json

            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load keywords from {config_path}: {e}")
            return self._load_industry_keywords()

    def classify(
        self, text: str, use_keywords: bool = True, threshold: float = CONFIDENCE_THRESHOLD
    ) -> ClassificationResult:
        """
        Classify text into industry categories.

        Args:
            text: Text to classify
            use_keywords: Whether to use keyword matching for validation
            threshold: Minimum confidence threshold

        Returns:
            ClassificationResult with industry predictions
        """
        if not text or not self.classifier:
            return ClassificationResult(
                primary_industry=Industry.OTHER.value,
                confidence=0.0,
                all_scores={},
                secondary_industries=[],
                keywords_matched=[],
            )

        try:
            # Truncate text if too long
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]

            # Perform zero-shot classification
            result = self.classifier(
                text,
                candidate_labels=self.industries,
                multi_label=True,
                hypothesis_template=DEFAULT_HYPOTHESIS_TEMPLATE,
            )

            # Extract scores
            scores = dict(zip(result["labels"], result["scores"]))

            # Apply keyword validation if enabled
            keywords_matched = []
            if use_keywords:
                keyword_scores = self._calculate_keyword_scores(text)
                keywords_matched = keyword_scores["matched_keywords"]

                # Boost scores based on keyword matches
                for industry, boost in keyword_scores["industry_boosts"].items():
                    if industry in scores:
                        scores[industry] = min(MAX_BOOST_MULTIPLIER, scores[industry] * (1 + boost))

            # Sort industries by score
            sorted_industries = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Get primary industry
            primary_industry = sorted_industries[0][0]
            confidence = sorted_industries[0][1]

            # Get secondary industries above threshold
            secondary_industries = [
                (industry, score) for industry, score in sorted_industries[1:] if score >= threshold
            ][:SECONDARY_INDUSTRIES_LIMIT]

            return ClassificationResult(
                primary_industry=primary_industry,
                confidence=confidence,
                all_scores=scores,
                secondary_industries=secondary_industries,
                keywords_matched=keywords_matched,
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                primary_industry=Industry.OTHER.value,
                confidence=0.0,
                all_scores={},
                secondary_industries=[],
                keywords_matched=[],
            )

    def _calculate_keyword_scores(self, text: str) -> Dict[str, Any]:
        """Calculate keyword-based scores for industries."""
        text_lower = text.lower()
        industry_boosts = {}
        all_matched_keywords = []

        for industry, keywords in self.industry_keywords.items():
            matches = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
                    matched_keywords.append(keyword)

            if matches > 0:
                # Calculate boost factor based on matches
                boost = min(MAX_KEYWORD_BOOST, matches * KEYWORD_BOOST_FACTOR)
                industry_boosts[industry] = boost
                all_matched_keywords.extend(matched_keywords)

        return {"industry_boosts": industry_boosts, "matched_keywords": list(set(all_matched_keywords))}

    def classify_batch(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[ClassificationResult]:
        """
        Classify multiple texts in batches.

        Args:
            texts: List of texts to classify
            batch_size: Batch size for processing

        Returns:
            List of classification results
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                result = self.classify(text)
                results.append(result)

        return results

    def get_industry_distribution(self, classifications: List[ClassificationResult]) -> Dict[str, float]:
        """
        Calculate industry distribution from classifications.

        Args:
            classifications: List of classification results

        Returns:
            Dictionary of industry percentages
        """
        if not classifications:
            return {}

        industry_counts = {}
        total = len(classifications)

        for result in classifications:
            industry = result.primary_industry
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

        # Calculate percentages
        distribution = {industry: (count / total) * 100 for industry, count in industry_counts.items()}

        # Sort by percentage
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def explain_classification(self, result: ClassificationResult) -> str:
        """
        Generate human-readable explanation of classification.

        Args:
            result: Classification result

        Returns:
            Explanation string
        """
        explanation = f"Primary Industry: {result.primary_industry} "
        explanation += f"(confidence: {result.confidence:.1%})"

        if result.secondary_industries:
            secondary = ", ".join(
                [f"{ind} ({score:.1%})" for ind, score in result.secondary_industries[:MAX_SECONDARY_TO_SHOW]]
            )
            explanation += f"\nSecondary Industries: {secondary}"

        if result.keywords_matched:
            keywords = ", ".join(result.keywords_matched[:MAX_KEYWORDS_TO_SHOW])
            explanation += f"\nKey indicators: {keywords}"

        return explanation


def classify_article_industry(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify a news article into industry categories.

    Args:
        article: Article dictionary with 'title' and 'content'

    Returns:
        Industry classification results
    """
    classifier = IndustryClassifier()

    # Combine title and content
    title = article.get("title", "")
    content = article.get("content", "")
    full_text = f"{title}\n\n{content}"

    # Classify
    result = classifier.classify(full_text)

    # Prepare response
    classification = {
        "primary_industry": result.primary_industry,
        "confidence": result.confidence,
        "secondary_industries": [{"industry": ind, "confidence": score} for ind, score in result.secondary_industries],
        "keywords_matched": result.keywords_matched,
        "explanation": classifier.explain_classification(result),
    }

    return classification


def analyze_industry_trends(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze industry trends across multiple articles.

    Args:
        articles: List of articles to analyze

    Returns:
        Industry trend analysis
    """
    if not articles:
        return {}

    classifier = IndustryClassifier()

    # Classify all articles
    texts = []
    for article in articles:
        title = article.get("title", "")
        content = article.get("content", "")
        texts.append(f"{title}\n\n{content}")

    classifications = classifier.classify_batch(texts)

    # Get distribution
    distribution = classifier.get_industry_distribution(classifications)

    # Analyze by time if dates available
    time_based_analysis = {}
    articles_with_dates = [a for a in articles if a.get("published_date")]

    if articles_with_dates:
        # Group by month
        from collections import defaultdict
        from datetime import datetime

        monthly_industries = defaultdict(lambda: defaultdict(int))

        for article, classification in zip(articles, classifications):
            if "published_date" in article:
                try:
                    date = article["published_date"]
                    if isinstance(date, str):
                        from dateutil import parser

                        date = parser.parse(date)

                    month_key = date.strftime("%Y-%m")
                    industry = classification.primary_industry
                    monthly_industries[month_key][industry] += 1
                except:
                    continue

        time_based_analysis = dict(monthly_industries)

    # Find dominant industries
    dominant_industries = []
    for industry, percentage in distribution.items():
        if percentage >= DOMINANT_INDUSTRY_THRESHOLD:
            dominant_industries.append(
                {"industry": industry, "percentage": percentage, "article_count": int(len(articles) * percentage / 100)}
            )

    return {
        "total_articles": len(articles),
        "industry_distribution": distribution,
        "dominant_industries": dominant_industries,
        "time_based_analysis": time_based_analysis,
        "unique_industries": len(distribution),
    }


if __name__ == "__main__":
    # Test the industry classifier
    logging.basicConfig(level=logging.INFO)

    test_articles = [
        {
            "title": "Apple Unveils New AI-Powered iPhone Features",
            "content": "Tech giant Apple announced revolutionary artificial intelligence features "
            "for its latest iPhone models. The new software uses machine learning "
            "to enhance photo quality and improve battery life through intelligent "
            "power management. CEO Tim Cook called it a breakthrough in mobile technology.",
        },
        {
            "title": "Major Bank Announces Blockchain Integration",
            "content": "JPMorgan Chase revealed plans to integrate blockchain technology "
            "into its payment systems. The financial institution aims to reduce "
            "transaction costs and improve security. This fintech innovation could "
            "revolutionize how banks handle international transfers.",
        },
        {
            "title": "New Cancer Treatment Shows Promise in Clinical Trials",
            "content": "A pharmaceutical company reported positive results from Phase 3 "
            "clinical trials of its new cancer treatment. The drug showed "
            "significant improvement in patient outcomes. The FDA is expected "
            "to review the application for approval next quarter.",
        },
        {
            "title": "Amazon Expands Same-Day Delivery Service",
            "content": "E-commerce giant Amazon announced expansion of its same-day delivery "
            "service to 50 new cities. The retail company is investing heavily "
            "in logistics infrastructure to compete with traditional retailers. "
            "Customer satisfaction with fast delivery continues to drive growth.",
        },
    ]

    print("Industry Classification Results:")
    print("-" * 50)

    for article in test_articles:
        result = classify_article_industry(article)
        print(f"\nTitle: {article['title']}")
        print(result["explanation"])
        print("-" * 50)

    # Analyze trends
    print("\nIndustry Trend Analysis:")
    trends = analyze_industry_trends(test_articles)
    print(f"Total articles: {trends['total_articles']}")
    print(f"Unique industries: {trends['unique_industries']}")
    print("\nIndustry Distribution:")
    for industry, percentage in trends["industry_distribution"].items():
        print(f"  - {industry}: {percentage:.1f}%")
