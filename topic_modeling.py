"""
Topic modeling module for Mimir.
Implements LDA (Latent Dirichlet Allocation) for dynamic topic discovery.
"""

import logging
import pickle  # TODO: Replace with joblib for model serialization
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_TOPICS = 10
DEFAULT_LANGUAGE = "en"
MAX_DF = 0.8  # Ignore terms in >80% of documents
MIN_DF = 5  # Ignore terms in <5 documents
MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)  # Include bigrams
MAX_ITER = 100
LEARNING_OFFSET = 50.0
RANDOM_STATE = 42
MIN_DOCUMENT_LENGTH = 20
OPTIMIZE_MIN_DOCUMENTS = 50
DEFAULT_MIN_TOPICS = 5
DEFAULT_MAX_TOPICS = 20
TOPIC_PROBABILITY_THRESHOLD = 0.1
TOPIC_EVOLUTION_THRESHOLD = 0.05  # 5% change threshold
DEFAULT_TOP_WORDS = 10
VISUALIZATION_TOP_WORDS = 15
LDA_PASSES = 10
DOCUMENT_DIVISOR = 10  # For max topics calculation
TOP_WORDS_FOR_LABEL = 3


class TopicModeler:
    """Advanced topic modeling for news articles."""

    def __init__(self, num_topics: int = DEFAULT_NUM_TOPICS, language: str = DEFAULT_LANGUAGE):
        """
        Initialize topic modeler.

        Args:
            num_topics: Default number of topics to discover
            language: Language for stopwords and preprocessing
        """
        self.num_topics = num_topics
        self.language = language
        self.lda_model = None
        self.vectorizer = None
        self.dictionary = None
        self.corpus = None
        self.topic_labels = {}

        # Language-specific stopwords (extend from preprocessing module)
        self.stopwords = self._get_stopwords(language)

    def _get_stopwords(self, language: str) -> List[str]:
        """Get stopwords for the specified language."""
        # Import from preprocessing module
        from nlp_preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()
        return list(preprocessor.stopwords.get(language, preprocessor.stopwords["en"]))

    def train_lda_model(
        self, documents: List[str], num_topics: Optional[int] = None, optimize_topics: bool = True
    ) -> Dict[str, Any]:
        """
        Train LDA model on documents.

        Args:
            documents: List of document texts
            num_topics: Number of topics (uses default if None)
            optimize_topics: Whether to optimize number of topics

        Returns:
            Dictionary with model info and topics
        """
        if not documents:
            logger.error("No documents provided for topic modeling")
            return {}

        num_topics = num_topics or self.num_topics

        try:
            # Preprocess documents
            processed_docs = self._preprocess_documents(documents)

            # Find optimal number of topics if requested
            if optimize_topics and len(documents) > OPTIMIZE_MIN_DOCUMENTS:
                optimal_topics = self._find_optimal_topics(processed_docs)
                logger.info(f"Optimal number of topics: {optimal_topics}")
                num_topics = optimal_topics

            # Create document-term matrix using sklearn
            self.vectorizer = CountVectorizer(
                max_df=MAX_DF,
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
                stop_words=self.stopwords,
                ngram_range=NGRAM_RANGE,
            )

            doc_term_matrix = self.vectorizer.fit_transform(processed_docs)

            # Train LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=num_topics,
                max_iter=MAX_ITER,
                learning_method="online",
                learning_offset=LEARNING_OFFSET,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            self.lda_model.fit(doc_term_matrix)

            # Extract topics
            topics = self._extract_topics()

            # Generate topic labels
            self.topic_labels = self._generate_topic_labels(topics)

            # Calculate topic coherence
            coherence_score = self._calculate_coherence(processed_docs)

            # Get document-topic distribution
            doc_topics = self.lda_model.transform(doc_term_matrix)

            result = {
                "num_topics": num_topics,
                "topics": topics,
                "topic_labels": self.topic_labels,
                "coherence_score": coherence_score,
                "document_topics": doc_topics.tolist(),
                "perplexity": self.lda_model.perplexity(doc_term_matrix),
                "model_params": {"max_features": MAX_FEATURES, "min_df": MIN_DF, "max_df": MAX_DF},
            }

            return result

        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return {}

    def _preprocess_documents(self, documents: List[str]) -> List[str]:
        """Preprocess documents for topic modeling."""
        from nlp_preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()

        processed = []
        for doc in documents:
            # Basic preprocessing
            text = preprocessor.normalize_text(doc, preserve_case=False)

            # Remove short documents
            if len(text.split()) > MIN_DOCUMENT_LENGTH:
                processed.append(text)

        return processed

    def _find_optimal_topics(
        self, documents: List[str], min_topics: int = DEFAULT_MIN_TOPICS, max_topics: int = DEFAULT_MAX_TOPICS
    ) -> int:
        """
        Find optimal number of topics using coherence scores.

        Args:
            documents: Preprocessed documents
            min_topics: Minimum number of topics to try
            max_topics: Maximum number of topics to try

        Returns:
            Optimal number of topics
        """
        logger.info(f"Finding optimal topics between {min_topics} and {max_topics}")

        # Use gensim for coherence calculation
        texts = [doc.split() for doc in documents]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        coherence_scores = []

        for n_topics in range(min_topics, min(max_topics + 1, len(documents) // DOCUMENT_DIVISOR)):
            try:
                # Train model
                lda = models.LdaModel(
                    corpus=corpus,
                    id2word=dictionary,
                    num_topics=n_topics,
                    random_state=RANDOM_STATE,
                    passes=LDA_PASSES,
                    alpha="auto",
                    per_word_topics=True,
                )

                # Calculate coherence
                coherence_model = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence="c_v")

                coherence = coherence_model.get_coherence()
                coherence_scores.append((n_topics, coherence))

                logger.info(f"Topics: {n_topics}, Coherence: {coherence:.4f}")

            except Exception as e:
                logger.warning(f"Failed to calculate coherence for {n_topics} topics: {e}")

        if coherence_scores:
            # Find elbow point or maximum coherence
            best_topics = max(coherence_scores, key=lambda x: x[1])[0]
            return best_topics

        return self.num_topics

    def _extract_topics(self, top_words: int = DEFAULT_TOP_WORDS) -> List[List[Tuple[str, float]]]:
        """Extract topics as word-probability pairs."""
        if not self.lda_model or not self.vectorizer:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic_dist in enumerate(self.lda_model.components_):
            # Get top word indices
            top_indices = topic_dist.argsort()[-top_words:][::-1]

            # Get words and their probabilities
            topic_words = []
            for idx in top_indices:
                word = feature_names[idx]
                prob = topic_dist[idx] / topic_dist.sum()
                topic_words.append((word, float(prob)))

            topics.append(topic_words)

        return topics

    def _generate_topic_labels(self, topics: List[List[Tuple[str, float]]]) -> Dict[int, str]:
        """Generate human-readable labels for topics."""
        labels = {}

        for idx, topic_words in enumerate(topics):
            # Use top words for label
            top_words = [word for word, _ in topic_words[:TOP_WORDS_FOR_LABEL]]
            label = " / ".join(top_words)
            labels[idx] = label

        return labels

    def _calculate_coherence(self, documents: List[str]) -> float:
        """Calculate topic coherence score."""
        try:
            # Convert to gensim format for coherence calculation
            texts = [doc.split() for doc in documents]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            # Convert sklearn model to gensim format
            topics = self._extract_topics()
            topic_words = [[word for word, _ in topic] for topic in topics]

            # Calculate coherence
            coherence_model = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence="c_v")

            return coherence_model.get_coherence()

        except Exception as e:
            logger.warning(f"Could not calculate coherence: {e}")
            return 0.0

    def predict_topics(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Predict topics for new documents.

        Args:
            documents: List of documents to analyze

        Returns:
            List of topic predictions for each document
        """
        if not self.lda_model or not self.vectorizer:
            logger.error("No trained model available")
            return []

        try:
            # Preprocess documents
            processed_docs = self._preprocess_documents(documents)

            # Transform to document-term matrix
            doc_term_matrix = self.vectorizer.transform(processed_docs)

            # Get topic distributions
            doc_topics = self.lda_model.transform(doc_term_matrix)

            results = []
            for idx, topic_dist in enumerate(doc_topics):
                # Get dominant topic
                dominant_topic_idx = topic_dist.argmax()
                dominant_topic_prob = topic_dist[dominant_topic_idx]

                # Get all topics above threshold
                threshold = 0.1
                relevant_topics = [
                    {"topic_id": i, "label": self.topic_labels.get(i, f"Topic {i}"), "probability": float(prob)}
                    for i, prob in enumerate(topic_dist)
                    if prob > threshold
                ]

                # Sort by probability
                relevant_topics.sort(key=lambda x: x["probability"], reverse=True)

                results.append(
                    {
                        "document_id": idx,
                        "dominant_topic": {
                            "id": int(dominant_topic_idx),
                            "label": self.topic_labels.get(dominant_topic_idx, f"Topic {dominant_topic_idx}"),
                            "probability": float(dominant_topic_prob),
                        },
                        "all_topics": relevant_topics,
                        "topic_diversity": self._calculate_topic_diversity(topic_dist),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Topic prediction failed: {e}")
            return []

    def _calculate_topic_diversity(self, topic_dist: np.ndarray) -> float:
        """Calculate topic diversity (entropy) for a document."""
        # Remove zero probabilities
        probs = topic_dist[topic_dist > 0]
        if len(probs) <= 1:
            return 0.0

        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))

        # Normalize by max entropy
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return float(normalized_entropy)

    def get_topic_trends(self, documents: List[Dict[str, Any]], time_window: str = "week") -> Dict[str, Any]:
        """
        Analyze topic trends over time.

        Args:
            documents: List of documents with 'content' and 'published_date'
            time_window: Time window for aggregation ('day', 'week', 'month')

        Returns:
            Topic trends analysis
        """
        if not documents:
            return {}

        try:
            # Group documents by time window
            time_groups = self._group_by_time(documents, time_window)

            trends = {}

            for time_period, period_docs in time_groups.items():
                # Extract content
                contents = [doc.get("content", "") for doc in period_docs]

                # Predict topics
                predictions = self.predict_topics(contents)

                # Aggregate topic distributions
                topic_counts = defaultdict(float)
                for pred in predictions:
                    for topic in pred["all_topics"]:
                        topic_counts[topic["label"]] += topic["probability"]

                # Normalize
                total = sum(topic_counts.values())
                if total > 0:
                    topic_dist = {topic: count / total for topic, count in topic_counts.items()}
                else:
                    topic_dist = {}

                trends[time_period] = {
                    "document_count": len(period_docs),
                    "topic_distribution": topic_dist,
                    "dominant_topic": max(topic_dist.items(), key=lambda x: x[1])[0] if topic_dist else None,
                }

            return {
                "time_window": time_window,
                "periods": trends,
                "topic_evolution": self._analyze_topic_evolution(trends),
            }

        except Exception as e:
            logger.error(f"Topic trend analysis failed: {e}")
            return {}

    def _group_by_time(self, documents: List[Dict[str, Any]], window: str) -> Dict[str, List[Dict]]:
        """Group documents by time window."""
        groups = defaultdict(list)

        for doc in documents:
            pub_date = doc.get("published_date")
            if not pub_date:
                continue

            # Parse date
            if isinstance(pub_date, str):
                try:
                    from dateutil import parser

                    pub_date = parser.parse(pub_date)
                except:
                    continue

            # Determine time period
            if window == "day":
                period = pub_date.strftime("%Y-%m-%d")
            elif window == "week":
                period = pub_date.strftime("%Y-W%U")
            elif window == "month":
                period = pub_date.strftime("%Y-%m")
            else:
                period = pub_date.strftime("%Y-%m-%d")

            groups[period].append(doc)

        return dict(groups)

    def _analyze_topic_evolution(self, trends: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze how topics evolve over time."""
        evolution = {"emerging_topics": [], "declining_topics": [], "stable_topics": []}

        if len(trends) < 2:
            return evolution

        # Get sorted time periods
        periods = sorted(trends.keys())

        # Compare first and last periods
        first_topics = trends[periods[0]].get("topic_distribution", {})
        last_topics = trends[periods[-1]].get("topic_distribution", {})

        # Find emerging topics (appear in last but not first)
        for topic, prob in last_topics.items():
            if topic not in first_topics and prob > 0.1:
                evolution["emerging_topics"].append({"topic": topic, "current_weight": prob})

        # Find declining topics
        for topic, prob in first_topics.items():
            if topic not in last_topics and prob > 0.1:
                evolution["declining_topics"].append({"topic": topic, "initial_weight": prob})

        # Find stable topics
        for topic in set(first_topics.keys()) & set(last_topics.keys()):
            change = abs(last_topics[topic] - first_topics[topic])
            if change < 0.05:  # Less than 5% change
                evolution["stable_topics"].append(
                    {"topic": topic, "average_weight": (first_topics[topic] + last_topics[topic]) / 2}
                )

        return evolution

    def visualize_topics(self, output_path: str = "topic_visualization.html") -> bool:
        """
        Create interactive visualization of topics.

        Args:
            output_path: Path to save HTML visualization

        Returns:
            Success status
        """
        if not self.lda_model:
            logger.error("No trained model to visualize")
            return False

        try:
            # Prepare data for pyLDAvis
            # Note: This requires the original corpus and dictionary
            # For now, we'll create a simple visualization

            # Get topic-term matrix
            topic_term_matrix = self.lda_model.components_

            # Create simple HTML visualization
            html_content = self._create_simple_visualization()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Topic visualization saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False

    def _create_simple_visualization(self) -> str:
        """Create a simple HTML visualization of topics."""
        topics = self._extract_topics(15)  # Get top 15 words per topic

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Topic Modeling Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .topic { margin-bottom: 30px; padding: 15px; background: #f5f5f5; border-radius: 5px; }
                .topic h3 { color: #333; margin-top: 0; }
                .words { display: flex; flex-wrap: wrap; gap: 10px; }
                .word { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; font-size: 14px; }
                .prob { font-size: 12px; opacity: 0.8; }
            </style>
        </head>
        <body>
            <h1>Topic Modeling Results</h1>
            <p>Number of topics: {num_topics}</p>
        """

        for idx, topic_words in enumerate(topics):
            label = self.topic_labels.get(idx, f"Topic {idx}")
            html += f'<div class="topic">'
            html += f"<h3>Topic {idx}: {label}</h3>"
            html += '<div class="words">'

            for word, prob in topic_words:
                html += f'<div class="word">{word} <span class="prob">({prob:.3f})</span></div>'

            html += "</div></div>"

        html += """
        </body>
        </html>
        """

        return html.format(num_topics=len(topics))

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.lda_model:
            logger.error("No model to save")
            return

        try:
            model_data = {
                "lda_model": self.lda_model,
                "vectorizer": self.vectorizer,
                "topic_labels": self.topic_labels,
                "num_topics": self.num_topics,
                "language": self.language,
            }

            # TODO: Replace pickle with joblib for security
            # WARNING: Only use with trusted model files
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""
        try:
            # TODO: Replace pickle with joblib for security
            # WARNING: Only load trusted model files
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.lda_model = model_data["lda_model"]
            self.vectorizer = model_data["vectorizer"]
            self.topic_labels = model_data["topic_labels"]
            self.num_topics = model_data["num_topics"]
            self.language = model_data.get("language", DEFAULT_LANGUAGE)

            logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")


def analyze_article_topics(
    articles: List[Dict[str, Any]], num_topics: int = 10, optimize: bool = True
) -> Dict[str, Any]:
    """
    Analyze topics across a collection of articles.

    Args:
        articles: List of articles with 'content' field
        num_topics: Number of topics to discover
        optimize: Whether to optimize number of topics

    Returns:
        Topic analysis results
    """
    if not articles:
        return {}

    # Extract content
    documents = []
    for article in articles:
        content = article.get("content", "")
        title = article.get("title", "")
        full_text = f"{title} {content}" if title else content
        if full_text.strip():
            documents.append(full_text)

    if not documents:
        logger.warning("No valid documents for topic modeling")
        return {}

    # Initialize and train model
    modeler = TopicModeler(num_topics=num_topics)
    results = modeler.train_lda_model(documents, optimize_topics=optimize)

    # Add document-level analysis
    if results:
        predictions = modeler.predict_topics(documents)
        results["document_topics"] = predictions

        # Add topic summary
        topic_summary = defaultdict(int)
        for pred in predictions:
            topic_summary[pred["dominant_topic"]["label"]] += 1

        results["topic_summary"] = dict(topic_summary)

    return results


if __name__ == "__main__":
    # Test topic modeling
    logging.basicConfig(level=logging.INFO)

    # Sample articles for testing
    test_articles = [
        {
            "title": "AI Breakthrough in Healthcare",
            "content": "Artificial intelligence is revolutionizing healthcare with new diagnostic tools. "
            "Machine learning algorithms can now detect diseases earlier than traditional methods. "
            "Deep learning models are being used to analyze medical images with high accuracy.",
        },
        {
            "title": "Stock Market Hits Record High",
            "content": "The stock market reached new heights today as investors showed confidence. "
            "Tech stocks led the rally with significant gains across the board. "
            "Financial analysts predict continued growth in the coming months.",
        },
        {
            "title": "Climate Change Summit Results",
            "content": "World leaders gathered to discuss climate change mitigation strategies. "
            "New agreements were reached on carbon emission reduction targets. "
            "Renewable energy investments will be increased significantly.",
        },
        {
            "title": "New Smartphone Technology",
            "content": "The latest smartphone features revolutionary camera technology. "
            "Battery life has been improved with new power management systems. "
            "5G connectivity enables faster data speeds than ever before.",
        },
        {
            "title": "Healthcare Innovation Conference",
            "content": "Medical professionals discussed the latest healthcare innovations. "
            "Telemedicine adoption has increased dramatically in recent years. "
            "AI-powered diagnostic tools are improving patient outcomes.",
        },
    ]

    # Analyze topics
    results = analyze_article_topics(test_articles, num_topics=3, optimize=False)

    print("Topic Modeling Results:")
    print(f"Number of topics: {results.get('num_topics', 0)}")
    print(f"Coherence score: {results.get('coherence_score', 0):.4f}")

    print("\nDiscovered Topics:")
    for idx, (topic_id, label) in enumerate(results.get("topic_labels", {}).items()):
        print(f"\nTopic {topic_id}: {label}")
        topic_words = results["topics"][topic_id][:5]  # Top 5 words
        for word, prob in topic_words:
            print(f"  - {word}: {prob:.3f}")

    print("\nTopic Summary:")
    for topic, count in results.get("topic_summary", {}).items():
        print(f"  - {topic}: {count} documents")
