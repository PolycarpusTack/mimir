"""
Advanced summarization engine for Mimir - EPIC 8.1
Provides multi-format summarization with BART/T5 models.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ai_optimization import ModelPool
import db_adapter

logger = logging.getLogger(__name__)


class SummarizationEngine:
    """Advanced summarization engine with multiple format support."""
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize the summarization engine.
        
        Args:
            model_pool: Optional ModelPool instance for model management
        """
        self.model_pool = model_pool or ModelPool()
        self.quality_metrics = SummaryQualityMetrics()
        
        # Summary format configurations
        self.format_configs = {
            'bullet_points': {
                'min_length': 50,
                'max_length': 150,
                'num_beams': 4,
                'template': "Create bullet points for: {text}"
            },
            'executive': {
                'min_length': 100,
                'max_length': 300,
                'num_beams': 6,
                'template': "Write an executive summary for: {text}"
            },
            'tweet': {
                'min_length': 20,
                'max_length': 280,
                'num_beams': 4,
                'template': "Summarize in one tweet: {text}"
            },
            'custom': {
                'min_length': 50,
                'max_length': 200,
                'num_beams': 4,
                'template': "Summarize: {text}"
            }
        }
        
    def get_summarization_model(self, model_type: str = "facebook/bart-large-cnn"):
        """Get or load a summarization model with caching."""
        return self.model_pool.get_transformer_model("summarization", model_type)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ensure text length is reasonable for models
        max_input_length = 1024
        if len(text) > max_input_length:
            # Truncate at sentence boundary
            sentences = text.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence) < max_input_length:
                    truncated += sentence + ". "
                else:
                    break
            text = truncated.strip()
            
        return text
    
    def generate_summary(
        self,
        text: str,
        format_type: str = "custom",
        custom_length: Optional[int] = None,
        model_type: str = "facebook/bart-large-cnn"
    ) -> Dict[str, Any]:
        """
        Generate a summary in the specified format.
        
        Args:
            text: Input text to summarize
            format_type: Type of summary ('bullet_points', 'executive', 'tweet', 'custom')
            custom_length: Custom max length for summary
            model_type: Model to use for summarization
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Preprocess input
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return self._create_error_result("Empty or invalid input text")
            
            # Get format configuration
            config = self.format_configs.get(format_type, self.format_configs['custom'])
            if custom_length:
                config = config.copy()
                config['max_length'] = custom_length
            
            # Get model
            summarizer = self.get_summarization_model(model_type)
            if not summarizer:
                return self._create_error_result(f"Failed to load model: {model_type}")
            
            # Generate summary
            start_time = datetime.now()
            
            summary_result = summarizer(
                processed_text,
                min_length=config['min_length'],
                max_length=config['max_length'],
                num_beams=config['num_beams'],
                early_stopping=True,
                do_sample=False,
                truncation=True
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            summary_text = summary_result[0]['summary_text']
            
            # Apply format-specific post-processing
            formatted_summary = self._format_summary(summary_text, format_type)
            
            # Calculate quality metrics
            quality_score = self.quality_metrics.calculate_quality_score(
                original_text=processed_text,
                summary=formatted_summary
            )
            
            return {
                'summary': formatted_summary,
                'format_type': format_type,
                'model_used': model_type,
                'processing_time_seconds': processing_time,
                'quality_score': quality_score,
                'input_length': len(processed_text),
                'output_length': len(formatted_summary),
                'compression_ratio': len(formatted_summary) / len(processed_text) if processed_text else 0,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._create_error_result(str(e))
    
    def _format_summary(self, summary: str, format_type: str) -> str:
        """Apply format-specific post-processing."""
        if format_type == 'bullet_points':
            # Convert to bullet points if not already formatted
            if not summary.startswith('•') and not summary.startswith('-'):
                sentences = summary.split('. ')
                if len(sentences) > 1:
                    bullet_points = []
                    for sentence in sentences:
                        if sentence.strip():
                            bullet_points.append(f"• {sentence.strip()}")
                    return '\n'.join(bullet_points)
            return summary
            
        elif format_type == 'executive':
            # Ensure professional executive summary format
            if not summary.lower().startswith(('executive summary:', 'summary:', 'overview:')):
                summary = f"Executive Summary: {summary}"
            return summary
            
        elif format_type == 'tweet':
            # Ensure tweet format with hashtags
            if len(summary) > 240:  # Leave room for hashtags
                summary = summary[:240].rsplit(' ', 1)[0] + "..."
            
            # Add relevant hashtags based on content
            hashtags = self._generate_hashtags(summary)
            if hashtags:
                summary = f"{summary} {hashtags}"
            
            return summary
            
        return summary
    
    def _generate_hashtags(self, text: str) -> str:
        """Generate relevant hashtags for tweet format."""
        # Simple hashtag generation based on keywords
        common_business_terms = {
            'technology': '#Tech',
            'business': '#Business',
            'finance': '#Finance',
            'market': '#Markets',
            'innovation': '#Innovation',
            'startup': '#Startup',
            'investment': '#Investment'
        }
        
        hashtags = []
        text_lower = text.lower()
        
        for term, hashtag in common_business_terms.items():
            if term in text_lower and hashtag not in hashtags:
                hashtags.append(hashtag)
                
        return ' '.join(hashtags[:3])  # Limit to 3 hashtags
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'summary': '',
            'format_type': 'error',
            'model_used': '',
            'processing_time_seconds': 0,
            'quality_score': 0,
            'input_length': 0,
            'output_length': 0,
            'compression_ratio': 0,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error_message
        }
    
    def batch_summarize(
        self,
        articles: List[Dict[str, Any]],
        format_type: str = "custom",
        custom_length: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch summarize multiple articles efficiently.
        
        Args:
            articles: List of article dictionaries with 'content' and 'id' keys
            format_type: Summary format type
            custom_length: Optional custom length
            
        Returns:
            List of summarization results
        """
        results = []
        
        logger.info(f"Starting batch summarization of {len(articles)} articles")
        
        for i, article in enumerate(articles):
            try:
                content = article.get('content', '')
                article_id = article.get('id', f'article_{i}')
                
                result = self.generate_summary(
                    text=content,
                    format_type=format_type,
                    custom_length=custom_length
                )
                
                result['article_id'] = article_id
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(articles)} articles")
                    
            except Exception as e:
                logger.error(f"Failed to summarize article {article.get('id', i)}: {e}")
                results.append({
                    'article_id': article.get('id', f'article_{i}'),
                    **self._create_error_result(str(e))
                })
        
        logger.info(f"Completed batch summarization: {len(results)} results")
        return results
    
    def save_summary_to_db(self, article_id: str, summary_result: Dict[str, Any]) -> bool:
        """Save summary result to database."""
        try:
            # Get current article metadata
            article = db_adapter.get_article_by_id(article_id)
            if not article:
                logger.error(f"Article not found: {article_id}")
                return False
            
            # Update metadata with summary information
            metadata = json.loads(article.get('metadata', '{}'))
            
            if 'summaries' not in metadata:
                metadata['summaries'] = {}
            
            metadata['summaries'][summary_result['format_type']] = {
                'summary': summary_result['summary'],
                'quality_score': summary_result['quality_score'],
                'compression_ratio': summary_result['compression_ratio'],
                'model_used': summary_result['model_used'],
                'timestamp': summary_result['timestamp']
            }
            
            # Update article metadata
            success = db_adapter.update_article_metadata(article_id, metadata)
            
            if success:
                logger.info(f"Saved summary for article {article_id}")
            else:
                logger.error(f"Failed to save summary for article {article_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving summary to database: {e}")
            return False


class SummaryQualityMetrics:
    """Calculate quality metrics for generated summaries."""
    
    def calculate_quality_score(self, original_text: str, summary: str) -> float:
        """
        Calculate a quality score for the summary.
        
        Args:
            original_text: Original article text
            summary: Generated summary
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            scores = []
            
            # Coverage score (important terms preserved)
            coverage_score = self._calculate_coverage(original_text, summary)
            scores.append(coverage_score * 0.4)
            
            # Compression score (appropriate length reduction)
            compression_score = self._calculate_compression_score(original_text, summary)
            scores.append(compression_score * 0.3)
            
            # Coherence score (grammatical and logical flow)
            coherence_score = self._calculate_coherence_score(summary)
            scores.append(coherence_score * 0.3)
            
            final_score = sum(scores)
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5  # Default score on error
    
    def _calculate_coverage(self, original: str, summary: str) -> float:
        """Calculate how well the summary covers important terms."""
        if not original or not summary:
            return 0.0
            
        # Extract important terms (simple approach)
        original_words = set(original.lower().split())
        summary_words = set(summary.lower().split())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        important_original = original_words - stop_words
        important_summary = summary_words - stop_words
        
        if not important_original:
            return 0.0
            
        coverage = len(important_summary & important_original) / len(important_original)
        return min(1.0, coverage * 2)  # Scale appropriately
    
    def _calculate_compression_score(self, original: str, summary: str) -> float:
        """Calculate compression quality (not too short, not too long)."""
        if not original or not summary:
            return 0.0
            
        compression_ratio = len(summary) / len(original)
        
        # Ideal compression ratio is between 0.1 and 0.3
        if 0.1 <= compression_ratio <= 0.3:
            return 1.0
        elif compression_ratio < 0.05:
            return 0.2  # Too compressed
        elif compression_ratio > 0.5:
            return 0.4  # Not compressed enough
        else:
            # Linear scaling for ratios outside ideal range
            if compression_ratio < 0.1:
                return compression_ratio / 0.1 * 0.8 + 0.2
            else:  # compression_ratio > 0.3
                return max(0.4, 1.0 - (compression_ratio - 0.3) / 0.2 * 0.6)
    
    def _calculate_coherence_score(self, summary: str) -> float:
        """Calculate coherence quality (simple heuristics)."""
        if not summary:
            return 0.0
            
        score = 1.0
        
        # Check for complete sentences
        sentences = summary.split('. ')
        if len(sentences) == 1 and not summary.endswith('.'):
            score -= 0.2
            
        # Check for proper capitalization
        if not summary[0].isupper():
            score -= 0.1
            
        # Check for excessive repetition
        words = summary.lower().split()
        if len(set(words)) < len(words) * 0.7:  # Less than 70% unique words
            score -= 0.2
            
        # Check for minimum length
        if len(summary) < 20:
            score -= 0.3
            
        return max(0.0, score)


# Extend ModelPool to support summarization models
def _extend_model_pool():
    """Extend ModelPool class to include summarization models."""
    original_get_transformer = ModelPool.get_transformer_model
    
    def get_transformer_model_extended(self, task: str, model_name: Optional[str] = None):
        """Extended transformer model getter with summarization support."""
        if task == "summarization":
            key = f"transformer_{task}_{model_name or 'default'}"
            
            with self.model_locks[key]:
                if key not in self.models:
                    logger.info(f"Loading summarization model: {model_name}")
                    try:
                        model = model_name or "facebook/bart-large-cnn"
                        summarizer = pipeline(
                            "summarization",
                            model=model,
                            tokenizer=model,
                            device=0 if torch.cuda.is_available() else -1,
                            framework="pt"
                        )
                        self.models[key] = summarizer
                    except Exception as e:
                        logger.error(f"Failed to load summarization model {model_name}: {e}")
                        return None
                
                self.usage_counts[key] += 1
                self.last_used[key] = time.time()
                return self.models[key]
        else:
            return original_get_transformer(self, task, model_name)
    
    # Monkey patch the method
    ModelPool.get_transformer_model = get_transformer_model_extended


# Initialize the extension
_extend_model_pool()