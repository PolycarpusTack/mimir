"""
Neural translation pipeline for Mimir - EPIC 8.2
Provides multi-language translation with quality estimation.
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


class TranslationEngine:
    """Neural translation engine with multi-language support."""
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize the translation engine.
        
        Args:
            model_pool: Optional ModelPool instance for model management
        """
        self.model_pool = model_pool or ModelPool()
        self.quality_estimator = TranslationQualityEstimator()
        
        # Supported language pairs and their models
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic'
        }
        
        # Model configurations for different language pairs
        self.model_configs = {
            'multilingual': {
                'model': 'facebook/mbart-large-50-many-to-many-mmt',
                'description': 'Multilingual BART for many-to-many translation',
                'languages': list(self.supported_languages.keys())
            },
            'en-de': {
                'model': 'Helsinki-NLP/opus-mt-en-de',
                'description': 'English to German translation',
                'languages': ['en', 'de']
            },
            'en-es': {
                'model': 'Helsinki-NLP/opus-mt-en-es',
                'description': 'English to Spanish translation',
                'languages': ['en', 'es']
            },
            'en-fr': {
                'model': 'Helsinki-NLP/opus-mt-en-fr',
                'description': 'English to French translation', 
                'languages': ['en', 'fr']
            }
        }
        
    def get_translation_model(self, source_lang: str, target_lang: str) -> str:
        """Determine the best model for a language pair."""
        # Check for specific language pair models
        pair_key = f"{source_lang}-{target_lang}"
        if pair_key in self.model_configs:
            return self.model_configs[pair_key]['model']
        
        # Check reverse direction for bilateral models
        reverse_pair = f"{target_lang}-{source_lang}"
        if reverse_pair in self.model_configs:
            return self.model_configs[reverse_pair]['model']
        
        # Default to multilingual model
        return self.model_configs['multilingual']['model']
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        try:
            # Use existing language detection from AI pipeline
            from nlp_preprocessing import detect_language_with_confidence
            return detect_language_with_confidence(text)
        except ImportError:
            # Fallback to simple heuristic detection
            return self._simple_language_detection(text)
    
    def _simple_language_detection(self, text: str) -> Tuple[str, float]:
        """Simple language detection fallback."""
        # Basic character-based detection
        if not text.strip():
            return 'en', 0.0
            
        # Count characters specific to certain languages
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06ff')
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04ff')
        
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'en', 0.0
            
        # Thresholds for language detection
        if chinese_chars / total_chars > 0.3:
            return 'zh', 0.8
        elif arabic_chars / total_chars > 0.3:
            return 'ar', 0.8
        elif cyrillic_chars / total_chars > 0.3:
            return 'ru', 0.7
        else:
            # Default to English for Latin-based scripts
            return 'en', 0.6
    
    def translate_text(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text to target language.
        
        Args:
            text: Input text to translate
            target_language: Target language code (e.g., 'es', 'fr', 'de')
            source_language: Source language code (auto-detected if None)
            model_override: Specific model to use for translation
            
        Returns:
            Dictionary containing translation and metadata
        """
        try:
            # Validate target language
            if target_language not in self.supported_languages:
                return self._create_error_result(
                    f"Unsupported target language: {target_language}"
                )
            
            # Detect source language if not provided
            if not source_language:
                detected_lang, confidence = self.detect_language(text)
                source_language = detected_lang
                
                if confidence < 0.5:
                    logger.warning(f"Low confidence language detection: {confidence}")
            
            # Validate source language
            if source_language not in self.supported_languages:
                return self._create_error_result(
                    f"Unsupported source language: {source_language}"
                )
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return {
                    'translated_text': text,
                    'source_language': source_language,
                    'target_language': target_language,
                    'model_used': 'none',
                    'quality_score': 1.0,
                    'processing_time_seconds': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'note': 'No translation needed - same language'
                }
            
            # Get appropriate model
            model_name = model_override or self.get_translation_model(
                source_language, target_language
            )
            
            # Get or load translation model
            translator = self.model_pool.get_transformer_model("translation", model_name)
            if not translator:
                return self._create_error_result(f"Failed to load model: {model_name}")
            
            # Preprocess text
            processed_text = self._preprocess_text_for_translation(text)
            
            # Perform translation
            start_time = datetime.now()
            
            # Handle different model types
            if 'mbart' in model_name.lower():
                translation_result = self._translate_with_mbart(
                    translator, processed_text, source_language, target_language
                )
            else:
                translation_result = translator(processed_text)
                if isinstance(translation_result, list) and translation_result:
                    translation_result = translation_result[0]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not translation_result or 'translation_text' not in translation_result:
                return self._create_error_result("Translation model returned no result")
            
            translated_text = translation_result['translation_text']
            
            # Post-process translation
            final_translation = self._postprocess_translation(translated_text)
            
            # Calculate quality score
            quality_score = self.quality_estimator.estimate_quality(
                source_text=processed_text,
                translated_text=final_translation,
                source_lang=source_language,
                target_lang=target_language
            )
            
            return {
                'translated_text': final_translation,
                'source_language': source_language,
                'target_language': target_language,
                'source_text': text,
                'model_used': model_name,
                'quality_score': quality_score,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return self._create_error_result(str(e))
    
    def _translate_with_mbart(
        self, 
        translator, 
        text: str, 
        source_lang: str, 
        target_lang: str
    ) -> Dict[str, str]:
        """Handle mBART-specific translation."""
        # mBART language code mapping
        mbart_lang_codes = {
            'en': 'en_XX',
            'es': 'es_XX', 
            'fr': 'fr_XX',
            'de': 'de_DE',
            'it': 'it_IT',
            'pt': 'pt_XX',
            'nl': 'nl_XX',
            'ru': 'ru_RU',
            'zh': 'zh_CN',
            'ja': 'ja_XX',
            'ko': 'ko_KR',
            'ar': 'ar_AR'
        }
        
        src_lang_code = mbart_lang_codes.get(source_lang, 'en_XX')
        tgt_lang_code = mbart_lang_codes.get(target_lang, 'en_XX')
        
        # Set source and target languages for mBART
        translator.tokenizer.src_lang = src_lang_code
        
        result = translator(text, forced_bos_token_id=translator.tokenizer.lang_code_to_id[tgt_lang_code])
        
        return result[0] if isinstance(result, list) else result
    
    def _preprocess_text_for_translation(self, text: str) -> str:
        """Preprocess text for better translation quality."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common abbreviations and entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        # Ensure proper sentence endings
        text = text.strip()
        if text and not text.endswith(('.', '!', '?', ':', ';')):
            text += '.'
        
        return text
    
    def _postprocess_translation(self, translation: str) -> str:
        """Post-process translation for better quality."""
        if not translation:
            return ""
        
        # Remove excessive whitespace
        translation = re.sub(r'\s+', ' ', translation)
        
        # Fix common capitalization issues
        translation = translation.strip()
        if translation:
            translation = translation[0].upper() + translation[1:]
        
        return translation
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'translated_text': '',
            'source_language': '',
            'target_language': '',
            'source_text': '',
            'model_used': '',
            'quality_score': 0.0,
            'processing_time_seconds': 0.0,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error_message
        }
    
    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (auto-detected if None)
            
        Returns:
            List of translation results
        """
        results = []
        
        logger.info(f"Starting batch translation of {len(texts)} texts to {target_language}")
        
        for i, text in enumerate(texts):
            try:
                result = self.translate_text(
                    text=text,
                    target_language=target_language,
                    source_language=source_language
                )
                
                result['text_index'] = i
                results.append(result)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Failed to translate text {i}: {e}")
                results.append({
                    'text_index': i,
                    **self._create_error_result(str(e))
                })
        
        logger.info(f"Completed batch translation: {len(results)} results")
        return results
    
    def save_translation_to_db(
        self, 
        article_id: str, 
        translation_result: Dict[str, Any]
    ) -> bool:
        """Save translation result to database."""
        try:
            # Get current article metadata
            article = db_adapter.get_article_by_id(article_id)
            if not article:
                logger.error(f"Article not found: {article_id}")
                return False
            
            # Update metadata with translation information
            metadata = json.loads(article.get('metadata', '{}'))
            
            if 'translations' not in metadata:
                metadata['translations'] = {}
            
            target_lang = translation_result['target_language']
            metadata['translations'][target_lang] = {
                'translated_text': translation_result['translated_text'],
                'source_language': translation_result['source_language'],
                'quality_score': translation_result['quality_score'],
                'model_used': translation_result['model_used'],
                'timestamp': translation_result['timestamp']
            }
            
            # Update article metadata
            success = db_adapter.update_article_metadata(article_id, metadata)
            
            if success:
                logger.info(f"Saved translation for article {article_id}")
            else:
                logger.error(f"Failed to save translation for article {article_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving translation to database: {e}")
            return False
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available translation models."""
        return self.model_configs.copy()


class TranslationQualityEstimator:
    """Estimate quality of translations using various metrics."""
    
    def estimate_quality(
        self,
        source_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str
    ) -> float:
        """
        Estimate translation quality using multiple metrics.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            scores = []
            
            # Length ratio score
            length_score = self._calculate_length_ratio_score(source_text, translated_text)
            scores.append(length_score * 0.2)
            
            # Character distribution score
            char_score = self._calculate_character_score(translated_text, target_lang)
            scores.append(char_score * 0.3)
            
            # Fluency score (simple heuristics)
            fluency_score = self._calculate_fluency_score(translated_text)
            scores.append(fluency_score * 0.3)
            
            # Completeness score
            completeness_score = self._calculate_completeness_score(source_text, translated_text)
            scores.append(completeness_score * 0.2)
            
            final_score = sum(scores)
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5  # Default score on error
    
    def _calculate_length_ratio_score(self, source: str, translation: str) -> float:
        """Calculate score based on length ratio."""
        if not source or not translation:
            return 0.0
        
        source_len = len(source.split())
        trans_len = len(translation.split())
        
        if source_len == 0:
            return 0.0
        
        ratio = trans_len / source_len
        
        # Ideal ratio depends on language pair, but 0.8-1.5 is generally good
        if 0.8 <= ratio <= 1.5:
            return 1.0
        elif ratio < 0.3 or ratio > 3.0:
            return 0.2  # Very poor ratio
        else:
            # Linear penalty outside ideal range
            if ratio < 0.8:
                return ratio / 0.8
            else:  # ratio > 1.5
                return max(0.2, 2.0 - ratio / 1.5)
    
    def _calculate_character_score(self, text: str, target_lang: str) -> float:
        """Calculate score based on expected character distribution."""
        if not text:
            return 0.0
        
        # Language-specific character expectations
        lang_expectations = {
            'zh': lambda t: sum(1 for c in t if '\u4e00' <= c <= '\u9fff') / len(t),
            'ar': lambda t: sum(1 for c in t if '\u0600' <= c <= '\u06ff') / len(t),
            'ru': lambda t: sum(1 for c in t if '\u0400' <= c <= '\u04ff') / len(t),
            'ja': lambda t: sum(1 for c in t if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff') / len(t)
        }
        
        if target_lang in lang_expectations:
            char_ratio = lang_expectations[target_lang](text)
            # Expect at least 30% of characters to be language-specific
            return min(1.0, char_ratio / 0.3)
        else:
            # For Latin-based languages, check for reasonable character distribution
            alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text) if text else 0
            return min(1.0, alpha_ratio / 0.7)  # Expect 70% alphabetic characters
    
    def _calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score using simple heuristics."""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Check for reasonable sentence structure
        sentences = text.split('.')
        if len(sentences) < 2 and len(text) > 100:
            score -= 0.2  # Long text without sentence breaks
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(set(words)) < len(words) * 0.6:  # Less than 60% unique words
            score -= 0.3
        
        # Check for proper capitalization
        if not text[0].isupper():
            score -= 0.1
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if punct_ratio > 0.2:  # More than 20% punctuation
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_completeness_score(self, source: str, translation: str) -> float:
        """Calculate how complete the translation appears."""
        if not source or not translation:
            return 0.0
        
        # Check if translation ends properly
        if not translation.strip().endswith(('.', '!', '?', ':', ';')):
            return 0.7  # Incomplete translation
        
        # Check if translation is too short relative to source
        source_words = len(source.split())
        trans_words = len(translation.split())
        
        if trans_words < source_words * 0.3:  # Less than 30% of source length
            return 0.5
        
        return 1.0


# Extend ModelPool to support translation models
def _extend_model_pool_for_translation():
    """Extend ModelPool class to include translation models."""
    original_get_transformer = ModelPool.get_transformer_model
    
    def get_transformer_model_extended(self, task: str, model_name: Optional[str] = None):
        """Extended transformer model getter with translation support."""
        if task == "translation":
            key = f"transformer_{task}_{model_name or 'default'}"
            
            with self.model_locks[key]:
                if key not in self.models:
                    logger.info(f"Loading translation model: {model_name}")
                    try:
                        model = model_name or "Helsinki-NLP/opus-mt-en-es"
                        translator = pipeline(
                            "translation",
                            model=model,
                            tokenizer=model,
                            device=0 if torch.cuda.is_available() else -1,
                            framework="pt"
                        )
                        self.models[key] = translator
                    except Exception as e:
                        logger.error(f"Failed to load translation model {model_name}: {e}")
                        return None
                
                self.usage_counts[key] += 1
                self.last_used[key] = time.time()
                return self.models[key]
        else:
            return original_get_transformer(self, task, model_name)
    
    # Monkey patch the method
    ModelPool.get_transformer_model = get_transformer_model_extended


# Initialize the extension
_extend_model_pool_for_translation()