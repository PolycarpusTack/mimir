"""
Translation API router for Mimir - EPIC 8.2
Provides REST endpoints for multi-language translation functionality.
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from api.dependencies import get_current_user, verify_api_key, get_db_connection
from auth.permissions import require_permission
from translation_engine import TranslationEngine
import db_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/translation", tags=["translation"])


# Pydantic models for request/response
class TranslationRequest(BaseModel):
    """Request model for text translation."""
    
    article_id: Optional[str] = Field(None, description="Article ID to translate from database")
    text: Optional[str] = Field(None, description="Raw text to translate")
    target_language: str = Field(..., description="Target language code (e.g., 'es', 'fr', 'de')")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detected if not provided)")
    model_override: Optional[str] = Field(None, description="Specific model to use for translation")
    save_to_db: bool = Field(True, description="Whether to save translation to database")
    
    @validator('target_language')
    def validate_target_language(cls, v):
        # This will be validated against engine.supported_languages in the endpoint
        if not v or len(v) < 2:
            raise ValueError("target_language must be a valid language code")
        return v.lower()
    
    @validator('source_language')
    def validate_source_language(cls, v):
        if v and len(v) < 2:
            raise ValueError("source_language must be a valid language code")
        return v.lower() if v else v
    
    @validator('text')
    def validate_text_or_id(cls, v, values):
        if not values.get('article_id') and not v:
            raise ValueError("Either article_id or text must be provided")
        return v


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""
    
    article_ids: List[str] = Field(..., description="List of article IDs to translate")
    target_language: str = Field(..., description="Target language code")
    source_language: Optional[str] = Field(None, description="Source language code (auto-detected if not provided)")
    model_override: Optional[str] = Field(None, description="Specific model to use for translation")
    save_to_db: bool = Field(True, description="Whether to save translations to database")
    
    @validator('article_ids')
    def validate_article_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("article_ids cannot be empty")
        if len(v) > 50:  # Limit batch size for translations
            raise ValueError("Maximum 50 articles per batch")
        return v
    
    @validator('target_language')
    def validate_target_language(cls, v):
        if not v or len(v) < 2:
            raise ValueError("target_language must be a valid language code")
        return v.lower()


class TranslationResponse(BaseModel):
    """Response model for translation results."""
    
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Detected or provided source language")
    target_language: str = Field(..., description="Target language")
    source_text: str = Field(..., description="Original text")
    model_used: str = Field(..., description="Model used for translation")
    quality_score: float = Field(..., description="Translation quality score (0-1)")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Timestamp of translation")
    success: bool = Field(..., description="Whether translation succeeded")
    article_id: Optional[str] = Field(None, description="Article ID if applicable")
    error: Optional[str] = Field(None, description="Error message if failed")
    note: Optional[str] = Field(None, description="Additional notes")


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation."""
    
    results: List[TranslationResponse] = Field(..., description="List of translation results")
    total_processed: int = Field(..., description="Total number of articles processed")
    successful: int = Field(..., description="Number of successful translations")
    failed: int = Field(..., description="Number of failed translations")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    target_language: str = Field(..., description="Target language for all translations")


class LanguageDetectionRequest(BaseModel):
    """Request model for language detection."""
    
    text: str = Field(..., description="Text to analyze for language detection")


class LanguageDetectionResponse(BaseModel):
    """Response model for language detection."""
    
    detected_language: str = Field(..., description="Detected language code")
    language_name: str = Field(..., description="Detected language name")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    text_sample: str = Field(..., description="Sample of analyzed text")


# Initialize translation engine
translation_engine = TranslationEngine()


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Translate a single text or article.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Validate target language
        supported_languages = translation_engine.get_supported_languages()
        if request.target_language not in supported_languages:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported target language: {request.target_language}. "
                       f"Supported: {list(supported_languages.keys())}"
            )
        
        # Validate source language if provided
        if request.source_language and request.source_language not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source language: {request.source_language}. "
                       f"Supported: {list(supported_languages.keys())}"
            )
        
        # Get text content
        if request.article_id:
            article = db_adapter.get_article_by_id(request.article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            text_content = article.get('content', '') or article.get('title', '')
            if not text_content:
                raise HTTPException(status_code=400, detail="Article has no content")
        else:
            text_content = request.text
        
        # Perform translation
        result = translation_engine.translate_text(
            text=text_content,
            target_language=request.target_language,
            source_language=request.source_language,
            model_override=request.model_override
        )
        
        # Add article_id to result if provided
        if request.article_id:
            result['article_id'] = request.article_id
        
        # Save to database if requested and article_id provided
        if request.save_to_db and request.article_id and result.get('success'):
            saved = translation_engine.save_translation_to_db(request.article_id, result)
            if not saved:
                logger.warning(f"Failed to save translation to database for article {request.article_id}")
        
        return TranslationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal translation error")


@router.post("/batch-translate", response_model=BatchTranslationResponse)
async def batch_translate_articles(
    request: BatchTranslationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Translate multiple articles in batch.
    
    Requires 'read_articles' permission.
    For large batches, processing continues in background.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Validate target language
        supported_languages = translation_engine.get_supported_languages()
        if request.target_language not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target language: {request.target_language}"
            )
        
        # Get articles from database
        texts = []
        articles_map = {}
        
        for article_id in request.article_ids:
            article = db_adapter.get_article_by_id(article_id)
            if article:
                content = article.get('content', '') or article.get('title', '')
                if content:
                    texts.append(content)
                    articles_map[len(texts) - 1] = article_id
        
        if not texts:
            raise HTTPException(status_code=400, detail="No valid articles with content found")
        
        # Process batch translation
        results = translation_engine.batch_translate(
            texts=texts,
            target_language=request.target_language,
            source_language=request.source_language
        )
        
        # Map results back to article IDs
        for result in results:
            text_index = result.get('text_index', 0)
            if text_index in articles_map:
                result['article_id'] = articles_map[text_index]
        
        # Save to database if requested
        if request.save_to_db:
            def save_batch_results():
                for result in results:
                    if result.get('success') and result.get('article_id'):
                        translation_engine.save_translation_to_db(
                            result['article_id'], 
                            result
                        )
            
            background_tasks.add_task(save_batch_results)
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        total_time = sum(r.get('processing_time_seconds', 0) for r in results)
        
        response_results = [TranslationResponse(**result) for result in results]
        
        return BatchTranslationResponse(
            results=response_results,
            total_processed=len(results),
            successful=successful,
            failed=failed,
            total_processing_time=total_time,
            target_language=request.target_language
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal batch translation error")


@router.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectionRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Detect the language of provided text.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Detect language
        detected_lang, confidence = translation_engine.detect_language(request.text)
        
        # Get language name
        supported_languages = translation_engine.get_supported_languages()
        language_name = supported_languages.get(detected_lang, "Unknown")
        
        # Create text sample (first 100 characters)
        text_sample = request.text[:100] + "..." if len(request.text) > 100 else request.text
        
        return LanguageDetectionResponse(
            detected_language=detected_lang,
            language_name=language_name,
            confidence=confidence,
            text_sample=text_sample
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal language detection error")


@router.get("/article/{article_id}/translations")
async def get_article_translations(
    article_id: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get all existing translations for an article.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        metadata = article.get('metadata', '{}')
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)
        
        translations = metadata.get('translations', {})
        
        return {
            'article_id': article_id,
            'translations': translations,
            'total_translations': len(translations),
            'available_languages': list(translations.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get article translations: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving translations")


@router.delete("/article/{article_id}/translations/{language}")
async def delete_article_translation(
    article_id: str,
    language: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Delete a specific translation for an article.
    
    Requires 'edit_articles' permission.
    """
    require_permission(current_user, "edit_articles")
    
    try:
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        metadata = article.get('metadata', '{}')
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)
        
        translations = metadata.get('translations', {})
        
        if language not in translations:
            raise HTTPException(status_code=404, detail="Translation not found")
        
        # Remove the translation
        del translations[language]
        metadata['translations'] = translations
        
        # Update article metadata
        success = db_adapter.update_article_metadata(article_id, metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete translation")
        
        return {
            'message': f"Translation for '{language}' deleted successfully",
            'article_id': article_id,
            'remaining_translations': len(translations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete translation: {e}")
        raise HTTPException(status_code=500, detail="Internal error deleting translation")


@router.get("/languages")
async def get_supported_languages():
    """
    Get list of supported languages.
    
    No authentication required - public endpoint.
    """
    try:
        supported_languages = translation_engine.get_supported_languages()
        available_models = translation_engine.get_available_models()
        
        return {
            'supported_languages': supported_languages,
            'total_languages': len(supported_languages),
            'available_models': available_models,
            'default_model': 'facebook/mbart-large-50-many-to-many-mmt'
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving languages")


@router.get("/stats")
async def get_translation_stats(
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get translation usage statistics.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get basic stats from database
        total_articles = db_adapter.count_articles()
        
        # Count articles with translations
        articles_with_translations = db_adapter.count_articles_with_metadata_key('translations')
        
        # Get language distribution (this would need a custom query)
        # For now, we'll provide basic stats
        
        return {
            'total_articles': total_articles,
            'articles_with_translations': articles_with_translations,
            'coverage_percentage': round((articles_with_translations / total_articles * 100), 2) if total_articles > 0 else 0,
            'supported_languages': len(translation_engine.get_supported_languages()),
            'translation_engine_status': 'active'
        }
        
    except Exception as e:
        logger.error(f"Failed to get translation stats: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving stats")