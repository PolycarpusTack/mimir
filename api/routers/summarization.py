"""
Summarization API router for Mimir - EPIC 8.1
Provides REST endpoints for article summarization functionality.
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from api.dependencies import get_current_user, verify_api_key, get_db_connection
from auth.permissions import require_permission
from summarization_engine import SummarizationEngine
import db_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarization", tags=["summarization"])


# Pydantic models for request/response
class SummarizationRequest(BaseModel):
    """Request model for single article summarization."""
    
    article_id: Optional[str] = Field(None, description="Article ID to summarize from database")
    text: Optional[str] = Field(None, description="Raw text to summarize")
    format_type: str = Field("custom", description="Summary format: bullet_points, executive, tweet, custom")
    custom_length: Optional[int] = Field(None, ge=20, le=1000, description="Custom maximum summary length")
    model_type: str = Field("facebook/bart-large-cnn", description="Model to use for summarization")
    save_to_db: bool = Field(True, description="Whether to save summary to database")
    
    @validator('format_type')
    def validate_format_type(cls, v):
        allowed_formats = ['bullet_points', 'executive', 'tweet', 'custom']
        if v not in allowed_formats:
            raise ValueError(f"format_type must be one of: {allowed_formats}")
        return v
    
    @validator('text')
    def validate_text_or_id(cls, v, values):
        if not values.get('article_id') and not v:
            raise ValueError("Either article_id or text must be provided")
        return v


class BatchSummarizationRequest(BaseModel):
    """Request model for batch summarization."""
    
    article_ids: List[str] = Field(..., description="List of article IDs to summarize")
    format_type: str = Field("custom", description="Summary format for all articles")
    custom_length: Optional[int] = Field(None, ge=20, le=1000, description="Custom maximum summary length")
    model_type: str = Field("facebook/bart-large-cnn", description="Model to use for summarization")
    save_to_db: bool = Field(True, description="Whether to save summaries to database")
    
    @validator('article_ids')
    def validate_article_ids(cls, v):
        if not v or len(v) == 0:
            raise ValueError("article_ids cannot be empty")
        if len(v) > 100:  # Limit batch size
            raise ValueError("Maximum 100 articles per batch")
        return v
    
    @validator('format_type')
    def validate_format_type(cls, v):
        allowed_formats = ['bullet_points', 'executive', 'tweet', 'custom']
        if v not in allowed_formats:
            raise ValueError(f"format_type must be one of: {allowed_formats}")
        return v


class SummarizationResponse(BaseModel):
    """Response model for summarization results."""
    
    summary: str = Field(..., description="Generated summary")
    format_type: str = Field(..., description="Summary format used")
    model_used: str = Field(..., description="Model used for summarization")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    quality_score: float = Field(..., description="Quality score (0-1)")
    input_length: int = Field(..., description="Original text length")
    output_length: int = Field(..., description="Summary length")
    compression_ratio: float = Field(..., description="Compression ratio")
    timestamp: str = Field(..., description="Timestamp of generation")
    success: bool = Field(..., description="Whether summarization succeeded")
    article_id: Optional[str] = Field(None, description="Article ID if applicable")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchSummarizationResponse(BaseModel):
    """Response model for batch summarization."""
    
    results: List[SummarizationResponse] = Field(..., description="List of summarization results")
    total_processed: int = Field(..., description="Total number of articles processed")
    successful: int = Field(..., description="Number of successful summarizations")
    failed: int = Field(..., description="Number of failed summarizations")
    total_processing_time: float = Field(..., description="Total processing time in seconds")


# Initialize summarization engine
summarization_engine = SummarizationEngine()


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_article(
    request: SummarizationRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Summarize a single article or text.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get text content
        if request.article_id:
            article = db_adapter.get_article_by_id(request.article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            text_content = article.get('content', '')
            if not text_content:
                raise HTTPException(status_code=400, detail="Article has no content")
        else:
            text_content = request.text
        
        # Generate summary
        result = summarization_engine.generate_summary(
            text=text_content,
            format_type=request.format_type,
            custom_length=request.custom_length,
            model_type=request.model_type
        )
        
        # Add article_id to result if provided
        if request.article_id:
            result['article_id'] = request.article_id
        
        # Save to database if requested and article_id provided
        if request.save_to_db and request.article_id and result.get('success'):
            saved = summarization_engine.save_summary_to_db(request.article_id, result)
            if not saved:
                logger.warning(f"Failed to save summary to database for article {request.article_id}")
        
        return SummarizationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Internal summarization error")


@router.post("/batch-summarize", response_model=BatchSummarizationResponse)
async def batch_summarize_articles(
    request: BatchSummarizationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Summarize multiple articles in batch.
    
    Requires 'read_articles' permission.
    For large batches, processing continues in background.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get articles from database
        articles = []
        for article_id in request.article_ids:
            article = db_adapter.get_article_by_id(article_id)
            if article and article.get('content'):
                articles.append({
                    'id': article_id,
                    'content': article.get('content', '')
                })
        
        if not articles:
            raise HTTPException(status_code=400, detail="No valid articles found")
        
        # Process batch summarization
        results = summarization_engine.batch_summarize(
            articles=articles,
            format_type=request.format_type,
            custom_length=request.custom_length
        )
        
        # Save to database if requested
        if request.save_to_db:
            def save_batch_results():
                for result in results:
                    if result.get('success') and result.get('article_id'):
                        summarization_engine.save_summary_to_db(
                            result['article_id'], 
                            result
                        )
            
            background_tasks.add_task(save_batch_results)
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        total_time = sum(r.get('processing_time_seconds', 0) for r in results)
        
        response_results = [SummarizationResponse(**result) for result in results]
        
        return BatchSummarizationResponse(
            results=response_results,
            total_processed=len(results),
            successful=successful,
            failed=failed,
            total_processing_time=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch summarization failed: {e}")
        raise HTTPException(status_code=500, detail="Internal batch summarization error")


@router.get("/article/{article_id}/summaries")
async def get_article_summaries(
    article_id: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get all existing summaries for an article.
    
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
        
        summaries = metadata.get('summaries', {})
        
        return {
            'article_id': article_id,
            'summaries': summaries,
            'total_summaries': len(summaries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get article summaries: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving summaries")


@router.delete("/article/{article_id}/summaries/{format_type}")
async def delete_article_summary(
    article_id: str,
    format_type: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Delete a specific summary for an article.
    
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
        
        summaries = metadata.get('summaries', {})
        
        if format_type not in summaries:
            raise HTTPException(status_code=404, detail="Summary not found")
        
        # Remove the summary
        del summaries[format_type]
        metadata['summaries'] = summaries
        
        # Update article metadata
        success = db_adapter.update_article_metadata(article_id, metadata)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete summary")
        
        return {
            'message': f"Summary '{format_type}' deleted successfully",
            'article_id': article_id,
            'remaining_summaries': len(summaries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete summary: {e}")
        raise HTTPException(status_code=500, detail="Internal error deleting summary")


@router.get("/formats")
async def get_summary_formats():
    """
    Get available summary formats and their configurations.
    
    No authentication required - public endpoint.
    """
    try:
        return {
            'available_formats': {
                'bullet_points': {
                    'name': 'Bullet Points',
                    'description': 'Key points in bullet format',
                    'typical_length': '50-150 characters',
                    'use_case': 'Quick overview, presentations'
                },
                'executive': {
                    'name': 'Executive Summary',
                    'description': 'Professional executive summary',
                    'typical_length': '100-300 characters',
                    'use_case': 'Business reports, decision making'
                },
                'tweet': {
                    'name': 'Tweet Format',
                    'description': 'Social media ready summary',
                    'typical_length': '20-280 characters',
                    'use_case': 'Social sharing, quick updates'
                },
                'custom': {
                    'name': 'Custom Length',
                    'description': 'Customizable length summary',
                    'typical_length': 'User-defined',
                    'use_case': 'Flexible use cases'
                }
            },
            'supported_models': [
                'facebook/bart-large-cnn',
                'google/pegasus-xsum',
                't5-small',
                't5-base'
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get summary formats: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving formats")


@router.get("/stats")
async def get_summarization_stats(
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get summarization usage statistics.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get basic stats from database
        total_articles = db_adapter.count_articles()
        
        # Count articles with summaries
        articles_with_summaries = db_adapter.count_articles_with_metadata_key('summaries')
        
        # Get format distribution
        format_stats = db_adapter.get_summary_format_distribution()
        
        return {
            'total_articles': total_articles,
            'articles_with_summaries': articles_with_summaries,
            'coverage_percentage': round((articles_with_summaries / total_articles * 100), 2) if total_articles > 0 else 0,
            'format_distribution': format_stats,
            'summarization_engine_status': 'active'
        }
        
    except Exception as e:
        logger.error(f"Failed to get summarization stats: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving stats")