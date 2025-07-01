"""
Content Generation API router for Mimir - EPIC 8.3
Provides REST endpoints for automated content generation (newsletters, reports).
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from api.dependencies import get_current_user, verify_api_key, get_db_connection
from auth.permissions import require_permission
from content_generation import ContentGenerationEngine
import db_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/content-generation", tags=["content-generation"])


# Pydantic models for request/response
class NewsletterRequest(BaseModel):
    """Request model for newsletter generation."""
    
    date_range_days: int = Field(1, ge=1, le=30, description="Number of days to look back for articles")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User interests and preferences")
    max_articles: int = Field(15, ge=5, le=50, description="Maximum number of articles to include")
    highlight_count: int = Field(3, ge=1, le=10, description="Number of top highlights")
    trending_count: int = Field(5, ge=1, le=15, description="Number of trending topics")
    personalized: bool = Field(True, description="Whether to apply personalization")
    
    @validator('user_preferences')
    def validate_user_preferences(cls, v):
        if v and not isinstance(v, dict):
            raise ValueError("user_preferences must be a dictionary")
        return v


class ExecutiveBriefingRequest(BaseModel):
    """Request model for executive briefing generation."""
    
    date_range_days: int = Field(7, ge=1, le=30, description="Number of days to analyze")
    focus_areas: Optional[List[str]] = Field(None, description="Areas of business focus")
    executive_level: str = Field("senior", description="Target executive level")
    max_articles: int = Field(10, ge=3, le=25, description="Maximum number of articles to analyze")
    include_recommendations: bool = Field(True, description="Whether to include strategic recommendations")
    
    @validator('executive_level')
    def validate_executive_level(cls, v):
        allowed_levels = ['senior', 'c-level', 'director', 'vp']
        if v not in allowed_levels:
            raise ValueError(f"executive_level must be one of: {allowed_levels}")
        return v
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        if v and len(v) > 10:
            raise ValueError("Maximum 10 focus areas allowed")
        return v


class TrendReportRequest(BaseModel):
    """Request model for trend report generation."""
    
    analysis_period_days: int = Field(7, ge=3, le=30, description="Period for trend analysis")
    trend_categories: Optional[List[str]] = Field(None, description="Specific categories to analyze")
    min_trend_strength: float = Field(0.3, ge=0.1, le=1.0, description="Minimum trend strength threshold")
    include_predictions: bool = Field(True, description="Whether to include trend predictions")
    
    @validator('trend_categories')
    def validate_trend_categories(cls, v):
        if v and len(v) > 15:
            raise ValueError("Maximum 15 trend categories allowed")
        return v


class ContentGenerationResponse(BaseModel):
    """Response model for generated content."""
    
    content: str = Field(..., description="Generated content")
    content_type: str = Field(..., description="Type of content generated")
    content_id: Optional[str] = Field(None, description="Unique identifier for the content")
    articles_used: int = Field(..., description="Number of articles analyzed")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Generation timestamp")
    success: bool = Field(..., description="Whether generation succeeded")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class ContentSectionsResponse(BaseModel):
    """Response model for content with sections breakdown."""
    
    content: str = Field(..., description="Full generated content")
    sections: Dict[str, str] = Field(..., description="Individual content sections")
    content_type: str = Field(..., description="Type of content generated")
    content_id: Optional[str] = Field(None, description="Unique identifier for the content")
    articles_used: int = Field(..., description="Number of articles analyzed")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Generation timestamp")
    success: bool = Field(..., description="Whether generation succeeded")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class ContentTemplateRequest(BaseModel):
    """Request model for custom content template."""
    
    template_name: str = Field(..., description="Name for the custom template")
    sections: List[str] = Field(..., description="List of sections to include")
    max_articles: int = Field(20, ge=5, le=100, description="Maximum articles to analyze")
    content_type: str = Field("custom", description="Type of content")
    
    @validator('sections')
    def validate_sections(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one section must be specified")
        if len(v) > 20:
            raise ValueError("Maximum 20 sections allowed")
        return v


# Initialize content generation engine
content_engine = ContentGenerationEngine()


@router.post("/newsletter", response_model=ContentSectionsResponse)
async def generate_newsletter(
    request: NewsletterRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Generate a personalized newsletter.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Prepare user preferences
        user_prefs = request.user_preferences if request.personalized else None
        
        # Create custom template if different from defaults
        custom_template = None
        if (request.max_articles != 15 or 
            request.highlight_count != 3 or 
            request.trending_count != 5):
            custom_template = {
                'sections': ['header', 'highlights', 'trending', 'analysis', 'footer'],
                'max_articles': request.max_articles,
                'highlight_count': request.highlight_count,
                'trending_count': request.trending_count
            }
        
        # Generate newsletter
        result = content_engine.generate_newsletter(
            user_preferences=user_prefs,
            date_range_days=request.date_range_days,
            custom_template=custom_template
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500, 
                detail=result.get('error', 'Newsletter generation failed')
            )
        
        # Generate unique content ID
        content_id = f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.get('id', 'unknown')}"
        
        # Save generated content
        saved = content_engine.save_generated_content(
            content_result=result,
            content_id=content_id,
            metadata={
                'user_id': current_user.get('id'),
                'request_params': request.dict()
            }
        )
        
        response_data = {
            'content': result['content'],
            'sections': result.get('sections', {}),
            'content_type': result['content_type'],
            'content_id': content_id if saved else None,
            'articles_used': result['articles_used'],
            'processing_time_seconds': result['processing_time_seconds'],
            'timestamp': result['timestamp'],
            'success': result['success'],
            'metadata': {
                'personalized': result.get('personalized', False),
                'date_range_days': result.get('date_range_days', 1)
            }
        }
        
        return ContentSectionsResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Newsletter generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal newsletter generation error")


@router.post("/executive-briefing", response_model=ContentSectionsResponse)
async def generate_executive_briefing(
    request: ExecutiveBriefingRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Generate an executive briefing report.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Generate executive briefing
        result = content_engine.generate_executive_briefing(
            focus_areas=request.focus_areas,
            date_range_days=request.date_range_days,
            executive_level=request.executive_level
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Executive briefing generation failed')
            )
        
        # Generate unique content ID
        content_id = f"exec_briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.get('id', 'unknown')}"
        
        # Save generated content
        saved = content_engine.save_generated_content(
            content_result=result,
            content_id=content_id,
            metadata={
                'user_id': current_user.get('id'),
                'request_params': request.dict()
            }
        )
        
        response_data = {
            'content': result['content'],
            'sections': result.get('sections', {}),
            'content_type': result['content_type'],
            'content_id': content_id if saved else None,
            'articles_used': result['articles_analyzed'],
            'processing_time_seconds': result['processing_time_seconds'],
            'timestamp': result['timestamp'],
            'success': result['success'],
            'metadata': {
                'focus_areas': result.get('focus_areas', []),
                'executive_level': result.get('executive_level', 'senior'),
                'date_range_days': result.get('date_range_days', 7)
            }
        }
        
        return ContentSectionsResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Executive briefing generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal executive briefing generation error")


@router.post("/trend-report", response_model=ContentSectionsResponse)
async def generate_trend_report(
    request: TrendReportRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Generate a trend analysis report.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Generate trend report
        result = content_engine.generate_trend_report(
            analysis_period_days=request.analysis_period_days,
            trend_categories=request.trend_categories
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Trend report generation failed')
            )
        
        # Generate unique content ID
        content_id = f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{current_user.get('id', 'unknown')}"
        
        # Save generated content
        saved = content_engine.save_generated_content(
            content_result=result,
            content_id=content_id,
            metadata={
                'user_id': current_user.get('id'),
                'request_params': request.dict()
            }
        )
        
        response_data = {
            'content': result['content'],
            'sections': result.get('sections', {}),
            'content_type': result['content_type'],
            'content_id': content_id if saved else None,
            'articles_used': result['articles_analyzed'],
            'processing_time_seconds': result['processing_time_seconds'],
            'timestamp': result['timestamp'],
            'success': result['success'],
            'metadata': {
                'trends_identified': result.get('trends_identified', 0),
                'analysis_period_days': result.get('analysis_period_days', 7)
            }
        }
        
        return ContentSectionsResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend report generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal trend report generation error")


@router.get("/templates")
async def get_content_templates():
    """
    Get available content templates and their configurations.
    
    No authentication required - public endpoint.
    """
    try:
        # Get templates from the engine
        templates = content_engine.templates
        
        return {
            'available_templates': {
                name: {
                    'sections': config.get('sections', []),
                    'description': _get_template_description(name),
                    'default_settings': {k: v for k, v in config.items() if k != 'sections'}
                }
                for name, config in templates.items()
            },
            'total_templates': len(templates),
            'supported_content_types': list(templates.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get content templates: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving templates")


def _get_template_description(template_name: str) -> str:
    """Get description for a template."""
    descriptions = {
        'newsletter': 'Personalized news digest with highlights and trending topics',
        'executive_briefing': 'Strategic briefing for executive decision-making',
        'trend_report': 'Comprehensive analysis of trending topics and patterns',
        'competitive_analysis': 'Market intelligence focused on competitive landscape'
    }
    return descriptions.get(template_name, 'Custom content template')


@router.get("/content/{content_id}")
async def get_generated_content(
    content_id: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Retrieve previously generated content by ID.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # In production, this would query a dedicated content storage table
        # For now, return a not implemented response
        raise HTTPException(
            status_code=501, 
            detail="Content retrieval not yet implemented. Content is generated on-demand."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve content: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving content")


@router.delete("/content/{content_id}")
async def delete_generated_content(
    content_id: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Delete previously generated content by ID.
    
    Requires 'edit_articles' permission.
    """
    require_permission(current_user, "edit_articles")
    
    try:
        # In production, this would delete from a dedicated content storage table
        raise HTTPException(
            status_code=501,
            detail="Content deletion not yet implemented. Content is generated on-demand."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete content: {e}")
        raise HTTPException(status_code=500, detail="Internal error deleting content")


@router.get("/stats")
async def get_content_generation_stats(
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get content generation usage statistics.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get basic stats
        total_articles = db_adapter.count_articles()
        
        return {
            'total_articles_available': total_articles,
            'supported_content_types': len(content_engine.templates),
            'available_templates': list(content_engine.templates.keys()),
            'personalization_enabled': True,
            'content_generation_engine_status': 'active',
            'supported_features': [
                'newsletter_generation',
                'executive_briefings',
                'trend_reports',
                'competitive_analysis',
                'personalization',
                'multi_format_output'
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get content generation stats: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving stats")


@router.post("/preview")
async def preview_content_generation(
    request: NewsletterRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Generate a preview of content without full processing.
    
    Requires 'read_articles' permission.
    Returns a limited preview for quick evaluation.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get limited articles for preview
        articles = db_adapter.get_recent_articles(limit=5, hours=request.date_range_days * 24)
        
        if not articles:
            raise HTTPException(status_code=404, detail="No recent articles found for preview")
        
        # Generate simple preview
        preview_content = f"# Preview: Newsletter for {datetime.now().strftime('%B %d, %Y')}\n\n"
        preview_content += f"Based on {len(articles)} recent articles from the last {request.date_range_days} day(s):\n\n"
        
        for i, article in enumerate(articles[:3], 1):
            title = article.get('title', 'Untitled')
            source = article.get('source', 'Unknown')
            preview_content += f"{i}. **{title}** ({source})\n"
        
        if len(articles) > 3:
            preview_content += f"\n...and {len(articles) - 3} more articles\n"
        
        preview_content += f"\n*This is a preview. Full generation will include detailed analysis, summaries, and personalization.*"
        
        return {
            'preview_content': preview_content,
            'articles_available': len(articles),
            'estimated_processing_time': f"{len(articles) * 2}-{len(articles) * 5} seconds",
            'would_be_personalized': bool(request.user_preferences),
            'preview_generated_at': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content preview failed: {e}")
        raise HTTPException(status_code=500, detail="Internal preview generation error")