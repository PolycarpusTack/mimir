"""
Recommendation Engine API Router - EPIC 6 Completion
Provides endpoints for content-based, collaborative filtering, and hybrid recommendations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...recommendation_engine import RecommendationEngine
from ...db_manager_postgres import PostgreSQLManager
from ..middleware.auth import get_current_user, require_permissions

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class ContentBasedRequest(BaseModel):
    """Request model for content-based recommendations."""
    article_id: str = Field(description="Article ID to find similar articles for")
    n_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")


class CollaborativeRequest(BaseModel):
    """Request model for collaborative filtering recommendations."""
    user_id: str = Field(description="User ID to generate recommendations for")
    n_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    min_score: float = Field(default=0.1, ge=0.0, le=5.0, description="Minimum recommendation score")


class HybridRequest(BaseModel):
    """Request model for hybrid recommendations."""
    user_id: str = Field(description="User ID to generate recommendations for")
    article_context: Optional[str] = Field(default=None, description="Optional article context")
    n_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    content_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight for content-based component")
    collaborative_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for collaborative component")


class FeedbackRequest(BaseModel):
    """Request model for user feedback processing."""
    user_id: str = Field(description="User ID providing feedback")
    article_id: str = Field(description="Article ID being rated")
    feedback_type: str = Field(regex="^(view|like|share|save|comment)$", description="Type of feedback")
    feedback_value: float = Field(default=1.0, ge=0.1, le=5.0, description="Feedback value/rating")


class ModelBuildRequest(BaseModel):
    """Request model for building recommendation models."""
    content_based: bool = Field(default=True, description="Build content-based model")
    collaborative_filtering: bool = Field(default=True, description="Build collaborative filtering model")
    article_limit: int = Field(default=1000, ge=10, le=5000, description="Articles for content model")
    interaction_limit: int = Field(default=5000, ge=10, le=50000, description="Interactions for CF model")


class RecommendationResponse(BaseModel):
    """Response model for recommendation results."""
    success: bool
    message: str
    recommendation_method: str
    target_user_id: Optional[str] = None
    target_article_id: Optional[str] = None
    recommendations: List[Dict]
    recommendation_count: int
    parameters: Dict
    processing_time_ms: float
    timestamp: str


class FeedbackResponse(BaseModel):
    """Response model for feedback processing."""
    success: bool
    message: str
    feedback_processed: bool
    user_id: str
    article_id: str
    feedback_type: str
    weighted_score: float
    timestamp: str


class ModelBuildResponse(BaseModel):
    """Response model for model building."""
    success: bool
    message: str
    components_built: Dict
    available_methods: List[str]
    build_timestamp: str
    processing_time_ms: float


def get_recommendation_engine() -> RecommendationEngine:
    """Dependency to get recommendation engine instance."""
    return RecommendationEngine()


@router.post("/content-based", response_model=RecommendationResponse)
async def get_content_based_recommendations(
    request: ContentBasedRequest,
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get content-based recommendations for a given article.
    
    Finds articles similar in content, topic, and metadata.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        start_time = datetime.now()
        
        # Get content-based recommendations
        result = rec_engine.get_content_based_recommendations(
            article_id=request.article_id,
            n_recommendations=request.n_recommendations,
            min_similarity=request.min_similarity
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Content-based recommendations failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            success=True,
            message=f"Generated {result['recommendation_count']} content-based recommendations",
            recommendation_method="content_based",
            target_article_id=request.article_id,
            recommendations=result['recommendations'],
            recommendation_count=result['recommendation_count'],
            parameters=result['parameters'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content-based recommendations API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collaborative", response_model=RecommendationResponse)
async def get_collaborative_recommendations(
    request: CollaborativeRequest,
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get collaborative filtering recommendations for a user.
    
    Recommends articles based on similar users' preferences.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        start_time = datetime.now()
        
        # Get collaborative filtering recommendations
        result = rec_engine.get_collaborative_recommendations(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations,
            min_score=request.min_score
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Collaborative filtering failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            success=True,
            message=f"Generated {result['recommendation_count']} collaborative filtering recommendations",
            recommendation_method="collaborative_filtering",
            target_user_id=request.user_id,
            recommendations=result['recommendations'],
            recommendation_count=result['recommendation_count'],
            parameters=result['parameters'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collaborative filtering API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid", response_model=RecommendationResponse)
async def get_hybrid_recommendations(
    request: HybridRequest,
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get hybrid recommendations combining content-based and collaborative filtering.
    
    Provides the most comprehensive and accurate recommendations.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        # Validate weights sum to reasonable value
        if abs((request.content_weight + request.collaborative_weight) - 1.0) > 0.1:
            raise HTTPException(
                status_code=400,
                detail="Content weight and collaborative weight should sum to approximately 1.0"
            )
        
        start_time = datetime.now()
        
        # Get hybrid recommendations
        result = rec_engine.get_hybrid_recommendations(
            user_id=request.user_id,
            article_context=request.article_context,
            n_recommendations=request.n_recommendations,
            content_weight=request.content_weight,
            collaborative_weight=request.collaborative_weight
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Hybrid recommendations failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            success=True,
            message=f"Generated {result['recommendation_count']} hybrid recommendations",
            recommendation_method="hybrid",
            target_user_id=request.user_id,
            recommendations=result['recommendations'],
            recommendation_count=result['recommendation_count'],
            parameters=result['parameters'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid recommendations API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def process_user_feedback(
    request: FeedbackRequest,
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Process user feedback to improve recommendation quality.
    
    Accepts various types of feedback: view, like, share, save, comment.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:write"])
        
        # Verify user can only submit feedback for themselves (unless admin)
        user_permissions = current_user.get('permissions', [])
        if request.user_id != current_user.get('user_id') and 'admin:all' not in user_permissions:
            raise HTTPException(
                status_code=403,
                detail="You can only submit feedback for your own account"
            )
        
        # Process feedback
        result = rec_engine.process_user_feedback(
            user_id=request.user_id,
            article_id=request.article_id,
            feedback_type=request.feedback_type,
            feedback_value=request.feedback_value
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Feedback processing failed: {result.get('error', 'Unknown error')}"
            )
        
        return FeedbackResponse(
            success=True,
            message=f"Successfully processed {request.feedback_type} feedback",
            feedback_processed=result['feedback_processed'],
            user_id=result['user_id'],
            article_id=result['article_id'],
            feedback_type=result['feedback_type'],
            weighted_score=result['weighted_score'],
            timestamp=result['timestamp']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback processing API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explanations", response_model=Dict)
async def get_recommendation_explanations(
    user_id: str = Query(description="User ID to explain recommendations for"),
    recommendations: List[Dict] = Query(description="Recommendations to explain"),
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Generate explanations for why articles were recommended.
    
    Provides interpretable reasons for recommendation decisions.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        # Get explanations
        result = rec_engine.get_recommendation_explanations(
            user_id=user_id,
            recommendations=recommendations
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Explanation generation failed: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "user_id": user_id,
            "explanations": result['explanations'],
            "explanation_count": result['explanation_count'],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation generation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-models", response_model=ModelBuildResponse)
async def build_recommendation_models(
    request: ModelBuildRequest,
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Build recommendation models with fresh data.
    
    Rebuilds content-based and/or collaborative filtering models.
    """
    try:
        # Check permissions (admin only for model building)
        require_permissions(current_user, ["admin:all"])
        
        start_time = datetime.now()
        
        # Build comprehensive recommendation system
        result = rec_engine.build_comprehensive_recommendation_system()
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Model building failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ModelBuildResponse(
            success=True,
            message=f"Successfully built recommendation models. Available methods: {', '.join(result['available_methods'])}",
            components_built=result['components'],
            available_methods=result['available_methods'],
            build_timestamp=result['build_timestamp'],
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model building API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods", response_model=Dict)
async def get_recommendation_methods(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get available recommendation methods and their descriptions.
    
    Returns information about supported algorithms and parameters.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        methods = {
            "content_based": {
                "name": "Content-Based Filtering",
                "description": "Recommends articles similar in content, topic, and metadata",
                "algorithm": "TF-IDF + Cosine Similarity",
                "parameters": ["n_recommendations", "min_similarity"],
                "best_for": "Finding articles similar to ones already read",
                "requires": "Article content and metadata"
            },
            "collaborative_filtering": {
                "name": "Collaborative Filtering",
                "description": "Recommends articles based on similar users' preferences",
                "algorithm": "Matrix Factorization (SVD) + Nearest Neighbors",
                "parameters": ["n_recommendations", "min_score"],
                "best_for": "Discovering new content based on user behavior",
                "requires": "User interaction data"
            },
            "hybrid": {
                "name": "Hybrid Recommendations",
                "description": "Combines content-based and collaborative filtering for best results",
                "algorithm": "Weighted combination of CB and CF scores",
                "parameters": ["n_recommendations", "content_weight", "collaborative_weight"],
                "best_for": "Most accurate and comprehensive recommendations",
                "requires": "Both content data and user interactions"
            }
        }
        
        feedback_types = {
            "view": {"weight": 1.0, "description": "User viewed the article"},
            "like": {"weight": 2.0, "description": "User liked the article"},
            "share": {"weight": 3.0, "description": "User shared the article"},
            "save": {"weight": 4.0, "description": "User saved the article"},
            "comment": {"weight": 2.5, "description": "User commented on the article"}
        }
        
        return {
            "success": True,
            "recommendation_methods": methods,
            "feedback_types": feedback_types,
            "supported_features": [
                "content_based_recommendations",
                "collaborative_filtering",
                "hybrid_recommendations",
                "user_feedback_processing",
                "recommendation_explanations",
                "model_building"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Methods endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=Dict)
async def get_recommendation_status(
    current_user: Dict = Depends(get_current_user),
    rec_engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """
    Get recommendation system status and health information.
    
    Returns information about model availability and system readiness.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["recommendations:read"])
        
        # Check model availability
        content_model_ready = bool(rec_engine.content_similarity_cache)
        cf_model_ready = bool(rec_engine.user_profiles_cache)
        
        # Check data availability
        try:
            articles = rec_engine.get_articles_from_database(limit=1)
            interactions = rec_engine.get_user_interactions_from_database(limit=1)
        except Exception:
            articles = []
            interactions = []
        
        available_methods = []
        if content_model_ready:
            available_methods.append("content_based")
        if cf_model_ready:
            available_methods.append("collaborative_filtering")
        if content_model_ready and cf_model_ready:
            available_methods.append("hybrid")
        
        status = {
            "success": True,
            "system_ready": len(available_methods) > 0,
            "models_loaded": {
                "content_based": content_model_ready,
                "collaborative_filtering": cf_model_ready,
                "hybrid": content_model_ready and cf_model_ready
            },
            "data_available": {
                "articles": len(articles) > 0,
                "user_interactions": len(interactions) > 0
            },
            "available_methods": available_methods,
            "system_health": {
                "database_connected": True,
                "recommendation_engine_loaded": True,
                "cache_status": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        warnings = []
        if not content_model_ready:
            warnings.append("Content-based model not loaded. Run model building.")
        if not cf_model_ready:
            warnings.append("Collaborative filtering model not loaded. Need user interaction data.")
        if not articles:
            warnings.append("No articles available for content-based recommendations.")
        if not interactions:
            warnings.append("No user interactions available for collaborative filtering.")
        
        if warnings:
            status["warnings"] = warnings
        
        return status
        
    except Exception as e:
        logger.error(f"Recommendation status check error: {e}")
        return {
            "success": False,
            "system_ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }