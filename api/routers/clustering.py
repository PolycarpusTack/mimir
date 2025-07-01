"""
Article Clustering API Router - EPIC 6 Completion
Provides endpoints for topic clustering, source clustering, and temporal clustering.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...article_clustering import ArticleClusteringEngine
from ...db_manager_postgres import PostgreSQLManager
from ..middleware.auth import get_current_user, require_permissions

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class ClusteringRequest(BaseModel):
    """Request model for clustering operations."""
    n_clusters: int = Field(default=10, ge=2, le=50, description="Number of clusters to create")
    method: str = Field(default="lda", description="Clustering method")
    limit: int = Field(default=1000, ge=10, le=5000, description="Maximum articles to analyze")
    source_filter: Optional[str] = Field(default=None, description="Filter by specific source")
    date_range_days: int = Field(default=30, ge=1, le=365, description="Date range in days")


class TopicClusteringRequest(ClusteringRequest):
    """Request model for topic clustering."""
    method: str = Field(default="lda", regex="^(lda|nmf|kmeans|dbscan)$", description="Topic clustering method")
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SourceClusteringRequest(BaseModel):
    """Request model for source clustering."""
    time_window_days: int = Field(default=30, ge=1, le=365, description="Analysis time window")
    limit: int = Field(default=1000, ge=10, le=5000, description="Maximum articles to analyze")


class TemporalClusteringRequest(BaseModel):
    """Request model for temporal clustering."""
    granularity: str = Field(default="daily", regex="^(hourly|daily|weekly)$", description="Time granularity")
    limit: int = Field(default=1000, ge=10, le=5000, description="Maximum articles to analyze")


class VisualizationRequest(BaseModel):
    """Request model for visualization generation."""
    visualization_type: str = Field(regex="^(topic|source|temporal)$", description="Type of visualization")
    clustering_result: Dict = Field(description="Clustering result to visualize")


class ClusteringResponse(BaseModel):
    """Response model for clustering operations."""
    success: bool
    message: str
    clustering_type: str
    n_clusters: int
    total_articles: int
    clusters: Dict
    quality_metrics: Dict
    processing_time_ms: float
    timestamp: str


class VisualizationResponse(BaseModel):
    """Response model for visualization data."""
    success: bool
    visualization_type: str
    data: Dict
    summary: Dict
    timestamp: str


def get_clustering_engine() -> ArticleClusteringEngine:
    """Dependency to get clustering engine instance."""
    return ArticleClusteringEngine()


@router.post("/topic", response_model=ClusteringResponse)
async def cluster_articles_by_topic(
    request: TopicClusteringRequest,
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Cluster articles by topic using various ML algorithms.
    
    Supports LDA, NMF, K-means, and DBSCAN clustering methods.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        start_time = datetime.now()
        
        # Get articles from database
        articles = clustering_engine.get_articles_from_database(
            limit=request.limit,
            source_filter=request.source_filter,
            date_range_days=request.date_range_days
        )
        
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No articles found for the specified criteria"
            )
        
        # Perform topic clustering
        result = clustering_engine.cluster_by_topic(
            articles=articles,
            n_clusters=request.n_clusters,
            method=request.method
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Topic clustering failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ClusteringResponse(
            success=True,
            message=f"Successfully clustered {len(articles)} articles into {result['n_clusters']} topic clusters using {request.method}",
            clustering_type="topic",
            n_clusters=result['n_clusters'],
            total_articles=result['total_articles'],
            clusters=result['clusters'],
            quality_metrics=result['quality_metrics'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Topic clustering API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/source", response_model=ClusteringResponse)
async def cluster_articles_by_source(
    request: SourceClusteringRequest,
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Cluster articles by source behavior and characteristics.
    
    Analyzes publishing patterns, content characteristics, and source diversity.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        start_time = datetime.now()
        
        # Get articles from database
        articles = clustering_engine.get_articles_from_database(
            limit=request.limit,
            date_range_days=request.time_window_days
        )
        
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No articles found for the specified time window"
            )
        
        # Perform source clustering
        result = clustering_engine.cluster_by_source(
            articles=articles,
            time_window_days=request.time_window_days
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Source clustering failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ClusteringResponse(
            success=True,
            message=f"Successfully clustered {result['total_sources']} sources into {result['n_clusters']} clusters",
            clustering_type="source",
            n_clusters=result['n_clusters'],
            total_articles=len(articles),
            clusters=result['clusters'],
            quality_metrics=result['quality_metrics'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Source clustering API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/temporal", response_model=ClusteringResponse)
async def cluster_articles_by_temporal_patterns(
    request: TemporalClusteringRequest,
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Cluster articles by temporal patterns and trends.
    
    Supports hourly, daily, and weekly granularity for pattern analysis.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        start_time = datetime.now()
        
        # Get articles from database
        articles = clustering_engine.get_articles_from_database(limit=request.limit)
        
        if not articles:
            raise HTTPException(
                status_code=404,
                detail="No articles found for temporal analysis"
            )
        
        # Perform temporal clustering
        result = clustering_engine.cluster_by_temporal_patterns(
            articles=articles,
            granularity=request.granularity
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Temporal clustering failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ClusteringResponse(
            success=True,
            message=f"Successfully identified {result['n_clusters']} temporal patterns with {request.granularity} granularity",
            clustering_type="temporal",
            n_clusters=result['n_clusters'],
            total_articles=len(articles),
            clusters=result['clusters'],
            quality_metrics=result['quality_metrics'],
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal clustering API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comprehensive", response_model=Dict)
async def comprehensive_clustering_analysis(
    limit: int = Query(default=1000, ge=10, le=5000, description="Maximum articles to analyze"),
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Perform comprehensive clustering analysis across all dimensions.
    
    Runs topic, source, and temporal clustering and provides unified insights.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        start_time = datetime.now()
        
        # Perform comprehensive analysis
        result = clustering_engine.analyze_all_patterns(limit=limit)
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Comprehensive analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Add API metadata
        result.update({
            'api_processing_time_ms': processing_time,
            'requested_by': current_user.get('user_id'),
            'api_timestamp': datetime.now().isoformat()
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive clustering API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualization", response_model=VisualizationResponse)
async def generate_clustering_visualization(
    request: VisualizationRequest,
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Generate visualization data for clustering results.
    
    Creates chart data and metadata for frontend visualization libraries.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        # Generate visualization
        result = clustering_engine.generate_cluster_visualization(
            clustering_result=request.clustering_result,
            visualization_type=request.visualization_type
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Visualization generation failed: {result.get('error', 'Unknown error')}"
            )
        
        return VisualizationResponse(
            success=True,
            visualization_type=result['visualization_type'],
            data=result['data'],
            summary=result['summary'],
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visualization generation API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/methods", response_model=Dict)
async def get_clustering_methods(
    current_user: Dict = Depends(get_current_user)
):
    """
    Get available clustering methods and their descriptions.
    
    Returns information about supported algorithms and parameters.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        methods = {
            "topic_clustering": {
                "lda": {
                    "name": "Latent Dirichlet Allocation",
                    "description": "Probabilistic topic modeling for discovering abstract topics",
                    "parameters": ["n_components", "random_state"],
                    "best_for": "General topic discovery with interpretable results"
                },
                "nmf": {
                    "name": "Non-negative Matrix Factorization",
                    "description": "Matrix decomposition for topic modeling with non-negative constraints",
                    "parameters": ["n_components", "random_state"],
                    "best_for": "Topic modeling with clearer topic separation"
                },
                "kmeans": {
                    "name": "K-Means Clustering",
                    "description": "Centroid-based clustering for grouping similar documents",
                    "parameters": ["n_clusters", "random_state"],
                    "best_for": "Fast clustering with predetermined number of clusters"
                },
                "dbscan": {
                    "name": "DBSCAN",
                    "description": "Density-based clustering that can find variable cluster sizes",
                    "parameters": ["eps", "min_samples"],
                    "best_for": "Finding clusters of varying densities and handling noise"
                }
            },
            "source_clustering": {
                "features": [
                    "articles_per_day",
                    "avg_title_length",
                    "avg_content_length",
                    "peak_hour",
                    "content_diversity"
                ],
                "algorithm": "K-Means with standardized features",
                "cluster_types": ["high_volume", "diverse_content", "long_form", "standard"]
            },
            "temporal_clustering": {
                "granularities": ["hourly", "daily", "weekly"],
                "features": ["article_count", "source_count", "content_characteristics"],
                "algorithm": "K-Means with time-based features",
                "pattern_types": ["high_activity", "diverse_sources", "standard"]
            }
        }
        
        return {
            "success": True,
            "clustering_methods": methods,
            "supported_visualizations": ["topic", "source", "temporal"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Methods endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=Dict)
async def get_clustering_status(
    current_user: Dict = Depends(get_current_user),
    clustering_engine: ArticleClusteringEngine = Depends(get_clustering_engine)
):
    """
    Get clustering system status and health information.
    
    Returns information about available data and system readiness.
    """
    try:
        # Check permissions
        require_permissions(current_user, ["analytics:read"])
        
        # Check data availability
        articles = clustering_engine.get_articles_from_database(limit=1)
        
        status = {
            "success": True,
            "system_ready": len(articles) > 0,
            "data_available": len(articles) > 0,
            "supported_methods": {
                "topic_clustering": True,
                "source_clustering": True,
                "temporal_clustering": True,
                "comprehensive_analysis": True,
                "visualization_generation": True
            },
            "system_health": {
                "database_connected": True,
                "clustering_engine_loaded": True,
                "cache_status": "active"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if not articles:
            status["warnings"] = ["No articles available for clustering analysis"]
        
        return status
        
    except Exception as e:
        logger.error(f"Clustering status check error: {e}")
        return {
            "success": False,
            "system_ready": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }