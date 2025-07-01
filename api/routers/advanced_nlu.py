"""
Advanced NLU API router for Mimir - EPIC 8.4
Provides REST endpoints for question answering and fact extraction.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from api.dependencies import get_current_user, verify_api_key, get_db_connection
from auth.permissions import require_permission
from advanced_nlu import AdvancedNLUEngine
import db_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advanced-nlu", tags=["advanced-nlu"])


# Pydantic models for request/response
class QuestionAnsweringRequest(BaseModel):
    """Request model for question answering."""
    
    question: str = Field(..., description="The question to answer")
    context: Optional[str] = Field(None, description="Specific context to use for answering")
    article_ids: Optional[List[str]] = Field(None, description="Specific article IDs to search in")
    search_articles: bool = Field(True, description="Whether to search article database for context")
    model_override: Optional[str] = Field(None, description="Specific QA model to use")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Question must be at least 5 characters long")
        return v.strip()
    
    @validator('article_ids')
    def validate_article_ids(cls, v):
        if v and len(v) > 10:
            raise ValueError("Maximum 10 article IDs allowed")
        return v


class FactExtractionRequest(BaseModel):
    """Request model for fact extraction."""
    
    text: str = Field(..., description="Text to extract facts from")
    article_id: Optional[str] = Field(None, description="Article ID if extracting from stored article")
    fact_types: Optional[List[str]] = Field(None, description="Types of facts to focus on")
    include_verification: bool = Field(True, description="Whether to verify extracted facts")
    
    @validator('text')
    def validate_text(cls, v, values):
        if not values.get('article_id') and (not v or len(v.strip()) < 10):
            raise ValueError("Text must be at least 10 characters long if no article_id provided")
        return v
    
    @validator('fact_types')
    def validate_fact_types(cls, v):
        allowed_types = ['financial', 'temporal', 'corporate', 'regulatory']
        if v:
            invalid_types = [t for t in v if t not in allowed_types]
            if invalid_types:
                raise ValueError(f"Invalid fact types: {invalid_types}. Allowed: {allowed_types}")
        return v


class ClaimDetectionRequest(BaseModel):
    """Request model for claim detection."""
    
    text: str = Field(..., description="Text to analyze for claims")
    article_id: Optional[str] = Field(None, description="Article ID if analyzing stored article")
    verify_claims: bool = Field(True, description="Whether to attempt claim verification")
    
    @validator('text')
    def validate_text(cls, v, values):
        if not values.get('article_id') and (not v or len(v.strip()) < 20):
            raise ValueError("Text must be at least 20 characters long if no article_id provided")
        return v


class QuestionAnsweringResponse(BaseModel):
    """Response model for question answering."""
    
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Answer confidence score (0-1)")
    quality_score: float = Field(..., description="Answer quality score (0-1)")
    question_type: str = Field(..., description="Detected question type")
    is_confident: bool = Field(..., description="Whether the answer meets confidence threshold")
    context_used: int = Field(..., description="Length of context used")
    source_info: Dict[str, Any] = Field(..., description="Information about answer source")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Answer generation timestamp")
    success: bool = Field(..., description="Whether answer generation succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class FactExtractionResponse(BaseModel):
    """Response model for fact extraction."""
    
    facts: List[Dict[str, Any]] = Field(..., description="List of extracted facts")
    facts_by_type: Dict[str, List[Dict[str, Any]]] = Field(..., description="Facts organized by type")
    total_facts: int = Field(..., description="Total number of facts extracted")
    fact_types_found: List[str] = Field(..., description="Types of facts found")
    text_length: int = Field(..., description="Length of analyzed text")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Extraction timestamp")
    success: bool = Field(..., description="Whether extraction succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class ClaimDetectionResponse(BaseModel):
    """Response model for claim detection."""
    
    claims: List[Dict[str, Any]] = Field(..., description="List of detected claims")
    total_claims: int = Field(..., description="Total number of claims detected")
    high_confidence_claims: int = Field(..., description="Number of high confidence claims")
    medium_confidence_claims: int = Field(..., description="Number of medium confidence claims")
    low_confidence_claims: int = Field(..., description="Number of low confidence claims")
    verification_performed: bool = Field(..., description="Whether verification was performed")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Detection timestamp")
    success: bool = Field(..., description="Whether detection succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchQARequest(BaseModel):
    """Request model for batch question answering."""
    
    questions: List[str] = Field(..., description="List of questions to answer")
    shared_context: Optional[str] = Field(None, description="Shared context for all questions")
    article_ids: Optional[List[str]] = Field(None, description="Article IDs to use as context")
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question must be provided")
        if len(v) > 20:
            raise ValueError("Maximum 20 questions per batch")
        
        for i, question in enumerate(v):
            if not question or len(question.strip()) < 5:
                raise ValueError(f"Question {i+1} must be at least 5 characters long")
        
        return v


# Initialize Advanced NLU engine
nlu_engine = AdvancedNLUEngine()


@router.post("/question-answering", response_model=QuestionAnsweringResponse)
async def answer_question(
    request: QuestionAnsweringRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Answer a question using available context or article database.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get context from article if article_id provided but no context
        context = request.context
        if not context and request.article_ids:
            context_parts = []
            for article_id in request.article_ids:
                article = db_adapter.get_article_by_id(article_id)
                if article:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    context_parts.append(f"{title}\n{content}")
            context = "\n\n".join(context_parts)
        
        # Answer the question
        result = nlu_engine.answer_question(
            question=request.question,
            context=context,
            article_ids=request.article_ids,
            search_articles=request.search_articles
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Question answering failed')
            )
        
        return QuestionAnsweringResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        raise HTTPException(status_code=500, detail="Internal question answering error")


@router.post("/fact-extraction", response_model=FactExtractionResponse)
async def extract_facts(
    request: FactExtractionRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Extract facts from text using multiple strategies.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get text from article if article_id provided
        text = request.text
        if request.article_id:
            article = db_adapter.get_article_by_id(request.article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            text = article.get('content', '') or article.get('title', '')
            if not text:
                raise HTTPException(status_code=400, detail="Article has no content")
        
        # Extract facts
        result = nlu_engine.extract_facts(
            text=text,
            fact_types=request.fact_types,
            include_verification=request.include_verification
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Fact extraction failed')
            )
        
        return FactExtractionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fact extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal fact extraction error")


@router.post("/claim-detection", response_model=ClaimDetectionResponse)
async def detect_claims(
    request: ClaimDetectionRequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Detect claims in text and optionally verify them.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get text from article if article_id provided
        text = request.text
        if request.article_id:
            article = db_adapter.get_article_by_id(request.article_id)
            if not article:
                raise HTTPException(status_code=404, detail="Article not found")
            text = article.get('content', '') or article.get('title', '')
            if not text:
                raise HTTPException(status_code=400, detail="Article has no content")
        
        # Detect claims
        result = nlu_engine.detect_claims(
            text=text,
            verify_claims=request.verify_claims
        )
        
        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Claim detection failed')
            )
        
        return ClaimDetectionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Claim detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal claim detection error")


@router.post("/batch-qa")
async def batch_question_answering(
    request: BatchQARequest,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Answer multiple questions in batch.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get shared context from articles if provided
        shared_context = request.shared_context
        if not shared_context and request.article_ids:
            context_parts = []
            for article_id in request.article_ids:
                article = db_adapter.get_article_by_id(article_id)
                if article:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    context_parts.append(f"{title}\n{content}")
            shared_context = "\n\n".join(context_parts)
        
        # Process each question
        results = []
        for i, question in enumerate(request.questions):
            try:
                result = nlu_engine.answer_question(
                    question=question,
                    context=shared_context,
                    article_ids=request.article_ids,
                    search_articles=not bool(shared_context or request.article_ids)
                )
                
                result['question_index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to answer question {i}: {e}")
                results.append({
                    'question_index': i,
                    'question': question,
                    'answer': '',
                    'confidence': 0.0,
                    'quality_score': 0.0,
                    'question_type': 'error',
                    'is_confident': False,
                    'context_used': 0,
                    'source_info': {},
                    'processing_time_seconds': 0.0,
                    'timestamp': '',
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r.get('success', False))
        failed = len(results) - successful
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success')) / successful if successful > 0 else 0
        total_time = sum(r.get('processing_time_seconds', 0) for r in results)
        
        return {
            'results': results,
            'total_questions': len(request.questions),
            'successful_answers': successful,
            'failed_answers': failed,
            'average_confidence': round(avg_confidence, 3),
            'total_processing_time': round(total_time, 3),
            'shared_context_used': bool(shared_context),
            'context_length': len(shared_context) if shared_context else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch question answering failed: {e}")
        raise HTTPException(status_code=500, detail="Internal batch QA error")


@router.get("/article/{article_id}/qa-summary")
async def get_article_qa_summary(
    article_id: str,
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get a comprehensive NLU analysis summary for an article.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        article = db_adapter.get_article_by_id(article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        content = article.get('content', '') or article.get('title', '')
        if not content:
            raise HTTPException(status_code=400, detail="Article has no content")
        
        # Extract facts
        facts_result = nlu_engine.extract_facts(text=content, include_verification=True)
        
        # Detect claims
        claims_result = nlu_engine.detect_claims(text=content, verify_claims=True)
        
        # Answer some standard questions about the article
        standard_questions = [
            "What is this article about?",
            "Who are the main entities mentioned?",
            "When did this happen?",
            "What are the key facts?"
        ]
        
        qa_results = []
        for question in standard_questions:
            qa_result = nlu_engine.answer_question(
                question=question,
                context=content,
                search_articles=False
            )
            if qa_result.get('success'):
                qa_results.append({
                    'question': question,
                    'answer': qa_result['answer'],
                    'confidence': qa_result['confidence']
                })
        
        return {
            'article_id': article_id,
            'article_title': article.get('title', 'Untitled'),
            'content_length': len(content),
            'facts_extracted': {
                'total_facts': facts_result.get('total_facts', 0),
                'facts_by_type': facts_result.get('facts_by_type', {}),
                'top_facts': facts_result.get('facts', [])[:5]  # Top 5 facts
            },
            'claims_detected': {
                'total_claims': claims_result.get('total_claims', 0),
                'high_confidence': claims_result.get('high_confidence_claims', 0),
                'medium_confidence': claims_result.get('medium_confidence_claims', 0),
                'low_confidence': claims_result.get('low_confidence_claims', 0)
            },
            'qa_analysis': qa_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Article QA summary failed: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis error")


@router.get("/capabilities")
async def get_nlu_capabilities():
    """
    Get information about Advanced NLU capabilities.
    
    No authentication required - public endpoint.
    """
    try:
        return {
            'question_answering': {
                'supported_question_types': [
                    'factual', 'analytical', 'temporal', 'numerical'
                ],
                'confidence_thresholds': {
                    'factual': 0.7,
                    'analytical': 0.6,
                    'temporal': 0.75,
                    'numerical': 0.8
                },
                'context_sources': [
                    'provided_text', 'article_database_search', 'specific_articles'
                ]
            },
            'fact_extraction': {
                'supported_fact_types': [
                    'financial', 'temporal', 'corporate', 'regulatory'
                ],
                'verification_enabled': True,
                'extraction_methods': [
                    'pattern_matching', 'named_entity_recognition', 'rule_based'
                ]
            },
            'claim_detection': {
                'confidence_levels': ['high', 'medium', 'low'],
                'verification_methods': [
                    'context_match', 'partial_match', 'evidence_checking'
                ],
                'claim_types': [
                    'general', 'attributed', 'assertive', 'uncertain'
                ]
            },
            'supported_models': [
                'distilbert-base-cased-distilled-squad',
                'bert-large-uncased-whole-word-masking-finetuned-squad',
                'roberta-base-squad2'
            ],
            'batch_processing': {
                'max_questions_per_batch': 20,
                'max_articles_per_context': 10
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get NLU capabilities: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving capabilities")


@router.get("/stats")
async def get_nlu_stats(
    current_user: Dict = Depends(get_current_user),
    api_key: Dict = Depends(verify_api_key)
):
    """
    Get Advanced NLU usage statistics.
    
    Requires 'read_articles' permission.
    """
    require_permission(current_user, "read_articles")
    
    try:
        # Get basic database stats
        total_articles = db_adapter.count_articles()
        
        return {
            'total_articles_available': total_articles,
            'nlu_capabilities': {
                'question_answering': 'active',
                'fact_extraction': 'active',
                'claim_detection': 'active',
                'batch_processing': 'active'
            },
            'supported_question_types': 4,
            'supported_fact_types': 4,
            'advanced_nlu_engine_status': 'active',
            'model_status': {
                'qa_models': 'loaded',
                'fact_extraction': 'ready',
                'claim_detection': 'ready'
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get NLU stats: {e}")
        raise HTTPException(status_code=500, detail="Internal error retrieving stats")