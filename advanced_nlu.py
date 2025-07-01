"""
Advanced Natural Language Understanding for Mimir - EPIC 8.4
Provides question answering and fact extraction capabilities.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from ai_optimization import ModelPool
import db_adapter

logger = logging.getLogger(__name__)


class AdvancedNLUEngine:
    """Advanced NLU engine for question answering and fact extraction."""
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        """
        Initialize the Advanced NLU engine.
        
        Args:
            model_pool: Optional ModelPool instance for model management
        """
        self.model_pool = model_pool or ModelPool()
        self.fact_extractor = FactExtractionEngine(model_pool)
        self.qa_validator = QAValidator()
        
        # Question types and their handling strategies
        self.question_types = {
            'factual': {
                'patterns': ['what is', 'who is', 'when did', 'where is', 'how many'],
                'strategy': 'extractive_qa',
                'confidence_threshold': 0.7
            },
            'analytical': {
                'patterns': ['why did', 'how does', 'what caused', 'what impact'],
                'strategy': 'context_analysis',
                'confidence_threshold': 0.6
            },
            'temporal': {
                'patterns': ['when', 'how long', 'since when', 'until when'],
                'strategy': 'temporal_extraction',
                'confidence_threshold': 0.75
            },
            'numerical': {
                'patterns': ['how much', 'how many', 'what percentage', 'what amount'],
                'strategy': 'numerical_extraction',
                'confidence_threshold': 0.8
            }
        }
        
        # Fact types for extraction
        self.fact_types = {
            'financial': {
                'patterns': [r'\$[\d,]+(?:\.\d+)?[BMK]?', r'\d+(?:\.\d+)?%', r'revenue', r'profit', r'loss'],
                'entities': ['MONEY', 'PERCENT']
            },
            'temporal': {
                'patterns': [r'\d{4}', r'Q[1-4]', r'january|february|march|april|may|june|july|august|september|october|november|december'],
                'entities': ['DATE', 'TIME']
            },
            'corporate': {
                'patterns': [r'CEO', r'merger', r'acquisition', r'IPO', r'shares'],
                'entities': ['ORG', 'PERSON']
            },
            'regulatory': {
                'patterns': [r'SEC', r'FDA', r'regulation', r'compliance', r'approved'],
                'entities': ['ORG', 'LAW']
            }
        }
    
    def get_qa_model(self, model_type: str = "distilbert-base-cased-distilled-squad"):
        """Get or load a question answering model."""
        return self.model_pool.get_transformer_model("question-answering", model_type)
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        article_ids: Optional[List[str]] = None,
        search_articles: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using available context or article database.
        
        Args:
            question: The question to answer
            context: Specific context to use (optional)
            article_ids: Specific article IDs to search in (optional)
            search_articles: Whether to search article database for context
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Determine question type
            question_type = self._classify_question(question)
            
            # Get context if not provided
            if not context:
                context = self._get_context_for_question(
                    question, article_ids, search_articles
                )
            
            if not context:
                return self._create_error_result("No relevant context found for the question")
            
            # Get appropriate QA model
            qa_model = self.get_qa_model()
            if not qa_model:
                return self._create_error_result("Failed to load question answering model")
            
            # Generate answer based on question type
            start_time = datetime.now()
            
            if question_type['strategy'] == 'extractive_qa':
                answer_result = self._extractive_qa(qa_model, question, context)
            elif question_type['strategy'] == 'context_analysis':
                answer_result = self._analytical_qa(question, context)
            elif question_type['strategy'] == 'temporal_extraction':
                answer_result = self._temporal_qa(question, context)
            elif question_type['strategy'] == 'numerical_extraction':
                answer_result = self._numerical_qa(question, context)
            else:
                answer_result = self._extractive_qa(qa_model, question, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Validate answer quality
            quality_score = self.qa_validator.validate_answer(
                question, answer_result.get('answer', ''), context
            )
            
            # Apply confidence threshold
            confidence = answer_result.get('confidence', 0.0)
            is_confident = confidence >= question_type['confidence_threshold']
            
            return {
                'question': question,
                'answer': answer_result.get('answer', ''),
                'confidence': confidence,
                'quality_score': quality_score,
                'question_type': question_type['strategy'],
                'is_confident': is_confident,
                'context_used': len(context),
                'source_info': answer_result.get('source_info', {}),
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return self._create_error_result(str(e))
    
    def extract_facts(
        self,
        text: str,
        fact_types: Optional[List[str]] = None,
        include_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Extract facts from text using multiple strategies.
        
        Args:
            text: Text to extract facts from
            fact_types: Types of facts to focus on (optional)
            include_verification: Whether to verify extracted facts
            
        Returns:
            Dictionary containing extracted facts and metadata
        """
        try:
            start_time = datetime.now()
            
            # Extract facts using multiple methods
            extracted_facts = self.fact_extractor.extract_all_facts(
                text, fact_types or list(self.fact_types.keys())
            )
            
            # Verify facts if requested
            if include_verification:
                verified_facts = []
                for fact in extracted_facts:
                    verification = self._verify_fact(fact, text)
                    fact['verification'] = verification
                    verified_facts.append(fact)
                extracted_facts = verified_facts
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Organize facts by type
            facts_by_type = {}
            for fact in extracted_facts:
                fact_type = fact.get('type', 'general')
                if fact_type not in facts_by_type:
                    facts_by_type[fact_type] = []
                facts_by_type[fact_type].append(fact)
            
            return {
                'facts': extracted_facts,
                'facts_by_type': facts_by_type,
                'total_facts': len(extracted_facts),
                'fact_types_found': list(facts_by_type.keys()),
                'text_length': len(text),
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return self._create_error_result(str(e))
    
    def detect_claims(
        self,
        text: str,
        verify_claims: bool = True
    ) -> Dict[str, Any]:
        """
        Detect claims in text and optionally verify them.
        
        Args:
            text: Text to analyze for claims
            verify_claims: Whether to attempt claim verification
            
        Returns:
            Dictionary containing detected claims and verification results
        """
        try:
            start_time = datetime.now()
            
            # Detect claims using pattern matching and NLP
            claims = self._detect_claims_in_text(text)
            
            # Verify claims if requested
            if verify_claims:
                for claim in claims:
                    verification = self._verify_claim(claim, text)
                    claim['verification'] = verification
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Categorize claims by confidence
            high_confidence = [c for c in claims if c.get('confidence', 0) > 0.8]
            medium_confidence = [c for c in claims if 0.5 < c.get('confidence', 0) <= 0.8]
            low_confidence = [c for c in claims if c.get('confidence', 0) <= 0.5]
            
            return {
                'claims': claims,
                'total_claims': len(claims),
                'high_confidence_claims': len(high_confidence),
                'medium_confidence_claims': len(medium_confidence),
                'low_confidence_claims': len(low_confidence),
                'verification_performed': verify_claims,
                'processing_time_seconds': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Claim detection failed: {e}")
            return self._create_error_result(str(e))
    
    def _classify_question(self, question: str) -> Dict[str, Any]:
        """Classify the type of question to determine handling strategy."""
        question_lower = question.lower()
        
        for q_type, config in self.question_types.items():
            for pattern in config['patterns']:
                if pattern in question_lower:
                    return {
                        'type': q_type,
                        'strategy': config['strategy'],
                        'confidence_threshold': config['confidence_threshold']
                    }
        
        # Default to factual question
        return {
            'type': 'factual',
            'strategy': 'extractive_qa',
            'confidence_threshold': 0.7
        }
    
    def _get_context_for_question(
        self,
        question: str,
        article_ids: Optional[List[str]] = None,
        search_articles: bool = True
    ) -> str:
        """Get relevant context for answering a question."""
        if article_ids:
            # Use specific articles
            context_parts = []
            for article_id in article_ids:
                article = db_adapter.get_article_by_id(article_id)
                if article:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    context_parts.append(f"{title}\n{content}")
            return "\n\n".join(context_parts)
        
        elif search_articles:
            # Search for relevant articles
            try:
                # Extract key terms from question for search
                search_terms = self._extract_search_terms(question)
                
                # Use existing search functionality
                articles = db_adapter.search_articles(search_terms, limit=5)
                
                context_parts = []
                for article in articles:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    # Limit content length for context
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    context_parts.append(f"{title}\n{content}")
                
                return "\n\n".join(context_parts)
                
            except Exception as e:
                logger.error(f"Failed to search articles for context: {e}")
                return ""
        
        return ""
    
    def _extract_search_terms(self, question: str) -> str:
        """Extract key terms from question for search."""
        # Remove question words and common words
        stop_words = {'what', 'who', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        search_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return ' '.join(search_terms[:5])  # Limit to 5 terms
    
    def _extractive_qa(self, qa_model, question: str, context: str) -> Dict[str, Any]:
        """Perform extractive question answering."""
        # Truncate context if too long
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        result = qa_model(question=question, context=context)
        
        return {
            'answer': result['answer'],
            'confidence': result['score'],
            'start_position': result.get('start', 0),
            'end_position': result.get('end', 0),
            'source_info': {
                'method': 'extractive_qa',
                'context_length': len(context)
            }
        }
    
    def _analytical_qa(self, question: str, context: str) -> Dict[str, Any]:
        """Handle analytical questions requiring reasoning."""
        # For analytical questions, look for causal relationships and explanations
        question_lower = question.lower()
        
        # Extract relevant sentences that might contain explanations
        sentences = context.split('.')
        relevant_sentences = []
        
        causal_indicators = ['because', 'due to', 'caused by', 'result of', 'led to', 'resulted in']
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in causal_indicators):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:3])  # Take top 3 relevant sentences
            confidence = 0.7
        else:
            # Fallback to first few sentences
            answer = '. '.join(sentences[:2]).strip()
            confidence = 0.4
        
        return {
            'answer': answer,
            'confidence': confidence,
            'source_info': {
                'method': 'analytical_reasoning',
                'relevant_sentences_found': len(relevant_sentences)
            }
        }
    
    def _temporal_qa(self, question: str, context: str) -> Dict[str, Any]:
        """Handle temporal questions about dates and times."""
        # Extract dates and temporal information
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
            r'\bQ[1-4]\s+\d{4}\b'  # Q1 2023
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            found_dates.extend(matches)
        
        if found_dates:
            # Return the most relevant date (simple heuristic: first one found)
            answer = found_dates[0]
            confidence = 0.8
        else:
            answer = "No specific date information found in the context."
            confidence = 0.2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'source_info': {
                'method': 'temporal_extraction',
                'dates_found': len(found_dates),
                'all_dates': found_dates[:5]  # Limit to 5 dates
            }
        }
    
    def _numerical_qa(self, question: str, context: str) -> Dict[str, Any]:
        """Handle numerical questions about quantities and amounts."""
        # Extract numerical information
        number_patterns = [
            r'\$[\d,]+(?:\.\d+)?[BMK]?',  # Money amounts
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Large numbers with commas
            r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b'  # Written out large numbers
        ]
        
        found_numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, context, re.IGNORECASE)
            found_numbers.extend(matches)
        
        if found_numbers:
            # Return the most relevant number (first one found)
            answer = found_numbers[0]
            confidence = 0.8
        else:
            answer = "No specific numerical information found in the context."
            confidence = 0.2
        
        return {
            'answer': answer,
            'confidence': confidence,
            'source_info': {
                'method': 'numerical_extraction',
                'numbers_found': len(found_numbers),
                'all_numbers': found_numbers[:5]  # Limit to 5 numbers
            }
        }
    
    def _detect_claims_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Detect claims in text using pattern matching and NLP."""
        claims = []
        
        # Claim indicators
        claim_patterns = [
            r'according to .+?,',
            r'.+ reported that .+',
            r'.+ announced .+',
            r'.+ stated .+',
            r'.+ claims? .+',
            r'research shows .+',
            r'studies indicate .+'
        ]
        
        sentences = text.split('.')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            confidence = 0.5  # Default confidence
            claim_type = 'general'
            
            # Check for claim patterns
            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    confidence += 0.2
                    claim_type = 'attributed'
                    break
            
            # Check for strong assertive language
            strong_indicators = ['will', 'must', 'always', 'never', 'definitely', 'certainly']
            if any(indicator in sentence.lower() for indicator in strong_indicators):
                confidence += 0.1
                claim_type = 'assertive'
            
            # Check for uncertainty indicators
            uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'allegedly']
            if any(indicator in sentence.lower() for indicator in uncertainty_indicators):
                confidence -= 0.2
                claim_type = 'uncertain'
            
            confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
            
            claims.append({
                'claim': sentence,
                'confidence': confidence,
                'type': claim_type,
                'position': i,
                'verification': {}
            })
        
        return claims
    
    def _verify_fact(self, fact: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Verify a fact against the source context."""
        fact_text = fact.get('text', '')
        
        # Simple verification based on context presence
        if fact_text.lower() in context.lower():
            return {
                'verified': True,
                'confidence': 0.9,
                'method': 'context_match',
                'details': 'Fact found directly in source text'
            }
        else:
            # Look for partial matches or synonyms
            fact_words = set(fact_text.lower().split())
            context_words = set(context.lower().split())
            
            overlap = len(fact_words & context_words) / len(fact_words) if fact_words else 0
            
            if overlap > 0.7:
                return {
                    'verified': True,
                    'confidence': 0.7,
                    'method': 'partial_match',
                    'details': f'High word overlap: {overlap:.2f}'
                }
            elif overlap > 0.4:
                return {
                    'verified': False,
                    'confidence': 0.5,
                    'method': 'weak_match',
                    'details': f'Moderate word overlap: {overlap:.2f}'
                }
            else:
                return {
                    'verified': False,
                    'confidence': 0.2,
                    'method': 'no_match',
                    'details': 'Fact not found in source context'
                }
    
    def _verify_claim(self, claim: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Verify a claim against available evidence."""
        claim_text = claim.get('claim', '')
        
        # For now, use similar logic to fact verification
        # In production, this would use more sophisticated claim verification
        return self._verify_fact({'text': claim_text}, context)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'question': '',
            'answer': '',
            'confidence': 0.0,
            'quality_score': 0.0,
            'question_type': 'error',
            'is_confident': False,
            'context_used': 0,
            'source_info': {},
            'processing_time_seconds': 0.0,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error_message
        }


class FactExtractionEngine:
    """Specialized engine for fact extraction from text."""
    
    def __init__(self, model_pool: Optional[ModelPool] = None):
        self.model_pool = model_pool or ModelPool()
    
    def extract_all_facts(self, text: str, fact_types: List[str]) -> List[Dict[str, Any]]:
        """Extract facts of specified types from text."""
        all_facts = []
        
        for fact_type in fact_types:
            facts = self._extract_facts_by_type(text, fact_type)
            all_facts.extend(facts)
        
        # Remove duplicates and rank by confidence
        unique_facts = self._deduplicate_facts(all_facts)
        ranked_facts = sorted(unique_facts, key=lambda x: x.get('confidence', 0), reverse=True)
        
        return ranked_facts
    
    def _extract_facts_by_type(self, text: str, fact_type: str) -> List[Dict[str, Any]]:
        """Extract facts of a specific type."""
        facts = []
        
        if fact_type == 'financial':
            facts.extend(self._extract_financial_facts(text))
        elif fact_type == 'temporal':
            facts.extend(self._extract_temporal_facts(text))
        elif fact_type == 'corporate':
            facts.extend(self._extract_corporate_facts(text))
        elif fact_type == 'regulatory':
            facts.extend(self._extract_regulatory_facts(text))
        
        return facts
    
    def _extract_financial_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial facts from text."""
        facts = []
        
        # Money amounts
        money_pattern = r'\$[\d,]+(?:\.\d+)?[BMK]?'
        money_matches = re.finditer(money_pattern, text)
        
        for match in money_matches:
            facts.append({
                'text': match.group(),
                'type': 'financial',
                'subtype': 'money_amount',
                'confidence': 0.9,
                'position': match.start(),
                'context': text[max(0, match.start()-50):match.end()+50]
            })
        
        # Percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        percent_matches = re.finditer(percent_pattern, text)
        
        for match in percent_matches:
            facts.append({
                'text': match.group(),
                'type': 'financial',
                'subtype': 'percentage',
                'confidence': 0.8,
                'position': match.start(),
                'context': text[max(0, match.start()-50):match.end()+50]
            })
        
        return facts
    
    def _extract_temporal_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal facts from text."""
        facts = []
        
        # Years
        year_pattern = r'\b(19|20)\d{2}\b'
        year_matches = re.finditer(year_pattern, text)
        
        for match in year_matches:
            facts.append({
                'text': match.group(),
                'type': 'temporal',
                'subtype': 'year',
                'confidence': 0.8,
                'position': match.start(),
                'context': text[max(0, match.start()-50):match.end()+50]
            })
        
        return facts
    
    def _extract_corporate_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract corporate facts from text."""
        facts = []
        
        # Corporate actions
        corporate_patterns = {
            'merger': r'merger|acquisition|acquired|bought',
            'ipo': r'IPO|initial public offering|went public',
            'leadership': r'CEO|CFO|CTO|president|chairman'
        }
        
        for action_type, pattern in corporate_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                facts.append({
                    'text': match.group(),
                    'type': 'corporate',
                    'subtype': action_type,
                    'confidence': 0.7,
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return facts
    
    def _extract_regulatory_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract regulatory facts from text."""
        facts = []
        
        # Regulatory bodies and actions
        regulatory_patterns = {
            'sec': r'SEC|Securities and Exchange Commission',
            'fda': r'FDA|Food and Drug Administration',
            'approval': r'approved|approval|authorized|cleared'
        }
        
        for reg_type, pattern in regulatory_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                facts.append({
                    'text': match.group(),
                    'type': 'regulatory',
                    'subtype': reg_type,
                    'confidence': 0.8,
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return facts
    
    def _deduplicate_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate facts."""
        seen_texts = set()
        unique_facts = []
        
        for fact in facts:
            fact_text = fact.get('text', '').lower()
            if fact_text not in seen_texts:
                seen_texts.add(fact_text)
                unique_facts.append(fact)
        
        return unique_facts


class QAValidator:
    """Validator for question answering quality."""
    
    def validate_answer(self, question: str, answer: str, context: str) -> float:
        """Validate the quality of a question-answer pair."""
        if not answer or answer.strip() == "":
            return 0.0
        
        scores = []
        
        # Relevance score
        relevance = self._calculate_relevance(question, answer)
        scores.append(relevance * 0.4)
        
        # Completeness score
        completeness = self._calculate_completeness(answer)
        scores.append(completeness * 0.3)
        
        # Context alignment score
        alignment = self._calculate_context_alignment(answer, context)
        scores.append(alignment * 0.3)
        
        final_score = sum(scores)
        return min(1.0, max(0.0, final_score))
    
    def _calculate_relevance(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        question_words -= stop_words
        answer_words -= stop_words
        
        if not question_words:
            return 0.5
        
        overlap = len(question_words & answer_words) / len(question_words)
        return min(1.0, overlap * 2)  # Scale appropriately
    
    def _calculate_completeness(self, answer: str) -> float:
        """Calculate how complete the answer appears."""
        if len(answer) < 10:
            return 0.3  # Very short answers are likely incomplete
        elif len(answer) < 50:
            return 0.7  # Short but potentially complete
        else:
            return 1.0  # Longer answers are likely more complete
    
    def _calculate_context_alignment(self, answer: str, context: str) -> float:
        """Calculate how well the answer aligns with the context."""
        if answer.lower() in context.lower():
            return 1.0  # Perfect alignment
        
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(answer_words & context_words) / len(answer_words)
        return overlap


# Extend ModelPool to support QA models
def _extend_model_pool_for_qa():
    """Extend ModelPool class to include question answering models."""
    original_get_transformer = ModelPool.get_transformer_model
    
    def get_transformer_model_extended(self, task: str, model_name: Optional[str] = None):
        """Extended transformer model getter with QA support."""
        if task == "question-answering":
            key = f"transformer_{task}_{model_name or 'default'}"
            
            with self.model_locks[key]:
                if key not in self.models:
                    logger.info(f"Loading QA model: {model_name}")
                    try:
                        model = model_name or "distilbert-base-cased-distilled-squad"
                        qa_pipeline = pipeline(
                            "question-answering",
                            model=model,
                            tokenizer=model,
                            device=0 if torch.cuda.is_available() else -1,
                            framework="pt"
                        )
                        self.models[key] = qa_pipeline
                    except Exception as e:
                        logger.error(f"Failed to load QA model {model_name}: {e}")
                        return None
                
                self.usage_counts[key] += 1
                self.last_used[key] = time.time()
                return self.models[key]
        else:
            return original_get_transformer(self, task, model_name)
    
    # Monkey patch the method
    ModelPool.get_transformer_model = get_transformer_model_extended


# Initialize the extension
_extend_model_pool_for_qa()