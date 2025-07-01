"""
Named Entity Recognition and Extraction module for Mimir.
Handles company, person, location, and product extraction with normalization.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)

# Constants
MIN_ENTITY_LENGTH = 2
MAX_ENTITY_LENGTH = 100
DEFAULT_CONFIDENCE_SCORE = 1.0
MIN_CONFIDENCE_THRESHOLD = 0.5
TOP_ENTITIES_LIMIT = 5
OVERLAP_THRESHOLD = 0.8
DEFAULT_LANGUAGE = 'en'
PHONE_CONFIDENCE_SCORE = 0.8
TICKER_CONFIDENCE_SCORE = 0.9
PATTERN_CONFIDENCE_SCORE = 1.0

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    text: str
    normalized: str
    type: str
    start: int
    end: int
    confidence: float = DEFAULT_CONFIDENCE_SCORE


class EntityExtractor:
    """Extract and normalize named entities from text."""
    
    # Class-level compiled regex patterns
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    URL_PATTERN = re.compile(
        r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
    )
    PHONE_PATTERN = re.compile(
        r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
    )
    TICKER_PATTERN = re.compile(
        r'\b(?:NYSE|NASDAQ|LSE|TSE|ASX|BSE|NSE|HKEX):\s*([A-Z]{1,5})\b|\$([A-Z]{1,5})\b'
    )
    
    def __init__(self):
        """Initialize entity extractor with language models."""
        self.models: Dict[str, Language] = {}
        self._load_models()
        
        # Entity type mappings for standardization
        self.entity_type_map = {
            'ORG': 'organization',
            'PERSON': 'person', 
            'PER': 'person',
            'GPE': 'location',  # Geopolitical entity
            'LOC': 'location',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'FAC': 'facility',  # Buildings, airports, highways, etc.
            'MONEY': 'money',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percentage',
            'QUANTITY': 'quantity'
        }
        
        # Common company suffixes for normalization
        self.company_suffixes = {
            'inc', 'inc.', 'incorporated', 'corp', 'corp.', 'corporation',
            'llc', 'l.l.c.', 'ltd', 'ltd.', 'limited', 'plc', 'p.l.c.',
            'gmbh', 'ag', 'sa', 's.a.', 'bv', 'b.v.', 'nv', 'n.v.',
            'co', 'co.', 'company', 'group', 'holdings', 'international'
        }
        
        # Technology and product indicators
        self.tech_indicators = {
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'deep learning', 'neural network', 'blockchain', 'crypto',
            'software', 'hardware', 'platform', 'system', 'technology',
            'cloud', 'saas', 'paas', 'iaas', 'api', 'sdk'
        }
    
    def _load_models(self) -> None:
        """Load spaCy models for supported languages."""
        model_configs = {
            'en': 'en_core_web_md',
            'nl': 'nl_core_news_md',
            'de': 'de_core_news_md',
            'fr': 'fr_core_news_md',
            'multi': 'xx_ent_wiki_sm'  # Multilingual NER
        }
        
        for lang, model_name in model_configs.items():
            try:
                self.models[lang] = spacy.load(model_name)
                logger.info(f"Loaded {lang} NER model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load {lang} model {model_name}: {e}")
    
    def extract_entities(self, text: str, language: str = DEFAULT_LANGUAGE) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Dictionary of entity types to list of entity objects
        """
        if not text:
            return {}
        
        # Select appropriate model
        nlp = self.models.get(language, self.models.get('en'))
        if not nlp:
            logger.error(f"No model available for language: {language}")
            return {}
        
        try:
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract entities by type
            entities = defaultdict(list)
            seen_entities = defaultdict(set)  # For deduplication
            
            # Extract spaCy entities
            for ent in doc.ents:
                entity_type = self.entity_type_map.get(ent.label_, ent.label_.lower())
                normalized_text = self._normalize_entity(ent.text, entity_type)
                
                # Skip if already seen
                entity_key = (entity_type, normalized_text.lower())
                if entity_key in seen_entities[entity_type]:
                    continue
                
                seen_entities[entity_type].add(entity_key)
                
                entity_obj = {
                    'text': ent.text,
                    'normalized': normalized_text,
                    'type': entity_type,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': self._calculate_confidence(ent, doc)
                }
                
                entities[entity_type].append(entity_obj)
            
            # Extract pattern-based entities
            pattern_entities = self._extract_pattern_entities(text)
            for entity_type, entity_list in pattern_entities.items():
                entities[entity_type].extend(entity_list)
            
            # Post-process and enrich entities
            entities = self._post_process_entities(entities, doc)
            
            return dict(entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    def _normalize_entity(self, text: str, entity_type: str) -> str:
        """
        Normalize entity text based on type.
        
        Args:
            text: Raw entity text
            entity_type: Type of entity
            
        Returns:
            Normalized entity text
        """
        # Basic normalization
        normalized = text.strip()
        
        if entity_type == 'organization':
            # Remove common suffixes
            words = normalized.split()
            if words and words[-1].lower() in self.company_suffixes:
                normalized = ' '.join(words[:-1])
            
            # Standardize common abbreviations
            normalized = normalized.replace('&', 'and')
            
        elif entity_type == 'person':
            # Ensure proper capitalization
            parts = normalized.split()
            normalized = ' '.join(word.capitalize() for word in parts)
            
        elif entity_type == 'location':
            # Standardize country names, cities, etc.
            # This could be expanded with a comprehensive mapping
            location_map = {
                'US': 'United States',
                'USA': 'United States',
                'UK': 'United Kingdom',
                'NL': 'Netherlands',
                'DE': 'Germany',
                'FR': 'France'
            }
            normalized = location_map.get(normalized.upper(), normalized)
        
        return normalized
    
    def _calculate_confidence(self, entity: Span, doc: Doc) -> float:
        """
        Calculate confidence score for an entity.
        
        Args:
            entity: spaCy entity span
            doc: spaCy document
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from model (if available)
        confidence = 0.7  # Default confidence
        
        # Adjust based on entity characteristics
        if len(entity.text.split()) > 1:
            confidence += 0.1  # Multi-word entities often more reliable
        
        if entity.text[0].isupper():
            confidence += 0.05  # Proper capitalization
        
        # Check if entity appears multiple times
        occurrences = sum(1 for ent in doc.ents if ent.text == entity.text)
        if occurrences > 1:
            confidence += min(0.15, occurrences * 0.05)
        
        return min(1.0, confidence)
    
    def _extract_pattern_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using regex patterns."""
        pattern_entities = defaultdict(list)
        
        # Extract emails
        for match in self.EMAIL_PATTERN.finditer(text):
            pattern_entities['email'].append({
                'text': match.group(),
                'normalized': match.group().lower(),
                'type': 'email',
                'start': match.start(),
                'end': match.end(),
                'confidence': PATTERN_CONFIDENCE_SCORE
            })
        
        # Extract URLs
        for match in self.URL_PATTERN.finditer(text):
            pattern_entities['url'].append({
                'text': match.group(),
                'normalized': match.group(),
                'type': 'url',
                'start': match.start(),
                'end': match.end(),
                'confidence': PATTERN_CONFIDENCE_SCORE
            })
        
        # Extract phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            pattern_entities['phone'].append({
                'text': match.group(),
                'normalized': match.group().replace(' ', '').replace('-', ''),
                'type': 'phone',
                'start': match.start(),
                'end': match.end(),
                'confidence': PHONE_CONFIDENCE_SCORE
            })
        
        # Extract stock tickers
        for match in self.TICKER_PATTERN.finditer(text):
            ticker = match.group(1) or match.group(2)
            if ticker:
                pattern_entities['ticker'].append({
                    'text': match.group(),
                    'normalized': ticker.upper(),
                    'type': 'ticker',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': TICKER_CONFIDENCE_SCORE
                })
        
        return dict(pattern_entities)
    
    def _post_process_entities(self, entities: Dict[str, List[Dict]], 
                             doc: Doc) -> Dict[str, List[Dict]]:
        """
        Post-process entities to improve quality.
        
        Args:
            entities: Raw extracted entities
            doc: spaCy document
            
        Returns:
            Processed entities
        """
        # Merge overlapping entities
        for entity_type in entities:
            entities[entity_type] = self._merge_overlapping_entities(
                entities[entity_type]
            )
        
        # Add context for organizations
        if 'organization' in entities:
            for org in entities['organization']:
                org['context'] = self._extract_entity_context(
                    doc, org['start'], org['end']
                )
                org['industry_hints'] = self._detect_industry_hints(
                    org['text'], org.get('context', '')
                )
        
        # Add roles for people
        if 'person' in entities:
            for person in entities['person']:
                person['role'] = self._extract_person_role(
                    doc, person['start'], person['end']
                )
        
        return entities
    
    def _merge_overlapping_entities(self, entity_list: List[Dict]) -> List[Dict]:
        """Merge overlapping or duplicate entities."""
        if not entity_list:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entity_list, key=lambda x: x.get('start', 0))
        merged = [sorted_entities[0]]
        
        for entity in sorted_entities[1:]:
            last_entity = merged[-1]
            
            # Check for overlap or same normalized text
            if (entity.get('start', 0) < last_entity.get('end', 0) or 
                entity.get('normalized', '').lower() == last_entity.get('normalized', '').lower()):
                # Merge: keep the one with higher confidence
                if entity.get('confidence', 0) > last_entity.get('confidence', 0):
                    merged[-1] = entity
            else:
                merged.append(entity)
        
        return merged
    
    def _extract_entity_context(self, doc: Doc, start: int, end: int, 
                               window: int = 50) -> str:
        """Extract context around an entity."""
        context_start = max(0, start - window)
        context_end = min(len(doc.text), end + window)
        return doc.text[context_start:context_end]
    
    def _detect_industry_hints(self, org_name: str, context: str) -> List[str]:
        """Detect potential industry based on organization name and context."""
        hints = []
        combined_text = f"{org_name} {context}".lower()
        
        # Technology indicators
        if any(indicator in combined_text for indicator in self.tech_indicators):
            hints.append('technology')
        
        # Financial indicators
        financial_terms = {'bank', 'financial', 'investment', 'capital', 'fund'}
        if any(term in combined_text for term in financial_terms):
            hints.append('finance')
        
        # Healthcare indicators
        healthcare_terms = {'health', 'medical', 'pharma', 'hospital', 'clinic'}
        if any(term in combined_text for term in healthcare_terms):
            hints.append('healthcare')
        
        # Retail indicators
        retail_terms = {'retail', 'store', 'shop', 'market', 'commerce'}
        if any(term in combined_text for term in retail_terms):
            hints.append('retail')
        
        return hints
    
    def _extract_person_role(self, doc: Doc, start: int, end: int) -> Optional[str]:
        """Extract role/title for a person entity."""
        # Look for common title patterns before the name
        before_text = doc.text[max(0, start-100):start].lower()
        
        # Common titles
        titles = {
            'ceo': 'Chief Executive Officer',
            'cto': 'Chief Technology Officer',
            'cfo': 'Chief Financial Officer',
            'president': 'President',
            'vp': 'Vice President',
            'director': 'Director',
            'manager': 'Manager',
            'founder': 'Founder',
            'chairman': 'Chairman',
            'analyst': 'Analyst'
        }
        
        for abbr, full_title in titles.items():
            if abbr in before_text:
                return full_title
        
        return None
    
    def extract_relationships(self, entities: Dict[str, List[Dict]], 
                            text: str) -> List[Dict]:
        """
        Extract relationships between entities.
        
        Args:
            entities: Extracted entities
            text: Original text
            
        Returns:
            List of relationship objects
        """
        relationships = []
        
        # Extract person-organization relationships
        if 'person' in entities and 'organization' in entities:
            for person in entities['person']:
                for org in entities['organization']:
                    # Check if they appear near each other
                    distance = abs(person['start'] - org['end'])
                    if distance < 100:  # Within ~100 characters
                        rel_type = 'affiliated_with'
                        if person.get('role'):
                            rel_type = 'works_for'
                        
                        relationships.append({
                            'subject': person['normalized'],
                            'subject_type': 'person',
                            'relation': rel_type,
                            'object': org['normalized'],
                            'object_type': 'organization',
                            'confidence': 0.7
                        })
        
        # Extract organization-location relationships
        if 'organization' in entities and 'location' in entities:
            for org in entities['organization']:
                for loc in entities['location']:
                    # Check proximity
                    if abs(org['start'] - loc['end']) < 50:
                        relationships.append({
                            'subject': org['normalized'],
                            'subject_type': 'organization',
                            'relation': 'located_in',
                            'object': loc['normalized'],
                            'object_type': 'location',
                            'confidence': 0.6
                        })
        
        return relationships


def extract_entities_from_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities from a news article.
    
    Args:
        article: Article dictionary with 'content' and optional 'title'
        
    Returns:
        Dictionary with extracted entities and relationships
    """
    extractor = EntityExtractor()
    
    # Combine title and content
    title = article.get('title', '')
    content = article.get('content', '')
    full_text = f"{title}\n\n{content}"
    
    # Get language from preprocessing if available
    language = 'en'
    if 'preprocessed' in article:
        language = article['preprocessed'].get('language', 'en')
    
    # Extract entities
    entities = extractor.extract_entities(full_text, language)
    
    # Extract relationships
    relationships = extractor.extract_relationships(entities, full_text)
    
    # Calculate entity statistics
    stats = {
        'total_entities': sum(len(ents) for ents in entities.values()),
        'entity_types': list(entities.keys()),
        'top_entities': _get_top_entities(entities)
    }
    
    return {
        'entities': entities,
        'relationships': relationships,
        'statistics': stats
    }


def _get_top_entities(
    entities: Dict[str,
    List[Dict[str,
    Any]]],
    top_n: int = TOP_ENTITIES_LIMIT
)) -> Dict[str, List[str]]:
    """Get most frequent entities by type."""
    top_entities = {}
    
    for entity_type, entity_list in entities.items():
        # Count occurrences
        counter = Counter(ent['normalized'] for ent in entity_list)
        top_entities[entity_type] = [
            {'entity': name, 'count': count}
            for name, count in counter.most_common(top_n)
        ]
    
    return top_entities


if __name__ == "__main__":
    # Test the entity extractor
    logging.basicConfig(level=logging.INFO)
    
    test_text = """
    Apple Inc. CEO Tim Cook announced today in Cupertino that the company 
    will invest $1 billion in artificial intelligence research. The tech giant, 
    traded as AAPL on NASDAQ, plans to open a new AI research facility in 
    Amsterdam, Netherlands by Q3 2024.
    
    Dr. Sarah Johnson, the company's Chief AI Officer, stated: "This investment 
    demonstrates our commitment to advancing AI technology responsibly."
    
    For more information, contact press@apple.com or visit https://apple.com/ai
    """
    
    extractor = EntityExtractor()
    entities = extractor.extract_entities(test_text)
    
    print("Extracted Entities:")
    for entity_type, entity_list in entities.items():
        print(f"\n{entity_type.upper()}:")
        for entity in entity_list:
            print(f"  - {entity['normalized']} (confidence: {entity['confidence']:.2f})")
    
    # Test relationship extraction
    relationships = extractor.extract_relationships(entities, test_text)
    print("\nExtracted Relationships:")
    for rel in relationships:
        print(f"  - {rel['subject']} {rel['relation']} {rel['object']}")