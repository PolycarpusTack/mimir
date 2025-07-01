"""
Entity linking module for Mimir.
Links extracted entities to Wikipedia and Wikidata knowledge bases.
"""

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
from SPARQLWrapper import JSON, SPARQLWrapper
from wikipedia import wikipedia

logger = logging.getLogger(__name__)


@dataclass
class EntityLink:
    """Represents a linked entity with knowledge base information."""

    entity_text: str
    entity_type: str
    wikipedia_url: Optional[str] = None
    wikidata_id: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = None
    categories: List[str] = None
    properties: Dict[str, Any] = None
    confidence: float = 0.0

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.categories is None:
            self.categories = []
        if self.properties is None:
            self.properties = {}


class EntityLinker:
    """Links entities to Wikipedia and Wikidata."""

    def __init__(self, language: str = "en", cache_size: int = 1000):
        """
        Initialize entity linker.

        Args:
            language: Language code for Wikipedia
            cache_size: Size of LRU cache for API calls
        """
        self.language = language
        self.cache_size = cache_size

        # Configure Wikipedia
        wikipedia.set_lang(language)
        wikipedia.set_rate_limiting(True)

        # Configure Wikidata SPARQL endpoint
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.sparql.setReturnFormat(JSON)

        # User agent for API requests
        self.headers = {"User-Agent": "Mimir-NewsAnalyzer/1.0 (https://github.com/mimir-news)"}

        # Cache for API responses
        self._setup_cache()

    def _setup_cache(self):
        """Setup LRU cache for API calls."""
        # Create cached versions of methods
        self._search_wikipedia_cached = lru_cache(maxsize=self.cache_size)(self._search_wikipedia_uncached)
        self._get_wikidata_info_cached = lru_cache(maxsize=self.cache_size)(self._get_wikidata_info_uncached)

    def link_entity(self, entity_text: str, entity_type: str, context: Optional[str] = None) -> EntityLink:
        """
        Link an entity to Wikipedia and Wikidata.

        Args:
            entity_text: The entity text to link
            entity_type: Type of entity (organization, person, location, etc.)
            context: Optional context to help disambiguation

        Returns:
            EntityLink object with linked information
        """
        try:
            # Search Wikipedia
            wiki_results = self._search_wikipedia_cached(entity_text, entity_type)

            if not wiki_results:
                return EntityLink(entity_text=entity_text, entity_type=entity_type, confidence=0.0)

            # Get best match
            best_match = self._select_best_match(wiki_results, entity_text, entity_type, context)

            if not best_match:
                return EntityLink(entity_text=entity_text, entity_type=entity_type, confidence=0.1)

            # Get Wikipedia page info
            wiki_url = best_match.get("url")
            wiki_title = best_match.get("title")

            # Get Wikidata ID and additional info
            wikidata_info = self._get_wikidata_from_wikipedia(wiki_title)

            # Create entity link
            entity_link = EntityLink(
                entity_text=entity_text,
                entity_type=entity_type,
                wikipedia_url=wiki_url,
                wikidata_id=wikidata_info.get("wikidata_id"),
                description=best_match.get("summary", ""),
                aliases=wikidata_info.get("aliases", []),
                categories=best_match.get("categories", []),
                properties=wikidata_info.get("properties", {}),
                confidence=best_match.get("confidence", 0.5),
            )

            return entity_link

        except Exception as e:
            logger.error(f"Entity linking failed for '{entity_text}': {e}")
            return EntityLink(entity_text=entity_text, entity_type=entity_type, confidence=0.0)

    def _search_wikipedia_uncached(self, query: str, entity_type: str) -> List[Dict[str, Any]]:
        """Search Wikipedia for entity (uncached version)."""
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=5)

            if not search_results:
                return []

            results = []
            for title in search_results[:3]:  # Check top 3 results
                try:
                    # Get page summary
                    page = wikipedia.page(title)

                    # Basic type checking based on categories
                    categories = [cat.lower() for cat in page.categories[:10]]
                    type_match = self._check_type_match(categories, entity_type)

                    result = {
                        "title": page.title,
                        "url": page.url,
                        "summary": page.summary[:500],
                        "categories": page.categories[:20],
                        "type_match": type_match,
                        "confidence": 0.5 if type_match else 0.3,
                    }

                    results.append(result)

                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    for option in e.options[:2]:
                        try:
                            page = wikipedia.page(option)
                            results.append(
                                {
                                    "title": page.title,
                                    "url": page.url,
                                    "summary": page.summary[:500],
                                    "categories": page.categories[:20],
                                    "type_match": False,
                                    "confidence": 0.3,
                                }
                            )
                        except:
                            continue
                except Exception as e:
                    logger.debug(f"Error processing Wikipedia page '{title}': {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []

    def _check_type_match(self, categories: List[str], entity_type: str) -> bool:
        """Check if Wikipedia categories match expected entity type."""
        entity_type_lower = entity_type.lower()

        # Define category indicators for each entity type
        type_indicators = {
            "organization": ["companies", "organizations", "corporations", "businesses"],
            "person": ["people", "births", "deaths", "living people", "biography"],
            "location": ["geography", "places", "cities", "countries", "regions"],
            "product": ["products", "brands", "software", "technology"],
        }

        indicators = type_indicators.get(entity_type_lower, [])

        for category in categories:
            for indicator in indicators:
                if indicator in category:
                    return True

        return False

    def _select_best_match(
        self, candidates: List[Dict], entity_text: str, entity_type: str, context: Optional[str]
    ) -> Optional[Dict]:
        """Select best matching candidate from search results."""
        if not candidates:
            return None

        # Score each candidate
        scored_candidates = []

        for candidate in candidates:
            score = 0.0

            # Title similarity
            title_similarity = self._calculate_similarity(entity_text.lower(), candidate["title"].lower())
            score += title_similarity * 0.4

            # Type match
            if candidate.get("type_match"):
                score += 0.3

            # Context relevance
            if context and candidate.get("summary"):
                context_words = set(context.lower().split())
                summary_words = set(candidate["summary"].lower().split())
                overlap = len(context_words & summary_words)
                score += min(0.3, overlap * 0.05)

            candidate["final_score"] = score
            candidate["confidence"] = min(0.9, score)
            scored_candidates.append(candidate)

        # Sort by score
        scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        # Return best match if score is high enough
        best = scored_candidates[0]
        if best["final_score"] >= 0.3:
            return best

        return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard coefficient."""
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union)

    def _get_wikidata_from_wikipedia(self, wikipedia_title: str) -> Dict[str, Any]:
        """Get Wikidata information from Wikipedia title."""
        try:
            # Escape Wikipedia title to prevent SPARQL injection
            escaped_title = wikipedia_title.replace('"', '\\"').replace("\\", "\\\\")

            # Query Wikidata for Wikipedia page
            query = f"""
            SELECT ?item ?itemLabel ?itemDescription WHERE {{
              ?article schema:about ?item ;
                      schema:isPartOf <https://{self.language}.wikipedia.org/> ;
                      schema:name "{escaped_title}"@{self.language} .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language},en". }}
            }}
            LIMIT 1
            """

            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            if results["results"]["bindings"]:
                binding = results["results"]["bindings"][0]
                wikidata_id = binding["item"]["value"].split("/")[-1]

                # Get additional Wikidata info
                wikidata_info = self._get_wikidata_info_cached(wikidata_id)
                wikidata_info["wikidata_id"] = wikidata_id

                return wikidata_info

        except Exception as e:
            logger.debug(f"Wikidata lookup failed: {e}")

        return {}

    def _get_wikidata_info_uncached(self, wikidata_id: str) -> Dict[str, Any]:
        """Get detailed information from Wikidata (uncached version)."""
        try:
            # Validate wikidata ID format
            if not wikidata_id or not wikidata_id.startswith("Q") or not wikidata_id[1:].isdigit():
                return {"properties": {}, "aliases": []}

            # Query for entity properties
            query = f"""
            SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
              wd:{wikidata_id} ?property ?value .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language},en". }}
              FILTER(?property IN (
                wdt:P31,     # instance of
                wdt:P279,    # subclass of
                wdt:P17,     # country
                wdt:P159,    # headquarters location
                wdt:P571,    # inception date
                wdt:P1128,   # employees
                wdt:P2139,   # total revenue
                wdt:P452,    # industry
                wdt:P1454    # legal form
              ))
            }}
            LIMIT 20
            """

            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            properties = {}
            for binding in results["results"]["bindings"]:
                prop_label = binding.get("propertyLabel", {}).get("value", "")
                value_label = binding.get("valueLabel", {}).get("value", "")

                if prop_label and value_label:
                    if prop_label in properties:
                        if isinstance(properties[prop_label], list):
                            properties[prop_label].append(value_label)
                        else:
                            properties[prop_label] = [properties[prop_label], value_label]
                    else:
                        properties[prop_label] = value_label

            # Get aliases
            aliases = self._get_wikidata_aliases(wikidata_id)

            return {"properties": properties, "aliases": aliases}

        except Exception as e:
            logger.debug(f"Failed to get Wikidata info: {e}")
            return {"properties": {}, "aliases": []}

    def _get_wikidata_aliases(self, wikidata_id: str) -> List[str]:
        """Get aliases from Wikidata."""
        try:
            # Validate wikidata ID format (Q followed by numbers)
            if not wikidata_id or not wikidata_id.startswith("Q") or not wikidata_id[1:].isdigit():
                return []

            query = f"""
            SELECT ?alias WHERE {{
              wd:{wikidata_id} skos:altLabel ?alias .
              FILTER(LANG(?alias) = "{self.language}" || LANG(?alias) = "en")
            }}
            LIMIT 10
            """

            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            aliases = []
            for binding in results["results"]["bindings"]:
                alias = binding.get("alias", {}).get("value", "")
                if alias and alias not in aliases:
                    aliases.append(alias)

            return aliases

        except Exception as e:
            logger.debug(f"Failed to get aliases: {e}")
            return []

    def link_entities_batch(self, entities: List[Tuple[str, str]], context: Optional[str] = None) -> List[EntityLink]:
        """
        Link multiple entities in batch.

        Args:
            entities: List of (entity_text, entity_type) tuples
            context: Optional shared context

        Returns:
            List of EntityLink objects
        """
        linked_entities = []

        for entity_text, entity_type in entities:
            # Add small delay to respect rate limits
            time.sleep(0.1)

            linked = self.link_entity(entity_text, entity_type, context)
            linked_entities.append(linked)

        return linked_entities

    def create_knowledge_graph(self, linked_entities: List[EntityLink]) -> Dict[str, Any]:
        """
        Create a simple knowledge graph from linked entities.

        Args:
            linked_entities: List of linked entities

        Returns:
            Knowledge graph structure
        """
        nodes = []
        edges = []

        # Create nodes
        for entity in linked_entities:
            if entity.confidence > 0.3:
                node = {
                    "id": entity.wikidata_id or entity.entity_text,
                    "label": entity.entity_text,
                    "type": entity.entity_type,
                    "url": entity.wikipedia_url,
                    "properties": entity.properties,
                }
                nodes.append(node)

        # Create edges based on shared properties
        for i, entity1 in enumerate(linked_entities):
            for j, entity2 in enumerate(linked_entities[i + 1 :], i + 1):
                # Check for relationships
                if entity1.properties and entity2.properties:
                    # Same industry
                    ind1 = entity1.properties.get("industry")
                    ind2 = entity2.properties.get("industry")
                    if ind1 and ind2 and ind1 == ind2:
                        edges.append(
                            {
                                "source": entity1.wikidata_id or entity1.entity_text,
                                "target": entity2.wikidata_id or entity2.entity_text,
                                "type": "same_industry",
                                "label": f"Both in {ind1}",
                            }
                        )

                    # Same location
                    loc1 = entity1.properties.get("headquarters location", entity1.properties.get("country"))
                    loc2 = entity2.properties.get("headquarters location", entity2.properties.get("country"))
                    if loc1 and loc2 and loc1 == loc2:
                        edges.append(
                            {
                                "source": entity1.wikidata_id or entity1.entity_text,
                                "target": entity2.wikidata_id or entity2.entity_text,
                                "type": "same_location",
                                "label": f"Both in {loc1}",
                            }
                        )

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_entities": len(linked_entities),
                "linked_entities": len([e for e in linked_entities if e.wikidata_id]),
                "relationships": len(edges),
            },
        }


def link_article_entities(article: Dict[str, Any], entities: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Link entities from an article to knowledge bases.

    Args:
        article: Article dictionary
        entities: Extracted entities from the article

    Returns:
        Linked entities and knowledge graph
    """
    linker = EntityLinker()

    # Get article context
    context = f"{article.get('title', '')} {article.get('content', '')[:500]}"

    # Collect unique entities to link
    entities_to_link = []
    seen = set()

    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            entity_key = (entity["normalized"], entity_type)
            if entity_key not in seen:
                seen.add(entity_key)
                entities_to_link.append(entity_key)

    # Link entities
    linked_entities = linker.link_entities_batch(entities_to_link, context)

    # Create knowledge graph
    knowledge_graph = linker.create_knowledge_graph(linked_entities)

    # Organize results
    linked_by_type = {}
    for linked in linked_entities:
        if linked.entity_type not in linked_by_type:
            linked_by_type[linked.entity_type] = []

        linked_by_type[linked.entity_type].append(
            {
                "text": linked.entity_text,
                "wikipedia_url": linked.wikipedia_url,
                "wikidata_id": linked.wikidata_id,
                "description": linked.description,
                "confidence": linked.confidence,
                "properties": linked.properties,
            }
        )

    return {
        "linked_entities": linked_by_type,
        "knowledge_graph": knowledge_graph,
        "enrichment_stats": {
            "total_entities": len(entities_to_link),
            "successfully_linked": len([e for e in linked_entities if e.wikipedia_url]),
            "average_confidence": sum(e.confidence for e in linked_entities) / len(linked_entities)
            if linked_entities
            else 0,
        },
    }


if __name__ == "__main__":
    # Test entity linking
    logging.basicConfig(level=logging.INFO)

    # Test entities
    test_entities = [
        ("Apple Inc.", "organization"),
        ("Tim Cook", "person"),
        ("Cupertino", "location"),
        ("Microsoft", "organization"),
        ("Bill Gates", "person"),
        ("Seattle", "location"),
    ]

    linker = EntityLinker()

    print("Entity Linking Results:")
    print("-" * 50)

    for entity_text, entity_type in test_entities:
        result = linker.link_entity(entity_text, entity_type)

        print(f"\nEntity: {entity_text} ({entity_type})")
        print(f"Wikipedia: {result.wikipedia_url}")
        print(f"Wikidata ID: {result.wikidata_id}")
        print(f"Description: {result.description[:100]}..." if result.description else "N/A")
        print(f"Confidence: {result.confidence:.2f}")

        if result.properties:
            print("Properties:")
            for key, value in list(result.properties.items())[:3]:
                print(f"  - {key}: {value}")

    # Test knowledge graph creation
    linked = linker.link_entities_batch(test_entities)
    kg = linker.create_knowledge_graph(linked)

    print(f"\n\nKnowledge Graph Statistics:")
    print(f"Nodes: {len(kg['nodes'])}")
    print(f"Edges: {len(kg['edges'])}")

    if kg["edges"]:
        print("\nRelationships found:")
        for edge in kg["edges"]:
            print(f"  - {edge['source']} -> {edge['target']} ({edge['label']})")
