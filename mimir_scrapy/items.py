"""
Scrapy items for Mimir news articles
"""

import scrapy
from scrapy import Field


class MimirArticleItem(scrapy.Item):
    """Item representing a scraped news article"""
    
    # Core article fields
    url = Field()
    title = Field()
    content_summary = Field()
    full_content = Field()
    publication_date = Field()
    author = Field()
    
    # Source information
    source_website = Field()
    category = Field()
    
    # Scraping metadata
    scraper_type = Field()
    spider_name = Field()
    rendering_method = Field()
    scraped_at = Field()
    
    # Additional metadata
    metadata = Field()
    tags = Field()
    
    # Processing fields
    keyword_matches = Field()
    relevance_score = Field()
    duplicate_check = Field()