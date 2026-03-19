"""
MCP Client for Hacker News integration.
Provides asynchronous access to Hacker News data.
"""

import asyncio
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
import logging

# Fix the import - use relative import or direct logger setup
import logging
logger = logging.getLogger(__name__)

class HackerNewsMCPClient:
    """
    Client for interacting with Hacker News MCP server.
    Provides methods to fetch stories, comments, and user data.
    """
    
    def __init__(self, command: str = "npx", args: list = None):
        self.command = command
        self.args = args or ["-y", "@modelcontextprotocol/server-hackernews"]
        self._process = None
        self._initialized = False
        self.base_url = "https://hacker-news.firebaseio.com/v0"
        self.client = None
    
    async def initialize(self):
        """Initialize HTTP client"""
        if self._initialized:
            return
        
        try:
            self.client = httpx.AsyncClient(timeout=30.0)
            self._initialized = True
            logger.info("Hacker News MCP client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            raise
    
    async def get_top_stories(self, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Fetch top stories from Hacker News.
        
        Args:
            limit: Maximum number of stories to fetch
        
        Returns:
            List of story objects with metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Fetch story IDs
            response = await self.client.get(f"{self.base_url}/topstories.json")
            story_ids = response.json()[:limit]
            
            # Fetch story details in parallel
            stories = await asyncio.gather(*[
                self.get_item(story_id) for story_id in story_ids
            ])
            
            # Filter out None values and add computed fields
            valid_stories = []
            for story in stories:
                if story and story.get("title") and story.get("url"):
                    # Add computed fields
                    story["domain"] = self._extract_domain(story.get("url", ""))
                    story["age_hours"] = self._calculate_age(story.get("time", 0))
                    story["engagement_score"] = (
                        story.get("score", 0) + story.get("descendants", 0) * 2
                    )
                    valid_stories.append(story)
            
            logger.info(f"Fetched {len(valid_stories)} top stories")
            return valid_stories
            
        except Exception as e:
            logger.error(f"Failed to fetch top stories: {e}")
            return []
    
    async def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific item (story, comment, etc.) by ID.
        
        Args:
            item_id: Hacker News item ID
        
        Returns:
            Item object or None if not found
        """
        try:
            response = await self.client.get(f"{self.base_url}/item/{item_id}.json")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch item {item_id}: {e}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            return domain.replace("www.", "")
        except:
            return "unknown"
    
    def _calculate_age(self, timestamp: int) -> float:
        """Calculate age in hours from Unix timestamp"""
        import time
        return (time.time() - timestamp) / 3600
    
    async def close(self):
        """Close client connections"""
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()
        logger.info("MCP client closed")