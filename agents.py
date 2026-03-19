"""
Agent definitions for the Hacker News trend analysis system.
Each agent has a specific role in the multi-agent workflow.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from typing import Dict, Any, List
import logging

# Fix imports - use relative imports or direct imports
from config import Config
from mcpclient import HackerNewsMCPClient

logger = logging.getLogger(__name__)

class NewsFetcherAgent(AssistantAgent):
    """
    Agent responsible for fetching news from Hacker News MCP server.
    Connects to MCP, retrieves top stories, and formats them for analysis.
    """
    
    def __init__(self, model_client: OpenAIChatCompletionClient, hn_client: HackerNewsMCPClient):
        super().__init__(
            name="NewsFetcher",
            model_client=model_client,
            system_message="""You are a news fetching specialist for Hacker News.
            Your ONLY job is to retrieve the latest technology and cybersecurity news.
            
            Use the Hacker News MCP client to:
            1. Fetch top stories from Hacker News
            2. Extract relevant metadata (title, url, points, comments, time)
            3. Filter for technology and cybersecurity content
            4. Format the data for the next agent
            
            Return ONLY the raw news data in a structured format.
            Do not analyze or summarize - just fetch and organize.
            
            Focus on stories with high engagement (points + comments).
            Prioritize cybersecurity and emerging technology topics.
            """,
            model_client_stream=True
        )
        self.hn_client = hn_client
        self.last_fetch_time = None
        self.cached_stories = []
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        """
        Handle incoming messages and fetch news data.
        
        Args:
            messages: List of incoming messages
            cancellation_token: Token for cancellation support
        
        Returns:
            Response with fetched news data
        """
        logger.info("NewsFetcherAgent: Fetching Hacker News stories")
        
        try:
            # Fetch top stories with limit from config
            config = Config()
            stories = await self.hn_client.get_top_stories(limit=config.system.story_limit)
            
            # Cache stories for potential recovery
            self.cached_stories = stories
            self.last_fetch_time = __import__('datetime').datetime.now()
            
            # Format response
            response = self._format_stories_response(stories)
            logger.info(f"NewsFetcherAgent: Successfully fetched {len(stories)} stories")
            
            return response
            
        except Exception as e:
            logger.error(f"NewsFetcherAgent: Error fetching stories: {e}")
            return f"Error fetching news: {str(e)}"
    
    def _format_stories_response(self, stories: List[Dict]) -> str:
        """Format stories for the next agent"""
        if not stories:
            return "No stories found."
        
        lines = ["📰 **HACKER NEWS TOP STORIES**", "=" * 50, ""]
        
        for i, story in enumerate(stories[:10], 1):  # Top 10 only
            lines.append(f"{i}. **{story.get('title', 'Untitled')}**")
            lines.append(f"   🔗 Domain: {story.get('domain', 'unknown')}")
            lines.append(f"   ⭐ Points: {story.get('score', 0)} | 💬 Comments: {story.get('descendants', 0)}")
            lines.append(f"   🕒 Age: {story.get('age_hours', 0):.1f} hours")
            lines.append(f"   📊 Engagement Score: {story.get('engagement_score', 0)}")
            if story.get('url'):
                lines.append(f"   🔍 URL: {story.get('url')}")
            lines.append("")
        
        return "\n".join(lines)


class TrendAnalyzerAgent(AssistantAgent):
    """
    Agent that analyzes news to identify key technology and cybersecurity trends.
    Extracts patterns, insights, and signals from the fetched news.
    """
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="TrendAnalyzer",
            model_client=model_client,
            system_message="""You are a technology trend analyst specializing in cybersecurity and emerging tech.
            
            Your task is to analyze Hacker News stories and identify:
            
            1. **Key Trends**: What patterns emerge? (e.g., AI security, cloud breaches, new vulnerabilities)
            2. **Important Signals**: What's gaining traction? (e.g., new tools, techniques, threats)
            3. **Industry Impact**: Which sectors are affected? (e.g., finance, healthcare, critical infrastructure)
            4. **Geopolitical Context**: Any regional or political implications?
            5. **Expert Insights**: What do the comments reveal about community sentiment?
            
            Provide a structured analysis with:
            - **Primary Trend**: The strongest signal (with evidence)
            - **Secondary Trends**: 2-3 related patterns
            - **Key Insights**: 3-5 important takeaways
            - **Sentiment Analysis**: Community reaction (positive/concerned/curious)
            
            Be specific, data-driven, and actionable.
            """,
            model_client_stream=True
        )
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        """
        Analyze news stories for trends.
        
        Args:
            messages: Contains the news data from NewsFetcher
            cancellation_token: Token for cancellation support
        
        Returns:
            Analysis of trends and insights
        """
        logger.info("TrendAnalyzerAgent: Analyzing trends")
        
        # Extract news data from messages
        news_data = messages[-1].content if messages else ""
        
        # The actual analysis happens in the LLM via the system message
        # We're just passing the data through with context
        analysis_prompt = f"""
        Analyze these Hacker News stories and provide trend analysis:
        
        {news_data}
        
        Focus on cybersecurity and technology trends. Identify patterns and signals.
        """
        
        # Return the prompt - the LLM will generate the analysis
        return analysis_prompt


class ContentWriterAgent(AssistantAgent):
    """
    Agent that converts trend analysis into engaging, informative summaries.
    Produces content suitable for tech audiences.
    """
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="ContentWriter",
            model_client=model_client,
            system_message="""You are a technology content writer specializing in crisp, engaging summaries.
            
            Transform trend analysis into compelling content with these characteristics:
            
            **Style Guidelines**:
            - Informative but accessible to technical audiences
            - Concise but knowledge-dense (every sentence adds value)
            - Engaging opening that hooks the reader
            - Clear structure with logical flow
            - Technical accuracy without unnecessary jargon
            
            **Format**:
            - Headline: Attention-grabbing but accurate
            - Opening paragraph: The big picture (what's happening and why it matters)
            - Key trends: 3-5 bullet points or short paragraphs
            - Insights: What this means for the industry
            - Closing: Forward-looking perspective
            
            **Tone**: Authoritative, insightful, slightly conversational
            **Length**: 400-600 words (perfect for LinkedIn/tech blog)
            
            Make every word count. Your audience are professionals who value substance over fluff.
            """,
            model_client_stream=True
        )
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        """
        Convert analysis into polished content.
        
        Args:
            messages: Contains trend analysis from TrendAnalyzer
            cancellation_token: Token for cancellation support
        
        Returns:
            Draft content for editor review
        """
        logger.info("ContentWriterAgent: Creating content draft")
        
        analysis = messages[-1].content if messages else ""
        
        content_prompt = f"""
        Create an engaging technology trend summary based on this analysis:
        
        {analysis}
        
        Follow the style guidelines in your system message.
        """
        
        return content_prompt


class EditorAgent(AssistantAgent):
    """
    Agent that reviews and polishes content for quality, accuracy, and readability.
    Final quality gate before output.
    """
    
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__(
            name="Editor",
            model_client=model_client,
            system_message="""You are a senior editor at a technology publication.
            
            Your role is to ensure content meets the highest standards:
            
            **Quality Checks**:
            ✓ **Accuracy**: Are technical claims correct and properly qualified?
            ✓ **Clarity**: Is the message clear and unambiguous?
            ✓ **Flow**: Does the content have logical progression?
            ✓ **Engagement**: Will it capture and hold reader attention?
            ✓ **Completeness**: Does it cover the key points without gaps?
            ✓ **Conciseness**: Is every sentence necessary? Remove fluff.
            
            **Editorial Standards**:
            - Fix grammar, spelling, and punctuation
            - Improve sentence structure and readability
            - Verify facts against original data
            - Ensure consistent tone throughout
            - Add missing context where needed
            - Remove redundant or repetitive content
            
            **Output Format**:
            Provide the FINAL POLISHED VERSION with a brief editor's note
            explaining key improvements made.
            
            Mark the final version clearly with "## FINAL OUTPUT" header.
            """,
            model_client_stream=True
        )
    
    async def on_messages(self, messages, cancellation_token: CancellationToken):
        """
        Review and polish content.
        
        Args:
            messages: Contains draft from ContentWriter
            cancellation_token: Token for cancellation support
        
        Returns:
            Final polished content
        """
        logger.info("EditorAgent: Reviewing and polishing content")
        
        draft = messages[-1].content if messages else ""
        
        editor_prompt = f"""
        Review and polish this technology trend summary:
        
        {draft}
        
        Apply all quality checks and editorial standards.
        Provide the final polished version.
        """
        
        return editor_prompt