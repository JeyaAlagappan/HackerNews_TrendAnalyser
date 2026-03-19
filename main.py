"""
Main application entry point for Hacker News Trend Analyzer.
Orchestrates the multi-agent team and manages the workflow.
"""

import asyncio
import sys
from typing import Optional
import signal
from datetime import datetime
import os

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("state", exist_ok=True)

# Fix imports
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import Config
from agents import (
    NewsFetcherAgent, 
    TrendAnalyzerAgent, 
    ContentWriterAgent, 
    EditorAgent
)
from mcpclient import HackerNewsMCPClient
from proj_utils import setup_logger,AgentStateManager # Fixed import

# Setup logging
logger = setup_logger(__name__, "INFO")

class HackerNewsTrendAnalyzer:
    """
    Main orchestrator for the Hacker News trend analysis system.
    Manages agent team, workflow, and state persistence.
    """
    
    def __init__(self):
        self.config = Config()
        self.model_client = None
        self.hn_client = None
        self.team = None
        self.state_manager = AgentStateManager()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        logger.info(f"Initializing HackerNewsTrendAnalyzer (Session: {self.session_id})")
        
        try:
            # Initialize model client
             # From your main.py initialization
            self.model_client = OpenAIChatCompletionClient(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            model_info={
            "vision": False,           # Does model support image inputs?
            "function_calling": True,   # Does model support tool/function calls?
            "json_output": True,        # Does model support JSON mode?
            "structured_output": True,  # Does model support structured outputs?
            "family": "gpt-4",          # Model family identifier
            # Optional but recommended:
            "token_limit": 8192,        # Max context tokens
            "multiple_system_messages": False  # Whether model accepts multiple system messages
        }
)
            
            # Initialize Hacker News client
            self.hn_client = HackerNewsMCPClient(
                command=self.config.mcp.command,
                args=self.config.mcp.args
            )
            await self.hn_client.initialize()
            
            # Create agents
            news_fetcher = NewsFetcherAgent(self.model_client, self.hn_client)
            trend_analyzer = TrendAnalyzerAgent(self.model_client)
            content_writer = ContentWriterAgent(self.model_client)
            editor = EditorAgent(self.model_client)
            
            # Define termination condition
            # Stop when editor produces final output (contains "FINAL OUTPUT")
            termination = TextMentionTermination("FINAL OUTPUT")
            
            # Create Round Robin team
            self.team = RoundRobinGroupChat(
                participants=[news_fetcher, trend_analyzer, content_writer, editor],
                termination_condition=termination,
                max_turns=self.config.system.max_rounds
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def run_analysis(self, custom_task: Optional[str] = None):
        """
        Run the trend analysis workflow.
        
        Args:
            custom_task: Optional custom task description
        """
        if not self.team:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        self.running = True
        
        # Default task if none provided
        task = custom_task or """
        Fetch the latest technology and cybersecurity news from Hacker News,
        analyze emerging trends, and create an insightful summary.
        Focus on stories with high engagement and cybersecurity relevance.
        """
        
        logger.info(f"Starting trend analysis with task: {task}")
        
        try:
            
            # Run the team with streaming output
            logger.info("Team execution started...")
            
            # Use Console for beautiful formatted output
            await Console(self.team.run_stream(task=task))
            
            # Save state for potential recovery
            await self.state_manager.save_team_state(self.team, self.session_id)
            
            logger.info("Team execution completed")
            
        except asyncio.CancelledError:
            logger.warning("Analysis cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            self.running = False
    
    async def shutdown(self):
        """Clean shutdown of all components"""
        logger.info("Shutting down...")
        
        if self.running:
            logger.warning("System still running, forcing shutdown...")
        
        if self.hn_client:
            await self.hn_client.close()
        
        if self.model_client:
            await self.model_client.close()
        
        logger.info("Shutdown complete")

async def main():
    """Main entry point"""
    analyzer = None
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Interrupt received, shutting down...")
        if analyzer and analyzer.running:
            analyzer.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize system
        analyzer = HackerNewsTrendAnalyzer()
        await analyzer.initialize()
        
        # Run analysis
        await analyzer.run_analysis()
        
    except KeyboardInterrupt:
        logger.info("User interrupted execution")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if analyzer:
            await analyzer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())