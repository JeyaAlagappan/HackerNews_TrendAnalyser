"""
Configuration module for Hacker News Trend Analyzer.
Manages environment variables, LLM settings, and system parameters.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """LLM configuration settings"""
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60

@dataclass
class MCPConfig:
    """MCP Server configuration"""
    server_name: str = "hackernews"
    command: str = "npx"
    args: list = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = ["-y", "@modelcontextprotocol/server-hackernews"]

@dataclass
class SystemConfig:
    """System-wide configuration"""
    max_rounds: int = 10
    story_limit: int = 15
    log_level: str = "INFO"
    state_save_path: str = "./team_state.json"
    
class Config:
    """Main configuration class"""
    
    def __init__(self):
        # LLM Configuration
        self.llm = LLMConfig(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000"))
        )
        
        # MCP Configuration
        self.mcp = MCPConfig()
        
        # System Configuration
        self.system = SystemConfig(
            max_rounds=int(os.getenv("MAX_ROUNDS", "10")),
            story_limit=int(os.getenv("STORY_LIMIT", "15")),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
        
        # Validate critical configuration
        self._validate()
    
    def _validate(self):
        """Validate required configuration"""
        if not self.llm.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
     #the model_info section:

    model_info={
    "vision": False,
    "function_calling": True,
    "json_output": True,  # Whether model supports JSON mode
    "structured_output": True,  # ← ADD THIS FIELD
    "family": "gpt-4"
     }