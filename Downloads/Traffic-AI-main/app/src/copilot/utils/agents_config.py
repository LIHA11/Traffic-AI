from pathlib import Path
from typing import Optional
import yaml

from src.copilot.copilot_v3 import CopilotConfig

class AgentConfig:
  
    @staticmethod
    def get_agent_config(config_file_name: str) -> Optional[CopilotConfig]:
        config_path = Path(f"./agent_config/{config_file_name}_v2.yaml")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        
        with config_path.open('r', encoding='utf-8') as file:
            return yaml.safe_load(file)
  
  