from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from src.connector.agentops.agentops import LogMessage
import uuid


@dataclass
class AgentProgressDto:
    id: str
    agent_name: str
    action: str
    content: str
    is_completed: bool
    timestamp: datetime
    references: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "id": self.id,
            "agentName": self.agent_name,
            "action": self.action,
            "content": self.content,
            "timestamp": self.timestamp.timestamp(),
            "isCompleted": self.is_completed,
        }

    @classmethod
    def from_log_message(cls, log_message: LogMessage):
        return cls(
            id=str(uuid.uuid4()),
            agent_name=log_message.agent_name,
            action=log_message.action,
            content=log_message.content,
            is_completed=log_message.is_complete,
            timestamp=log_message.timestamp,
            references=log_message.references,
        )
