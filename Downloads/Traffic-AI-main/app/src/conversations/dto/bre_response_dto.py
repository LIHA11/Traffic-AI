from dataclasses import dataclass
from typing import Optional
from src.conversations.vo.message import Message

@dataclass
class BreResponseDto:
    id: Optional[str] 
    message: Optional[Message]
    
    def to_dict(self):
        return {
            "id": self.id,
            "message": self.message.to_dict() if self.message else None,
        }