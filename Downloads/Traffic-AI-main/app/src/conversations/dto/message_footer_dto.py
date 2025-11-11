from dataclasses import dataclass
from typing import List, Optional
from src.conversations.enum.message_footer_type_enum import MessageFooterTypeEnum


@dataclass
class MessageFooterDto:
    key: str
    type: MessageFooterTypeEnum
    content: Optional[str]
    options: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "MessageFooterDto":
        return cls(
            key=data.get("key"),
            type=data.get("type"),
            content=data.get("content"),
            options=data.get("options"),
        )

    def to_dict(self) -> dict:
        footer = {
            "key": self.key,
            "type": self.type,
        }
        if self.content:
            footer["content"] = self.content
        if self.options:
            footer["options"] = self.options

        return footer
