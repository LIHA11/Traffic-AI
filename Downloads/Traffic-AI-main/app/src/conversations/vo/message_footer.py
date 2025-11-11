from typing import List, Optional
from src.conversations.enum.message_footer_type_enum import MessageFooterTypeEnum
from src.conversations.dto.message_footer_dto import MessageFooterDto
from src.conversations.document.message_footer import (
    MessageFooter as MessageFooterDocument,
)


class MessageFooter:
    def __init__(
        self,
        key: str,
        type: MessageFooterTypeEnum,
        content: Optional[str] = None,
        options: Optional[List[str]] = None,
    ):
        self.key = key
        self.type = type
        self.content = content
        self.options = options

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

    @classmethod
    def from_dict(cls, data: dict) -> "MessageFooter":
        return cls(
            key=data.get("key"),
            type=data.get("type"),
            content=data.get("content"),
            options=data.get("options"),
        )

    @classmethod
    def from_dto(cls, dto: MessageFooterDto) -> "MessageFooter":
        return cls(
            key=dto.key,
            type=dto.type,
            content=dto.content,
            options=dto.options,
        )

    @classmethod
    def from_document(cls, document: MessageFooterDocument):
        return cls(
            key=document.key,
            type=document.type,
            content=document.content,
            options=document.options,
        )

    def to_document(self) -> MessageFooterDocument:
        return {
            "key": self.key,
            "type": self.type,
            "content": self.content,
            "options": self.options,
        }
