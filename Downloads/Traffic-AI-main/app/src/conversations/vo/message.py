from typing import Optional, Dict, Any
from datetime import datetime, timezone
from src.conversations.vo.message_footer import MessageFooter
from src.conversations.vo.user_feedback import UserFeedback
from src.conversations.enum.message_type_enum import MessageTypeEnum
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.document.message import Message as MessageDocument


class Message:
    def __init__(
        self,
        id: Optional[str] = None,
        content: Optional[str] = "",
        role: Optional[RoleEnum] = None,
        type: Optional[MessageTypeEnum] = None,
        footer: Optional[MessageFooter] = None,
        timestamp: Optional[datetime] = None,
        group_id: Optional[str] = None,
        userFeedback: Optional[UserFeedback] = None,
        metadata: Optional[Dict[str, Any]] = None,
        others: Optional[Dict[str, Any]] = None,
    ):
        self.id = id
        self.role = role or RoleEnum.USER
        self.type = type or MessageTypeEnum.CONVERSATION
        self.content = content
        self.footer = footer
        self.timestamp = timestamp or datetime.now(tz=timezone.utc)
        self.group_id = group_id
        self.userFeedback = userFeedback
        self.metadata = metadata or {}
        self.others = others or {}

    def to_dict(self):
        result = {
            "id": self.id,
            "role": self.role,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp.timestamp(),
            "metadata": self.metadata,
        }
        if self.group_id:
            result["groupId"] = self.group_id
        if self.userFeedback:
            result["userFeedback"] = self.userFeedback.to_dict()
        if self.footer:
            result["footer"] = self.footer.to_dict()
        return result

    @classmethod
    def from_dict(cls, document: Dict[str, Any]):
        return cls(
            id=document.get("id"),
            role=document.get("role"),
            type=document.get("type"),
            content=document.get("content"),
            footer=(
                MessageFooter.from_dict(document.get("footer"))
                if document.get("footer")
                else None
            ),
            userFeedback=(
                UserFeedback.from_dict(document.get("userFeedback"))
                if document.get("userFeedback")
                else None
            ),
            timestamp=document.get("timestamp"),
            group_id=document.get("groupId"),
            metadata=document.get("metadata"),
            others=document.get("others"),
        )

    @classmethod
    def from_content(cls, content: str):
        return cls(
            content=content,
            role=RoleEnum.USER,
            timestamp=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def from_document(cls, document: MessageDocument):
        return cls(
            id=str(document.id),
            content=document.content,
            role=document.role,
            type=document.type,
            timestamp=document.timestamp.replace(tzinfo=timezone.utc),
            group_id=document.group_id,
            footer=(
                MessageFooter.from_document(document.footer)
                if document.footer
                else None
            ),
            userFeedback=(
                UserFeedback.from_document(document.userFeedback)
                if document.userFeedback
                else None
            ),
            metadata=document.metadata,
            others=document.others,
        )

    def to_document(self) -> MessageDocument:
        return MessageDocument(
            content=self.content,
            role=self.role,
            type=self.type,
            timestamp=self.timestamp,
            group_id=self.group_id,
            footer=self.footer.to_document() if self.footer else None,
            userFeedback=self.userFeedback.to_document() if self.userFeedback else None,
            metadata=self.metadata,
            others=self.others,
        )
