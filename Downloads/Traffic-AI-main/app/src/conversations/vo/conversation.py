from typing import Optional, List
from datetime import datetime, timezone
from src.conversations.vo.message import Message
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.document.conversation import Conversation as ConversationDocument


class Conversation:
    def __init__(
        self,
        creator: str,
        messages: Optional[List[Message]] = None,
        is_deleted: Optional[bool] = None,
        created_date_time_utc: Optional[datetime] = None,
        last_modified_date_time_utc: Optional[datetime] = None,
        id: Optional[str] = None,
    ):
        now = datetime.now(tz=timezone.utc)

        self.creator = creator
        self.messages = messages or []
        self.is_deleted = is_deleted or False
        self.created_date_time_utc = created_date_time_utc or now
        self.last_modified_date_time_utc = last_modified_date_time_utc or now
        self.id = id

    def add_message(self, message: Message):
        self.messages.append(message)
        self.last_modified_date_time_utc = datetime.now(tz=timezone.utc)

    def to_dict(self):
        return {
            "id": self.id,
            "creator": self.creator,
            "messages": [message.to_dict() for message in self.messages],
            "createdDateTimeUtc": self.created_date_time_utc.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "lastModifiedDateTimeUtc": self.last_modified_date_time_utc.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }

    @classmethod
    def from_document(cls, document: ConversationDocument):
        return cls(
            creator=document.creator,
            messages=[Message.from_document(msg) for msg in document.messages],
            is_deleted=document.is_deleted,
            id=str(document.id),
            created_date_time_utc=document.created_date_time_utc,
            last_modified_date_time_utc=document.last_modified_date_time_utc,
        )
