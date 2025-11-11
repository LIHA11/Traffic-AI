from dataclasses import dataclass
from typing import Optional
from src.conversations.vo.conversation import Conversation
from src.conversations.vo.message import Message


@dataclass
class UserFeedbackResponseDto:
    liked: bool
    feedback: Optional[str]
    summary: Optional[str]
    creator: str
    content: str
    metadata: Optional[dict]
    message_id: str
    conversation_id: str

    @classmethod
    def from_message_and_conversation(
        cls, message: Message, conversation: Conversation
    ) -> "UserFeedbackResponseDto":
        return cls(
            liked=message.userFeedback.liked if message.userFeedback else None,
            feedback=message.userFeedback.feedback if message.userFeedback else None,
            summary=message.userFeedback.summary if message.userFeedback else None,
            creator=conversation.creator,
            content=message.content,
            metadata=message.metadata,
            message_id=message.id,
            conversation_id=conversation.id,
        )

    def to_dict(self):
        return {
            "liked": self.liked,
            "feedback": self.feedback,
            "summary": self.summary,
            "creator": self.creator,
            "content": self.content,
            "metadata": self.metadata,
            "messageId": self.message_id,
            "conversationId": self.conversation_id,
        }
