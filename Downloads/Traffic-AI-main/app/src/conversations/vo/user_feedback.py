from typing import Optional
from datetime import datetime, timezone
from src.conversations.dto.user_feedback_dto import UserFeedbackDto
from src.conversations.document.user_feedback import (
    UserFeedback as UserFeedbackDocument,
)


class UserFeedback:
    def __init__(
        self,
        id: Optional[str] = None,
        liked: Optional[bool] = None,
        feedback: Optional[str] = None,
        summary: Optional[str] = None,
        created_date_time_utc: Optional[datetime] = None,
        last_modified_date_time_utc: Optional[datetime] = None,
    ):
        self.id = id
        self.liked = liked
        self.feedback = feedback
        self.summary = summary
        self.created_date_time_utc = created_date_time_utc or datetime.now(
            tz=timezone.utc
        )
        self.last_modified_date_time_utc = last_modified_date_time_utc or datetime.now(
            tz=timezone.utc
        )

    def to_dict(self):
        return {
            "liked": self.liked,
            "feedback": self.feedback,
            "summary": self.summary,
        }

    @classmethod
    def from_dto(cls, dto: UserFeedbackDto):
        return cls(
            liked=dto.liked,
            feedback=dto.feedback,
            summary=dto.summary,
            created_date_time_utc=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def from_document(cls, document: UserFeedbackDocument):
        return cls(
            id=str(document.id),
            liked=document.liked,
            feedback=document.feedback,
            summary=document.summary,
            created_date_time_utc=document.created_date_time_utc,
            last_modified_date_time_utc=document.last_modified_date_time_utc,
        )

    def to_document(self) -> UserFeedbackDocument:
        return UserFeedbackDocument(
            liked=self.liked,
            feedback=self.feedback,
            summary=self.summary,
            created_date_time_utc=self.created_date_time_utc,
            last_modified_date_time_utc=self.last_modified_date_time_utc,
        )
