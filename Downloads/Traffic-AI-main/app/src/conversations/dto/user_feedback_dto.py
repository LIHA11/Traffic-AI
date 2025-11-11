from dataclasses import dataclass
from typing import Optional


@dataclass
class UserFeedbackDto:
    liked: bool
    feedback: Optional[str]
    summary: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "UserFeedbackDto":
        return cls(
            liked=data.get("liked"),
            feedback=data.get("feedback"),
            summary=data.get("summary")
        )
