from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class PreferenceExtractionResultDto:
    message_group_id: Optional[str]
    summary: Optional[str]
    keywords: List[dict[str, str]]
    
    def to_dict(self) -> dict:
        return {
            "messageGroupId": self.message_group_id,
            "summary": self.summary,
            "keywords": self.keywords,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceExtractionResultDto":
        return cls(
            message_group_id=data.get("message_group_id"),
            summary=data.get("summary"),
            keywords=data.get("keywords", [])
        )

    @classmethod
    def from_str(cls, data: str) -> "PreferenceExtractionResultDto | None":
        try:
            return cls.from_dict(json.loads(data))
        except json.JSONDecodeError:
            return None