from dataclasses import dataclass


@dataclass
class UserPreferenceRequestDto:
    content: str
    top_n: int = 10
    threshold: float = 0.1

    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferenceRequestDto":
        return cls(
            content=data.get("content"),
            top_n=data.get("topN", 10),
            threshold=data.get("threshold", 0.1),
        )

