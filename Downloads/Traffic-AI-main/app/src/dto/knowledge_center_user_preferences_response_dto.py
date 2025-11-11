from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KnowledgeCenterUserPreferenceResponseDto:
    error: Optional[str]
    results: List[str]