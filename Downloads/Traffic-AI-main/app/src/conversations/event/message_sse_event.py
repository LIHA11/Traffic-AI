from dataclasses import dataclass
from src.conversations.enum.message_event_enum import MessageEventEnum
import json


@dataclass
class MessageSSEEvent:
    data: str | None = None
    event: MessageEventEnum = MessageEventEnum.MESSAGE

    def encode(self) -> bytes:
        message = f"data:{self.data if self.data else {}}"
        if self.event is not None:
            message = f"{message}\nevent:{self.event.value}"
        message = f"{message}\n\n"
        return message.encode("utf-8")
