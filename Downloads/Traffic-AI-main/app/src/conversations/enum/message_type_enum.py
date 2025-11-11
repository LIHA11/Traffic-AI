from enum import Enum


class MessageTypeEnum(str, Enum):
    CONVERSATION = "conversation"
    ERROR = "error"
    AGENT_PROGRESS = "agent_progress"
