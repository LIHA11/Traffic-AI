from enum import Enum


class MessageEventEnum(str, Enum):
    MESSAGE = "message"
    AGENT_PROGRESS = "agent_progress"
    START_STREAM = "start_stream"
    END_STREAM = "end_stream"
    KEEP_ALIVE = "keep_alive"