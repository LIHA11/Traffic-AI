from enum import Enum


class MessageMetadataKeyEnum(str, Enum):
    SHIPMENT = "SHIPMENT"
    AGENT_PROGRESS = "AGENT_PROGRESS"
    TABLE = "TABLE"
