from typing import TypedDict
from enum import Enum

class LoggingLevelEnum(str, Enum):
    CRITICAL = "CRITICAL",
    ERROR = "ERROR",
    WARNING = "WARNING",
    INFO = "INFO",
    DEBUG = "DEBUG",
    NOTSET = "NOTSET"

class LoggingConfig(TypedDict):
    level: LoggingLevelEnum
