from logging import Formatter
from datetime import datetime, timezone

class LoggingFormatter(Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat()
