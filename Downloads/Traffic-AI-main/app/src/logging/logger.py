import logging

from src.logging.logging_formatter import LoggingFormatter

class Logger:
    _root_logger = logging.getLogger()
    _autogen_logger = logging.getLogger("autogen_core")
    _httpcore_logger = logging.getLogger("httpcore")
    _urllib3_logger = logging.getLogger("urllib3")
    _mcp_logger = logging.getLogger("mcp")
    _openai_logger = logging.getLogger("openai")
    _git_logger = logging.getLogger("git")

    @staticmethod
    def initialize(level: str = "INFO", library_level: str = "WARNING"):
        Logger._root_logger.setLevel(level)

        formatter = LoggingFormatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        if Logger._root_logger.hasHandlers():
            Logger._root_logger.handlers.clear()
        Logger._root_logger.addHandler(console_handler)
                
        # Set library logger to WARNING level by default
        Logger._autogen_logger.setLevel(library_level)
        Logger._httpcore_logger.setLevel(library_level)
        Logger._urllib3_logger.setLevel(library_level)
        Logger._mcp_logger.setLevel(library_level)
        Logger._openai_logger.setLevel(library_level)
        Logger._git_logger.setLevel(library_level)