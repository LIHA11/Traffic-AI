from abc import ABC, abstractmethod
import logging
from time import sleep
import threading

logger = logging.getLogger(__name__)

class TokenManager(ABC):
    _REFRESH_ACCESS_TOKEN_INTERVAL = 60

    def __init__(
        self,
        callback_funcs: list,
        refresh_access_token_interval: int = _REFRESH_ACCESS_TOKEN_INTERVAL,
        auto_refresh: bool = False,
    ):
        self._access_token = None
        self._callback_funcs = callback_funcs

        def __periodic_run(interval_sec: int, func):
            while True:
                sleep(interval_sec)
                func()

        self.__refresh_access_token_with_callback()
        if auto_refresh:
            threading.Thread(
                target = __periodic_run,
                args = (
                    refresh_access_token_interval,
                    self.__refresh_access_token_with_callback,
                ),
                daemon = True,
            ).start()

    def get_access_token(self) -> str:
        return self._access_token

    @abstractmethod
    def refresh_access_token(self) -> str:
        pass

    def __refresh_access_token_with_callback(self):
        self.refresh_access_token()
        if self.get_access_token() is not None:
            for func in self._callback_funcs:
                func(self.get_access_token())
