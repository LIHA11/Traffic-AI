import asyncio
import logging
import warnings
import openai

from typing import Any, Mapping, Optional, Sequence, ClassVar
from typing_extensions import Unpack

from autogen_core import CancellationToken
from autogen_core.models import CreateResult, LLMMessage
from autogen_core.tools import Tool, ToolSchema

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.models.openai.config import AzureOpenAIClientConfiguration

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
class ChatClient(AzureOpenAIChatCompletionClient):
    RETRY_NUM: ClassVar[int] = 5
    RETRY_DELAY_S: ClassVar[int] = 60
    RETRY_EXCEPTIONS: ClassVar[tuple] = (openai.RateLimitError, openai.InternalServerError, openai.BadRequestError, openai.APITimeoutError)

    def __init__(self, **kwargs: Unpack[AzureOpenAIClientConfiguration]):
        super().__init__(**kwargs)
        self.params = kwargs 

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[Sequence[Tool | ToolSchema]] = None,
        extra_create_args: Optional[Mapping[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[CreateResult]:
        tools = tools or []
        extra_create_args = extra_create_args or {}

        for attempt in range(1, self.RETRY_NUM + 1):
            try:
                return await super().create(
                    messages,
                    tools=tools,
                    extra_create_args=extra_create_args,
                    cancellation_token=cancellation_token
                )
            except self.RETRY_EXCEPTIONS as e:
                if attempt < self.RETRY_NUM:
                    if (e.__class__.__name__ == "BadRequestError") or (e.__class__.__name__ == "APITimeoutError"):
                        T = 2
                        logger.error(
                            f"{e.__class__.__name__}: {e} (attempt {attempt}/{self.RETRY_NUM}). "
                            f"Retrying in {T} seconds..."
                        )
                        await asyncio.sleep(T)
                    else:
                        logger.error(e)
                        logger.error(
                            f"{e.__class__.__name__}: {e} (attempt {attempt}/{self.RETRY_NUM}). "
                            f"Retrying in {self.RETRY_DELAY_S} seconds..."
                        )
                        await asyncio.sleep(self.RETRY_DELAY_S)
        return None