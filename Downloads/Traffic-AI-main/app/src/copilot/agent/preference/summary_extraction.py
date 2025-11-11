# Standard Library Imports
import logging
from typing import (
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
    List
)

# Third-party library imports
from pydantic import ValidationError

# Core Module Imports
from autogen_core import (
    MessageContext,
    TopicId,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    CreateResult,
    SystemMessage,
    UserMessage,
)

# Project-Specific Imports
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.agent.preference.constant import (
    SUMMARY_EXTRACTION_REQUEST,
    SUMMARY_EXTRACTION_TOPIC,
    SummaryExtractionRequest,
    SummaryExtractionResponse,
)

from pydantic import BaseModel

class SummaryExtractionOutput(BaseModel):
    summary: str
    thought: str

logger = logging.getLogger(__name__)

class SummaryExtraction(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
    ):
        super().__init__(
            name=SUMMARY_EXTRACTION_TOPIC,
            description="Analyze a user-assistant conversation and extract a summary.",
            chat_client=chat_client,
            prompt_templates=prompt_templates,
            response_format=SummaryExtractionOutput,
            agent_ops=agent_ops,
            report_message=report_message
        )
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ) -> "SummaryExtraction":
        return await SummaryExtraction.register(
            runtime,
            type=SUMMARY_EXTRACTION_TOPIC,
            factory=lambda: SummaryExtraction(
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )     
            
    @message_handler
    async def on_request(
        self,
        message: SummaryExtractionRequest,
        ctx: MessageContext,
    ) -> Optional[SummaryExtractionResponse]:
        """
        Handles incoming summary extraction requests and forwards results to the keyword extractor.
        """
        self._chat_history = []

        if len(message.messages) < 2:
            response = SummaryExtractionResponse(
                summary=None
            )
            return response
            
        def messages_to_str(messages: List[dict]) -> str:
            """Convert conversation turns to numbered string."""
            return '\n'.join(
                f"{i+1}. {turn['role']}: {turn['content']}"
                for i, turn in enumerate(messages)
            )

        prompt = self.get_prompt(SUMMARY_EXTRACTION_REQUEST, {})
        self._chat_history.append(SystemMessage(content=prompt))
        
        max_retries = 10
        done = False
        for attempt in range(max_retries):
            llm_result = await self.generate(
                ctx, new_message=UserMessage(content=messages_to_str(message.messages), source=self.id.key), session_id=self.get_id()
            ) if attempt == 0 else await self.generate(
                ctx, UserMessage(content=f"{response}, please try again.", source=self.id.key), session_id=self.get_id()
            )
            done, response = self._handle_llm_result(llm_result)
            if done:
                break
            
        if done:
            response = SummaryExtractionResponse(
                summary=response.summary
            )
        else:
            logger.error(f"Failed to parse LLM result after {max_retries} attempts: {response}")
            response = SummaryExtractionResponse(
                summary=None
            )

        return response
        
    @staticmethod
    def _handle_llm_result(
        llm_result: CreateResult,
    ) -> Tuple[bool, Union[SummaryExtractionOutput, str]]:
        """
        Attempts to parse LLM result content into SummaryExtractionOutput.
        Returns (success, result or error).
        """
        content = getattr(llm_result, 'content', None)
        if not content:
            return False, "No content in LLM result."
        try:
            return True, SummaryExtractionOutput.model_validate_json(content)
        except ValidationError as e:
            return False, f"Validation error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
