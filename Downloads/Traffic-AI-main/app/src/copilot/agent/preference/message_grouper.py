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
    MESSAGE_GROUPER_REQUEST,
    MESSAGE_GROUPER_TOPIC,
    MessageGrouperRequest,
    MessageGrouperResponse,
)

from pydantic import BaseModel

class MessageGrouperOutput(BaseModel):
    requests: List[Tuple[int, int]]
    thoughts: List[str]

logger = logging.getLogger(__name__)

class MessageGrouper(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
    ):
        super().__init__(
            name=MESSAGE_GROUPER_TOPIC,
            description="Analyze a user-assistant conversation and group related requests into distinct groups.",
            chat_client=chat_client,
            prompt_templates=prompt_templates,
            response_format=MessageGrouperOutput,
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
    ) -> "MessageGrouper":
        return await MessageGrouper.register(
            runtime,
            type=MESSAGE_GROUPER_TOPIC,
            factory=lambda: MessageGrouper(
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )     
            
    @message_handler
    async def on_request(
        self,
        message: MessageGrouperRequest,
        ctx: MessageContext,
    ) -> Optional[MessageGrouperResponse]:
        """
        Handles incoming message grouping requests and forwards results to the keyword extractor.
        """
        self._chat_history = []
        
        if len(message.messages) > 1:
            start_idx = len(message.messages) - 1
            if message.refresh_all_msgs:
                start_idx = 0
            else:
                while start_idx > 0 and message.messages_groups.get(message.messages[start_idx]["id"], None) is None:
                    start_idx -= 1
                last_group_id = message.messages_groups.get(message.messages[start_idx]["id"], None)

                # Move back until messages belong to same group
                while start_idx > 0 and message.messages_groups.get(message.messages[start_idx-1]["id"], None) == last_group_id:
                    start_idx -= 1
                    
                
            def messages_to_str(messages: List[dict]) -> str:
                """Convert conversation turns to numbered string."""
                return '\n'.join(
                    f"{i+1}. {turn['role']}: {turn['content']}"
                    for i, turn in enumerate(messages)
                )

            prompt = self.get_prompt(MESSAGE_GROUPER_REQUEST, {})
            self._chat_history.append(SystemMessage(content=prompt))
            
            max_retries = 3
            done = False
            for attempt in range(max_retries):
                llm_result = await self.generate(
                    ctx, new_message=UserMessage(content=messages_to_str(message.messages[start_idx:]), source=self.id.key), session_id=self.get_id()
                ) if attempt == 0 else await self.generate(
                    ctx, UserMessage(content=f"{response}, please try again.", source=self.id.key), session_id=self.get_id()
                )
                done, response = self._handle_llm_result(llm_result)
                if done:
                    break
                
            grouped_contents = []
            for start, end in response.requests:
                contents = []
                for msg in message.messages[start_idx + (int(start) - 1):start_idx + int(end)]:
                    contents.append(msg)
                grouped_contents.append(contents)
        
            if done:
                response = MessageGrouperResponse(
                    messages_by_groups=grouped_contents
                )
            else:
                logger.error(f"Failed to parse LLM result after {max_retries} attempts: {response}")
                response = MessageGrouperResponse(
                    messages_by_groups=None
                )
        else:
            response = MessageGrouperResponse(
                messages_by_groups=None
            )
            
        return response
        
    @staticmethod
    def _handle_llm_result(
        llm_result: CreateResult,
    ) -> Tuple[bool, Union[MessageGrouperOutput, str]]:
        """
        Attempts to parse LLM result content into MessageGrouperOutput.
        Returns (success, result or error).
        """
        content = getattr(llm_result, 'content', None)
        if not content:
            return False, "No content in LLM result."
        try:
            return True, MessageGrouperOutput.model_validate_json(content)
        except ValidationError as e:
            return False, f"Validation error: {e}"
        except Exception as e:
            return False, f"Error: {e}"
    
