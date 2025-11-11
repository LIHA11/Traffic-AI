# Standard Library
import logging
from typing import Awaitable, Callable, Dict, List, Optional, Any

# Third-party Libraries
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

# Local Modules
from autogen_core import MessageContext, RoutedAgent
from autogen_core.models import AssistantMessage, CreateResult, LLMMessage
from autogen_core.tools import Tool
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.utils.mcp import MCPTool
from src.connector.agentops.mlflow_ops import MLflowOps
from mlflow.entities import Span, SpanType

logger = logging.getLogger(__name__)

class Agent(RoutedAgent):
    PROMPT_TEMPLATE_FOLDER_PATH = "prompt"
    _jinja_env = Environment(loader=FileSystemLoader(PROMPT_TEMPLATE_FOLDER_PATH))

    def __init__(
        self, 
        name: str, 
        description: str, 
        prompt_templates: Dict[str, str], 
        chat_client: ChatClient, 
        agent_ops: MLflowOps,
        response_format: Optional[BaseModel] = None, 
        tools: Optional[List[Tool]] = None, 
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ):
        """
        Initialize the Agent.

        Args:
            name (str): Name of the agent.
            description (str): Description of the agent.
            prompt_templates (Dict[str, str]): Mapping of prompt keys to template names.
            chat_client (ChatClient): Chat client instance.
            response_format (Optional[BaseModel]): Expected response format.
            tools (Optional[List[Tool]]): List of tools available to the agent.
        """
        super().__init__(description)
        self._name = name
        self._prompt_templates = prompt_templates
        self._chat_history: List[LLMMessage] = []
        self._tools = tools or []
        self._chat_client = chat_client
        self._response_format = response_format
        self._agent_ops = agent_ops
        self._report_message = report_message
        self._span : Optional[Span] = None
    
    def get_id(self) -> str:
        return self.id.key.split("_")[0]
    
    def get_agent_ops(self) -> MLflowOps:
        return self._agent_ops
    
    def get_prompt(self, key: str, vars: dict) -> str:
        """
        Render a prompt template with the provided variables.

        Args:
            key (str): Template key.
            vars (dict): Variables to render in the template.

        Returns:
            str: Rendered prompt string.
        """
        template_name = self._prompt_templates.get(key)
        if not template_name:
            raise ValueError(f"Prompt template '{key}' is missing.")
        return self._jinja_env.get_template(template_name).render(vars)
    
    async def generate(
        self, 
        ctx: MessageContext, 
        new_message: Optional[LLMMessage] = None, 
        append_generated_message: bool = True,
        tools: Optional[List[Tool]] = None,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> CreateResult:
        """
        Generate a response using chat history and tools.

        Args:
            ctx (MessageContext): The message context.
            new_message (Optional[LLMMessage]): New message to add to history.
            append_generated_message (bool): Whether to append the generated message to history.
            tools (Optional[List[Tool]]): Tools to use for this generation.
            session_id (Optional[str]): Session identifier.

        Returns:
            CreateResult: The result of the chat client generation.
        """
        if new_message:
            self._chat_history.append(new_message)
        
        params = self._chat_client.params
        model_name = params["model"]
        model_parameters = {
            "ai.model.max_tokens": params.get("max_tokens"),
            "ai.model.temperature": params["temperature"],
            "ai.model.version": params["default_headers"]["api-version"],
        }
        
        span = self._agent_ops.create_model_span(
            run_id=session_id,
            name=name or "chat.completion",
            model=model_name,
            model_parameters=model_parameters,
            inputs=self._chat_history,
            parent_id=self._span.span_id if self._span is not None else None,
        )

        # Use instance tools unless overridden
        tools = self._tools if tools is None else tools

        # Handle MCPTool instances
        for idx, tool in enumerate(tools):
            if isinstance(tool, MCPTool):
                try:
                    tools[idx] = await tool.get(session_id=session_id)
                except Exception as e:
                    exception = ''
                    if (isinstance(e, ExceptionGroup)):
                        exception = [str(ex) for ex in e.exceptions]
                    else:
                        exception = str(e)
                    logger.error(f"Failed to get mcp tools for session {session_id}: {exception}")

        result = await self._chat_client.create(
            messages=self._chat_history,
            tools=tools,
            cancellation_token=ctx.cancellation_token
        )
        
        if span is not None:
            self._agent_ops.end_model_span(
                run_id=session_id,
                span_id=span.span_id,
                outputs=self._chat_history + [AssistantMessage(content=result.content, source=self.get_id())],
            )

        if append_generated_message:
            self._chat_history.append(
                AssistantMessage(content=result.content, source=self.get_id())
            )
        
        return result