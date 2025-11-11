# ===== Standard Library Imports =====
import asyncio
import json
import logging
import re
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, Annotated, Tuple
from mlflow.entities.span import Span, SpanType

# ===== Third-Party Imports =====
from pydantic import BaseModel

# ===== Project-Specific Imports =====
from autogen_core import (
    FunctionCall,
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool

from src.conversations.constant.conversation_constant import GENERIC_ERROR_MESSAGE
from src.connector.agentops.agentops import LogMessage
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.utils.gather import run_tool_with_retries
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.connector.agentops.mlflow_ops import MLflowOps

logger = logging.getLogger(__name__)

# ===== Constants and Enums =====

class ReporterState(Enum):
    GET_ANSWER = "GET_ANSWER"
    GET_REFERENCE = "GET_REFERENCE"
    REMOVE_INTERNAL_STUFF = "REMOVE_INTERNAL_STUFF"
    REPORT_READY = auto()

NAME_DONE_TOOL = "done"
DESCRIPTION_DONE_TOOL = "Invoke this upon task completion."
NAME_GET_REFERENCE_TOOL = "get_from_memory"
REPORTER_TOPIC = "reporter"

REPORTER_STATE_TRANSITION = {
    ReporterState.GET_ANSWER: ReporterState.GET_REFERENCE,
    ReporterState.GET_REFERENCE: ReporterState.REMOVE_INTERNAL_STUFF,
    ReporterState.REMOVE_INTERNAL_STUFF: ReporterState.REPORT_READY
}

REPORTER_STATE_TRANSITION_NO_REFERENCE = {
    ReporterState.GET_ANSWER: ReporterState.REMOVE_INTERNAL_STUFF,
    ReporterState.REMOVE_INTERNAL_STUFF: ReporterState.REPORT_READY
}

class Reference:
    def __init__(
        self,
        content: str,
        headers: str,
        content_type: str,
        data_description: Dict[str, Any]
    ):
        self.content = content
        self.headers = headers
        self.content_type = content_type
        self.meta_data = data_description

# ===== Message Models =====

class ReporterRequestMessage(BaseModel):
    task: str
    chat_history: List[LLMMessage]
    sent_from: str

class ReporterReplyMessage(BaseModel):
    answer: str

# --- Reporter Agent ---
class Reporter(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        tools: Optional[List[Tool]] = None,
        agent_ops: Optional[MLflowOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ):
        tools = list(tools) if tools else []

        has_reference_tool = any(t.name == NAME_GET_REFERENCE_TOOL for t in tools)
        required_states = [
            ReporterState.GET_ANSWER,
            ReporterState.REMOVE_INTERNAL_STUFF,
            *( [ReporterState.GET_REFERENCE] if has_reference_tool else [] )
        ]

        for state in required_states:
            if prompt_templates.get(state) is None:
                raise ValueError(f"Missing prompt template for '{state}' state in prompt_templates.")

        # Define the 'done' tool
        def done(
            thought: Annotated[str, "Brief reasoning or thought process."],
            result: str
        ) -> str:
            return result

        tools.append(FunctionTool(done, DESCRIPTION_DONE_TOOL, NAME_DONE_TOOL))

        super().__init__(
            name=REPORTER_TOPIC,
            description="Extracts the final answer from the chat history",
            chat_client=chat_client,
            tools=tools,
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message
        )

    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        tools: Optional[List[Tool]] = None,
        agent_ops: Optional[MLflowOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ) -> "Reporter":
        return await Reporter.register(
            runtime,
            type=REPORTER_TOPIC,
            factory=lambda: Reporter(
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                tools=tools,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )

    @staticmethod
    def _remove_thought(llm_result: str) -> str:
        """Remove <think> tags from the LLM result."""
        return re.sub(r"<think>.*?</think>", "", llm_result, flags=re.DOTALL)

    @message_handler
    async def on_request(self, message: ReporterRequestMessage, ctx: MessageContext) -> Optional[ReporterReplyMessage]:
        """
        Main entry for handling a reporter request.
        """
        status = ReporterState.GET_ANSWER
        self._chat_history = []
        agent_ops = self.get_agent_ops()
        
        self._span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )
        
        # Prepare chat history for prompt
        messages = [
            {"role": "user" if isinstance(msg, UserMessage) else "assistant", "content": msg.content}
            for msg in message.chat_history if isinstance(msg, (UserMessage, AssistantMessage))
        ]

        prompt_vars = {
            "chat_history": messages,
            "task": message.task
        }

        tools_wo_reference = [t for t in self._tools if t.name != NAME_GET_REFERENCE_TOOL]
        has_reference_tool = any(t.name == NAME_GET_REFERENCE_TOOL for t in self._tools)

        llm_result = await self.generate(
            ctx,
            SystemMessage(content=self.get_prompt(status, prompt_vars)),
            append_generated_message=False,
            tools=tools_wo_reference,
            session_id=self.get_id()
        )

        states_result: List[str] = []
        references: List[Reference] = []

        retry_count = 0
        max_retries = 3

        while status != ReporterState.REPORT_READY and retry_count < max_retries:
            retry_count += 1
            state_complete, filtered_result = await self._handle_llm_result(llm_result, ctx)

            if isinstance(filtered_result, Exception):
                logger.error("Error handling LLM result: %s", filtered_result)
                llm_result = await self.generate(ctx, new_message=None, append_generated_message=False, session_id=self.get_id())
                states_result.append(GENERIC_ERROR_MESSAGE)
                continue

            if isinstance(filtered_result, str):
                states_result.append(filtered_result)
            elif isinstance(filtered_result, list):
                references.extend(filtered_result)

            if state_complete:
                status = (
                    REPORTER_STATE_TRANSITION.get(status)
                    if has_reference_tool
                    else REPORTER_STATE_TRANSITION_NO_REFERENCE.get(status)
                )

            if status != ReporterState.REPORT_READY:
                next_prompt = self.get_prompt(status, {})
                next_message = UserMessage(content=next_prompt, source=self.get_id())
                next_tools = tools_wo_reference if status != ReporterState.GET_REFERENCE else None
                llm_result = await self.generate(
                    ctx, next_message, False, next_tools, session_id=self.get_id()
                )

        reply_message = ReporterReplyMessage(answer=states_result[-1])
        
        await self._report_message(
            LogMessage(
                agent_name=self._name,
                action="Your request has been completed. The results are as follows",
                content=states_result[-1], 
                id=self.get_id(),
                references=[{"content": ref.content, "headers": ref.headers, "meta_data": { "data_description": ref.meta_data, "data_type": ref.content_type } } for ref in references],
                is_complete=True
            )
        )
        
        if self._span is not None:
            agent_ops.end_span(
                run_id=self.get_id(),
                span_id=self._span.span_id,
                outputs=reply_message
            )
        
        agent_ops.end_run(
            run_id=self.get_id(),
            outputs={
                "content": states_result[-1],
                "references (Maximum content displayed: 2000)": [{"content": ref.content[:2000], "meta_data": { "data_description": ref.meta_data, "data_type": ref.content_type } } for ref in references],
            }
        )
        
        # Only publish if not from human
        if message.sent_from != "human":
            await self.publish_message(
                reply_message, topic_id=TopicId(self.sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token,
            )
            return None

        return reply_message

    async def _handle_llm_result(
        self, llm_result: CreateResult, ctx: MessageContext
    ) -> Tuple[bool, Union[Exception, List[Reference], str]]:
        """
        Handle the result of a tool invocation.
        Returns (state_complete, result)
        """
        
        # Validate the LLM result format
        if not (isinstance(llm_result.content, list) and all(isinstance(call, FunctionCall) for call in llm_result.content)):
            reply_result = (
                "Only function calls are allowed. The format is incorrect; "
                "please try again with exactly the same content but with a suitable function call."
            )
            self._chat_history.extend([
                AssistantMessage(content=llm_result.content, source=self.id.type),
                UserMessage(content=reply_result, source=self.id.type)
            ])
            return False, ValueError(reply_result)

        tools_dict = {tool.name: tool for tool in self._tools}

        tool_calls = []
        for call in llm_result.content:
            tool = tools_dict.get(call.name)
            if not tool:
                reply_result = f"Unknown tool: {call.name}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, ValueError(reply_result)
            try:
                arguments = json.loads(call.arguments)
            except Exception as e:
                reply_result = f"Invalid arguments for tool '{call.name}': {e}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, ValueError(reply_result)
            tool_calls.append((arguments, tool, call))
            
        agent_ops = self.get_agent_ops()
            
        tool_calls_span = agent_ops.create_span(
            run_id=self.get_id(),
            name="handle_tool_calls",
            inputs=llm_result.content,
            span_type=SpanType.TOOL,
            parent_id=self._span.span_id if self._span is not None else None,
        )

        tool_call_results: List[FunctionExecutionResult] = await asyncio.gather(
            *[run_tool_with_retries(args, tool, call, ctx, self._agent_ops, self.get_id(), tool_calls_span.span_id if tool_calls_span is not None else None) for args, tool, call in tool_calls]
        )

        filtered_results, state_complete = [], False

        for ind, result in enumerate(tool_call_results):
            if result.name == NAME_GET_REFERENCE_TOOL:
                try:
                    response = json.loads(result.content)
                except json.JSONDecodeError:
                    return False, ValueError(
                        f"Failed to parse JSON from the tool call result: {result.content}. "
                        "Please review the implementation of get_memory_function"
                    )
                tool_call_results[ind].content = response.get("message")
                if response.get("error") is None:
                    res = response["result"]
                    filtered_results.append(
                        Reference(
                            json.loads(res["content"]),
                            res["headers"],
                            res["type"],
                            res["meta_data"]
                        )
                    )
                else:
                    tool_call_results[ind].is_error = True
            elif result.name == NAME_DONE_TOOL:
                filtered_results = result.content
                tool_call_results[ind].content = "OK"
                state_complete = True
            
        # Update chat history once
        self._chat_history.extend([
            AssistantMessage(content=llm_result.content, source=self.id.type),
            FunctionExecutionResultMessage(content=tool_call_results)
        ])
        
        if tool_calls_span is not None:
            agent_ops.end_span(
                run_id=self.get_id(),
                span_id=tool_calls_span.span_id,
                outputs=tool_call_results
            )
        
        return state_complete, filtered_results