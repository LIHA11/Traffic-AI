import asyncio
import json
import logging
from dataclasses import asdict
from typing import (
    Any, Annotated, Awaitable, Callable, Dict, List, Optional, Set, Tuple
)
from mlflow.entities import SpanType, Document

from pydantic import BaseModel

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
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool, Tool
from mlflow.entities.span import Span, SpanType

from src.connector.agentops.agentops import LogMessage, rename_and_remove_keys
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.utils.gather import run_tool_with_retries
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.utils.message import WorkingMemoryService

logger = logging.getLogger(__name__)

# --- Constants ---
NAME_RESOLVED_TOOL = "resolved"
DESCRIPTION_RESOLVED_TOOL = "Marks the task as completed and returns the result."
NAME_FAIL_TOOL = "failed"
DESCRIPTION_FAIL_TOOL = "Marks the task as failed and returns the reason."
EXECUTOR_REQUEST = "executor_request"

# --- Tool Functions ---
def resolved_tool_func(
    result: Annotated[str, "Result with Brief reasoning or thought process."],
    user_message:  Annotated[str, "A short closing message for users, with no technical or memory-related information"],
    result_memory_key: Annotated[Optional[str | None], "Shared memory location for storing the result, if applicable."] = None,
) -> str:
    return result

def fail_tool_func(
    reason: Annotated[str, "Reason with Brief reasoning or thought process."],
    user_message:  Annotated[str, "A short closing message for users, with no technical or memory-related information"]
) -> str:
    return reason

# --- Message Models ---
class ExecutorRequestMessage(BaseModel):
    task: str
    success_criteria: str
    sent_from: Optional[str] = None
    user_message: Optional[str] = None
    parent_id: Optional[str] = None

class ExecutorReplyMessage(BaseModel):
    content: str
    is_success: bool

# --- Executor Agent ---
class Executor(Agent):
    """
    Executor agent that handles tasks and invokes tools as needed.
    """
    def __init__(
        self,
        name: str,
        description: str,
        prompt_templates: Dict[str, str],
        chat_client: 'ChatClient',
        knowledge_center: Optional['KnowledgeCenter'] = None,
        tools: Optional[List['Tool']] = None,
        notes: Optional[str] = None,
        agent_ops: Optional[LangfuseOps] = None,
        max_call_count: Optional[int] = 20,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
        wms: Optional[WorkingMemoryService] = None,
        report_intermediate_steps: bool = False
    ):
        if EXECUTOR_REQUEST not in prompt_templates:
            raise ValueError(f"Missing required prompt template: '{EXECUTOR_REQUEST}'")

        # Ensure tools is a list and add resolved/fail tools only if not present
        tools = list(tools) if tools else []
        tool_names: Set[str] = {tool.name for tool in tools}

        # Add essential tools if missing
        essential_tools = [
            (NAME_RESOLVED_TOOL, resolved_tool_func, DESCRIPTION_RESOLVED_TOOL),
            (NAME_FAIL_TOOL, fail_tool_func, DESCRIPTION_FAIL_TOOL)
        ]
        for tool_name, func, desc in essential_tools:
            if tool_name not in tool_names:
                tools.append(FunctionTool(func, desc, tool_name))

        super().__init__(
            name=name,
            description=description,
            chat_client=chat_client,
            tools=tools,
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message
        )
        self._knowledge_center = knowledge_center
        self._max_call_count = max_call_count
        self._notes = notes
        self._report_intermediate_steps = report_intermediate_steps
        self._wms = wms
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        name: str,
        description: str,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        knowledge_center: Optional[KnowledgeCenter] = None,
        tools: Optional[List[Tool]] = None,
        notes: Optional[str] = None,
        agent_ops: Optional[LangfuseOps] = None,
        max_call_count: Optional[int] = 20,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
        wms: Optional[WorkingMemoryService] = None,
        report_intermediate_steps: bool = False
    ) -> 'Executor':
        return await Executor.register(
            runtime,
            type=name,
            factory=lambda: Executor(
                name=name,
                description=description,
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                knowledge_center=knowledge_center,
                tools=tools,
                notes=notes,
                agent_ops=agent_ops,
                report_message=report_message,
                max_call_count=max_call_count,
                report_intermediate_steps=report_intermediate_steps,
                wms=wms
            ),
        )
        
    @message_handler
    async def on_request(
        self, message: ExecutorRequestMessage, ctx: MessageContext
    ) -> Optional[ExecutorReplyMessage]:
        """
        Handles incoming ExecutorRequestMessage, generates response, and publishes if required.
        """
        self._chat_history = []
        call_count = self._max_call_count
        
        agent_ops = self.get_agent_ops()
        self._span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )
        
        await self._report_message(
                LogMessage(
                    agent_name=self._name,
                    action="Starting Subtask:",
                    content=message.model_dump_json(), 
                    id=self.get_id(),
                )
            )
        
        retriever_span: Optional[Span] = None
        if self._span is not None:
            retriever_span = agent_ops.create_span(
                run_id=self.get_id(),
                name="search_past_experience",
                parent_id=self._span.span_id,
                span_type=SpanType.RETRIEVER,
                inputs=message.task,
            )

        docs = (
            await self._knowledge_center.get_documents(query_text=message.task, domains=self._name)
            if self._knowledge_center else []
        )
        
        if retriever_span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                            span_id=retriever_span.span_id,
                            outputs=[Document(page_content=doc.content) for doc in docs])


        prompt_vars = {
            **message.model_dump(),
            "docs": docs,
            "name": self._name,
            "notes": self._notes
        }
        
        system_message = SystemMessage(
            content=self.get_prompt(EXECUTOR_REQUEST, prompt_vars)
        )
        
        llm_result: CreateResult = await self.generate(ctx, system_message, append_generated_message=False, session_id=self.get_id())

        masked_tools = [tool for tool in self._tools if tool.name in {NAME_FAIL_TOOL, NAME_RESOLVED_TOOL}]
        
        done, reply_result, is_success = await self._handle_llm_result(llm_result, ctx)
        while not done:
            call_count -= int(is_success)
            llm_result = await self.generate(
                ctx, 
                append_generated_message=False, 
                session_id=self.get_id(), 
                tools=masked_tools if call_count == 0 else None
            )
            done, reply_result, is_success = await self._handle_llm_result(llm_result, ctx)

        reply_message = ExecutorReplyMessage(
            content=reply_result,
            is_success=is_success
            
        )
        
        if self._span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                        span_id=self._span.span_id,
                        outputs=reply_message,
                        attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
            
        if message.sent_from:
            await self.publish_message(
                reply_message,
                topic_id=TopicId(message.sent_from, source=self.get_id()),
                cancellation_token=ctx.cancellation_token,
            )
            return None
        else:
            return reply_message
        
    async def _handle_llm_result(
        self, llm_result: CreateResult, ctx: MessageContext
    ) -> Tuple[bool, Any, bool]:
        """
        Handles the result from the LLM and invokes tools if needed.
        Returns (done, reply_result, is_success).
        """
        if not isinstance(llm_result.content, list) or not all(isinstance(call, FunctionCall) for call in llm_result.content):
            reply_result = "Invalid format: Only function calls are allowed. Please retry with the correct format."
            self._chat_history.extend([
                AssistantMessage(content=llm_result.content, source=self.id.type),
                UserMessage(content=reply_result, source=self.id.type)
            ])
            return False, None, False

        tools_dict = {tool.name: tool for tool in self._tools}
        tool_calls = []
        
        agent_ops = self.get_agent_ops()
        
        tool_calls_span = agent_ops.create_span(
            run_id=self.get_id(),
            name="handle_tool_calls",
            span_type=SpanType.TOOL,
            inputs=llm_result.content,
            parent_id=self._span.span_id if self._span is not None else None,
        )
        
        for call in llm_result.content:
            tool = tools_dict.get(call.name)
            if not tool:
                reply_result = f"Unknown tool: {call.name}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, None, False
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as e:
                reply_result = f"Invalid arguments for tool '{call.name}': {e}"
                self._chat_history.extend([
                    AssistantMessage(content=llm_result.content, source=self.id.type),
                    UserMessage(content=reply_result, source=self.id.type)
                ])
                return False, None, False
            tool_calls.append((arguments, tool, call))
            
        for call in llm_result.content:
            if call.name not in {NAME_FAIL_TOOL, NAME_RESOLVED_TOOL}:
                if self._report_intermediate_steps:
                    await self._report_message(
                        LogMessage(
                            agent_name=self._name,
                            action="Executing external tools ...",
                            content=json.loads(call.arguments)["thought"], 
                            id=self.get_id(),
                            is_complete=False,
                        )
                    )

        # Run all tools concurrently
        tool_call_results: List[FunctionExecutionResult] = await asyncio.gather(
            *[run_tool_with_retries(args, tool, call, ctx, self._agent_ops, self.get_id(), tool_calls_span.span_id if tool_calls_span is not None else None) for args, tool, call in tool_calls]
        )
        
        # Update chat history once
        self._chat_history.extend([
            AssistantMessage(content=llm_result.content, source=self.id.type),
            FunctionExecutionResultMessage(content=tool_call_results)
        ])
        
        if tool_calls_span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                span_id=tool_calls_span.span_id,
            )

        # Find if any tool call signals done/failed
        for result in tool_call_results:
            if result.name not in {NAME_FAIL_TOOL, NAME_RESOLVED_TOOL}:
                if self._report_intermediate_steps:
                    if result.is_error:
                        try:
                            tool_output = json.loads(result.content)["message"]
                            await self._report_message(
                                LogMessage(
                                    agent_name=self._name,
                                    action="Received output from the tool:",
                                    content=tool_output, 
                                    id=self.get_id(),
                                    is_complete=False,
                                )
                            )
                        except json.JSONDecodeError:
                            await self._report_message(
                                LogMessage(
                                    agent_name=self._name,
                                    action="Received output from the tool:",
                                    content="Fail to run the tool", 
                                id=self.get_id(),
                                is_complete=False,
                            )
                        )
                    else:
                        await self._report_message(
                            LogMessage(
                                agent_name=self._name,
                                action="Received response from the external tool:",
                                content=json.loads(result.content)["message"], 
                                id=self.get_id(),
                                is_complete=False,
                            )
                        )
            else:
                for call in llm_result.content:
                    if call.id == result.call_id:
                        user_msg = json.loads(call.arguments).get("user_message", "No user message provided.")
                        memory_location = json.loads(call.arguments).get("result_memory_key", None)
                        res = []
                        if self._wms is not None and memory_location is not None:
                            ## Get file from memory
                            try:
                                wms_result = await self._wms.get(
                                    session_id=self.get_id(),
                                    resource_id=memory_location
                                )
                                res = [{"content": json.loads(wms_result["content"]), "headers": wms_result["headers"], "meta_data": { "data_description": wms_result["meta_data"], "data_type": wms_result["content_type"] }}]
                            except Exception as e:
                                logger.error("Failed to get memory from WMS: %s", str(e))
                                res = []  # Ensure res is always a list
                            
                        await self._report_message(
                            LogMessage(
                                agent_name=self._name,
                                action="Final results of the subtask:",
                                content=user_msg, 
                                id=self.get_id(),
                                references=res,
                                is_complete=False,
                            )
                        )
                return True, result.content, result.name == NAME_RESOLVED_TOOL

        return False, None, True