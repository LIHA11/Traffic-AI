# Standard library imports
import json
import logging
from typing import (
    Awaitable, Callable, Dict, List, Optional, Tuple, Annotated
)
from collections import defaultdict, deque
from dataclasses import asdict

# Third-party imports
from pydantic import BaseModel

# Autogen core imports
from autogen_core import (
    AgentId,
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
from mlflow.entities import SpanType, Document
from mlflow.entities import Span

# Local imports
from src.connector.agentops.agentops import LogMessage
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.agent.agent import Agent
from src.copilot.utils.handoff import Handoff
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.agent.dag_generator import (
    DAG,
    DAGGeneratorRequestMessage,
    DAG_GENERATOR_TOPIC,
    DAGGeneratorReplyMessage,
    Node,
)
from src.copilot.agent.executor import ExecutorRequestMessage, ExecutorReplyMessage
from src.copilot.utils.gather import gather_with_retries

logger = logging.getLogger(__name__)

# --- Constants ---
RESOLVED_TOOL_NAME = "resolved"
FAILED_TOOL_NAME = "failed"
RESOLVED_TOOL_DESCRIPTION = "Marks the assigned task as completed."
FAILED_TOOL_DESCRIPTION = "Marks the assigned task as failed."
AGENT_NAME = "task_dispatcher"
AGENT_DESCRIPTION = """
Responsible for:
    - Execute tasks assigned by the Planner using appropriate tools and function calls.
    - Communicate results to the Planner clearly and in a structured format.
    - Escalate issues with diagnostics and terminate the process if a task cannot be completed.
"""
STAGE_RECEIVE_DAG = "RECEIVE_DAG"
REQUIRED_PROMPT_TEMPLATES = [STAGE_RECEIVE_DAG]

# --- Data Models ---
class TaskDispatcherRequestMessage(BaseModel):
    plan: str
    sent_from: Optional[str] = None
    user_request: str
    parent_id: Optional[str] = None

class WorkerReply(BaseModel):
    worker_name: str
    is_sucess: bool
    reply: str

class TaskDispatcherReplyMessage(BaseModel):
    task_description: str
    success_criteria: str
    dispatcher_reply: str
    worker_replies: Optional[List[WorkerReply]] = None
    is_success: bool

class TaskDispatcher(Agent):
    def __init__(
        self,
        name: str,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        knowledge_center: Optional[KnowledgeCenter] = None,
        tools: Optional[List[Tool]] = None,
        proactive_planning: bool = True,
        agent_ops: Optional[LangfuseOps] = None,
        report_message: Optional[Callable[[LogMessage], Awaitable[None]]] = None
    ):
        missing = [req for req in REQUIRED_PROMPT_TEMPLATES if req not in prompt_templates]
        if missing:
            raise ValueError(f"Missing required prompt templates: {missing}")

        tools = tools or []
        
        def resolved(reason: Annotated[str, "Reason for success"]) -> str:
            return reason
        
        def failed(reason: Annotated[str, "Reason for failure"]) -> str:
            return reason
        
        tools.append(FunctionTool(resolved, RESOLVED_TOOL_DESCRIPTION, RESOLVED_TOOL_NAME))
        tools.append(FunctionTool(failed, FAILED_TOOL_DESCRIPTION, FAILED_TOOL_NAME))

        super().__init__(
            name=name,
            description=AGENT_DESCRIPTION,
            chat_client=chat_client,
            tools=tools,
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message
        )

        self._knowledge_center = knowledge_center
        self._sent_from = None
        self._proactive_planning = proactive_planning
        self._dag = None
        self._user_message = None
        self._tools_dict = {tool.name: tool for tool in self._tools}
        self._chat_history = []

    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        name: str,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        knowledge_center: Optional[KnowledgeCenter] = None,
        tools: Optional[List[Tool]] = None,
        agent_ops: Optional[LangfuseOps] = None,
        report_message: Optional[Callable[[LogMessage], Awaitable[None]]] = None
    ) -> 'TaskDispatcher':
        return await TaskDispatcher.register(
            runtime,
            type=AGENT_NAME,
            factory=lambda: TaskDispatcher(
                name=name,
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                knowledge_center=knowledge_center,
                tools=tools,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )

    @message_handler
    async def on_request(self, message: TaskDispatcherRequestMessage, ctx: MessageContext) -> None:
        self._sent_from = message.sent_from
        self._user_message = message.user_request
        self._chat_history = [] 
       
        agent_ops = self.get_agent_ops()
        self._span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )

        logger.debug(
            f"Task Dispatcher - {self._name} received request from {self._sent_from} with plan: {message.plan}. "
            f"User message: {self._user_message}"
        )

        dag_request = DAGGeneratorRequestMessage(
            plan=message.plan,
            sent_from=self._name,
            parent_id=self._span.span_id if self._span is not None else None,
        )
        topic_id = TopicId(DAG_GENERATOR_TOPIC, source=self.get_id())
        await self.publish_message(dag_request, topic_id=topic_id, cancellation_token=ctx.cancellation_token)

    @staticmethod    
    def _topological_sort(dag: DAG) -> List[Node]:
        num_nodes = len(dag.nodes)

        # Build adjacency list and in-degree count using indices
        adj = defaultdict(list)      # key: source index, value: list of target indices
        in_degree = [0] * num_nodes  # in-degree count for each node by index

        # Build the graph
        for edge in dag.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1

        # Start with nodes with in-degree 0
        queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
        result = []

        while queue:
            u = queue.popleft()
            result.append(dag.nodes[u])
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(result) != num_nodes:
            raise ValueError("Graph has at least one cycle, topological sort not possible.")

        return result
    
    @message_handler
    async def on_receive_dag(
        self, message: DAGGeneratorReplyMessage, ctx: MessageContext
    ) -> Optional[TaskDispatcherReplyMessage]:
        self._dag = message.dag
        if not self._proactive_planning:
            raise NotImplementedError("Non Proactive planning is not implemented yet.")

        nodes = self._topological_sort(self._dag)
        await self._report_message(LogMessage(
            agent_name=self._sent_from,
            action="Execution plan has been prepared to fulfill your request:",
            content="[" + ",".join([node.model_dump_json() for node in nodes]) + "]",
            id=self.get_id(),
        ))

        target_node = nodes[0]

        agent_ops = self.get_agent_ops()
        
        retriever_span: Optional[Span] = None
        if self._span is not None:
            retriever_span = agent_ops.create_span(
                run_id=self.get_id(),
                name="search_past_experience",
                parent_id=self._span.span_id,
                span_type=SpanType.RETRIEVER,
                inputs=target_node.task_description,
            )

        docs = []
        if self._knowledge_center:
            docs = await self._knowledge_center.get_documents(
                query_text=target_node.task_description, domains=self._name
            )
        
        if retriever_span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                span_id=retriever_span.span_id,
                outputs=[Document(page_content=doc.content) for doc in docs])

        prompt_vars = {
            **message.model_dump(),
            "docs": docs,
            "task": target_node.task_description,
            "success_criteria": target_node.success_criteria,
            "user_message": self._user_message
        }

        system_message = SystemMessage(
            content=self.get_prompt(STAGE_RECEIVE_DAG, prompt_vars)
        )

        reply_results = []
        llm_result: CreateResult = await self.generate(ctx, new_message=system_message, append_generated_message=False, session_id=self.get_id())
        done, reply_result, is_success = await self._handle_llm_result(llm_result, target_node, ctx)
        reply_results.append(reply_result)

        while not done:
            llm_result = await self.generate(ctx, append_generated_message=False, session_id=self.get_id())
            done, reply_result, is_success = await self._handle_llm_result(llm_result, target_node, ctx)
            reply_results.append(reply_result)
            
        worker_replies = []
        for reply_result in reversed(reply_results):
            for reply in reply_result:
                if isinstance(reply, WorkerReply):
                    worker_replies.append(reply)

            if len(worker_replies) > 0:
                break

        reply_message = TaskDispatcherReplyMessage(
            task_description=target_node.task_description,
            success_criteria=target_node.success_criteria,
            dispatcher_reply=reply_results[-1][-1] if reply_results[-1] else "",
            worker_replies=worker_replies or None,
            is_success=is_success
        )  

        if self._span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                        span_id=self._span.span_id,
                        outputs=reply_message,
                        attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
            

        if self._sent_from:
            await self.publish_message(
                reply_message,
                topic_id=TopicId(self._sent_from, source=self.get_id()),
                cancellation_token=ctx.cancellation_token,
            )
            return None

        return reply_message
            
        
    async def _handle_llm_result(
        self,
        llm_result: CreateResult,
        node: Node,
        ctx: MessageContext
    ) -> Tuple[bool, List[str], bool]:
        """
        Handles the result from the LLM and invokes tools if needed.
        Returns (done, reply_results, is_success).
        """

        def _add_history_and_return(msg, content, success=False):
            self._chat_history.extend([
                AssistantMessage(content=f"My generation: {content}", source=self.id.type),
                UserMessage(content=msg, source=self.id.type)
            ])
            return False, [msg], success

        if not isinstance(llm_result.content, list):
            return _add_history_and_return(
                "Invalid format: Only function calls are allowed. Please retry with the correct format.",
                llm_result.content,
            )

        tool_call_results = []
        delegate_targets = []
        reply_results = []
        done = False
        is_success = False

        for call in llm_result.content:
            if call.name in {FAILED_TOOL_NAME, RESOLVED_TOOL_NAME} and len(llm_result.content) > 1:
                return _add_history_and_return(
                    f"Only one tool call is expected for failure or resolved",
                    llm_result.content,
                )

            tool = self._tools_dict.get(call.name)
            if not tool:
                return _add_history_and_return(
                    f"Unknown tool: {call.name}",
                    llm_result.content,
                )
            try:
                arguments = json.loads(call.arguments)
                result = await tool.run_json(arguments, ctx.cancellation_token)
            except Exception as e:
                return _add_history_and_return(
                    f"Invalid arguments for tool '{call.name}': {e}",
                    llm_result.content,
                )
                
            result_as_str = tool.return_value_as_string(result)
            tool_call_results.append(FunctionExecutionResult(
                call_id=call.id,
                content=result_as_str,
                is_error=False,
                name=call.name
            ))

            if isinstance(tool, Handoff):
                delegate_targets.append((call.id, result_as_str))

            if call.name in {FAILED_TOOL_NAME, RESOLVED_TOOL_NAME}:
                done = True
                reply_results = [result_as_str]
                is_success = call.name == RESOLVED_TOOL_NAME

        if delegate_targets:
            message_agent_pairs = [
                (
                    ExecutorRequestMessage(
                        task=node.task_description,
                        success_criteria=node.success_criteria,
                        user_message=self._user_message,
                        parent_id=self._span.span_id if self._span is not None else None,
                    ),
                    AgentId(agent, f"{self.get_id()}_{i}")
                )
                for i, (_, agent) in enumerate(delegate_targets)
            ]
            task_factories = [
                lambda m=msg, a=aid: self.send_message(m, a)
                for msg, aid in message_agent_pairs
            ]
            
            replies_message: List[ExecutorReplyMessage] = await gather_with_retries(task_factories, max_retries=3)
            call_id_to_result = {t.call_id: t for t in tool_call_results}

            for idx, reply_message in enumerate(replies_message):
                call_id = delegate_targets[idx][0]
                tool_call_result = call_id_to_result.get(call_id)
                if not tool_call_result:
                    continue

                if isinstance(reply_message, Exception):
                    reply_results.append(
                        WorkerReply(
                            worker_name=tool_call_result.name,
                            is_sucess=False,
                            reply=str(reply_message)
                        )
                    )
                    tool_call_result.content = (
                        f"Task failed to complete. Reason: Agent raised an Exception: {str(reply_message)}"
                    )
                    tool_call_result.is_error = True
                else:
                    reply_results.append(
                        WorkerReply(
                            worker_name=tool_call_result.name,
                            is_sucess=reply_message.is_success,
                            reply=reply_message.content
                        )
                    )

                    if not reply_message.is_success:
                        tool_call_result.content = f"Task failed to complete. Reason: {reply_message.content}"
                        tool_call_result.is_error = True
                    else:
                        tool_call_result.content = reply_message.content
                        tool_call_result.is_error = False

        if tool_call_results:
            self._chat_history.extend([
                AssistantMessage(content=llm_result.content, source=self.id.type),
                FunctionExecutionResultMessage(content=tool_call_results),
            ])

        return done, reply_results, is_success
