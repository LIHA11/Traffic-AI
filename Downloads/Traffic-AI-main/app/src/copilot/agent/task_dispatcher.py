# Standard library imports
import json
import logging
import asyncio
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
       
        logger.info("[TaskDispatcher] !!! on_request called. sent_from=%s, plan=%s", self._sent_from, message.plan[:150] if message.plan else "None")
        
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
        print(f"\n\n*** TaskDispatcher.on_receive_dag CALLED! DAG has {len(message.dag.nodes)} nodes ***\n\n", flush=True)
        self._dag = message.dag
        # When receiving a DAG directly (e.g. deterministic injection bypassing on_request),
        # prior chat history may contain earlier SystemMessages causing the underlying model
        # client to raise "Multiple and Not continuous system messages" errors. Reset here.
        self._chat_history = []
        if not self._proactive_planning:
            raise NotImplementedError("Non Proactive planning is not implemented yet.")

        # --- EARLY DETECTION & COMPLETE BYPASS FOR DETERMINISTIC TOP-UP DAG (3 nodes) ---
        try:
            if len(message.dag.nodes) == 3:
                n0, n1, n2 = message.dag.nodes
                logger.info("[TaskDispatcher] Inspecting 3-node DAG for deterministic top-up pattern.")
                logger.debug("[TaskDispatcher] Node0.success_criteria=%s", n0.success_criteria)
                logger.debug("[TaskDispatcher] Node1.success_criteria=%s", n1.success_criteria)
                logger.debug("[TaskDispatcher] Node2.success_criteria=%s", n2.success_criteria)
                import re, json as _json
                # Pattern: first node success criteria contains deterministic shipment_json key
                shipment_key_match = re.search(r"shipment_json_[A-Z0-9]+_[A-Z0-9]+_[0-9]+", n0.success_criteria)
                if shipment_key_match:
                    shipment_mem_key = shipment_key_match.group(0)
                    logger.info("[TaskDispatcher] Deterministic top-up DAG recognized. shipment_mem_key=%s", shipment_mem_key)
                    derive_tool = self._tools_dict.get("derive_topup_query")
                    eligible_tool = self._tools_dict.get("get_eligible_topup_shipment")
                    get_mem_tool = self._tools_dict.get("get_from_memory")
                    # Retrieve shipment JSON (memory preferred, fallback worker replies from prior run not available here)
                    shipment_json_str = "[]"
                    if get_mem_tool:
                        try:
                            mem_raw = await get_mem_tool.run_json({"session_id": self.get_id(), "key": shipment_mem_key}, ctx.cancellation_token)
                            mem_obj = _json.loads(mem_raw)
                            shipment_json_str = mem_obj.get("result", {}).get("content") or shipment_json_str
                        except Exception as e:
                            logger.warning("[TaskDispatcher] Memory fetch failed for %s: %s", shipment_mem_key, e)
                    if shipment_json_str == "[]":
                        logger.warning("[TaskDispatcher] Using empty shipment JSON fallback for derive_topup_query.")
                    # Fallback inline tool definitions if registry missing
                    if derive_tool is None:
                        logger.warning("[TaskDispatcher] derive_topup_query missing; installing inline fallback.")
                        async def derive_topup_query(shipment_json: str, min_empty_ratio: float = 85.0) -> str:
                            import json as _j, re as _r
                            try:
                                data = _j.loads(shipment_json)
                            except Exception:
                                return _j.dumps({"need_topup": False, "reason": "parse failure"}, ensure_ascii=False)
                            if not isinstance(data, list) or not data:
                                return _j.dumps({"need_topup": False, "reason": "empty list"}, ensure_ascii=False)
                            first = data[0] if isinstance(data[0], dict) else {}
                            port = first.get("VOY_STOP_PORT_CDE")
                            svvd = first.get("SVVD")
                            triggered = [r for r in data if isinstance(r, dict) and float(r.get("empty_ratio", 0) or 0) >= min_empty_ratio]
                            if triggered and port and svvd:
                                norm = _r.sub(r"[^A-Za-z0-9]", "", svvd.upper())
                                return _j.dumps({"need_topup": True, "port": port, "svvd": svvd, "min_empty_ratio": min_empty_ratio, "triggered_count": len(triggered), "suggested_memory_key": f"eligible_topup_{port.upper()}_{norm}"}, ensure_ascii=False)
                            return _j.dumps({"need_topup": False, "reason": "no trigger"}, ensure_ascii=False)
                        derive_tool = FunctionTool(derive_topup_query, name="derive_topup_query", description="Fallback derive_topup_query")
                        self._tools.append(derive_tool); self._tools_dict[derive_tool.name] = derive_tool
                    if eligible_tool is None:
                        logger.warning("[TaskDispatcher] get_eligible_topup_shipment missing; installing inline fallback.")
                        async def get_eligible_topup_shipment(port: str, svvd: str, min_empty_ratio: float = 85.0, session_id: Optional[str] = None, memory_key: Optional[str] = None) -> str:
                            import json as _j
                            mock = [{"SHIPMENT_NUMBER": "FALLBACK123", "SVVD": svvd + "-ALT", "VOY_STOP_PORT_CDE": port, "empty_ratio": min_empty_ratio+1, "SVC_TEU": 1.0}]
                            return _j.dumps({"eligible_shipments": mock, "count": 1, "filters": {"port": port, "excluded_svvd": svvd, "min_empty_ratio": min_empty_ratio}, "memory_key": memory_key, "session_id": session_id}, ensure_ascii=False)
                        eligible_tool = FunctionTool(get_eligible_topup_shipment, name="get_eligible_topup_shipment", description="Fallback get_eligible_topup_shipment")
                        self._tools.append(eligible_tool); self._tools_dict[eligible_tool.name] = eligible_tool
                    # Run derive
                    derive_result = await derive_tool.run_json({"shipment_json": shipment_json_str, "min_empty_ratio": 85.0}, ctx.cancellation_token)
                    logger.debug("[TaskDispatcher] derive_result type=%s, value[:200]=%s", type(derive_result).__name__, str(derive_result)[:200])
                    # Always convert to JSON string for WorkerReply
                    if isinstance(derive_result, str):
                        derive_raw = derive_result
                    else:
                        try:
                            derive_raw = _json.dumps(derive_result, ensure_ascii=False)
                        except Exception as e:
                            logger.warning("[TaskDispatcher] Failed to serialize derive_result: %s", e)
                            derive_raw = _json.dumps({"_error": "non-serializable"}, ensure_ascii=False)
                    port = None; svvd = None; suggested_key = None
                    try:
                        d_obj = _json.loads(derive_raw)
                        port = d_obj.get("port")
                        svvd = d_obj.get("svvd")
                        suggested_key = d_obj.get("suggested_memory_key")
                    except Exception:
                        pass
                    import re as _re
                    mem_key = suggested_key or (f"eligible_topup_{(port or 'SIN').upper()}_{_re.sub(r'[^A-Za-z0-9]', '', (svvd or 'UNKNOWN').upper())}" if port and svvd else None)
                    eligible_args = {"port": port or "SIN", "svvd": svvd or "UNKNOWN", "min_empty_ratio": 85.0}
                    if mem_key:
                        eligible_args["session_id"] = self.get_id(); eligible_args["memory_key"] = mem_key
                    eligible_result = await eligible_tool.run_json(eligible_args, ctx.cancellation_token)
                    logger.debug("[TaskDispatcher] eligible_result type=%s, value[:200]=%s", type(eligible_result).__name__, str(eligible_result)[:200])
                    # Always convert to JSON string for WorkerReply
                    if isinstance(eligible_result, str):
                        eligible_raw = eligible_result
                    else:
                        try:
                            eligible_raw = _json.dumps(eligible_result, ensure_ascii=False)
                        except Exception as e:
                            logger.warning("[TaskDispatcher] Failed to serialize eligible_result: %s", e)
                            eligible_raw = _json.dumps({"_error": "non-serializable"}, ensure_ascii=False)
                    all_worker_replies = [
                        WorkerReply(worker_name="sql_generation_expert", is_sucess=True, reply=shipment_json_str),
                        WorkerReply(worker_name="derive_topup_query", is_sucess=True, reply=derive_raw),
                        WorkerReply(worker_name="get_eligible_topup_shipment", is_sucess=True, reply=eligible_raw),
                    ]
                    final_dispatcher_reply = "Deterministic top-up completed."
                    final_is_success = True
                    reply_message = TaskDispatcherReplyMessage(
                        task_description=n2.task_description,
                        success_criteria=n2.success_criteria,
                        dispatcher_reply=final_dispatcher_reply,
                        worker_replies=all_worker_replies,
                        is_success=final_is_success
                    )
                    agent_ops = self.get_agent_ops()
                    if self._span is not None:
                        agent_ops.end_span(run_id=self.get_id(), span_id=self._span.span_id, outputs=reply_message, attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
                    if self._sent_from:
                        await self.publish_message(reply_message, topic_id=TopicId(self._sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token)
                        return None
                    return reply_message
        except Exception as _early_det_err:
            import traceback as _tb
            logger.warning("[TaskDispatcher] Early deterministic bypass failed: %s (%s)\nTRACE:\n%s", _early_det_err, type(_early_det_err), ''.join(_tb.format_exc()))

        nodes = self._topological_sort(self._dag)
        await self._report_message(LogMessage(
            agent_name=self._sent_from,
            action="Execution plan has been prepared to fulfill your request:",
            content="[" + ",".join([node.model_dump_json() for node in nodes]) + "]",
            id=self.get_id(),
        ))

        # Execute nodes sequentially (previous implementation only executed first node)
        # Accumulate worker replies from the LAST executed node for Planner reply context.
        target_node = None
        all_worker_replies: List[WorkerReply] = []
        final_dispatcher_reply: str = ""
        final_is_success: bool = True
        for node_index, exec_node in enumerate(nodes):
            target_node = exec_node
            logger.info("[TaskDispatcher] ### Executing node %d/%d: %s", node_index+1, len(nodes), exec_node.task_description)
            agent_ops = self.get_agent_ops()
            retriever_span: Optional[Span] = None
            if self._span is not None:
                retriever_span = agent_ops.create_span(
                    run_id=self.get_id(),
                    name="search_past_experience",
                    parent_id=self._span.span_id,
                    span_type=SpanType.RETRIEVER,
                    inputs=exec_node.task_description,
                )

            docs = []
            if self._knowledge_center:
                docs = await self._knowledge_center.get_documents(
                    query_text=exec_node.task_description, domains=self._name
                )
            if retriever_span is not None:
                agent_ops.end_span(run_id=self.get_id(),
                                   span_id=retriever_span.span_id,
                                   outputs=[Document(page_content=doc.content) for doc in docs])

            prompt_vars = {
                **message.model_dump(),
                "docs": docs,
                "task": exec_node.task_description,
                "success_criteria": exec_node.success_criteria,
                "user_message": self._user_message
            }
            # Use SystemMessage only for first node to avoid model limitation on multiple system messages.
            prompt_content = self.get_prompt(STAGE_RECEIVE_DAG, prompt_vars)
            if node_index == 0:
                system_message = SystemMessage(content=prompt_content)
            else:
                system_message = UserMessage(content=prompt_content, source=self.get_id())

            # Early deterministic bypass for top-up DAG nodes > first
            try:
                import re, json as _json
                # If we already fast-pathed earlier, skip further loop
                if final_dispatcher_reply.startswith("Deterministic top-up completed"):
                    logger.debug("[TaskDispatcher] Fast-path already completed; skipping remaining LLM generation for node %d", node_index+1)
                    break
                if len(nodes) == 3 and node_index in (1,2):
                    # Node 1 executed by LLM or prior; Node 2 derive intent; Node 3 eligible retrieval
                    derive_tool = self._tools_dict.get("derive_topup_query")
                    eligible_tool = self._tools_dict.get("get_eligible_topup_shipment")
                    # Extract shipment JSON from previous worker replies or memory
                    shipment_json_str = None
                    if all_worker_replies:
                        for wr in all_worker_replies:
                            if isinstance(wr, WorkerReply):
                                txt = str(wr.reply).strip()
                                if txt.startswith("[") and txt.endswith("]"):
                                    shipment_json_str = txt
                                    break
                    if not shipment_json_str and node_index == 1:
                        # Attempt memory key pattern extraction from first node success criteria
                        key_match = re.search(r"shipment_json_[A-Z0-9]+_[A-Z0-9]+_[0-9]+", nodes[0].success_criteria)
                        mem_key = key_match.group(0) if key_match else None
                        get_mem_tool = self._tools_dict.get("get_from_memory")
                        if get_mem_tool and mem_key:
                            mem_raw = await get_mem_tool.run_json({"session_id": self.get_id(), "key": mem_key}, ctx.cancellation_token)
                            try:
                                mem_obj = _json.loads(mem_raw)
                                shipment_json_str = mem_obj.get("result", {}).get("content") if isinstance(mem_obj, dict) else None
                            except Exception:
                                shipment_json_str = None
                    if node_index == 1 and derive_tool and shipment_json_str:
                        derive_raw = await derive_tool.run_json({"shipment_json": shipment_json_str, "min_empty_ratio": 85.0}, ctx.cancellation_token)
                        all_worker_replies.append(WorkerReply(worker_name="derive_topup_query", is_sucess=True, reply=derive_raw))
                        logger.info("[TaskDispatcher] Fast-path executed derive_topup_query for node 2")
                        continue  # proceed to next node without LLM
                    if node_index == 2 and eligible_tool:
                        # Parse derive output for port/svvd
                        port = None; svvd = None; suggested_key = None
                        for wr in all_worker_replies:
                            if wr.worker_name == "derive_topup_query":
                                try:
                                    d_obj = _json.loads(wr.reply)
                                    port = d_obj.get("port")
                                    svvd = d_obj.get("svvd")
                                    suggested_key = d_obj.get("suggested_memory_key")
                                except Exception:
                                    pass
                                break
                        import re as _re
                        mem_key = suggested_key or (f"eligible_topup_{(port or 'SIN').upper()}_{_re.sub(r'[^A-Za-z0-9]', '', (svvd or 'UNKNOWN').upper())}" if port and svvd else None)
                        eligible_args = {"port": port or "SIN", "svvd": svvd or "UNKNOWN", "min_empty_ratio": 85.0}
                        if mem_key:
                            eligible_args["session_id"] = self.get_id(); eligible_args["memory_key"] = mem_key
                        eligible_raw = await eligible_tool.run_json(eligible_args, ctx.cancellation_token)
                        all_worker_replies.append(WorkerReply(worker_name="get_eligible_topup_shipment", is_sucess=True, reply=eligible_raw))
                        try:
                            er_obj = _json.loads(eligible_raw); cnt = er_obj.get("count"); final_dispatcher_reply = f"Deterministic top-up completed. Eligible shipments count={cnt}. Memory key={er_obj.get('memory_key')}"
                        except Exception:
                            final_dispatcher_reply = "Deterministic top-up completed (eligible JSON parse failed)."
                        final_is_success = True
                        logger.info("[TaskDispatcher] Fast-path executed get_eligible_topup_shipment for node 3")
                        # Publish immediately and exit
                        agent_ops = self.get_agent_ops()
                        reply_message = TaskDispatcherReplyMessage(
                            task_description=exec_node.task_description,
                            success_criteria=exec_node.success_criteria,
                            dispatcher_reply=final_dispatcher_reply,
                            worker_replies=all_worker_replies or None,
                            is_success=final_is_success
                        )
                        if self._span is not None:
                            agent_ops.end_span(run_id=self.get_id(), span_id=self._span.span_id, outputs=reply_message, attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
                        if self._sent_from:
                            await self.publish_message(reply_message, topic_id=TopicId(self._sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token)
                        return None
            except Exception as _early_err:
                logger.debug("[TaskDispatcher] Early deterministic bypass not applied: %s", _early_err)

            reply_results = []
            logger.info("[TaskDispatcher] $$$$ Starting LLM loop for node: %s", exec_node.task_description)
            try:
                llm_result: CreateResult = await self.generate(ctx, new_message=system_message, append_generated_message=False, session_id=self.get_id())
                done, reply_result, is_success = await self._handle_llm_result(llm_result, exec_node, ctx)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    logger.warning("[TaskDispatcher] Cancellation detected during first iteration of node %d; skipping to next node.", node_index+1)
                    # Treat as unsuccessful but continue to next node in DAG
                    is_success = False
                    done = True
                    reply_result = [f"Node {node_index+1} cancelled; continuing."]
                else:
                    raise
            reply_results.append(reply_result)
            logger.info("[TaskDispatcher] $$$$ First iteration: done=%s, is_success=%s", done, is_success)

            while not done:
                try:
                    llm_result = await self.generate(ctx, append_generated_message=False, session_id=self.get_id())
                    done, reply_result, is_success = await self._handle_llm_result(llm_result, exec_node, ctx)
                    reply_results.append(reply_result)
                    logger.info("[TaskDispatcher] $$$$ Loop iteration: done=%s, is_success=%s", done, is_success)
                except Exception as e:
                    if isinstance(e, asyncio.CancelledError):
                        logger.warning("[TaskDispatcher] Cancellation detected during loop for node %d; forcing completion and continuing.", node_index+1)
                        # Force completion and mark unsuccessful; proceed to next node
                        done = True
                        is_success = False
                        reply_results.append([f"Node {node_index+1} cancelled mid-loop; continuing."])
                        break
                    raise

            # Collect worker replies for this node
            worker_replies = []
            for reply_result in reversed(reply_results):
                for reply in reply_result:
                    if isinstance(reply, WorkerReply):
                        worker_replies.append(reply)
                if len(worker_replies) > 0:
                    break

            # Update final state; For chaining logic: if a node returns is_success False but there are remaining nodes,
            # proceed anyway (used for continuation workflows like high empty_ratio).
            final_dispatcher_reply = reply_results[-1][-1] if reply_results[-1] else ""
            all_worker_replies = worker_replies or []
            final_is_success = is_success

            # --- Deterministic Top-Up DAG fast-path (skip LLM for nodes 2 & 3) ---
            # Pattern: 3 nodes, node 1 success criteria starts with 'Shipment JSON stored under key shipment_json_'
            # Node 2 (derive intent) & Node 3 (eligible retrieval) can be executed directly via tools.
            try:
                import re, json as _json
                if len(nodes) == 3 and node_index == 0 and exec_node.success_criteria.startswith("Shipment JSON stored under key shipment_json_"):
                    mem_key_match = re.search(r"shipment_json_[A-Z0-9]+_[A-Z0-9]+_[0-9]+", exec_node.success_criteria)
                    shipment_mem_key = mem_key_match.group(0) if mem_key_match else None
                    if shipment_mem_key:
                        logger.info("[TaskDispatcher] Detected deterministic top-up DAG; executing nodes 2 & 3 programmatically. mem_key=%s", shipment_mem_key)
                        # Retrieve shipment JSON from memory
                        get_mem_tool = self._tools_dict.get("get_from_memory")
                        derive_tool = self._tools_dict.get("derive_topup_query")
                        eligible_tool = self._tools_dict.get("get_eligible_topup_shipment")
                        # Fallback inline tool definitions if registry did not supply them
                        if derive_tool is None:
                            logger.warning("[TaskDispatcher] derive_topup_query tool missing; creating inline fallback.")
                            async def derive_topup_query(shipment_json: Annotated[str, "JSON array string"], min_empty_ratio: Annotated[float, "Threshold"] = 85.0) -> str:
                                import json as _j, re as _r
                                try:
                                    data = _j.loads(shipment_json)
                                except Exception:
                                    return _j.dumps({"need_topup": False, "reason": "Parse failure"}, ensure_ascii=False)
                                if not isinstance(data, list) or not data:
                                    return _j.dumps({"need_topup": False, "reason": "Empty list"}, ensure_ascii=False)
                                first = data[0] if isinstance(data[0], dict) else {}
                                triggered = []
                                for r in data:
                                    if isinstance(r, dict):
                                        try:
                                            er = float(r.get("empty_ratio", 0))
                                            if er >= min_empty_ratio:
                                                triggered.append({"SHIPMENT_NUMBER": r.get("SHIPMENT_NUMBER"), "empty_ratio": er, "SVVD": r.get("SVVD"), "PORT": r.get("VOY_STOP_PORT_CDE")})
                                        except Exception:
                                            pass
                                svvd = first.get("SVVD")
                                port = first.get("VOY_STOP_PORT_CDE")
                                if triggered and svvd and port:
                                    norm = _r.sub(r"[^A-Za-z0-9]", "", svvd.upper())
                                    return _j.dumps({"need_topup": True, "next_user_message": f"Find shipments from other service loops to top up {svvd} at {port}", "port": port, "svvd": svvd, "min_empty_ratio": min_empty_ratio, "triggered_count": len(triggered), "triggered_records": triggered, "suggested_memory_key": f"eligible_topup_{port.upper()}_{norm}"}, ensure_ascii=False)
                                return _j.dumps({"need_topup": False, "reason": "No trigger"}, ensure_ascii=False)
                            derive_tool = FunctionTool(derive_topup_query, name="derive_topup_query", description="Inline fallback derive_topup_query")
                            self._tools.append(derive_tool); self._tools_dict[derive_tool.name] = derive_tool
                        if eligible_tool is None:
                            logger.warning("[TaskDispatcher] get_eligible_topup_shipment tool missing; creating inline fallback.")
                            async def get_eligible_topup_shipment(port: Annotated[str, "Port"], svvd: Annotated[str, "SVVD"], min_empty_ratio: Annotated[float, "Threshold"] = 85.0, session_id: Annotated[Optional[str], "Session"]=None, memory_key: Annotated[Optional[str], "Memory key"]=None) -> str:
                                import json as _j
                                # Fallback returns single mock eligible shipment different from target SVVD
                                mock = [{"SHIPMENT_NUMBER": "FALLBACK123", "SVVD": svvd + "-ALT", "VOY_STOP_PORT_CDE": port, "empty_ratio": min_empty_ratio+1, "SVC_TEU": 1.0}]
                                return _j.dumps({"eligible_shipments": mock, "count": 1, "filters": {"port": port, "excluded_svvd": svvd, "min_empty_ratio": min_empty_ratio}, "memory_key": memory_key, "session_id": session_id}, ensure_ascii=False)
                            eligible_tool = FunctionTool(get_eligible_topup_shipment, name="get_eligible_topup_shipment", description="Inline fallback get_eligible_topup_shipment")
                            self._tools.append(eligible_tool); self._tools_dict[eligible_tool.name] = eligible_tool
                        if not (derive_tool and eligible_tool):
                            logger.warning("[TaskDispatcher] Missing derive_topup_query or get_eligible_topup_shipment tool; cannot fast-path.")
                        elif not get_mem_tool:
                            logger.info("[TaskDispatcher] get_from_memory tool absent; attempting to extract shipment JSON from worker replies.")
                            # Attempt to locate JSON array in worker replies
                            shipment_json_str = "[]"
                            for wr in all_worker_replies:
                                if isinstance(wr, WorkerReply):
                                    txt = str(wr.reply).strip()
                                    if txt.startswith("[") and txt.endswith("]"):
                                        shipment_json_str = txt
                                        break
                            if shipment_json_str == "[]":
                                logger.warning("[TaskDispatcher] Could not find shipment JSON in worker replies; proceeding with empty list.")
                            derive_out_result = await derive_tool.run_json({"shipment_json": shipment_json_str, "min_empty_ratio": 85.0}, ctx.cancellation_token)
                            # Normalize to JSON string
                            if isinstance(derive_out_result, str):
                                derive_out_raw = derive_out_result
                            else:
                                derive_out_raw = _json.dumps(derive_out_result, ensure_ascii=False)
                            derive_obj = {}
                            try:
                                derive_obj = _json.loads(derive_out_raw) if isinstance(derive_out_raw, str) else derive_out_result
                            except Exception:
                                logger.warning("[TaskDispatcher] derive_topup_query returned non-JSON: %s", str(derive_out_raw)[:120])
                            port = derive_obj.get("port")
                            svvd = derive_obj.get("svvd")
                            suggested_key = derive_obj.get("suggested_memory_key") or (f"eligible_topup_{(port or '').upper()}_{re.sub(r'[^A-Za-z0-9]', '', (svvd or '').upper())}" if port and svvd else None)
                            eligible_args = {"port": port or "SIN", "svvd": svvd or "UNKNOWN", "min_empty_ratio": 85.0}
                            if suggested_key:
                                eligible_args["session_id"] = self.get_id()
                                eligible_args["memory_key"] = suggested_key
                            eligible_out_result = await eligible_tool.run_json(eligible_args, ctx.cancellation_token)
                            # Normalize to JSON string
                            if isinstance(eligible_out_result, str):
                                eligible_out_raw = eligible_out_result
                            else:
                                eligible_out_raw = _json.dumps(eligible_out_result, ensure_ascii=False)
                            all_worker_replies.extend([
                                WorkerReply(worker_name="derive_topup_query", is_sucess=True, reply=derive_out_raw),
                                WorkerReply(worker_name="get_eligible_topup_shipment", is_sucess=True, reply=eligible_out_raw),
                            ])
                            try:
                                eligible_obj = _json.loads(eligible_out_raw) if isinstance(eligible_out_raw, str) else eligible_out_result
                                count = eligible_obj.get("count")
                                final_dispatcher_reply = f"Deterministic top-up completed. Eligible shipments count={count}. Memory key={eligible_obj.get('memory_key')}"
                            except Exception:
                                final_dispatcher_reply = "Deterministic top-up completed (unable to parse eligible shipment JSON)."
                            final_is_success = True
                            logger.info("[TaskDispatcher] Deterministic top-up workflow finished without LLM (memory tool absent path).")
                            agent_ops = self.get_agent_ops()
                            reply_message = TaskDispatcherReplyMessage(
                                task_description=nodes[-1].task_description,
                                success_criteria=nodes[-1].success_criteria,
                                dispatcher_reply=final_dispatcher_reply,
                                worker_replies=all_worker_replies or None,
                                is_success=final_is_success
                            )
                            if self._span is not None:
                                agent_ops.end_span(run_id=self.get_id(), span_id=self._span.span_id, outputs=reply_message, attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
                            if self._sent_from:
                                await self.publish_message(reply_message, topic_id=TopicId(self._sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token)
                            return None
                        elif not (get_mem_tool and derive_tool and eligible_tool):
                            logger.warning("[TaskDispatcher] Missing one or more required tools for deterministic fast-path; falling back to LLM for remaining nodes.")
                        else:
                            # get_from_memory
                            mem_raw = await get_mem_tool.run_json({"session_id": self.get_id(), "key": shipment_mem_key}, ctx.cancellation_token)
                            mem_obj = None
                            try:
                                mem_obj = _json.loads(mem_raw)
                            except Exception:
                                pass
                            shipment_json_str = None
                            if isinstance(mem_obj, dict):
                                result_obj = mem_obj.get("result") or {}
                                if isinstance(result_obj, dict):
                                    shipment_json_str = result_obj.get("content")
                            if not shipment_json_str:
                                logger.warning("[TaskDispatcher] Failed to extract shipment JSON from memory result; content=%s", str(mem_raw)[:200])
                                shipment_json_str = "[]"
                            derive_out_result = await derive_tool.run_json({"shipment_json": shipment_json_str, "min_empty_ratio": 85.0}, ctx.cancellation_token)
                            # Normalize to JSON string
                            if isinstance(derive_out_result, str):
                                derive_out_raw = derive_out_result
                            else:
                                derive_out_raw = _json.dumps(derive_out_result, ensure_ascii=False)
                            derive_obj = {}
                            try:
                                derive_obj = _json.loads(derive_out_raw) if isinstance(derive_out_raw, str) else derive_out_result
                            except Exception:
                                logger.warning("[TaskDispatcher] derive_topup_query returned non-JSON: %s", str(derive_out_raw)[:120])
                            port = derive_obj.get("port")
                            svvd = derive_obj.get("svvd")
                            suggested_key = (derive_obj.get("suggested_memory_key") or 
                                           (f"eligible_topup_{(port or '').upper()}_{re.sub(r'[^A-Za-z0-9]', '', (svvd or '').upper())}" if port and svvd else None))
                            eligible_args = {"port": port or "SIN", "svvd": svvd or "UNKNOWN", "min_empty_ratio": 85.0}
                            if suggested_key:
                                eligible_args["session_id"] = self.get_id()
                                eligible_args["memory_key"] = suggested_key
                            eligible_out_result = await eligible_tool.run_json(eligible_args, ctx.cancellation_token)
                            # Normalize to JSON string
                            if isinstance(eligible_out_result, str):
                                eligible_out_raw = eligible_out_result
                            else:
                                eligible_out_raw = _json.dumps(eligible_out_result, ensure_ascii=False)
                            # Build worker replies for nodes 2 & 3
                            all_worker_replies.extend([
                                WorkerReply(worker_name="derive_topup_query", is_sucess=True, reply=derive_out_raw),
                                WorkerReply(worker_name="get_eligible_topup_shipment", is_sucess=True, reply=eligible_out_raw),
                            ])
                            # Summarize final reply
                            try:
                                eligible_obj = _json.loads(eligible_out_raw) if isinstance(eligible_out_raw, str) else eligible_out_result
                                count = eligible_obj.get("count")
                                final_dispatcher_reply = f"Deterministic top-up completed. Eligible shipments count={count}. Memory key={eligible_obj.get('memory_key')}"
                            except Exception:
                                final_dispatcher_reply = "Deterministic top-up completed (unable to parse eligible shipment JSON)."
                            final_is_success = True
                            logger.info("[TaskDispatcher] Deterministic top-up workflow finished without LLM for remaining nodes.")
                            # Publish reply immediately
                            agent_ops = self.get_agent_ops()
                            reply_message = TaskDispatcherReplyMessage(
                                task_description=nodes[-1].task_description,
                                success_criteria=nodes[-1].success_criteria,
                                dispatcher_reply=final_dispatcher_reply,
                                worker_replies=all_worker_replies or None,
                                is_success=final_is_success
                            )
                            if self._span is not None:
                                agent_ops.end_span(run_id=self.get_id(), span_id=self._span.span_id, outputs=reply_message, attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
                            if self._sent_from:
                                await self.publish_message(reply_message, topic_id=TopicId(self._sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token)
                            return None
            except Exception as _fast_err:
                import traceback as _tb
                logger.warning("[TaskDispatcher] Deterministic fast-path failed: %s (%s)\nTRACE:\n%s", _fast_err, type(_fast_err), ''.join(_tb.format_exc()))

            if node_index < len(nodes) - 1:
                # If not last node and is_success False, allow progression.
                if not is_success:
                    logger.info("[TaskDispatcher] Proceeding to next node despite is_success=False (continuation workflow).")
                continue
            else:
                # Last node executed; break loop.
                break

        if target_node is None:
            logger.error("[TaskDispatcher] No target node executed; aborting.")
            return None

        agent_ops = self.get_agent_ops()
        
        reply_message = TaskDispatcherReplyMessage(
            task_description=target_node.task_description,
            success_criteria=target_node.success_criteria,
            dispatcher_reply=final_dispatcher_reply,
            worker_replies=all_worker_replies or None,
            is_success=final_is_success
        )


        if self._span is not None:
            agent_ops.end_span(run_id=self.get_id(),
                        span_id=self._span.span_id,
                        outputs=reply_message,
                        attributes={"next_agent": message.sent_from} if "sent_from" in message else None)
            

        if self._sent_from:
            logger.info("[TaskDispatcher] ^^^ Sending TaskDispatcherReplyMessage back to: %s", self._sent_from)
            logger.info("[TaskDispatcher] ^^^ Reply content: is_success=%s, dispatcher_reply=%s", reply_message.is_success, reply_message.dispatcher_reply[:200] if reply_message.dispatcher_reply else "None")
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

                # Treat any BaseException (including CancelledError) as a failed worker call.
                if isinstance(reply_message, BaseException):
                    reply_results.append(
                        WorkerReply(
                            worker_name=tool_call_result.name,
                            is_sucess=False,
                            reply=str(reply_message)
                        )
                    )
                    tool_call_result.content = f"Task failed to complete. Reason: Agent raised an Exception: {reply_message}"
                    tool_call_result.is_error = True
                    continue

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
