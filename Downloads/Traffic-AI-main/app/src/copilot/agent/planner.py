# Standard library imports
import logging
import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Set,
)

print("\n\n##### planner.py MODULE LOADED #####\n\n", flush=True)

# Third-party library imports
from langfuse import Langfuse
from pydantic import BaseModel
from mlflow.entities.span import SpanType
from mlflow.entities import SpanType, Document

# Project-specific imports
from autogen_core import (
    MessageContext,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
)
from autogen_core.models import (
    CreateResult,
    SystemMessage,
    UserMessage,
)
from src.connector.agentops.agentops import LogMessage
from src.copilot.agent.agent import Agent
from src.copilot.agent.reporter import REPORTER_TOPIC, ReporterRequestMessage
from src.copilot.agent.task_dispatcher import (
    AGENT_NAME,
    TaskDispatcherReplyMessage,
    TaskDispatcherRequestMessage,
)
from src.copilot.agent.dag_generator import Node, DAG, Edge, DAGGeneratorReplyMessage
from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.connector.agentops.langfuse_ops import LangfuseOps
from mlflow.entities import Span

logger = logging.getLogger(__name__)

# --- Constants ---
PLANNER_REQUEST = "planner_request"
HUMAN_SENDER = "human"
REQUIRED_PROMPT_TEMPLATES = [PLANNER_REQUEST]

# --- Message Models ---
class PlannerRequestMessage(BaseModel):
    task: str
    sent_from: Optional[str] = None
    domain_knowledges: Optional[str] = None

class PlannerReplyMessage(BaseModel):
    content: str
    is_success: bool
    
# --- Planner Agent ---
class Planner(Agent):
    async def on_message(self, message: Any, ctx: MessageContext) -> Any:
        """Override to log all incoming messages for debugging."""
        logger.info("[Planner] @@@ Received message type: %s", type(message).__name__)
        return await super().on_message(message, ctx)
    
    def __init__(
        self,
        name: str,
        description: str,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        knowledge_center: Optional[KnowledgeCenter] = None,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ):        
        # Ensure all required prompt templates are provided
        for required in REQUIRED_PROMPT_TEMPLATES:
            if required not in prompt_templates:
                raise ValueError(f"Missing required prompt template: '{required}'")
        
        super().__init__(
            name=name,
            description=description,
            chat_client=chat_client,
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message
        )
        self._task: Optional[Any] = None
        self._sent_from: Optional[Any] = None
        self._knowledge_center: Optional[KnowledgeCenter] = knowledge_center
        # Track (port, svvd) pairs already scheduled for top-up to avoid duplication
        self._scheduled_topup_pairs: Set[Tuple[str, str]] = set()
        
        logger.info("[Planner.__init__] +++++ Planner initialized: name=%s", self._name)
        logger.debug(
            f"Planner initialized: name={self._name}, description={self._description}"
        )
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        name: str,
        description: str,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        knowledge_center: Optional[KnowledgeCenter] = None,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ) -> 'Planner':
        return await Planner.register(
            runtime,
            type=name,
            factory = lambda: Planner(
                name=name,
                description=description,
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                knowledge_center=knowledge_center,
                agent_ops=agent_ops,
                report_message=report_message
            )
        )
        
    @message_handler
    async def on_task_dispatcher_reply(
        self,
        message: TaskDispatcherReplyMessage,
        ctx: MessageContext
    ) -> Optional[Union[TaskDispatcherRequestMessage, ReporterRequestMessage, PlannerReplyMessage]]:
        """
        Handles TaskDispatcher reply messages and generates an appropriate response.
        """
        print(f"\n\n===== PLANNER on_task_dispatcher_reply CALLED! is_success={message.is_success} =====\n\n", flush=True)
        logger.info("[Planner] ===== on_task_dispatcher_reply CALLED! Task: %s, Success: %s", message.task_description, message.is_success)
        # Build the response content based on the task status
        content = (
            f"Task: {message.task_description}\n"
            f"Successful Status: {message.is_success}\n"
            f"{'Result' if message.is_success else 'Reason'}: {message.dispatcher_reply}\n"
        )
        
        # Separate succeeded and failed workers
        succeeded = []
        failed = []
        
        if message.worker_replies:
            for wr in message.worker_replies:
                # Ensure is_sucess is a boolean
                if getattr(wr, 'is_sucess', False):
                    succeeded.append(wr)
                else:
                    failed.append(wr)
                logger.debug("[Planner] Worker reply from %s (success=%s): %s", getattr(wr, 'worker_name', 'unknown'), getattr(wr, 'is_sucess', False), wr.reply)
                    
        def format_reply(reply):
            # Indent all lines of the reply by 4 spaces
            return '\n'.join('' + line for line in str(reply).splitlines())

        if succeeded:
            content += "Succeeded Workers:\n"
            for wr in succeeded:
                content += f"- {wr.worker_name}:\n{format_reply(wr.reply)}\n\n"

        if failed:
            content += "Failed Workers:\n"
            for wr in failed:
                content += f"- {wr.worker_name}:\n{format_reply(wr.reply)}\n\n"
        # --- Deterministic Top-Up Injection (bypass LLM when threshold met) ---
        # Conditions (conceptual):
        # 1. Previous worker produced shipment JSON (typically sql_generation_expert for shipment details).
        # 2. Any record has empty_ratio >= 85.
        # 3. (port, svvd) pair not already scheduled for top-up in this Planner instance.
        if True:
            import json
            import re

            def _extract_shipments(raw: Union[str, Dict[str, Any], List[Any]]) -> Optional[List[Dict[str, Any]]]:
                """Best-effort extraction of a shipment list from raw content.

                - Accepts a pure JSON list, or a JSON list embedded in text.
                - Returns a list[dict] when successful, otherwise None.
                """
                if isinstance(raw, str):
                    text = raw.strip()
                    parsed_obj: Any = None
                    # Try direct JSON first
                    try:
                        parsed_obj = json.loads(text)
                    except Exception:
                        # Fallback: scan for first JSON array/object and raw_decode from there
                        decoder = json.JSONDecoder()
                        for idx, ch in enumerate(text):
                            if ch not in "[{":
                                continue
                            try:
                                candidate, _ = decoder.raw_decode(text[idx:])
                                parsed_obj = candidate
                                break
                            except json.JSONDecodeError:
                                continue
                        if parsed_obj is None:
                            return None
                else:
                    parsed_obj = raw

                if isinstance(parsed_obj, list) and parsed_obj and all(isinstance(item, dict) for item in parsed_obj):
                    return parsed_obj  # type: ignore[return-value]
                return None

            # --- Diagnostics container ---
            debug_reasons: List[str] = []
            candidate_payloads: List[List[Dict[str, Any]]] = []

            # Helper: explicit locate JSON after 'Copilot:' if generic parse fails
            def _fallback_copilot_json(text: str) -> Optional[List[Dict[str, Any]]]:
                try:
                    m = re.search(r"Copilot:\s*(\[.*\])", text, flags=re.DOTALL)
                    if not m:
                        return None
                    arr_txt = m.group(1)
                    return json.loads(arr_txt) if arr_txt.strip().startswith("[") else None
                except Exception as e:
                    debug_reasons.append(f"fallback_copilot_json failed: {e}")
                    return None

            # 1) Try dispatcher_reply
            candidate_payloads: List[List[Dict[str, Any]]] = []

            # 1) Try dispatcher_reply (often a summary; may or may not contain embedded JSON)
            logger.debug("[Planner] Dispatcher reply content: %s", message.dispatcher_reply)
            shipments_from_reply = _extract_shipments(message.dispatcher_reply)
            if shipments_from_reply:
                candidate_payloads.append(shipments_from_reply)
            else:
                fb = _fallback_copilot_json(str(message.dispatcher_reply)) if isinstance(message.dispatcher_reply, str) else None
                if fb:
                    candidate_payloads.append(fb)
                    debug_reasons.append("dispatcher_reply parsed via fallback_copilot_json")
                else:
                    debug_reasons.append("dispatcher_reply no shipments parsed")

            # 2) Prefer worker replies, which TaskDispatcher builds directly from ExecutorReplyMessage.content
            # IMPORTANT: High empty_ratio shipments are intentionally marked is_success False in Executor to force follow-up planning.
            # Therefore we must parse BOTH succeeded and failed worker replies for shipment JSON.
            all_workers = []
            all_workers.extend(succeeded)
            # Include failed replies too; they may carry the shipment JSON when high empty_ratio triggered continuation.
            all_workers.extend(failed)
            for worker in all_workers:
                shipments_from_worker = _extract_shipments(worker.reply)
                if shipments_from_worker:
                    logger.info(
                        "[Planner] Parsed shipment payload from worker %s with %d record(s)",
                        getattr(worker, 'worker_name', 'unknown'),
                        len(shipments_from_worker),
                    )
                    candidate_payloads.append(shipments_from_worker)
                else:
                    fbw = _fallback_copilot_json(str(worker.reply)) if isinstance(worker.reply, str) else None
                    if fbw:
                        candidate_payloads.append(fbw)
                        debug_reasons.append(f"worker {getattr(worker,'worker_name','unknown')} parsed via fallback_copilot_json")
                    else:
                        debug_reasons.append(f"worker {getattr(worker,'worker_name','unknown')} no shipments parsed")

            # 3) For any shipment list with high empty_ratio, inject a deterministic top-up plan
            logger.info("[Planner] Top-up injection diagnostics: payload_sets=%d reasons=%s", len(candidate_payloads), "; ".join(debug_reasons))
            for payload in candidate_payloads:
                try:
                    triggering = [r for r in payload if float(r.get("empty_ratio", 0) or 0) >= 85.0]
                except Exception as e:
                    logger.warning("[Planner] Failed evaluating empty_ratio trigger: %s", e)
                    continue
                if not triggering:
                    logger.debug("[Planner] Payload skipped: no record meets empty_ratio threshold")
                    continue
                trig = triggering[0]
                port = trig.get("VOY_STOP_PORT_CDE")
                svvd = trig.get("SVVD")
                if not isinstance(port, str) or not isinstance(svvd, str):
                    logger.debug("[Planner] Skipping trigger: port or SVVD missing (%s, %s)", port, svvd)
                    continue
                norm_svvd_sched = re.sub(r"[^A-Za-z0-9]", "", svvd.upper())
                pair = (port.upper(), norm_svvd_sched)
                if pair in self._scheduled_topup_pairs:
                    logger.info("[Planner] Top-up already scheduled for normalized pair (%s, %s); skipping duplicate.", port, norm_svvd_sched)
                    continue
                self._scheduled_topup_pairs.add(pair)
                memory_key = f"eligible_topup_{port.upper()}_{norm_svvd_sched}"
                # Revised deterministic plan to ensure shipment JSON is freshly available for handoff_agent:
                # 1) Re-run targeted SQL to retrieve the single shipment details (guarantees memory availability).
                # 2) Handoff agent derives top-up query intent using freshly stored JSON.
                # 3) Retrieve eligible top-up shipments.
                plan_parts = [
                    f"Subtask 1: Description: Generate and execute a SQL query to retrieve details for shipment number {trig.get('SHIPMENT_NUMBER')} at port {port} (table NOE_TFC_COPILOT_CMS_DATA). Dependencies: None. Success Criteria: Shipment JSON stored under key shipment_json_{port.upper()}_{re.sub(r'[^A-Za-z0-9]', '', svvd.upper())}_{trig.get('SHIPMENT_NUMBER')}.",
                    "Subtask 2: Description: Invoke handoff_agent to derive top-up query intent using the deterministic shipment_json_* memory key (no modification). Dependencies: Subtask 1. Success Criteria: Produce need_topup decision and suggested memory key (or reason when not needed).",
                    f"Subtask 3: Description: Retrieve eligible top-up shipments (exclude special cargo DG/RD/AD/PCT) for SVVD {svvd} at port {port} using get_eligible_topup_shipment with parameters port={port}, svvd={svvd}, min_empty_ratio=85, memory_key={memory_key}. Dependencies: Subtask 2. Success Criteria: Store result JSON in shared memory under {memory_key} and return summary.",
                ]
                # Previous implementation relied on LLM parsing of a textual plan -> DAG.
                # To eliminate ordering/merging ambiguity, we now construct the DAG deterministically and
                # send it directly to TaskDispatcher, bypassing the DAGGenerator LLM.
                nodes: List[Node] = []
                edges: List[Edge] = []
                for idx, part in enumerate(plan_parts):
                    # Extract task_description and success_criteria segments.
                    # Format: "Subtask X: Description: <desc>. Dependencies: <deps>. Success Criteria: <criteria>."
                    desc = ""
                    crit = ""
                    try:
                        # Split once on 'Success Criteria:' to isolate criteria
                        before_sc, after_sc = part.split("Success Criteria:", 1)
                        crit = after_sc.strip().rstrip('.')
                        # Remove leading subtask + description labels
                        m = re.search(r"Description:\s*(.*?)\.\s*Dependencies:", before_sc)
                        if m:
                            desc = m.group(1).strip()
                        else:
                            # Fallback: take entire pre-success part sans prefix
                            desc = before_sc.strip()
                    except Exception as _parse_err:
                        logger.warning("[Planner] Failed to parse deterministic plan part %d: %s", idx+1, _parse_err)
                        desc = part.strip()
                        crit = "Must satisfy stated success criteria."
                    nodes.append(Node(task_description=desc, success_criteria=crit))
                # Build linear edges 0->1, 1->2, ...
                for i in range(len(nodes) - 1):
                    edges.append(Edge(source=i, target=i+1))
                deterministic_dag = DAG(nodes=nodes, edges=edges)
                dag_reply = DAGGeneratorReplyMessage(dag=deterministic_dag)
                # IMPORTANT: TaskDispatcher instance is named 'task_dispatcher_<planner_name>'
                # so we must target that composite topic, not just 'task_dispatcher'.
                dispatcher_topic = f"{AGENT_NAME}_{self._name}"
                await self.publish_message(
                    dag_reply,
                    topic_id=TopicId(dispatcher_topic, source=self.get_id()),
                    cancellation_token=ctx.cancellation_token,
                )
                # Emit explicit log message to help non-interactive runner detect injection
                try:
                    if self._report_message:
                        await self._report_message(LogMessage(
                            agent_name=self._name,
                            action="Injected deterministic top-up DAG",
                            content=json.dumps({
                                "port": port,
                                "svvd": svvd,
                                "normalized_svvd": norm_svvd_sched,
                                "memory_key": memory_key,
                                "trigger_empty_ratio": float(trig.get("empty_ratio", 0) or 0),
                                "plan_lines": len(plan_parts),
                                "nodes": [n.task_description for n in nodes]
                            }, ensure_ascii=False),
                            id=self.get_id(),
                            is_complete=False
                        ))
                except Exception as _log_err:
                    logger.debug("[Planner] Failed to emit injection log: %s", _log_err)
                logger.info("[Planner] Injected deterministic top-up DAG for (%s, %s) memory_key=%s trigger_empty_ratio=%.2f", port, svvd, memory_key, float(trig.get("empty_ratio", 0) or 0))
                return None

        # Fall back to LLM-driven planning
        llm_result: CreateResult = await self.generate(
            ctx,
            new_message=UserMessage(content=content, source=self.get_id()),
            session_id=self.get_id()
        )
        return await self._handle_and_publish(llm_result, ctx)

    @message_handler
    async def on_request(
        self, 
        message: PlannerRequestMessage, 
        ctx: MessageContext
    ) -> Optional[Union[TaskDispatcherRequestMessage, ReporterRequestMessage, PlannerReplyMessage]]:
        """
        Handle Planner request messages, retrieve few-shot samples, 
        prepare prompt, and generate a response.
        """
        agent_ops = self.get_agent_ops()
        self._span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )
        
        self._sent_from = message.sent_from
        self._task = message.task
        self._chat_history = []

        # Retrieve few-shot samples if knowledge center is available
        retriever_span: Optional[Span] = None
        if self._span is not None:
            retriever_span = agent_ops.create_span(
                run_id=self.get_id(),
                name="search_past_experience",
                inputs=message.task,
                parent_id=self._span.span_id if self._span is not None else None,
                span_type=SpanType.RETRIEVER
            )

        docs = (
            await self._knowledge_center.get_documents(query_text=message.task, domains=self._name)
            if self._knowledge_center else []
        )
        
        if retriever_span is not None:
            agent_ops.end_span(
                run_id=self.get_id(),
                span_id=retriever_span.span_id if retriever_span is not None else None,
                outputs=[Document(page_content=doc.content) for doc in docs]
            )
        
        # Prepare prompt variables
        prompt_vars = {
            **message.model_dump(), 
            "docs": docs, 
            "name": self._name
        }

        system_message = SystemMessage(
            content=self.get_prompt(PLANNER_REQUEST, prompt_vars)
        )

        llm_result: CreateResult = await self.generate(ctx, system_message, session_id=self.get_id())
        return await self._handle_and_publish(llm_result, ctx)

    async def _handle_and_publish(
        self, llm_result: CreateResult, ctx: MessageContext
    ) -> Optional[Union[TaskDispatcherRequestMessage, ReporterRequestMessage, PlannerReplyMessage]]:
        """Handle the LLM result and publish messages as needed."""
        done, reply_message = await self._handle_llm_result(llm_result)
        agent_ops = self.get_agent_ops()
        
        if done:     
            if self._sent_from == HUMAN_SENDER:
                msg = ReporterRequestMessage(
                        task=self._task,
                        chat_history=self._chat_history,
                        sent_from=self._sent_from,
                    )
                
                await self.publish_message(
                    msg,
                    topic_id=TopicId(REPORTER_TOPIC, source=self.get_id()),
                    cancellation_token=ctx.cancellation_token,
                )

                if self._span is not None:
                    agent_ops.end_span(
                        run_id=self.get_id(),
                        span_id=self._span.span_id,
                        outputs=msg,
                        attributes={"next_agent": REPORTER_TOPIC}
                    )
                
                return None
            elif self._sent_from:
                await self.publish_message(
                    reply_message, topic_id=TopicId(self._sent_from, source=self.get_id()), cancellation_token=ctx.cancellation_token,
                )
                return None
            return reply_message

        await self.publish_message(
            reply_message, topic_id=TopicId(AGENT_NAME + "_" + self._name, source=self.get_id()), cancellation_token=ctx.cancellation_token,
        )
        return None
            
    @staticmethod
    def _remove_thought(llm_result: str) -> str:
        """Remove <think>...</think> blocks from LLM output."""
        # Non-greedy match for <think> blocks, case-insensitive
        return re.sub(r"<think>.*?</think>", "", llm_result, flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _is_terminate(llm_result: str) -> bool:
        """Check if LLM result signals termination."""
        # Lowercase once for efficiency
        return "<terminate " in llm_result.lower()
    
    @staticmethod
    def _extract_terminate_fields(text: str) -> Optional[Dict[str, str]]:
        pattern = r'is_success="([^"]*)",?\s*reason="([^"]*)",?\s*result="([^"]*)"'
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            is_success, reason, result = match.groups()
            return {"is_success": is_success, "reason": reason, "result": result}
        return None
           
    async def _handle_llm_result(
        self, llm_result: 'CreateResult'
    ) -> Tuple[bool, 'BaseModel']:
        """Process the LLM result and return the appropriate message."""
        content = llm_result.content

        if not isinstance(content, str):
            raise ValueError("Content type must be str.")

        if self._is_terminate(content):
            fields = self._extract_terminate_fields(content)
            if not fields:
                raise ValueError("Malformed terminate block in LLM output.")
            is_success = fields["is_success"].strip().lower() == "true"
            reply_content = fields["result"] if is_success else fields["reason"]
            return True, PlannerReplyMessage(
                content=reply_content,
                is_success=is_success
            )

        plan = self._remove_thought(content).strip()
        return False, TaskDispatcherRequestMessage(
            plan=plan,
            sent_from=self._name,
            user_request=self._task,
            parent_id=self._span.span_id if self._span is not None else None,
        )
    