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
        
        # --- Special deterministic shortcut for handoff_agent: auto-build derive_topup_query + resolved calls ---
        if self._name == "handoff_agent":
            # Deterministic execution: directly run derive_topup_query, then create a single resolved call embedding its output.
            import uuid, re
            shipment_json = "[]"
            port = None
            svvd = None
            ship_no = None
            try:
                m_num = re.search(r"shipment(?:\s+number)?\s+(\d+)", message.task, re.IGNORECASE)
                m_port = re.search(r"port\s+([A-Z]{3})", message.task, re.IGNORECASE)
                m_svvd = re.search(r"SVVD\s+([A-Z0-9\- ]+)", message.task, re.IGNORECASE)
                ship_no = m_num.group(1) if m_num else None
                port = m_port.group(1) if m_port else None
                svvd = m_svvd.group(1).strip() if m_svvd else None
            except Exception as e:
                logger.debug("handoff_agent parse failed: %s", e)

            # Attempt to load shipment JSON from memory (best-effort)
            if self._wms and ship_no and port and svvd:
                candidates = []
                norm_svvd_us = svvd.replace(" ", "_")
                norm_svvd_rm = re.sub(r"[^A-Za-z0-9]", "", svvd)
                candidates.append(f"shipment_{ship_no}_{port}_{norm_svvd_us}")
                candidates.append(f"shipment_{ship_no}_{port}_{norm_svvd_rm}")
                # Deterministic new key format
                candidates.append(f"shipment_json_{port.upper()}_{norm_svvd_rm.upper()}_{ship_no}")
                for ck in candidates:
                    try:
                        resp = await self._wms.get(session_id=self.get_id(), resource_id=ck)
                        if resp and resp.get("content"):
                            shipment_json = resp.get("content")
                            break
                    except Exception as e:
                        logger.debug("handoff_agent memory lookup failed for %s: %s", ck, e)
                if shipment_json == "[]":
                    logger.info("[handoff_agent] No shipment JSON found. Tried keys=%s", candidates)

            # Run derive_topup_query tool function directly
            derive_tool = next((t for t in self._tools if t.name == "derive_topup_query"), None)
            derive_result: Dict[str, Any] = {"error": "derive_topup_query tool not found"}
            if derive_tool is not None:
                try:
                    # FunctionTool exposes an async _func attribute usually
                    func = getattr(derive_tool, "_func", None) or getattr(derive_tool, "func", None)
                    if func:
                        derive_result = await func(shipment_json=shipment_json, min_empty_ratio=85.0, default_port=port, default_svvd=svvd)
                except Exception as e:
                    derive_result = {"error": f"Failed to run derive_topup_query: {e}"}

            need_topup = bool(derive_result.get("need_topup"))
            triggered = derive_result.get("triggered_count", 0)
            suggested_key = derive_result.get("suggested_memory_key")
            summary_msg = (
                f"Top-up recommended: {need_topup}. Triggered records: {triggered}." if suggested_key
                else f"Top-up not recommended. Reason: {derive_result.get('reason','unknown')}"
            )

            # Optionally store decision in memory (separate from future top-up list key) 
            decision_key = None
            if self._wms and port and svvd:
                try:
                    norm_svvd_decision = re.sub(r"[^A-Za-z0-9]", "", svvd.upper()) if svvd else ""
                    decision_key = f"handoff_decision_{port.upper()}_{norm_svvd_decision}"
                    await self._wms.set(
                        content=json.dumps(derive_result, ensure_ascii=False),
                        session_id=self.get_id(),
                        resource_id=decision_key,
                        data_description={
                            "source": "handoff_agent",
                            "ship_no": ship_no,
                            "port": port,
                            "svvd": svvd,
                            "derived": True,
                            "suggested_topup_key": suggested_key
                        }
                    )
                    logger.info("[handoff_agent] Stored decision to memory key %s", decision_key)
                except Exception as e:
                    logger.warning("[handoff_agent] Failed to store decision: %s", e)

            resolved_args = {
                "result": json.dumps(derive_result, ensure_ascii=False),
                "user_message": summary_msg,
                "result_memory_key": decision_key
            }
            llm_result = type("_FakeResult", (), {"content": [FunctionCall(id=str(uuid.uuid4()), name=NAME_RESOLVED_TOOL, arguments=json.dumps(resolved_args, ensure_ascii=False))]})()
        else:
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
            
        # Emit a completion log message so upstream runtime can mark flow complete when this is a terminal worker
        if not message.sent_from:
            # Only mark is_complete when this truly represents a terminal success.
            # For high empty_ratio (is_success=False) we keep the queue open so Planner can inject top-up workflow.
            try:
                await self._report_message(
                    LogMessage(
                        agent_name=self._name,
                        action="Subtask completed",
                        content=reply_result if isinstance(reply_result, str) else json.dumps(reply_result, ensure_ascii=False),
                        id=self.get_id(),
                        is_complete=bool(is_success),
                    )
                )
            except Exception as e:
                logger.warning("Failed to emit completion log message for %s: %s", self._name, e)

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

            # --- Auto-inject session_id / memory_key heuristics ---
            try:
                if hasattr(tool, "_args_type"):
                    model_fields = getattr(tool._args_type, "model_fields", {})
                    if "session_id" in model_fields and "session_id" not in arguments:
                        arguments["session_id"] = getattr(ctx.topic_id, "source", None) or self.get_id()
                    if call.name == "get_eligible_topup_shipment" and "memory_key" in model_fields and "memory_key" not in arguments:
                        port_val = arguments.get("port")
                        svvd_val = arguments.get("svvd")
                        if isinstance(port_val, str) and isinstance(svvd_val, str):
                            import re
                            norm_svvd = re.sub(r'[^A-Za-z0-9]', '', svvd_val.upper())
                            arguments["memory_key"] = f"eligible_topup_{port_val.upper()}_{norm_svvd}"
            except Exception as _auto_inject_err:
                logger.debug("Auto injection skipped: %s", _auto_inject_err)

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
                        try:
                            call_args = json.loads(call.arguments)
                        except json.JSONDecodeError:
                            call_args = {}
                        user_msg = call_args.get("user_message", "No user message provided.")
                        memory_location = call_args.get("result_memory_key") or call_args.get("memory_key")
                        session_arg = call_args.get("session_id")
                        res = []
                        inline_result_raw = call_args.get("result") or result.content
                        # Attempt proactive store if memory key provided but not yet present
                        if self._wms is not None and memory_location is not None:
                            lookup_session = session_arg or self.get_id()
                            need_store = False
                            # First try to see if it exists
                            try:
                                existing = await self._wms.get(session_id=lookup_session, resource_id=memory_location)
                                if isinstance(existing, dict) and existing.get("error") == "not_found":
                                    need_store = True
                            except Exception:
                                need_store = True
                            if need_store:
                                # Parse inline_result_raw (may be JSON list/string) and store
                                try:
                                    import json as _json
                                    parsed_inline_for_store = inline_result_raw
                                    if isinstance(parsed_inline_for_store, str):
                                        try:
                                            parsed_inline_for_store = _json.loads(parsed_inline_for_store)
                                        except Exception:
                                            # keep raw string
                                            pass
                                    await self._wms.set(
                                        content=_json.dumps(parsed_inline_for_store, ensure_ascii=False) if not isinstance(parsed_inline_for_store, str) else parsed_inline_for_store,
                                        session_id=lookup_session,
                                        resource_id=memory_location,
                                        data_description={
                                            "source": "executor_auto_store",
                                            "agent": self._name,
                                            "original_user_msg": user_msg[:120]
                                        }
                                    )
                                    logger.info("[Executor] Auto-stored result to memory key %s (session=%s)", memory_location, lookup_session)
                                except Exception as e:
                                    logger.error("[Executor] Failed auto-store to memory key %s: %s", memory_location, e)
                        if self._wms is not None and memory_location is not None:
                            # Try provided session_id first, then fallback to executor run id
                            lookup_session = session_arg or self.get_id()
                            try:
                                wms_result = await self._wms.get(
                                    session_id=lookup_session,
                                    resource_id=memory_location
                                )
                                if isinstance(wms_result, dict) and "content" in wms_result:
                                    try:
                                        parsed_content = json.loads(wms_result["content"]) if isinstance(wms_result["content"], str) else wms_result["content"]
                                    except Exception:
                                        parsed_content = wms_result["content"]
                                    res = [{
                                        "content": parsed_content,
                                        "headers": wms_result.get("headers"),
                                        "meta_data": {"data_description": wms_result.get("meta_data"), "data_type": wms_result.get("content_type")}
                                    }]
                                else:
                                    logger.error("Unexpected WMS response structure for key %s: %s", memory_location, wms_result)
                            except Exception as e:
                                logger.error("Failed to get memory from WMS (session=%s key=%s): %s", lookup_session, memory_location, str(e))
                                res = []  # Ensure res is always a list
                        # Fallback: if no references from memory, attempt to parse inline result JSON
                        if not res:
                            inline_result = inline_result_raw
                            try:
                                parsed_inline = json.loads(inline_result) if isinstance(inline_result, str) else inline_result
                                # Heuristic: only attach if it looks like structured data (dict with eligible_shipments / list)
                                if isinstance(parsed_inline, (dict, list)):
                                    res = [{
                                        "content": parsed_inline,
                                        "headers": None,
                                        "meta_data": {"data_description": {"source": "inline_fallback"}, "data_type": type(parsed_inline).__name__}
                                    }]
                                    logger.warning("[Executor] Memory lookup failed; using inline result fallback for reporting (key=%s)", memory_location)
                            except Exception:
                                pass
                        # If we have data but user_msg is failure-like, adjust
                        if res and any(tok in user_msg.lower() for tok in ["unable", "cannot", "fail"]):
                            try:
                                count_hint = 0
                                first_block = res[0]["content"]
                                if isinstance(first_block, list):
                                    count_hint = len(first_block)
                                elif isinstance(first_block, dict) and "eligible_shipments" in first_block:
                                    count_hint = len(first_block.get("eligible_shipments", []))
                                if count_hint:
                                    user_msg = f"Retrieved {count_hint} record(s) successfully."  # overwrite negative message
                            except Exception:
                                pass
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
                # Treat shipment-details with high empty_ratio as non-terminal (is_success=False)
                # so that Planner/TaskDispatcher can continue into top-up workflow.
                if result.name == NAME_RESOLVED_TOOL:
                    try:
                        parsed = json.loads(result.content) if isinstance(result.content, str) else result.content
                        # Heuristic: list of shipment dicts with empty_ratio field
                        if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
                            # --- Single-shipment filtering enhancement ---
                            # If original user_message or task references exactly one shipment number, filter out others.
                            import re
                            single_ship_match = None
                            if hasattr(ctx, 'topic_id'):
                                pass  # placeholder if future context needed
                            # Prefer message.task if available in chat history earlier (we have message in outer scope as 'message' variable)
                            # We don't have direct access to ExecutorRequestMessage here, so infer from last system message stored.
                            # Simpler: scan chat history content for 'shipment number' pattern.
                            ship_nums = set()
                            for h in self._chat_history:
                                try:
                                    m = re.findall(r"shipment(?:\s+number)?\s+(\d{6,})", str(h.content), re.IGNORECASE)
                                    ship_nums.update(m)
                                except Exception:
                                    continue
                            if len(ship_nums) == 1:
                                target_ship = list(ship_nums)[0]
                                before_len = len(parsed)
                                filtered = [row for row in parsed if str(row.get("SHIPMENT_NUMBER")) == target_ship]
                                if filtered and len(filtered) != before_len:
                                    logger.info("[Executor] Filtered shipment list from %d to 1 for single-shipment query %s", before_len, target_ship)
                                    parsed = filtered
                                    # Overwrite result.content with filtered JSON
                                    result.content = json.dumps(parsed, ensure_ascii=False)
                            # Persist deterministic shipment JSON key for downstream handoff_agent
                            try:
                                if self._wms and parsed:
                                    first = parsed[0]
                                    ship_no = str(first.get("SHIPMENT_NUMBER"))
                                    port = first.get("VOY_STOP_PORT_CDE")
                                    svvd = first.get("SVVD")
                                    if ship_no and port and svvd:
                                        norm_svvd = re.sub(r"[^A-Za-z0-9]", "", str(svvd).upper())
                                        det_key = f"shipment_json_{str(port).upper()}_{norm_svvd}_{ship_no}"
                                        await self._wms.set(
                                            content=json.dumps(parsed, ensure_ascii=False),
                                            session_id=self.get_id(),
                                            resource_id=det_key,
                                            data_description={
                                                "source": "executor_single_shipment_store",
                                                "port": port,
                                                "svvd": svvd,
                                                "ship_no": ship_no,
                                                "record_count": len(parsed)
                                            }
                                        )
                                        logger.info("[Executor] Stored shipment JSON under deterministic key %s", det_key)
                            except Exception as store_err:
                                logger.warning("[Executor] Failed storing deterministic shipment JSON key: %s", store_err)
                            if any(float(item.get("empty_ratio", 0) or 0) >= 85.0 for item in parsed):
                                logger.info("[Executor] Detected shipment details with empty_ratio>=85; marking is_success=False to allow top-up continuation.")
                                return True, result.content, False
                    except Exception as _err:
                        logger.debug("[Executor] empty_ratio check skipped due to error: %s", _err)
                return True, result.content, result.name == NAME_RESOLVED_TOOL

        return False, None, True