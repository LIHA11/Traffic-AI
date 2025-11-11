# Standard library imports
import asyncio
import logging
import socket
import threading
import time
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    Any,
    Annotated,
)
import logging
import uuid

# Third-party imports (autogen_core)
from autogen_core import (
    CancellationToken,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core.models import LLMMessage
from autogen_core.tools import Tool, FunctionTool
from termcolor import RESET

# Local application imports
from src.conversations.enum.message_metadata_key_enum import MessageMetadataKeyEnum
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.mlflow_ops import MLflowOps
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.vo.message import Message
from src.copilot.agent.dag_generator import (
    DAG_GENERATOR_REQUEST,
    DAG_GENERATOR_TOPIC,
    DAGGenerator,
    DAGGeneratorOutput,
)
from src.copilot.agent.executor import Executor
from src.copilot.agent.intention import (
    AGENT_NAME as INTENTION_AGENT_NAME,
    INTENTION_REQUEST,
    Intention,
    IntentionRequestMessage,
)
from src.copilot.agent.preference.constant import (
    MESSAGE_GROUPER_REQUEST,
    MESSAGE_GROUPER_TOPIC,
    KEYWORDS_EXTRACTION_TOPIC,
    KEYWORDS_EXTRACTION_REQUEST,
    SUMMARY_EXTRACTION_TOPIC,
    SUMMARY_EXTRACTION_REQUEST,
    PREFERENCE_EXTRACTOR_MASTER_TOPIC
)
from src.copilot.agent.preference.keywords_extraction import KeyWordsExtraction
from src.copilot.agent.preference.message_grouper import MessageGrouper
from src.copilot.agent.preference.summary_extraction import SummaryExtraction
from src.copilot.agent.preference.preference_extraction_master import PreferenceExtractionMaster

from src.copilot.agent.planner import HUMAN_SENDER, Planner
from src.copilot.agent.reporter import REPORTER_TOPIC, Reporter, ReporterState
from src.copilot.agent.task_dispatcher import AGENT_NAME, TaskDispatcher
from src.copilot.chat_client.chat_client_creator import ChatClientCreator
from src.copilot.utils.handoff import Handoff
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.copilot.utils.message import (
    WorkingMemoryService,
    convert_to_llm_message,
)
from src.copilot.agent.preference.constant import PREFERENCE_EXTRACTOR_MASTER_TOPIC, PreferenceExtractionMasterRequest
from mlflow.entities import RunStatus

logger = logging.getLogger(__name__)

class WorkflowTask(Enum):
    PREFERENCE_EXTRACTION = "preference_extraction"
        
logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    EXECUTOR = "executor"
    PLANNER = "planner"

class AgentConfig(TypedDict):
    name: str
    description: str
    type: AgentType
    prompt_templates: Dict[str, str]
    tools: List[str]
    model: Optional[str]
    family: Optional[str]
    notes: Optional[str]
class CopilotConfig(TypedDict):
    entry_agent_name: str
    agents: List[AgentConfig]
    research_agent_name: Optional[str]
    
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tools(self, names: List[str]) -> List[Tool]:
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise ValueError(f"Requested tool(s) not found: {', '.join(missing)}")
        return [self._tools[n] for n in names]
    
'''
async def queue_streamer(user_id: str):
    queue = id_queues[user_id]
    try:
        while True:
            msg = await queue.get()
            yield msg.model_dump_json() + "\n"
            if msg.is_complete:
                break
    finally:
        # Cleanup: after the stream is complete, delete the user's queue for GC
        id_queues.pop(user_id, None)

'''

class CopilotAgentRuntime(SingleThreadedAgentRuntime):
    def __init__(
        self,
        chat_client_creator: ChatClientCreator,
        copilot_config: CopilotConfig,
        working_memory_service: WorkingMemoryService,
        tool_registry: Optional[ToolRegistry] = None,
        knowledge_center: Optional[KnowledgeCenter] = None,
        agent_ops: Optional[MLflowOps] = None,  # Placeholder for Langfuse or similar
        use_mock_sql: bool = False,  # Enable mock SQL tool for testing
    ):
        super().__init__()
        self._chat_client_creator = chat_client_creator
        self._tool_registry = tool_registry or ToolRegistry()
        self._copilot_config = copilot_config
        self._working_memory_service = working_memory_service
        self._knowledge_center = knowledge_center
        self._agent_ops = agent_ops
        self._research_agent = copilot_config.get("research_agent_name")
        self._use_mock_sql = use_mock_sql
        
        self._report_message_queue: Dict[str, asyncio.Queue] = {}
        self.lock = threading.Lock()  # Lock for thread-safe operations

        self._register_all_tools()

        if self._research_agent and not self._tool_registry.get_tool(self._research_agent):
            raise ValueError(
                f"Research agent '{self._research_agent}' must be registered as a tool."
            )
            
    async def send_to_user(self, msg: LogMessage):
        with self.lock:
            await self._report_message_queue[msg.id].put(msg)
            
    async def create_queue(self, request_id: str) -> asyncio.Queue:
        """Create a message queue for a user to stream messages."""
        with self.lock:
            if request_id not in self._report_message_queue:
                self._report_message_queue[request_id] = asyncio.Queue()
            return self._report_message_queue[request_id]
            
    async def release_queue(self, request_id: str):
        """Release the queue for a user after streaming is complete."""
        with self.lock:
            if request_id in self._report_message_queue:
                self._report_message_queue.pop(request_id, None)
            
    def _register_all_tools(self):
        for tool in self._working_memory_service.get_tools():
            self._tool_registry.register_tool(tool)
        if self._knowledge_center:
            for tool in self._knowledge_center.get_tools():
                self._tool_registry.register_tool(tool)
        for agent in self._copilot_config["agents"]:
            self._tool_registry.register_tool(Handoff(agent["name"], agent["description"]))

        # --- Built-in fallback tools for local/dev mode ---
        # Some agent configs (e.g. agent_config_v2.yaml) reference tools that may normally
        # come from an external MCP tool service. In local/offline mode these might be
        # missing, so we register lightweight implementations if absent to prevent
        # startup failures. These can be replaced by richer implementations later.

        if not self._tool_registry.get_tool("run_python_code"):
            async def run_python_code(
                code: Annotated[str, "Python code snippet to execute (no network/db calls)."],
            ) -> str:
                """Executes a small Python code snippet in a restricted namespace.
                Returns JSON (string) containing stdout and any top-level variables (basic types).
                WARNING: This is a dev/local helper; do not use in production without hardening.
                """
                import io, sys, json, contextlib, ast
                # Basic static check to avoid obviously dangerous statements
                banned_nodes = (ast.ImportFrom,)
                try:
                    tree = ast.parse(code, mode='exec')
                    for n in ast.walk(tree):
                        if isinstance(n, banned_nodes):
                            return json.dumps({"error": "Import statements not allowed in local run_python_code."})
                except Exception as e:
                    return json.dumps({"error": f"Syntax error: {e}"})

                stdout_buffer = io.StringIO()
                # Provide a very small safe namespace
                safe_builtins = {"print": print, "len": len, "range": range, "sum": sum}
                env = {"__builtins__": safe_builtins}
                # Optional: provide pandas if present for DataFrame operations
                try:
                    import pandas as pd  # noqa: F401
                    env["pd"] = pd
                except Exception:
                    pass

                try:
                    with contextlib.redirect_stdout(stdout_buffer):
                        exec(code, env, env)
                except Exception as e:
                    return json.dumps({"error": f"Execution error: {e}"})

                # Collect simple (JSON-serializable) variable results
                simple_vars = {}
                for k, v in env.items():
                    if k.startswith("__"):
                        continue
                    if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                        simple_vars[k] = v
                return json.dumps({
                    "stdout": stdout_buffer.getvalue(),
                    "variables": simple_vars
                }, ensure_ascii=False)

            self._tool_registry.register_tool(FunctionTool(run_python_code, name="run_python_code", description="Execute limited Python code locally (development only). Returns stdout and simple variables as JSON."))

        if not self._tool_registry.get_tool("sql_execution"):
            # Use mock SQL tool if enabled
            if self._use_mock_sql:
                from src.copilot.utils.mock_sql_tool import create_mock_sql_tool
                self._tool_registry.register_tool(create_mock_sql_tool())
                logger.info("[CopilotAgentRuntime] Registered MOCK sql_execution tool")
            else:
                async def sql_execution(
                    sql: Annotated[str, "SQL query string to execute (SELECT / WITH only, no DML/DDL)."],
                    db_url: Annotated[str, "Database URL, e.g., sqlite:///path.db or mysql+pymysql://user:pass@host/db"],
                    limit: Annotated[int, "Max rows to return (after filtering)." ] = 100,
                    params: Annotated[Optional[Dict[str, Any]], "Bind parameters dict for :name placeholders in SQL."] = None,
                    explain: Annotated[bool, "Return execution plan if dialect supports (EXPLAIN)." ] = False,
                ) -> Dict[str, Any]:
                    """执行只读 SQL 查询并返回结构化结果 (开发/本地使用)。\n\n安全约束: 仅允许以 SELECT 或 WITH 开头的单条语句, 禁止出现 update/delete/insert/alter/drop/create/truncate 等写操作关键字。\n\n返回示例: {\n  'columns': [...],\n  'rows': [[...],[...]],\n  'row_count': 10,\n  'truncated': False,\n  'dialect': 'sqlite',\n  'sql': 'SELECT ...',\n  'explain': { 'columns': [...], 'rows': [[...]] } | None,\n  'error': None | '错误信息'\n}"""
                    import re
                    from sqlalchemy import create_engine, text
                    from sqlalchemy.exc import SQLAlchemyError

                    cleaned = sql.strip()
                    lowered = cleaned.lower()
                    if not (lowered.startswith("select") or lowered.startswith("with")):
                        return {"error": "Only SELECT / WITH queries allowed", "sql": sql}
                    if ";" in cleaned[:-1]:  # allow trailing semicolon only
                        return {"error": "Multiple statements not allowed", "sql": sql}
                    forbidden = [r"\bupdate\b", r"\bdelete\b", r"\binsert\b", r"\balter\b", r"\bdrop\b", r"\bcreate\b", r"\btruncate\b"]
                    for pat in forbidden:
                        if re.search(pat, lowered):
                            return {"error": f"Forbidden keyword detected: {pat}", "sql": sql}

                    try:
                        engine = create_engine(db_url)
                    except Exception as e:
                        return {"error": f"DB connection error: {e}", "sql": sql}

                    payload: Dict[str, Any] = {
                        "columns": [],
                        "rows": [],
                        "row_count": 0,
                        "truncated": False,
                        "dialect": None,
                        "sql": sql,
                        "explain": None,
                        "error": None,
                    }
                    try:
                        with engine.connect() as conn:
                            stmt = text(cleaned)
                            result = conn.execute(stmt, params or {})
                            rows = result.fetchall()
                            cols = result.keys()
                            payload["dialect"] = conn.dialect.name
                            if len(rows) > limit:
                                payload["truncated"] = True
                                rows = rows[:limit]
                            payload["columns"] = list(cols)
                            payload["rows"] = [list(r) for r in rows]
                            payload["row_count"] = len(rows)

                            if explain:
                                try:
                                    dialect = conn.dialect.name
                                    explain_sql = None
                                    if dialect.startswith("sqlite"):
                                        explain_sql = f"EXPLAIN QUERY PLAN {cleaned.rstrip(';')}"
                                    elif dialect in ("postgresql", "mysql"):
                                        explain_sql = f"EXPLAIN {cleaned.rstrip(';')}"
                                    if explain_sql:
                                        er = conn.execute(text(explain_sql), params or {})
                                        payload["explain"] = {
                                            "columns": list(er.keys()),
                                            "rows": [list(rr) for rr in er.fetchall()],
                                        }
                                except Exception as e2:
                                    payload["explain"] = {"error": f"Explain failed: {e2}"}
                    except SQLAlchemyError as sqle:
                        payload["error"] = f"SQLAlchemy error: {sqle}"
                    except Exception as e:
                        payload["error"] = f"Unexpected error: {e}"
                    finally:
                        try:
                            engine.dispose()
                        except Exception:
                            pass
                    return payload

                self._tool_registry.register_tool(
                    FunctionTool(
                        sql_execution,
                        name="sql_execution",
                        description="只读 SQL 查询工具: 执行 SELECT / WITH 语句并返回列+行 JSON, 可选 explain。开发环境使用。"
                    )
                )

        
        if not self._tool_registry.get_tool("port_name_mapper"):
            async def port_name_mapper(
                port_name: Annotated[str, "Full port name or UNLOCODE / internal port code to normalize."],
                fuzzy: Annotated[bool, "Use fuzzy matching against knowledge graph"] = True,
                limit: Annotated[int, "Max candidate codes"] = 5,
            ) -> str:
                """Map a full port name / alias to candidate 3-letter port codes using the preference graph.
                If the input is already a 3-letter code, returns it directly. Results are heuristic.
                """
                import json
                pn = (port_name or '').strip().upper()
                if len(pn) == 3 and pn.isalpha():
                    return json.dumps({"input": port_name, "candidates": [pn], "method": "direct"})
                try:
                    from src.copilot.agent.preference.keywords_extraction import graph  # graph built at import
                    # Perform a fuzzy search; graph.search already uppercases internally
                    results = await graph.search(pn, fuzzy=fuzzy, limit=limit, max_hops=2)
                    # Extract 3-letter nodes as candidate codes
                    codes = [n for n in results.get("nodes", []) if len(n) == 3 and n.isalpha()]
                    return json.dumps({
                        "input": port_name,
                        "candidates": codes[:limit],
                        "similarity": results.get("similarity", {}),
                        "method": "graph_fuzzy" if fuzzy else "graph_exact"
                    })
                except Exception as e:
                    return json.dumps({"input": port_name, "candidates": [], "error": str(e)})

            self._tool_registry.register_tool(FunctionTool(port_name_mapper, name="port_name_mapper", description="Map full port name/alias to candidate 3-letter port codes using local knowledge graph (development)."))
        
        if not self._tool_registry.get_tool("get_eligible_topup_shipment"):
            async def get_eligible_topup_shipment(
                shipments: Annotated[list, "List of shipment dict. Each dict should include keys like 'svvd', 'port', 'remaining_space', 'empty_ratio', etc."],
                port: Annotated[str, "Target port for top up."],
                svvd: Annotated[str, "Target SVVD/service loop for top up."],
                min_empty_ratio: Annotated[float, "Minimum empty ratio threshold for eligible top up. Default 85."] = 85,
            ) -> list:
                """
                Returns a list of shipment dict that are eligible for top up at the given port and SVVD,
                i.e. empty_ratio >= min_empty_ratio and matching port/SVVD.
                """
                # 防御性编程，筛选数据
                eligible = []
                for ship in shipments:
                    try:
                        if (
                            ship.get("port") == port and
                            ship.get("svvd") != svvd and  # 不包括目标本身
                            float(ship.get("empty_ratio", 0)) >= min_empty_ratio
                        ):
                            eligible.append(ship)
                    except Exception:
                        continue
                return eligible

            self._tool_registry.register_tool(
                FunctionTool(get_eligible_topup_shipment, name="get_eligible_topup_shipment",
                            description="筛选可用于补舱的运单，按港口/服务号过滤，空舱率超过指定阈值。输入list[dict]，返回筛选后的list[dict]")
            )

    async def start(self):
        """Initialize all agents concurrently and start."""
        await asyncio.gather(
            *(self._init_agent(cfg) for cfg in self._copilot_config["agents"]),
            self._init_default_agents()
        )
        await self._init_preference_extraction_agents()
        super().start()

    async def _init_agent(self, cfg: AgentConfig):
        """Initialize a single agent based on its type."""
        if cfg["type"] == AgentType.PLANNER:
            await self._init_planner(cfg)
        elif cfg["type"] == AgentType.EXECUTOR:
            await self._init_executor(cfg)
        else:
            raise ValueError(f"Unknown agent type: {cfg['type']}")
        
    async def _register_and_subscribe(self, topic: str, agent_type_obj):
        sub = TypeSubscription(topic_type=topic, agent_type=agent_type_obj.type)
        await self.add_subscription(sub)    
        
    async def _init_executor(self, cfg: AgentConfig):
        exe_type = await Executor.register_agent(
            runtime=self,
            name=cfg["name"],
            description=cfg["description"],
            prompt_templates=cfg["prompt_templates"],
            chat_client=self._chat_client_creator.create(
                model=cfg["model"],
                family=cfg["family"],
            ),
            knowledge_center=self._knowledge_center,
            tools=self._tool_registry.get_tools(cfg["tools"]),
            notes=cfg.get("notes"),
            agent_ops=self._agent_ops,
            max_call_count=cfg.get("call_count", 20),
            report_message=self.send_to_user,
            report_intermediate_steps=cfg.get("report_intermediate_steps", False),
            wms=self._working_memory_service
        )
        await self._register_and_subscribe(cfg["name"], exe_type)
        
    async def _init_preference_extraction_agents(self):
        exe_type = await KeyWordsExtraction.register_agent(
            runtime=self,
            prompt_templates={
                KEYWORDS_EXTRACTION_REQUEST: "preference/keywords_extraction_request.j2"
            },
            chat_client=self._chat_client_creator.create(),
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(KEYWORDS_EXTRACTION_TOPIC, exe_type)
        exe_type = await SummaryExtraction.register_agent(
            runtime=self,
            prompt_templates={
                SUMMARY_EXTRACTION_REQUEST: "preference/summary_extraction_request.j2"
            },
            chat_client=self._chat_client_creator.create(),
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(SUMMARY_EXTRACTION_TOPIC, exe_type)
        exe_type = await MessageGrouper.register_agent(
            runtime=self,
            prompt_templates={
                MESSAGE_GROUPER_REQUEST: "preference/message_grouper_request.j2"
            },
            chat_client=self._chat_client_creator.create(),
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(MESSAGE_GROUPER_TOPIC, exe_type)
        exe_type = await PreferenceExtractionMaster.register_agent(
            runtime=self,
            kc=self._knowledge_center,
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(PREFERENCE_EXTRACTOR_MASTER_TOPIC, exe_type)

    async def _init_planner(self, cfg: AgentConfig):
        planner_type = await Planner.register_agent(
            runtime=self,
            name=cfg["name"],
            description=cfg["description"],
            prompt_templates=cfg["prompt_templates"],
            chat_client=self._chat_client_creator.create(
                model=cfg["model"],
                family=cfg["family"],
                fc=False,
            ),
            knowledge_center=self._knowledge_center,
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(cfg["name"], planner_type)
        await self._init_task_dispatcher(cfg["name"], cfg["tools"])
        
    async def _init_task_dispatcher(self, planner_name: str, tools: List[str]):
        task_dispatcher_type = await TaskDispatcher.register_agent(
            name=AGENT_NAME + "_" + planner_name,
            runtime=self,
            prompt_templates={
                "RECEIVE_DAG": "task_dispatcher_receive_dag_v2.j2"
            },
            chat_client=self._chat_client_creator.create(),
            knowledge_center=self._knowledge_center,
            tools=self._tool_registry.get_tools(tools),
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(AGENT_NAME + "_" + planner_name, task_dispatcher_type)
        
    async def _init_default_agents(self):
        """Initialize system default agents concurrently."""
        await asyncio.gather(
            self._init_intention(),
            self._init_dag_generator(),
            self._init_reporter(),
        )
        
    async def _init_intention(self):
        intent_type = await Intention.register_agent(
            runtime=self,
            prompt_templates={
                INTENTION_REQUEST: "intention_request.j2"
            },
            chat_client=self._chat_client_creator.create(),
            research_agent=self._research_agent,
            agent_ops=self._agent_ops,
            knowledge_center=self._knowledge_center,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(INTENTION_AGENT_NAME, intent_type)
        
    async def _init_dag_generator(self):
        dg_type = await DAGGenerator.register_agent(
            runtime=self,
            prompt_templates={
                DAG_GENERATOR_REQUEST: "dag_generator_request.j2"
            },
            chat_client=self._chat_client_creator.create_mini(response_format=DAGGeneratorOutput),
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(DAG_GENERATOR_TOPIC, dg_type)
        
    async def _init_reporter(self):
        get_from_memory_tool = self._tool_registry.get_tool("get_from_memory")
        if not get_from_memory_tool:
            raise ValueError("Tool 'get_from_memory' is not registered.")
        
        rep_type = await Reporter.register_agent(
            runtime=self,
            prompt_templates={
                ReporterState.GET_ANSWER: "reporter_get_answer.j2",
                ReporterState.GET_REFERENCE: "reporter_get_reference.j2",
                ReporterState.REMOVE_INTERNAL_STUFF: "reporter_remove_internal_stuff.j2",
            },
            chat_client=self._chat_client_creator.create(),
            tools=[get_from_memory_tool],
            agent_ops=self._agent_ops,
            report_message=self.send_to_user,
        )
        await self._register_and_subscribe(REPORTER_TOPIC, rep_type)
        
    async def _create(
        self,
        id: str,
        messages: Sequence[LLMMessage],
        cancellation_token: Optional[CancellationToken] = None,
    ):
        request = IntentionRequestMessage(
            chat_history=messages,
            sent_from=HUMAN_SENDER,
            forward_to=self._copilot_config["entry_agent_name"],
        )
        await self.publish_message(
            message=request,
            topic_id=TopicId(type=INTENTION_AGENT_NAME, source=id),
            cancellation_token=cancellation_token,
        )

    async def create(
        self,
        messages: Sequence[Message],
        user_id: str,
        cancellation_token: Optional[CancellationToken] = None,
        conversation_id: Optional[str] = None
    )-> Union[asyncio.Queue, str]:
        if conversation_id is not None:
            conversation_id = str(conversation_id) if not isinstance(conversation_id, str) else conversation_id

        latest_message: Message = messages[-1]

        run = self._agent_ops.create_run(
            description=f"{latest_message.content} (with messages history)" if len(messages) > 1 else latest_message.content,
            user_id=user_id,
            tags={
                "hostname": socket.gethostname(),
                "conversation_id": conversation_id,
                "message_id": latest_message.id,
            },
        )

        run_id: str = str(uuid.uuid4())

        if run is not None:
            self._agent_ops.create_trace(run_id=run.info.run_id,
                                        name="Traffic Copilot Agent",
                                        inputs=[msg.to_dict() for msg in messages])
            run_id = run.info.run_id
        else:
            logger.warning(f"Failed to create MLflow run. Proceeding the request of {user_id} without tracing.")
            
        queue = await self.create_queue(run_id)
            
        msgs, all_metadatas = [], []
        for msg in messages:
            llm_message, metadatas = convert_to_llm_message(msg)
            msgs.append(llm_message)
            all_metadatas.extend(metadatas)

        await asyncio.gather(*[
            self._working_memory_service.set(
                content=metadata.content,
                resource_id=metadata.id,
                session_id=run_id,
                data_description=metadata.data_description,
            ) for metadata in all_metadatas
        ])

        await self._create(
            id=run_id,
            messages=msgs,
            cancellation_token=cancellation_token,
        )
        return queue, run_id
    
    async def create_until_finish(
        self,
        messages: Sequence[Message],
        cancellation_token: Optional[CancellationToken] = None,
        user_id: str = socket.gethostname(),
        print_log: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Tuple[Message, float]:
        start_time = time.time()
        
        def format_timestamp(ts):
            from datetime import datetime
            # Formats datetime to a readable string
            if isinstance(ts, datetime):
                return ts.strftime("%Y-%m-%d %H:%M:%S")
            return str(ts)

        def print_log_message(msg, step_num=None, indent=0):
            import pandas as pd
            
            ts = format_timestamp(msg.timestamp)
            agent = getattr(msg, 'agent_name', 'Copilot')
            action = getattr(msg, 'action', '')
            content = getattr(msg, 'content', '')
            references = getattr(msg, 'references', [])
            prefix = f"Step {step_num}: " if step_num else ""
            indentation = "    " * indent
            
            if references:
                if "tsPort" in references[0]["content"][0]:
                    _metadata = {MessageMetadataKeyEnum.SHIPMENT: references[0]["content"]} # For top-up schema
                    _others = {MessageMetadataKeyEnum.SHIPMENT: references[0]["meta_data"]}
                else:
                    _metadata = {MessageMetadataKeyEnum.TABLE: {"headers": references[0]['headers'], "data": references[0]["content"]}} # For shipment details schema
                    _others = {MessageMetadataKeyEnum.TABLE: references[0]["meta_data"]}
            else:
                _metadata = None
                _others = None
            
            print(f"{indentation}[{ts}] {agent}: {prefix}{action}\n{indentation}{content}")
            
            if _metadata:
                if 'SHIPMENT' in _metadata:
                    shipment = _metadata['SHIPMENT']
                elif 'TABLE' in _metadata and 'data' in _metadata['TABLE']:
                    shipment = _metadata['TABLE']['data']
                if shipment:
                    print(pd.DataFrame(shipment))

            if _others is not None:
                print(f"Others:{RESET} {_others}")

        queue, id = await self.create(
            messages=messages,
            user_id=user_id,
            conversation_id=conversation_id,
            cancellation_token=cancellation_token,
        )

        is_error: bool = True
        is_aborted: bool = False
        try:
            while True:
                msg: LogMessage = await queue.get()
                if msg.is_complete:
                    is_error = False
                    break
                if print_log:
                    print_log_message(msg)  # Use the user-friendly print function
        except Exception as e:
            is_error = True
            if isinstance(e, asyncio.CancelledError):
                is_aborted = True
        finally:
            self.release_queue(id)
            logger.info(f"Released queue for {id}")
            self._agent_ops.end_run(id = (
                    RunStatus.KILLED
                ))

        if not msg.is_complete:
            raise ValueError("Copilot execution did not complete successfully.")

        references = msg.references

        if references:
            if "tsPort" in references[0]["content"][0]:
                _metadata = {MessageMetadataKeyEnum.SHIPMENT: references[0]["content"]} # For top-up schema
                _others = {MessageMetadataKeyEnum.SHIPMENT: references[0]["meta_data"]}
            else:
                _metadata = {MessageMetadataKeyEnum.TABLE: {"headers": references[0]['headers'], "data": references[0]["content"]}} # For shipment details schema
                _others = {MessageMetadataKeyEnum.TABLE: references[0]["meta_data"]}
        else:
            _metadata = {}
            _others = {}

        message = Message(
            role=RoleEnum.ASSISTANT,
            content=msg.content,
            metadata=_metadata, # headers (for shipment details) and shipment data
            others=_others # column descriptions
        )

        duration = time.time() - start_time
        return message, duration

    async def _create_workflow_task(
        self,
        id: str,
        task: 'WorkflowTask',
        inputs: Optional[Dict] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ):
        
        if task == WorkflowTask.PREFERENCE_EXTRACTION:
            request = PreferenceExtractionMasterRequest(
                user_id=inputs.get("user_id", None),
                message_id=inputs.get("message_id", None),
                first_n_conversations=inputs.get("first_n_conversations", 10),
            )
            await self.publish_message(
                message=request,
                topic_id=TopicId(type=PREFERENCE_EXTRACTOR_MASTER_TOPIC, source=id),
                cancellation_token=cancellation_token,
            )
        else:
            raise ValueError(f"Unknown workflow task: {task}")
        
    async def create_workflow_task_until_finish(
        self,
        task: 'WorkflowTask',
        cancellation_token: Optional[CancellationToken] = None,
        user_id: Optional[str] = socket.gethostname(),
        inputs: Optional[Dict[str, str]] = {}
    ) -> Tuple[Dict, float]:
        """
        Creates a workflow task, initializes trace, and returns the queue and id.
        """

        queue, id = await self.create_workflow_task(
            task=task,
            cancellation_token=cancellation_token,
            user_id=user_id,
            inputs=inputs,
        )

        is_error: bool = True
        is_aborted: bool = False
        try:
            while True:
                msg: LogMessage = await queue.get()
                if msg.is_complete:
                    is_error = False
                    break
        except Exception as e:
            is_error = True
            if isinstance(e, asyncio.CancelledError):
                is_aborted = True
        finally:
            self.release_queue(id)
            logger.info(f"Released queue for {id}")
            self._agent_ops.end_run(id, status = (
                    RunStatus.KILLED
                    if is_aborted
                    else (RunStatus.FAILED if is_error else RunStatus.FINISHED)
                ))
            
        if not msg.is_complete:
            raise ValueError("Preference extraction did not complete successfully.")
        
        
        logger.info(f"Released queue for {msg.content}")
        
        return msg.content, 0 #TODO: add duration
        
    async def create_workflow_task(
        self,
        task: 'WorkflowTask',
        cancellation_token: Optional[CancellationToken] = None,
        user_id: Optional[str] = socket.gethostname(),
        inputs: Optional[Dict[str, str]] = {}
    ) -> Union[asyncio.Queue, str]:
        """
        Creates a workflow task, initializes trace, and returns the queue and id.
        """

        description = f"Workflow task: {task.value}"
        message_id = inputs.get("message_id", None)
        top_n_conversations = inputs.get("first_n_conversations", 10)
        
        if message_id is None:
            description += f" for first {top_n_conversations} conversations"

        run = self._agent_ops.create_run(
            description=description,
            user_id=user_id,
            tags={
                "hostname": socket.gethostname(),
                "message_id": message_id,
            },
        )
        
        run_id: str = str(uuid.uuid4())
        
        if run is not None:
            self._agent_ops.create_trace(run_id=run.info.run_id,
                                        name="Extraction",
                                        inputs=inputs)
            run_id = run.info.run_id
        else:
            logger.warning(f"Failed to create MLflow run. Proceeding the request of {user_id} without tracing.")
        
        queue = await self.create_queue(run_id)

        await self._create_workflow_task(
            id=run_id,
            task=task,
            inputs=inputs,
            cancellation_token=cancellation_token,
        )

        return queue, run_id
