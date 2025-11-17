import asyncio
import json
from argparse import ArgumentParser
from typing import List, Tuple
import uuid
import os
import logging
import socket
import multiprocessing
import atexit
from contextlib import closing

import pandas as pd
import mlflow

from src.copilot.utils.autogen_patching import apply_mlflow_autogen_patch
from src.configurator.configurator import Configurator
from src.connector.connector import Connector
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.vo.message import Message
from src.copilot.chat_client.chat_client_creator import ChatClientCreator
from src.copilot.copilot_v3 import CopilotAgentRuntime, ToolRegistry
from src.copilot.evaluate.evaluate import evaluate as evaluate_fn
from src.copilot.evaluate.evaluate_set import get_eval_set, get_sample_from_dict
from src.copilot.utils.agents_config import AgentConfig
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.copilot.utils.mcp import MCPTool
from src.copilot.utils.message import WorkingMemoryService
from src.copilot.evaluate.evaluate import evaluate
from src.copilot.copilot_v3 import WorkflowTask


apply_mlflow_autogen_patch()
mlflow.config.enable_async_logging()

async def init_environment(local: bool = False, use_mock_sql: bool = False) -> CopilotAgentRuntime:
    """Initialize runtime.

    When local=True, override remote endpoints with local mock services and disable MLflow tracking.
    When use_mock_sql=True, use hardcoded SQL responses instead of real database queries.
    """
    Configurator.load_config()
    config = Configurator.get_config()

    if local:
        logging.warning("Running in LOCAL MODE: overriding endpoints to localhost and disabling MLflow tracking.")
        # Merge local overrides if example file exists
        local_cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "json_config_local.example.json")
        try:
            with open(local_cfg_path, "r", encoding="utf-8") as f:
                local_cfg = json.load(f)
            # apply selective overrides
            for section in ["api_host", "mcp_tool", "mlflow"]:
                if section in local_cfg:
                    config[section] = local_cfg[section]
        except Exception as e:
            logging.warning(
                f"Failed loading local example config: {e}. Falling back to static overrides."
            )
            config["api_host"]["sana_knowledge_center"] = "http://127.0.0.1:8001"
            config["api_host"]["working_memory_service"] = "http://127.0.0.1:8003"
            # Keep realtime cms optional
            config["mcp_tool"] = {"local_mcp": "http://127.0.0.1:8002/traffic-copilot/sse"}
            config["mlflow"]["tracking_uri"] = None
    
    if use_mock_sql:
        logging.warning("MOCK SQL MODE: sql_execution tool will return hardcoded data.")

    await Connector.initiate()

    kc = Connector.get_keycloak()
    aops = Connector.get_mlflow()
    gateway_host = config["llm_gateway"]["host"]
    # Debug log of resolved endpoints to help diagnose 404/mis-port issues
    wm_host = config.get("api_host", {}).get("working_memory_service")
    kc_host = config.get("api_host", {}).get("sana_knowledge_center")
    mcp_cfg = config.get("mcp_tool", {})
    logging.info(f"[init_environment] Resolved KnowledgeCenter host: {kc_host}")
    logging.info(f"[init_environment] Resolved WorkingMemory host: {wm_host}")
    logging.info(f"[init_environment] Resolved MCP tool endpoints: {mcp_cfg}")
    # Quick validation: local mode expects MCP SSE on port 8002 (mcp_tool_local) not 8003 (working_memory)
    if local:
        bad_ports = [url for url in mcp_cfg.values() if ":8003" in url]
        if bad_ports:
            logging.warning(f"[init_environment] MCP tool endpoints incorrectly point to working memory port 8003: {bad_ports}. "
                            "Ensure they use 8002. Auto-correcting.")
            # Auto-correct by rewriting endpoints to 8002 if pattern matches
            corrected = {}
            for name, url in mcp_cfg.items():
                if ":8003" in url:
                    corrected[name] = url.replace(":8003", ":8002")
                else:
                    corrected[name] = url
            config["mcp_tool"] = corrected
            logging.info(f"[init_environment] Corrected MCP tool endpoints: {config['mcp_tool']}")
    kdc = KnowledgeCenter(config["api_host"]["sana_knowledge_center"])
    wms = WorkingMemoryService(config["api_host"]["working_memory_service"], keycloak=kc)
    aops = Connector.get_mlflow()
    if local or not config.get("mlflow", {}).get("tracking_uri"):
        # Disable mlflow ops usage inside runtime gracefully
        
        class DummyAgentOps:
            def create_run(self, **kwargs):
                logging.info("[DummyAgentOps] Skipping create_run.")
                return None
            
            
            def create_trace(self, **kwargs):
                logging.info("[DummyAgentOps] Skipping create_trace.")

                
            def create_span(self, **kwargs):
                logging.info("[DummyAgentOps] create_span called but skipped.")
                return None

            
            def end_span(self, **kwargs):
                logging.info("[DummyAgentOps] end_span called but skipped.")
                return None

            
            def end_run(self, **kwargs):
                logging.info("[DummyAgentOps] end_run called but skipped.")
                return None
            
            def create_model_span(self, **kwargs):
                logging.info("[DummyAgentOps] create_model_span called but skipped.")
                return None


        aops = DummyAgentOps()

    agent_configs = AgentConfig.get_agent_config("agent_config")

    tool_registry = ToolRegistry()
    for name, url in config.get("mcp_tool", {}).items():
        # In local mode we may not want auth header; MCPTool should handle missing keycloak gracefully
        tool_registry.register_tool(MCPTool(name, url, keycloak=None if local else kc))

    runtime = CopilotAgentRuntime(
        chat_client_creator=ChatClientCreator(kc, gateway_host),
        copilot_config=agent_configs,
        tool_registry=tool_registry,
        knowledge_center=kdc,
        working_memory_service=wms,
        agent_ops=aops,
        use_mock_sql=use_mock_sql  # Enable mock SQL if requested
    )
    await runtime.start()
    return runtime

# --------------------- Local Stack Helpers ---------------------
LOCAL_SERVICES: list[tuple[str, int]] = [
    ("local_services.knowledge_center_local:app", 8001),
    ("local_services.mcp_tool_local:app", 8002),
    ("local_services.working_memory_local:app", 8003),
]
_LOCAL_PROCS: list[multiprocessing.Process] = []

def _is_port_open(port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0

def _spawn_uvicorn(target: str, port: int):
    import uvicorn
    dev_reload = os.getenv("DEV_RELOAD", "0") == "1"
    logging.info(f"[local-stack] starting {target} on {port} (reload={'ON' if dev_reload else 'OFF'})")
    uvicorn.run(target, host="127.0.0.1", port=port, reload=dev_reload, access_log=False, log_level="info")

def maybe_spawn_local_stack(enable: bool):
    if not enable:
        return
    missing = [(t, p) for (t, p) in LOCAL_SERVICES if not _is_port_open(p)]
    if not missing:
        logging.info("[local-stack] all services already running")
        return
    logging.warning(f"[local-stack] spawning {len(missing)} services: {missing}")
    for target, port in missing:
        proc = multiprocessing.Process(target=_spawn_uvicorn, args=(target, port), daemon=True)
        proc.start()
        _LOCAL_PROCS.append(proc)
    def _cleanup():
        for p in _LOCAL_PROCS:
            if p.is_alive():
                logging.info(f"[local-stack] terminating {p.pid}")
                p.terminate()
    atexit.register(_cleanup)

async def chat(runtime: CopilotAgentRuntime):
    messages = []
    conversation_id = uuid.uuid4()
    
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    USER_COLOR = Fore.GREEN
    BOT_COLOR = Fore.CYAN
    RESET = Style.RESET_ALL

    print(f"{BOT_COLOR}Chat started! Conversation ID: {conversation_id}{RESET}")
    print(f"{BOT_COLOR}Type 'exit' to quit, 'clear' to reset history, 'help' for commands.{RESET}\n")
    
    last_metadata = None  # Store last metadata for column description
    
    while True:
        user_input = input(f"{USER_COLOR}You: {RESET}").strip()
        if user_input.lower() == 'exit':
            print(f"{BOT_COLOR}Chat ended.{RESET}")
            break
        if user_input.lower() == 'clear':
            messages.clear()
            print(f"{BOT_COLOR}Messages cleared.{RESET}\n")
            continue
        if user_input.lower() == 'help':
            print(f"{BOT_COLOR}Commands:\n  exit  - quit\n  clear - clear history\n  help  - show this help\n{RESET}")
            continue
        if user_input.lower() == 'showdesc':
            if last_metadata and 'data_description' in last_metadata:
                print(f"{BOT_COLOR}Column Descriptions:{RESET}")
                for col, desc in last_metadata['data_description'].items():
                    print(f"  {Fore.YELLOW}{col}{RESET}: {desc}")
            else:
                print(f"{Fore.RED}No column descriptions available.{RESET}")
            continue
        
        messages.append(Message(content=user_input, role=RoleEnum.USER, metadata=None))
        try:
            message, duration = await runtime.create_until_finish(
                messages=messages,
                print_log=True,
                conversation_id=conversation_id
            )
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{RESET}")
            continue
        
        messages.append(message)
        message_dict = message.to_dict()
        print(f"{BOT_COLOR}Copilot: {message_dict['content']}{RESET}")

        # Print metadata if available
        metadata = message_dict.get('metadata', {})
        shipment = None
        if metadata:
            if 'SHIPMENT' in metadata:
                shipment = metadata['SHIPMENT']
            elif 'TABLE' in metadata and 'data' in metadata['TABLE']:
                shipment = metadata['TABLE']['data']
            if shipment:
                print(f"{BOT_COLOR}Metadata:{RESET}")
                print(pd.DataFrame(shipment))

        if getattr(message, 'others', None):
            print(f"{BOT_COLOR}Others:{RESET} {message.others}")

        print(f"{BOT_COLOR}(duration: {duration} ms){RESET}\n")

async def predict(runtime: CopilotAgentRuntime) -> Tuple[Message, int]:
    with open("./verification/predict_in/predict_in.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Data should be a dictionary with sections as keys.")

    message, duration = await runtime.create_until_finish(messages=get_sample_from_dict(data))
    message_dict = message.to_dict()
    print(f"Copilot response (content): {message_dict['content']}")
    print(f"Copilot response (duration): {duration} ms")
    print(f"Copilot response (metadata): {message_dict['metadata']}")
    return message, duration

async def preference_extraction(runtime: CopilotAgentRuntime):
    print(await runtime.create_workflow_task_until_finish(task=WorkflowTask.PREFERENCE_EXTRACTION, inputs={"message_id": "690021ba4e1858e774d09681", "user_id": "CHENGST"}))


async def evaluate(runtime: CopilotAgentRuntime, eval_set_name: str, num_trials: int = 4, batch_size: int = 0):
    eval_set_file = f"./verification/eval_sets/{eval_set_name}.json"
    eval_set: List[Tuple[List[Message], str]] = get_eval_set(eval_set_file)
    if batch_size == 0:
        batch_size = len(eval_set)
    accuracy, _ = await evaluate_fn(runtime, eval_set, num_trials, batch_size)
    print(f"\nCopilot accuracy@{num_trials} for {len(eval_set)} samples: {accuracy}")

async def main_async(args):
    # Optionally spawn local mock services before initializing runtime
    if args.local and args.spawn_local_stack:
        maybe_spawn_local_stack(True)
        # wait for ports to be ready (max ~10s)
        for _ in range(40):
            if all(_is_port_open(p) for _, p in LOCAL_SERVICES):
                break
            await asyncio.sleep(0.25)
        else:
            logging.warning("[local-stack] timeout waiting for services; proceeding anyway")
    runtime = await init_environment(local=args.local, use_mock_sql=args.mock_sql)
    if args.mode == "evaluate":
        await evaluate(runtime, eval_set_name=args.eval_set_name, num_trials=args.num_trials, batch_size=args.batch_size)
    elif args.mode == "predict":
        await predict(runtime)
    elif args.mode == "pe":
        await preference_extraction(runtime)
    else:
        # Enhanced non-interactive single message support with optional top-up waiting
        if getattr(args, "user_message", None):
            from src.conversations.vo.message import Message
            from src.conversations.enum.role_enum import RoleEnum
            user_msg = args.user_message
            wait_topup = getattr(args, "wait_topup", False)
            if not wait_topup:
                try:
                    message, duration = await runtime.create_until_finish(
                        messages=[Message(content=user_msg, role=RoleEnum.USER, metadata=None)],
                        print_log=True,
                        conversation_id=uuid.uuid4()
                    )
                    print(f"Single chat response: {message.to_dict().get('content')} (duration: {duration} ms)")
                except Exception as e:
                    print(f"Error running single chat message: {e}")
                return
            # wait_topup logic: manually consume queue until top-up completes or timeout
            try:
                queue, run_id = await runtime.create(
                    messages=[Message(content=user_msg, role=RoleEnum.USER, metadata=None)],
                    user_id=socket.gethostname(),
                    conversation_id=uuid.uuid4()
                )
            except Exception as e:
                print(f"Failed to start runtime create: {e}")
                return
            import re, time, json
            start = time.time()
            injection_detected = False
            topup_completed = False
            high_empty_ratio_detected = False
            extracted_port = None
            extracted_svvd = None
            extracted_ship_no = None
            timeout = getattr(args, "wait_topup_timeout", 35)
            fallback_sent = False
            while True:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    print("Timeout waiting for top-up completion.")
                    break
                try:
                    log_msg = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    print("Timeout while awaiting queue log message.")
                    break
                content = getattr(log_msg, 'content', '') or ''
                action = getattr(log_msg, 'action', '') or ''
                if action == "Injected deterministic top-up plan":
                    injection_detected = True
                    try:
                        meta = json.loads(content)
                        extracted_port = meta.get("port", extracted_port)
                        extracted_svvd = meta.get("svvd", extracted_svvd)
                    except Exception:
                        pass
                # Detect high empty_ratio from shipment details content
                if not high_empty_ratio_detected and re.search(r'empty ratio is (\d+)%', content):
                    try:
                        ratio = int(re.search(r'empty ratio is (\d+)%', content).group(1))
                        if ratio >= 85:
                            high_empty_ratio_detected = True
                    except Exception:
                        pass
                # Extract ship number, port, svvd from JSON references if available
                if log_msg.references:
                    try:
                        ref_data = log_msg.references[0].get('content')
                        if isinstance(ref_data, list) and ref_data and isinstance(ref_data[0], dict):
                            row = ref_data[0]
                            extracted_ship_no = str(row.get("SHIPMENT_NUMBER", extracted_ship_no))
                            extracted_port = row.get("VOY_STOP_PORT_CDE", extracted_port)
                            extracted_svvd = row.get("SVVD", extracted_svvd)
                    except Exception:
                        pass
                # Detect top-up completion by source meta_data
                if log_msg.references:
                    try:
                        meta_data = log_msg.references[0].get('meta_data', {})
                        if isinstance(meta_data, dict):
                            data_desc = meta_data.get('data_description', {})
                            if isinstance(data_desc, dict) and data_desc.get('source') == 'get_eligible_topup_shipment':
                                topup_completed = True
                    except Exception:
                        pass
                # Print log line (compact)
                print(f"[LOG] {action} | {content[:240]}")
                if topup_completed:
                    print("Top-up workflow completed.")
                    break
                # Fallback: if high empty ratio detected but no injection after 10s, trigger second message
                if high_empty_ratio_detected and not injection_detected and not fallback_sent and (time.time() - start) > 10:
                    fallback_sent = True
                    if extracted_svvd and extracted_port:
                        follow_msg = f"Find shipments from other services to top up {extracted_svvd} at {extracted_port}."
                        print(f"[FALLBACK] Sending second message: {follow_msg}")
                        from src.conversations.vo.message import Message as Msg2
                        try:
                            queue2, run_id2 = await runtime.create(
                                messages=[Msg2(content=follow_msg, role=RoleEnum.USER, metadata=None)],
                                user_id=socket.gethostname(),
                                conversation_id=uuid.uuid4()
                            )
                            # Consume until eligible shipments appear or immediate completion
                            while True:
                                try:
                                    log2 = await asyncio.wait_for(queue2.get(), timeout=timeout - (time.time() - start))
                                except asyncio.TimeoutError:
                                    print("Fallback timeout.")
                                    break
                                c2 = getattr(log2, 'content', '') or ''
                                a2 = getattr(log2, 'action', '') or ''
                                print(f"[FALLBACK-LOG] {a2} | {c2[:200]}")
                                if log2.references:
                                    try:
                                        md2 = log2.references[0].get('meta_data', {})
                                        if isinstance(md2, dict):
                                            ds2 = md2.get('data_description', {})
                                            if isinstance(ds2, dict) and ds2.get('source') == 'get_eligible_topup_shipment':
                                                topup_completed = True
                                                print("Top-up completion detected in fallback path.")
                                                break
                                    except Exception:
                                        pass
                                if getattr(log2, 'is_complete', False):
                                    break
                        except Exception as fe:
                            print(f"Fallback second message failed: {fe}")
                if getattr(log_msg, 'is_complete', False) and not (high_empty_ratio_detected and not topup_completed):
                    # Normal completion when not waiting for top-up continuation
                    break
            print("Non-interactive chat session ended.")
            return
        await chat(runtime)

def main():
    parser = ArgumentParser(description="Copilot evaluation, prediction, and chat utility.")
    parser.add_argument(
        'mode',
        type=str,
        choices=['evaluate', 'predict', 'chat', 'pe'],
        help='Choose between `evaluate`, `predict`, `pe` or `chat` mode.'
    )
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials for stability evaluation (default: 4)')
    parser.add_argument('--batch_size', type=int, default=0, help='Number of samples to run evaluation in parallel (default: number of samples)')
    parser.add_argument('--eval_set_name', type=str, default="2025_06_11", help='Evaluation set name (default: 2025_06_11)')
    parser.add_argument('--local', action='store_true', help='Run using local mock services (knowledge center, MCP, working memory) without remote auth/MLflow.')
    parser.add_argument('--spawn-local-stack', action='store_true', help='Auto-start local mock services if not already running.')
    parser.add_argument('--mock-sql', action='store_true', help='Use hardcoded SQL responses for testing (bypasses database).')
    parser.add_argument('--user-message', type=str, default=None, help='(Chat mode only) Provide a single user message to run non-interactively and exit.')
    parser.add_argument('--wait-topup', action='store_true', help='(Chat mode only) Wait for deterministic top-up workflow completion when high empty_ratio detected.')
    parser.add_argument('--wait-topup-timeout', type=int, default=35, help='Timeout (seconds) for waiting top-up workflow completion (default: 35).')
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
