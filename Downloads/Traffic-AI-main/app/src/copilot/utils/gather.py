import asyncio
import logging
import json
import traceback
from mlflow.entities import Span
from typing import Any, Optional

from mlflow.entities.span import SpanType
from autogen_core.models import FunctionExecutionResult
from autogen_core.tools import Tool
from autogen_core import (
    FunctionCall,
)

from src.connector.agentops.mlflow_ops import MLflowOps
from src.copilot.utils.mcp import parse_mcp_return

logger = logging.getLogger(__name__)

async def gather_with_retries(task_factories, max_retries=5):
    """
    task_factories: list of zero-arg callables that return awaitables
    """
    results = [None] * len(task_factories)
    attempts = [0] * len(task_factories)
    pending = set(range(len(task_factories)))

    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        logger.debug(f"Gather attempt {attempt} for indices: {pending}")
        # Only (re)run tasks that haven't succeeded yet
        current_tasks = [task_factories[i]() for i in pending]
        gathered = await asyncio.gather(*current_tasks)
        for idx, res in zip(list(pending), gathered):
            if not isinstance(res, Exception):
                results[idx] = res
                pending.remove(idx)
            else:
                attempts[idx] += 1
                if attempt == max_retries:
                    results[idx] = res  # Give up after last attempt
                    pending.remove(idx)
    logger.debug(f"Gather completed with {len(results)} results")
    return results

async def run_tool_with_retries(
    arguments: Any,
    tool: 'Tool',
    call: FunctionCall,
    ctx,
    agent_ops: MLflowOps,
    session_id: str,
    parent_span_id: Optional[str],
    max_retries: int = 2,
    timeout_seconds: int = 240,
    delay: int = 2,
) -> 'FunctionExecutionResult':
    for attempt in range(1, max_retries + 1):
        try:
            span: Span | None = None
            if parent_span_id is not None:
                span = agent_ops.create_span(
                    run_id=session_id,
                    name=f"{call.name}_({call.id[-5:]})",
                    inputs=json.loads(call.arguments),
                    parent_id=parent_span_id,
                    span_type=SpanType.TOOL,
                    attributes={"call_id": call.id},
                )
            result = await asyncio.wait_for(tool.run_json(arguments, ctx.cancellation_token), timeout=timeout_seconds)
            result_str = parse_mcp_return(tool.return_value_as_string(result))
            
            if span is not None:
                agent_ops.end_span(
                    run_id=session_id,
                    span_id=span.span_id,
                    outputs=result_str,
                )
            
            return FunctionExecutionResult(
                call_id=call.id,
                content=result_str,
                is_error=False,
                name=call.name
            )
        except Exception as e:
            log = traceback.format_exc()
            error_msg = str(e).lower()
            logger.error(f"Attempt {attempt} for {call.name} failed: {log}")
            if "Input validation error".lower() in error_msg:
                return FunctionExecutionResult(
                    call_id=call.id,
                    content=f"Validation error: {e}. Please verify the tool call format and try again. (Ensure your input matches the required format including the parameter's type)",
                    is_error=True,
                    name=call.name
                )
            if attempt < max_retries:
                logger.error(f"Retrying {call.name} in {delay} seconds ")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff, optional
            else:
                exception = ''
                if (isinstance(e, ExceptionGroup)):
                    exception = [str(ex) for ex in e.exceptions]
                else:
                    exception = str(e)
                logger.error(f"[{call.name}] All retries failed with Exception {exception}.")
            
                return FunctionExecutionResult(
                    call_id=call.id,
                    content=f"All retries failed for {call.name} with Exception {exception}.",
                    is_error=True,
                    name=call.name
                )