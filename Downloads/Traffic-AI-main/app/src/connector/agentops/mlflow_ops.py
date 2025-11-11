from datetime import datetime
import logging
from typing import Any, Optional, Dict
from functools import wraps

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run, Span, RunStatus
from mlflow.entities.span import SpanType
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_BRANCH,
)
from src.copilot.utils.autogen_patching import apply_mlflow_autogen_patch
from src.connector.agentops.agentops import AgentOps
from src.copilot.utils.ssl_utils import fix_local_ssl

import os

os.environ["MLFLOW_TRACE_TIMEOUT_SECONDS"] = "500"
os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
os.environ["MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT"] = "1"
os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "true"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"

fix_local_ssl()

logger = logging.getLogger(__name__)


def require_initialization(func):
    """Decorator to check if MLflow is initialized before executing the method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            self.initialized = self.initialize()
            if not self.initialized:
                return None
        return func(self, *args, **kwargs)

    return wrapper


class MLflowOps(AgentOps):
    tracking_uri: str
    experiment: Experiment | None
    experiment_name: str
    client: MlflowClient | None
    runs: Dict[str, Run]
    root_spans: Dict[str, Span]
    descriptions: Dict[str, str]
    initialized: bool

    """
    MLflowOps handles MLflow trace and span management with thread-safe caching.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.runs = {}
        self.root_spans = {}
        self.descriptions = {}

        self.initialized = self.initialize()

    def initialize(self) -> bool:
        try:
            apply_mlflow_autogen_patch()
            mlflow.config.enable_async_logging()
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

            try:
                self.experiment = self.client.get_experiment_by_name(
                    self.experiment_name
                )
            except:
                experiment_id = self.client.create_experiment(self.experiment_name)
                self.experiment = self.client.get_experiment(experiment_id)
        except Exception as e:
            logger.error(f"Failed to initialize MLflowOps: {e}")
            return False
        return True

    def run(self, *args, **kwargs) -> Any:
        pass

    @require_initialization
    def create_run(
        self,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ) -> Run | None:
        run_name: str = datetime.now().strftime("%Y%m%d%H%M%S")
        if tags is None:
            tags = {}

        try:
            tags.update(
                {
                    MLFLOW_USER: user_id or "unknown",
                    MLFLOW_SOURCE_NAME: os.path.basename(__file__),
                    MLFLOW_SOURCE_TYPE: "PROJECT",
                }
            )

            for key in list(tags.keys()):
                if tags[key] is None:
                    del tags[key]
                elif not isinstance(tags[key], str):
                    tags[key] = str(tags[key])

            run: Run = self.client.create_run(
                run_name=run_name,
                experiment_id=self.experiment.experiment_id,
                tags=tags,
            )
            self.runs[run.info.run_id] = run
            if description:
                self.descriptions[run.info.run_id] = description

            logger.info(
                f"Created MLflow run with ID: {run.info.run_id}, name: {run_name}"
            )

            return run
        except Exception as e:
            logger.error(f"Failed to create MLflow run: {e}")
            return None

    @require_initialization
    def end_run(
        self,
        run_id: str,
        outputs: Optional[Any] = None,
        status: RunStatus = RunStatus.FINISHED,
    ) -> None:
        run = self.runs.get(run_id)
        if run is None:
            return

        try:
            self.end_trace(run_id=run_id, outputs=outputs)
            self.client.set_terminated(
                run_id=run.info.run_id, status=RunStatus.to_string(status)
            )

            logger.info(
                f"Ending MLflow run with ID: {run.info.run_id}, name: {run.info.run_name}"
            )

            del self.runs[run_id]
            if run_id in self.descriptions:
                del self.descriptions[run_id]
            return
        except Exception as e:
            logger.error(f"Failed to end MLflow run {run_id}: {e}")
            return        
        

    @require_initialization
    def create_trace(
        self,
        run_id: str,
        name: str,
        inputs: Optional[Any] = None,
        tags: Optional[dict[str, str]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span | None:
        run = self.runs.get(run_id)
        if run is None:
            return None

        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
                
            mlflow.start_run(
                run_id=run.info.run_id,
                description=self.descriptions.get(run_id),
                experiment_id=self.experiment.experiment_id,
            )

            self.root_spans[run.info.run_id] = self.client.start_trace(
                name=name,
                span_type=SpanType.AGENT,
                inputs=inputs,
                tags=tags,
                attributes=attributes,
            )

            mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))

            return self.root_spans[run.info.run_id]
        except Exception as e:
            logger.error(f"Failed to create trace for run {run_id}: {e}")
            return None

    @require_initialization
    def end_trace(self, run_id: str, outputs: Optional[Any] = None) -> None:
        run = self.runs.get(run_id)
        if run is None:
            return
        root_span = self.root_spans.get(run_id)
        if root_span is None:
            return

        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
                
            mlflow.start_run(
                run_id=run.info.run_id,
                description=self.descriptions.get(run_id),
                experiment_id=self.experiment.experiment_id,
            )

            self.client.end_trace(
                trace_id=root_span.trace_id,
                outputs=outputs,
            )
            del self.root_spans[run_id]

            mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
        except Exception as e:
            logger.error(f"Failed to end trace for run {run_id}: {e}")
            return

    @require_initialization
    def create_span(
        self,
        run_id: str,
        name: str,
        parent_id: Optional[str] = None,
        span_type: Optional[SpanType] = SpanType.AGENT,
        attributes: Optional[dict[str, Any]] = None,
        inputs: Optional[Any] = None,
    ) -> Span | None:
        run = self.runs.get(run_id)
        root_span = self.root_spans.get(run_id)
        if run is None or root_span is None:
            return None

        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
                
            trace_id = root_span.trace_id
            mlflow.start_run(
                run_id=run.info.run_id,
                description=self.descriptions.get(run_id),
                experiment_id=self.experiment.experiment_id,
            )

            span = self.client.start_span(
                name=name,
                trace_id=trace_id,
                parent_id=parent_id if parent_id else root_span.span_id,
                span_type=span_type,
                inputs=inputs,
                attributes=attributes,
            )

            mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))

            return span
        except Exception as e:
            logger.error(f"Failed to create span for run {run_id}: {e}")
            return None

    @require_initialization
    def end_span(
        self,
        run_id: str,
        span_id: str,
        outputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        run = self.runs.get(run_id)
        root_span = self.root_spans.get(run_id)
        if run is None or root_span is None:
            return

        try:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
                
            mlflow.start_run(
                run_id=run.info.run_id,
                description=self.descriptions.get(run_id),
                experiment_id=self.experiment.experiment_id,
            )

            self.client.end_span(
                trace_id=root_span.trace_id,
                span_id=span_id,
                outputs=outputs,
                attributes=attributes,
            )
            mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))
            return
        except Exception as e:
            logger.error(f"Failed to end span {span_id} for run {run_id}: {e}")
            return

    @require_initialization
    def create_model_span(
        self,
        run_id: str,
        name: str,
        model: str,
        model_parameters: Optional[Dict[str, Any]],
        parent_id: Optional[str] = None,
        inputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Span | None:
        run = self.runs.get(run_id)
        root_span = self.root_spans.get(run_id)
        if run is None or root_span is None:
            return None

        try:
            trace_id = root_span.trace_id
            messages = self._llm_messages_to_openai(inputs)
            attributes = {
                "ai.model.name": model,
                "ai.model.provider": "openai",
                "mlflow.chat.messages": messages,
            }
            if model_parameters:
                attributes.update(model_parameters)

            return self.client.start_span(
                name=name,
                trace_id=trace_id,
                parent_id=parent_id if parent_id else root_span.span_id,
                span_type=SpanType.CHAT_MODEL,
                inputs=inputs,
                attributes=attributes,
            )
        except Exception as e:
            logger.error(f"Failed to create model span for run {run_id}: {e}")
            return None

    @require_initialization
    def end_model_span(
        self,
        run_id: str,
        span_id: str,
        outputs: Optional[Any] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        if attributes is None:
            attributes = {}

        try:
            model_parameters = {
                "mlflow.chat.messages": self._llm_messages_to_openai(outputs)
            }
            attributes.update(model_parameters)
            self.end_span(
                run_id=run_id,
                span_id=span_id,
                outputs=outputs,
                attributes=attributes,
            )
        except Exception as e:
            logger.error(f"Failed to end model span {span_id} for run {run_id}: {e}")
            return

    @staticmethod
    def _llm_messages_to_openai(messages: list) -> list:
        """
        Convert a list of LLMMessage (SystemMessage, UserMessage, AssistantMessage, FunctionExecutionResultMessage)
        to OpenAI-style message dicts.
        """
        result = []

        for msg in messages:
            # Discriminator: msg.type
            if msg.type == "SystemMessage":
                result.append({"role": "system", "content": msg.content})
            elif msg.type == "UserMessage":
                # msg.content can be str or List[str | Image]
                # For simplicity, join list into string if needed
                if isinstance(msg.content, list):
                    content = " ".join(
                        item if isinstance(item, str) else "[Image]"
                        for item in msg.content
                    )
                else:
                    content = msg.content
                result.append({"role": "user", "content": content})
            elif msg.type == "AssistantMessage":
                # msg.content can be str or List[FunctionCall]
                if isinstance(msg.content, list):
                    tool_calls = []
                    for call in msg.content:
                        name = call.name
                        id = call.id
                        arguments = call.arguments
                        tool_calls.append(
                            {
                                "type": "function",
                                "id": id,
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        )
                    msg_dict = {"role": "assistant", "tool_calls": tool_calls}
                else:
                    content = msg.content
                    msg_dict = {"role": "assistant", "content": content}
                result.append(msg_dict)
            elif msg.type == "FunctionExecutionResultMessage":
                for res in msg.content:
                    result.append(
                        {"role": "tool", "content": res.content, "call_id": res.call_id}
                    )
        return result
