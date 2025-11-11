# Standard library imports
import logging
import re
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

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
        
        # Generate a response using the language model
        llm_result: CreateResult = await self.generate(
            ctx, 
            new_message=UserMessage(content=content, source=self.get_id()),
            session_id=self.get_id()
        )
        
        # Handle and publish the generated response
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
    