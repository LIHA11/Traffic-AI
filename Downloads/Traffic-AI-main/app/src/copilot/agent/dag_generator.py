# Standard Library Imports
import logging
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# Third-party library imports
from pydantic import BaseModel, ValidationError

# Core Module Imports
from autogen_core import (
    MessageContext,
    TopicId,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import (
    CreateResult,
    SystemMessage,
    UserMessage,
)

# Project-Specific Imports
from src.copilot.agent.agent import Agent
from src.copilot.chat_client.chat_client import ChatClient
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.langfuse_ops import LangfuseOps

logger = logging.getLogger(__name__)

DAG_GENERATOR_REQUEST = "dag_generator_request"
DAG_GENERATOR_TOPIC = "dag_generator"

class Node(BaseModel):
    task_description: str
    success_criteria: str

class Edge(BaseModel):
    source: int
    target: int

class DAG(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class DAGGeneratorOutput(BaseModel):
    dag: DAG

class DAGGeneratorRequestMessage(BaseModel):
    plan: str
    sent_from: Optional[str] = None
    parent_id: Optional[str] = None

class DAGGeneratorReplyMessage(BaseModel):
    dag: DAG
    

    
class DAGGenerator(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None,
    ):
        super().__init__(
            name=DAG_GENERATOR_TOPIC,
            description="Converts a plan into a Directed Acyclic Graph (DAG) representation.",
            chat_client=chat_client,
            prompt_templates=prompt_templates,
            response_format=DAGGeneratorOutput,
            agent_ops=agent_ops,
            report_message=report_message
        )
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        agent_ops: Optional[LangfuseOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ) -> "DAGGenerator":
        return await DAGGenerator.register(
            runtime,
            type=DAG_GENERATOR_TOPIC,
            factory=lambda: DAGGenerator(
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                agent_ops=agent_ops,
                report_message=report_message
            ),
        )
        
            
    @message_handler
    async def on_request(
        self,
        message: DAGGeneratorRequestMessage,
        ctx: MessageContext,
    ) -> Optional[DAGGeneratorReplyMessage]:
        """
        Handles incoming DAG generation requests.
        """
        self._chat_history = []
        agent_ops = self.get_agent_ops()
        span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )
        
        prompt = self.get_prompt(DAG_GENERATOR_REQUEST, message)
        llm_result = await self.generate(ctx, new_message=SystemMessage(content=prompt), session_id=self.get_id())
        done, parsed_result = self._handle_llm_result(llm_result)

        # Retry loop for LLM result parsing
        while not done:
            llm_result = await self.generate(ctx, UserMessage(content=f"{parsed_result}, please try again.", source=self.get_id()), session_id=self.get_id())
            done, parsed_result = self._handle_llm_result(llm_result)

        reply_message = DAGGeneratorReplyMessage(
            dag=parsed_result.dag,
        )
        
        if span is not None:
            agent_ops.end_span(
                run_id=self.get_id(),
                span_id=span.span_id,
                outputs=reply_message
            )

        if message.sent_from:
            await self.publish_message(
                reply_message,
                topic_id=TopicId(message.sent_from, source=self.get_id()),
                cancellation_token=ctx.cancellation_token,
            )
            return None
        else:
            return reply_message
        
    @staticmethod
    def _handle_llm_result(
        llm_result: CreateResult,
    ) -> Tuple[bool, Union[DAGGeneratorOutput, str]]:
        """
        Handles and parses the result from the LLM.
        """
        # If llm_result is a CreateResult, extract content
        content = getattr(llm_result, 'content', None)
        if content is None:
            return False, "No content in LLM result."

        try:
            dag_generator_out = DAGGeneratorOutput.model_validate_json(content)
            if len(dag_generator_out.dag.nodes) == 1 and len(dag_generator_out.dag.edges) == 1:
                raise Exception(
                    "No cycle in the edges is allowed, meaning no node can depend on itself or create a loop. (If there are only one node, no edges should be present.)"
                )
        except ValidationError as e:
            return False, f"Validation error: {e}"
        except Exception as e:
            return False, f"Error: {e}"

        return True, dag_generator_out
    
    
