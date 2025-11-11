import logging
import json
from typing import (
    Dict, Optional, List, Union, Callable, Awaitable, Annotated
)

from pydantic import BaseModel

from autogen_core import (
    MessageContext,
    TopicId,
    message_handler,
    SingleThreadedAgentRuntime,
)
from autogen_core.models import (
    CreateResult,
    LLMMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import FunctionTool

from src.copilot.chat_client.chat_client import ChatClient
from src.copilot.agent.planner import PlannerRequestMessage
from src.copilot.agent.executor import (
    ExecutorRequestMessage,
    ExecutorReplyMessage,
)
from src.copilot.agent.agent import Agent
from src.connector.agentops.agentops import LogMessage
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.connector.agentops.mlflow_ops import MLflowOps

# Tool and agent configuration constants
DONE_TOOL_NAME = "done"
DONE_TOOL_DESCRIPTION = "Call this tool to mark the task as complete."
AGENT_NAME = "intention"
AGENT_DESCRIPTION = "Refine and research the user's question."
INTENTION_REQUEST = "intention_request"
REQUIRED_PROMPT_TEMPLATES = [INTENTION_REQUEST]
RESEARCH_TOPIC = "request_research"

logger = logging.getLogger(__name__)

class IntentionRequestMessage(BaseModel):
    chat_history: List[LLMMessage]
    sent_from: str
    forward_to: str

class IntentionReplyMessage(BaseModel):
    task: str
    domain_knowledges: Optional[str] = None

class Intention(Agent):
    def __init__(
        self,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        research_agent: Optional[str] = None,
        agent_ops: Optional[MLflowOps] = None,
        knowledge_center: Optional[KnowledgeCenter] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ):
        # Ensure all required prompt templates are provided
        for required in REQUIRED_PROMPT_TEMPLATES:
            if required not in prompt_templates:
                raise ValueError(f"Missing required prompt template: '{required}'")

        def done(
            thought: Annotated[str, "Brief reasoning or thought process."],
            result: Annotated[str, "Latest user request after reviewing chat history (include memory key if applicable)."],
        ) -> str:
            return result

        super().__init__(
            name=AGENT_NAME,
            description=AGENT_DESCRIPTION,
            chat_client=chat_client,
            tools=[FunctionTool(done, DONE_TOOL_DESCRIPTION, DONE_TOOL_NAME)],
            prompt_templates=prompt_templates,
            agent_ops=agent_ops,
            report_message=report_message
        )
        self._task = None
        self._topic = None
        self._sent_from = None
        self._research_agent = research_agent
        self._knowledge_center: Optional[KnowledgeCenter] = knowledge_center
        
    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        prompt_templates: Dict[str, str],
        chat_client: ChatClient,
        research_agent: Optional[str] = None,
        knowledge_center: Optional[KnowledgeCenter] = None,
        agent_ops: Optional[MLflowOps] = None,
        report_message : Callable[[LogMessage], Awaitable[None]] = None
    ) -> 'Intention':
        """
        Register the Intention agent with the agent runtime.
        """
        return await Intention.register(
            runtime,
            type=AGENT_NAME,
            factory=lambda: Intention(
                prompt_templates=prompt_templates,
                chat_client=chat_client,
                research_agent=research_agent,
                knowledge_center=knowledge_center,
                agent_ops=agent_ops,
                report_message=report_message
            )
        )
        
    async def publish(self, ctx: MessageContext, research_results: str = None) -> Optional[Union[PlannerRequestMessage, IntentionReplyMessage]]:
        """
        Publish the intention or research results to the appropriate topic.
        """
        if self._topic is None:
            return IntentionReplyMessage(
                task=self._task,
                domain_knowledges=research_results
            )

        # Prepare and publish planner request message
        msg = PlannerRequestMessage(
            task=self._task,
            sent_from=self._sent_from,
            domain_knowledges=research_results,
        )
        await self.publish_message(
            msg,
            topic_id=TopicId(self._topic, source=self.get_id()),
            cancellation_token=ctx.cancellation_token,
        )

        # End span for tracing
        agent_ops = self.get_agent_ops()
        
        if self._span is not None:
            agent_ops.end_span(
                run_id=self.get_id(),
                span_id=self._span.span_id,
                outputs=msg,
                attributes={"next_agent": self._topic}
            )

        return None
        
    @message_handler
    async def on_researcher_reply(
        self,
        message: ExecutorReplyMessage,
        ctx: MessageContext
    ) -> Optional[Union[PlannerRequestMessage, IntentionReplyMessage]]:
        """
        Handle researcher reply and publish results.
        """
        return await self.publish(ctx, message.content)
        
          
    @message_handler
    async def on_intention_request(
        self,
        message: IntentionRequestMessage,
        ctx: MessageContext
    ) -> Optional[Union[PlannerRequestMessage, IntentionReplyMessage]]:
        """
        Handle incoming intention requests, generate task, and optionally trigger research.
        """
        self._chat_history = []
        self._topic = message.forward_to
        self._sent_from = message.sent_from
        
        agent_ops = self.get_agent_ops()
        self._span = agent_ops.create_span(
            run_id=self.get_id(),
            name=self._name,
            inputs=message,
            parent_id=message.parent_id if "parent_id" in message else None
        )
        tools_dict = {tool.name: tool for tool in self._tools}
        
  
        if len(message.chat_history) > 1:
            # Prepare chat history for prompt
            messages = []
            for msg in message.chat_history:
                if isinstance(msg, UserMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AssistantMessage):
                    messages.append({"role": "assistant", "content": msg.content})

            prompt_vars = {"chat_history": messages}
            llm_result: CreateResult = await self.generate(
                ctx, SystemMessage(content=self.get_prompt(INTENTION_REQUEST, prompt_vars)), session_id=self.get_id()
            )

            if not isinstance(llm_result.content, list):
                raise ValueError(f"Unexpected LLM result type: {type(llm_result.content)}")

            for call in llm_result.content:
                tool = tools_dict.get(call.name)
                if not tool:
                    raise ValueError(f"Unknown tool: {call.name}")
                arguments = json.loads(call.arguments)
                if call.name == DONE_TOOL_NAME:
                    self._task = arguments["result"]
                    break

            if self._task is None:
                raise ValueError("No valid tool call found in LLM result.")
            
            await self._report_message(
                LogMessage(
                    action="Your request has been analyzed and categorized as:",
                    agent_name=self._name,
                    content=self._task,
                    id=self.get_id(),
            ))
        else:
            self._task = message.chat_history[0].content
            
        if self._research_agent:
            # Trigger research for the task

            await self.publish_message(
                ExecutorRequestMessage(
                    task=f"Gather background information relevant to the user's shipping question: {self._task}",
                    success_criteria="Relevant background knowledge for the user's question",
                    sent_from=self._name,
                    user_message=None,
                    parent_span_id=self._span.span_id if self._span is not None else None,
                ),
                topic_id=TopicId(self._research_agent, source=self.get_id()),
                cancellation_token=ctx.cancellation_token
            )
            return None

        return await self.publish(ctx, await self._knowledge_center.search_domain_knowledge_basic(self._task) if self._knowledge_center else "")
        


    