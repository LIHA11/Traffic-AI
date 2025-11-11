import logging
import json
from typing import Awaitable, Callable, Optional, List

from autogen_core import (
    MessageContext,
    SingleThreadedAgentRuntime,
    message_handler,
)

import uuid

from autogen_core import (
    AgentId,
    MessageContext,
    SingleThreadedAgentRuntime,
    message_handler,
)

from src.copilot.utils.gather import gather_with_retries

from src.copilot.agent.agent import Agent
from src.connector.agentops.agentops import LogMessage
from src.connector.agentops.langfuse_ops import LangfuseOps
from src.copilot.agent.preference.constant import (
    PreferenceExtractionMasterRequest,
    PREFERENCE_EXTRACTOR_MASTER_TOPIC,
    MESSAGE_GROUPER_TOPIC,
    MessageGrouperResponse,
    MessageGrouperRequest,
    KeywordsExtractionRequest,
    KeywordsExtractionResponse,
    KEYWORDS_EXTRACTION_TOPIC,
    SummaryExtractionRequest,
    SUMMARY_EXTRACTION_TOPIC,
    SummaryExtractionResponse,
)
from src.copilot.utils.knowledge_center import KnowledgeCenter
from src.conversations.vo.message import Message

logger = logging.getLogger(__name__)

def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

class PreferenceExtractionMaster(Agent):
    def __init__(
        self,
        kc: KnowledgeCenter,
        agent_ops: Optional[LangfuseOps] = None,
        report_message: Optional[Callable[[LogMessage], Awaitable[None]]] = None,
    ):
        super().__init__(
            name=PREFERENCE_EXTRACTOR_MASTER_TOPIC,
            description="Master agent to handle preference extraction by coordinating message grouping and keyword extraction.",
            agent_ops=agent_ops,
            report_message=report_message,
            chat_client=None,
            prompt_templates={}
        )
        self._kc = kc

    @staticmethod
    async def register_agent(
        runtime: SingleThreadedAgentRuntime,
        kc: KnowledgeCenter,
        agent_ops: Optional[LangfuseOps] = None,
        report_message: Optional[Callable[[LogMessage], Awaitable[None]]] = None
    ) -> "PreferenceExtractionMaster":
        return await PreferenceExtractionMaster.register(
            runtime,
            type=PREFERENCE_EXTRACTOR_MASTER_TOPIC,
            factory=lambda: PreferenceExtractionMaster(
                agent_ops=agent_ops,
                kc=kc,
                report_message=report_message
            ),
        )

    @message_handler
    async def on_request(
        self,
        message: PreferenceExtractionMasterRequest,
        ctx: MessageContext,
    ) -> None:
        from src.conversations.svc.conversation_svc import fetch_conversations_by_user_id, get_conversation_by_message_id, get_message_groups, update_messages_group_id
        inputs = []
        
        # TODO: Filtered by like
        # TODO: Whole update mode
        # TODO: Updated message group

        user_id = message.user_id
        messages: List[Message] = []
        if message.message_id is not None:
            conversation = get_conversation_by_message_id(message.message_id)
            messages = conversation.messages if conversation else []
            
            if not messages or len(messages) == 0:
                logger.warning(f"No conversation found for message_id: {message.message_id}")
                return None

            messages_groups = get_message_groups(messages)
            inputs.append((messages, messages_groups, user_id))
        else:
            conversations = fetch_conversations_by_user_id(user_id)
            conversations = conversations[: message.first_n_conversations if message.first_n_conversations else 10]
            print(f"Using the first {len(conversations)} conversations for user_id: {user_id} for preference extraction")
            messages_by_conversation = [conv.messages for conv in conversations if conv.messages]
            if not messages_by_conversation or len(messages_by_conversation) == 0:
                logger.warning(f"No messages found for user_id: {user_id}")
                return None
            for messages in messages_by_conversation:
                messages_groups = get_message_groups(messages)
                inputs.append((messages, messages_groups, user_id))

        message_agent_pairs = [
            (
                MessageGrouperRequest(messages=[message.to_dict() for message in messages], messages_groups=messages_groups, refresh_all_msgs=(True)),
                AgentId(MESSAGE_GROUPER_TOPIC, f"{self.id.key}_{i}"),
            )
            for i, (messages, messages_groups, _) in enumerate(inputs)
        ]
        
        task_factories = [
            lambda m=msg, a=aid: self.send_message(m, a)
            for msg, aid in message_agent_pairs
        ]
        
        BATCH_SIZE = 15
        grouper_responses: List[MessageGrouperResponse] = []

        for batch in batched(task_factories, BATCH_SIZE):
            # Run gather_with_retries on the current batch
            responses = await gather_with_retries(batch, max_retries=3)
            grouper_responses.extend(responses)
            # Print the progress
            print(f"Processed {len(grouper_responses)} out of {len(inputs)} conversation grouping tasks.")

        print(f"{sum([len(resp.messages_by_groups) if resp.messages_by_groups is not None else 0 for resp in grouper_responses])} of messages groups are generated for {len(inputs)} conversations.")
        print(f"Starting keyword and summary extraction.")

        extraction_inputs = []
        
        for i, response in enumerate(grouper_responses):
            if response.messages_by_groups is not None:
                for messages in response.messages_by_groups:
                    message_group_id = str(uuid.uuid4())
                    update_messages_group_id(user_id=user_id, message_ids=[msg.get("id") for msg in messages if msg.get("id") is not None], group_id=message_group_id)
                    if len(messages) > 1:
                        old_message_group_ids = [msg.get("groupId") for msg in messages if msg.get("groupId") is not None]
                        extraction_inputs.append((messages, message_group_id, inputs[i][2], old_message_group_ids))
                
        message_agent_pairs = [
            (
                KeywordsExtractionRequest(messages=messages),
                AgentId(KEYWORDS_EXTRACTION_TOPIC, f"{self.id.key}_{i}"),
            )
            for i, (messages, _, _, _) in enumerate(extraction_inputs)
        ]
        
        task_factories = [
            lambda m=msg, a=aid: self.send_message(m, a)
            for msg, aid in message_agent_pairs
        ]
        
        ke_responses: List[KeywordsExtractionResponse] = []
        
        for batch in batched(task_factories, BATCH_SIZE):
            # Run gather_with_retries on the current batch
            responses = await gather_with_retries(batch, max_retries=3)
            ke_responses.extend(responses)
            # Print the progress
            print(f"Processed {len(ke_responses)} out of {len(extraction_inputs)} keyword extraction tasks.")

        logger.info("Updating Knowledge Centre.")

        # Insert into RAG store
        for ind, response in enumerate(ke_responses):
            if response.keywords is not None:
                await self._kc.insert_keywords(response.keywords, extraction_inputs[ind][1], extraction_inputs[ind][2], extraction_inputs[ind][3])
            
        message_agent_pairs = [
            (
                SummaryExtractionRequest(messages=messages),
                AgentId(SUMMARY_EXTRACTION_TOPIC, f"{self.id.key}_{i}"),
            )
            for i, (messages, _, _, _) in enumerate(extraction_inputs)
        ]
        
        task_factories = [
            lambda m=msg, a=aid: self.send_message(m, a)
            for msg, aid in message_agent_pairs
        ]
        
        su_responses: List[SummaryExtractionResponse] = []
        
        for batch in batched(task_factories, BATCH_SIZE):
            # Run gather_with_retries on the current batch
            responses = await gather_with_retries(batch, max_retries=3)
            su_responses.extend(responses)
            # Print the progress
            logger.info(f"Processed {len(su_responses)} out of {len(extraction_inputs)} summary extraction tasks.")

        logger.info("Completed summary extraction.")
        logger.info("Updating Knowledge Centre.")

        # Insert into RAG store
        for ind, response in enumerate(su_responses):
            if response.summary is not None:
                await self._kc.insert_summary(response.summary, extraction_inputs[ind][1], extraction_inputs[ind][2], extraction_inputs[ind][3])

        # List of dict with message group id, summary and keywords
        results = []
        for i in range(len(extraction_inputs)):
            result = {
                "message_group_id": extraction_inputs[i][1],
                "summary": su_responses[i].summary if i < len(su_responses) else None,
                "keywords": ke_responses[i].keywords if i < len(ke_responses) else None
            }
            results.append(result)
            
        logger.error(results)

        await self._report_message(
            LogMessage(
                agent_name="PreferenceExtractionMaster",
                action="Completed preference extraction workflow",
                content=json.dumps(results, indent=2),
                id=self.get_id(),
                is_complete=True
            )
        )
        
        logger.info("Completed preference extraction workflow.")
        
        return None
