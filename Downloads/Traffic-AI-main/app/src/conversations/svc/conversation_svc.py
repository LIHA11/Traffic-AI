from typing import Any, Dict, List, Optional, Union
import logging
import aiohttp
from autogen_core import CancellationToken

from src.dto.preference_extraction_result_dto import PreferenceExtractionResultDto
from src.dto.knowledge_center_user_preferences_response_dto import (
    KnowledgeCenterUserPreferenceResponseDto,
)
from src.conversations.dto.user_preference_request_dto import UserPreferenceRequestDto
from src.conversations.vo.message_footer import MessageFooter
from src.conversations.dto.preset_message_request_dto import PresetMessageRequestDto
from src.conversations.dto.user_feedback_response_dto import UserFeedbackResponseDto
from src.conversations.dto.user_feedback_dto import UserFeedbackDto
from src.conversations.constant.conversation_constant import GENERIC_ERROR_MESSAGE
from src.conversations.enum.message_type_enum import MessageTypeEnum
from src.conversations.dto.bre_shipment_request_dto import BreShipmentRequestDto
from src.conversations.dto.bre_response_dto import BreResponseDto
from src.conversations.dto.message_request_dto import MessageRequestDto
from src.error.bad_request_error import BadRequestError
from src.error.not_found_error import NotFoundError
from src.error.unauthorized_error import UnauthorizedError
from src.conversations.dto.bre_request_dto import BreRequestDto
from src.configurator.configurator import Configurator
from src.connector.connector import Connector
from src.conversations.vo.conversation import Conversation
from src.conversations.vo.message import Message
from src.conversations.vo.user_feedback import UserFeedback
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.document.conversation import Conversation as ConversationDocument
from src.conversations.document.message import Message as MessageDocument
from src.copilot.copilot_v3 import CopilotAgentRuntime, WorkflowTask
from werkzeug.datastructures import ImmutableMultiDict
from asyncio import Queue
import json

logger = logging.getLogger(__name__)



def create_conversation(user_id: str) -> Conversation:
    conversation_document = ConversationDocument(creator=user_id, messages=[])
    conversation_document.save()

    return Conversation.from_document(conversation_document)


def fetch_conversations_by_user_id(user_id: str) -> List[Conversation]:
    documents = ConversationDocument.objects(
        creator=user_id, is_deleted=False
    ).order_by("-lastModifiedDateTimeUtc")

    if len(documents) == 0:
        new_conversation = create_conversation(user_id)
        return [new_conversation]

    return [Conversation.from_document(document) for document in documents]


def fetch_conversation_by_id(conversation_id: str) -> Conversation:
    document = ConversationDocument.objects.get(id=conversation_id, is_deleted=False)

    return Conversation.from_document(document)


def get_conversation_by_message_id(message_id: str) -> Conversation:
    message_document: MessageDocument = MessageDocument.objects.get(id=message_id)
    conversation_document: ConversationDocument = message_document.conversation

    if not conversation_document:
        return None
    return Conversation.from_document(conversation_document)


def get_message_groups(messages: List[Message]) -> Dict[str, str]:
    map = {}
    for message in messages:
        map[message.id] = message.group_id

    return map


def update_messages_group_id(user_id: str, message_ids: list[str], group_id: str):
    message_documents = [
        get_message_document_by_id(user_id, message_id) for message_id in message_ids
    ]
    for message_document in message_documents:
        message_document.group_id = group_id
        message_document.save()


async def create_message_stream(
    user_id: str, conversation_id: str, message: MessageRequestDto
) -> Union[Queue, str, CancellationToken]:
    add_message_to_conversation(
        user_id, conversation_id, Message.from_content(message.content)
    )
    conversation_document = get_conversation_document_by_id(user_id, conversation_id)
    messages = [
        Message.from_document(message_document)
        for message_document in conversation_document.messages
        if message_document.type == MessageTypeEnum.CONVERSATION
    ]

    cancellation_token = CancellationToken()

    copilot: CopilotAgentRuntime = Connector.get_copilot()

    queue, request_id = await copilot.create(
        messages=messages,
        user_id=user_id,
        conversation_id=conversation_id,
        cancellation_token=cancellation_token,
    )

    return queue, request_id, cancellation_token


async def create_message(
    user_id: str, conversation_id: str, message: MessageRequestDto
) -> Message | None:
    try:
        add_message_to_conversation(
            user_id, conversation_id, Message.from_content(message.content)
        )
        conversation_document = get_conversation_document_by_id(
            user_id, conversation_id
        )
        messages = [
            Message.from_document(message_document)
            for message_document in conversation_document.messages
            if message_document.type == MessageTypeEnum.CONVERSATION
        ]

        copilot: CopilotAgentRuntime = Connector.get_copilot()

        system_message, _ = await copilot.create_until_finish(
            messages=messages, user_id=user_id, conversation_id=conversation_id
        )

        system_message_document = add_message_to_conversation(
            user_id, conversation_id, system_message
        )

        return Message.from_document(system_message_document)
    except Exception as e:
        logger.error(e, exc_info=True)

        system_message = create_system_message(
            user_id, conversation_id, GENERIC_ERROR_MESSAGE, type=MessageTypeEnum.ERROR
        )

        return system_message


async def create_preset_message(
    user_id: str, conversation_id: str, preset_message_request: PresetMessageRequestDto
) -> Message | None:
    try:
        add_message_to_conversation(
            user_id,
            conversation_id,
            Message.from_content(preset_message_request.request_content),
        )
        system_message = create_system_message(
            user_id,
            conversation_id,
            preset_message_request.preset_system_content,
            footer=(
                MessageFooter.from_dto(preset_message_request.preset_system_footer)
                if preset_message_request.preset_system_footer
                else None
            ),
        )
        return system_message

    except Exception as e:
        logger.error(e, exc_info=True)

        system_message = create_system_message(
            user_id, conversation_id, GENERIC_ERROR_MESSAGE, type=MessageTypeEnum.ERROR
        )

        return system_message


def create_system_message(
    user_id: str,
    conversation_id: str,
    content: str,
    type: MessageTypeEnum = MessageTypeEnum.CONVERSATION,
    footer: Optional[MessageFooter] = None,
    metadata: Optional[Dict[str, Any]] = None,
    others: Optional[Dict[str, Any]] = None,
) -> Message:
    message = Message(
        role=RoleEnum.ASSISTANT,
        type=type,
        content=content,
        footer=footer,
        metadata=metadata,
        others=others,
    )
    message_document = add_message_to_conversation(user_id, conversation_id, message)
    return Message.from_document(message_document)


def soft_delete_conversation(user_id: str, conversation_id: str) -> None:
    conversation_document = get_conversation_document_by_id(
        user_id, conversation_id, check_ownership=False
    )
    conversation_document.is_deleted = True
    conversation_document.save()


def delete_conversation(user_id: str, conversation_id: str) -> None:
    conversation_document = get_conversation_document_by_id(
        user_id, conversation_id, check_ownership=False
    )
    conversation_document.delete()


async def request_bre(
    user_id: str,
    user_cookies: ImmutableMultiDict[str, str],
    conversation_id: str,
    bre_request: BreShipmentRequestDto,
) -> BreResponseDto:
    if bre_request is None or len(bre_request.shipments) == 0:
        raise BadRequestError("No shipments provided")

    shipments = bre_request.shipments

    check_conversation_ownership(user_id, conversation_id)

    request_dtos: List[BreRequestDto] = []

    for shipment in shipments:
        request_dtos.append(
            BreRequestDto(
                shipment_numbers=[shipment.shipment_number],
                original_leg_load_stop_id=shipment.orig_next_seg_load_id,
                original_leg_dsch_stop_id=shipment.orig_next_seg_dsch_id,
                new_leg_load_stop_id=shipment.tgt_next_seg_load_id,
                new_leg_dsch_stop_id=shipment.tgt_next_seg_dsch_id,
            )
        )

    logger.info("POST to sana_realtime_cms /bre/external with")
    logger.info(json.dumps([dto.to_dict() for dto in request_dtos], indent=2))

    request_id = None
    message: Message | None = None
    try:
        async with aiohttp.ClientSession(cookies=user_cookies) as session:
            async with session.post(
                Configurator.get_config()["api_host"]["sana_realtime_cms"]
                + "/bre/external",
                json=[dto.to_dict() for dto in request_dtos],
            ) as response:
                response.raise_for_status()
                request_id = await response.text()

        logger.info(f"BRE request successful with ID: {request_id}")
        message = Message(
            role=RoleEnum.ASSISTANT,
            content=f"BRE request for {', '.join([shipment.shipment_number for shipment in shipments])} successful. Request ID: {request_id}",
        )
    except Exception as e:
        if response.status == 400:
            logger.error(f"Bad request to BRE for {user_id}: {response.text}")
            message = Message(
                role=RoleEnum.ASSISTANT,
                type=MessageTypeEnum.ERROR,
                content="Invalid request to BRE. Please check the shipment details.",
            )
        elif response.status == 401:
            logger.error(f"Unauthorized request to BRE for {user_id}")
            message = Message(
                role=RoleEnum.ASSISTANT,
                type=MessageTypeEnum.ERROR,
                content="Unauthorized request to BRE. Please check if you have the permission to access TOP.",
            )
        else:
            logger.error(f"Error requesting BRE for {user_id}: {e}")
            message = Message(
                role=RoleEnum.ASSISTANT,
                type=MessageTypeEnum.ERROR,
                content="An error occurred while processing your request. Please try again later.",
            )

    if message:
        message_document = add_message_to_conversation(
            user_id, conversation_id, message
        )
        message = Message.from_document(message_document)

    return BreResponseDto(id=request_id, message=message)


def check_conversation_ownership(user_id: str, conversation_id: str):
    conversation_document = get_conversation_document_by_id(user_id, conversation_id)

    if conversation_document.creator != user_id:
        raise UnauthorizedError()


def get_conversation_document_by_id(
    user_id: str, conversation_id: str, check_ownership: bool = True
) -> ConversationDocument:
    conversation_document: ConversationDocument = ConversationDocument.objects.get(
        id=conversation_id
    )

    if not conversation_document:
        raise NotFoundError("Conversation not found")

    if check_ownership and conversation_document.creator != user_id:
        raise UnauthorizedError()

    return conversation_document


def get_message_document_by_id(user_id: str, message_id: str) -> MessageDocument:
    message_document: MessageDocument = MessageDocument.objects.get(id=message_id)

    if not message_document:
        raise NotFoundError("Message not found")

    #if message_document.conversation.creator != user_id:
    #    raise UnauthorizedError()

    return message_document


def add_message_to_conversation(
    user_id: str, conversation_id: str, message: Message
) -> MessageDocument:
    conversation_document = get_conversation_document_by_id(user_id, conversation_id)
    message_document = message.to_document()
    message_document.conversation = conversation_document
    message_document.save()
    conversation_document.messages.append(message_document)
    conversation_document.save()
    return conversation_document.messages[-1]


def update_feedback(user_id: str, message_id: str, feedback: UserFeedbackDto) -> bool:
    message_document = get_message_document_by_id(user_id, message_id)
    feedback_document = message_document.userFeedback

    is_created = bool(feedback_document)

    if not is_created:
        feedback_document = UserFeedback.from_dto(feedback).to_document()
        feedback_document.message = message_document
        feedback_document.save()

        message_document.userFeedback = feedback_document

        message_document.save()
    else:
        feedback_document.liked = feedback.liked
        feedback_document.feedback = feedback.feedback
        feedback_document.save()

    return is_created

async def get_user_preferences(
    user_id: str, conversation_id: str, requestDto: UserPreferenceRequestDto
) -> KnowledgeCenterUserPreferenceResponseDto:
    check_conversation_ownership(user_id, conversation_id)
    knowledge_center = Connector.get_knowledge_center()
    preferences = await knowledge_center.get_user_preferences(user_id=user_id, query_text=requestDto.content)
    return preferences

async def create_user_preferences(
    user_id: str, message_id: Optional[str] = None, first_n_conversations: int = 10
) -> List[PreferenceExtractionResultDto] | None:
    message_document = None

    if message_id:
        message_document = get_message_document_by_id(user_id, message_id)
        message = Message.from_document(message_document)
        message_id = message.id

    copilot = Connector.get_copilot()

    content, _duration = await copilot.create_workflow_task_until_finish(
        task=WorkflowTask.PREFERENCE_EXTRACTION,
        user_id=user_id,
        inputs={
            "user_id": user_id,
            "message_id": message_id,
            "first_n_conversations": first_n_conversations,
        },
    )

    try:
        content_obj = json.loads(content)
        if message_document is not None:
            feedback_document = message_document.userFeedback
            feedback_document.summary = content_obj[0]["summary"]
            feedback_document.save()

        if content_obj and isinstance(content_obj, list):
            dtos = [
                PreferenceExtractionResultDto.from_dict(item) for item in content_obj
            ]
        return dtos
    except Exception as e:
        return None


async def get_user_preferences(
    user_id: str, conversation_id: str, requestDto: UserPreferenceRequestDto
) -> KnowledgeCenterUserPreferenceResponseDto:
    check_conversation_ownership(user_id, conversation_id)
    knowledge_center = Connector.get_knowledge_center()
    preferences = await knowledge_center.get_user_preferences(
        user_id=user_id, query_text=requestDto.content, top_n=requestDto.top_n, threshold=requestDto.threshold
    )
    return preferences

def fetch_feedback(message_id: str) -> UserFeedbackResponseDto | None:
    message_document: MessageDocument = MessageDocument.objects.get(id=message_id)
    conversation_document: ConversationDocument = message_document.conversation

    if not message_document:
        return None

    message: Message = Message.from_document(message_document)
    conversation: Conversation = Conversation.from_document(conversation_document)

    return UserFeedbackResponseDto.from_message_and_conversation(message, conversation)


def fetch_feedback_by_user_id(user_id: Optional[str]) -> List[UserFeedbackResponseDto]:
    conversation_documents = []
    if user_id:
        conversation_documents = ConversationDocument.objects(creator=user_id)
    else:
        conversation_documents = ConversationDocument.objects()

    conversations = [
        Conversation.from_document(conversation_document)
        for conversation_document in conversation_documents
    ]

    responses: List[UserFeedbackResponseDto] = []

    for conversation in conversations:
        for message in conversation.messages:
            if message.userFeedback:
                responses.append(
                    UserFeedbackResponseDto.from_message_and_conversation(
                        message, conversation
                    )
                )

    return responses

def fetch_similar_messages_by_user_id_and_message(user_id: str, message: str) -> List[str]:
    results =  ConversationDocument.objects(creator = user_id).aggregate([
        {
            '$lookup': {
                'from': 'messages',
                'localField': 'messages',
                'foreignField': '_id',
                'as': 'messages'
            }
        },
        {
            '$project': {
                'messages': 1
            }
        },
        {
            '$unwind': {
                'path': '$messages'
            }
        },
        {
            '$match': {
                'messages.role': 'user',
                'messages.content': {
                    '$regex': f'.*{message}.*',
                    '$options': 'i'
                }
            }
        },
        {
            '$project': {
                '_id': 0,
                'similarContent': '$messages.content'
            }
        }
    ])

    return [result['similarContent'] for result in results]
