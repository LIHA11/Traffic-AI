from typing import Any, Dict, List
from autogen_core import CancellationToken
from quart import jsonify, make_response, request, g, stream_with_context
from quart_schema import validate_request, validate_response
from src.error.internal_server_error import InternalServerError
from src.dto.preference_extraction_result_dto import PreferenceExtractionResultDto
from src.conversations.dto.user_preference_request_dto import UserPreferenceRequestDto
from src.copilot.copilot_v3 import CopilotAgentRuntime
from src.conversations.dto.preset_message_request_dto import PresetMessageRequestDto
from src.conversations.dto.user_feedback_response_dto import UserFeedbackResponseDto
from src.error.not_found_error import NotFoundError
from src.conversations.dto.user_feedback_dto import UserFeedbackDto
from src.conversations.enum.message_metadata_key_enum import MessageMetadataKeyEnum
from src.conversations.enum.message_type_enum import MessageTypeEnum
from src.connector.connector import Connector
from src.connector.agentops.agentops import LogMessage
from src.conversations.dto.agent_progress_dto import AgentProgressDto
from src.conversations.dto.bre_shipment_request_dto import BreShipmentRequestDto
from src.conversations.constant.conversation_constant import (
    ABORT_ERROR_MESSAGE,
    MESSAGE_EVENT_KEEP_ALIVE_INTERVAL,
    GENERIC_ERROR_MESSAGE,
    MESSAGE_EVENT_TIMEOUT,
    TIMEOUT_ERROR_MESSAGE,
)
from src.conversations.dto.bre_response_dto import BreResponseDto
from src.conversations.dto.message_request_dto import MessageRequestDto
from src.conversations.enum.message_event_enum import MessageEventEnum
from src.conversations.event.message_sse_event import MessageSSEEvent
from src.error.bad_request_error import BadRequestError
from src.common.dataclass.access_token import AccessToken
from src.error.unauthorized_error import UnauthorizedError
import src.conversations.svc.conversation_svc as conversation_svc
from src.conversations.vo.message import Message
import logging
import asyncio
import json
from quart import Quart
from mlflow.entities import RunStatus

logger = logging.getLogger(__name__)


def init_conversations_routes(app: Quart):
    @app.get("/conversations/<conversation_id>")
    def fetch_conversation(conversation_id: str):
        access_token: AccessToken = g.access_token
        user_id: str = g.user_id

        logger.info(f"Getting conversation {conversation_id}")
        conversation = conversation_svc.fetch_conversation_by_id(conversation_id)

        if not conversation:
            raise NotFoundError()

        if conversation.creator != user_id and not access_token.is_in_scope(
            "sana.simulator.support"
        ):
            raise UnauthorizedError()

        return jsonify(conversation.to_dict()), 200

    @app.get("/conversations")
    def fetch_conversations():
        user_id: str = g.user_id

        logger.info(f"Getting conversation for user {user_id}")
        conversations = conversation_svc.fetch_conversations_by_user_id(user_id)
        return jsonify([conversation.to_dict() for conversation in conversations]), 200

    @app.post("/conversations")
    def create_conversation():
        user_id: str = g.user_id

        logger.info(f"Creating conversation for user {user_id}")
        conversation = conversation_svc.create_conversation(user_id)
        return jsonify(conversation.to_dict()), 201

    @validate_request(MessageRequestDto)
    @app.post("/conversations/<conversation_id>/message")
    async def create_message(conversation_id: str):
        if "text/event-stream" not in request.accept_mimetypes:
            raise BadRequestError(
                "Invalid request. Expected 'text/event-stream' Mime type."
            )

        user_id: str = g.user_id

        logger.info(
            f"Creating message for conversation {conversation_id} for user {user_id}"
        )

        @stream_with_context
        async def send_events():
            queue_id: str | None = None
            cancellation_token: CancellationToken | None = None
            agent_progresses: List[AgentProgressDto] = []
            is_agent_progress_message_created = False
            is_error = False
            is_aborted = False
            is_timed_out = False

            start_event = MessageSSEEvent(event=MessageEventEnum.START_STREAM)
            yield start_event.encode()

            messageDto: MessageRequestDto = MessageRequestDto(
                **await request.get_json()
            )

            # Create a queue to collect events from both keep-alive and message creation
            event_queue = asyncio.Queue()
            ended = False

            async def timeout():
                await asyncio.sleep(MESSAGE_EVENT_TIMEOUT)
                system_message = conversation_svc.create_system_message(
                    user_id,
                    conversation_id,
                    TIMEOUT_ERROR_MESSAGE,
                    type=MessageTypeEnum.ERROR,
                )

                event = MessageSSEEvent(
                    data=json.dumps(system_message.to_dict()),
                    event=MessageEventEnum.MESSAGE,
                )
                
                nonlocal is_timed_out
                is_timed_out = True
                
                await event_queue.put(event)
                await event_queue.put(None)

            async def send_keep_alive():
                nonlocal ended
                while not ended:
                    await asyncio.sleep(MESSAGE_EVENT_KEEP_ALIVE_INTERVAL)
                    keep_alive_event = MessageSSEEvent(
                        event=MessageEventEnum.KEEP_ALIVE
                    )
                    await event_queue.put(keep_alive_event)

            async def create_message_task():
                nonlocal queue_id
                nonlocal cancellation_token
                nonlocal agent_progresses
                nonlocal is_agent_progress_message_created
                nonlocal ended

                queue, request_id, cancellation_token = (
                    await conversation_svc.create_message_stream(
                        user_id, conversation_id, messageDto
                    )
                )
                queue_id = request_id

                while not ended:
                    log_message: LogMessage = await queue.get()
                    agent_progress_dto = AgentProgressDto.from_log_message(log_message)

                    agent_progresses.append(agent_progress_dto)
                    event = MessageSSEEvent(
                        data=json.dumps(agent_progress_dto.to_dict()),
                        event=MessageEventEnum.AGENT_PROGRESS,
                    )
                    await event_queue.put(event)

                    if agent_progress_dto.is_completed:
                        reference = (
                            agent_progress_dto.references[0]
                            if agent_progress_dto.references and len(agent_progress_dto.references) > 0
                            else None
                        )

                        conversation_svc.create_system_message(
                            user_id,
                            conversation_id,
                            content="",
                            type=MessageTypeEnum.AGENT_PROGRESS,
                            metadata={
                                MessageMetadataKeyEnum.AGENT_PROGRESS: [
                                    agent_progress.to_dict()
                                    for agent_progress in agent_progresses
                                ]
                            },
                        )
                        is_agent_progress_message_created = True

                        metadata: Dict[str, Any] = {}
                        others: Dict[str, Any] = {}

                        if reference:
                            content = reference.get("content")
                            headers = reference.get("headers")
                            if content and len(content) > 0:
                                if "tsPort" in content[0]:
                                    metadata = {MessageMetadataKeyEnum.SHIPMENT: content}
                                    others = {MessageMetadataKeyEnum.SHIPMENT: reference.get("meta_data", {})}
                                elif headers:
                                    metadata = {
                                        MessageMetadataKeyEnum.TABLE: {
                                            "headers": headers,
                                            "data": content,
                                        }
                                    }
                                    others = {MessageMetadataKeyEnum.TABLE: reference.get("meta_data", {})}

                        system_message = conversation_svc.create_system_message(
                            user_id,
                            conversation_id,
                            agent_progress_dto.content,
                            metadata=metadata,
                            others=others,
                        )

                        event = MessageSSEEvent(
                            data=json.dumps(system_message.to_dict()),
                            event=MessageEventEnum.MESSAGE,
                        )
                        await event_queue.put(event)
                        await event_queue.put(None)

            keep_alive_task = asyncio.create_task(send_keep_alive())
            message_task = asyncio.create_task(create_message_task())
            timeout_task = asyncio.create_task(timeout())

            # Process events from queue
            while not ended:
                try:
                    event = await event_queue.get()
                    if event is None:  # Ended
                        ended = True
                        is_error = False
                        is_aborted = False
                        break
                    else:  # Regular event (keep-alive or message)
                        yield event.encode()
                except asyncio.CancelledError:
                    logger.info(
                        f"Request for conversation {conversation_id} was cancelled by the client."
                    )
                    ended = True
                    is_error = True
                    is_aborted = True
                except Exception as e:
                    logger.error(
                        f"Error while sending events for conversation {conversation_id}: {e}",
                        exc_info=True,
                    )
                    ended = True
                    is_error = True
                    is_aborted = False

            if not is_agent_progress_message_created and len(agent_progresses) > 0:
                logger.info(
                    f"Saving intermediate agent progress message for conversation {conversation_id} before cancellation."
                )
                conversation_svc.create_system_message(
                    user_id,
                    conversation_id,
                    content="",
                    type=MessageTypeEnum.AGENT_PROGRESS,
                    metadata={
                        MessageMetadataKeyEnum.AGENT_PROGRESS: [
                            agent_progress.to_dict()
                            for agent_progress in agent_progresses
                        ]
                    },
                )

            error_message: str | None = None
            if is_error and is_aborted:
                error_message = ABORT_ERROR_MESSAGE
            elif is_error:
                error_message = GENERIC_ERROR_MESSAGE

            if error_message:
                system_message = conversation_svc.create_system_message(
                    user_id,
                    conversation_id,
                    error_message,
                    type=MessageTypeEnum.ERROR,
                )

                event = MessageSSEEvent(
                    data=json.dumps(system_message.to_dict()),
                    event=MessageEventEnum.MESSAGE,
                )
                yield event.encode()

            if cancellation_token:
                cancellation_token.cancel()

            # Clean up tasks
            keep_alive_task.cancel()
            message_task.cancel()
            timeout_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass
            try:
                await message_task
            except asyncio.CancelledError:
                pass
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass

            copilot = Connector.get_copilot()
            if queue_id is not None and copilot is not None:
                await copilot.release_queue(queue_id)
                logger.info(
                    f"Released queue {queue_id} for conversation {conversation_id}"
                )

            agent_ops = Connector.get_mlflow()
            if agent_ops is not None and queue_id is not None:
                status = (
                    RunStatus.KILLED
                    if is_aborted
                    else (RunStatus.FAILED if is_error or is_timed_out else RunStatus.FINISHED)
                )
                agent_ops.end_run(queue_id, status=status)

            end_event = MessageSSEEvent(event=MessageEventEnum.END_STREAM)
            yield end_event.encode()

        response = await make_response(
            send_events(),
            {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
                "X-Accel-Buffering": "no",
            },
        )
        response.timeout = None
        return response

    @validate_request(MessageRequestDto)
    @app.post("/conversations/<conversation_id>/message/rest")
    async def create_message_rest(conversation_id: str):
        user_id: str = g.user_id

        logger.info(
            f"Creating message for conversation {conversation_id} for user {user_id}"
        )
        try:
            messageDto: MessageRequestDto = MessageRequestDto(
                **await request.get_json()
            )
            system_message = await conversation_svc.create_message(
                user_id, conversation_id, messageDto
            )
        except asyncio.CancelledError as error:
            logger.error(
                f"Request aborted for conversation {conversation_id} for user {user_id}: {error}",
                exc_info=True,
            )

            system_message = conversation_svc.create_system_message(
                user_id,
                conversation_id,
                ABORT_ERROR_MESSAGE,
                type=MessageTypeEnum.ERROR,
            )

        return jsonify(system_message.to_dict()), 201

    @validate_request(PresetMessageRequestDto)
    @app.post("/conversations/<conversation_id>/message/preset")
    async def create_preset_system_message(conversation_id: str):
        user_id: str = g.user_id

        logger.info(
            f"Creating preset messages for conversation {conversation_id} for user {user_id}"
        )

        system_message: Message | None = None
        try:
            preset_message_request_dto: PresetMessageRequestDto = (
                PresetMessageRequestDto.from_dict(await request.get_json())
            )
            system_message = await conversation_svc.create_preset_message(
                user_id, conversation_id, preset_message_request_dto
            )
        except asyncio.CancelledError as error:
            logger.error(
                f"Request aborted for conversation {conversation_id} for user {user_id}: {error}",
                exc_info=True,
            )

            system_message = conversation_svc.create_system_message(
                user_id,
                conversation_id,
                ABORT_ERROR_MESSAGE,
                type=MessageTypeEnum.ERROR,
            )

        return jsonify(system_message.to_dict()), 201

    @app.delete("/conversations/<conversation_id>")
    async def delete_conversation(conversation_id: str):
        access_token: AccessToken = g.access_token
        user_id: str = g.user_id
        hard = request.args.get("hard", "false").lower() == "true"

        logger.info(
            f"{'Hard' if hard else 'Soft'} deleting conversation {conversation_id} for user {user_id}"
        )

        conversation = conversation_svc.get_conversation_document_by_id(
            user_id, conversation_id, check_ownership=False
        )
        if hard:
            # Only support can hard delete conversations
            if not access_token.is_in_scope("sana.simulator.support"):
                raise UnauthorizedError(
                    "You are not allowed to hard delete this conversation."
                )
            conversation_svc.delete_conversation(user_id, conversation_id)
        else:
            if conversation.creator != user_id and not access_token.is_in_scope(
                "sana.simulator.support"
            ):
                raise UnauthorizedError(
                    "You are not allowed to soft delete this conversation."
                )
            conversation_svc.soft_delete_conversation(user_id, conversation_id)
        return "", 204

    @validate_request(BreShipmentRequestDto)
    @validate_response(BreResponseDto)
    @app.post("/conversations/<conversation_id>/bre")
    async def request_bre(conversation_id: str):
        user_id: str = g.user_id

        logger.info(f"Requesting BRE for conversation {conversation_id}")
        bre_request = BreShipmentRequestDto.from_dict(await request.get_json())
        breResponseDto = await conversation_svc.request_bre(
            user_id, request.cookies, conversation_id, bre_request
        )
        return jsonify(breResponseDto.to_dict()), 200

    @validate_request(UserFeedbackDto)
    @app.put("/messages/<message_id>/feedback")
    async def update_feedback(message_id: str):
        user_id: str = g.user_id

        logger.info(f"Updating feedback for message {message_id} for user {user_id}")
        feedback = UserFeedbackDto.from_dict(await request.get_json())

        is_created = conversation_svc.update_feedback(user_id, message_id, feedback)
        if is_created:
            return "", 204
        else:
            return "", 201

    @validate_response(UserFeedbackResponseDto)
    @app.get("/messages/<message_id>/feedback")
    def fetch_feedback(message_id: str):
        access_token: AccessToken = g.access_token
        user_id: str = g.user_id

        logger.info(f"Getting feedback for message {message_id}")
        feedback = conversation_svc.fetch_feedback(message_id)

        if not feedback:
            raise NotFoundError()

        if feedback.creator != user_id and not access_token.is_in_scope(
            "sana.simulator.support"
        ):
            raise UnauthorizedError()

        return jsonify(feedback.to_dict()), 200

    @validate_response([UserFeedbackResponseDto])
    @app.get("/messages/feedback")
    def fetch_messages_with_feedback():
        access_token: AccessToken = g.access_token
        current_user_id = access_token.get_user_id()
        if not current_user_id:
            raise UnauthorizedError()

        user_id = request.args.get("userId")

        if not access_token.is_in_scope("sana.simulator.support"):
            raise UnauthorizedError()

        if not user_id:
            logger.info(f"Getting feedbacks for all users")
        else:
            logger.info(f"Getting feedbacks for user {user_id}")

        feedback_responses = conversation_svc.fetch_feedback_by_user_id(user_id)

        return jsonify([feedback.to_dict() for feedback in feedback_responses]), 200

    @validate_response(List[PreferenceExtractionResultDto])
    @app.post("/messages/<message_id>/preferences")
    async def create_user_preferences(message_id: str):
        user_id: str = g.user_id

        dtos = await conversation_svc.create_user_preferences(
            user_id=user_id,
            message_id=message_id,
        )

        if not dtos:
            raise InternalServerError("Failed to create user preferences")

        return jsonify([dto.to_dict() for dto in dtos]), 200

    @validate_response(List[PreferenceExtractionResultDto])
    @app.post("/preferences")
    async def create_full_user_preferences():
        user_id: str = g.user_id
        access_token: AccessToken = g.access_token

        request_user_id = request.args.get("userId", user_id)
        first_n_conversations = int(request.args.get("firstNConversations", "10"))

        if request_user_id != user_id and not access_token.is_in_scope(
            "sana.simulator.support"
        ):
            raise UnauthorizedError()

        dtos = await conversation_svc.create_user_preferences(
            user_id=user_id, first_n_conversations=first_n_conversations
        )

        if not dtos:
            raise InternalServerError("Failed to create user preferences")

        return jsonify([dto.to_dict() for dto in dtos]), 200

    @validate_response([str])
    @app.get("/conversations/<conversation_id>/preferences")
    async def get_user_preferences(conversation_id: str):
        user_id: str = g.user_id

        content = request.args.get("content")
        top_n = int(request.args.get("topN", "10"))
        threshold = float(request.args.get("threshold", "0.1"))
        
        if not content or top_n <= 0:
            raise BadRequestError()

        logger.info(
            f"Getting user preferences for conversation {conversation_id} for user {user_id}"
        )
        requestDto: UserPreferenceRequestDto = UserPreferenceRequestDto(
            content=content, top_n=top_n, threshold=threshold
        )
        preference = await conversation_svc.get_user_preferences(
            user_id=user_id, conversation_id=conversation_id, requestDto=requestDto
        )
        if (not preference) or (preference.error is not None):
            raise NotFoundError(preference.error)

        return jsonify(preference.results), 200
    
    @app.get("/messages")
    async def fetch_messages() -> List[str]:
        user_id: str = g.user_id
        message = request.args.get('message')

        return conversation_svc.fetch_similar_messages_by_user_id_and_message(
            user_id, message
        )
