import logging
import uuid

from dataclasses import field
from typing import Any, Dict, List, Tuple, Type
from urllib.parse import urljoin

import aiohttp

from autogen_core.models import AssistantMessage, LLMMessage, UserMessage
from autogen_core.tools import FunctionTool
from src.connector.tokenmanager.keycloak import Keycloak
from src.conversations.enum.role_enum import RoleEnum
from src.conversations.vo.message import Message

logger = logging.getLogger(__name__)

# Assuming these are defined elsewhere:
# class Message, class LLMMessage, class UserMessage, class AssistantMessage, class RoleEnum

class MetaData:
    """
    Represents metadata for a message, including additional information.
    """
    def __init__(self, data_type: str, content: Any, data_description: str):
        self.data_type = data_type
        self.content = content
        self.data_description = data_description
        self.id: str = str(uuid.uuid4())[:8]

def convert_to_llm_message(message: 'Message') -> Tuple['LLMMessage', List[MetaData]]:
    """
    Converts a Message object to an LLMMessage and extracts MetaData.
    """
    role_to_message_cls: Dict[RoleEnum, Type[LLMMessage]] = {
        RoleEnum.USER: UserMessage,
        RoleEnum.ASSISTANT: AssistantMessage,
    }
    msg_cls = role_to_message_cls.get(message.role)
    if not msg_cls:
        raise ValueError(f"Unknown role: {message.role}")

    metadatas: List[MetaData] = []
    remarks: List[str] = []
    content = getattr(message, 'content', '')

    for key, value in message.metadata.items():
        if key not in message.others:
            logger.warning(
                "Key '%s' not found in message.others, skipping metadata extraction.", key
            )
            continue
        value = value.get("data") if isinstance(value, dict) and "data" in value else value
        meta_info = message.others[key]
        meta_data = MetaData(
            data_type=meta_info.get("data_type", ""),
            content=value,
            data_description=meta_info.get("data_description", "")
        )
        metadatas.append(meta_data)
        remarks.append(f" [Memory key of the returned data: {meta_data.id}]")

    if remarks:
        content += ''.join(remarks)

    source = message.others.get("source", "")
    return msg_cls(content=content, source=source), metadatas

class WorkingMemoryService:
    """Service for interacting with agent's working memory.

    Previously this registered a MCP SSE endpoint on the working memory host (port 8003),
    which caused 404 errors because /traffic-copilot/sse lives on the MCP tool service (port 8002).
    We now expose a direct FunctionTool that fetches a memory item via the /memory/get POST endpoint.
    """

    def __init__(self, url: str, keycloak: Keycloak):
        self.url = url.rstrip('/')
        self.keycloak = keycloak

    def get_tools(self) -> List[FunctionTool]:
        async def get_from_memory(
            session_id: str,  # Working memory session / run id
            key: str,         # Resource id stored earlier
        ) -> Dict[str, Any]:
            """Retrieve a memory item (content + meta_data) from the working memory service.

            Returns: { 'content': ..., 'meta_data': ..., 'session_id': ..., 'key': ... }
            or { 'error': 'not_found', ... }
            """
            url = f"{self.url}/memory/get"
            access_token = self.keycloak.get_access_token() if self.keycloak else None
            headers = {"Content-Type": "application/json"}
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
            payload = {"session_id": session_id, "key": key}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, ssl=False) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as e:
                logger.error("get_from_memory failed: %s", e, extra={"url": url, "payload": payload})
                return {"error": str(e), "session_id": session_id, "key": key}

        return [
            FunctionTool(
                get_from_memory,
                name="get_from_memory",
                description="Retrieve an item from working memory by session_id and key."
            )
        ]

    async def set(
        self,
        content: Any,
        session_id: str,
        resource_id: str,
        data_description: Dict
    ) -> None:
        """
        Sets memory for the agent by posting to the memory server.
        """
        url = f"{self.url}/memory/set"
        access_token = self.keycloak.get_access_token() if self.keycloak else None
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
            
        payload = {
            "content": content,
            "key": resource_id,
            "session_id": session_id,
            "meta_data": data_description
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, ssl=False) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    logger.info("Response JSON: %s", resp_json)
        except aiohttp.ClientError as e:
            logger.error(
                "WorkingMemoryService.set failed: %s", str(e),
                exc_info=True,
                extra={
                    "url": url,
                    "payload": payload,
                    "error": str(e)
                }
            )
            raise RuntimeError(
                f"Failed to set memory for agent at {url} with payload {payload}: {e}"
            ) from e


    async def get(
        self,
        session_id: str,
        resource_id: str,
    ) -> dict:
        """
        Gets memory for the agent by querying the memory server (using POST).
        """
        url = f"{self.url}/memory/get"
        access_token = self.keycloak.get_access_token() if self.keycloak else None
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        params = {
            "session_id": session_id,
            "key": resource_id
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=params, ssl=False) as response:
                    response.raise_for_status()
                    resp_json = await response.json()
                    logger.info("Response JSON: %s", resp_json)
                    return resp_json
        except aiohttp.ClientError as e:
            logger.error(
                "WorkingMemoryService.get failed: %s", str(e),
                exc_info=True,
                extra={
                    "url": url,
                    "params": params,
                    "error": str(e)
                }
            )
            raise RuntimeError(
                f"Failed to get memory for agent at {url} with params {params}: {e}"
            ) from e