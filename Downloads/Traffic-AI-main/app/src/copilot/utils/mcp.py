import asyncio
import json
import logging
from typing import Optional

from autogen_core.tools import FunctionTool
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams

from src.connector.tokenmanager.keycloak import Keycloak

logger = logging.getLogger(__name__)

class MCPTool(FunctionTool):
    def __init__(self, name: str, url: str, keycloak: Keycloak):
        super().__init__(self.get, f"Retrieve the MCP tool description for {name} from {url}.", name)
        self.url = url
        self.keycloak = keycloak

    async def get(
        self,
        session_id: Optional[str] = None,
        retries: int = 3,
        delay: float = 2.0
    ) -> SseMcpToolAdapter:
        """
        Connects to the MCP tool using the provided URL and tool name.

        Args:
            session_id (Optional[str]): Optional session ID for authentication.
            retries (int): Number of retry attempts.
            delay (float): Delay in seconds between retries.

        Returns:
            SseMcpToolAdapter: An instance connected to the MCP tool.
        """
        access_token = self.keycloak.get_access_token() if self.keycloak else None
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["session_id"] = session_id
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        server_params = SseServerParams(
            url=self.url,
            headers=headers,
        )

        return await retry_from_server_params(server_params, self.name, retries, delay)

def parse_mcp_return(result: str) -> str:
    """
    Parses the MCP return JSON string and extracts the 'text' field from the first element.
    If parsing fails or the structure is unexpected, returns the original input.
    """
    try:
        return json.loads(result)[0]['text']
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return result

async def retry_from_server_params(
    server_params: SseServerParams,
    tool_name: str,
    retries: int = 3,
    delay: float = 2.0
) -> SseMcpToolAdapter:
    """
    Attempts to create an SseMcpToolAdapter from server_params, with retries.

    Args:
        server_params (SseServerParams): The server parameters.
        tool_name (str): The tool's name.
        retries (int): Number of retry attempts.
        delay (float): Delay in seconds between retries.

    Returns:
        SseMcpToolAdapter: An instance connected to the MCP tool.

    Raises:
        Exception: If all retries fail.
    """
    for attempt in range(1, retries + 1):
        try:
            return await SseMcpToolAdapter.from_server_params(server_params, tool_name)
        except Exception as e:
            exception = ''
            if (isinstance(e, ExceptionGroup)):
                exception = [str(ex) for ex in e.exceptions]
            else:
                exception = str(e)
            logger.error("Attempt %d/%d failed: %s", attempt, retries, exception)
            if attempt == retries:
                logger.error("All %d attempts failed.", retries)
                raise e
            await asyncio.sleep(delay)
