from typing import TypedDict

from src.configurator.logging_config import LoggingConfig
from src.configurator.sana_gateway_config import SanaGatewayConfig
from src.configurator.llm_gateway_config import LlmGatewayConfig
from src.configurator.mcp_tool_config import McpToolConfig
from src.configurator.mongo_config import MongoConfig
from src.configurator.keycloak_config import KeycloakConfig
from src.configurator.mlflow_config import MLflowConfig
from src.configurator.api_host_config import ApiHostConfig

class Config(TypedDict):
    token_manager: KeycloakConfig
    mongodb: MongoConfig
    mlflow: MLflowConfig
    mcp_tool: McpToolConfig
    llm_gateway: LlmGatewayConfig
    sana_gateway: SanaGatewayConfig
    logging: LoggingConfig
    api_host: ApiHostConfig