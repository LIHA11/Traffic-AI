import config_agent
from typing import Callable, List, TypedDict

from src.common.auth.bearer_auth_provider import BearerAuthProvider
from src.configurator.mlflow_config import MLflowConfig
from src.connector.agentops.mlflow_ops import MLflowOps
from src.connector.agentops.agentops import AgentOps
from src.configurator.configurator import Configurator
from src.configurator.mongo_config import MongoConfig
from src.connector.database.mongo_creds import MongoCreds
from src.connector.database.mongo import MongoDB, PyMongoDB
from src.connector.tokenmanager.keycloak_creds import KeycloakCreds
from src.configurator.keycloak_config import KeycloakConfig
from src.connector.tokenmanager.keycloak import Keycloak
from src.configurator.mcp_tool_config import McpToolConfig
from src.copilot.copilot_v3 import CopilotAgentRuntime, ToolRegistry
from src.copilot.utils.mcp import MCPTool
from src.copilot.utils.agents_config import AgentConfig
from src.copilot.chat_client.chat_client_creator import ChatClientCreator
from src.configurator.config import Config
from src.copilot.utils.message import WorkingMemoryService
from src.copilot.utils.knowledge_center import KnowledgeCenter


class ConnectorType(TypedDict):
    keycloak: Keycloak
    mongodb: MongoDB
    mlflow: MLflowOps
    mcp_tool: ToolRegistry
    copilot: CopilotAgentRuntime
    bearer_auth_provider: BearerAuthProvider
    pymongodb: PyMongoDB


class Connector:
    connectors: ConnectorType | None = None

    def connector_config(connector_name):
        def decorator(function):
            async def wrapper(*args, **kwargs):
                # Add your desired check logic here using check_arg_value
                if not isinstance(args[0], dict):
                    raise Exception("arg 1 must dictionary for configurations")

                if not isinstance(args[1], dict):
                    raise Exception("arg 2 must dictionary for connectors")

                if not isinstance(connector_name, str):
                    raise Exception("decorator input must be the name of the connector")

                app_config: dict = args[0]
                connectors: dict = args[1]
                if connector_name in app_config:
                    connector = await function(
                        *((app_config[connector_name],) + args[1:]), **kwargs
                    )
                    connectors[connector_name] = connector
                    return connectors[connector_name]
                return None

            return wrapper

        return decorator

    @connector_config("keycloak")
    @staticmethod
    async def __initiate_keycloak(
        config: KeycloakConfig, connectors: dict, callbacks: List[Callable] = []
    ) -> Keycloak:
        keycloak_creds: KeycloakCreds = config_agent.get_creds(config["service_name"])

        return Keycloak(
            keycloak_creds["client_id"],
            keycloak_creds["client_secret"],
            keycloak_creds["base_url"],
            keycloak_creds["realm"],
            callbacks,
        )

    @connector_config("mongodb")
    @staticmethod
    async def __initiate_mongodb(config: MongoConfig, connectors: dict) -> MongoDB:
        mongodb_creds: MongoCreds = config_agent.get_creds(config["service_name"])

        return MongoDB(
            user=mongodb_creds["user"],
            password=mongodb_creds["password"],
            hosts=mongodb_creds["hosts"],
            port=mongodb_creds["port"],
            database=mongodb_creds["database"],
            replica_set=(
                mongodb_creds["replica_set"] if "replica_set" in mongodb_creds else None
            ),
            conversations_collection=config["conversations_collection"],
        )

    @staticmethod
    async def __initiate_pymongodb(config: MongoConfig, connectors: dict) -> PyMongoDB:
        mongodb_creds: MongoCreds = config_agent.get_creds(
            "sana-llm-traffic-copilot-mongo-0"
        )

        pymongodb = PyMongoDB(
            user=mongodb_creds["user"],
            password=mongodb_creds["password"],
            hosts=mongodb_creds["hosts"],
            port=mongodb_creds["port"],
            database=mongodb_creds["database"],
            replica_set=(
                mongodb_creds["replica_set"] if "replica_set" in mongodb_creds else None
            ),
            conversations_collection="",
        )
        connectors["pymongodb"] = pymongodb
        return pymongodb

    @connector_config("mlflow")
    @staticmethod
    async def __initiate_mlflow(config: MLflowConfig, connectors: dict) -> AgentOps:
        return MLflowOps(config["tracking_uri"], config["experiment_name"])

    @connector_config("mcp_tool")
    @staticmethod
    async def __initiate_mcp_tool(
        config: McpToolConfig, connectors: dict
    ) -> ToolRegistry:
        tool_registry = ToolRegistry()
        for name, url in config.items():
            tool = MCPTool(name, url, keycloak=Connector.get_keycloak())
            tool_registry.register_tool(tool)

        return tool_registry

    @staticmethod
    async def __initiate_knowledge_center(
        config: Config, connectors: dict
    ) -> KnowledgeCenter:
        knowledge_center = KnowledgeCenter(config["api_host"]["sana_knowledge_center"])

        connectors["knowledge_center"] = knowledge_center
        return knowledge_center

    @staticmethod
    async def __initiate_copilot(
        config: Config, connectors: dict
    ) -> CopilotAgentRuntime:
        runtime = CopilotAgentRuntime(
            chat_client_creator=ChatClientCreator(
                Connector.get_keycloak(), config["llm_gateway"]["host"]
            ),
            copilot_config=AgentConfig.get_agent_config("agent_config"),
            tool_registry=Connector.get_mcp_tool(),
            knowledge_center=Connector.get_knowledge_center(),
            working_memory_service=WorkingMemoryService(
                config["api_host"]["working_memory_service"],
                keycloak=Connector.get_keycloak(),
            ),
            agent_ops=Connector.get_mlflow(),
        )
        await runtime.start()

        connectors["copilot"] = runtime
        return runtime

    @connector_config("mcp_tool")
    @staticmethod
    async def __initiate_mcp_tool(
        config: McpToolConfig, connectors: dict
    ) -> ToolRegistry:
        tool_registry = ToolRegistry()
        for name, url in config.items():
            tool = MCPTool(name, url, keycloak=Connector.get_keycloak())
            tool_registry.register_tool(tool)

        return tool_registry

    @staticmethod
    async def __initiate_bearer_auth_provider(
        config: Config, connectors: dict
    ) -> BearerAuthProvider:
        keycloak = Connector.get_keycloak()
        bearer_auth_provider = BearerAuthProvider(
            issuer=keycloak.get_issuer(),
            jwks_uri=keycloak.get_jwks_uri(),
        )
        connectors["bearer_auth_provider"] = bearer_auth_provider
        return bearer_auth_provider

    @staticmethod
    async def initiate() -> ConnectorType:
        connectors = {}
        Connector.connectors = connectors

        app_config = Configurator.get_config()

        await Connector.__initiate_mlflow(app_config, connectors)
        await Connector.__initiate_keycloak(app_config, connectors)
        await Connector.__initiate_knowledge_center(app_config, connectors)
        await Connector.__initiate_mongodb(app_config, connectors)
        await Connector.__initiate_mcp_tool(app_config, connectors)
        await Connector.__initiate_copilot(app_config, connectors)
        await Connector.__initiate_bearer_auth_provider(app_config, connectors)
        await Connector.__initiate_pymongodb(app_config, connectors)

        return connectors

    @staticmethod
    def get_connectors() -> ConnectorType:
        if not Connector.connectors:
            raise Exception("Connectors not initiated")

        return Connector.connectors

    @staticmethod
    def get_keycloak() -> Keycloak:
        return Connector.get_connectors()["keycloak"]

    @staticmethod
    def get_mongodb() -> MongoDB:
        return Connector.get_connectors()["mongodb"]

    @staticmethod
    def get_pymongodb() -> PyMongoDB:
        return Connector.get_connectors()["pymongodb"]

    @staticmethod
    def get_mlflow() -> MLflowOps:
        return Connector.get_connectors()["mlflow"]

    @staticmethod
    def get_mcp_tool() -> ToolRegistry:
        return Connector.get_connectors()["mcp_tool"]

    @staticmethod
    def get_copilot() -> CopilotAgentRuntime:
        return Connector.get_connectors()["copilot"]

    @staticmethod
    def get_bearer_auth_provider() -> BearerAuthProvider:
        return Connector.get_connectors()["bearer_auth_provider"]

    @staticmethod
    def get_knowledge_center() -> KnowledgeCenter:
        return Connector.get_connectors()["knowledge_center"]