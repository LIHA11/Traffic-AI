from enum import Enum
from typing import Union, Optional, Dict, Any

from pydantic import BaseModel

from autogen_core.models import ModelInfo, ModelFamily
from src.connector.tokenmanager.token_manager import TokenManager
from src.copilot.chat_client.chat_client import ChatClient

class Model(Enum):
    GPT_4O = "gpt-4o-241120-deploy-gs" 
    QWQ_32B = "vllm-aliyuncs-1" 
    O3_MINI = "o3-mini-deploy-gs"
    O4_MINI = "o4-mini-deploy-gs"
    GPT_41 = "gpt-4.1-deploy-gs"
    GPT_41_MINI = "gpt-4.1-mini-deploy-gs"
    GPT_41_NANO = "gpt-4.1-nano-deploy-gs"

class ChatClientCreator:
    API_VERSION = "2024-12-01-preview"
    DEFAULTS = {
        "temperature": 0.01,
        "family": ModelFamily.GPT_41, 
        "vision": False,
        "model": Model.GPT_41,
        "fc": True,
    }
    CREATE_API_VERSION_KEY = "api-version"

    def __init__(
        self,
        token_manager: TokenManager,
        endpoint: str,
        api_version: str = API_VERSION
    ):
        self._endpoint = endpoint
        self._api_version = api_version
        self._token_manager = token_manager

    @staticmethod
    def _normalize_model(model: Union[Model, str]) -> str:
        return model.value if isinstance(model, Model) else model

    def _get_default(self, key: str, value: Any) -> Any:
        return value if value is not None else self.DEFAULTS[key]

    def _build_params(
        self,
        model: Union[Model, str] = None,
        response_format: Optional[BaseModel] = None,
        family: ModelFamily = None,
        temperature: float = None,
        vision: bool = None,
        fc: bool = None,
    ) -> Dict[str, Any]:
        model = self._get_default("model", model)
        family = self._get_default("family", family)
        temperature = self._get_default("temperature", temperature)
        vision = self._get_default("vision", vision)
        fc = self._get_default("fc", fc)

        model_value = self._normalize_model(model)
        params = {
            "response_format": response_format,
            "api_version": "",  # Must be empty for compatibility reasons
            "model": model_value,
            "azure_ad_token_provider": self._token_manager.refresh_access_token,
            "azure_deployment": model_value,
            "azure_endpoint": self._endpoint,
            "default_headers": {
                self.CREATE_API_VERSION_KEY: self._api_version
            },
            "model_info": ModelInfo(
                family=family,
                function_calling=fc,
                json_output=False,
                structured_output=bool(response_format),
                vision=vision
            )
        }
        if model_value not in {Model.O3_MINI.value, Model.O4_MINI.value}:
            params['temperature'] = temperature
        return params

    def create(
        self,
        model: Union[Model, str] = None,
        response_format: Optional[BaseModel] = None,
        family: ModelFamily = None,
        temperature: float = None,
        vision: bool = None,
        fc: bool = None
    ) -> ChatClient:
        params = self._build_params(
            model=model,
            response_format=response_format,
            family=family,
            temperature=temperature,
            vision=vision,
            fc=fc
        )
        return ChatClient(**params)
    
    def create_mini(
        self,
        response_format: Optional[BaseModel] = None,
    ) -> ChatClient:
        return self.create(response_format=response_format, model=Model.GPT_41_MINI, family=ModelFamily.GPT_41)
    
    def create_nano(
        self,
        response_format: Optional[BaseModel] = None,
    ) -> ChatClient:
        return self.create(response_format=response_format, model=Model.GPT_41_NANO, family=ModelFamily.GPT_41)
