import requests
from time import sleep
from typing import List, Callable
from src.connector.tokenmanager.token_manager import TokenManager
import logging

logger = logging.getLogger(__name__)

class Keycloak(TokenManager):
    url: str
    issuer: str
    realm: str
    _access_token: str | None
    client_id: str
    client_secret: str

    _REFRESH_NUM_RETRIES = 10
    _REFRESH_RETRY_INTERVAL = 2
    _REFRESH_ACCESS_TOKEN_INTERVAL = 120

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        url: str,
        realm: str,
        callbacks: List[Callable] = [],
    ):
        self.url = url
        self.issuer = f"{url}/auth/realms/{realm}"
        self.jwks_uri = f"{self.issuer}/protocol/openid-connect/certs"
        self._access_token = None
        self.realm = realm
        self.client_id = client_id
        self.client_secret = client_secret
        super().__init__(callbacks, Keycloak._REFRESH_ACCESS_TOKEN_INTERVAL, auto_refresh=True)

    def get_url(self) -> str:
        return self.url

    def get_issuer(self) -> str:
        return self.issuer

    def get_jwks_uri(self) -> str:
        return self.jwks_uri
    
    def get_client_id(self) -> str:
        return self.client_id

    def refresh_access_token(self) -> str:

        url = f"{self.url}/auth/realms/{self.realm}/protocol/openid-connect/token"

        keycloak_body = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        keycloak_headers = {
            "Request-client": "IAM_PORTAL",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        last_exception = None
        for _ in range(Keycloak._REFRESH_NUM_RETRIES):
            try:
                response = requests.post(url, data=keycloak_body, headers=keycloak_headers)
                if response.ok:
                    break
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error: {e}. Retrying...")
            sleep(Keycloak._REFRESH_RETRY_INTERVAL)
        else:
            # All retries failed; raise the last connection error
            if last_exception:
                raise last_exception
            else:
                response.raise_for_status()  # Raise HTTPError if response was never ok

        if response is not None:
            if response.ok:
                self._access_token = response.json()["access_token"]
            else:
                self._access_token = None
        else:
            self._access_token = None

        return self._access_token