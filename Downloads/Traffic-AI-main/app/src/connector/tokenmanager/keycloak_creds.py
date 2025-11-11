from typing import TypedDict

class KeycloakCreds(TypedDict):
    client_id: str
    client_secret: str
    base_url: str
    realm: str
