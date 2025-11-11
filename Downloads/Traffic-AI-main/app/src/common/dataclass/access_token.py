from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AccessToken:
    exp: Optional[int]
    iat: Optional[int]
    auth_time: Optional[int]
    jti: Optional[str]
    iss: Optional[str]
    sub: Optional[str]
    typ: Optional[str]
    azp: Optional[str]
    session_state: Optional[str]
    allowed_origins: Optional[List[str]]
    scopes: List[str]
    sid: Optional[str]
    tenant_id: Optional[str]
    aud: Optional[List[str]]
    upn: Optional[str]
    email_verified: Optional[bool]
    name: Optional[str]
    preferred_username: Optional[str]
    locale: Optional[str]
    given_name: Optional[str]
    family_name: Optional[str]
    email: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "AccessToken":
        """Create AccessToken instance from dictionary"""
        return cls(
            exp=data.get("exp"),
            iat=data.get("iat"),
            auth_time=data.get("auth_time"),
            jti=data.get("jti"),
            iss=data.get("iss"),
            sub=data.get("sub"),
            typ=data.get("typ"),
            azp=data.get("azp"),
            session_state=data.get("session_state"),
            allowed_origins=data.get("allowed-origins"),
            scopes=(
                data.get("scope", []).split()
                if isinstance(data.get("scope"), str)
                else data.get("scope", [])
            ),
            sid=data.get("sid"),
            tenant_id=data.get("tenant_id"),
            aud=data.get("aud"),
            upn=data.get("upn"),
            email_verified=data.get("email_verified"),
            name=data.get("name"),
            preferred_username=data.get("preferred_username"),
            locale=data.get("locale"),
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            email=data.get("email"),
        )

    def is_valid(self) -> bool:
        return self.get_user_id() is not None

    def get_user_id(self) -> Optional[str]:
        if self.upn:
            return self.upn.split("@")[0]
        return None

    def get_scopes(self) -> Optional[str]:
        if self.scopes:
            return self.scopes
        return None

    def is_in_scope(self, scope: str) -> bool:
        return scope in self.scopes if self.scopes else False
