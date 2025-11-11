import logging
import time

import httpx
from pydantic import AnyHttpUrl, ValidationError
from authlib.jose import JsonWebKey, JsonWebToken
from authlib.jose.errors import JoseError

from src.common.dataclass.access_token import AccessToken


class BearerAuthProvider:
    """
    Simple JWT Bearer Token validator.
    Uses RS256 asymmetric encryption by default but supports all JWA algorithms. Supports either static public key
    or JWKS URI for key rotation.

    Note that this provider DOES NOT permit client registration or revocation, or any OAuth flows.
    It is intended to be used with a control plane that manages clients and tokens.
    """

    def __init__(
        self,
        public_key: str | None = None,
        jwks_uri: str | None = None,
        issuer: str | None = None,
        algorithm: str | None = None,
        audience: str | list[str] | None = None,
    ):
        """
        Initialize the provider. Either public_key or jwks_uri must be provided.

        Args:
            public_key: RSA public key in PEM format (for static key)
            jwks_uri: URI to fetch keys from (for key rotation)
            issuer: Expected issuer claim (optional)
            algorithm: Algorithm to use for verification (optional, defaults to RS256)
            audience: Expected audience claim - can be a string or list of strings (optional)
            required_scopes: List of required scopes for access (optional)
        """
        if not (public_key or jwks_uri):
            raise ValueError("Either public_key or jwks_uri must be provided")
        if public_key and jwks_uri:
            raise ValueError("Provide either public_key or jwks_uri, not both")

        if not algorithm:
            algorithm = "RS256"
        if algorithm not in {
            "HS256",
            "HS384",
            "HS512",
            "RS256",
            "RS384",
            "RS512",
            "ES256",
            "ES384",
            "ES512",
            "PS256",
            "PS384",
            "PS512",
        }:
            raise ValueError(f"Unsupported algorithm: {algorithm}.")

        # Only pass issuer to parent if it's a valid URL, otherwise use default
        # This allows the issuer claim validation to work with string issuers per RFC 7519
        try:
            issuer_url = AnyHttpUrl(issuer)
        except ValidationError:
            raise ValueError(f"Invalid issuer: {issuer}.")

        self.algorithm = algorithm
        self.issuer = issuer
        self.issuer_url = issuer_url
        self.audience = audience
        self.public_key = public_key
        self.jwks_uri = jwks_uri
        self.jwt = JsonWebToken([self.algorithm])  # Use RS256 by default
        self.logger = logging.getLogger(__name__)

        # Simple JWKS cache
        self._jwks_cache: dict[str, str] = {}
        self._jwks_cache_time: float = 0
        self._cache_ttl = 3600  # 1 hour

    async def _get_verification_key(self, token: str) -> str:
        """Get the verification key for the token."""
        if self.public_key:
            return self.public_key

        # Extract kid from token header for JWKS lookup
        try:
            import base64
            import json

            header_b64 = token.split(".")[0]
            header_b64 += "=" * (4 - len(header_b64) % 4)  # Add padding
            header = json.loads(base64.urlsafe_b64decode(header_b64))
            kid = header.get("kid")

            return await self._get_jwks_key(kid)

        except Exception as e:
            raise ValueError(f"Failed to extract key ID from token: {e}")

    async def _get_jwks_key(self, kid: str | None) -> str:
        """Fetch key from JWKS with simple caching."""
        if not self.jwks_uri:
            raise ValueError("JWKS URI not configured")

        current_time = time.time()

        # Check cache first
        if current_time - self._jwks_cache_time < self._cache_ttl:
            if kid and kid in self._jwks_cache:
                return self._jwks_cache[kid]
            elif not kid and len(self._jwks_cache) == 1:
                # If no kid but only one key cached, use it
                return next(iter(self._jwks_cache.values()))

        # Fetch JWKS
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                jwks_data = response.json()

            # Cache all keys
            self._jwks_cache = {}
            for key_data in jwks_data.get("keys", []):
                key_kid = key_data.get("kid")
                jwk = JsonWebKey.import_key(key_data)
                public_key = jwk.get_public_key()  # type: ignore

                if key_kid:
                    self._jwks_cache[key_kid] = public_key
                else:
                    # Key without kid - use a default identifier
                    self._jwks_cache["_default"] = public_key

            self._jwks_cache_time = current_time

            # Select the appropriate key
            if kid:
                if kid not in self._jwks_cache:
                    self.logger.debug(
                        "JWKS key lookup failed: key ID '%s' not found", kid
                    )
                    raise ValueError(f"Key ID '{kid}' not found in JWKS")
                return self._jwks_cache[kid]
            else:
                # No kid in token - only allow if there's exactly one key
                if len(self._jwks_cache) == 1:
                    return next(iter(self._jwks_cache.values()))
                elif len(self._jwks_cache) > 1:
                    raise ValueError(
                        "Multiple keys in JWKS but no key ID (kid) in token"
                    )
                else:
                    raise ValueError("No keys found in JWKS")

        except Exception as e:
            self.logger.debug("JWKS fetch failed: %s", str(e))
            raise ValueError(f"Failed to fetch JWKS: {e}")

    async def load_access_token(self, token: str) -> AccessToken | None:
        """
        Validates the provided JWT bearer token.

        Args:
            token: The JWT token string to validate

        Returns:
            AccessToken object if valid, None if invalid or expired
        """
        try:
            # Get verification key (static or from JWKS)
            verification_key = await self._get_verification_key(token)

            # Decode and verify the JWT token
            claims = self.jwt.decode(token, verification_key)

            # Extract client ID early for logging
            client_id = claims.get("client_id") or claims.get("sub") or "unknown"

            # Validate expiration
            exp = claims.get("exp")
            if exp and exp < time.time():
                self.logger.debug(
                    "Token validation failed: expired token for client %s", client_id
                )
                self.logger.info("Bearer token rejected for client %s", client_id)
                return None

            # Validate issuer - note we use issuer instead of issuer_url here because
            # issuer is optional, allowing users to make this check optional
            if self.issuer:
                if claims.get("iss") != self.issuer:
                    self.logger.debug(
                        "Token validation failed: issuer mismatch for client %s",
                        client_id,
                    )
                    self.logger.info("Bearer token rejected for client %s", client_id)
                    return None

            # Validate audience if configured
            if self.audience:
                aud = claims.get("aud")

                # Handle different combinations of audience types
                audience_valid = False
                if isinstance(self.audience, list):
                    # self.audience is a list - check if any expected audience is present
                    if isinstance(aud, list):
                        # Both are lists - check for intersection
                        audience_valid = any(
                            expected in aud for expected in self.audience
                        )
                    else:
                        # aud is a string - check if it's in our expected list
                        audience_valid = aud in self.audience
                else:
                    # self.audience is a string - use original logic
                    if isinstance(aud, list):
                        audience_valid = self.audience in aud
                    else:
                        audience_valid = aud == self.audience

                if not audience_valid:
                    self.logger.debug(
                        "Token validation failed: audience mismatch for client %s",
                        client_id,
                    )
                    self.logger.info("Bearer token rejected for client %s", client_id)
                    return None

            return AccessToken.from_dict(claims)

        except JoseError:
            self.logger.debug("Token validation failed: JWT signature/format invalid")
            return None
        except Exception as e:
            self.logger.debug("Token validation failed: %s", str(e))
            return None

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify a bearer token and return access info if valid.

        This method implements the TokenVerifier protocol by delegating
        to our existing load_access_token method.

        Args:
            token: The JWT token string to validate

        Returns:
            AccessToken object if valid, None if invalid or expired
        """
        return await self.load_access_token(token)
