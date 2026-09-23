"""Isolated HF OIDC login for the read-only admin portal.

Only opaque, bounded, expiring sessions live in memory. No middleware, cookies,
network calls or storage changes are applied to the public application.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlencode

import httpx
import jwt

HF_ISSUER = "https://huggingface.co"
# Public HF profile verified on 2026-09-19: user=SAkizuki, fullname=Sakizuki.
# Must match the verified OIDC sub. There is deliberately no username fallback.
ADMIN_SUB = "693d7b970b708e7e897cabcf"
TARGET_SPACE = "SAkizuki/DanbooruSearch"
SESSION_COOKIE = "ds_admin_session"
FLOW_COOKIE = "ds_admin_flow"
SESSION_TTL = 2 * 60 * 60
FLOW_TTL = 5 * 60


class AdminCallbackLogFilter(logging.Filter):
    """Redact OAuth callback queries only; preserve public access log behavior."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and len(record.args) == 5:
            args = list(record.args)
            target = str(args[2])
            if target.split("?", 1)[0] in {"/admin/callback", "/developer/callback"}:
                args[2] = target.split("?", 1)[0]
                record.args = tuple(args)
        return True


def install_callback_log_filter() -> None:
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, AdminCallbackLogFilter) for item in logger.filters):
        logger.addFilter(AdminCallbackLogFilter())


@dataclass(frozen=True)
class AdminConfig:
    client_id: str = ""
    client_secret: str = field(default="", repr=False)
    origin: str = ""
    admin_sub: str = ADMIN_SUB
    enabled: bool = False
    cookie_secure: bool = True

    @classmethod
    def from_env(cls) -> "AdminConfig":
        host = os.environ.get("SPACE_HOST", "")
        valid_host = bool(re.fullmatch(r"[a-zA-Z0-9-]+\.hf\.space", host))
        return cls(
            client_id=os.environ.get("OAUTH_CLIENT_ID", ""),
            client_secret=os.environ.get("OAUTH_CLIENT_SECRET", ""),
            origin=f"https://{host}" if valid_host else "",
            enabled=os.environ.get("SPACE_ID") == TARGET_SPACE,
        )

    @property
    def ready(self) -> bool:
        return bool(self.enabled and self.client_id and self.client_secret and self.origin)

    @property
    def callback_url(self) -> str:
        return self.origin + "/admin/callback"


@dataclass
class LoginFlow:
    expires: float
    nonce: str
    verifier: str = field(repr=False)


@dataclass
class AdminSession:
    expires: float
    issuer: str
    sub: str
    username: str
    csrf: str = field(repr=False)


class LoginRejected(Exception):
    """Safe, intentionally detail-free authentication failure."""


class HFIdentityProvider:
    """HF authorization code + S256 PKCE; PyJWT verifies the ID token.

    Endpoints and issuer are pinned to HF, never taken from request headers or
    unverified tokens. JWKS fetches are serialized, cached and timeout-bounded.
    """

    def __init__(self, config: AdminConfig, transport=None):
        self.config = config
        self.transport = transport
        self._keys: list[dict] = []
        self._keys_at = 0.0
        self._lock = asyncio.Lock()

    async def _signing_key(self, client: httpx.AsyncClient, token: str):
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
            raise LoginRejected()
        async with self._lock:
            known = any(k.get("kid") == header["kid"] for k in self._keys)
            age = time.monotonic() - self._keys_at
            if not self._keys or age >= 3600 or (not known and age >= 30):
                response = await client.get(HF_ISSUER + "/oauth/jwks")
                response.raise_for_status()
                self._keys = response.json()["keys"]
                self._keys_at = time.monotonic()
            for key in self._keys:
                if (key.get("kid") == header["kid"] and key.get("kty") == "RSA"
                        and key.get("use", "sig") == "sig"
                        and key.get("alg", "RS256") == "RS256"):
                    return jwt.PyJWK.from_dict(key, algorithm="RS256").key
        raise LoginRejected()

    async def exchange(self, code: str, flow: LoginFlow) -> dict:
        try:
            async with httpx.AsyncClient(
                timeout=10, follow_redirects=False, transport=self.transport,
            ) as client:
                response = await client.post(
                    HF_ISSUER + "/oauth/token",
                    auth=(self.config.client_id, self.config.client_secret),
                    data={"grant_type": "authorization_code", "code": code,
                          "redirect_uri": self.config.callback_url,
                          "code_verifier": flow.verifier},
                )
                response.raise_for_status()
                tokens = response.json()
                token = tokens["id_token"]
                key = await self._signing_key(client, token)
                claims = jwt.decode(
                    token, key, algorithms=["RS256"], issuer=HF_ISSUER,
                    audience=self.config.client_id, leeway=30,
                    options={"require": ["iss", "sub", "aud", "exp", "iat", "nonce"]},
                )
                if not secrets.compare_digest(str(claims["nonce"]).encode(), flow.nonce.encode()):
                    raise LoginRejected()
                multiple_audiences = isinstance(claims["aud"], list) and len(claims["aud"]) > 1
                if ("azp" in claims or multiple_audiences) and claims.get("azp") != self.config.client_id:
                    raise LoginRejected()
                access_token = tokens["access_token"]
                if "at_hash" in claims:
                    digest = hashlib.sha256(access_token.encode("ascii")).digest()[:16]
                    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                    if not secrets.compare_digest(str(claims["at_hash"]).encode(), expected.encode()):
                        raise LoginRejected()
                # Userinfo is used only for display, and must describe the same identity.
                response = await client.get(
                    HF_ISSUER + "/oauth/userinfo",
                    headers={"Authorization": "Bearer " + access_token},
                )
                response.raise_for_status()
                profile = response.json()
                if profile.get("sub") != claims["sub"]:
                    raise LoginRejected()
                return {"issuer": claims["iss"], "sub": claims["sub"],
                        "username": str(profile.get("preferred_username") or "管理员")[:100]}
        except (httpx.HTTPError, jwt.PyJWTError, KeyError, ValueError, TypeError, UnicodeError):
            # Never print code, tokens, HTTP bodies or exception strings.
            raise LoginRejected() from None


class AdminAuth:
    max_sessions = 32
    def __init__(self, config: AdminConfig, provider=None, clock: Callable = time.monotonic):
        self.config = config
        self.provider = provider or HFIdentityProvider(config)
        self.clock = clock
        self.flows: dict[str, LoginFlow] = {}
        self.sessions: dict[str, AdminSession] = {}
        self.exchanges = 0

    def prune(self) -> None:
        now = self.clock()
        for store in (self.flows, self.sessions):
            for key in list(store):
                if store[key].expires <= now:
                    store.pop(key, None)

    def start(self, old_flow: str = "") -> tuple[str, str]:
        self.prune()
        self.flows.pop(old_flow, None)
        if len(self.flows) >= 128:
            raise LoginRejected()
        state = secrets.token_urlsafe(32)
        flow = LoginFlow(self.clock() + FLOW_TTL, secrets.token_urlsafe(32), secrets.token_urlsafe(48))
        self.flows[state] = flow
        challenge = base64.urlsafe_b64encode(hashlib.sha256(flow.verifier.encode()).digest()).rstrip(b"=").decode()
        query = urlencode({"client_id": self.config.client_id, "redirect_uri": self.config.callback_url,
                           "response_type": "code", "scope": "openid profile", "state": state,
                           "nonce": flow.nonce, "code_challenge": challenge, "code_challenge_method": "S256"})
        return state, HF_ISSUER + "/oauth/authorize?" + query

    def consume(self, state: str, cookie: str) -> LoginFlow:
        self.prune()
        if not state or not cookie or not secrets.compare_digest(state.encode(), cookie.encode()):
            raise LoginRejected()
        flow = self.flows.pop(state, None)
        if flow is None:
            raise LoginRejected()
        return flow

    def issue(self, identity: dict) -> str:
        self.prune()
        if not self.allowed(identity.get("issuer"), identity.get("sub")):
            raise PermissionError()
        if len(self.sessions) >= self.max_sessions:
            raise LoginRejected()
        token = secrets.token_urlsafe(32)
        self.sessions[token] = AdminSession(
            self.clock() + SESSION_TTL, HF_ISSUER, identity["sub"],
            identity["username"], secrets.token_urlsafe(32),
        )
        return token

    def get_session(self, token: str) -> AdminSession | None:
        self.prune()
        session = self.sessions.get(token)
        if session and self.allowed(session.issuer, session.sub):
            return session
        return None

    def allowed(self, issuer, sub):
        return issuer == HF_ISSUER and sub == self.config.admin_sub
