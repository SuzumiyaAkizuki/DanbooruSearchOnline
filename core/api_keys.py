"""API Key policy and HTTPS RPC adapter. No I/O until explicitly enabled/used."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import math
import os
import re
import secrets
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.admin_auth import ADMIN_SUB, TARGET_SPACE

TERMS_VERSION = "2026-09-22-v1"
COSTS = {"search": 3, "related": 2, "artists": 1}
GOVERNED = ContextVar("api_key_governed", default=False)


@dataclass(frozen=True)
class KeyConfig:
    mode: str = "off"
    testers: frozenset[str] = frozenset({ADMIN_SUB})
    url: str = ""
    service_key: str = field(default="", repr=False)
    hmac_keys: dict[str, str] = field(default_factory=dict, repr=False)
    hmac_version: str = "v1"
    anonymous_daily: int = 15000
    anonymous_per_minute: int = 30
    key_per_minute: int = 60
    rest_per_minute: int = 120
    anonymous_burst: int = 6
    key_burst: int = 6
    rest_burst: int = 12
    anonymous_concurrency: int = 1
    key_concurrency: int = 1
    rest_concurrency: int = 2
    search_concurrency: int = 1
    recommendation_concurrency: int = 1
    timeout: float = 120.0

    @classmethod
    def from_env(cls):
        mode = os.getenv("API_KEY_MODE", "off")
        # Other deployments keep their existing behavior. Tests inject configuration.
        if os.getenv("SPACE_ID") != TARGET_SPACE:
            mode = "off"
        if mode == "off":
            return cls()
        if mode not in {"preview", "public"}:
            # Bad deployment config must not crash the public UI. REST stays closed
            # rather than accidentally reverting to unrestricted operation.
            return cls(mode="invalid")
        try:
            keys = json.loads(os.getenv("API_KEY_HMAC_KEYS", "{}")) if mode != "off" else {}
            if not isinstance(keys, dict) or any(not re.fullmatch(r"v[0-9]+", k) or not isinstance(v, str) or len(v) < 32 for k, v in keys.items()):
                raise ValueError()
            daily = int(os.getenv("API_KEY_ANONYMOUS_DAILY", "15000"))
            if daily < 1:
                raise ValueError()
            limits = {}
            for name in ("anonymous_per_minute", "key_per_minute", "rest_per_minute", "anonymous_burst", "key_burst", "rest_burst",
                         "anonymous_concurrency", "key_concurrency", "rest_concurrency", "search_concurrency", "recommendation_concurrency"):
                limits[name] = int(os.getenv("API_KEY_" + name.upper(), str(getattr(cls(),name))))
                if limits[name] < (3 if name.endswith("burst") else 1) or limits[name] > 100000:
                    raise ValueError()
        except (ValueError, TypeError):
            return cls(mode=mode)  # unready; no secrets in errors and no UI startup failure
        return cls(mode=mode, testers=frozenset({ADMIN_SUB} | set(filter(None, os.getenv("API_KEY_TEST_SUBS", "").split(",")))),
                   url=os.getenv("SUPABASE_URL", "").rstrip("/"),
                   service_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""), hmac_keys=keys,
                   hmac_version=os.getenv("API_KEY_HMAC_VERSION", "v1"), anonymous_daily=daily, **limits)

    @property
    def ready(self):
        return bool(self.mode in {"preview", "public"} and re.fullmatch(r"https://[a-z0-9-]+\.supabase\.co", self.url)
                    and self.service_key and len(self.hmac_keys.get(self.hmac_version, "")) >= 32)

    def allows(self, sub):
        return self.mode == "public" or (self.mode == "preview" and sub in self.testers)


class ApplicationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submission_id: uuid.UUID
    kind: str
    client: str
    site: str = ""
    purpose: str = Field("", max_length=1000)
    daily: int = Field(3000, ge=1, le=10000000, strict=True)
    reason: str = Field("", max_length=1000)
    terms: str
    accepted: bool
    grant_id: uuid.UUID | None = None

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value):
        if value not in {"personal", "public"}:
            raise ValueError("请选择个人或公开业务")
        return value

    @field_validator("client")
    @classmethod
    def valid_client(cls, value):
        value = value.strip()
        if not re.fullmatch(r"[\x20-\x7e]{1,80}", value):
            raise ValueError("Client 必须为 1–80 个可打印 ASCII 字符")
        return value

    @field_validator("site")
    @classmethod
    def valid_site(cls, value):
        value = value.strip()
        if not value:
            return value
        try:
            parsed = urlsplit(value)
            if (len(value) > 500 or parsed.scheme != "https" or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or any(ord(c) <= 32 or ord(c) >= 127 for c in value) or "\\" in value):
                raise ValueError()
            parsed.port
        except ValueError:
            raise ValueError("Site 必须是单个不含账号密码的 HTTPS 地址（域名使用 ASCII）") from None
        return value

    def payload(self):
        if not self.accepted or self.terms != TERMS_VERSION:
            raise HTTPException(422, "请阅读并同意当前版本须知")
        if self.kind == "public" and not self.site:
            raise HTTPException(422, "公开业务须登记 HTTPS Site")
        if self.kind == "personal" and self.daily > 6000:
            raise HTTPException(422, "个人业务最高 6000 点/日")
        if (self.kind == "public" or self.daily > 3000 or self.grant_id) and not self.purpose.strip():
            raise HTTPException(422, "需要人工审核，请填写用途说明")
        data = self.model_dump(mode="json")
        if self.kind == "personal":
            data["site"] = ""
        return data


class RPC:
    def __init__(self, config, transport=None):
        self.config, self.transport = config, transport

    async def call(self, name, **payload):
        if not self.config.ready:
            raise HTTPException(503, "key_service_unconfigured")
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=False, transport=self.transport) as client:
                response = await client.post(self.config.url + "/rest/v1/rpc/" + name,
                    headers={"apikey": self.config.service_key, "Authorization": "Bearer " + self.config.service_key},
                    json={"p": payload})
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError()
        except (httpx.HTTPError, ValueError):
            raise HTTPException(503, "key_service_unavailable") from None
        if result.get("error"):
            headers = {"Retry-After": str(result["retry_after"])} if "retry_after" in result else None
            raise HTTPException(int(result.get("status", 409)), result["error"], headers=headers)
        return result


@dataclass
class Bucket:
    tokens: float
    at: float
    active: int = 0


class Limiter:
    """One event loop/worker, bounded buckets; never evict an active/depleted bucket."""
    def __init__(self, clock=time.monotonic, capacity=4096, config=None):
        self.clock, self.capacity = clock, capacity
        self.config = config or KeyConfig()
        self.buckets = {}
        self.total = Bucket(self.config.rest_burst, clock())
        self.lanes = {"search": 0, "recommendation": 0}
        self.rejections = 0

    def acquire(self, subject, endpoint):
        now, cost = self.clock(), COSTS[endpoint]
        anonymous = subject.startswith("anonymous")
        c = self.config
        rate = (c.anonymous_per_minute if anonymous else c.key_per_minute) / 60
        burst = c.anonymous_burst if anonymous else c.key_burst
        if subject not in self.buckets:
            for key, bucket in list(self.buckets.items()):
                full_after = (max(c.anonymous_burst,c.key_burst)*60 / min(c.anonymous_per_minute,c.key_per_minute))
                if not bucket.active and now - bucket.at >= max(60,full_after):
                    self.buckets.pop(key)
            if len(self.buckets) >= self.capacity:
                raise HTTPException(503, "key_limiter_capacity")
            self.buckets[subject] = Bucket(burst, now)
        bucket = self.buckets[subject]
        bucket.tokens = min(burst, bucket.tokens + (now - bucket.at) * rate)
        self.total.tokens = min(c.rest_burst, self.total.tokens + (now - self.total.at) * c.rest_per_minute / 60)
        bucket.at = self.total.at = now
        lane = "search" if endpoint == "search" else "recommendation"
        prefix = "anonymous" if anonymous else "key"
        error, retry = None, 1
        if bucket.active >= (c.anonymous_concurrency if anonymous else c.key_concurrency):
            error = prefix + "_concurrency_limited"
        elif self.total.active >= c.rest_concurrency or self.lanes[lane] >= (c.search_concurrency if lane=="search" else c.recommendation_concurrency):
            error = "rest_concurrency_limited"
        elif bucket.tokens < cost:
            error, retry = prefix + "_rate_limited", math.ceil((cost - bucket.tokens) / rate)
        elif self.total.tokens < cost:
            error, retry = "rest_rate_limited", math.ceil((cost - self.total.tokens) * 60 / c.rest_per_minute)
        if error:
            self.rejections += 1
            raise HTTPException(429, error, headers={"Retry-After": str(retry)})
        bucket.tokens -= cost
        self.total.tokens -= cost
        bucket.active += 1
        self.total.active += 1
        self.lanes[lane] += 1
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                bucket.active -= 1
                self.total.active -= 1
                self.lanes[lane] -= 1
        return release


class KeyService:
    def __init__(self, config, rpc=None, limiter=None):
        self.config = config
        self.rpc = rpc or RPC(config)
        self.limiter = limiter or Limiter(config=config)
        self.tasks = set()
        self.pending_refunds = {}  # bounded, no user content; manual recovery if process is lost
        self.auth_gate = Bucket(12, time.monotonic())

    def track(self, task):
        self.tasks.add(task)
        def done(future):
            self.tasks.discard(future)
            if not future.cancelled():
                future.exception()  # retrieve exceptions even after HTTP client disconnects
        task.add_done_callback(done)
        return task

    def manages(self, headers):
        if self.config.mode == "off":
            return False
        if self.config.mode != "preview":
            return True
        # Preview only opts in our credential namespace. Existing callers are untouched.
        return bool(re.match(r"(?i)^bearer\s+dsk_", headers.get("authorization", "")))

    def digest(self, key):
        match = re.fullmatch(r"dsk_(v[0-9]+)_([a-f0-9]{32})_([A-Za-z0-9_-]{43})", key)
        if not match or match[1] not in self.config.hmac_keys:
            raise HTTPException(401, "invalid_api_key")
        return match[2], hmac.new(self.config.hmac_keys[match[1]].encode(), key.encode(), hashlib.sha256).hexdigest()

    async def issue(self, actor, grant_id, rotate, version):
        if not self.config.ready:
            raise HTTPException(503, "key_service_unconfigured")
        key_id = uuid.uuid4().hex
        key = f"dsk_{self.config.hmac_version}_{key_id}_{secrets.token_urlsafe(32)}"
        _, digest = self.digest(key)
        result = await self.rpc.call("ds_key_portal", action="rotate" if rotate else "claim", actor=actor,
                    grant_id=grant_id, version=version, key_id=key_id, digest=digest,
                    hmac_version=self.config.hmac_version, prefix=key[:len(key)-43] + "…")
        return {**result, "key": key}

    async def refund(self, request_id):
        try:
            await self.rpc.call("ds_key_meter", action="refund", request_id=request_id)
            self.pending_refunds.pop(request_id, None)
        except HTTPException:
            self.pending_refunds[request_id] = True
            # Record intent separately if the failure was limited to the refund transaction.
            try:
                await self.rpc.call("ds_key_meter", action="mark_refund", request_id=request_id)
            except HTTPException:
                pass

    async def retry_refunds(self):
        for request_id in list(self.pending_refunds):
            await self.refund(request_id)
        await self.rpc.call("ds_key_meter", action="retry_refunds")

    async def execute(self, headers, endpoint, business, *, preview_anonymous=False):
        if not preview_anonymous and not self.manages(headers):
            return await business()
        if not self.config.ready:
            raise HTTPException(503, "key_service_unconfigured")
        if len(self.pending_refunds) >= 128:
            raise HTTPException(503, "refund_recovery_required")
        key_id, digest = "", ""
        subject = "anonymous_preview" if preview_anonymous else "anonymous"
        if "authorization" in headers and not preview_anonymous:
            value = headers["authorization"]
            if not value.lower().startswith("bearer "):
                raise HTTPException(401, "invalid_api_key")
            key_id, digest = self.digest(value[7:])
            # Resolve the stable logical grant before bucket lookup; rotation keeps its bucket.
            gate, now = self.auth_gate, time.monotonic()
            gate.tokens = min(12, gate.tokens + (now-gate.at)*2)
            gate.at = now
            if gate.tokens < 1 or gate.active >= 4:
                raise HTTPException(429, "authentication_rate_limited", headers={"Retry-After": "1"})
            gate.tokens -= 1
            gate.active += 1
            try:
                identity = await self.rpc.call("ds_key_resolve", key_id=key_id, digest=digest)
            finally:
                gate.active -= 1
            if not self.config.allows(identity["sub"]):
                raise HTTPException(403, "preview_account_required")
            subject = identity["grant_id"]
        release = self.limiter.acquire(subject, endpoint)
        request_id = str(uuid.uuid4())
        try:
            await self.rpc.call("ds_key_meter", action="debit", request_id=request_id,
                subject=subject, key_id=key_id, digest=digest, endpoint=endpoint,
                client=headers.get("x-danboorusearch-client", ""), site=headers.get("x-danboorusearch-site", ""),
                anonymous_daily=self.config.anonymous_daily)
        except BaseException as exc:
            # A lost debit response must never start work; refund with the same ID is safe.
            if not isinstance(exc, HTTPException) or exc.status_code >= 500:
                self.track(asyncio.create_task(self.refund(request_id)))
            release()
            raise

        async def run():
            token = GOVERNED.set(True)
            try:
                result = await business()
                if isinstance(result, dict) and "error" in result:
                    await self.refund(request_id)  # semantic input rejection, before computation
                return result
            except BaseException:
                await self.refund(request_id)
                raise
            finally:
                GOVERNED.reset(token)
                release()

        task = self.track(asyncio.create_task(run()))
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.config.timeout)
        except asyncio.TimeoutError:
            await self.refund(request_id)
            raise HTTPException(503, "business_timeout") from None
        # Cancellation of the HTTP waiter does not cancel business or release its slot.


@lru_cache(maxsize=1)
def get_key_service():
    return KeyService(KeyConfig.from_env())


def governed(endpoint):
    def decorate(function):
        @wraps(function)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("http_request")
            if request is None:  # direct internal invocation, existing contract tests
                return await function(*args, **kwargs)
            service = get_key_service()
            return await service.execute(request.headers, endpoint, lambda: function(*args, **kwargs))
        # FastAPI must resolve endpoint annotations in the endpoint's own module.
        wrapper.__signature__ = inspect.signature(function, eval_str=True)
        return wrapper
    return decorate


async def engine_call(tagger, method, *args, **kwargs):
    if not GOVERNED.get():
        return await getattr(tagger, method + "_async")(*args, **kwargs)
    # Keep the existing engine semaphore until the actual worker finishes. The outer
    # HTTP deadline is shielded, so wait_for cannot prematurely release either slot.
    slot = tagger._cpu_slot() if method == "search" else tagger._get_recommendation_sem()
    async with slot:
        return await asyncio.to_thread(getattr(tagger, method), *args, **kwargs)
