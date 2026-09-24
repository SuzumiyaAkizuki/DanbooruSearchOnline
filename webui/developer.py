"""Small HF developer portal, isolated from the anonymous NiceGUI application."""
from __future__ import annotations

import json as json_module
import secrets
import re
from html import escape
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import ValidationError
from markdown_it import MarkdownIt

from core.ui_text import load_ui_text

from core.admin_auth import AdminAuth, AdminConfig, HF_ISSUER, LoginRejected, FLOW_TTL, SESSION_TTL, install_callback_log_filter
from core.api_keys import ApplicationIn, TERMS_VERSION, get_key_service

ASSETS = Path(__file__).parent / "static" / "developer"
COOKIE, FLOW = "ds_developer_session", "ds_developer_flow"
HEADERS = {"Cache-Control": "no-store, private", "Pragma": "no-cache", "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff", "X-Robots-Tag": "noindex, nofollow",
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'self' https://huggingface.co"}


def response(data, status=200):
    return JSONResponse(data, status_code=status, headers=HEADERS)


def render_key_page():
    template = (ASSETS / "index.html").read_text(encoding="utf-8")
    copy = load_ui_text()["developer"]
    markdown = MarkdownIt("commonmark", {"html": False})
    replacements = {}
    for name in ("rules", "terms"):
        replacements[name + "_title"] = escape(copy[name + "_title"])
        replacements[name + "_body"] = markdown.render(copy[name + "_markdown"])
    return re.sub(r"\{\{(rules_title|rules_body|terms_title|terms_body)\}\}",
                  lambda match: replacements[match[1]], template)


def key_page():
    return HTMLResponse(render_key_page(), headers=HEADERS)


def embedded_key_panel():
    html = render_key_page()
    body = html.split("<main>", 1)[1].split("</main>", 1)[0]
    body = re.sub(r"<header\b[^>]*>.*?</header>|<footer\b[^>]*>.*?</footer>", "", body, flags=re.S)
    body = re.sub(r'id="([^"]+)"', lambda m: 'id="key-' + m[1] + '"', body)
    body = re.sub(r'aria-controls="([^"]+)"', lambda m: 'aria-controls="key-' + m[1] + '"', body)
    return '<div id="key-workspace">' + body + '</div>'


def static(name):
    types = {"portal.js": "text/javascript", "portal_view.js": "text/javascript", "portal.css": "text/css"}
    if name not in types:
        return response({"error": "not_found"}, 404)
    return Response((ASSETS / name).read_text(encoding="utf-8"), media_type=types[name], headers=HEADERS)


@dataclass(frozen=True)
class DeveloperConfig(AdminConfig):
    @property
    def callback_url(self):
        return self.origin + "/developer/callback"


class DeveloperAuth(AdminAuth):
    max_sessions = 512

    def __init__(self, config, keys, **kwargs):
        super().__init__(config, **kwargs)
        self.keys = keys

    def allowed(self, issuer, sub):
        return issuer == HF_ISSUER and isinstance(sub, str) and bool(sub) and self.keys.allows(sub)


def actor(current):
    return {"issuer": current.issuer, "sub": current.sub, "username": current.username}


def check_csrf(request, current, origin):
    if (request.headers.get("origin") not in (None, origin)
            or request.headers.get("sec-fetch-site") == "cross-site"
            or not secrets.compare_digest(request.headers.get("x-csrf-token", "").encode(), current.csrf.encode())):
        raise HTTPException(403, "forbidden")


async def read_body(request):
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 12000:
            raise HTTPException(413, "request_too_large")
    try:
        data = json_module.loads(body)
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (ValueError, UnicodeError):
        raise HTTPException(422, "invalid_json") from None


def identity_id(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise HTTPException(422, "invalid_id") from None


def safe_error(exc):
    result = response({"error": exc.detail}, exc.status_code)
    if exc.headers:
        result.headers.update(exc.headers)
    return result


def create_developer_router(service=None, config=None, auth=None):
    service = service or get_key_service()
    config = config or DeveloperConfig.from_env()
    auth = auth or DeveloperAuth(config, service.config)
    install_callback_log_filter()
    router = APIRouter(prefix="/developer", include_in_schema=False)

    def current(request):
        if not config.ready or service.config.mode == "off":
            raise HTTPException(404, "not_open")
        value = auth.get_session(request.cookies.get(COOKIE, ""))
        if not value:
            raise HTTPException(401, "login_required")
        return value

    def cookie(result, name, value, ttl):
        result.set_cookie(name, value, max_age=ttl, path="/developer", secure=config.cookie_secure, httponly=True, samesite="lax")

    @router.get("")
    @router.get("/apply")
    @router.get("/keys")
    async def page():
        if service.config.mode == "off":
            return response({"error": "not_open"}, 404)
        return key_page()

    @router.get("/assets/{name}")
    async def asset(name: str):
        return static(name)

    @router.get("/api/session")
    async def session(request: Request):
        try:
            value = current(request)
            return response({"authenticated": True, "username": value.username, "csrf": value.csrf,
                "mode": service.config.mode, "terms": TERMS_VERSION, "ready": service.config.ready,
                "expires_in": max(0, int(value.expires-auth.clock()))})
        except HTTPException as exc:
            return response({"error": exc.detail, "mode": service.config.mode, "authenticated": False,
                "login_url": config.origin + "/developer/login" if config.ready and service.config.mode != "off" else None}, exc.status_code)

    @router.get("/login")
    async def login(request: Request):
        if not config.ready or service.config.mode == "off":
            return response({"error": "not_open"}, 404)
        try:
            state, url = auth.start(request.cookies.get(FLOW, ""))
        except LoginRejected:
            return response({"error": "login_busy"}, 429)
        result = RedirectResponse(url, 303, headers=HEADERS)
        cookie(result, FLOW, state, FLOW_TTL)
        return result

    @router.get("/callback")
    async def callback(request: Request):
        result = RedirectResponse("/developer/keys?notice=login_failed", 303, headers=HEADERS)
        try:
            if not config.ready or service.config.mode == "off":
                raise LoginRejected()
            flow = auth.consume(request.query_params.get("state", ""), request.cookies.get(FLOW, ""))
            code = request.query_params.get("code", "")
            if not code or len(code)>4096 or request.query_params.get("error") or auth.exchanges>=4:
                raise LoginRejected()
            auth.exchanges += 1
            try:
                identity = await auth.provider.exchange(code, flow)
            finally:
                auth.exchanges -= 1
            token = auth.issue(identity)
            auth.sessions.pop(request.cookies.get(COOKIE, ""), None)
            result = RedirectResponse("/developer/keys", 303, headers=HEADERS)
            cookie(result, COOKIE, token, SESSION_TTL)
        except (PermissionError, LoginRejected):
            pass
        cookie(result, FLOW, "", 0)
        return result

    @router.post("/logout")
    async def logout(request: Request):
        try:
            value = current(request)
            check_csrf(request, value, config.origin)
            auth.sessions.pop(request.cookies.get(COOKIE, ""), None)
            result = response({"ok": True})
            cookie(result, COOKIE, "", 0)
            return result
        except HTTPException as exc:
            return safe_error(exc)

    @router.get("/api/data")
    async def mine(request: Request, page: int = 0):
        try:
            value = current(request)
            return response(await service.rpc.call("ds_key_portal", action="mine", actor=actor(value), page=max(0,page)))
        except HTTPException as exc:
            return safe_error(exc)

    @router.post("/api/apply")
    async def apply(request: Request):
        try:
            value = current(request)
            check_csrf(request, value, config.origin)
            application = ApplicationIn.model_validate(await read_body(request))
            return response(await service.rpc.call("ds_key_portal", action="apply", actor=actor(value), application=application.payload()))
        except ValidationError:
            return response({"error": "invalid_application_fields"}, 422)
        except HTTPException as exc:
            return safe_error(exc)

    @router.post("/api/grants/{grant_id}/{action}")
    async def grant_action(request: Request, grant_id: str, action: str):
        try:
            value = current(request)
            check_csrf(request, value, config.origin)
            grant_id = identity_id(grant_id)
            data = await read_body(request)
            version = data.get("version")
            if type(version) is not int or action not in {"claim", "rotate", "revoke"}:
                raise HTTPException(422, "invalid_action")
            if action in {"claim", "rotate"}:
                return response(await service.issue(actor(value), grant_id, action=="rotate", version))
            return response(await service.rpc.call("ds_key_portal", action="revoke", actor=actor(value), grant_id=grant_id, version=version))
        except HTTPException as exc:
            return safe_error(exc)

    @router.post("/api/preview/{endpoint}")
    async def preview(request: Request, endpoint: str):
        """Authenticated anonymous-pool probe, never changes ordinary anonymous calls."""
        try:
            value = current(request)
            check_csrf(request, value, config.origin)
            if service.config.mode != "preview" or value.sub not in service.config.testers:
                raise HTTPException(404, "not_found")
            from api_fastapi import SearchIn, RelatedIn, ArtistIn, search, related, artists
            targets = {"search": (SearchIn, search), "related": (RelatedIn, related), "artists": (ArtistIn, artists)}
            if endpoint not in targets:
                raise HTTPException(404, "not_found")
            model, function = targets[endpoint]
            body = model.model_validate(await read_body(request))
            result = await service.execute({}, endpoint, lambda: function(body), preview_anonymous=True)
            return response(result.model_dump() if hasattr(result,"model_dump") else result)
        except ValidationError:
            return response({"error": "invalid_parameters"}, 422)
        except HTTPException as exc:
            return safe_error(exc)

    return router


def install_admin_key_routes(router, session, same_origin, service=None):
    service = service or get_key_service()

    def admin(request, mutate=False):
        value = session(request)  # existing admin OIDC whitelist; never accepts developer cookies
        if not value:
            raise HTTPException(401, "unauthorized")
        if not same_origin(request) or (mutate and not secrets.compare_digest(request.headers.get("x-csrf-token", "").encode(), value.csrf.encode())):
            raise HTTPException(403, "forbidden")
        if service.config.mode == "off":
            raise HTTPException(404, "not_open")
        return value

    @router.get("/api/key-data")
    async def data(request: Request, page: int = 0, search: str = "", state: str = "", grant_state: str = ""):
        try:
            value = admin(request)
            result = await service.rpc.call("ds_key_portal", action="admin_list", actor=actor(value), page=max(0,page), search=search[:80], state=state, grant_state=grant_state)
            result["pool"] = await service.rpc.call("ds_key_portal", action="pool", actor=actor(value))
            result["audit"] = await service.rpc.call("ds_key_portal", action="audit", actor=actor(value), page=max(0,page))
            result.update(mode=service.config.mode, anonymous_daily=service.config.anonymous_daily,
                in_flight={k:b.active for k,b in service.limiter.buckets.items() if k.startswith("anonymous")},
                rejections=service.limiter.rejections, local_pending_refunds=len(service.pending_refunds))
            return response(result)
        except HTTPException as exc:
            return safe_error(exc)

    @router.post("/api/key-action")
    async def action(request: Request):
        try:
            value = admin(request, True)
            data = await read_body(request)
            op = data.get("action")
            if op == "retry_refunds":
                await service.retry_refunds()
                return response({"ok": True})
            if op not in {"review", "update", "revoke", "reopen"}:
                raise HTTPException(422, "invalid_action")
            reason = data.get("reason", "")
            if not isinstance(reason,str) or not 1<=len(reason.strip())<=1000:
                raise HTTPException(422, "reason_required")
            payload = {"action":op, "actor":actor(value), "reason":reason}
            if op == "review":
                payload.update(id=identity_id(data.get("id")), decision=data.get("decision"), internal_note=str(data.get("internal_note", ""))[:1000])
                if payload["decision"] not in {"approve", "reject"}:
                    raise HTTPException(422,"invalid_decision")
            else:
                payload.update(grant_id=identity_id(data.get("grant_id")), version=data.get("version"))
                if type(payload["version"]) is not int:
                    raise HTTPException(422,"version_required")
            if op in {"update","review"}:
                daily = data.get("daily")
                if type(daily) is not int or not 1<=daily<=10000000:
                    raise HTTPException(422,"invalid_quota")
                payload.update(daily=daily)
            if op == "update":
                if data.get("state") not in {"active","paused"}:
                    raise HTTPException(422,"invalid_state")
                payload.update(state=data["state"])
            return response(await service.rpc.call("ds_key_portal", **payload))
        except HTTPException as exc:
            return safe_error(exc)
