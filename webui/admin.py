"""HTTP admin pages mounted into the existing app, independent of NiceGUI clients."""
from __future__ import annotations

import asyncio
import inspect
import secrets
import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from core.admin_auth import (
    AdminAuth, AdminConfig, FLOW_COOKIE, FLOW_TTL,
    LoginRejected, SESSION_COOKIE, SESSION_TTL, install_callback_log_filter,
)
from core.admin_metrics import read_dashboard

ASSETS = Path(__file__).parent / "static" / "admin"
SECURITY_HEADERS = {
    "Cache-Control": "no-store, private", "Pragma": "no-cache",
    "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
    "X-Robots-Tag": "noindex, nofollow",
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'self' https://huggingface.co",
}


@lru_cache(maxsize=3)
def asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def create_admin_router(config: AdminConfig | None = None, *, auth: AdminAuth | None = None,
                        snapshot_reader=read_dashboard) -> APIRouter:
    config = config or AdminConfig.from_env()
    auth = auth or AdminAuth(config)
    install_callback_log_filter()
    router = APIRouter(prefix="/admin", include_in_schema=False)
    cache: dict = {}
    snapshot_lock = asyncio.Lock()

    def json(data: dict, status: int = 200):
        return JSONResponse(data, status_code=status, headers=SECURITY_HEADERS)

    def redirect(path: str):
        return RedirectResponse(path, status_code=303, headers=SECURITY_HEADERS)

    def set_cookie(response: Response, name: str, value: str, age: int):
        response.set_cookie(name, value, max_age=age, path="/admin", secure=config.cookie_secure,
                            httponly=True, samesite="lax")

    def clear_cookie(response: Response, name: str):
        response.delete_cookie(name, path="/admin", secure=config.cookie_secure,
                               httponly=True, samesite="lax")

    def session(request: Request):
        if not config.ready:
            return None
        return auth.get_session(request.cookies.get(SESSION_COOKIE, ""))

    def same_origin(request: Request) -> bool:
        origin = request.headers.get("origin")
        return (not origin or origin == config.origin) and request.headers.get("sec-fetch-site") != "cross-site"

    @router.get("")
    @router.get("/")
    @router.get("/api-keys")
    async def page(request: Request):
        # Public shell contains no metrics, identity, secrets or hidden admin data.
        return HTMLResponse(asset("index.html"), headers=SECURITY_HEADERS)

    @router.get("/assets/{name}")
    async def static_asset(name: str):
        types = {"admin.css": "text/css", "admin.js": "text/javascript"}
        if name not in types:
            return json({"error": "not_found"}, 404)
        return Response(asset(name), media_type=types[name], headers=SECURITY_HEADERS)

    @router.get("/api/session")
    async def identity(request: Request):
        if not same_origin(request):
            return json({"error": "forbidden"}, 403)
        current = session(request)
        if not current:
            return json({"authenticated": False, "configured": config.ready,
                         "login_url": config.origin + "/admin/login" if config.ready else None}, 401)
        return json({"authenticated": True, "username": current.username,
                     "issuer": current.issuer, "sub": current.sub, "csrf": current.csrf,
                     "expires_in": max(0, int(current.expires - auth.clock()))})

    @router.get("/login")
    async def login(request: Request):
        if not config.ready:
            return redirect("/admin?notice=unconfigured")
        try:
            state, url = auth.start(request.cookies.get(FLOW_COOKIE, ""))
        except LoginRejected:
            return redirect("/admin?notice=busy")
        response = redirect(url)
        set_cookie(response, FLOW_COOKIE, state, FLOW_TTL)
        return response

    @router.get("/callback")
    async def callback(request: Request):
        response = redirect("/admin?notice=login_failed")
        try:
            if not config.ready:
                raise LoginRejected()
            flow = auth.consume(request.query_params.get("state", ""), request.cookies.get(FLOW_COOKIE, ""))
            auth.sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
            code = request.query_params.get("code", "")
            if request.query_params.get("error") or not code or len(code) > 4096 or auth.exchanges >= 4:
                raise LoginRejected()
            auth.exchanges += 1
            try:
                identity = await auth.provider.exchange(code, flow)
            finally:
                auth.exchanges -= 1
            # Permission is checked on the verified issuer/sub, never the display name.
            token = auth.issue(identity)
            auth.sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
            response = redirect("/admin")
            set_cookie(response, SESSION_COOKIE, token, SESSION_TTL)
        except PermissionError:
            response = redirect("/admin?notice=forbidden")
        except LoginRejected:
            pass
        clear_cookie(response, FLOW_COOKIE)
        return response

    @router.post("/logout")
    async def logout(request: Request):
        current = session(request)
        if not current:
            return json({"error": "unauthorized"}, 401)
        if not same_origin(request) or not secrets.compare_digest(request.headers.get("x-csrf-token", "").encode(), current.csrf.encode()):
            return json({"error": "forbidden"}, 403)
        auth.sessions.pop(request.cookies.get(SESSION_COOKIE, ""), None)
        response = json({"ok": True})
        clear_cookie(response, SESSION_COOKIE)
        return response

    @router.get("/api/overview")
    async def overview(request: Request):
        if not session(request):
            return json({"error": "unauthorized"}, 401)
        if not same_origin(request):
            return json({"error": "forbidden"}, 403)
        # At most one read per 30 seconds across tabs. No OSS fetch/sync on refresh.
        async with snapshot_lock:
            if time.monotonic() - cache.get("at", -float("inf")) >= 30:
                try:
                    data = snapshot_reader()
                    if inspect.isawaitable(data):
                        data = await data
                except Exception:
                    return json({"error": "metrics_unavailable"}, 503)
                cache.update(data=data, at=time.monotonic())
        return json(cache["data"])

    return router
