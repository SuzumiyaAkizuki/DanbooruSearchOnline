"""
api_fastapi.py
──────────────
FastAPI 适配层（可选）。

演示如何在完全不修改 core/ 的情况下将引擎 API 化。

启动方式：
    uvicorn api_fastapi:app --host 0.0.0.0 --port 8000

请求示例：
    POST /search
    {
        "query": "白色水手服的女孩",
        "top_k": 5,
        "limit": 20
    }

    POST /related
    {
        "tags": ["white_serafuku", "sailor_collar"],
        "limit": 20,
        "show_nsfw": false
    }
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from core.engine import DanbooruTagger
from core.models import SearchRequest, SearchResponse
from core.prompt_formats import PROMPT_FORMATS
from core.api_keys import governed, engine_call, get_key_service
import core.counter as counter
import core.telemetry as telemetry
import core.traffic_attribution as traffic_attribution


# ── Pydantic I/O 模型（API 层专用，与 core.models 解耦）──

LayerName = Literal['英文', '中文扩展词', '释义', '中文核心词', 'artist']
CategoryName = Literal['General', 'Artist', 'Copyright', 'Character', 'Meta']
GroupMode = Literal['off', 'expand', 'diverse']

API_KEY_NOTICE = (
    "即将上线 API Key 与限流政策，当前尚未生效，申请入口及生效时间将另行公告。"
    "上线后每把 Key 默认 3000 点/日、60 点/分钟、并发 1，各 Key 独立；"
    "无 Key 调用共享 15000 点/日、30 点/分钟、并发 1 的试用池。"
    "search/related/artists/health 每次分别消耗 3/2/1/0 点，每日北京时间 00:00 重置。"
    "入口开放后使用 Hugging Face 账号登录申请；仅个人业务首把且不超默认日额度可自动批准，所有公开业务须人工审核；"
    "第二把及后续 Key、首把超默认额度或后续超默认额度增额需说明原因并人工审核。"
    "超限请求届时将返回 429，各 Key 仍受服务整体容量保护。网页搜索无需申请 Key。"
)
if get_key_service().config.mode == "public":
    API_KEY_NOTICE = (
        "API Key 与 REST 限流政策已启用。使用 Hugging Face 账号访问 /developer/apply 申请；"
        "仅个人首把且不超过 3000 点/日可自动批准，个人上限 6000 点/日；所有公开业务、第二把及增额须人工审核。"
        "每把 Key 独立计额，60 点/分钟、并发 1；无 Key 共享 15000 点/日、30 点/分钟、并发 1。"
        "search/related/artists/health 每次 3/2/1/0 点，北京时间零点重置。"
        "个人调用须匹配 Client，公开调用须匹配 Client 与 Site；无效 Key 不回退匿名。网页搜索及 MCP 保持原有方式。"
    )


class SearchIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "白色水手服的女孩",
                "top_k": 5,
                "limit": 80,
                "popularity_weight": 0.15,
                "show_nsfw": True,
                "use_segmentation": True,
                "target_layers": ['英文', '中文扩展词', '释义', '中文核心词', 'artist'],
                "target_categories": ['General', 'Character', 'Copyright'],
                "group_mode": "off",
                "max_per_group": 2,
            }
        }
    )

    query: str
    top_k: int = Field(5, ge=1, le=50)
    limit: int = Field(80, ge=1, le=500)
    popularity_weight: float = Field(0.15, ge=0.0, le=1.0)
    show_nsfw: bool = True
    use_segmentation: bool = True
    target_layers: list[LayerName] = Field(
        default_factory=lambda: ['英文', '中文扩展词', '释义', '中文核心词'],
        description="匹配层；可显式加入 'artist' 以返回编辑距离<=1的画师标签行。",
    )
    target_categories: list[CategoryName] = Field(
        default_factory=lambda: ['General', 'Character', 'Copyright'],
    )
    group_mode: GroupMode = "off"
    max_per_group: int = 2


class TagOut(BaseModel):
    tag: str
    cn_name: str
    category: str
    nsfw: str
    final_score: float
    semantic_score: float
    count: int
    source: str
    layer: str
    wiki: str = ""
    artist_top_tags: list[str] = Field(default_factory=list)
    alias_from: str | None = Field(default=None, exclude_if=lambda value: value is None)


class RelatedIn(BaseModel):
    tags: list[str]
    limit: int = Field(50, ge=1, le=200)
    show_nsfw: bool = True
    target_categories: list[CategoryName] | None = None


class RelatedTagOut(BaseModel):
    tag: str
    cn_name: str
    category: str = ""
    sources: list[str]
    wiki: str = ""


class SearchOut(BaseModel):
    api_key_notice: str = Field(default=API_KEY_NOTICE, description="当前 API Key 与限流政策说明")
    tags_all: str
    tags_sfw: str
    results: list[TagOut]
    keywords: list[str]


class ArtistIn(BaseModel):
    tags: list[str]
    limit: int = Field(30, ge=1, le=100)
    min_cooc: int = Field(3, ge=1, le=100)
    show_nsfw: bool = True


class ArtistOut(BaseModel):
    artist: str
    cooc_count: int
    post_count: int
    sources: list[str]
    top_tags: list[str]


async def _correct_tags(tagger: DanbooruTagger, tags: list[str]) -> tuple[list[str], list[str], dict[str, str]]:
    """Resolve input tags through the deterministic canonical-tag resolver."""
    corrected_tags: list[str] = []
    invalid_tags: list[str] = []
    corrections: dict[str, str] = {}

    for raw_tag in tags:
        resolved = tagger.resolve_tag_name(raw_tag)
        canonical = resolved.get("tag")
        if canonical:
            corrected_tags.append(canonical)
            if canonical != raw_tag:
                corrections[raw_tag] = canonical
        else:
            invalid_tags.append(raw_tag)

    return corrected_tags, invalid_tags, corrections


def _with_corrections(results: list[dict[str, Any]], corrections: dict[str, str]) -> dict[str, Any]:
    if not corrections:
        return {"results": results, "api_key_notice": API_KEY_NOTICE}
    correction_notes = [f"{bad} → {good}" for bad, good in corrections.items()]
    return {
        "correction_note": "标签拼写错误，已经纠错: " + ", ".join(correction_notes),
        "corrections": corrections,
        "results": results,
        "api_key_notice": API_KEY_NOTICE,
    }


# ── FastAPI 子应用（挂载到 NiceGUI 的 /api 路径下）──
# lifespan / 预热由 ui_nicegui.py 的 @app.on_startup 统一管理，此处不重复。
app = FastAPI(
    title="Danbooru Tag Searcher API",
    description=API_KEY_NOTICE + "\n\n申请与管理：[开发者页面](/developer/apply)。"
        "凭证使用 Authorization: Bearer <API_KEY>；登记头为 X-DanbooruSearch-Client 和 X-DanbooruSearch-Site。",
    version="1.0.0",
    docs_url=None,
)


async def _policy_notice_error(request: Request, exc):
    # Preserve FastAPI's error status, detail and headers; only append public copy.
    handler = request_validation_exception_handler if isinstance(exc, RequestValidationError) else http_exception_handler
    response = await handler(request, exc)
    if response.body:
        payload = json.loads(response.body)
        payload["api_key_notice"] = API_KEY_NOTICE
        response.body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        response.headers["content-length"] = str(len(response.body))
    return response


app.add_exception_handler(StarletteHTTPException, _policy_notice_error)
app.add_exception_handler(RequestValidationError, _policy_notice_error)


app.mount(
    "/docs-assets",
    StaticFiles(directory=Path(__file__).resolve().parent / "webui" / "static" / "swagger-ui"),
    name="docs-assets",
)


@app.get("/docs", include_in_schema=False)
async def swagger_docs(request: Request):
    # root_path 包含 /api 挂载前缀及反向代理前缀，独立运行时为空。
    root_path = request.scope.get("root_path", "").rstrip("/")
    return get_swagger_ui_html(
        openapi_url=f"{root_path}{app.openapi_url}",
        title=f"{app.title} - Swagger UI",
        swagger_js_url=f"{root_path}/docs-assets/swagger-ui-bundle.js",
        swagger_css_url=f"{root_path}/docs-assets/swagger-ui.css",
        swagger_favicon_url="data:,",
        oauth2_redirect_url=f"{root_path}{app.swagger_ui_oauth2_redirect_url}",
        swagger_ui_parameters={"validatorUrl": None},
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_oauth2_redirect():
    return get_swagger_ui_oauth2_redirect_html()


def _attribution_endpoint(request: Request) -> str | None:
    path = request.url.path.rstrip("/")
    method = request.method.upper()
    if method == "POST":
        for endpoint in ("search", "related", "artists"):
            if path.endswith(f"/{endpoint}"):
                return endpoint
    if method == "GET" and path.endswith("/health"):
        return "health"
    return None


@app.middleware("http")
async def observe_rest_attribution(request: Request, call_next):
    """Observe aggregate REST source signals without gating or rejecting calls."""
    endpoint = _attribution_endpoint(request)
    if endpoint is None:
        return await call_next(request)

    started_at = time.perf_counter()
    observation = None
    try:
        observation = traffic_attribution.start_request(request.headers)
    except Exception as exc:
        print(f"[TrafficAttribution] 请求观察启动失败，已忽略: {type(exc).__name__}", flush=True)

    try:
        response = await call_next(request)
    except Exception:
        if observation is not None:
            try:
                await traffic_attribution.finish_request(
                    observation,
                    endpoint=endpoint,
                    status_code=500,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                )
            except Exception as exc:
                print(f"[TrafficAttribution] 请求观察写入失败，已忽略: {type(exc).__name__}", flush=True)
        raise

    if observation is not None:
        try:
            await traffic_attribution.finish_request(
                observation,
                endpoint=endpoint,
                status_code=response.status_code,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        except Exception as exc:
            print(f"[TrafficAttribution] 请求观察写入失败，已忽略: {type(exc).__name__}", flush=True)

    for name, value in traffic_attribution.ATTRIBUTION_RESPONSE_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


# ── 端点 ──

@app.get("/get_anima_format", response_class=PlainTextResponse)
async def get_anima_format() -> str:
    """返回完整 Anima Hybrid 提示词格式规范，与同名 MCP 工具一致。"""
    return PROMPT_FORMATS["anima"]


@app.get("/get_newbie_format", response_class=PlainTextResponse)
async def get_newbie_format() -> str:
    """返回完整 NewBie XML 提示词格式规范，与同名 MCP 工具一致。"""
    return PROMPT_FORMATS["newbie"]


@app.get("/get_qwen_image_2_1_format", response_class=PlainTextResponse)
async def get_qwen_image_2_1_format(mode: Literal["T2I", "I2I"]) -> str:
    """返回 Qwen-Image-2.1 规范；必填 mode：T2I 文生图，I2I 图生图。"""
    return PROMPT_FORMATS[{"T2I": "qwen_t2i", "I2I": "qwen_i2i"}[mode]]


@app.post("/search", response_model=SearchOut)
@governed("search")
async def search(body: SearchIn, http_request: Request = None) -> SearchOut:
    await telemetry.increment("rest_search")
    traffic_attribution.note_safe_parameters(
        limit=body.limit,
        top_k=body.top_k,
        group_mode=body.group_mode,
        use_segmentation=body.use_segmentation,
    )
    tagger = await DanbooruTagger.get_instance()

    # SearchIn → core.models.SearchRequest（两者字段一一对应，直接解包）
    request = SearchRequest(**body.model_dump())

    # 并发安全的异步 search（信号量串行化 + 线程池执行）
    try:
        response: SearchResponse = await engine_call(tagger, "search", request)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="搜索超时（120s），请简化查询或稍后重试")

    # 旧累计口径继续保留，但接口成功不再冒充真实 UI 复制。
    await counter.increment()
    await counter.increment_success()

    return SearchOut(
        tags_all=response.tags_all,
        tags_sfw=response.tags_sfw,
        results=[TagOut(**vars(result)) for result in response.results],
        keywords=response.keywords,
    )


@app.post("/related")
@governed("related")
async def related(body: RelatedIn, http_request: Request = None) -> dict[str, Any]:
    """
    给定已选标签列表，返回基于共现表的关联推荐。

    - tags：种子标签列表（Danbooru 英文标签名）
    - limit：最多返回条数，默认 50
    - show_nsfw：是否包含 NSFW 标签，默认 True
    - target_categories：仅返回指定类别；未传入时不过滤
    """
    await telemetry.increment("rest_related")
    traffic_attribution.note_safe_parameters(limit=body.limit)
    tagger = await DanbooruTagger.get_instance()
    corrected_tags, invalid_tags, corrections = await _correct_tags(tagger, body.tags)
    if not corrected_tags:
        return {
            "error": "所有传入的标签均不存在于标签表中",
            "invalid_tags": invalid_tags,
            "api_key_notice": API_KEY_NOTICE,
        }
    results = await engine_call(tagger, "get_related",
        corrected_tags,
        set(corrected_tags),
        body.limit,
        body.show_nsfw,
        set(body.target_categories) if body.target_categories is not None else None,
    )
    # 旧累计口径继续保留，但接口成功不再冒充真实 UI 复制。
    await counter.increment()
    await counter.increment_success()

    output: list[dict[str, Any]] = []
    for result in results:
        item = {
            "tag": result.tag,
            "cn_name": result.cn_name,
            "category": result.category,
            "sources": result.sources,
        }
        item["wiki"] = result.wiki
        output.append(item)

    return _with_corrections(output, corrections)


@app.post("/artists")
@governed("artists")
async def artists(body: ArtistIn, http_request: Request = None) -> dict[str, Any]:
    """
    给定标签列表，推荐擅长绘制这些标签的画师（基于 NPMI 共现数据）。

    - tags：种子标签列表（Danbooru 英文标签名）
    - limit：最多返回条数，默认 30
    - min_cooc：单个 (tag, artist) 对的最小共现次数，默认 3
    """
    await telemetry.increment("rest_artists")
    traffic_attribution.note_safe_parameters(limit=body.limit, min_cooc=body.min_cooc)
    tagger = await DanbooruTagger.get_instance()
    if not body.tags:
        return {"error": "tags 列表不能为空", "api_key_notice": API_KEY_NOTICE}

    corrected_tags, invalid_tags, corrections = await _correct_tags(tagger, body.tags)
    if not corrected_tags:
        return {
            "error": "所有传入的标签均不存在于标签表中",
            "invalid_tags": invalid_tags,
            "api_key_notice": API_KEY_NOTICE,
        }

    results = await engine_call(tagger, "search_artists_by_tags",
        corrected_tags, limit=body.limit, min_cooc=body.min_cooc,
    )
    artist_names = [result.artist for result in results]
    top_tags_map = tagger.get_artist_top_tags(artist_names, show_nsfw=body.show_nsfw)
    # 计数
    await counter.increment()
    await counter.increment_success()

    output = [
        {
            "artist": result.artist,
            "cooc_count": result.cooc_count,
            "post_count": result.post_count,
            "sources": result.sources,
            "top_tags": top_tags_map.get(result.artist, []),
        }
        for result in results
    ]
    return _with_corrections(output, corrections)


@app.get("/health")
async def health():
    tagger = await DanbooruTagger.get_instance()
    return {"status": "ok", "loaded": tagger.is_loaded, "api_key_notice": API_KEY_NOTICE}


def key_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["APIKey"] = {
        "type": "http", "scheme": "bearer", "description": "本站签发的 API Key；不是 Hugging Face Token。无 Key 试用请完全不发送 Authorization。"
    }
    for endpoint in ("search", "related", "artists"):
        operation = schema["paths"]["/" + endpoint]["post"]
        operation["security"] = [{}, {"APIKey": []}]
        operation.setdefault("parameters", []).extend([
            {"name": "X-DanbooruSearch-Client", "in": "header", "required": False,
             "schema": {"type": "string"}, "description": "持 Key 调用时须与登记 Client 精确匹配。"},
            {"name": "X-DanbooruSearch-Site", "in": "header", "required": False,
             "schema": {"type": "string"}, "description": "公开业务持 Key 调用时须与登记 HTTPS Site 精确匹配。"},
        ])
    app.openapi_schema = schema
    return schema


app.openapi = key_openapi
