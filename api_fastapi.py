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
from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.engine import DanbooruTagger
from core.models import SearchRequest, SearchResponse, TagResult, RelatedTag


# ── Pydantic I/O 模型（API 层专用，与 core.models 解耦）──

class SearchIn(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)
    limit: int = Field(80, ge=1, le=500)
    popularity_weight: float = Field(0.15, ge=0.0, le=1.0)
    show_nsfw: bool = True
    use_segmentation: bool = True
    target_layers: list[str] = ['英文', '中文扩展词', '释义', '中文核心词']
    target_categories: list[str] = ['General', 'Character', 'Copyright']


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


class RelatedIn(BaseModel):
    tags: list[str]
    limit: int = Field(50, ge=1, le=200)
    show_nsfw: bool = True


class RelatedTagOut(BaseModel):
    tag: str
    cn_name: str
    category: str
    nsfw: str
    cooc_count: int
    cooc_score: float
    sources: list[str]


class SearchOut(BaseModel):
    tags_all: str
    tags_sfw: str
    results: list[TagOut]
    keywords: list[str]


# ── FastAPI 子应用（挂载到 NiceGUI 的 /api 路径下）──
# lifespan / 预热由 ui_nicegui.py 的 @app.on_startup 统一管理，此处不重复。
app = FastAPI(
    title="Danbooru Tag Searcher API",
    description="通过 /api/docs 查看完整接口文档。",
    version="1.0.0",
)


# ── 端点 ──

@app.post("/search", response_model=SearchOut)
async def search(body: SearchIn) -> SearchOut:
    tagger = await DanbooruTagger.get_instance()

    # SearchIn → core.models.SearchRequest（两者字段一一对应，直接解包）
    request = SearchRequest(**body.model_dump())

    # 在线程池中运行阻塞的 search()
    response: SearchResponse = await asyncio.to_thread(tagger.search, request)

    return SearchOut(
        tags_all=response.tags_all,
        tags_sfw=response.tags_sfw,
        results=[TagOut(**vars(r)) for r in response.results],
        keywords=response.keywords,
    )


@app.post("/related", response_model=list[RelatedTagOut])
async def related(body: RelatedIn) -> list[RelatedTagOut]:
    """
    给定已选标签列表，返回基于共现表的关联推荐。

    - tags：种子标签列表（Danbooru 英文标签名）
    - limit：最多返回条数，默认 50
    - show_nsfw：是否包含 NSFW 标签，默认 True
    """
    tagger = await DanbooruTagger.get_instance()
    results = await asyncio.to_thread(
        tagger.get_related,
        body.tags,
        set(body.tags),   # exclude 已选标签自身
        body.limit,
        body.show_nsfw,
    )
    return [
        RelatedTagOut(
            tag=r.tag,
            cn_name=r.cn_name,
            category=r.category,
            nsfw=r.nsfw,
            cooc_count=r.cooc_count,
            cooc_score=r.cooc_score,
            sources=r.sources,
        )
        for r in results
    ]


@app.get("/health")
async def health():
    tagger = await DanbooruTagger.get_instance()
    return {"status": "ok", "loaded": tagger.is_loaded}
