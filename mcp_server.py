"""
mcp_server.py
─────────────
MCP 服务层

挂载方式（在 ui_nicegui.py 中）：
    from mcp_server import mcp
    app.mount('/mcp', mcp.streamable_http_app())

接入地址：
    https://sakizuki-danboorusearch.hf.space/mcp/mcp

支持的工具：
    search_tags        自然语言搜索标签
    get_related_tags   基于共现表查关联推荐
    get_artist_profile 查询单个画师常见共现标签
    get_anima_format   返回 Anima 模型 Hybrid 提示词格式规范
    get_newbie_format  返回 NewBie 模型 XML 提示词格式规范
    get_qwen_image_2_1_format
                       返回 Qwen-Image-2.1 模型的 T2I/I2I 提示词格式规范
"""

import json
import asyncio
import logging
from typing import Annotated
from pydantic import Field
from anyio import BrokenResourceError, ClosedResourceError
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from core.engine import DanbooruTagger, run_in_slot
from core.models import MAX_INPUT_TAGS, MAX_INPUT_TAG_LENGTH, SearchRequest
from core.prompt_formats import PROMPT_FORMATS
from core.runtime_diagnostics import install_mcp_diagnostics
import core.counter as counter
import core.telemetry as telemetry
import re


# ── 过滤客户端断连/超时产生的无害报错噪音 ──────────────────────────────
class _SuppressClientDisconnect(logging.Filter):
    _SUPPRESSED: tuple = ()
    _HAS_STARLETTE: bool = False

    @classmethod
    def _init_suppressed(cls):
        if cls._SUPPRESSED:
            return
        types: list = [BrokenResourceError, ClosedResourceError, asyncio.CancelledError]
        try:
            from starlette.requests import ClientDisconnect
            types.append(ClientDisconnect)
            cls._HAS_STARLETTE = True
        except ImportError:
            pass
        cls._SUPPRESSED = tuple(types)

    def filter(self, record: logging.LogRecord) -> bool:
        self._init_suppressed()
        exc = record.exc_info[1] if record.exc_info else None
        if isinstance(exc, self._SUPPRESSED):
            return False
        # 用类名字符串兜底（避免 starlette 版本差异导致 import 失败）
        if exc is not None and not self._HAS_STARLETTE:
            name = type(exc).__name__
            if name in ('ClientDisconnect',):
                return False
        return True


_disconnect_filter = _SuppressClientDisconnect()
logging.getLogger("mcp.server.streamable_http").addFilter(_disconnect_filter)
logging.getLogger("mcp.server").addFilter(_disconnect_filter)
logging.getLogger("uvicorn.error").addFilter(_disconnect_filter)


# ── 过滤 Trae 非标准会话结束通知产生的校验噪音 ─────────────────────────────
class _SuppressTraeSessionStopNoise(logging.Filter):
    _MARKERS = (
        "Failed to validate notification",
        "notifications/trae/session_stop",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not all(marker in message for marker in self._MARKERS)


# MCP 依赖在 shared.session 中直接使用根 logger 记录此校验警告。
logging.getLogger().addFilter(_SuppressTraeSessionStopNoise())


mcp = FastMCP(
    name="danbooru-searcher",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
install_mcp_diagnostics(mcp._mcp_server)


def _resolve_canonical_tags(tagger: DanbooruTagger, tags: list[str]) -> tuple[list[str], list[str], dict[str, str], dict[str, list[str]]]:
    """轻量解析 canonical tag 名，不调用语义搜索。"""
    resolved_tags: list[str] = []
    invalid_tags: list[str] = []
    corrections: dict[str, str] = {}
    candidates: dict[str, list[str]] = {}

    for raw_tag in tags:
        resolved = tagger.resolve_tag_name(raw_tag)
        tag = resolved.get("tag")
        if tag:
            resolved_tags.append(tag)
            if tag != raw_tag:
                corrections[raw_tag] = tag
            continue

        invalid_tags.append(raw_tag)
        if resolved.get("candidates"):
            candidates[raw_tag] = resolved["candidates"]

    return resolved_tags, invalid_tags, corrections, candidates


@mcp.tool()
async def search_tags(
    query: str,
    search_mode: str = "full_scene",
    category: str = "all",
    show_nsfw: bool = True,
    include_wiki: bool = False,
) -> str:
    """
使用自然语言搜索 Danbooru 视觉标签、角色标签、作品标签，并返回可直接用于提示词的 tag 列表。

本工具适合搜索可见画面内容：主体、服装、姿势、动作、表情、背景、构图、角色名、作品名等。

不要用本工具搜索画师名、画师风格、creator/artist lookup，也不要用它验证某个画师标签是否存在。
遇到 "Mika Pikazo style"、"画师 mika_pikazo"、"by redjuice"、"这个画师常画什么" 这类请求时，
应改用 get_artist_profile。若用户同时给出画师/风格参考和可见画面描述，只把可见画面描述交给
search_tags，不要把画师名放进 query。

## 参数
- query: 自然语言画面描述。推荐使用中文。
- search_mode: 搜索策略。**默认是 "full_scene"；除非用户明确想探索多种候选，否则保持默认。**
    "full_scene"       — **默认。** 用户给出具体画面描述时使用：场景、主体、服装、姿势、动作、
                         背景等，不管描述多长、元素多少。用户想要的是一张图的一组连贯提示词。
                         (e.g. "一个穿着白色水手服的少女在雨中奔跑", "金发双马尾女孩坐在教室窗边看书，夕阳",
                              "芙兰朵露 金发 辫子 发带 连衣裙 围裙 灯笼裤")
    "concept_explore"  — **只用于开放式概念浏览。** 当用户想看某个模糊/单一概念有哪些类型、
                         想从大量候选中挑选时使用。会返回最多 80 个候选，token 成本较高。
                         不要因为描述元素多就使用此模式；详细场景仍然属于 "full_scene"。
                         (e.g. "各种各样的汉服", "兔耳朵都有哪些", "赛博朋克服装有什么风格")
    "subject_describe" — **只用于描述一个单一视觉概念。** 此模式关闭分词，不能解析多元素 query。
                         如果 query 包含角色名 + 属性、多个服装物件、或任何组合场景，应使用
                         "full_scene"。
                         适合："EVA中蓝发的驾驶员"（单一角色概念）、"灯笼裤"（单一物件）、
                         "两侧有开口，前方有拉绳的运动短裤"（带细节的单一物件）。
    "precise_lookup"   — 中文或英文的单一概念精确查词 / 拼写纠错，例如“水手服”、
                         "selafuku"、"thighhigh"。仍使用语义搜索；官方 Tag Alias 仅用于
                         将召回的废弃标签规范化为当前标签。
- 判断规则：用户是想得到一张具体图的提示词（→ full_scene），还是想浏览某个概念的多种候选
  （→ concept_explore）？元素数量不是判断依据，探索意图才是。
- 重要：只要 query 是具体场景、多元素组合、角色 + 属性，就用 "full_scene"。拿不准时也用
  "full_scene"，它能处理具体画面描述。
- category: 限定搜索类别。默认 "all"。
    "all"       — 全部（通用 + 作品 + 角色）
    "general"   — 可见属性、服装、姿势、背景等通用标签
    "character" — 角色标签
    "copyright" — 动画/游戏/作品名等版权标签
- show_nsfw: 是否包含 NSFW 标签。默认 True。
- include_wiki: 是否在结果中附带 wiki 说明。默认 False。
    当标签含义不熟悉、需要消歧时设为 True。

## query 写法建议

可以使用**空格、换行、中文逗号（，）、顿号（、）**手动分隔概念。
被分隔符包围且长度不超过 7 个汉字的片段会尽量保持原子性，搜索引擎会尊重你的拆分意图。

| 写法 | 示例 |
|---|---|
| 空格分隔概念 | `运动社团 校队 比赛 运动会` |
| 顿号分隔概念 | `反乌托邦、赛博朋克、蒸汽朋克` |
| 自然句子 | `一个穿着白色水手服的少女在雨中奔跑` |
| 混合写法 | `运动社团 一个穿水手服的少女` |

## 工作流

调用 search_tags 后，可以把选中的标签传给 get_related_tags，通过共现关系发现互补标签。
可按 search_tags → get_related_tags → get_related_tags → search_tags 多跳探索。

## 返回

JSON 对象，包含 prompt（逗号分隔 tag）、keywords、results。
每个 result 包含 tag、cn_name；搜索结果经官方 Alias 规范化时包含 alias_from；
当 include_wiki=True 时额外包含 wiki。
    """
    await telemetry.increment("mcp_search_tags")
    _SEARCH_MODE_PRESETS: dict[str, dict] = {
        "precise_lookup":   {"top_k": 10, "limit": 10, "popularity_weight": 0.15, "use_segmentation": False, "group_mode": "off",    "max_per_group": 2},
        "concept_explore":  {"top_k": 80, "limit": 80, "popularity_weight": 0.15, "use_segmentation": True,  "group_mode": "expand",  "max_per_group": 2},
        "subject_describe": {"top_k": 20, "limit": 20, "popularity_weight": 0.15, "use_segmentation": False, "group_mode": "off",    "max_per_group": 2},
        "full_scene":       {"top_k": 5,  "limit": 80, "popularity_weight": 0.15, "use_segmentation": True,  "group_mode": "diverse", "max_per_group": 2},
    }
    preset = _SEARCH_MODE_PRESETS.get(search_mode, _SEARCH_MODE_PRESETS["full_scene"])

    _CATEGORY_MAP: dict[str, list[str]] = {
        "all":       ["General", "Character", "Copyright", "Artist", "Meta"],
        "general":   ["General"],
        "character": ["Character"],
        "copyright": ["Copyright"],
    }
    target_categories = _CATEGORY_MAP.get(
        category,
        _CATEGORY_MAP["all"],
    )

    tagger = await DanbooruTagger.get_instance()
    request = SearchRequest(
        query=query,
        top_k=preset["top_k"],
        limit=preset["limit"],
        popularity_weight=preset["popularity_weight"],
        show_nsfw=show_nsfw,
        use_segmentation=preset["use_segmentation"],
        target_categories=target_categories,
        group_mode=preset["group_mode"],
        max_per_group=preset["max_per_group"],
    )
    try:
        response = await tagger.search_async(request)
    except asyncio.TimeoutError:
        return json.dumps({
            "error": "搜索超时（120s），请简化查询或稍后重试",
        }, ensure_ascii=False, indent=2)
    # 旧累计口径继续保留，但 MCP 成功不再冒充真实 UI 复制。
    await counter.increment()
    await counter.increment_success()
    await counter.increment_mcp()

    results = []
    for r in response.results:
        if r.nsfw == '1' and not show_nsfw:
            continue
        item = {
            "tag":         r.tag,
            "cn_name":     r.cn_name,
        }
        if r.alias_from:
            item["alias_from"] = r.alias_from
        if include_wiki:
            item["wiki"] = r.wiki
        results.append(item)

    payload = {
        "prompt":   response.tags_sfw if not show_nsfw else response.tags_all,
        "keywords": response.keywords,
        "results":  results,
    }
    han_chars = re.findall(r'[\u4e00-\u9fff]', query)
    if len(query) > 0 and len(han_chars) / len(query) < 0.5:
        payload["hint"] = (
            "检测到英文查询，该搜索引擎对中文查询优化更好，如果搜索结果不合预期，推荐用中文重试"
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_related_tags(
    tags: Annotated[list[Annotated[str, Field(max_length=MAX_INPUT_TAG_LENGTH)]], Field(max_length=MAX_INPUT_TAGS)],
    limit: int = 50,
    show_nsfw: bool = True,
    include_wiki: bool = False,
) -> str:
    """
根据已给定的 Danbooru 标签列表，返回基于 NPMI 共现评分的关联标签推荐。
本工具只支持通用标签、作品标签、角色标签；**不支持画师标签和 meta 标签。**

不要用本工具搜索画师名、画师风格、creator/artist lookup，也不要用它验证某个画师标签是否存在。
如果用户询问某个具体画师常画什么，或询问画师风格参考，应使用 get_artist_profile。

本工具会找出在 Danbooru 中经常与种子标签共同出现的标签。结果会按设计混合
General / Character / Copyright 类别。

## 典型用法

- 属性 → 拥有该属性的角色
  例如 ["fingerless_gloves"] → tifa_lockhart, cammy_white, bridget_(guilty_gear), ...
- 作品 → 作品中的角色
  例如 ["overlord_(maruyama)"] → shalltear_bloodfallen, ainz_ooal_gown, albedo_(overlord), ...
- 角色 → 该角色常见视觉属性
  例如 ["amiya_(arknights)"] → 服装、表情、配饰等
- 主题探索
  例如 ["fighter_jet"] → 飞机类型、动作、背景等
- 多标签交集
  例如 ["maid", "twintails"] → 与该组合强相关的标签，按聚合 NPMI 评分排序

如果要做同类别内部探索，例如“更多类似 X 的服装标签”，请使用 search_tags 并设置 category。

## 工作流

可按 search_tags → get_related_tags → get_related_tags → search_tags 链式调用。
沿共现图多跳探索时，可以发现单纯语义搜索不容易召回的标签。

## 参数

- tags: canonical Danbooru tag 名列表，使用下划线，不使用空格。
        最多 128 个标签，每个标签最多 256 个字符。
        例如 ["white_serafuku", "sailor_collar"]
- limit: 最多返回的推荐数量。默认 50。
- show_nsfw: 是否包含 NSFW 标签。默认 True。
- include_wiki: 是否在结果中附带 wiki 说明。默认 False。
        当结果标签不熟悉、需要消歧时设为 True。

## 返回

JSON 对象，results 按聚合 NPMI 分数降序排序。每个结果包含：
- tag, cn_name
- sources: 对该推荐有贡献的种子标签
- wiki: 仅当 include_wiki=True 时返回
    """
    await telemetry.increment("mcp_get_related_tags")
    tagger = await DanbooruTagger.get_instance()

    corrected_tags, invalid_tags, corrections, candidates = await run_in_slot(
        DanbooruTagger._get_recommendation_sem(),
        lambda: asyncio.to_thread(_resolve_canonical_tags, tagger, tags),
    )

    if not corrected_tags:
        payload = {
            "error": "所有传入的标签均不存在于标签表中",
            "invalid_tags": invalid_tags,
        }
        if candidates:
            payload["candidates"] = candidates
        return json.dumps(payload, ensure_ascii=False, indent=2)

    results = await tagger.get_related_async(
        corrected_tags,
        set(corrected_tags),
        limit,
        show_nsfw,
    )
    # 旧累计口径继续保留，但 MCP 成功不再冒充真实 UI 复制。
    await counter.increment()
    await counter.increment_success()
    await counter.increment_mcp()

    output = []
    for r in results:
        item = {
            "tag":        r.tag,
            "cn_name":    r.cn_name,
            "sources":    r.sources,
        }
        if include_wiki:
            item["wiki"] = r.wiki
        output.append(item)

    payload = {"results": output}
    if corrections:
        correction_notes = [
            f"{bad} → {good}" for bad, good in corrections.items()
        ]
        payload = {
            "correction_note": "标签拼写错误，已经纠错: " + ", ".join(correction_notes),
            "corrections": corrections,
            "results": output,
        }

    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_artist_recommendations(
    tags: Annotated[list[Annotated[str, Field(max_length=MAX_INPUT_TAG_LENGTH)]], Field(max_length=MAX_INPUT_TAGS)],
    limit: int = 30,
    min_cooc: int = 3,
    show_nsfw: bool = True,
) -> str:
    """
    根据标签-画师 NPMI 共现数据，推荐擅长绘制给定标签的画师。

    输入一组 canonical Danbooru 标签（例如角色名、服装、主题、视觉元素），本工具会返回作品中
    经常与这些标签共同出现的画师，并按聚合 NPMI 分数排序。

    本工具用于 tag → artist 推荐。输入必须是 canonical Danbooru tag 名，不是画师名。
    不要用本工具查询某个具体画师；画师 → 常见标签应使用 get_artist_profile。

    ## 参数
    - tags: canonical Danbooru tag 名列表，使用下划线，不使用空格。
            最多 128 个标签，每个标签最多 256 个字符。
            例如 ["1girl", "blue_hair", "school_uniform"]
    - limit: 最多返回的画师数量。默认 30。
    - min_cooc: 单个 (tag, artist) 组合进入计算所需的最小共现次数。默认 3。
    - show_nsfw: 是否包含 NSFW 画师数据。默认 True。

    ## 返回

    JSON 对象，results 按 NPMI 分数降序排序。每个结果包含：
    - artist: Danbooru 画师 tag 名
    - cooc_count: 所有输入标签上的累计共现次数
    - post_count: 该画师在 Danbooru 的作品数
    - sources: 命中该画师的输入标签
    - top_tags: 该画师最常画的前 10 个标签（带中文名）
    """
    await telemetry.increment("mcp_get_artist_recommendations")
    tagger = await DanbooruTagger.get_instance()

    if not tags:
        return json.dumps({"error": "tags 列表不能为空"}, ensure_ascii=False, indent=2)

    corrected_tags, invalid_tags, corrections, candidates = await run_in_slot(
        DanbooruTagger._get_recommendation_sem(),
        lambda: asyncio.to_thread(_resolve_canonical_tags, tagger, tags),
    )

    if not corrected_tags:
        payload = {
            "error": "所有传入的标签均不存在于标签表中",
            "invalid_tags": invalid_tags,
        }
        if candidates:
            payload["candidates"] = candidates
        return json.dumps(payload, ensure_ascii=False, indent=2)

    results = await tagger.search_artists_by_tags_async(
        corrected_tags, limit=limit, min_cooc=min_cooc,
    )

    # 获取每个画师最常画的标签
    artist_names = [r.artist for r in results]
    top_tags_map = tagger.get_artist_top_tags(artist_names, show_nsfw=show_nsfw)

    output = []
    for r in results:
        item = {
            "artist":     r.artist,
            "cooc_count": r.cooc_count,
            "post_count": r.post_count,
            "sources":    r.sources,
            "top_tags":   top_tags_map.get(r.artist, []),
        }
        output.append(item)

    # 计数
    await counter.increment()
    await counter.increment_success()
    await counter.increment_mcp()

    payload = {"results": output}
    if corrections:
        correction_notes = [
            f"{bad} → {good}" for bad, good in corrections.items()
        ]
        payload = {
            "correction_note": "标签拼写错误，已经纠错: " + ", ".join(correction_notes),
            "corrections": corrections,
            "results": output,
        }

    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_artist_profile(
    artist_name: str,
    top_n: int = 20,
    show_nsfw: bool = True,
) -> str:
    """
在画师-标签共现数据库中查询单个 Danbooru 画师，并返回该画师常见共现标签。

当用户询问某个具体画师或画师风格参考时使用本工具，例如：
"Mika Pikazo style"、"画师 mika_pikazo"、"by redjuice"、"这个画师常画什么"。
本工具查询的是画师数据库，不是普通视觉 tag 搜索索引。

画师名会在查询前自动规范化。因此，当数据库中存在 "mika_pikazo" 时，
"Mika Pikazo"、"mika pikazo"、"mika_pikazo"、"MikaPikazo" 都可以解析到它。

## 参数
- artist_name: 画师名或 Danbooru 画师 tag。允许大小写差异和空格。
- top_n: 最多返回的常见标签数量。默认 20。
- show_nsfw: 是否包含 NSFW 常见标签。默认 True。

## 返回

JSON 对象，包含：
- artist: 解析后的 canonical Danbooru 画师 tag
- input: 原始输入
- matched_by: 匹配方式，可能是 exact / normalized_exact / compact_exact / fuzzy
- post_count: 该画师在共现数据库中的作品数
- top_tags: 常见共现标签列表，每项只包含 tag 和 cn_name
- note: 说明这些常见标签只能作为风格参考，不等于完整画风语义描述

如果没有找到唯一画师，会返回 artist_not_found 和候选画师名。这不代表该画师 tag 在 Danbooru
不存在，也不要改用 search_tags 验证画师名。
    """
    await telemetry.increment("mcp_get_artist_profile")
    tagger = await DanbooruTagger.get_instance()
    profile = tagger.get_artist_profile(
        artist_name,
        top_n=max(1, min(int(top_n), 100)),
        show_nsfw=show_nsfw,
    )

    await counter.increment()
    await counter.increment_mcp()
    if "error" not in profile:
        await counter.increment_success()

    return json.dumps(profile, ensure_ascii=False, indent=2)


# ── Anima 提示词格式说明 ─────────────────────────────────────────────────


@mcp.tool()
async def get_anima_format() -> str:
    """
    返回 Anima 文生图模型的 Hybrid 混合提示词格式规范。

    当用户提到「Anima 提示词」「Anima 格式」「Anima Prompt」「Anima 模型」等关键词时，
    应调用此工具，以获取完整的提示词组装规范。

    ## 适用场景

    - 用户明确要求输出 Anima 模型的提示词
    - 用户提到 anima、Anima 等关键词
    - 需要将标签转换为 Anima 的 Hybrid 混合格式

    ## Returns

    包含完整 Anima 提示词格式规范的 Markdown 文本，涵盖标签格式化规则、
    自然语言段落规则、权重语法、多人物防串扰规则等。
    """
    await telemetry.increment("mcp_get_anima_format")
    return PROMPT_FORMATS["anima"]


# ── NewBie 提示词格式说明 ─────────────────────────────────────────────────


@mcp.tool()
async def get_newbie_format() -> str:
    """
    返回 NewBie 文生图模型的 XML 格式提示词规范。

    当用户提到「NewBie 提示词」「NewBie 格式」「NewBie Prompt」「NewBie 模型」等关键词时，
    应调用此工具，以获取完整的 XML 格式组装规范。

    ## 适用场景

    - 用户明确要求输出 NewBie 模型的提示词
    - 用户提到 newbie、NewBie 等关键词
    - 需要将标签转换为 NewBie 的 XML 格式

    ## Returns

    包含完整 NewBie 提示词格式规范的文本，涵盖 XML 结构、标签处理规则、多人物规则等。
    """
    await telemetry.increment("mcp_get_newbie_format")
    return PROMPT_FORMATS["newbie"]


# ── Qwen-Image-2.1 提示词格式说明 ────────────────────────────────────────


@mcp.tool()
async def get_qwen_image_2_1_format(mode: str) -> str:
    """
    获取 Qwen-Image-2.1 模型的提示词格式规范。

    ## 参数
    - mode: 必须是 "T2I" 或 "I2I"。T2I 返回文生图格式，I2I 返回图生图格式。

    ## 返回
    返回对应模式的完整 Qwen-Image-2.1 提示词格式文本。
    如果 mode 不是 "T2I" 或 "I2I"，返回错误提示。
    """
    await telemetry.increment("mcp_get_qwen_image_2_1_format")

    prompt = PROMPT_FORMATS.get({"T2I": "qwen_t2i", "I2I": "qwen_i2i"}.get(mode))
    if prompt is None:
        return '错误：mode 必须是 "T2I" 或 "I2I"。'

    return prompt
