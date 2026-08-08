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
"""

import json
import asyncio
import logging
from anyio import BrokenResourceError, ClosedResourceError
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from core.engine import DanbooruTagger
from core.models import SearchRequest
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
    tags: list[str],
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

    corrected_tags, invalid_tags, corrections, candidates = _resolve_canonical_tags(tagger, tags)

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
    tags: list[str],
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

    corrected_tags, invalid_tags, corrections, candidates = _resolve_canonical_tags(tagger, tags)

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
_ANIMA_FORMAT_INSTRUCTION = """
# Anima Hybrid Prompt Format Specification

请严格按照以下规范，将已确认的 Danbooru 标签和用户描述整理为 Anima Hybrid 提示词。

## 核心原则

Anima 同时理解 Danbooru 标签和自然语言：

- **Hard Tags** 负责人物身份、外观、服装、动作、道具和场景锚点。
- **Natural Language** 负责构图、主体占比、空间关系、光照、曝光和色彩。
- 背景自然语言过多时，模型容易拉远镜头，因此必须同时明确人物大小和背景层级。
- `full body` 只表示身体完整可见，不代表人物会占据主要画面。
- 权重只能强化标签，不能代替构图、主体占比和曝光描述。

不得为了增加画面感，擅自加入用户没有要求的前景、天气、道具、人物、戏剧冲突或复杂光效。

---

## 组装前确认

先确定以下内容：

主体是谁 → 正在做什么 → 视觉焦点是什么 → 背景如何辅助 → 人物如何被照亮

- 有剧情时，只选择一个可见瞬间，不描述连续事件。
- 安静日常场景不强制添加内在冲突、动势或戏剧性光影。
- 动作、视线、道具接触和环境关系必须符合物理逻辑。

---

## 两层 Prompt 结构

### 第一层：Hard Tags

使用用户已提供或经检索确认的逗号分隔标签。

**包含：**

- 质量、年代和安全分级
- 人数、性别、角色、作品
- `@artist`
- 发色、瞳色、发型、体型
- 服装、道具
- 姿势、表情、动作
- 简洁场景锚点，如 `library, bookshelf, window`

**不包含：**

- 完整英文句子
- 未确认或编造的标签
- 主体占比、画面布局和前景限制
- 光源方向、主体曝光和色彩主次
- `warm colors`、`dim lighting` 等自由形式的审美短语

### 第二层：Natural Language

单人物通常使用2～3句简洁、具体、可视化的英文。多人物场景优先为每个角色单独写一句，通常可扩展为3～6句；角色更多时可随人数适度增加，禁止为了满足句数限制把多名角色压进同一个长句。

推荐职责：

1. 第一句：景别、主体占比、整体布局、背景层级、前景限制。
2. 单人物时，第二句说明动作、视线、道具接触和空间关系；多人物时，先按角色分别说明画面位置、身份，以及按可见类别选出的代表性特征。
3. 多人物的角色锚定完成后，再用独立句子说明互动，明确写出动作发起者、承受者、接触对象和视线方向。
4. 最后一句：主光源、主体曝光、色彩主次和景深。

单人物不要机械复述 Hard Tags 中已经明确的外观和服装。多人物为了明确属性归属，每个角色的锚定句必须复述“多人物特征分离规则”要求的各类代表性特征，但不要继续抄写同类别的其余次要 Hard Tags。
背景描述不要只写 `behind the character`、`recedes into the background` 或整体 `softly blurred`；应说明主要环境位于画面的左侧、中央、人物后方或地面，并仅让最远处细节轻微虚化。
多人物场景必须使用“画面方位 + 角色名/身份 + 关键辨识特征”的独立短句建立角色锚点；详细规则见“多人物特征分离规则”。

---

## 构图规则

### 景别与主体大小

景别和主体占比必须分别确定：

- `close-up`、`upper body`、`cowboy_shot`、`full body` 等决定可见范围。
- `dominates the frame`、`occupies most of the frame` 等描述决定人物大小。
- Hard Tags 必须与最终景别一致。用户已有标签即使本身正确，只要对应内容不会出现在当前取景范围内，也应删除，不能为了“保留已有标签”继续放入提示词。
- `close-up`：优先保留脸部、发型、头部饰品和表情，删除画面外的身体、下装、腿部、鞋袜及全身姿势标签。
- `upper body`：保留上半身服装、手臂和画面内可见动作；删除裙子、裤子等下装细节，以及腿部姿势、袜子、长筒袜和鞋子标签。
- `cowboy shot` 或大腿以上取景：可以保留可见的下装，通常删除袜子、鞋子、脚部动作及依赖完整身体的姿势标签。
- `full body`：身体、下装、腿部和鞋袜均实际可见时，才保留对应标签。

参考范围：

| 构图目标 | 主体占画面高度 |
|---|---:|
| 人物主导 | 65%～85% |
| 人景平衡 | 45%～65% |
| 环境主导 | 25%～45% |

不必机械输出百分比，可以写：

- `the character dominates the frame`
- `the seated figure occupies about two-thirds of the frame height`
- `the full body remains large and clearly readable`
- `the background stays secondary and occupies limited space`

### 默认构图

- 用户没有要求环境主视觉时，默认采用人物主导构图。
- 全身图必须完整显示身体、鞋子和必要的承载物，但不得默认缩成远景小人物。
- 坐姿全身图可以显示完整椅子，但人物仍应是主要视觉焦点。
- 已知画幅比例时，构图必须与画幅匹配。横向全身图不得用大量空白、墙壁或黑暗物体填充两侧。
- 宽画幅中人物位于左侧或右侧时，必须使用用户已有的环境元素明确填充另一侧及中间区域；除非用户要求留白或极简构图，不得留下未定义的大面积纯色空间。

### 背景层级

- 只确定一个主要背景锚点，其余元素作为辅助。
- 窗户、太阳、极光、霓虹等高亮背景不得抢过人物。
- 背景复杂时，必须同时说明人物占比和背景的次要地位。
- 背景保持次要只表示视觉优先级低于人物，不表示背景可以缺失、变成纯色或被整体严重虚化；用户明确要求的主要环境仍需清晰可辨。
- 背景元素应位于人物后方，不得无故延伸到镜头前方形成狭窄通道。

### 前景限制

- 不得为了增加空间感自动添加前景遮挡。
- 除非用户明确要求，不得生成大片黑暗前景、严重遮挡、门框式夹景或隧道式构图。
- 前景必须具有明确作用，且不能遮挡脸、手、主要动作或大部分身体。
- 需要空间层次时，优先使用中景、背景、引导线和景深。

---

## 光照、曝光与色彩

光照描述应明确：

```text
主光源 → 光线方向 → 照亮的主体部位 → 背景光作用 → 暗部细节
```

### 主体曝光

- 需要正常可见的人物图，应明确脸、眼睛、手或关键服装被主光照亮。
- 窗户、夕阳、极光、霓虹位于人物背后时，必须说明它是背景光或轮廓光。
- 强背景光场景如果要求人物清晰，应增加正面或侧前方主光、柔和补光。
- 除非用户明确要求剪影，否则人物不能完全落入死黑暗部。
- 必要时使用：
  - `well-exposed subject`
  - `visible facial features`
  - `clear details in the shadows`
  - `no silhouette`

### `dim lighting`

- `dim lighting` 表示低照度，不等于温馨或柔和。
- 只有用户明确要求昏暗、低调光、压抑氛围或剪影时才使用。
- 温馨室内场景优先使用：
  - `soft warm interior light`
  - `gentle warm key light`
  - `cozy ambient lighting`
- 夜景中需要人物清晰时，必须同时指定照亮人物的光源。

### 色彩主次

- 确定一个主色倾向和最多两个辅助色。
- 冷暖对比必须说明哪一方占主导。
- 高饱和背景色不得压过人物肤色、眼睛和服装。
- 避免同时堆叠多个互相竞争的调色词，却不说明主次。
- 用户要求明亮画面时，应明确使用 `bright`、`luminous`、`well-exposed` 或 `clear midtones`。

---

## 标签规则

### 标签顺序

```text
[quality/meta/year/safety] → [人数] → [character] → [series] → [@artist] → [外观/服装/动作/场景标签]
```

多人时，先完整列出人数，再按角色连续排列身份和作品锚点；同一角色的专属外观、服装和道具标签必须保持相邻，不与其他角色的同类属性交叉排列。

### 标签数量

以下数量是推荐的**目标范围**，不是必须满足的硬性上下限：

| 场景复杂度 | 总标签数 |
|---|---:|
| 简单 | 16～30 |
| 标准 | 22～38 |
| 复杂多人或剧情主视觉 | 30～48 |

用户已有标签视为可信标签，通常不需要重新检索，但不代表必须全部保留。应尽量保留与最终画面一致、对人物识别或用户意图有帮助的标签；当已有标签过多时，可以按需删除一部分。

裁剪已有标签时按以下顺序处理：

1. 先删除与最终景别不一致、实际不会出现在画面中的标签。
2. 再删除与人数、动作、视线、姿势、服装状态或安全分级冲突的标签。
3. 再删除宽泛标签、近义重复标签，以及被更准确标签覆盖的标签。
4. 最后删除对人物区分和主要画面贡献较低的次要装饰、身体细节或场景细节。

应优先保留用户强调的内容、人物身份与作品、关键发型和瞳色、主要服装、核心道具、主要表情和动作。不要为了达到目标范围的下限而补充无关标签；如果完成必要裁剪后仍略高于目标范围，可以保留真正重要的标签，不要为了机械计数继续删除。

只保留关键且有区分度的标签：

- 宽泛标签与更准确的具体标签重复时，保留更准确的一项。
- `holding book, open book, reading` 等近义动作按实际需要精简。
- 同一身体部位的细节标签不超过两个。
- 同一概念不在 Hard Tags 和 Natural Language 中机械重复。
- Tag Dropout 意味着不需要列出所有相关标签。

### 格式

- 标签使用小写，空格替换下划线。
- `score_1` 到 `score_9` 保留下划线。
- 标签内括号使用反斜线转义。
- 画师标签必须带 `@`，最多使用三个画师。
- 标签之间使用一个逗号和一个空格。
- 不确定是否存在的标签不得编造，应改写到 Natural Language。
- 必须包含一个安全分级：`safe`、`sensitive`、`nsfw` 或 `explicit`。

### 冲突检查

输出前消解明显冲突：

- `close-up` 与 `full body`
- `from front` 与 `from behind`
- `looking at viewer` 与 `facing away`
- `solo` 与多人互动标签
- `open mouth` 与 `closed mouth`
- `spread fingers` 与 `clenched fist`
- `spread legs` 与 `legs together`
- 完全裸露与具体服装
- 同一动作的多个互斥姿态

### 视线

- 肖像或无明确动作对象时，可以默认看向观众。
- 阅读、工作、睡眠、观察道具等场景，视线必须服从当前动作。
- 用户要求背影、侧脸或看向画外时，以用户要求为准。
- 多人物场景根据互动关系确定视线，不强制看向观众。

---

## 默认前缀

### Anima Base

```text
masterpiece, best quality, score_7, safe,
```

### Anima Aesthetic

```text
masterpiece, best quality, safe,
```

Anima Aesthetic 默认不使用 `score_*`，避免对已完成美学微调的模型施加过强偏置。

### 其他规则

- 模型版本未知时使用 Anima Base 默认前缀。
- `very aesthetic`、`newest`、`year 2025` 不作为强制默认标签。
- 年代标签只在用户要求特定时期或画风时加入。
- Anima Turbo 的 CFG、步数等参数由外部工作流控制，不写入提示词。
- `ye-pop`、`deviantart` 只在用户明确要求对应的非动漫数据集风格时使用。

---

## 权重规则

Anima 支持 Prompt Weighting，但权重只是辅助控制：

- 从 `(tag:2)` 开始。
- 必要时提高到 `(tag:3)`～`(tag:5)`。
- 不得超过5。
- 一段提示词最多强调四个标签。
- 不得使用权重代替主体占比、背景层级和曝光关系。
- `(full body:2)` 只能强化全身可见，不能保证人物足够大。
- 背景导致景别漂移时，应先补全构图描述，再考虑提高景别权重。

---

## 多人物特征分离规则

Anima 的多人生成可以通过清晰的角色边界减少特征混淆，但提示词不能保证彻底消除串色。必须严格遵守：

1. **先声明准确人数与性别构成**：使用 `2girls`、`1girl, 1boy`、`3girls` 等与画面一致的标签；不得同时保留 `solo`，也不得用 `multiple girls` 代替已知的精确人数。
2. **先建立角色身份，再描述互动**：人数之后先写角色名及各自作品名。不要把互动标签插在角色身份之间，也不要只列角色名后立刻进入复杂动作。
3. **Hard Tags 按角色分组**：同一角色的专属发型、瞳色、服装、体型和道具连续出现后再切换到下一角色。严禁把不同角色的同类属性交叉排列，例如 `blue hair, red hair, short hair, long hair`。
4. **每个角色使用独立的 Natural Language 锚定句**：推荐结构为 `On the left side of the image is Character A from Series A, with [按规则5覆盖各可见类别的代表性特征].`；下一角色另起一句。不得把多名角色的外观塞进同一个嵌套长句。
5. **各类别至少一个代表性特征**：每个角色在设定中存在且在当前景别中可见的发型、上衣、下装、鞋袜、道具、姿势类别，各至少写一个代表性特征。某类别本来不存在，或因 `close-up`、`upper body` 等景别不会出现在画面中时，不得为了凑齐类别而编造或保留画面外特征。
6. **空间位置以画面/观众视角为基准并保持稳定**：使用 `on the left side of the image`、`on the right side of the image`、`in the center`、`in the foreground`、`in the background`。不要混用画面左右与角色自身左右；后文不得交换已分配的位置。
7. **互动句必须明确主语和宾语**：完成所有角色锚定后，再写 `Character A holds Character B's right hand`、`Character B looks at Character A` 等。避免连续使用含义不明的 `she`、`he`、`they`，避免笼统的 `interacting`、`together` 代替可见动作。
8. **区分专属属性与共享属性**：专属外观、服装、表情和道具必须放进对应角色的分组或锚定句；两人共有的服装、姿势或环境状态使用 `Both characters...` 单独说明，不得复制成含混的全局属性。
9. **每类只保留少量高区分度特征**：在满足规则5的前提下，每个可见类别通常只选择一个最有辨识度的代表性特征；只有人物识别确实需要时，同一类别才增加第二个。角色相似、人数达到3人以上或互动复杂时，应删除同类别的次要细节和不必要的同时动作，但不得删掉某个实际存在且可见类别的唯一代表性特征。
10. **权重不能代替属性归属**：只有在角色分组和自然语言锚定已经清楚时，才可谨慎强化关键特征。不得仅靠 `(blue hair:2)`、`(red hair:2)` 分离角色，也不得同时堆叠大量高权重属性。
11. **Natural Language 负责明确归属而非完整抄写**：角色锚定句必须复述规则5要求的各类别代表性特征，并补充空间位置、互动动作、光影对象和构图取景；同类别的其余次要 Hard Tags 不再重复。

推荐示意：

```text
2girls, character a, series a, short black hair, blue eyes, white jacket, blue skirt, black boots, shoulder bag, standing, character b, series b, long blonde hair, red eyes, black blouse, red skirt, white boots, suitcase, standing, holding hands, railway station

The image is divided into a left side and a right side, with both characters shown at the same readable scale. On the left side of the image is Character A from Series A, standing with short black hair, a white jacket, a blue skirt, black boots, and a shoulder bag. On the right side of the image is Character B from Series B, standing with long blonde hair, a black blouse, a red skirt, white boots, and a suitcase held in her left hand. Character A holds Character B's right hand while Character B looks at Character A. A soft side light keeps both faces clearly visible while the railway platform remains secondary in the background.
```

---

## 输出格式

````markdown
## Prompt
```
[Hard Tags：逗号分隔，单行]

[Natural Language：单人物2～3句英文；多人物按角色拆句，通常3～6句]
```

## 中文解释

[分点解释实际使用的标签、构图、空间和光照设计，并完整翻译 Natural Language]
````

禁止在规定部分之外添加开场白、寒暄或总结。

---

## 最终自检

输出前确认：

1. 是否只有一个明确景别？
2. 是否说明人物在画面中的大小和主次？
3. 背景是否保持辅助但仍清晰可辨；宽画幅偏置主体时，另一侧是否由用户要求的环境内容合理填充，而非纯色留白？
4. 是否避免无意义前景、大片暗部和隧道式构图？
5. 是否明确主光源照亮人物的哪些部位？
6. 夜景或背光场景是否避免意外剪影？
7. 标签是否去重并尽量落入目标范围；超出目标范围时，是否只保留了与取景、人物识别和用户重点真正相关的标签？
8. 动作、视线、姿势和道具关系是否一致？
9. Natural Language 是否保持职责清晰：单人物2～3句，多人物按角色拆句且没有嵌套长句？
10. 多人物时，人数是否准确，每个角色是否都有稳定的画面位置、独立锚定、明确的动作归属，并为设定中存在且当前景别可见的发型、上衣、下装、鞋袜、道具、姿势类别各保留至少一个代表性特征？
11. 是否只加入用户要求或画面逻辑真正需要的内容？

## 中文解释规则

- 只解释本次提示词中实际采用的设计。
- 说明关键标签、主体占比、背景层级、前景限制和光照主次。
- 多人物时说明角色分组和动作归属。
- 必须完整翻译 Natural Language。
- 使用中立、简洁、技术化的语言。
"""


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
    return _ANIMA_FORMAT_INSTRUCTION


# ── NewBie 提示词格式说明 ─────────────────────────────────────────────────
_NEWBIE_OUTPUT_FORMAT = """
# NewBie XML Prompt Format Specification

## 输出格式要求

你的输出包括两部分：一个 XML 代码块和代码块外的中文翻译。

---

## 情境因果锁（组装前必做）

组装 prompt 前，先建立情境因果链，再拆解为 XML 各字段内容：

```
发生了什么 → 角色的情感/欲望/冲突 → 具体反应（表情+肢体） → 环境如何参与 → 最抓人眼球的画面瞬间
```

- 先定情境，再填充各 XML 字段。
- 情境必须包含因果链：事件起因 → 角色反应 → 可见后果。
- 即使是单人图，也要有内在张力（例：偷穿大衣的体温升高 → 颤抖+脸红+抓衣服）。
- 只选一个最有张力的瞬间，不描述连续剧情。

### 因果可见性

- 每个关键动作必须产生至少一个可见后果。
- 环境事件必须影响角色、道具、服装、头发、表情或构图层次。
- 角色情绪必须落到表情、视线、手势、身体重心或距离变化。
- 手部动作必须明确接触对象、接触位置和结果。
- 天气/季节不能只写 tag，必须落到可见物理效果。
- 看不见后果的动作不写；无法明确归属的动作改写进 `<caption>`。

---

## 标签处理规则

- 标签内部的空格必须替换为下划线 `_`（如 `red eyes` → `red_eyes`）
- 标签名内的括号必须用反斜杠转义（如 `momoko (momopoco)` → `momoko_\\(momopoco\\)`）
- 权重括号（如 `(daito:1.2)`）保持原样，不转义
- 括号内包含多个独立标签时，拆解为独立标签

---

## XML 结构

```xml
<img>
 <character_1>
  <n>角色名</n>
  <gender>性别标签 (如 1girl)</gender>
  <appearance>外貌特征 (发色, 瞳色, 身体特征等)</appearance>
  <clothing>衣着 (具体服饰)</clothing>
  <expression>表情</expression>
  <action>动作</action>
  <position>位置</position>
 </character_1>

 <!-- 若有多个角色，按 character_2, character_3 顺延 -->

 <general_tags>
  <count>人数标签</count>
  <style>画风标签（若用户未指定，默认 anime_style,realistic_shading）</style>
  <background>背景标签</background>
  <atmosphere>画面情绪、氛围标签</atmosphere>
  <quality>very_aesthetic, masterpiece, no_text</quality>
  <resolution>max_high_resolution</resolution>
  <artist>画师标签</artist>
  <objects>各种物品（包括武器、饰品等）</objects>
  <other>其它标签</other>
 </general_tags>

 <caption>
  将所有标签串联为一段流畅、详细的英文场景描述。包含光线、情绪、角色和背景。
  不要在此处提及 style 或 quality 类词汇。
 </caption>
</img>
```

在 XML 代码块结束后，输出 `<caption>` 内容的中文翻译。

---

## XML 字段职责划分

### character_N 块（离散标签层）

负责角色的结构化属性，使用 Danbooru 标签格式：

- `<n>`：角色名（经检索确认的 canonical name）
- `<gender>`：人数/性别标签
- `<appearance>`：发色、瞳色、发型、体型等外观特征（经检索确认）
- `<clothing>`：服装、配饰（经检索确认）
- `<expression>`：表情标签
- `<action>`：动作/姿势标签
- `<position>`：空间位置（left/right/foreground/background）

### general_tags 块（画面全局标签）

负责画面整体的结构化属性：

- `<count>`：人数标签
- `<style>`：画风标签
- `<background>`：场景/背景标签
- `<atmosphere>`：氛围/情绪标签
- `<quality>`：质量标签
- `<resolution>`：分辨率标签
- `<artist>`：画师标签
- `<objects>`：道具/物品标签
- `<other>`：其他标签

### caption 块（空间叙事层）

负责 hard tags 难以精确表达的内容，使用自然语言：

**包含：**
- 镜头取景：angle, shot distance, framing
- 光线：方向、质感、色温
- 色彩调性：palette, color grading
- 空间布局：角色间的位置关系、前后层次
- 多角色动作归属与互动
- 手和道具的精确接触关系
- 因果链的可见后果
- 景深、虚化、清晰区域

**规则：**
- 流畅的英文段落，不是标签列表。
- 不重复 character_N 和 general_tags 中已出现的标签内容。
- 不写 style 或 quality 类词汇。
- 使用客观、具体、视觉化的描述。

---

## 八维补全检查（输出前必做）

组装完成后，自查以下 8 个维度，**至少触发 3 维以上**。缺失的维度用 `<caption>` 补全，不硬塞更多标签。

| 维度 | 检查问题 | 缺失表现 | 补全方向 |
|------|----------|----------|----------|
| **互动** | 元素之间有无行为联系？ | 各自独立摆 pose，零交集 | 对视、触碰、动作呼应、人与环境互动 |
| **情感** | 表情+肢体传递了什么情绪？ | generic smile / 面无表情 | 微表情、身体语言（前倾/缩肩/攥拳） |
| **视线** | 目光或引导线指向哪里？ | 所有人看镜头或闭眼 | 角色间对视、偷瞄、看向画外某物 |
| **联动** | 环境是否影响主体？ | 环境是纯背景装饰 | 风雨→反应、光线→塑型、材质受环境影响 |
| **动势** | 冻结画面暗示了运动吗？ | 像摆拍立绘，重心正中 | 重心偏移、布料飞扬、头发飘动、失衡感 |
| **空间** | 有前后层次和呼吸感吗？ | 平铺直叙，贴脸输出 | 前景遮挡、景深虚化、正负空间、引导线 |
| **质感** | 材质有真实细节吗？ | 塑料感/卡通化 | 湿润反光、粗糙纹理、丝滑垂坠、水珠凝结 |
| **因果** | 观众能看出前因后果吗？ | 不知道在发生什么 | 行为起因→当前姿态→暗示后续 |

**规则：**
- 补全内容必须服务于已有情境因果链，不能凭空插入无关元素。
- 单人图：互动维转为「主体与环境的互动」（风吹头发、踩水溅起、光影打在脸侧）。
- `<caption>` 是补全八维的主要载体，character_N 和 general_tags 维持结构化标签干净。

---

## 冲突检查（输出前必做）

组装前必须消解以下冲突，逐项通过后才输出：

| 冲突对 | 规则 |
|--------|------|
| `solo` vs 多人 | 选一个，不共存 |
| `close-up` vs `full body` | 选一个景别 |
| `from above` vs `from below` | 选一个视角 |
| `from front` vs `from behind` | 选一个朝向 |
| `closed eyes` vs `looking at viewer` | 选一个视线 |
| 裸体 vs 服装 | 选一个着装状态 |
| 多角色属性归属 | 发色/服装必须绑定具体角色，不串 |
| 室内光源 vs 室外背景 | 光源和背景必须同空间 |
| 背光 | 必须补脸部补光或轮廓保护 |

单人正面默认保护脸部：保留 `looking at viewer` 或 `facing viewer`，`<caption>` 补一句脸部清晰。

多人必须在 `<position>` 和 `<caption>` 中明确空间方位。

---

## 多人物规则（防特征混淆）

如果用户提到了多个人物，必须严格遵循以下规则：

1. **角色分组**：每个 character_N 块内连续排列该角色的所有专属属性（发型、瞳色、服装、体型、表情、动作），然后再切换到下一角色。
2. **外观标签充分**：每个角色至少 5 个角色特征标签。可使用 `get_related_tags` 获得更多特征。
3. **属性不交叉**：禁止将不同角色的同类属性交叉排列。不同角色的特征混淆是多人场景最常见的失败模式。
4. **空间锚定**：在 `<position>` 和 `<caption>` 中明确每个角色的空间位置（如"左侧"、"右侧"、"前景"等）。
5. **caption 角色锚定**：在 `<caption>` 中为每个角色写一句外观锚定短语，使用"[角色名] with [关键特征]"的句式，明确指出视觉归属。
6. **caption 中不重复标签内容**——`<caption>` 补充空间关系、互动动作、光影氛围、构图取景。

---

## 默认值

**质量标签**（无特殊要求时的默认值）：
```xml
<quality>very_aesthetic, masterpiece, no_text</quality>
<resolution>max_high_resolution</resolution>
```

**画风标签**（用户未指定时的默认值）：
```xml
<style>anime_style, realistic_shading</style>
```

**取景默认**：若用户未指定，默认近景人物、人物面向观众。若用户有描述则以用户描述为准。

---

## 中文翻译规则

在 XML 代码块结束后，输出 `<caption>` 内容的完整中文翻译。
"""


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
    return _NEWBIE_OUTPUT_FORMAT
