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
    "precise_lookup"   — 精确查词 / 拼写纠错，例如 "selafuku"、"thighhigh"。
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
每个 result 包含 tag、cn_name；当 include_wiki=True 时额外包含 wiki。
    """
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
    # 计数：每次 MCP 搜索调用均计入搜索、成功、复制；访问不变
    await counter.increment()
    await counter.increment_success()
    await counter.increment_copy()
    await counter.increment_mcp()

    results = []
    for r in response.results:
        if r.nsfw == '1' and not show_nsfw:
            continue
        item = {
            "tag":         r.tag,
            "cn_name":     r.cn_name,
        }
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
    # 计数：每次 MCP related 调用均计入搜索、成功、复制；访问不变
    await counter.increment()
    await counter.increment_success()
    await counter.increment_copy()
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
    await counter.increment_copy()
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
        await counter.increment_copy()

    return json.dumps(profile, ensure_ascii=False, indent=2)


# ── Anima 提示词格式说明 ─────────────────────────────────────────────────
_ANIMA_FORMAT_INSTRUCTION = """
# Anima Hybrid Prompt Format Specification

请严格按以下 Anima 混合提示词（Hybrid Prompt）规范，基于提供的标签和用户描述，输出最终结果。

## Overview

将已有的 Danbooru 风格标签数据整合为 Anima 模型的最优 Hybrid 提示词。该 Skill 假定调用方已经拥有充足的标签信息（通过 Tagger、Captioner 或用户输入），仅负责按 Anima 的格式规范与社区验证的最佳实践进行结构化组装。

Anima 是一个 2B 参数的文生图模型（CircleStone Labs × Comfy Org），基于 NVIDIA Cosmos-Predict2-2B，使用 Qwen 3 0.6B 文本编码器。它同时理解 Danbooru 标签和自然语言，但两者的行为有本质差异——标签掌控结构与精度，自然语言掌控氛围与构图。

社区的共识结论：
- **纯标签提示词**：线条锐利、色彩平整、几乎没有解剖错误，但画面扁平，缺乏光影、氛围、构图的精确控制。
- **纯自然语言提示词**：细节丰富、光影动态、气氛到位，但超过 2~3 段后结构崩塌，手部最先出问题。
- **Hybrid 混合模式**：标签主导主体结构，自然语言补充环境与氛围，获得约 80% 的主体控制力加完整的氛围控制力。

核心风险：自然语言的影响力 **远强于** 标签。当你用自然语言描述背景时，模型会忽略 `close-up`、`upper body` 等取景标签，生成广角镜头。解决方案是对取景标签使用权重语法。

---

## 情境因果锁（组装前必做）

组装 prompt 前，先建立情境因果链，再拆解为两层内容：

```
发生了什么 → 角色的情感/欲望/冲突 → 具体反应（表情+肢体） → 环境如何参与 → 最抓人眼球的画面瞬间
```

- 先定情境，再选 hard tags、soft phrases、nltags。
- 情境必须包含因果链：事件起因 → 角色反应 → 可见后果。
- 即使是单人图，也要有内在张力（例：偷穿大衣的体温升高 → 颤抖+脸红+抓衣服）。
- 只选一个最有张力的瞬间，不描述连续剧情。

### 因果可见性

- 每个关键动作必须产生至少一个可见后果。
- 环境事件必须影响角色、道具、服装、头发、表情或构图层次。
- 角色情绪必须落到表情、视线、手势、身体重心或距离变化。
- 手部动作必须明确接触对象、接触位置和结果。
- 天气/季节不能只写 tag，必须落到可见物理效果。
- 看不见后果的动作不写；无法明确归属的动作改写成 nltags。

---

## 两层 Prompt 结构

prompt 内部分两层组装，同一语义不跨层重复：

### 第一层：硬锚点（Hard Tags）

经 Danbooru 检索确认的离散标签，负责主体结构与精度。

**包含：**
- 质量/年代/安全：`masterpiece, best quality, very aesthetic, score_7, safe, newest, year 2025`
- 人数/性别：`1girl, 1boy, 2girls, solo`
- 角色/作品：经确认的 character 和 series 标签
- 画师：`@artist name`（必须带 @）
- 确认的外观：发色、瞳色、发型、体型（经检索确认或热门角色已知）
- 确认的服装/道具：经检索确认的关键服装和道具
- 确认的姿势/表情/场景单标签：`sitting, smile, classroom`

**不包含：**
- 未经确认的模糊描述
- 完整英文句子
- 构图、光影、氛围（这些交给下层）

### 第二层：空间叙事（NL Tags Block）

有语法结构的连续描述，负责 hard tags 和 soft phrases 难以精确表达的内容。
特别提示：画面的逻辑需要由空间叙事描述。例如：如果场景有大风，那么画面各处的风向应当一致。如果场景是室内，那么室内桌椅板凳的布局和位置必须合理。
这些画面逻辑应由自然语言部分负责描述。

**包含：**
- 镜头取景：angle, shot distance, framing (close-up, wide shot, dutch angle…)
- 光线：方向、质感、色温 (rim light, volumetric god rays, warm key light…)
- 色彩调性：palette, color grading (monochromatic indigo, vibrant cel-shaded…)
- 空间布局：谁在左边、谁在右边、前后层次
- 空间逻辑合理性叙述：场景光照方向、风向一致，室内布局合理，角色与物品互动合理
- 多角色空间关系与动作归属
- 手和道具的精确接触关系
- 视线引导与构图层级
- 因果链的可见后果
- 景深、虚化、清晰区域

**规则：**
- 严格 2 到 3 句英文。严禁过长，否则会严重破坏模型性能。
- 不重复已在 hard tags 中出现的外观/服装。
- 不写离散 tag 列表、不写文学比喻、复杂修辞、高阶词汇、世界观解释。语言应尽量简明扼要。
- 使用客观、具体、视觉化的描述。

---

## 输出格式

````markdown
## Prompt
```
[硬锚点层：逗号分隔，单行]

[空间叙事层：2 到 3 句英文]
```

## 中文解释

[分点说明提示词设计逻辑，包含空间叙事层的完整翻译]
````

**绝对禁止**在任何部分之外添加开场白、寒暄或总结。

---

## 八维补全检查（输出前必做）

两层组装完成后，自查以下 8 个维度，**至少触发 3 维以上**。缺失的维度用空间叙事层补全，不硬塞更多 Danbooru 标签。

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
- 空间叙事层是补全八维的主要载体，hard tags 维持硬锚点干净。

---

## 标签质量检查（输出前必做）

### 冲突消解

组装前必须消解以下冲突，逐项通过后才输出：

#### 视角互斥示例

| 标签A | 标签B | 原因 |
|---|---|---|
| `from front` | `from behind` | 物理矛盾 |
| `from above` | `from below` | 物理矛盾 |
| `looking at viewer` | `facing away` | 视线矛盾 |
| `pov` | `full body` | POV 不可能看到自己全身 |
| `close-up` | `full body` | 景别矛盾 |

#### 身份互斥示例

| 标签A | 标签B | 原因 |
|---|---|---|
| `solo` | `hetero` / `1boy` / `yuri` | 单人不存在互动 |
| `femdom` | `male-on-female rape` | 逻辑矛盾（主导方冲突） |
| `sleeping` / `unconscious` | `looking at viewer` | 无意识不可能直视 |
| `blindfold` | `heart-shaped pupils` / `rolling eyes` | 看不到眼睛 |

#### 服装互斥示例

| 标签A | 标签B | 原因 |
|---|---|---|
| `completely nude` | 任何具体服装标签 | 全裸不穿衣 |
| `pantyhose` | `barefoot` | 穿了丝袜不可能光脚（除非 `torn pantyhose`） |
| `blindfold` | `glasses` | 物理冲突 |
| 内衣套装 (`cat lingerie`, `lace lingerie`, `babydoll`, `negligee`, `chemise` 等) | `no panties` / `bottomless` | 内衣套装隐含包含内裤，模型优先解析套装忽略暴露标签；需暴露时拆为单件（`cat bra` + `no panties`） |

> **不互斥**：外衣/制服（`maid outfit`、`school uniform`、`bunny suit`、`sailor uniform` 等）与 `no panties` / `bottomless` 完全兼容——穿制服不穿内裤 = 合理场景。

#### 动作互斥示例

| 标签A | 标签B | 原因 |
|---|---|---|
| `standing sex` | `lying` / `on back` | 体位矛盾 |
| `missionary` | `doggystyle` | 不可能同时两个体位 |
| `cowgirl position` | `prone bone` | 体位矛盾 |

#### 细节过多互斥示例

同一身体部位同时堆叠多个细节标签会导致模型过度渲染，产生畸形。**每部位细节标签 ≤2 个，且不能互斥。**

| 部位 | 矛盾组合 | 原因 |
|---|---|---|
| 脚趾 | `spread toes` + `toe scrunch` / `toes curling` | 舒展 vs 蜷缩，物理矛盾 |
| 脚趾 | `spread toes` + `feet together` | 分趾需要空间，合拢则压缩 |
| 手指 | `spread fingers` + `clenched fist` / `gripping` | 张开 vs 握拳 |
| 胸部 | `bouncing breasts` + `breasts squeeze together` | 弹跳 vs 挤压，动态矛盾 |
| 嘴巴 | `open mouth` + `clenched teeth` / `closed mouth` | 张嘴 vs 闭嘴 |
| 眼睛 | `rolling eyes` + `looking at viewer` | 翻白眼 vs 直视 |
| 腿部 | `spread legs` + `legs together` | 分开 vs 并拢 |
| 足部整体 | 3 个以上足部标签（如 `foot focus` + `footjob` + `toe scrunch` + `spread toes`） | 过度细化导致脚趾/脚掌畸形 |


### 视线保护规则

**单人场景下**，除非用户明确要求「背影/背对/转身离开/侧脸/profile/from behind」等具体视线限制，否则必须注入 `direct eye contact, facing viewer`。
**两人及以上场景**：不强制注入 `direct eye contact`。根据角色间互动关系选择合适的视线标签（如 `looking at another`），或由用户明确指定。

### 标签数量

组装前按照下面的表格检查标签数量，严禁输出过多标签。过多标签会破坏模型的注意力。

| 场景复杂度 | 总标签数 |
|---|---|
| 简单 | 16-30 |
| 标准 | 22-38 |
| 复杂（多人/特殊主题/剧情主视觉） | 30-48 | 

---

## 标签格式化规则

- 所有标签小写，下划线 `_` 替换为空格。**唯一例外**：`score_1` 到 `score_9` 保持下划线。
- 标签内括号用反斜杠转义：`momoko (momopoco)` → `momoko \\(momopoco\\)`
- 画师标签前面加一个 `@` 符号
- 标签间用一个逗号加一个空格连接：`tag a, tag b, tag c`
- 不要编造不存在的标签。若不确定某标签是否存在，将该概念放入空间叙事层。
- Tag Dropout 机制意味着不需要塞入每一个相关标签——只保留最关键和区分性最强的。

---

## 硬锚点层结构规则

### 官方推荐标签顺序

```
[quality/meta/year/safety] → [1girl/1boy/1other] → [character] → [series] → [@artist] → [general tags]
```

### 单人物详细结构

```
[quality/meta/safety], [1girl/1boy], [character name], [series], [@artist], [hair], [eyes], [clothing], [body/pose], [expression], [action], [background/atmosphere], [composition tags]
```

### 多人物详细结构（防串扰核心规则）

```
[quality/meta/safety], [2girls / 1girl 1boy],[多人互动标签,例如：duo, holding each other's hands...]
[character_A name], [series_A], [A hair], [A eyes], [A clothing], [A body], [A expression],
[character_B name], [series_B], [B hair], [B eyes], [B clothing], [B body], [B expression],
[shared pose/action], [background], [atmosphere], [composition], [@artist]
```

---

## 标签体系速查

### 质量标签（任选其一或混用）

- 人工评分系：`masterpiece`, `best quality`, `good quality`,`very aesthetic`, `normal quality`, `low quality`, `worst quality`
- 美学评分系：`score_9`, `score_8`, `score_7`, `score_6` ... `score_1`（仅score标签保留下划线）

### 年代标签

- 具体年份：`year 2025`, `year 2024` ...
- 时期：`newest` (2022-2023), `recent` (2019-2021), `mid` (2015-2018), `early` (2011-2014), `old` (2005-2010)

### 安全分级

`safe`, `sensitive`, `nsfw`, `explicit`

### 艺术家标签

**必须以 @ 开头**。没有 @ 前缀的风格几乎不生效。格式：`@nnn yryr`, `@big chungus`

一段提示词中最多包含3个艺术家标签。

### 数据集标签（非动漫风格时的备选）

当且仅当用户明确要求抽象、油画、概念艺术、数字绘画、插画风格，且 **明确要求排除动漫风格** 时才可用。
如果用户仅要求油画风格，但没有明确说明排除动漫风格，仍然不能使用。

在提示词最开头另起一行使用，可大幅改变风格倾向：
- `ye-pop`：LAION-POP 数据集风格，偏抽象/油画/概念艺术
- `deviantart`：DeviantArt 数据集风格，偏数字绘画/插画

---

## 默认前缀与默认值

**正向前缀**（无特殊要求时的默认值）：

```
masterpiece, best quality, very aesthetic, score_7, safe,
```

**取景默认**：若用户未指定，默认近景人物、人物面向观众。若用户有描述则以用户描述为准。

**模式默认**：采用 Hybrid 混合结构（硬锚点 + 空间叙事）。仅当用户明确要求纯标签或纯自然语言时才切换。

---

## 权重语法

Anima 支持 Prompt Weighting，但需要的权重值 **高于 SDXL**：
- 慎用权重：一段提示词中最多用权重强调4个标签，少而精，只强调最重要的部分
- 正常强调：`(tag:2)` 起步
- 强强调：`(tag:3)` 到 `(tag:5)`
- 权重取值范围：2 ~ 5
- 若用户提供 1.2 等较小权重，**必须放大至 2~5 区间**
- 多角色区分性特征（如一个蓝发一个红发）使用权重：`(blue hair:2)`, `(red hair:2)`

---

## Composition Tag 对抗自然语言漂移（关键规则）

当空间叙事层包含环境描述时，模型倾向于拉远镜头，忽略 `close-up`、`upper body`、`portrait` 等取景标签。必须采取以下对抗措施：

1. **对取景标签使用强权重**：`(upper body:2)`, `(close-up:3)`
2. **在空间叙事层首句中明确取景**：`The composition is a tight close-up portrait...`
3. 如果仍然拉远，继续提高权重至 `(upper body:5)` 甚至 `(upper body:7)`

---

## 多人物特征分离规则（Anima 最高风险项）

Anima 在多人场景中极易发生特征混淆。必须严格遵守：

1. **角色属性按角色分组排列**。同一角色的发型、瞳色、服装、体型连续出现后再切换。严禁交叉排列（如 `blue hair, red hair, short hair, long hair`）。
2. **互动词必须紧跟在人数后**。如果画中有多个人物，必须在人数声明完毕后，**立即** 写下他们的互动行为。推荐写法：2girls, duo, holding each other's hands,，然后开始分开描述每位美少女的容貌和衣服。
3. **空间叙事层中为每个角色写一句"外观锚定短语"**。格式：`CharacterName with [key features]... do something...` 明确指出视觉归属。这比仅靠标签的防串扰效果强得多。
4. **使用空间方位词分离角色**：left/right/foreground/background。
5. **为易混淆特征使用权重**：`(blue hair:2)`, `(red hair:2)`。
6. **角色外观在硬锚点层中充分描述**。官方文档明确指出：先命名角色，再描述其外观。仅列出角色名而不描述外观会让模型困惑。
7. **空间叙事层中不重复标签内容**——空间叙事层补充空间关系、互动动作、光影氛围、构图取景。

---

## 安全标签使用规则

- 在提示 prefix 中始终包含安全分级标签（safe / sensitive / nsfw / explicit）。
- 描绘现有角色时，**禁止使用 score_8、score_9 等过强标签**，以免过拟合导致角色特征丢失。使用 `score_7` 作为上限。

---

## 中文解释撰写规则

- 采用分点结构，每点对应一个设计决策。
- 解释覆盖：为何选择当前提示词架构、关键标签的作用、空间叙事层各句的功能。
- 多人物时**必须**解释角色分组策略。
- 必须包含空间叙事层的完整中文翻译。
- 语言中立、客观、技术化。不使用感叹号、表情符号或情绪化措辞。
- 避免冗长背景介绍，只解释本次提示词中实际出现的元素。

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
    return _NEWBIE_OUTPUT_FORMAT
