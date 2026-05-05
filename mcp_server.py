"""
mcp_server.py
─────────────
MCP 服务层（本地 stdio 模式）

启动方式：
    python mcp_server.py

Claude Desktop 配置（claude_desktop_config.json）：
    {
      "mcpServers": {
        "danbooru-searcher": {
          "command": "python",
          "args": ["mcp_server.py的完整路径"]
        }
      }
    }

支持的工具：
    search_tags      自然语言搜索标签
    get_related_tags 基于共现表查关联推荐
"""

import json
import asyncio
import os
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from core.engine import DanbooruTagger
from core.models import SearchRequest
import re

# 确保工作目录为脚本所在目录（stdio 模式下 MCP 客户端可能从任意目录启动）
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

mcp = FastMCP(name="danbooru-searcher")


@mcp.tool()
async def search_tags(
    query: str,
    use_segmentation: bool = True,
    top_k: int = 5,
    limit: int = 80,
    popularity_weight: float = 0.15,
    show_nsfw: bool = True,
    include_wiki: bool = False,
    category: str = "all",
) -> str:
    """
Search Danbooru tags using natural language and return a ready-to-use prompt.

## Args
- query: Natural language description (Chinese recommended).
- use_segmentation: Split multi-concept input into segments for separate retrieval. True for scene descriptions, False for single-concept queries.
- top_k: Candidates recalled per segment. Semantics change with use_segmentation — see guide below.
- limit: Max tags returned.
- popularity_weight: Influence of tag post count on ranking (0.0–1.0). Default 0.15.
- show_nsfw: Include NSFW tags. Default True.
- include_wiki: Append wiki description to each result. Default False.
- category: Filter results to a specific tag category. Default "all".
    "all"       —  All categories (通用 + 版权 + 人物 )
    "general"   —  General: visual attributes, clothing, pose, background, etc.
    "copyright" —  Copyright: specific anime/game/franchise titles
    "character" —  Character: named characters from any series
    Use this when you know what kind of tag you need — e.g. looking for a
    character name vs. describing a scene visually.

## Query writing guide

The `query` parameter supports explicit delimiter control for precise segmentation.

### Explicit delimiters

Use **spaces, newlines, Chinese commas (，), or Chinese dunhao (、)** to manually separate concepts.
Each delimiter-bounded segment ≤7 characters stays atomic — the engine respects your intent.

| Query style | Example | When to use |
|---|---|---|
| Concept list (spaces) | `运动社团 校队 比赛 运动会` | You know the exact concepts to search |
| Concept list (dun hao) | `反乌托邦、赛博朋克、蒸汽朋克` | Same, with Chinese list punctuation |
| Natural sentence | `一个穿着白色水手服的少女在雨中奔跑` | Scene description, let the engine auto-split |
| Mixed | `运动社团 一个穿水手服的少女` | Mix concepts with descriptive phrases |

Segments >7 characters are still auto-split by jieba, but the raw segment is kept as an additional query
to preserve clause-level semantics (multi-granularity retrieval).

### Recommendations

1. **Concept lists → use explicit delimiters:** Group independent concepts with spaces or dunhao.
   `运动社团 校队 比赛 体育祭 田径部` is better than `运动社团校队比赛体育祭田径部`.

2. **Scene descriptions → write naturally:** Natural Chinese with Chinese commas works well for full scenes.
   `一个穿着白色水手服，蓝色短裙的少女在雨中奔跑` — commas here are grammatical, not delimiters.

3. **Precise lookup → turn off segmentation:** For finding a specific character or copyright title,
   set `use_segmentation=False` and combine with `category` filter.
   e.g. `query="EVA中蓝发的零号机驾驶员"` with `category="character"` and `use_segmentation=False`.

4. **Category filtering:** Use `category` to narrow results. Looking for a character?
   `category="character"`. Building a scene prompt? `category="general"`.

## Parameter guide

### Step 1 — Decide use_segmentation + top_k together

top_k means "candidates per segment"; its effect depends on whether segmentation is on.

Multi-concept input (scene description) → use_segmentation=True

| Sub-scenario          | top_k | Reason                                                   |
|-----------------------|-------|----------------------------------------------------------|
| Full scene → prompt   | 5     | Many segments; low top_k distributes result slots fairly |
| Vague concept explore | 80    | Few segments; high top_k needed for broad recall         |

Single-concept input → use_segmentation=False

top_k acts as total candidate pool size. Use 20 for all single-concept cases.

| Sub-scenario                  | top_k |
|-------------------------------|-------|
| Describe subject / find tag   | 20    |
| Precise lookup / spell fix    | 20    |

### Step 2 — Decide limit independently

| Goal                         | limit |
|------------------------------|-------|
| Full prompt for image gen    | 80    |
| Concept exploration          | 20–80 |
| Precise lookup / role search | 10–20 |

### Auxiliary params

popularity_weight (default 0.15, rarely needs changing):
- Higher (0.3+): favor common, well-established tags
- Lower (0.0): surface niche/rare tags

include_wiki (default False):
- True: The meaning of the tag is important — disambiguation, explaining tags to users, exploring unfamiliar domains, or when you are unsure of the tag's meaning
- False: Prompt generation (Wiki is irrelevant to the downstream task), tags are known

### Quick reference

| Scenario                      | use_segmentation | top_k | limit |
|-------------------------------|------------------|-------|-------|
| Full scene → prompt (default) | True             | 5     | 80    |
| Vague concept exploration     | True             | 80    | 80    |
| Describe subject / find tag   | False            | 20    | 20    |
| Precise lookup / spell fix    | False            | 20    | 10    |

### Workflow

After search_tags, pass selected tags to get_related_tags to discover complementary tags via co-occurrence (accessories, character features, scene atmosphere).
Supports chained exploration / iterative loops – take the interesting tags from the returned results as input to call get_related_tags again,
and use the results from get_related to feed back into a new round of search, enabling multi-hop deep traversal along the co-occurrence graph.

## Examples

Precise lookup / spell fix — e.g. "selafuku", "thighhigh", "twintail"
→ use_segmentation=False, top_k=20, limit=10

Vague concept exploration — e.g. "兔耳朵", "赛博朋克服装", "假肢"
→ use_segmentation=True, top_k=80, limit=80

Describe subject / find tag — e.g. "EVA中蓝发的零号机驾驶员", "命运石之门中的助手"
→ use_segmentation=False, top_k=20, limit=20

Full scene → prompt — e.g. "一个穿着白色水手服，蓝色短裙的少女在雨中的城市里奔跑"
→ use_segmentation=True, top_k=5, limit=80

## Returns
JSON with: prompt (comma-separated tags), keywords, results.
Each result: tag, cn_name, category, final_score, count[, wiki if include_wiki=True].
    """
    _CATEGORY_MAP: dict[str, list[str]] = {
        "all":       ["General", "Character", "Copyright", "Artist", "Meta"],
        "general":   ["General"],
        "character": ["Character"],
        "copyright": ["Copyright"],
    }
    target_categories = _CATEGORY_MAP.get(
        category,
        _CATEGORY_MAP["all"],  # unrecognized value → fall back to all
    )

    tagger = await DanbooruTagger.get_instance()
    request = SearchRequest(
        query=query,
        top_k=top_k,
        limit=limit,
        popularity_weight=popularity_weight,
        show_nsfw=show_nsfw,
        use_segmentation=use_segmentation,
        target_categories=target_categories,
    )
    response = await asyncio.to_thread(tagger.search, request)

    results = []
    for r in response.results:
        if r.nsfw == '1' and not show_nsfw:
            continue
        item = {
            "tag":         r.tag,
            "cn_name":     r.cn_name,
            "category":    r.category,
            "final_score": r.final_score,
            "count":       r.count,
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
Return co-occurrence-based tag recommendations for a given tag list (NPMI scoring).

This tool surfaces tags that frequently appear alongside the seeds in
Danbooru, mixing categories (General / Character / Copyright) by design.

## Typical use cases

- Attribute → characters who have it
  e.g. ["fingerless_gloves"] → tifa_lockhart, cammy_white, bridget_(guilty_gear), ...
- Work → characters in it
  e.g. ["overlord_(maruyama)"] → shalltear_bloodfallen, ainz_ooal_gown, albedo_(overlord), ...
- Character → their visual attributes
  e.g. ["amiya_(arknights)"] → outfits, expressions, accessories
- Theme exploration
  e.g. ["fighter_jet"] → aircraft types, actions, backgrounds
- Multi-tag intersection
  e.g. ["maid", "twintails"] → tags specific to the combination, scored by summed NPMI

For within-category exploration (e.g. "more clothing tags like X"), use search_tags
with the `category` parameter instead.

## Workflow

Chain freely: search_tags → get_related_tags → get_related_tags → search_tags.
Each hop along the co-occurrence graph reveals tags unreachable by semantic search alone.

## Args

- tags: List of canonical Danbooru tag names (underscores, no spaces).
        e.g. ["white_serafuku", "sailor_collar"]
- limit: Max recommendations returned. Default 50.
- show_nsfw: Include NSFW tags. Default True.
- include_wiki: Append wiki description to each result. Default False.
        Set True when result tags are unfamiliar and need disambiguation.

## Returns

JSON array sorted by aggregated NPMI score (descending). Each result:
- tag, cn_name, category, count (post_count), cooc_score (normalized to [0,1])
- sources: seed tags that contributed to this score
- wiki: only if include_wiki=True
    """
    tagger = await DanbooruTagger.get_instance()
    results = await asyncio.to_thread(
        tagger.get_related,
        tags,
        set(tags),
        limit,
        show_nsfw,
    )

    output = []
    for r in results:
        item = {
            "tag":        r.tag,
            "cn_name":    r.cn_name,
            "category":   r.category,
            "count":      r.post_count,
            "cooc_score": r.cooc_score,
            "sources":    r.sources,
        }
        if include_wiki:
            item["wiki"] = r.wiki
        output.append(item)

    return json.dumps(output, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    mcp.run(transport='stdio')