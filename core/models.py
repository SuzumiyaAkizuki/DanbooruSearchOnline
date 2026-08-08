"""
core/models.py
数据结构
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TagResult:
    """单条标签搜索结果。"""
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
    artist_top_tags: list[str] = field(default_factory=list)
    alias_from: str | None = None  # 搜索结果由官方 Tag Alias 规范化时记录原标签


@dataclass
class RelatedTag:
    """单条关联推荐结果"""
    tag: str
    cn_name: str
    category: str
    nsfw: str
    cooc_count: int       # 累计共现次数
    cooc_score: float     # 归一化分数
    sources: list[str] = field(default_factory=list)  # 触发该推荐的种子 tag 列表
    post_count: int = 0   # 该标签的发帖总数（与 search_tags 的 count 对齐）
    wiki: str = ""        # 标签 wiki 描述


@dataclass
class ArtistResult:
    """单条画师搜索结果。"""
    artist: str           # 画师名
    score: float          # 综合 NPMI 得分
    cooc_count: int       # 累计共现次数
    post_count: int       # 画师在 Danbooru 的作品数
    sources: list[str]    # 命中标签列表
    hit_count: int        # 命中标签数


@dataclass
class SearchRequest:
    """搜索参数"""
    query: str
    top_k: int = 5
    limit: int = 80
    popularity_weight: float = 0.15
    show_nsfw: bool = True
    use_segmentation: bool = True
    target_layers: list[str] = field(
        default_factory=lambda: ['英文', '中文扩展词', '释义', '中文核心词']
    )
    target_categories: list[str] = field(
        default_factory=lambda: ['General', 'Character', 'Copyright']
    )
    group_mode: str = "off"        # "off" / "expand" / "diverse"
    max_per_group: int = 2         # 仅 diverse 模式生效


@dataclass
class SearchResponse:
    """搜索返回值。"""
    tags_all: str
    tags_sfw: str
    results: list[TagResult]
    keywords: list[str]
    segments: list[str] = field(default_factory=list)  # 分隔符切分后的原始从句级片段
    cached_queries: list[str] = field(default_factory=list)  # 命中 emb 缓存的查询文本
