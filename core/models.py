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


@dataclass
class SearchResponse:
    """搜索返回值。"""
    tags_all: str
    tags_sfw: str
    results: list[TagResult]
    keywords: list[str]