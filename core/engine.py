"""
core/engine.py
──────────────
DanbooruTagger 核心引擎

缓存格式（存于 cache_dir/ 目录）：
  embeddings.safetensors   — 四路向量矩阵（FP16），行顺序与 metadata.parquet 完全对齐
  metadata.parquet         — DataFrame（name/cn_name/cn_core/wiki/nsfw/category/post_count）
  meta.json                — 标量元数据（max_log_count、schema_version）
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import jieba
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file as st_save
from safetensors import safe_open
from sentence_transformers import SentenceTransformer

from .models import SearchRequest, SearchResponse, TagResult
from .tag_aliases import (
    TagAliasRecord,
    TagAliasSnapshotError,
    load_tag_alias_snapshot,
)
from platform_utils import (
    PLATFORM,
    is_cloud,
    download_file,
    resolve_model_path,
)


def _positive_int_env(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_torch_threads(cpu_count: Optional[int] = None) -> int:
    """Resolve PyTorch intra-op threads for CPU-only Space deployments."""
    override = _positive_int_env('DANBOORU_TORCH_THREADS')
    if override is not None:
        return override
    cpus = cpu_count or os.cpu_count() or 2
    if cpus <= 2:
        return 1
    return max(1, min(4, cpus // 2))


def _resolve_cpu_sem_limit(cpu_count: Optional[int] = None) -> int:
    """Resolve process-wide CPU task concurrency."""
    override = _positive_int_env('DANBOORU_CPU_CONCURRENCY')
    if override is not None:
        return override
    cpus = cpu_count or os.cpu_count() or 2
    return 2 if cpus >= 8 else 1


# 限制 PyTorch CPU 线程数，给 asyncio 事件循环和 UI 留出核心。
torch.set_num_threads(_resolve_torch_threads())


# LRU 缓存
class LRUCache:
    def __init__(self, maxsize: int):
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: Any) -> Any:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

STOP_WORDS: frozenset[str] = frozenset({
    ',', '.', ':', ';', '?', '!', '"', "'", '`',
    '(', ')', '[', ']', '{', '}', '<', '>',
    '-', '_', '=', '+', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '~',
    '，', '。', '：', '；', '？', '！', '\u201c', '\u201d', '\u2018', '\u2019',
    '（', '）', '【', '】', '《', '》', '、', '…', '—', '·',
    ' ', '\t', '\n', '\r',
    '的', '地', '得', '了', '着', '过',
    '是', '为', '被', '给', '把', '让', '由',
    '在', '从', '自', '向', '往', '对', '于',
    '和', '与', '及', '或', '且', '而', '但', '并', '即', '又', '也',
    '啊', '吗', '吧', '呢', '噢', '哦', '哈', '呀', '哇',
    '我', '你', '他', '她', '它', '我们', '你们', '他们',
    '这', '那', '此', '其', '谁', '啥', '某', '每',
    '这个', '那个', '这些', '那些', '这里', '那里',
    '个', '位', '只', '条', '张', '幅', '件', '套', '双', '对', '副',
    '种', '类', '群', '些', '点', '份', '部', '名',
    '很', '太', '更', '最', '挺', '特', '好', '真',
    '一', '一个', '一种', '一下', '一点', '一些',
    '有', '无', '非', '没', '不',
    '正在', '已经', '正', '刚', '开始', '继续', '一直', '不断',
    '穿着', '戴着', '穿', '戴',
    '带有', '具有', '拥有',
    '看起来', '看上去', '显得', '仿佛', '似乎',
    '十分', '非常', '特别', '比较',
    '图片', '画面', '图像',
    '位于', '处于',
    '许多', '大量', '各种', '所有', '其他', '其它',
    # ── 英文停用词 ──
    'a', 'an', 'the',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'into',
    'about', 'between', 'through', 'after', 'before', 'above', 'below',
    'and', 'or', 'but', 'nor', 'so', 'yet',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'do', 'does', 'did', 'done',
    'have', 'has', 'had', 'having',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'not', 'no', 'very', 'too', 'also', 'just', 'only', 'even', 'still',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their',
    'this', 'that', 'these', 'those', 'which', 'who', 'whom', 'what',
    'there', 'here', 'where', 'when', 'how', 'all', 'each', 'every',
    'some', 'any', 'few', 'more', 'most', 'other', 'such',
    'than', 'up', 'out', 'if', 'then', 'else', 'while', 'during',
    'both', 'same', 'own', 'now',
})

CAT_MAP: dict[str, str] = {
    '0': 'General', '1': 'Artist', '3': 'Copyright', '4': 'Character', '5': 'Meta',
}

SCHEMA_VERSION = 4   # 升级此值将自动触发全量重建，用于破坏性格式变更

# 用户显式分隔后，纯 CJK 片段超过此长度仍用 jieba 切分（避免长句被当作原子概念）
_ATOMIC_CJK_MAX_LEN = 7

# 四路 embedding 层配置: (层名, tensor 属性名, DataFrame 列名)
_LAYER_SPEC: list[tuple[str, str, str]] = [
    ('英文',   'emb_en',      'name'),
    ('中文扩展词', 'emb_cn',      'cn_name'),
    ('释义',   'emb_wiki',    'wiki'),
    ('中文核心词', 'emb_cn_core', 'cn_core'),
]
_ALL_LAYER_NAMES = [ln for ln, _, _ in _LAYER_SPEC]


# ──────────────────────────────────────────────
# 缓存路径助手
# ──────────────────────────────────────────────

class _CachePaths:
    def __init__(self, cache_dir: str | Path):
        self.dir        = Path(cache_dir)
        self.embeddings = self.dir / 'danbooru_multiview_embeddings.safetensors'
        self.metadata   = self.dir / 'tags_metadata.parquet'
        self.meta_json  = self.dir / 'version_data.json'

    def exists(self) -> bool:
        return (
            self.embeddings.is_file()
            and self.metadata.is_file()
            and self.meta_json.is_file()
        )

    def ensure_dir(self):
        self.dir.mkdir(parents=True, exist_ok=True)


class DanbooruTagger:
    """核心搜索引擎（单例）"""

    _instance: Optional['DanbooruTagger'] = None
    _lock: Optional[asyncio.Lock] = None
    # 进程级 CPU 并发闸门：串行化所有 CPU 密集型操作（search / get_related /
    # get_group_candidates），避免并发抢占 CPU 而拖垮 asyncio 事件循环。
    _cpu_sem: Optional[asyncio.Semaphore] = None

    @classmethod
    def is_ready(cls) -> bool:
        return cls._instance is not None and cls._instance.is_loaded

    @classmethod
    async def get_instance(cls, **kwargs) -> 'DanbooruTagger':
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        async with cls._lock:
            if cls._instance is None:
                inst = cls(**kwargs)
                await asyncio.to_thread(inst.load)
                cls._instance = inst
            return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        csv_file:   str = 'origin_database/tags_enhanced.csv',
        cache_dir:  str = 'tags_embedding',
        cooc_file:          str = 'origin_database/cooccurrence_clean.csv',
        group_file:         str = 'origin_database/tag_groups.json',
        tag_artist_file:    str = 'origin_database/tag_artist_cooc.parquet',
        tag_alias_file:     str = 'origin_database/tag_aliases.parquet',
        tag_alias_meta_file: str = 'origin_database/tag_aliases_metadata.json',
    ):
        # 模型路径：优先使用显式传入，否则交由 platform_utils 解析
        self.model_path = model_path or resolve_model_path()

        self.csv_path  = csv_file
        self.device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.paths     = _CachePaths(cache_dir)
        self.cooc_file = cooc_file
        self.group_file = group_file
        self.tag_artist_file = tag_artist_file
        self.tag_alias_file = tag_alias_file
        self.tag_alias_meta_file = tag_alias_meta_file

        self.model:         Optional[SentenceTransformer]         = None
        self.df:            Optional[pd.DataFrame]                = None
        self.emb_en:        Optional[torch.Tensor]                = None
        self.emb_cn:        Optional[torch.Tensor]                = None
        self.emb_wiki:      Optional[torch.Tensor]                = None
        self.emb_cn_core:   Optional[torch.Tensor]                = None
        self.max_log_count: float                                 = 15.0
        self.cooc: dict[str, list[tuple[str, int]]]               = {}
        self._name_to_idx: dict[str, int]                         = {}
        self._tag_to_groups: dict[str, set[str]]                  = {}
        self._group_to_tags_idx: dict[str, np.ndarray]            = {}
        self._group_cn_names: dict[str, str]                      = {}
        self._tag_artist_index: dict[str, list[tuple]]           = {}
        self._artist_top_tags: dict[str, list[tuple]]            = {}  # artist → [(tag, npmi, cn_short), ...]
        self._artist_post_count: dict[str, int]                  = {}
        self._tag_aliases: dict[str, TagAliasRecord]             = {}
        self._tag_alias_metadata: dict[str, Any]                 = {}
        self.is_loaded:     bool                                  = False

        # 预提取的列数组，避免热点路径上反复执行 df.iloc[idx]
        self._arr_name:       Optional[np.ndarray] = None
        self._arr_cn_name:    Optional[np.ndarray] = None
        self._arr_category:   Optional[np.ndarray] = None
        self._arr_nsfw:       Optional[np.ndarray] = None
        self._arr_wiki:       Optional[np.ndarray] = None
        self._arr_post_count: Optional[np.ndarray] = None
        self._arr_pop_score:  Optional[np.ndarray] = None

        # 三层 LRU 缓存（纯内存，重启后自动重热）
        # embedding 缓存：key=文本, value=归一化后的 1-D Tensor (D,)，约 40 MB
        self._emb_cache:     LRUCache = LRUCache(maxsize=10_000)
        # 搜索结果缓存：key=请求参数 tuple, value=SearchResponse，约 100 MB
        self._search_cache:  LRUCache = LRUCache(maxsize=5_000)
        # 关联推荐缓存：key=(seed_tuple, limit, show_nsfw), value=list[RelatedTag]，约 20 MB
        self._related_cache: LRUCache = LRUCache(maxsize=2_000)

    # ── 初始化 ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """同步加载，在线程池中调用。"""
        if self.is_loaded:
            return
        t0 = time.time()

        # ── 云端环境：从对应平台 Hub 拉取数据文件 ──────────────────────────
        if is_cloud():
            self._pull_cloud_files()

        # ── 缓存校验与构建 ─────────────────────────────────────────────────
        if not self.paths.exists():
            print('\n' + '=' * 50)
            print('[Engine] 未找到缓存，开始首次构建（约 1~3 分钟）...')
            print('=' * 50 + '\n')
            self._load_model()
            self._build_full()
        else:
            print(f'[Engine] 加载缓存 ({self.paths.dir}) ...')
            self._load_from_cache()
            if self._cached_schema_version() != SCHEMA_VERSION:
                print('[Engine] 缓存格式版本不符，触发全量重建...')
                self._load_model()
                self._build_full()
            elif os.path.exists(self.csv_path):
                self._load_model()
                self._smart_update()

        if self.model is None:
            self._load_model()

        self._setup_jieba_from_memory()
        self._load_cooc()
        self._load_tag_artist_cooc()
        self._name_to_idx = {n: i for i, n in enumerate(self.df['name'])}
        self._tag_names_set: set[str] = set(self._name_to_idx.keys())
        self._load_tag_aliases()
        self._rebuild_arrays_from_df()
        self._normalize_embeddings()
        self._load_groups()
        self.is_loaded = True
        print(f'[Engine] 初始化完成，耗时 {time.time() - t0:.2f}s')

    def _normalize_embeddings(self) -> None:
        """
        对四路 embedding 矩阵做 L2 归一化（in-place 替换 self.emb_*）。
        归一化后 search 阶段可直接用矩阵乘法得到 cosine similarity，
        无需每次调用 util.semantic_search 内部再做一次归一化。
        """
        for _, attr, _ in _LAYER_SPEC:
            t = getattr(self, attr)
            if t is None:
                continue
            setattr(self, attr, torch.nn.functional.normalize(t, p=2, dim=1))

    def _rebuild_arrays_from_df(self) -> None:
        """
        将 DataFrame 中搜索热点路径需要的列预提取为 numpy 数组。
        任何修改 self.df 行内容或行数的操作之后都必须调用此方法刷新。
        """
        if self.df is None:
            return
        self._arr_name       = self.df['name'].to_numpy()
        self._arr_cn_name    = self.df['cn_name'].to_numpy()
        self._arr_category   = self.df['category'].astype(str).to_numpy()
        self._arr_nsfw       = self.df['nsfw'].astype(str).to_numpy()
        self._arr_wiki       = self.df['wiki'].astype(str).to_numpy()
        self._arr_post_count = self.df['post_count'].to_numpy()
        # 预算热度归一化分，避免 search 中每次 np.log1p
        max_log = self.max_log_count if self.max_log_count > 0 else 1.0
        self._arr_pop_score  = np.log1p(self._arr_post_count) / max_log

    def _pull_cloud_files(self) -> None:
        """
        从当前云平台拉取所有数据文件，并将路径写回实例属性。
        HF / MS 的差异完全由 platform_utils.download_file() 屏蔽。
        """
        print(f'[Engine] 云端环境 ({PLATFORM})，开始拉取数据文件...')

        # ── HF 平台需要额外指定 repo_id（SPACE_ID）和 repo_type ──────────
        extra_hf_kwargs = {}
        if PLATFORM == 'hf':
            extra_hf_kwargs = {
                'hf_repo_id':   os.environ.get('SPACE_ID'),
                'hf_repo_type': 'space',
            }

        def pull(filename: str) -> str:
            try:
                return download_file(filename, **extra_hf_kwargs)
            except Exception as e:
                print(f'[Engine] 拉取 {filename} 失败（非致命）: {e}')
                return filename   # 回退到原始路径，让后续逻辑决定是否重建

        self.csv_path         = pull('origin_database/tags_enhanced.csv')
        self.cooc_file        = pull('origin_database/cooccurrence_clean.parquet')
        self.group_file       = pull('origin_database/tag_groups.json')
        self.tag_artist_file  = pull('origin_database/tag_artist_cooc.parquet')

        alias_path = pull('origin_database/tag_aliases.parquet')
        alias_meta_path = pull('origin_database/tag_aliases_metadata.json')
        if Path(alias_path).is_file() and Path(alias_meta_path).is_file():
            self.tag_alias_file = alias_path
            self.tag_alias_meta_file = alias_meta_path
        else:
            print('[Engine] Tag Alias 快照不完整，本次启动不启用 Alias。')

        meta_path = pull('tags_embedding/tags_metadata.parquet')
        emb_path  = pull('tags_embedding/danbooru_multiview_embeddings.safetensors')
        json_path = pull('tags_embedding/version_data.json')

        # 只有三个缓存文件都成功拉取才覆盖路径，防止部分失败导致 exists() 误判
        if all(
            Path(p).is_file()
            for p in (meta_path, emb_path, json_path)
        ):
            self.paths.metadata   = Path(meta_path)
            self.paths.embeddings = Path(emb_path)
            self.paths.meta_json  = Path(json_path)
            print('[Engine] 云端数据文件拉取完毕。')
        else:
            print('[Engine] 部分缓存文件拉取失败，将触发本地重建。')

    # ── 搜索 ──────────────────────────────────────────────────────────────

    def _encode_queries(self, queries: list[str]) -> tuple[torch.Tensor, list[bool]]:
        """批量编码查询词，命中 embedding 缓存的跳过 model.encode。

        Returns:
            (q_emb, hit_mask): 编码后的张量 (Q, D) 以及每个 query 是否命中缓存。
        """
        cached_vecs: list[Optional[torch.Tensor]] = [self._emb_cache.get(q) for q in queries]
        uncached_idx = [i for i, v in enumerate(cached_vecs) if v is None]
        hit_mask = [v is not None for v in cached_vecs]

        if uncached_idx:
            uncached_texts = [queries[i] for i in uncached_idx]
            new_embs = self.model.encode(
                uncached_texts, convert_to_tensor=True, show_progress_bar=False,
            ).float()
            new_embs = torch.nn.functional.normalize(new_embs, p=2, dim=1)
            for j, i in enumerate(uncached_idx):
                emb = new_embs[j]
                self._emb_cache.put(queries[i], emb)
                cached_vecs[i] = emb

        return torch.stack(cached_vecs), hit_mask  # type: ignore[arg-type]

    def search(self, request: SearchRequest) -> SearchResponse:
        if not self.is_loaded:
            self.load()

        cache_key = (
            request.query,
            request.top_k,
            request.limit,
            request.popularity_weight,
            request.use_segmentation,
            tuple(sorted(request.target_layers)),
            tuple(sorted(request.target_categories)),
            request.group_mode,
            request.max_per_group,
        )
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            return cached

        tag_results, keywords, extra_segments, cached_queries = self._search_tag_results(request)
        artist_results = (
            self.search_artist_rows(request.query, request.limit, show_nsfw=request.show_nsfw)
            if "artist" in request.target_layers else []
        )
        if artist_results:
            artist_tags = {r.tag for r in artist_results}
            tag_results = [r for r in tag_results if r.tag not in artist_tags]
        valid = artist_results + tag_results

        tags_all = ', '.join(r.tag for r in valid)
        tags_sfw = ', '.join(r.tag for r in valid if r.nsfw != '1')
        response = SearchResponse(
            tags_all=tags_all, tags_sfw=tags_sfw,
            results=valid, keywords=keywords, segments=extra_segments,
            cached_queries=cached_queries,
        )
        self._search_cache.put(cache_key, response)
        return response

    def _search_tag_results(
        self, request: SearchRequest,
    ) -> tuple[list[TagResult], list[str], list[str], list[str]]:
        if request.use_segmentation:
            raw_kw, raw_segments = self._smart_split(request.query)
            keywords = [w.strip() for w in raw_kw if w.strip() and w.strip() not in STOP_WORDS]
            # raw_segments: 分隔符切分后的原始片段（未经 jieba），作为从句级查询插入。
            # 排除与完整 query 相同、以及已出现在 keywords 中的片段，避免重复编码和权重膨胀。
            keywords_set = set(keywords)
            extra_segments = [s for s in raw_segments if s != request.query and s not in keywords_set]
            queries = [request.query] + extra_segments + keywords
        else:
            keywords = []
            extra_segments = []
            queries  = [request.query]

        q_emb, hit_mask = self._encode_queries(queries)

        tl    = request.target_layers
        k     = request.top_k

        # 每个查询词单独做意图识别，避免长句意图污染短分词
        query_weights = [self._detect_intent(q) for q in queries]
        active_layers = [ln for ln in _ALL_LAYER_NAMES if ln in tl]

        # 预算每个 query × 每个 layer 的 top_k 配额
        # cur_pvk_per_q[i][ln] = 第 i 个 query 在 layer ln 的配额
        cur_pvk_per_q: list[dict[str, int]] = []
        for cur_weights in query_weights:
            if active_layers:
                aw       = {l: cur_weights.get(l, 1.0) for l in active_layers}
                total_aw = sum(aw.values())
                cur_pvk  = {l: max(1, round(k * aw[l] / total_aw)) for l in active_layers}
            else:
                cur_pvk = {}
            cur_pvk_per_q.append(cur_pvk)

        target_cats = request.target_categories
        w_pop       = request.popularity_weight

        final: dict[str, TagResult] = {}

        # 按 layer 批量做矩阵乘 + topk，合并 Q×L=20 次小调用为 L=4 次大调用
        for ln, attr, _ in _LAYER_SPEC:
            if ln not in tl:
                continue
            emb_matrix = getattr(self, attr)   # (N, D)，已归一化
            if emb_matrix is None:
                continue

            # 该 layer 在所有 query 中的最大配额（少数 query 会算到多余的 hit，最后按各自配额截断）
            k_max = max((cur_pvk_per_q[i].get(ln, 1) for i in range(len(queries))), default=1)
            k_max = min(k_max, emb_matrix.shape[0])

            scores = q_emb @ emb_matrix.T                       # (Q, N)
            top_v, top_i = scores.topk(k_max, dim=1)            # (Q, k_max)
            top_v_list = top_v.tolist()
            top_i_list = top_i.tolist()

            for i, source_word in enumerate(queries):
                cur_weights = query_weights[i]
                kq = cur_pvk_per_q[i].get(ln, 1)
                layer_w = cur_weights.get(ln, 1.0)
                row_v = top_v_list[i]
                row_i = top_i_list[i]
                # 仅取该 query 自己的配额条数
                for j in range(min(kq, len(row_v))):
                    score = row_v[j]
                    if score < 0.35:
                        # topk 已按分数降序，后续都低于阈值，可提前结束
                        break
                    idx = row_i[j]
                    cat_text = CAT_MAP.get(self._arr_category[idx], 'Other')
                    if cat_text not in target_cats:
                        continue
                    tag_name    = self._arr_name[idx]
                    count       = self._arr_post_count[idx]
                    pop_score   = self._arr_pop_score[idx]
                    final_score = score * layer_w * (1 - w_pop) + pop_score * w_pop
                    if tag_name not in final or final_score > final[tag_name].final_score:
                        final[tag_name] = TagResult(
                            tag=tag_name, cn_name=self._arr_cn_name[idx], category=cat_text,
                            nsfw=self._arr_nsfw[idx],
                            final_score=round(float(final_score), 4),
                            semantic_score=round(float(score), 4),
                            count=int(count), source=source_word, layer=ln,
                            wiki=self._arr_wiki[idx],
                        )

        # ── 全句语义一致性软重排 ──────────────────────────────────────────
        # 对每个候选标签，计算其与完整原始查询（而非分词片段）的语义相似度，
        # 将相似度作为软因子乘入 final_score，使仅由分词碎片匹配到的噪声
        # 标签自然下沉，同时不硬过滤任何结果。
        # 批量矩阵乘法替代逐条 torch.dot，O(R*L) 降为 O(L) + O(R)。
        full_q = q_emb[0]          # queries[0] 始终为完整原始查询
        alpha  = 0.3 if request.use_segmentation else 0   # 一致性调节强度（0=不调节, 1=完全按一致性重排），仅在启用分词时有意义
        if alpha > 0 and final:
            tag_list = list(final.keys())
            tag_indices = [self._name_to_idx[t] for t in tag_list]
            idx_tensor = torch.tensor(tag_indices, dtype=torch.long, device=full_q.device)
            max_co = torch.zeros(len(tag_indices), device=full_q.device)
            for ln, attr, _ in _LAYER_SPEC:
                if ln not in tl:
                    continue
                emb_selected = getattr(self, attr)[idx_tensor]       # (R, D)
                co = (full_q.unsqueeze(0) @ emb_selected.T).squeeze(0)  # (R,)
                max_co = torch.maximum(max_co, co)
            for i, tag in enumerate(tag_list):
                r = final[tag]
                r.final_score = round(r.final_score * (1.0 - alpha + alpha * float(max_co[i])), 4)

        # Group expand 处理（在 guaranteed_tags 之前，因为会改分数）
        if request.group_mode == "expand" and self._tag_to_groups:
            self._apply_group_expand(final)

        # 收集每个查询源的 top-1 结果（高于阈值）
        guaranteed_tags: set[str] = set()
        for source_word in queries:
            best: TagResult | None = None
            for r in final.values():
                if r.source == source_word and r.final_score > 0.45:
                    if best is None or r.final_score > best.final_score:
                        best = r
            if best is not None:
                guaranteed_tags.add(best.tag)

        # 对所有候选进行排序，然后在保留保证结果的同时截断至限制数量
        sorted_results = sorted(final.values(), key=lambda r: r.final_score, reverse=True)
        valid: list[TagResult] = []

        if request.group_mode == "diverse" and self._tag_to_groups:
            # diverse 模式：每个 group 最多保留 max_per_group 个标签
            group_counter: dict[str, int] = {}
            max_per = request.max_per_group
            for r in sorted_results:
                if r.final_score <= 0.45:
                    continue
                if r.tag in guaranteed_tags:
                    # guaranteed_tags 豁免 group 上限
                    valid.append(r)
                    continue
                groups = self._tag_to_groups.get(r.tag)
                if not groups:
                    # 无 group 信息，不受限制
                    if len(valid) < request.limit:
                        valid.append(r)
                    continue
                # 检查是否有任一 group 达上限
                if any(group_counter.get(g, 0) >= max_per for g in groups):
                    continue
                if len(valid) < request.limit:
                    valid.append(r)
                    for g in groups:
                        group_counter[g] = group_counter.get(g, 0) + 1
        else:
            for r in sorted_results:
                if r.final_score <= 0.45:
                    continue
                if len(valid) < request.limit or r.tag in guaranteed_tags:
                    valid.append(r)

        cached_queries = [q for q, hit in zip(queries, hit_mask) if hit]
        return valid, keywords, extra_segments, cached_queries

    # ── CPU 并发闸门（类级信号量，所有 CPU 密集型操作共享）───────────────

    @classmethod
    def _get_cpu_sem(cls) -> asyncio.Semaphore:
        if cls._cpu_sem is None:
            cls._cpu_sem = asyncio.Semaphore(_resolve_cpu_sem_limit())
        return cls._cpu_sem

    async def search_async(self, request: SearchRequest) -> SearchResponse:
        """search() 的并发安全异步封装：共享闸门串行化 + 线程池执行。

        所有异步入口（MCP / API / UI）都应改用本方法，而非各自
        asyncio.to_thread(self.search)，以共享同一个 CPU 并发闸门。
        包含 60 秒超时，防止异常卡死导致信号量永久泄漏。
        """
        async with self._get_cpu_sem():
            return await asyncio.wait_for(
                asyncio.to_thread(self.search, request),
                timeout=120.0,
            )

    async def get_related_async(
        self,
        seed_tags: list[str],
        exclude: set[str] | None = None,
        limit: int = 20,
        show_nsfw: bool = True,
        target_categories: set[str] | None = None,
    ) -> list:
        """get_related() 的并发安全异步封装，共享同一个 CPU 闸门。"""
        async with self._get_cpu_sem():
            return await asyncio.to_thread(
                self.get_related, seed_tags, exclude, limit, show_nsfw, target_categories,
            )

    async def get_group_candidates_async(
        self,
        selected_tags: list[str],
        show_nsfw: bool = True,
    ) -> list[dict]:
        """get_group_candidates() 的并发安全异步封装，共享同一个 CPU 闸门。"""
        async with self._get_cpu_sem():
            return await asyncio.to_thread(
                self.get_group_candidates, selected_tags, show_nsfw,
            )

    async def search_artists_by_tags_async(
        self,
        tags: list[str],
        limit: int = 30,
        min_cooc: int = 5,
    ) -> list:
        """search_artists_by_tags() 的并发安全异步封装。"""
        async with self._get_cpu_sem():
            return await asyncio.to_thread(
                self.search_artists_by_tags, tags, limit, min_cooc,
            )

    async def search_artists_pipeline_async(
        self,
        query: str,
        limit: int = 30,
        min_cooc: int = 5,
        target_layers: list[str] | None = None,
        target_categories: list[str] | None = None,
    ) -> tuple:
        """search_artists_pipeline() 的并发安全异步封装。"""
        async with self._get_cpu_sem():
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.search_artists_pipeline,
                    query, limit, min_cooc, target_layers, target_categories,
                ),
                timeout=120.0,
            )

    def _apply_group_expand(self, final: dict[str, TagResult]) -> None:
        """expand 模式：提升同 group 标签的分数。"""
        BETA = 0.2
        TOP_N = 20

        sorted_items = sorted(final.values(), key=lambda r: r.final_score, reverse=True)
        top_n = min(TOP_N, len(sorted_items))
        anchor_results = sorted_items[:top_n]

        # 收集锚点结果所属的所有 group
        active_groups: set[str] = set()
        for r in anchor_results:
            groups = self._tag_to_groups.get(r.tag)
            if groups:
                active_groups.update(groups)

        if not active_groups:
            return

        # 预计算每个 group 的锚点最大分
        group_max_score: dict[str, float] = {}
        for g in active_groups:
            group_max_score[g] = max(
                (r.final_score for r in anchor_results
                 if g in self._tag_to_groups.get(r.tag, set())),
                default=0.0,
            )

        # 对所有候选应用 boost
        for r in final.values():
            groups = self._tag_to_groups.get(r.tag)
            if not groups:
                continue
            overlap = groups & active_groups
            if not overlap:
                continue
            best_group_score = max(group_max_score[g] for g in overlap)
            boost = 1.0 + BETA * best_group_score
            r.final_score = round(r.final_score * boost, 4)

    # ── 全量构建 ──────────────────────────────────────────────────────────

    def _build_full(self) -> None:
        print(f'[Engine] 全量读取 {self.csv_path} ...')
        raw_df         = self._read_csv_robust(self.csv_path)
        self.df        = self._preprocess_raw_df(raw_df)
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._encode_all_and_save()

    def _encode_all_and_save(self) -> None:
        print('[Engine] 全量编码...')
        for _, attr, col in _LAYER_SPEC:
            texts = self.df[col].tolist()
            if col == 'name':  # 英文层：编码时将下划线替换为空格
                texts = [t.replace('_', ' ') for t in texts]
            setattr(self, attr, self._encode_texts(texts))
        self._save_cache()

    # ── 增量更新 ──────────────────────────────────────────────────────────

    def _smart_update(self) -> None:
        print('[Engine] 检查增量变更...')
        t0 = time.time()

        raw_df = self._read_csv_robust(self.csv_path)
        new_df = self._preprocess_raw_df(raw_df)

        _SIG_COLS = ['cn_name', 'wiki', 'cn_core']

        def _sig(df: pd.DataFrame, iloc_idx: int) -> tuple:
            row = df.iloc[iloc_idx]
            return tuple(str(row.get(c, '')) for c in _SIG_COLS)

        cached_idx: dict[str, int] = {n: i for i, n in enumerate(self.df['name'])}
        new_idx:    dict[str, int] = {n: i for i, n in enumerate(new_df['name'])}

        added_names   = [n for n in new_idx if n not in cached_idx]
        deleted_names = [n for n in cached_idx if n not in new_idx]
        changed_names = [
            n for n in new_idx
            if n in cached_idx and _sig(new_df, new_idx[n]) != _sig(self.df, cached_idx[n])
        ]

        if not added_names and not deleted_names and not changed_names:
            print('[Engine] 数据已是最新，无需更新。')
            return

        print(f'[Engine] 变更 → 新增: {len(added_names)}  修改: {len(changed_names)}  删除: {len(deleted_names)}')

        if deleted_names:
            keep_mask = ~self.df['name'].isin(set(deleted_names))
            keep_pos  = [i for i, v in enumerate(keep_mask) if v]
            self.df = self.df[keep_mask].reset_index(drop=True)
            for _, attr, _ in _LAYER_SPEC:
                setattr(self, attr, getattr(self, attr)[keep_pos])
            cached_idx = {n: i for i, n in enumerate(self.df['name'])}

        if changed_names:
            changed_rows = new_df[new_df['name'].isin(set(changed_names))].reset_index(drop=True)
            _vecs = {}
            for _, attr, col in _LAYER_SPEC:
                texts = changed_rows[col].tolist()
                if col == 'name':  # 英文层：编码时将下划线替换为空格
                    texts = [t.replace('_', ' ') for t in texts]
                _vecs[attr] = self._encode_texts(texts)
            for j, name in enumerate(changed_rows['name']):
                ci = cached_idx[name]
                for _, attr, _ in _LAYER_SPEC:
                    getattr(self, attr)[ci] = _vecs[attr][j]
                for col in changed_rows.columns:
                    self.df.at[ci, col] = changed_rows.at[j, col]

        if added_names:
            added_rows = new_df[new_df['name'].isin(set(added_names))].reset_index(drop=True)
            for _, attr, col in _LAYER_SPEC:
                texts = added_rows[col].tolist()
                if col == 'name':  # 英文层：编码时将下划线替换为空格
                    texts = [t.replace('_', ' ') for t in texts]
                vecs = self._encode_texts(texts)
                setattr(self, attr, torch.cat([getattr(self, attr), vecs], dim=0))
            self.df = pd.concat([self.df, added_rows], ignore_index=True)

        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        self._name_to_idx = {n: i for i, n in enumerate(self.df['name'])}
        self._tag_names_set = set(self._name_to_idx.keys())
        self._rebuild_arrays_from_df()
        self._load_groups()
        self._normalize_embeddings()
        self._save_cache()
        print(f'[Engine] 增量更新完成，耗时 {time.time() - t0:.2f}s（共 {len(self.df)} 条）')

    # ── 缓存 I/O ──────────────────────────────────────────────────────────

    def _save_cache(self) -> None:
        self.paths.ensure_dir()
        st_save(
            {attr: getattr(self, attr).half() for _, attr, _ in _LAYER_SPEC},
            str(self.paths.embeddings),
        )
        save_cols = ['name', 'cn_name', 'cn_core', 'wiki', 'nsfw', 'category', 'post_count']
        self.df[save_cols].to_parquet(str(self.paths.metadata), index=False)

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.paths.meta_json, 'w', encoding='utf-8') as f:
            json.dump({
                'schema_version': SCHEMA_VERSION,
                'updated_at': current_time,
            }, f, ensure_ascii=False, indent=4)

        print(f'[Engine] 缓存保存完成（{len(self.df)} 条记录），生成时间: {current_time}')

    def _load_from_cache(self) -> None:
        t0 = time.time()
        emb_path = str(self.paths.embeddings)
        meta_path = str(self.paths.metadata)
        emb_size_mb = self.paths.embeddings.stat().st_size / 1024 / 1024
        meta_size_mb = self.paths.metadata.stat().st_size / 1024 / 1024

        # ── 步骤 1/3: embedding 文件（逐层加载以显示进度）─────────────────
        n_layers = len(_LAYER_SPEC)
        print(f'  [1/3] 加载 embedding ({emb_size_mb:.0f} MB, {n_layers} 层) ...')
        with safe_open(emb_path, framework="pt", device=self.device) as f:
            for i, (name, attr, _) in enumerate(_LAYER_SPEC, 1):
                print(f'    [{i}/{n_layers}] {attr:12s} ...', end=' ', flush=True)
                _t = time.time()
                tensor = f.get_tensor(attr)
                setattr(self, attr, tensor.float())
                print(f'✓ {time.time() - _t:.1f}s')
        print(f'  [1/3] ✓ {time.time() - t0:.1f}s')

        # ── 步骤 2/3: 元数据 ──────────────────────────────────────────────
        print(f'  [2/3] 加载元数据 ({meta_size_mb:.0f} MB) ...', end=' ', flush=True)
        _t = time.time()
        self.df = pd.read_parquet(meta_path)
        print(f'✓ {time.time() - _t:.1f}s  ({len(self.df):,} 条)')

        # ── 步骤 3/3: 统计信息 ────────────────────────────────────────────
        print(f'  [3/3] 计算统计信息 ...', end=' ', flush=True)
        _t = time.time()
        self.max_log_count = float(np.log1p(self.df['post_count'].max()))
        print(f'✓ {time.time() - _t:.1f}s')

        print(f'[Engine] 缓存加载完成 (总耗时 {time.time() - t0:.1f}s)')

    def _cached_schema_version(self) -> int:
        try:
            with open(self.paths.meta_json, 'r', encoding='utf-8') as f:
                return int(json.load(f).get('schema_version', 1))
        except Exception:
            return 0

    # ── 编码 & 预处理 ──────────────────────────────────────────────────────

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_tensor=True,
        ).float()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        print(f'[Engine] 加载模型 (path={self.model_path}, device={self.device})...')
        try:
            self.model = SentenceTransformer(self.model_path, device=self.device)
        except Exception as e:
            print(f'[Engine] 指定路径加载失败，尝试重新解析: {e}')
            fallback = resolve_model_path()
            self.model = SentenceTransformer(fallback, device=self.device)

    def _read_csv_robust(self, path: str) -> pd.DataFrame:
        for enc in ['utf-8', 'gbk', 'gb18030']:
            try:
                return pd.read_csv(path, dtype=str, encoding=enc).fillna('')
            except UnicodeDecodeError:
                continue
        raise ValueError('CSV 读取失败，请检查编码')

    def _preprocess_raw_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.dropna(subset=['name'], inplace=True)
        df = df[df['name'].str.strip() != '']
        for col in ['cn_name', 'category', 'wiki', 'nsfw']:
            if col not in df.columns:
                df[col] = ''
        df['category']   = df['category'].fillna('0')
        df['nsfw']       = df['nsfw'].fillna('0')
        for char in ['，', '|', '、']:
            df['cn_name'] = df['cn_name'].str.replace(char, ',', regex=False)
        if 'post_count' not in df.columns:
            df['post_count'] = 0
        df['post_count'] = pd.to_numeric(df['post_count'], errors='coerce').fillna(0)
        df['cn_name']    = df['cn_name'].fillna('')
        df['wiki']       = df['wiki'].fillna('')
        df['cn_core']    = df['cn_name'].str.split(',', n=1).str[0].str.strip().fillna('')
        df.drop_duplicates(subset=['name'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _setup_jieba_from_memory(self) -> None:
        if self.df is None:
            return
        unique_words: set[str] = set()
        for text in self.df['cn_name'].dropna().astype(str):
            for part in text.replace(',', ' ').split():
                part = part.strip()
                if len(part) > 1:
                    unique_words.add(part)
        for word in unique_words:
            jieba.add_word(word, 2000)

    def _detect_intent(self, query: str) -> dict[str, float]:
        """
        根据查询词特征返回各视图的语义分系数。
        系数 > 1 表示加权，< 1 表示降权。
        """
        stripped = query.replace(' ', '')
        cn_chars = sum(1 for c in stripped if '\u4e00' <= c <= '\u9fff')
        en_chars = sum(1 for c in stripped if c.isascii() and c.isalpha())
        total    = max(len(stripped), 1)
        is_long  = len(query) > 8
        is_cn    = cn_chars / total > 0.5
        is_en    = en_chars / total > 0.5

        if is_long and is_cn:
            return {'英文': 0.8, '中文核心词': 0.9, '中文扩展词': 1.1, '释义': 1.4}
        if is_long and is_en:
            return {'英文': 1.0, '中文核心词': 0.8, '中文扩展词': 0.9, '释义': 1.3}
        if is_cn:
            return {'英文': 0.8, '中文核心词': 1.3, '中文扩展词': 1.1, '释义': 0.6}
        if is_en:
            return {'英文': 1.3, '中文核心词': 1.0, '中文扩展词': 0.8, '释义': 0.6}
        return {'英文': 1.0, '中文核心词': 1.0, '中文扩展词': 1.0, '释义': 1.0}

    # ── 英文分词辅助 ──────────────────────────────────────────────────────

    _EN_MAX_COMPOUND = 4  # 复合标签最大单词数

    def _tokenize_en_chunk(self, chunk: str) -> list[str]:
        """对一段英文文本做分词：清洗 → 按空格切分 → 过滤停用词/纯数 → 变体规范化 → 合并已知复合标签。

        变体规范化指：未直接命中 tag_set 的 token 尝试 `连字符→下划线` /
        复数还原（s/es/ies→y），仅在变体落在 tag_set 才采用。
        """
        cleaned = re.sub(r'[,()\[\]{}:]', ' ', chunk)
        raw = [p for p in cleaned.split() if p]
        tag_set = getattr(self, '_tag_names_set', None)
        tokens: list[str] = []
        for part in raw:
            low = part.lower()
            # 已知标签直接保留（如用户输入了带下划线的 tag 名）
            if tag_set and low in tag_set:
                tokens.append(low)
                continue
            if low in STOP_WORDS:
                continue
            # 连字符/复数变体探测（仅当变体落在 tag_set 才采用）
            variant = self._resolve_tag_variant(low)
            if variant:
                tokens.append(variant)
                continue
            if part.isdigit():      # 仅过滤纯数字，保留 3d/2b 等含数字的词
                continue
            tokens.append(low)
        if not tokens:
            return []
        return self._merge_compound_english(tokens)

    def _resolve_tag_variant(self, low: str) -> str | None:
        """对未直接命中 tag_set 的英文 token 探测常见变体。

        覆盖：连字符→下划线（cat-ears → cat_ears）、复数→单数
        （cats → cat / dresses → dress / bunnies → bunny）。
        仅在变体落在 _tag_names_set 中才返回，避免 'glass'→'glas' 之类误伤。

        Returns:
            命中的 tag 名；都不命中返回 None。
        """
        tag_set = getattr(self, '_tag_names_set', None)
        if not tag_set:
            return None

        # 连字符直接换成下划线若直接命中 tag 则优先返回
        bases = [low]
        if '-' in low:
            hyphen_normalized = low.replace('-', '_')
            if hyphen_normalized in tag_set:
                return hyphen_normalized
            bases.append(hyphen_normalized)

        # 对每个基串尝试复数还原（按"剥离短→长"顺序，避免 houses→hous 误判）
        for base in bases:
            if base.endswith('s') and len(base) > 1:
                v = base[:-1]
                if v in tag_set:
                    return v
            if base.endswith('es') and len(base) > 2:
                v = base[:-2]
                if v in tag_set:
                    return v
            if base.endswith('ies') and len(base) > 3:
                v = base[:-3] + 'y'
                if v in tag_set:
                    return v
        return None

    def _merge_compound_english(self, tokens: list[str]) -> list[str]:
        """将相邻英文单词合并为已知的 Danbooru 下划线复合标签。

        贪心最长匹配：优先 4-gram，依次递减到 bigram，匹配到即消耗。
        例: ['beam', 'rifle', 'scope'] → 如果 'beam_rifle' 是标签则合并，
             否则保留原样。
        """
        tag_set = getattr(self, '_tag_names_set', None)
        if tag_set is None or len(tokens) < 2:
            return tokens

        result: list[str] = []
        i = 0
        max_w = min(self._EN_MAX_COMPOUND, len(tokens))
        while i < len(tokens):
            merged = False
            for w in range(max_w, 1, -1):          # 4, 3, 2
                if i + w > len(tokens):
                    continue
                candidate = '_'.join(tokens[i:i + w])
                if candidate in tag_set:
                    result.append(candidate)
                    i += w
                    merged = True
                    break
            if not merged:
                result.append(tokens[i])
                i += 1
        return result

    # ── 查询切分 ──────────────────────────────────────────────────────────

    def _smart_split(self, text: str) -> tuple[list[str], list[str]]:
        """将查询文本拆分为关键词列表，同时返回从句级片段。

        先把文本切成交替的 CN-region（含 CJK 字符）与 EN-region（无 CJK）；
        中英文用各自的规则处理：

        1. CN-region 按空格/CJK 标点切出"子句"（segments），中文自然语句里
           这些符号是显式概念边界。子句的 token 切分策略由"整句是否含任何
           分隔符"决定：
           - 整句无任何分隔符 → 视作自然句，jieba 切分；
           - 整句有分隔符 → 用户已标边界，每个短纯 CJK 子句原子保留，
             超过 _ATOMIC_CJK_MAX_LEN 才走 jieba。
        2. EN-region 仅走 _tokenize_en_chunk（停用词过滤 + 复合词合并），
           不产出 segments——英文里空格是词内分隔而非概念边界。

        Returns:
            (tokens, segments):
            - tokens: 处理后的关键词列表
            - segments: CN-region 切出的子句片段；纯英文查询为空列表
        """
        user_pieces = [s.strip() for s in re.split(r'[\s\n\r，、；。]+', text) if s.strip()]
        if not user_pieces:
            return [], []
        has_boundary = len(user_pieces) > 1   # 整句是否含任何用户标注的概念边界

        tokens: list[str] = []
        segments: list[str] = []

        # 把文本切成交替的 CN-region 与 EN-region。
        # CN-region 允许内部以空格/CJK 标点连接相邻 CJK 块。
        cjk_region = r'[一-龥]+(?:[\s\n\r，、；。]+[一-龥]+)*'
        parts = re.split(f'({cjk_region})', text)

        for part in parts:
            if not part.strip():
                continue

            if re.search(r'[一-龥]', part):
                # CN region：产出子句 + tokens
                cn_segs = [s.strip() for s in re.split(r'[\s\n\r，、；。]+', part) if s.strip()]
                for seg in cn_segs:
                    segments.append(seg)
                    if has_boundary and re.match(r'^[一-龥]+$', seg) and len(seg) <= _ATOMIC_CJK_MAX_LEN:
                        tokens.append(seg)              # 短 → 原子概念
                    else:
                        for chunk in re.split(r'([一-龥]+)', seg):
                            if not chunk.strip():
                                continue
                            if re.match(r'[一-龥]+', chunk):
                                tokens.extend(jieba.cut(chunk))
                            else:
                                tokens.extend(self._tokenize_en_chunk(chunk))
            else:
                # EN region：仅 tokenize，不产出子句
                tokens.extend(self._tokenize_en_chunk(part))

        return tokens, segments

    # ── 关联推荐 ──────────────────────────────────────────────────────────

    def get_related(
            self,
            seed_tags: list[str],
            exclude: set[str] | None = None,
            limit: int = 20,
            show_nsfw: bool = True,
            target_categories: set[str] | None = None,
    ) -> list:
        from .models import RelatedTag
        import math

        if not self.cooc or not seed_tags:
            return []
        exclude = exclude or set()

        related_key = (
            tuple(sorted(seed_tags)), tuple(sorted(exclude)), limit, show_nsfw,
            tuple(sorted(target_categories)) if target_categories is not None else None,
        )
        cached = self._related_cache.get(related_key)
        if cached is not None:
            return cached

        # 估算语料库总大小 N，取数据集中发帖量的最大值，并设置合理下限
        total_posts = float(max(self.df['post_count'].max(), 7000000.0))

        npmi_scores: dict[str, float] = {}
        total_cooc: dict[str, int] = {}
        tag_sources: dict[str, list[str]] = {}
        name_to_idx = self._name_to_idx
        arr_post_count = self._arr_post_count

        for seed in seed_tags:
            if seed not in name_to_idx:
                continue

            seed_count = float(arr_post_count[name_to_idx[seed]] or 1)

            for neighbor, cnt in self.cooc.get(seed, []):
                if neighbor in exclude or neighbor == seed:
                    continue
                if neighbor not in name_to_idx:
                    continue

                neighbor_count = float(arr_post_count[name_to_idx[neighbor]] or 1)

                cooc = min(float(cnt), seed_count, neighbor_count)
                if cooc <= 0:
                    continue

                # 计算分子：(Cooc * N) / (Count(A) * Count(B))
                numerator = (cooc * total_posts) / (seed_count * neighbor_count)

                # 忽略负相关或完全不相关的词条
                if numerator <= 1.0:
                    continue

                pmi = math.log(numerator)

                # 计算分母：-log(P(A, B))
                p_a_b = cooc / total_posts
                if p_a_b >= 1.0:
                    npmi = 1.0
                else:
                    npmi = pmi / -math.log(p_a_b)

                # 多词条搜索时累加 NPMI
                npmi_scores[neighbor] = npmi_scores.get(neighbor, 0.0) + npmi
                total_cooc[neighbor] = total_cooc.get(neighbor, 0) + cnt
                tag_sources.setdefault(neighbor, []).append(seed)

        if not npmi_scores:
            return []

        # 归一化用于前端展示
        max_score = max(npmi_scores.values())

        sorted_candidates = sorted(npmi_scores.items(), key=lambda x: x[1], reverse=True)

        # ── 构建 NPMI 结果 ─────────────────────────────────────────────
        results: list = []

        for tag_name, raw_score in sorted_candidates:
            if len(results) >= limit:
                break
            idx = name_to_idx[tag_name]
            nsfw = self._arr_nsfw[idx]
            if nsfw == '1' and not show_nsfw:
                continue
            cat = CAT_MAP.get(self._arr_category[idx], 'Other')
            if target_categories is not None and cat not in target_categories:
                continue
            results.append(RelatedTag(
                tag=tag_name,
                cn_name=str(self._arr_cn_name[idx]),
                category=cat,
                nsfw=nsfw,
                cooc_count=total_cooc.get(tag_name, 0),
                cooc_score=round(raw_score / max_score, 4),
                sources=tag_sources.get(tag_name, []),
                post_count=int(self._arr_post_count[idx]),
                wiki=str(self._arr_wiki[idx]) if self._arr_wiki is not None else '',
            ))

        self._related_cache.put(related_key, results)
        return results

    def get_group_candidates(
            self,
            selected_tags: list[str],
            show_nsfw: bool = True,
    ) -> list[dict]:
        """根据已选标签，返回候选 Group 及其成员标签。"""
        if not self._tag_to_groups or not selected_tags:
            return []

        group_sources: dict[str, list[str]] = {}
        for tag_name in selected_tags:
            groups = self._tag_to_groups.get(tag_name)
            if groups:
                for g in groups:
                    group_sources.setdefault(g, []).append(tag_name)

        if not group_sources:
            return []

        selected_set = set(selected_tags)
        sorted_groups = sorted(group_sources.items(), key=lambda x: -len(x[1]))

        results = []
        for group_name, sources in sorted_groups:
            member_idxs = self._group_to_tags_idx.get(group_name)
            if member_idxs is None:
                continue

            tags = []
            for idx in member_idxs:
                tag_name = str(self._arr_name[idx])
                if tag_name in selected_set:
                    continue
                nsfw = self._arr_nsfw[idx]
                if nsfw == '1' and not show_nsfw:
                    continue
                cat = CAT_MAP.get(self._arr_category[idx], 'Other')
                tags.append({
                    'tag': tag_name,
                    'cn_name': str(self._arr_cn_name[idx]),
                    'category': cat,
                    'nsfw': nsfw,
                    'post_count': int(self._arr_post_count[idx]),
                    'wiki': str(self._arr_wiki[idx]) if self._arr_wiki is not None else '',
                })

            tags.sort(key=lambda x: -x['post_count'])
            cn_name = self._group_cn_names.get(group_name, group_name)
            results.append({
                'group': group_name,
                'group_cn_name': cn_name,
                'hit_count': len(sources),
                'sources': sources,
                'tags': tags,
            })

        return results

    # ── 画师查找 ──────────────────────────────────────────────────────────

    def search_artists_by_tags(
        self,
        tags: list[str],
        limit: int = 30,
        min_cooc: int = 5,
    ) -> list:
        """按标签查找画师：聚合多个标签的 NPMI 得分，返回排名靠前的画师。"""
        from .models import ArtistResult

        if not self._tag_artist_index or not tags:
            return []

        artist_scores: dict[str, float] = {}
        artist_cooc: dict[str, int] = {}
        artist_post_count: dict[str, int] = {}
        artist_sources: dict[str, list[str]] = {}
        artist_hits: dict[str, int] = {}

        for tag in tags:
            tag = tag.strip().lower()
            if not tag or tag not in self._tag_artist_index:
                continue
            for artist, npmi, cooc, post_count in self._tag_artist_index[tag]:
                if cooc < min_cooc:
                    continue
                artist_scores[artist] = artist_scores.get(artist, 0.0) + npmi
                artist_cooc[artist] = artist_cooc.get(artist, 0) + cooc
                artist_post_count[artist] = max(artist_post_count.get(artist, 0), post_count)
                artist_sources.setdefault(artist, []).append(tag)
                artist_hits[artist] = artist_hits.get(artist, 0) + 1

        if not artist_scores:
            return []

        scored = []
        for artist, raw_score in artist_scores.items():
            hit_bonus = 1.0 + 0.3 * (artist_hits[artist] - 1)
            final_score = raw_score * hit_bonus
            scored.append((
                artist, final_score, artist_cooc[artist],
                artist_post_count[artist], artist_sources[artist],
                artist_hits[artist],
            ))

        scored.sort(key=lambda x: -x[1])
        results = []
        for artist, score, cooc, post_count, sources, hits in scored[:limit]:
            results.append(ArtistResult(
                artist=artist,
                score=round(score, 4),
                cooc_count=cooc,
                post_count=post_count,
                sources=sources,
                hit_count=hits,
            ))
        return results

    def get_artist_top_tags(self, artist_names: list[str], top_n: int = 10,
                            show_nsfw: bool = True) -> dict[str, list[str]]:
        """获取画师的 top-N 共现标签（含中文简称）。

        Returns:
            {artist: [f"{tag} ({cn_short})", ...]}
        """
        result: dict[str, list[str]] = {}
        for artist in artist_names:
            entries = self._artist_top_tags.get(artist, [])
            items: list[str] = []
            for tag, npmi, cooc in entries:
                if len(items) >= top_n:
                    break
                # NSFW 过滤
                if not show_nsfw and self._name_to_idx is not None and tag in self._name_to_idx:
                    idx = self._name_to_idx[tag]
                    if self._arr_nsfw is not None and self._arr_nsfw[idx] == '1':
                        continue
                cn_short = ""
                if self._name_to_idx is not None and tag in self._name_to_idx:
                    idx = self._name_to_idx[tag]
                    cn_full = str(self._arr_cn_name[idx]) if self._arr_cn_name is not None else ""
                    cn_short = cn_full.split(',')[0].strip() if cn_full else ""
                display = f"{tag}"
                if cn_short:
                    display += f" ({cn_short})"
                items.append(display)
            result[artist] = items
        return result

    @staticmethod
    def _normalize_artist_name(name: str) -> str:
        """Normalize user-entered artist names toward Danbooru tag form."""
        text = str(name or "").strip().lower()
        if text.startswith("@"):
            text = text[1:].strip()
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")

    @staticmethod
    def _compact_artist_key(name: str) -> str:
        return re.sub(r"[\W_]+", "", str(name or "").lower())

    @staticmethod
    def _edit_distance_at_most_one(left: str, right: str) -> bool:
        if left == right:
            return True
        if abs(len(left) - len(right)) > 1:
            return False

        if len(left) == len(right):
            mismatches = 0
            for a, b in zip(left, right):
                if a != b:
                    mismatches += 1
                    if mismatches > 1:
                        return False
            return True

        short, long = (left, right) if len(left) < len(right) else (right, left)
        i = j = edits = 0
        while i < len(short) and j < len(long):
            if short[i] == long[j]:
                i += 1
                j += 1
                continue
            edits += 1
            if edits > 1:
                return False
            j += 1
        return True

    def search_artist_rows(
        self, query: str, limit: int = 20, show_nsfw: bool = True,
    ) -> list[TagResult]:
        """Return artist rows whose normalized name is within edit distance 1."""
        normalized = self._normalize_artist_name(query)
        compact_query = self._compact_artist_key(normalized)
        if not compact_query:
            return []

        matches: list[TagResult] = []
        for artist in sorted(self._artist_top_tags.keys()):
            if not self._edit_distance_at_most_one(compact_query, self._compact_artist_key(artist)):
                continue
            top_tags = self.get_artist_top_tags(
                [artist], top_n=10, show_nsfw=show_nsfw,
            ).get(artist, [])
            matches.append(TagResult(
                tag=artist,
                cn_name="画师标签",
                category="Artist",
                nsfw="0",
                final_score=1.0,
                semantic_score=1.0,
                count=int(self._artist_post_count.get(artist, 0)),
                source=query,
                layer="artist",
                wiki="",
                artist_top_tags=top_tags,
            ))

        matches.sort(key=lambda r: (r.count, r.tag), reverse=True)
        return matches[:limit]

    def resolve_artist_name(self, artist_name: str) -> dict[str, Any]:
        """Resolve a user-entered artist name to the artist co-occurrence index."""
        artists = set(self._artist_top_tags.keys())
        normalized = self._normalize_artist_name(artist_name)

        if artist_name in artists:
            return {
                "artist": artist_name,
                "matched_by": "exact",
                "candidates": [],
            }

        if normalized in artists:
            return {
                "artist": normalized,
                "matched_by": "normalized_exact",
                "candidates": [],
            }

        compact_query = self._compact_artist_key(artist_name)
        compact_map: dict[str, list[str]] = {}
        for artist in artists:
            compact_map.setdefault(self._compact_artist_key(artist), []).append(artist)
        compact_matches = compact_map.get(compact_query, [])
        if len(compact_matches) == 1:
            return {
                "artist": compact_matches[0],
                "matched_by": "compact_exact",
                "candidates": [],
            }
        if len(compact_matches) > 1:
            return {
                "artist": None,
                "matched_by": "ambiguous_compact",
                "candidates": sorted(compact_matches)[:10],
            }

        close = difflib.get_close_matches(normalized, sorted(artists), n=5, cutoff=0.78)
        if len(close) == 1:
            return {
                "artist": close[0],
                "matched_by": "fuzzy",
                "candidates": close,
            }
        return {
            "artist": None,
            "matched_by": "not_found",
            "candidates": close,
        }

    def resolve_tag_name(self, tag_name: str) -> dict[str, Any]:
        """Resolve a user-entered tag name to the canonical tag index without semantic search."""
        tags = set(self._name_to_idx.keys())
        normalized = self._normalize_artist_name(tag_name)

        if tag_name in tags:
            return {
                "tag": tag_name,
                "matched_by": "exact",
                "candidates": [],
            }

        if normalized in tags:
            return {
                "tag": normalized,
                "matched_by": "normalized_exact",
                "candidates": [],
            }

        # 官方 Alias 是确定性名称替换，优先于复数、紧凑拼写与模糊纠错。
        alias = self._tag_aliases.get(normalized)
        if alias is not None:
            target = alias.consequent_name
            if alias.target_in_tag_db and target in tags:
                return {
                    "tag": target,
                    "matched_by": "alias",
                    "candidates": [],
                    "alias_from": normalized,
                    "alias_target": target,
                }
            return {
                "tag": None,
                "matched_by": "alias_target_missing",
                "candidates": [target],
                "alias_from": normalized,
                "alias_target": target,
            }

        plural = f"{normalized}s"
        if plural in tags:
            return {
                "tag": plural,
                "matched_by": "plural_exact",
                "candidates": [],
            }
        if normalized.endswith("s") and normalized[:-1] in tags:
            return {
                "tag": normalized[:-1],
                "matched_by": "singular_exact",
                "candidates": [],
            }

        compact_query = self._compact_artist_key(tag_name)
        compact_map: dict[str, list[str]] = {}
        for tag in tags:
            compact_map.setdefault(self._compact_artist_key(tag), []).append(tag)
        compact_matches = compact_map.get(compact_query, [])
        if len(compact_matches) == 1:
            return {
                "tag": compact_matches[0],
                "matched_by": "compact_exact",
                "candidates": [],
            }
        if len(compact_matches) > 1:
            return {
                "tag": None,
                "matched_by": "ambiguous_compact",
                "candidates": sorted(compact_matches)[:10],
            }

        close = difflib.get_close_matches(normalized, sorted(tags), n=5, cutoff=0.78)
        if len(close) == 1:
            return {
                "tag": close[0],
                "matched_by": "fuzzy",
                "candidates": close,
            }
        return {
            "tag": None,
            "matched_by": "not_found",
            "candidates": close,
        }

    def get_tag_workspace_metadata(self, tag_name: str) -> dict[str, Any] | None:
        """Return stable local metadata used by Prompt import and workspace grouping."""
        if tag_name not in self._name_to_idx:
            return None
        idx = self._name_to_idx[tag_name]
        return {
            "tag": tag_name,
            "cn_name": str(self._arr_cn_name[idx]) if self._arr_cn_name is not None else "",
            "category": CAT_MAP.get(
                str(self._arr_category[idx]) if self._arr_category is not None else "",
                "Other",
            ),
            "nsfw": str(self._arr_nsfw[idx]) if self._arr_nsfw is not None else "0",
            "groups": sorted(self._tag_to_groups.get(tag_name, set())),
        }

    def get_artist_profile(self, artist_name: str, top_n: int = 20,
                           show_nsfw: bool = True) -> dict[str, Any]:
        """Return a resolved artist and their common co-occurring tags."""
        resolved = self.resolve_artist_name(artist_name)
        artist = resolved["artist"]
        if not artist:
            return {
                "error": "artist_not_found",
                "input": artist_name,
                "matched_by": resolved["matched_by"],
                "candidates": resolved["candidates"],
                "message": (
                    "未在画师共现库中找到唯一画师；这不代表 Danbooru 标签不存在，"
                    "也不应使用 search_tags 验证画师名。"
                ),
            }

        top_tags: list[dict[str, str]] = []
        for tag, _npmi, _cooc in self._artist_top_tags.get(artist, []):
            if len(top_tags) >= top_n:
                break
            if not show_nsfw and self._name_to_idx is not None and tag in self._name_to_idx:
                idx = self._name_to_idx[tag]
                if self._arr_nsfw is not None and self._arr_nsfw[idx] == '1':
                    continue

            cn_short = ""
            if self._name_to_idx is not None and tag in self._name_to_idx:
                idx = self._name_to_idx[tag]
                cn_full = str(self._arr_cn_name[idx]) if self._arr_cn_name is not None else ""
                cn_short = cn_full.split(',')[0].strip() if cn_full else ""
            top_tags.append({
                "tag": tag,
                "cn_name": cn_short,
            })

        return {
            "artist": artist,
            "input": artist_name,
            "matched_by": resolved["matched_by"],
            "post_count": self._artist_post_count.get(artist, 0),
            "top_tags": top_tags,
            "note": (
                "这些是该画师作品中常共现的标签，可作为风格参考；"
                "不是对画风的完整语义描述。"
            ),
        }

    def search_artists_pipeline(
        self,
        query: str,
        limit: int = 30,
        min_cooc: int = 5,
        target_layers: list[str] | None = None,
        target_categories: list[str] | None = None,
    ) -> tuple[list, list[str], list[str], list[str]]:
        """画师查找完整管线：自然语言 → 标签搜索 → 提取最佳标签 → 画师查询。

        Returns:
            (artist_results, seed_tags, found_tags, missing_tags)
        """
        if target_layers is None:
            target_layers = ['英文', '中文扩展词', '释义', '中文核心词']
        if target_categories is None:
            target_categories = ['General', 'Character', 'Copyright']

        # Step 1: 自然语言 → 标签搜索
        tag_request = SearchRequest(
            query=query,
            top_k=10,
            limit=80,
            popularity_weight=0.15,
            show_nsfw=True,
            use_segmentation=True,
            target_layers=target_layers,
            target_categories=target_categories,
        )
        tag_response = self.search(tag_request)

        if not tag_response.results:
            return [], [], [], []

        # Step 2: 按 source 分组，每组按分数降序取前 10 个候选，找到能命中画师的第一个
        tag_artist = self._tag_artist_index
        source_candidates: dict[str, list[str]] = {}
        for r in tag_response.results:
            if r.final_score < 0.45:
                continue
            source_candidates.setdefault(r.source, []).append(r.tag)

        seed_tags: list[str] = []
        seen = set()
        for candidates in source_candidates.values():
            for tag in candidates[:10]:
                tag_lower = tag.strip().lower()
                if tag_lower in tag_artist and tag_lower not in seen:
                    seed_tags.append(tag_lower)
                    seen.add(tag_lower)
                    break

        if not seed_tags:
            return [], [], [], []

        # Step 3: 标签 → 画师查找
        artist_results = self.search_artists_by_tags(seed_tags, limit, min_cooc)

        # 统计 found / missing（seed_tags 中哪些匹配到了画师）
        matched = {s for r in artist_results for s in r.sources}
        found_tags = [t for t in seed_tags if t in matched]
        missing_tags = [t for t in seed_tags if t not in matched]

        return artist_results, seed_tags, found_tags, missing_tags

    # ── 共现数据加载 ──────────────────────────────────────────────────────

    def _load_cooc(self) -> None:
        csv_path     = Path(self.cooc_file)
        parquet_path = csv_path.with_suffix('.parquet')

        if parquet_path.is_file() and (
            not csv_path.is_file()
            or parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
        ):
            read_path  = parquet_path
            is_parquet = True
        elif csv_path.is_file():
            read_path  = csv_path
            is_parquet = False
        else:
            print(f'[Engine] 未找到共现表 ({self.cooc_file})，关联推荐功能不可用。')
            return

        print(f'[Engine] 加载共现表 ({read_path.name})...')
        t0 = time.time()
        try:
            if is_parquet:
                df = pd.read_parquet(str(read_path))
            else:
                df = self._read_csv_robust(str(read_path))
                df['count'] = pd.to_numeric(df['count'], errors='coerce').fillna(0).astype(int)
                df.to_parquet(str(parquet_path), index=False)
                print(f'[Engine] 已将共现表缓存为 {parquet_path.name}，下次启动将直接加载。')

            tag_a  = df['tag_a'].astype(str).to_numpy()
            tag_b  = df['tag_b'].astype(str).to_numpy()
            counts = df['count'].astype(int).to_numpy()

            src = np.concatenate([tag_a, tag_b])
            dst = np.concatenate([tag_b, tag_a])
            cnt = np.concatenate([counts, counts])

            sort_idx = np.lexsort((-cnt, src))
            src = src[sort_idx]
            dst = dst[sort_idx]
            cnt = cnt[sort_idx]

            unique_srcs, first_pos = np.unique(src, return_index=True)
            end_pos = np.append(first_pos[1:], len(src))

            cooc: dict[str, list[tuple[str, int]]] = {}
            for s, start, end in zip(unique_srcs, first_pos, end_pos):
                cooc[s] = list(zip(dst[start:end].tolist(), cnt[start:end].tolist()))

            self.cooc = cooc
            print(
                f'[Engine] 共现表加载完成，{len(cooc):,} 个 tag，'
                f'耗时 {time.time() - t0:.2f}s'
            )
        except Exception as e:
            print(f'[Engine] 共现表加载失败: {e}')

    def _load_tag_artist_cooc(self) -> None:
        """加载标签-画师共现数据（tag_artist_cooc.parquet）。"""
        path = Path(self.tag_artist_file)
        if not path.is_file():
            print(f'[Engine] 未找到标签-画师共现表 ({self.tag_artist_file})，画师查找功能不可用。')
            return

        print(f'[Engine] 加载标签-画师共现表 ({path.name})...')
        t0 = time.time()
        try:
            import math
            df = pd.read_parquet(str(path))

            # 预计算语料库总大小和 tag → post_count 映射
            total_posts = float(max(self.df['post_count'].max(), 7_000_000))
            tag_post_count = dict(zip(self.df['name'], self.df['post_count']))

            index: dict[str, list[tuple]] = {}
            skipped = 0
            for _, row in df.iterrows():
                tag = str(row["tag"])
                artist = str(row["artist"])
                cooc = int(row["cooc_count"])
                artist_pc = int(row["artist_post_count"])

                # 获取标签的 post_count
                tag_pc = tag_post_count.get(tag)
                if tag_pc is None or tag_pc <= 0:
                    skipped += 1
                    continue

                # 即时计算 NPMI（与 get_related 相同算法）
                cooc_capped = min(float(cooc), float(tag_pc), float(artist_pc))
                if cooc_capped <= 0:
                    skipped += 1
                    continue

                numerator = (cooc_capped * total_posts) / (float(tag_pc) * float(artist_pc))
                if numerator <= 1.0:
                    skipped += 1
                    continue

                pmi = math.log(numerator)
                p_ab = cooc_capped / total_posts
                if p_ab >= 1.0:
                    npmi_val = 1.0
                else:
                    npmi_val = pmi / -math.log(p_ab)

                if math.isnan(npmi_val):
                    skipped += 1
                    continue

                index.setdefault(tag, []).append(
                    (artist, npmi_val, cooc, artist_pc))

            self._tag_artist_index = index

            # 构建反向索引：artist → top tags（按 NPMI 降序）
            artist_tags: dict[str, list[tuple]] = {}
            artist_post_count: dict[str, int] = {}
            for tag, entries in index.items():
                for artist, npmi, cooc, artist_pc in entries:
                    artist_tags.setdefault(artist, []).append((tag, npmi, cooc))
                    artist_post_count[artist] = max(
                        artist_post_count.get(artist, 0), artist_pc,
                    )
            for artist, entries in artist_tags.items():
                entries.sort(key=lambda x: -x[1])
            self._artist_top_tags = artist_tags
            self._artist_post_count = artist_post_count

            print(
                f'[Engine] 标签-画师共现表加载完成，{len(index):,} 个标签，'
                f'跳过 {skipped:,} 行，耗时 {time.time() - t0:.2f}s'
            )
        except Exception as e:
            print(f'[Engine] 标签-画师共现表加载失败: {e}')

    def _load_tag_aliases(self) -> None:
        """加载并校验本地 Alias 快照；失败时保留已有的有效快照。"""
        try:
            snapshot = load_tag_alias_snapshot(
                self.tag_alias_file,
                self.tag_alias_meta_file,
            )
        except TagAliasSnapshotError as exc:
            print(f'[Engine] Tag Alias 快照不可用，继续使用 canonical 解析: {exc}')
            return

        self._tag_aliases = snapshot.alias_by_name
        self._tag_alias_metadata = snapshot.metadata
        fetched_at = snapshot.metadata.get('fetched_at', 'unknown')
        print(
            f'[Engine] Tag Alias loaded, {snapshot.count:,} aliases, '
            f'fetched_at={fetched_at}'
        )

    def _load_groups(self) -> None:
        """加载 Tag Group 数据，构建 tag→group 和 group→idx 索引。"""
        if not Path(self.group_file).is_file():
            print('[Engine] 未找到 Tag Group 数据，group 功能不可用。')
            return

        with open(self.group_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        raw_t2g = data.get('tag_to_groups', {})
        name_to_idx = self._name_to_idx

        self._tag_to_groups = {}
        group_members: dict[str, list[int]] = {}

        for tag_name, groups in raw_t2g.items():
            if tag_name not in name_to_idx:
                continue
            group_set = set(groups)
            self._tag_to_groups[tag_name] = group_set
            idx = name_to_idx[tag_name]
            for g in group_set:
                group_members.setdefault(g, []).append(idx)

        self._group_to_tags_idx = {
            g: np.array(idxs, dtype=np.int64) for g, idxs in group_members.items()
        }

        self._group_cn_names = data.get("group_cn_names", {})

        print(f'[Engine] Tag Group loaded, {len(self._tag_to_groups)} tags, '
              f'{len(self._group_to_tags_idx)} groups, '
              f'{len(self._group_cn_names)} cn_names')
