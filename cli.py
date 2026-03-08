"""
cli.py
──────
命令行适配层（可选）。

子命令：
    search   语义搜索标签
    related  基于共现表查关联推荐

用法：
    python cli.py search "白色水手服的女孩" --limit 10 --no-nsfw
    python cli.py related "white_serafuku,sailor_collar" --limit 20
    python cli.py related "white_serafuku,sailor_collar" --no-nsfw --show-sources
"""

import argparse
import asyncio

from core.engine import DanbooruTagger
from core.models import SearchRequest


# ── search ────────────────────────────────────────────────────────────

async def cmd_search(args):
    tagger = await DanbooruTagger.get_instance()
    request = SearchRequest(
        query=args.query,
        top_k=args.top_k,
        limit=args.limit,
        popularity_weight=args.weight,
        show_nsfw=not args.no_nsfw,
        use_segmentation=not args.no_seg,
    )
    resp = await asyncio.to_thread(tagger.search, request)

    print(f"\n{'='*60}")
    print(f"查询：{args.query}  |  共 {len(resp.results)} 条结果")
    print(f"{'='*60}")
    print(f"推荐 Prompt：\n  {resp.tags_sfw if args.no_nsfw else resp.tags_all}\n")

    for r in resp.results:
        nsfw_mark = "🔴" if r.nsfw == '1' else "🟢"
        print(f"  {nsfw_mark} [{r.final_score:.3f}] {r.tag:<30}  {r.cn_name[:20]:<20}  {r.category}")


# ── related ───────────────────────────────────────────────────────────

async def cmd_related(args):
    seed_tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    if not seed_tags:
        print("错误：请提供至少一个种子标签，多个标签以逗号分隔。")
        return

    tagger = await DanbooruTagger.get_instance()
    results = await asyncio.to_thread(
        tagger.get_related,
        seed_tags,
        set(seed_tags),   # exclude 种子标签自身
        args.limit,
        not args.no_nsfw,
    )

    if not results:
        print("未找到关联推荐（共现表可能未加载，或种子标签不在库中）。")
        return

    print(f"\n{'='*60}")
    print(f"种子标签：{', '.join(seed_tags)}  |  共 {len(results)} 条推荐")
    print(f"{'='*60}")

    # 输出逗号分隔的标签串（方便直接复制使用）
    tag_list = [r.tag for r in results if not (r.nsfw == '1' and args.no_nsfw)]
    print(f"推荐标签：\n  {', '.join(tag_list)}\n")

    # 明细表
    for r in results:
        if r.nsfw == '1' and args.no_nsfw:
            continue
        nsfw_mark   = "🔴" if r.nsfw == '1' else "🟢"
        sources_str = f"  ← {', '.join(r.sources)}" if args.show_sources else ""
        print(
            f"  {nsfw_mark} [{r.cooc_score:.3f}] {r.tag:<30}"
            f"  {r.cn_name[:20]:<20}  {r.category}"
            f"  (共现:{r.cooc_count:,}){sources_str}"
        )


# ── 入口 ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Danbooru Tag CLI Searcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python cli.py search "白色水手服的女孩" --limit 10
  python cli.py related "white_serafuku,sailor_collar"
  python cli.py related "white_serafuku" --limit 30 --no-nsfw --show-sources
        """,
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── search 子命令 ──
    p_search = sub.add_parser('search', help='语义搜索标签')
    p_search.add_argument('query',       help='搜索词（支持中英文自然语言）')
    p_search.add_argument('--top-k',   type=int,   default=5,    help='每层返回数量（默认 5）')
    p_search.add_argument('--limit',   type=int,   default=20,   help='结果上限（默认 20）')
    p_search.add_argument('--weight',  type=float, default=0.15, help='热度权重（默认 0.15）')
    p_search.add_argument('--no-nsfw', action='store_true',      help='过滤 NSFW 内容')
    p_search.add_argument('--no-seg',  action='store_true',      help='禁用智能分词')

    # ── related 子命令 ──
    p_related = sub.add_parser('related', help='基于共现表查关联推荐')
    p_related.add_argument('tags',           help='种子标签，以英文逗号分隔（如 white_serafuku,sailor_collar）')
    p_related.add_argument('--limit',        type=int, default=50, help='推荐结果上限（默认 50）')
    p_related.add_argument('--no-nsfw',      action='store_true',  help='过滤 NSFW 内容')
    p_related.add_argument('--show-sources', action='store_true',  help='显示每条推荐由哪个种子触发')

    args = parser.parse_args()
    if args.cmd == 'search':
        await cmd_search(args)
    elif args.cmd == 'related':
        await cmd_related(args)


if __name__ == "__main__":
    asyncio.run(main())
