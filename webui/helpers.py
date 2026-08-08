"""无 NiceGUI 控件依赖的前端格式化与数据适配函数。"""

import json
import os
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from core.workspace_insights import semantic_candidate_reason
from webui.constants import (
    GROUP_RENDER_TAG_LIMIT,
    SEARCH_MODE_OPTIONS,
    SEARCH_MODE_PRESETS,
)


_HISTORY_DISPLAY_TIMEZONE = timezone(timedelta(hours=8))


def format_history_time(value: object) -> str:
    raw = str(value or '').strip()
    if not raw:
        return '--'
    try:
        parsed = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_HISTORY_DISPLAY_TIMEZONE)
        return parsed.astimezone(_HISTORY_DISPLAY_TIMEZONE).strftime('%y-%m-%d %H:%M:%S')
    except ValueError:
        return raw


def format_history_settings(settings: object) -> str:
    if not isinstance(settings, dict):
        settings = {}
    mode = settings.get('search_mode')
    preset = SEARCH_MODE_PRESETS.get(mode)
    if preset and all(settings.get(key) == value for key, value in preset.items()):
        return f'预设：{mode}'
    top_k = settings.get('top_k', '--')
    limit = settings.get('limit', '--')
    segmentation = settings.get('use_segmentation')
    segmentation_text = '开启' if segmentation is True else '关闭' if segmentation is False else '--'
    return f'Top K：{top_k} · 数量上限：{limit} · 分词：{segmentation_text}'


def sanitize_restored_config(cfg: dict) -> dict:
    safe: dict = {}
    numeric_fields = {
        'top_k': (int, 1, 200),
        'limit': (int, 10, 500),
        'popularity_weight': (float, 0.0, 1.0),
        'max_per_group': (int, 1, 10),
    }
    for key, (caster, minimum, maximum) in numeric_fields.items():
        if key not in cfg:
            continue
        try:
            value = caster(cfg[key])
        except (TypeError, ValueError):
            continue
        if minimum <= value <= maximum:
            safe[key] = value
    for key in ('show_nsfw', 'use_segmentation', 'sw_semantic', 'sw_layer', 'sw_source'):
        if isinstance(cfg.get(key), bool):
            safe[key] = cfg[key]
    for key in ('selected_layers', 'selected_cats'):
        value = cfg.get(key)
        if isinstance(value, dict):
            safe[key] = {str(k): v for k, v in value.items() if isinstance(v, bool)}
    if cfg.get('prompt_format') in ('sdxl', 'nai', 'anima'):
        safe['prompt_format'] = cfg['prompt_format']
    if cfg.get('search_mode') in SEARCH_MODE_OPTIONS:
        safe['search_mode'] = cfg['search_mode']
    if cfg.get('group_mode') in ('off', 'expand', 'diverse'):
        safe['group_mode'] = cfg['group_mode']
    if isinstance(cfg.get('rows_per_page'), int) and cfg['rows_per_page'] in {0, 5, 7, 10, 15, 20, 25, 50}:
        safe['rows_per_page'] = cfg['rows_per_page']
    if isinstance(cfg.get('search_query'), str):
        safe['search_query'] = cfg['search_query'][:4_000]
    if isinstance(cfg.get('dismissed_announcement_version'), str):
        safe['dismissed_announcement_version'] = cfg['dismissed_announcement_version'][:100]
    return safe


def next_group_render_limit(current: int, total: int, page_size: int) -> int:
    if page_size <= 0:
        return total
    return min(total, max(page_size, current + page_size))


def limit_group_render_tags(tags: list[dict], visible_limit: int | None = None) -> tuple[list[dict], int]:
    limit = GROUP_RENDER_TAG_LIMIT if visible_limit is None else visible_limit
    if limit <= 0 or len(tags) <= limit:
        return tags, 0
    return tags[:limit], len(tags) - limit


def should_group_start_expanded(group_name: str, expanded_groups: set[str]) -> bool:
    return group_name in expanded_groups


def group_names_key(group_data: list[dict]) -> tuple[str, ...]:
    return tuple(sorted({str(group.get('group', '')) for group in group_data}))


def group_scroll_dom_id(group_name: str) -> str:
    safe_name = re.sub(r'[^0-9A-Za-z_-]+', '_', group_name)
    return f'group-scroll-{safe_name}'


def scroll_state_restore_script(positions: dict[str, int]) -> str:
    js_positions = json.dumps(positions)
    return f"""
        (() => {{
            const positions = {js_positions};
            const restore = () => {{
                const windowTop = positions.__window__;
                if (typeof windowTop === 'number') {{
                    window.scrollTo({{ top: windowTop, behavior: 'auto' }});
                    const root = document.scrollingElement || document.documentElement || document.body;
                    if (root) root.scrollTop = windowTop;
                }}
                for (const [id, top] of Object.entries(positions)) {{
                    if (id === '__window__' || id.endsWith('__bottom__')) continue;
                    const el = document.getElementById(id);
                    if (!el) continue;
                    const bottom = positions[`${{id}}__bottom__`];
                    el.scrollTop = typeof bottom === 'number'
                        ? Math.max(0, el.scrollHeight - bottom) : top;
                }}
            }};
            requestAnimationFrame(() => {{ restore(); requestAnimationFrame(restore); }});
            setTimeout(restore, 80);
        }})();
    """


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return os.environ.get('COMMIT_SHA', 'unknown')[:7]


def result_to_row(result, nsfw_visible: bool) -> dict:
    row = asdict(result)
    row['_nsfw_blocked'] = (result.nsfw == '1') and not nsfw_visible
    row['reason'] = semantic_candidate_reason(result.source, result.layer, result.alias_from)
    return row


def apply_nsfw_filter(rows: list[dict], show_nsfw: bool) -> list[dict]:
    result = []
    for row in rows:
        copied = dict(row)
        copied['_nsfw_blocked'] = (copied.get('nsfw') == '1') and not show_nsfw
        result.append(copied)
    return result


def format_tag_with_weight(tag: str, weight: float, fmt: str = 'sdxl') -> str:
    tag = tag.replace('(', '\\(').replace(')', '\\)')
    if fmt == 'anima':
        tag = tag.replace('_', ' ')
    if weight == 1.0:
        return tag
    if fmt == 'nai':
        return f'{weight:.1f}::{tag}::'
    return f'({tag}:{weight:.1f})'


def format_selected_tag_label(tag: str, cn_name: str = '') -> str:
    cn_first = (cn_name or '').split(',', 1)[0].strip()
    return f'{tag} | {cn_first}' if cn_first else tag
