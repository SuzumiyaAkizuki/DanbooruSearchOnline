"""不依赖 NiceGUI 控件的页面会话状态。"""

from dataclasses import dataclass, field
from typing import Any

from core.workspace import empty_favorites, empty_history, new_workspace
from webui.constants import LOCAL_STORAGE_NAMES


@dataclass
class PageState:
    """单个浏览器页面的纯数据状态；控件引用由控制器单独持有。"""

    full_table_data: list[dict] = field(default_factory=list)
    current_segments: list[str] = field(default_factory=list)
    current_keywords: list[str] = field(default_factory=list)
    current_cached_queries: set[str] = field(default_factory=set)
    current_filter_keyword: str = 'ALL'
    current_query_str: str = ''
    full_tags_str: str = ''
    full_tags_str_sfw: str = ''

    current_related: list[Any] = field(default_factory=list)
    chip_extra_selected: set[str] = field(default_factory=set)
    selected_order: list[str] = field(default_factory=list)
    rendered_selected_chip_tags: set[str] = field(default_factory=set)
    workspace: dict = field(default_factory=new_workspace)
    history: dict = field(default_factory=empty_history)
    favorites: dict = field(default_factory=empty_favorites)
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)
    pending_selection_meta: dict[str, dict[str, str]] = field(default_factory=dict)
    workspace_artist_tags: set[str] = field(default_factory=set)

    tag_weights: dict[str, float] = field(default_factory=dict)
    prompt_format: str = 'sdxl'
    selected_layers: dict[str, bool] = field(default_factory=lambda: {
        '英文': True,
        '中文扩展词': True,
        '释义': True,
        '中文核心词': True,
        'artist': True,
    })
    selected_categories: dict[str, bool] = field(default_factory=lambda: {
        'General': True,
        'Copyright': True,
        'Character': True,
    })

    related_results: list[Any] = field(default_factory=list)
    related_show_nsfw: bool = True
    related_page: int = 1
    related_page_count: int = 0
    group_render_limits: dict[str, int] = field(default_factory=dict)
    group_expanded_names: set[str] = field(default_factory=set)
    group_scroll_positions: dict[str, int] = field(default_factory=dict)
    group_render_key: tuple[str, ...] = ()
    group_candidate_sources: dict[str, str] = field(default_factory=dict)

    artist_rec_rows: list[dict] = field(default_factory=list)
    artist_rec_results: list[Any] = field(default_factory=list)
    artist_rec_top_tags: dict[str, list[str]] = field(default_factory=dict)
    artist_rec_show_nsfw: bool = True
    artist_rec_sources: dict[str, str] = field(default_factory=dict)
    artist_rec_page: int = 1
    artist_rec_page_count: int = 0
    current_artist_rec_tags: set[str] = field(default_factory=set)
    artist_result_tags: set[str] = field(default_factory=set)
    last_recommendation_seed_tags: list[str] = field(default_factory=list)
    pending_recommendation_request: dict | None = None
    recommendation_generation: int = 0

    storage_states: dict[str, str] = field(
        default_factory=lambda: {name: 'pending' for name in LOCAL_STORAGE_NAMES}
    )
    storage_session_dirty: set[str] = field(default_factory=set)
    storage_applying: set[str] = field(default_factory=set)
    storage_raw_values: dict[str, str | None] = field(default_factory=dict)


class StateField:
    """把控制器的旧属性名兼容地代理到 ``controller.state``。"""

    def __init__(self, state_name: str | None = None):
        self.state_name = state_name

    def __set_name__(self, owner, name: str) -> None:
        if self.state_name is None:
            self.state_name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance.state, self.state_name)

    def __set__(self, instance, value) -> None:
        setattr(instance.state, self.state_name, value)
