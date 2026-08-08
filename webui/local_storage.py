"""浏览器 localStorage 协议的共享边界。"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from core.workspace import (
    FAVORITES_STORAGE_KEY,
    HISTORY_STORAGE_KEY,
    LEGACY_STAGED_STORAGE_KEY,
    WORKSPACE_STORAGE_KEY,
    WorkspaceDataError,
    empty_favorites,
    empty_history,
    merge_favorites,
    merge_history,
    merge_workspaces,
    migrate_legacy_workspace,
    normalize_favorites,
    normalize_history,
    normalize_workspace,
)
from webui.constants import CONFIG_LS_KEY
from webui.constants import (
    LOCAL_STORAGE_MAX_READ_CHARS,
    LOCAL_STORAGE_READ_CHUNK_CHARS,
    LOCAL_STORAGE_RESTORE_CACHE,
)


def storage_keys() -> dict[str, str]:
    return {
        'workspace': WORKSPACE_STORAGE_KEY,
        'legacy': LEGACY_STAGED_STORAGE_KEY,
        'config': CONFIG_LS_KEY,
        'history': HISTORY_STORAGE_KEY,
        'favorites': FAVORITES_STORAGE_KEY,
    }


def write_is_ready(name: str, states: dict[str, str], applying: set[str]) -> bool:
    """只有已安全恢复且未处于应用阶段的数据域允许写回浏览器。"""
    return states.get(name) == 'ready' and name not in applying


async def backup_key(client, source_key: str, backup_key: str) -> bool:
    """在覆盖前仅创建一次浏览器内备份；已有备份绝不覆盖。"""
    source_key_js = json.dumps(source_key, ensure_ascii=False)
    backup_key_js = json.dumps(backup_key, ensure_ascii=False)
    try:
        status = await client.run_javascript(
            f'''(() => {{
                const source = localStorage.getItem({source_key_js});
                if (source === null) return 'missing';
                if (localStorage.getItem({backup_key_js}) !== null) return 'exists';
                try {{ localStorage.setItem({backup_key_js}, source); return 'created'; }}
                catch (error) {{ return `error:${{error && error.name ? error.name : 'unknown'}}`; }}
            }})()''',
            timeout=5.0,
        )
    except Exception:
        return False
    return status in {'created', 'exists'}


def storage_listener_script() -> str:
    keys = storage_keys()
    return f'''
        if (!window.__danbooruWorkspaceStorageListenerV1) {{
            window.__danbooruWorkspaceStorageListenerV1 = true;
            window.addEventListener('storage', (event) => {{
                const watchedKeys = new Set([{keys['workspace']!r}, {keys['history']!r}, {keys['favorites']!r}]);
                if (watchedKeys.has(event.key) && event.newValue !== event.oldValue) {{
                    if (window.confirm('工作区数据已在另一个标签页更新。是否重新加载当前页面以同步最新内容？')) window.location.reload();
                }}
            }});
        }}
    '''


async def read_prepared_value(client, connected, name: str, key: str, length) -> str | None:
    """按 UTF-16 安全边界分块读取准备好的浏览器缓存值。"""
    if length is None:
        return None
    if isinstance(length, bool) or not isinstance(length, (int, float)):
        raise RuntimeError(f'localStorage key {key!r} returned an invalid length')
    length = int(length)
    if length < 0 or length > LOCAL_STORAGE_MAX_READ_CHARS:
        raise WorkspaceDataError(f'localStorage key {key!r} exceeds the read limit')
    if length == 0:
        return ''
    name_js, key_js = json.dumps(name, ensure_ascii=False), json.dumps(key, ensure_ascii=False)
    cache_key_js = json.dumps(LOCAL_STORAGE_RESTORE_CACHE)
    chunks: list[str] = []
    offset = 0
    while offset < length:
        if not connected():
            raise RuntimeError('client disconnected during localStorage restore')
        result = await client.run_javascript(
            f'''(() => {{
                const cache = window[{cache_key_js}];
                const value = cache && Object.prototype.hasOwnProperty.call(cache, {name_js}) ? cache[{name_js}] : localStorage.getItem({key_js});
                if (value === null) return null;
                let end = Math.min(value.length, {offset + LOCAL_STORAGE_READ_CHUNK_CHARS});
                if (end < value.length) {{ const last = value.charCodeAt(end - 1); if (last >= 0xD800 && last <= 0xDBFF) end += 1; }}
                return {{chunk: value.slice({offset}, end), next_offset: end}};
            }})()''', timeout=5.0,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f'localStorage key {key!r} disappeared during restore')
        chunk, next_offset = result.get('chunk'), result.get('next_offset')
        if not isinstance(chunk, str) or not isinstance(next_offset, (int, float)):
            raise RuntimeError(f'localStorage key {key!r} returned an invalid chunk')
        next_offset = int(next_offset)
        if next_offset <= offset or next_offset > length:
            raise RuntimeError(f'localStorage key {key!r} returned an invalid offset')
        chunks.append(chunk)
        offset = next_offset
    return ''.join(chunks)


def clear_restore_cache(client) -> None:
    try:
        client.run_javascript(f'delete window[{json.dumps(LOCAL_STORAGE_RESTORE_CACHE)}];')
    except RuntimeError:
        pass


async def prepare_restore_snapshot(client, connected, keys: dict[str, str]) -> dict:
    """快照请求域；仅在浏览器内存中预压缩旧 history，绝不直接覆盖存储。"""
    if not connected():
        raise RuntimeError('client is disconnected')
    keys_js = json.dumps(keys, ensure_ascii=False)
    cache_key_js = json.dumps(LOCAL_STORAGE_RESTORE_CACHE)
    result = await client.run_javascript(
        f'''(() => {{
            const keys = {keys_js}, values = {{}}, manifest = {{}};
            for (const [name, key] of Object.entries(keys)) {{
                let value = localStorage.getItem(key), prepared = false;
                const originalLength = value === null ? null : value.length;
                if (name === 'history' && value) try {{
                    const data = JSON.parse(value);
                    if (data && typeof data === 'object' && (data.schema_version === 1 || data.schema_version === 2) && Array.isArray(data.items)) {{
                        let changed = data.schema_version !== 2;
                        const items = data.items.map((item) => {{
                            if (!item || typeof item !== 'object' || !item.workspace || typeof item.workspace !== 'object' || typeof item.query !== 'string' || !item.query.trim() || !item.settings || typeof item.settings !== 'object' || Array.isArray(item.settings)) return item;
                            const query = item.query.trim().slice(0, 4000);
                            const searchedAt = typeof item.searched_at === 'string' && item.searched_at ? item.searched_at : new Date().toISOString();
                            const oldQueries = item.workspace.queries;
                            if (!Array.isArray(oldQueries) || oldQueries.length !== 1 || !oldQueries[0] || oldQueries[0].query !== query) changed = true;
                            const workspace = {{...item.workspace, queries: [{{query, searched_at: searchedAt, settings: item.settings}}], updated_at: searchedAt}};
                            return {{...item, workspace_id: workspace.workspace_id, workspace}};
                        }});
                        if (changed) {{ value = JSON.stringify({{...data, schema_version: 2, items}}); prepared = true; }}
                    }}
                }} catch (_) {{ /* Python validates and backs up corruption. */ }}
                values[name] = value;
                manifest[name] = {{length: value === null ? null : value.length, original_length: originalLength, prepared}};
            }}
            window[{cache_key_js}] = values;
            return manifest;
        }})()''', timeout=5.0,
    )
    if not isinstance(result, dict):
        raise RuntimeError('localStorage manifest is invalid')
    return result


async def restore_staged_storage(
    controller: Any,
    names: tuple[str, ...],
    config_version: int,
) -> tuple[dict[str, str], set[str], list[str]]:
    """执行一次恢复尝试；控制器只提供页面状态和应用回调。"""
    unresolved = [
        name for name in names
        if controller._storage_states.get(name) != 'ready'
    ]
    if not unresolved:
        return {}, set(), []

    failures: dict[str, str] = {}
    persist: set[str] = set()
    warnings: list[str] = []
    keys = controller._local_storage_keys()
    try:
        manifest = await controller._prepare_local_storage_restore(unresolved)
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        return {name: message for name in unresolved}, persist, warnings

    try:
        for name in unresolved:
            meta = manifest.get(name)
            if not isinstance(meta, dict):
                failures[name] = 'missing manifest entry'
                continue
            try:
                raw = await controller._read_local_storage_value(
                    name, keys[name], meta.get('length')
                )
            except Exception as exc:
                failures[name] = str(exc) or type(exc).__name__
                continue
            controller._storage_raw_values[name] = raw
    finally:
        controller._clear_local_storage_restore_cache()

    if 'legacy' in unresolved and 'legacy' not in failures:
        controller._storage_states['legacy'] = 'ready'

    if 'config' in unresolved and 'config' not in failures:
        raw_config = controller._storage_raw_values.get('config')
        config_dirty = 'config' in controller._storage_session_dirty
        try:
            cfg = json.loads(raw_config) if raw_config else {}
            if not isinstance(cfg, dict):
                raise WorkspaceDataError('config must be a JSON object')
        except Exception as exc:
            config_key = keys['config']
            if raw_config and not await controller._backup_local_storage_key(
                config_key, f'{config_key}_corrupt_backup'
            ):
                failures['config'] = f'corrupt config backup failed: {exc}'
            else:
                warnings.append('config_corrupt')
                persist.add('config')
        else:
            if cfg and cfg.get('version') != config_version:
                warnings.append('config_schema_migrated')
                persist.add('config')
            if not config_dirty:
                controller._storage_applying.add('config')
                try:
                    controller._apply_config_state(cfg)
                finally:
                    controller._storage_applying.discard('config')
            else:
                persist.add('config')
        if 'config' not in failures:
            controller._storage_states['config'] = 'ready'

    if 'workspace' in unresolved and 'workspace' not in failures:
        raw_workspace = controller._storage_raw_values.get('workspace')
        if not raw_workspace and any(
            controller._storage_states.get(name) != 'ready'
            for name in ('legacy', 'config')
        ):
            failures['workspace'] = 'legacy workspace inputs are not available'
        else:
            workspace_warnings: list[str] = []
            try:
                if raw_workspace:
                    workspace, workspace_warnings = normalize_workspace(raw_workspace)
                else:
                    workspace, workspace_warnings = migrate_legacy_workspace(
                        controller._storage_raw_values.get('legacy'),
                        controller._storage_raw_values.get('config'),
                    )
                    persist.add('workspace')
            except WorkspaceDataError as exc:
                backed_up = await controller._backup_local_storage_key(
                    keys['workspace'], f"{keys['workspace']}_corrupt_backup"
                )
                if raw_workspace and not backed_up:
                    failures['workspace'] = f'corrupt workspace backup failed: {exc}'
                else:
                    workspace, migration_warnings = migrate_legacy_workspace(
                        controller._storage_raw_values.get('legacy'),
                        controller._storage_raw_values.get('config'),
                    )
                    workspace_warnings = ['workspace_corrupt'] + migration_warnings
                    persist.add('workspace')
            if 'workspace' not in failures:
                if 'workspace' in controller._storage_session_dirty:
                    workspace = merge_workspaces(
                        controller.workspace_state,
                        workspace,
                        origin='local_restore',
                        source='浏览器本地恢复',
                    )
                    persist.add('workspace')
                controller._storage_applying.add('workspace')
                try:
                    controller._apply_workspace_state(
                        workspace, persist=False, refresh_recommendations=False
                    )
                finally:
                    controller._storage_applying.discard('workspace')
                warnings.extend(workspace_warnings)
                if workspace_warnings:
                    persist.add('workspace')
                controller._storage_states['workspace'] = 'ready'
                if raw_workspace:
                    controller._storage_states['legacy'] = 'ready'
                    failures.pop('legacy', None)

    if 'history' in unresolved and 'history' not in failures:
        raw_history = controller._storage_raw_values.get('history')
        history_prepared = bool(
            isinstance(manifest.get('history'), dict)
            and manifest['history'].get('prepared')
        )
        try:
            history, history_warnings = normalize_history(raw_history)
        except WorkspaceDataError as exc:
            backed_up = await controller._backup_local_storage_key(
                keys['history'], f"{keys['history']}_corrupt_backup"
            )
            if raw_history and not backed_up:
                failures['history'] = f'corrupt history backup failed: {exc}'
            else:
                history, history_warnings = empty_history(), ['history_corrupt']
                persist.add('history')
        if 'history' not in failures:
            if history_prepared:
                if not await controller._backup_history_before_compaction():
                    failures['history'] = 'legacy history backup failed'
                else:
                    history_warnings.extend([
                        'history_schema_migrated',
                        'history_workspace_queries_compacted',
                    ])
                    persist.add('history')
            if 'history' not in failures:
                if 'history' in controller._storage_session_dirty:
                    history = merge_history(controller.search_history, history)
                    persist.add('history')
                controller.search_history = history
                warnings.extend(history_warnings)
                if history_warnings:
                    persist.add('history')
                controller._storage_states['history'] = 'ready'

    if 'favorites' in unresolved and 'favorites' not in failures:
        raw_favorites = controller._storage_raw_values.get('favorites')
        try:
            favorites, favorite_warnings = normalize_favorites(raw_favorites)
        except WorkspaceDataError as exc:
            backed_up = await controller._backup_local_storage_key(
                keys['favorites'], f"{keys['favorites']}_corrupt_backup"
            )
            if raw_favorites and not backed_up:
                failures['favorites'] = f'corrupt favorites backup failed: {exc}'
            else:
                favorites, favorite_warnings = empty_favorites(), ['favorites_corrupt']
                persist.add('favorites')
        if 'favorites' not in failures:
            if 'favorites' in controller._storage_session_dirty:
                favorites = merge_favorites(controller.favorites, favorites)
                persist.add('favorites')
            controller.favorites = favorites
            warnings.extend(favorite_warnings)
            if favorite_warnings:
                persist.add('favorites')
            controller._storage_states['favorites'] = 'ready'

    for name in failures:
        controller._storage_states[name] = 'failed'
    controller._update_undo_buttons()
    controller._update_workspace_counts()
    return failures, persist, warnings


@dataclass(frozen=True)
class RestoreRetryResult:
    """一次恢复任务的无界面结果，交由页面层决定日志和提示。"""

    completed: bool
    client_stopped: bool
    failures: dict[str, str]
    warnings: list[str]


async def restore_with_retries(
    controller: Any,
    retry_delays: tuple[float, ...],
) -> RestoreRetryResult:
    """在连接可用时执行有限次恢复，不触碰 NiceGUI 的通知上下文。"""
    last_failures: dict[str, str] = {}
    all_warnings: list[str] = []
    for delay in retry_delays:
        if delay:
            await asyncio.sleep(delay)
        if not controller._client_alive():
            return RestoreRetryResult(False, True, last_failures, all_warnings)
        if not controller._client_connected():
            last_failures = {'connection': 'client is disconnected'}
            continue
        controller._storage_restoring = True
        try:
            failures, persist, warnings = await controller._restore_staged_tags()
        finally:
            controller._storage_restoring = False
        persist_restored_storage(controller, persist)
        all_warnings.extend(warnings)
        last_failures = failures
        if not failures:
            return RestoreRetryResult(True, False, {}, all_warnings)
    return RestoreRetryResult(False, False, last_failures, all_warnings)


def persist_restored_storage(controller: Any, names: set[str]) -> None:
    """按既定顺序写回已恢复的数据域，具体写入仍由页面控制器完成。"""
    for name, saver_name in (
        ('config', '_save_config'),
        ('workspace', '_save_staged_tags'),
        ('history', '_save_history'),
        ('favorites', '_save_favorites'),
    ):
        if name in names:
            getattr(controller, saver_name)()


def flush_storage_session_changes(controller: Any) -> None:
    """只刷新已安全恢复的数据域，失败或未恢复域绝不覆盖浏览器旧值。"""
    ready_dirty = {
        name for name in controller._storage_session_dirty
        if controller._storage_states.get(name) == 'ready'
    }
    persist_restored_storage(controller, ready_dirty)


def start_storage_restore_task(controller: Any, worker) -> Any:
    """仅在仍有待恢复域或待写回数据时创建一个恢复任务。"""
    task = controller._storage_restore_task
    if task is not None and not task.done():
        return task
    needs_work = any(
        state != 'ready' for state in controller._storage_states.values()
    ) or bool(controller._storage_session_dirty)
    if not needs_work:
        return None
    controller._storage_restore_task = asyncio.create_task(worker())
    return controller._storage_restore_task


def pause_storage_restore(controller: Any) -> None:
    """取消当前恢复任务；重连后的重启决定仍由页面生命周期处理。"""
    task = controller._storage_restore_task
    if task is not None and not task.done():
        task.cancel()


def finish_storage_restore_task(
    controller: Any,
    current_task: Any,
    was_cancelled: bool,
) -> bool:
    """清理当前任务，并告知页面层是否应在已连接时重新启动恢复。"""
    controller._storage_restoring = False
    if controller._storage_restore_task is not current_task:
        return False
    controller._storage_restore_task = None
    return was_cancelled and controller._client_connected()
