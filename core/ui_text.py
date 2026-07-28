from pathlib import Path
from typing import Any

import yaml


DEFAULT_UI_TEXT_PATH = Path(__file__).resolve().parents[1] / "config" / "ui_text.yaml"
REQUIRED_TEXT_PATHS = (
    ("sponsor", "body"),
    ("sponsor", "toolchain_prompt"),
    ("help", "update_title"),
    ("help", "update_summary"),
    ("help", "guide_markdown"),
    ("documentation", "title"),
    ("documentation", "subtitle"),
    ("documentation", "copyright_markdown"),
    ("notice", "title"),
    ("notice", "subtitle"),
    ("notice", "body_markdown"),
    ("dialogs", "prompt_import_description"),
    ("dialogs", "backup_description"),
    ("dialogs", "search_feedback_privacy"),
    ("dialogs", "translation_feedback_privacy"),
)


def load_ui_text(path: Path = DEFAULT_UI_TEXT_PATH) -> dict[str, Any]:
    """Load editable UI copy and fail with a clear startup error if it is invalid."""
    try:
        with path.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取 UI 文案配置 {path}: {exc}") from exc

    if not isinstance(content, dict):
        raise RuntimeError(f"UI 文案配置顶层必须是对象: {path}")

    for section, key in REQUIRED_TEXT_PATHS:
        section_content = content.get(section)
        value = section_content.get(key) if isinstance(section_content, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"UI 文案配置缺少非空文本: {section}.{key} ({path})")

    links = content.get("documentation", {}).get("links")
    if not isinstance(links, list) or not links:
        raise RuntimeError(f"UI 文案配置缺少链接列表: documentation.links ({path})")
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            raise RuntimeError(f"UI 文案配置链接格式错误: documentation.links[{index}] ({path})")
        if not isinstance(link.get("label"), str) or not link["label"].strip():
            raise RuntimeError(f"UI 文案配置链接缺少名称: documentation.links[{index}] ({path})")
        if not isinstance(link.get("url"), str) or not link["url"].strip():
            raise RuntimeError(f"UI 文案配置链接缺少地址: documentation.links[{index}] ({path})")
    return content
