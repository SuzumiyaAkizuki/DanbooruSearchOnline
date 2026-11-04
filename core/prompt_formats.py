"""MCP 和 REST 共用的提示词规范；配置修改后重启服务生效。"""

from pathlib import Path
from typing import Literal

import yaml


DEFAULT_PROMPT_FORMATS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompt_formats.yaml"
ArtProfile = Literal["illustration", "graphic_design"]
MODEL_KEYS = ("anima", "newbie", "qwen_t2i", "qwen_i2i", "moss")


def _load_prompt_sources(path: Path) -> dict[str, str]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取提示词规范配置: {path}") from exc
    if not isinstance(content, dict):
        raise RuntimeError(f"提示词规范配置顶层必须是对象: {path}")
    for key in ("art_direction", "graphic_design_art_direction", *MODEL_KEYS):
        value = content.get(key)
        if not isinstance(value, str) or (key != "newbie" and not value.strip()):
            raise RuntimeError(f"提示词规范配置缺少非空文本: {key} ({path})")
    return content


def _default_prompt_formats(content: dict[str, str]) -> dict[str, str]:
    formats = {key: content[key] for key in MODEL_KEYS}
    art_direction = content["art_direction"]
    for key in ("anima", "newbie", "qwen_t2i", "moss"):
        model_format = content[key]
        formats[key] = art_direction + "\n\n" + model_format if model_format.strip() else art_direction
    return formats


def load_prompt_formats(path: Path = DEFAULT_PROMPT_FORMATS_PATH) -> dict[str, str]:
    """加载默认规范，保持现有调用的返回内容。"""
    return _default_prompt_formats(_load_prompt_sources(path))


_PROMPT_SOURCES = _load_prompt_sources(DEFAULT_PROMPT_FORMATS_PATH)
PROMPT_FORMATS = _default_prompt_formats(_PROMPT_SOURCES)


def get_prompt_format(model: str, art_profile: ArtProfile | None = None) -> str:
    """为 Qwen T2I 或 MOSS 选择艺术规范；省略参数时沿用默认规范。"""
    if model not in PROMPT_FORMATS:
        raise ValueError(f"不支持的模型规范：{model}。")
    if art_profile is None:
        return PROMPT_FORMATS[model]
    if art_profile not in ("illustration", "graphic_design"):
        raise ValueError("art_profile 必须是 illustration 或 graphic_design。")
    if model not in ("qwen_t2i", "moss"):
        raise ValueError("art_profile 仅支持 Qwen T2I 和 MOSS；Qwen I2I 请省略该参数。")
    if art_profile == "illustration":
        return PROMPT_FORMATS[model]
    return _PROMPT_SOURCES["graphic_design_art_direction"] + "\n\n" + _PROMPT_SOURCES[model]
