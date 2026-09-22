"""MCP 和 REST 共用的提示词规范；配置修改后重启服务生效。"""

from pathlib import Path

import yaml


DEFAULT_PROMPT_FORMATS_PATH = Path(__file__).resolve().parent.parent / "config" / "prompt_formats.yaml"


def load_prompt_formats(path: Path = DEFAULT_PROMPT_FORMATS_PATH) -> dict[str, str]:
    """加载完整规范，保留原始空白，并在配置无效时明确报错。"""
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取提示词规范配置: {path}") from exc
    if not isinstance(content, dict):
        raise RuntimeError(f"提示词规范配置顶层必须是对象: {path}")
    for key in ("anima", "newbie", "qwen_t2i", "qwen_i2i"):
        value = content.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"提示词规范配置缺少非空文本: {key} ({path})")
    return content


PROMPT_FORMATS = load_prompt_formats()
