"""从可编辑配置生成明暗主题；配色随 Quasar 的 body--dark 状态切换。"""

import re
from pathlib import Path

import yaml


THEME_PATH = Path(__file__).resolve().parents[1] / 'config' / 'ui_theme.yaml'
REQUIRED_COLORS = frozenset('''
primary secondary accent page surface section-surface section-border cell-border
section-heading text muted subtle link link-hover info-bg info-border info-text
success-bg success-border success-text warning-bg warning-border warning-text
danger-bg danger-border danger-text notice-bg notice-border notice-text notice-action
notice-hover notice-close chip-bg chip-border chip-text boost-bg boost-border boost-text
reduce-bg reduce-border button-text help-bg help-border help-title help-success-border
help-success-text help-warning-text hover-overlay button-bg button-hover focus-ring
shadow pending-border general-bg character-bg copyright-bg artist-bg artist-row-bg
blue green purple pink orange teal amber red on-color tooltip-bg tooltip-text
positive negative warning info action-bg action-text
service-ready-bg service-ready-border service-ready-text selection-bg
'''.split())


def load_theme(path: Path = THEME_PATH) -> dict:
    try:
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'无法读取主题配置 {path}: {exc}') from exc
    if not isinstance(config, dict) or config.get('default') not in ('light', 'dark'):
        raise ValueError(f'{path}: default 必须是 light 或 dark')
    for mode in ('light', 'dark'):
        palette = config.get(mode)
        if not isinstance(palette, dict) or set(palette) != REQUIRED_COLORS:
            missing = REQUIRED_COLORS - set(palette or {}) if isinstance(palette, dict) else REQUIRED_COLORS
            raise ValueError(f'{path}: {mode} 色名不完整或有未知键，缺少 {sorted(missing)}')
        for name, value in palette.items():
            if not isinstance(value, str) or not re.fullmatch(r'#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})', value):
                raise ValueError(f'{path}: {mode}.{name} 必须是加引号的十六进制颜色')
    return config


def build_theme_css(config: dict) -> str:
    rules = []
    for mode, selector in (('light', 'body'), ('dark', 'body.body--dark')):
        colors = config[mode]
        variables = [f'--{name}: {value};' for name, value in colors.items()]
        variables += [f'--q-{name}: var(--{name});' for name in (
            'primary', 'secondary', 'accent', 'positive', 'negative', 'info', 'warning',
        )]
        variables += ['--q-dark: var(--surface);', '--q-dark-page: var(--page);']
        rules.append(f'{selector} {{ {" ".join(variables)} color-scheme: {mode}; }}')

    rules.append('''
body, body.body--dark { background: var(--page); color: var(--text); }
body .q-card:not(.section-surface), body .q-menu, body .q-table__container {
    background: var(--surface); color: var(--text);
}
body .q-field__native, body .q-field__input { color: var(--text); }
body .q-field__label, body .q-field__marginal { color: var(--muted); }
body .q-field--outlined .q-field__control:before { border-color: var(--section-border); }
body .q-separator { background: var(--cell-border); }
body .q-table tbody td:before, body .q-item.q-router-link--active { background: var(--hover-overlay); }
body .q-table__bottom, body .q-table th, body .q-table td { border-color: var(--cell-border); }
body .q-tooltip { background: var(--tooltip-bg) !important; color: var(--tooltip-text) !important; }
body .motion-search-button { background: var(--action-bg) !important; color: var(--action-text) !important; }
body .bg-white { background: var(--surface) !important; }
body .selection-surface { background: var(--selection-bg); }
body .bg-primary.text-white, body .q-btn.bg-primary { color: var(--on-color) !important; }
body .q-chip.bg-primary { color: var(--on-color) !important; }
body .bg-positive, body .bg-negative, body .bg-warning, body .bg-info { color: var(--on-color) !important; }
body .q-field input::placeholder, body .q-field textarea::placeholder { color: var(--subtle); opacity: 1; }
body :focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
body .nicegui-markdown a { color: var(--link); }
''')
    # 兼容现有 Tailwind / Quasar 色类，让弹窗、动态标签和控件共用配置。
    for family in ('gray', 'grey', 'slate', 'blue-grey'):
        for level in (1, 2, 3, 4, 5, 6, 7, 8, 9, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900):
            rank = level // 100 if level >= 100 else level
            text = 'text' if rank >= 7 else 'muted'
            rules.append(f'body .text-{family}-{level} {{ color: var(--{text}) !important; }}')
            rules.append(f'body .bg-{family}-{level} {{ background-color: var(--section-surface) !important; }}')
            rules.append(f'body .border-{family}-{level} {{ border-color: var(--cell-border) !important; }}')
        rules.append(f'body .text-{family} {{ color: var(--muted) !important; }}')
    for family, state in (
        ('blue', 'info'), ('green', 'success'), ('teal', 'success'),
        ('orange', 'warning'), ('amber', 'warning'), ('red', 'danger'),
        ('purple', 'reduce'), ('pink', 'reduce'),
    ):
        for suffix in ('', '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9',
                       '-50', '-100', '-200', '-300', '-400', '-500', '-600', '-700', '-800', '-900'):
            pale = suffix in ('-1', '-2', '-3', '-50', '-100', '-200', '-300')
            background = f'{state}-bg' if pale else family
            rules.append(f'body .text-{family}{suffix} {{ color: var(--{family}) !important; }}')
            rules.append(f'body .bg-{family}{suffix} {{ background-color: var(--{background}) !important; }}')
            rules.append(f'body .border-{family}{suffix} {{ border-color: var(--{state}-border) !important; }}')
            if not pale:
                rules.append(f'body .q-badge.bg-{family}{suffix} {{ color: var(--on-color) !important; }}')
                rules.append(
                    f'body.body--dark .q-badge.bg-{family}{suffix} {{ '
                    f'background-color: var(--{state}-bg) !important; color: var(--{family}) !important; }}'
                )
        for level in (400, 500, 600, 700, 800):
            rules.append(f'body .hover\\:text-{family}-{level}:hover {{ color: var(--link-hover) !important; }}')
    # NiceGUI 将 Quasar 的 !important 色类放入 quasar_importants 层。
    # important 的层优先级与普通规则相反；放入更早的 overrides 层才能覆盖，
    # 单纯提高未分层选择器的 specificity 仍会输给框架颜色。
    return '@layer overrides {\n' + '\n'.join(rules) + '\n}'


def apply_theme():
    from nicegui import ui

    config = load_theme()
    ui.add_head_html(f'<style>{build_theme_css(config)}</style>')
    return ui.dark_mode(config['default'] == 'dark')
