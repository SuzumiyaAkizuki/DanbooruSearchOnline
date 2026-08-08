"""与业务逻辑无关的页面样式片段。"""


MOTION_STYLE = '''
<style>
:root {
    --motion-ease-out: cubic-bezier(0.22, 1, 0.36, 1);
    --motion-fast: 180ms;
    --motion-content: 250ms;
    --motion-chip: 210ms;
}
.motion-search-input .q-field__control {
    transition: box-shadow var(--motion-fast) var(--motion-ease-out),
                background-color var(--motion-fast) ease;
}
.motion-search-input.q-field--focused .q-field__control {
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.12);
}
.motion-search-button {
    transition: transform var(--motion-fast) var(--motion-ease-out),
                box-shadow var(--motion-fast) ease, opacity var(--motion-fast) ease;
}
@media (hover: hover) {
    .motion-search-button:not(.disabled):hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(51, 65, 85, 0.18);
    }
}
.motion-search-button:active { transform: translateY(0); }
.motion-search-button.disabled { opacity: 0.82; }
.motion-search-spinner { animation: motion-chip-enter var(--motion-fast) var(--motion-ease-out) both; }
@keyframes motion-content-enter {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes motion-chip-enter {
    from { opacity: 0; transform: translateY(3px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes motion-recommendation-enter-from-right {
    from { opacity: 0; transform: translateX(10px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes motion-recommendation-enter-from-left {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}
.motion-results-enter, .motion-secondary-enter, .motion-refresh-enter {
    animation: motion-content-enter var(--motion-content) var(--motion-ease-out) both;
}
.motion-recommendation-enter-right {
    animation: motion-recommendation-enter-from-right var(--motion-content) var(--motion-ease-out) both;
}
.motion-recommendation-enter-left {
    animation: motion-recommendation-enter-from-left var(--motion-content) var(--motion-ease-out) both;
}
.motion-chip-enter { animation: motion-chip-enter var(--motion-chip) var(--motion-ease-out) both; }
@media (prefers-reduced-motion: reduce) {
    .motion-search-input .q-field__control, .motion-search-button, .motion-search-spinner,
    .related-item, .weight-btn { transition-duration: 1ms !important; }
    .motion-results-enter, .motion-secondary-enter, .motion-refresh-enter,
    .motion-recommendation-enter-right, .motion-recommendation-enter-left,
    .motion-chip-enter, .motion-search-spinner { animation: none !important; }
}
</style>
'''
