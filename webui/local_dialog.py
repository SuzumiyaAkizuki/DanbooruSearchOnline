"""由浏览器立即开关、异步同步 Python value 的展示弹窗。"""

from nicegui import ui


class LocalDialog(ui.dialog, component='local_dialog.js'):
    # Vue 组件自行更新值，避免服务器回传较早的值覆盖连续开关。
    LOOPBACK = None

    def _handle_value_change(self, value: bool) -> None:
        super()._handle_value_change(value)
        if self._send_update_on_value_change:
            # 和 NiceGUI Input 一样，确保相同服务端 prop 也能再次驱动本地状态。
            self.run_method('syncValue', value)

    def browser_action(self, opened: bool) -> str:
        return f'() => runMethod({self.id}, "setOpen", [{str(opened).lower()}])'
