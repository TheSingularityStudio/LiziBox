
"""
简单工具栏实现 - 不依赖ImGui，使用键盘快捷键和鼠标操作
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ui.toolbar import Toolbar
from core.events import EventBus, Event, EventType
from core.state import state_manager

class SimpleToolbar(Toolbar):
    """简单工具栏实现，不依赖ImGui"""
    def __init__(self):
        super().__init__()
        self._show_help = False
        print("[工具栏] 使用简单工具栏（键盘快捷键和鼠标操作）")

    def render(self) -> None:
        """渲染工具栏（在控制台输出帮助信息）"""
        if self._show_help:
            self._print_help()
            self._show_help = False

    def _print_help(self) -> None:
        """打印帮助信息"""
        print("=== LiziEngine 控制面板 ===")
        print("快捷键:")
        print("  G - 切换网格显示")
        print("  R - 重置视图")
        print("  C - 清空网格")
        print("  T - 切换工具栏显示")
        print("  H - 显示此帮助信息")
        print("  +/- - 增加/减少画笔大小")
        print("  </> - 增加/减少向量大小")
        print("  V - 切换向量方向")
        print("鼠标操作:")
        print("  左键点击 - 在网格上绘制向量")
        print("  滚轮 - 缩放视图")
        print("当前设置:")
        print(f"  画笔大小: {self._brush_size}")
        print(f"  向量大小: {self._magnitude}")
        print(f"  显示网格: {self._show_grid}")
        print(f"  反转向量: {self._reverse_vector}")
        print("========================")

    def _on_key_pressed(self, event: Event) -> None:
        """处理键盘按键事件"""
        # 获取按键代码和动作
        key = event.data.get("key")
        action = event.data.get("action")
        
        # 只处理按键按下事件
        if action != 1:  # GLFW_PRESS
            return

        # 处理特定按键
        if key == 71:  # G键
            print(f"[工具栏] G键被按下，切换网格显示")
            self.toggle_grid()
        elif key == 82:  # R键
            self.reset_view()
        elif key == 67:  # C键
            self.clear_grid()
        elif key == 84:  # T键
            self.toggle_toolbar()
        elif key == 72:  # H键
            self._show_help = True
        elif key == 61 or key == 171:  # +键
            self.set_brush_size(self._brush_size + 1)
        elif key == 45 or key == 173:  # -键
            self.set_brush_size(self._brush_size - 1)
        elif key == 44:  # ,键
            self.set_magnitude(self._magnitude - 0.1)
        elif key == 46:  # .键
            self.set_magnitude(self._magnitude + 0.1)
        elif key == 86:  # V键
            self.toggle_reverse_vector()

        # 如果按下H键，显示帮助信息
        if key == 72:
            self._show_help = True
