"""
UI 管理模块：封装用户交互回调、鼠标拖拽与滚轮缩放处理
将与输入相关的逻辑从示例中抽离，便于维护和复用。
使用 Controller 模块处理业务逻辑。
"""
from typing import Tuple
import numpy as np
from lizi_engine.input import input_handler, KeyMap, MouseMap
from lizi_engine.core.state import state_manager


class UIManager:
    def __init__(self, app_core, window, controller, marker_system):
        self.app_core = app_core
        self.window = window
        self.controller = controller
        self.marker_system = marker_system

        self.enable_update = True

        # 保持最后鼠标位置以便拖拽计算（像素坐标）
        self._last_mouse_x = None
        self._last_mouse_y = None

        # 鼠标按钮状态
        self._mouse_buttons_pressed = set()  # 使用集合跟踪按下的按钮

        # 左键按下时选择的标记
        self._selected_marker = None
        # 弹簧连接模式下选择的标记列表
        self._spring_selected_markers = []

    def register_callbacks(self, grid: np.ndarray, on_space=None, on_r=None, on_g=None, on_c=None, on_u=None, on_v=None, on_f=None):
        self._grid = grid

        def on_u_press():
            if callable(on_u):
                try:
                    on_u()
                    return
                except Exception as e:
                    print(f"[错误] on_u 回调异常: {e}")
        
        def on_v_press():
            if callable(on_v):
                try:
                    on_v()
                    return
                except Exception as e:
                    print(f"[错误] on_v 回调异常: {e}")
            try:
                self.controller.switch_vector_field_direction()
            except Exception as e:
                print(f"[错误] 切换向量场方向 异常: {e}")

        def on_f_press():
            if callable(on_f):
                try:
                    on_f()
                    return
                except Exception as e:
                    print(f"[错误] on_f 回调异常: {e}")
            try:          
                mx = state_manager.get("mouse_x", 0.0)
                my = state_manager.get("mouse_y", 0.0)
                self.controller.place_vector_field(mx, my)
            except Exception as e:
                print(f"[错误] 处理f键按下时发生异常: {e}")

        def on_mouse_left_press():
            try:
                # 添加左键到按下按钮集合
                self._mouse_buttons_pressed.add(1)  # 左键

                mx = state_manager.get("mouse_x", 0.0)
                my = state_manager.get("mouse_y", 0.0)

                # 检查鼠标模式
                mouse_mode = state_manager.get("mouse_mode", "drag")
                if mouse_mode == "place_marker":
                    # 放置标记模式
                    self.controller.place_vector_field(mx, my)
                elif mouse_mode == "spring_connect":
                    # 弹簧连接模式
                    selected_marker = self.controller.handle_mouse_left_press(mx, my)
                    if selected_marker:
                        self._spring_selected_markers.append(selected_marker["id"])
                        print(f"Selected marker {selected_marker['id']} for spring connection")
                        if len(self._spring_selected_markers) == 2:
                            # 连接两个标记
                            id1, id2 = self._spring_selected_markers
                            self.marker_system.connect_spring(id1, id2)
                            self._spring_selected_markers = []  # 重置选择
                else:
                    # 拖动模式（默认）
                    self._selected_marker = self.controller.handle_mouse_left_press(mx, my)
            except Exception as e:
                print(f"[错误] 处理鼠标左键按下时发生异常: {e}")

        def on_mouse_left_release():
            # 从按下按钮集合中移除左键
            self._mouse_buttons_pressed.discard(1)
            self._selected_marker = None

        def on_mouse_middle_press():
            # 添加中键到按下按钮集合
            self._mouse_buttons_pressed.add(2)  # 中键
            # 初始化最后鼠标位置以避免拖拽时的跳跃
            from lizi_engine.core.state import state_manager
            self._last_mouse_x = state_manager.get("mouse_x", 0.0)
            self._last_mouse_y = state_manager.get("mouse_y", 0.0)

        def on_mouse_middle_release():
            # 从按下按钮集合中移除中键
            self._mouse_buttons_pressed.discard(2)

        # 注册键盘和鼠标回调 (使用PyQt6动作常量: 1=PRESS, 0=RELEASE)
        input_handler.register_key_callback(KeyMap.U, 1, on_u_press)  # PRESS
        input_handler.register_key_callback(KeyMap.F, 1, on_f_press)  # PRESS
        input_handler.register_key_callback(KeyMap.V, 1, on_v_press)  # PRESS

        input_handler.register_mouse_callback(MouseMap.LEFT, 1, on_mouse_left_press)  # PRESS
        input_handler.register_mouse_callback(MouseMap.LEFT, 0, on_mouse_left_release)  # RELEASE
        input_handler.register_mouse_callback(MouseMap.MIDDLE, 1, on_mouse_middle_press)  # PRESS
        input_handler.register_mouse_callback(MouseMap.MIDDLE, 0, on_mouse_middle_release)  # RELEASE

    def process_mouse_drag(self):
        from lizi_engine.core.state import state_manager
        # 处理鼠标左键持续按下，在标记位置添加向量
        if 1 in self._mouse_buttons_pressed and self._selected_marker is not None:
            try:
                mx = state_manager.get("mouse_x", 0.0)
                my = state_manager.get("mouse_y", 0.0)
                self.controller.handle_mouse_drag(mx, my, self._selected_marker)
            except Exception as e:
                print(f"[错误] 处理左键持续按下时发生异常: {e}")

        # 只在鼠标中键按下时才允许拖动视图
        if 2 in self._mouse_buttons_pressed:  # 中键
            try:
                x = state_manager.get("mouse_x", 0.0)
                y = state_manager.get("mouse_y", 0.0)

                dx = x - (self._last_mouse_x if self._last_mouse_x is not None else x)
                dy = y - (self._last_mouse_y if self._last_mouse_y is not None else y)

                self.controller.handle_mouse_drag_view(dx, dy)

                self._last_mouse_x = x
                self._last_mouse_y = y
            except Exception as e:
                print(f"[错误] process_mouse_drag_view 异常: {e}")
        else:
            # 清除上次位置，避免下次拖拽跳跃
            self._last_mouse_x = None
            self._last_mouse_y = None

    def process_scroll(self):
        window = self.window
        if hasattr(window, "_scroll_y") and window._scroll_y != 0:
            try:
                self.controller.handle_scroll_zoom(window._scroll_y)
            except Exception as e:
                print(f"[错误] process_scroll 异常: {e}")

            window._scroll_y = 0
