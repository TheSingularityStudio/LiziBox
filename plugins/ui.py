"""
UI 管理模块：封装用户交互回调、鼠标拖拽与滚轮缩放处理
将与输入相关的逻辑从示例中抽离，便于维护和复用。
"""
from typing import Tuple
import numpy as np
from lizi_engine.input import input_handler, KeyMap, MouseMap
from plugins.marker_system import MarkerSystem


class UIManager:
    def __init__(self, app_core, window, vector_calculator):
        self.app_core = app_core
        self.window = window
        self.vector_calculator = vector_calculator
        self.enable_update = True
        
        # 向量场方向状态：True表示朝外，False表示朝内
        self.vector_field_direction = True
        
        # 向量场模式：True表示径向模式（发散），False表示切线模式（旋转）
        self.vector_field_pattern = True

        # 保持最后鼠标位置以便拖拽计算（像素坐标）
        self._last_mouse_x = None
        self._last_mouse_y = None

        # 初始化标记系统
        self.marker_system = MarkerSystem(app_core)

        # 左键按下标志
        self._mouse_left_pressed = False

        # 左键按下时选择的标记
        self._selected_marker = None

    def register_callbacks(self, grid: np.ndarray, on_space=None, on_r=None, on_g=None, on_c=None, on_u=None, on_v=None):
        self._grid = grid

        def on_space_press():
            # 优先使用外部提供的回调（用于切换模式等），否则做一个默认视图重置。
            if callable(on_space):
                try:
                    on_space()
                    return
                except Exception as e:
                    print(f"[错误] on_space 回调异常: {e}")
            try:
                self.app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
            except Exception:
                pass

        def on_r_press():
            if callable(on_r):
                try:
                    on_r()
                    return
                except Exception as e:
                    print(f"[错误] on_r 回调异常: {e}")
            self.app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])

        def on_g_press():
            if callable(on_g):
                try:
                    on_g()
                    return
                except Exception as e:
                    print(f"[错误] on_g 回调异常: {e}")
            show_grid = self.app_core.state_manager.get("show_grid", True)
            self.app_core.state_manager.set("show_grid", not show_grid)

        def on_c_press():
            if callable(on_c):
                try:
                    on_c()
                    return
                except Exception as e:
                    print(f"[错误] on_c 回调异常: {e}")
            grid.fill(0.0)
            
        def on_v_press():
            if callable(on_v):
                try:
                    on_v()
                    return
                except Exception as e:
                    print(f"[错误] on_v 回调异常: {e}")
            self.vector_field_direction = not self.vector_field_direction
            direction = "朝外" if self.vector_field_direction else "朝内"
            print(f"[示例] 向量场方向已切换为: {direction}")
            
        def on_f_press():
            try:
                mx, my = input_handler.get_mouse_position()

                cam_x = self.app_core.state_manager.get("cam_x", 0.0)
                cam_y = self.app_core.state_manager.get("cam_y", 0.0)
                cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
                viewport_width = self.app_core.state_manager.get("viewport_width", self.window._width)
                viewport_height = self.app_core.state_manager.get("viewport_height", self.window._height)
                cell_size = self.app_core.config_manager.get("cell_size", 1.0)

                world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
                world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

                gx = int(world_x / cell_size)
                gy = int(world_y / cell_size)

                h, w = grid.shape[:2]
                if gx < 0 or gx >= w or gy < 0 or gy >= h:
                    print(f"[示例] 点击位置超出网格: ({gx}, {gy})")
                    return

                radius = 2
                magnitude = 1 if self.vector_field_direction else -1

                direction = "朝外" if self.vector_field_direction else "朝内"                
                print(f"[示例] 在网格位置放置向量场: ({gx}, {gy}), radius={radius}, mag={magnitude}, 方向={direction}")

                self.vector_calculator.create_radial_pattern(grid, center=(gx, gy), radius=radius, magnitude=magnitude)                

                # 同时创建一个标记，初始放在点击处（浮点位置）
                self.marker_system.add_marker(float(gx), float(gy), float(magnitude))

                self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
            except Exception as e:
                print(f"[错误] 处理f键按下时发生异常: {e}")

        def on_mouse_left_press():
            try:
                # 设置左键按下标志
                self._mouse_left_pressed = True

                mx, my = input_handler.get_mouse_position()

                cam_x = self.app_core.state_manager.get("cam_x", 0.0)
                cam_y = self.app_core.state_manager.get("cam_y", 0.0)
                cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
                viewport_width = self.app_core.state_manager.get("viewport_width", self.window._width)
                viewport_height = self.app_core.state_manager.get("viewport_height", self.window._height)
                cell_size = self.app_core.config_manager.get("cell_size", 1.0)

                world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
                world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

                gx = int(world_x / cell_size)
                gy = int(world_y / cell_size)

                h, w = grid.shape[:2]
                if gx < 0 or gx >= w or gy < 0 or gy >= h:
                    print(f"[示例] 点击位置超出网格: ({gx}, {gy})")
                    return

                # 获取所有标记
                markers = self.marker_system.get_markers()
                if not markers:
                    print("[示例] 没有可用的标记")
                    return
                    
                # 找到最近的标记
                min_dist = float('inf')
                closest_marker = None
                for marker in markers:
                    marker_x = marker["x"]
                    marker_y = marker["y"]
                    dist = ((marker_x - gx) ** 2 + (marker_y - gy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_marker = marker
                
                if closest_marker is None:
                    print("[示例] 未找到最近的标记")
                    return
                    
                # 计算从标记到鼠标位置的方向向量
                vx = gx - closest_marker["x"]
                vy = gy - closest_marker["y"]
                
                # 归一化向量
                vec_len = (vx ** 2 + vy ** 2) ** 0.5
                if vec_len > 0:
                    vx /= vec_len
                    vy /= vec_len
                    
                # 使用标记系统的新功能在标记位置添加向量
                #self.marker_system.add_vector_at_position(grid, closest_marker["x"], closest_marker["y"], vx, vy, radius=0.5)
                # 使用微小向量创建函数
                self.marker_system.create_tiny_vector(grid, closest_marker["x"], closest_marker["y"], vx=vx, vy=vy)

                # 保存选定的标记，以便在持续按下的过程中使用
                self._selected_marker = closest_marker
                
                print(f"[示例] 在标记位置({closest_marker['x']:.2f}, {closest_marker['y']:.2f})添加向量({vx:.2f}, {vy:.2f})")
                
                self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
            except Exception as e:
                print(f"[错误] 处理鼠标左键按下时发生异常: {e}")

        # 注册键盘和鼠标回调
        input_handler.register_key_callback(KeyMap.SPACE, MouseMap.PRESS, on_space_press)
        input_handler.register_key_callback(KeyMap.R, MouseMap.PRESS, on_r_press)
        input_handler.register_key_callback(KeyMap.G, MouseMap.PRESS, on_g_press)
        input_handler.register_key_callback(KeyMap.C, MouseMap.PRESS, on_c_press)
        #input_handler.register_key_callback(KeyMap.U, MouseMap.PRESS, on_u_press)
        input_handler.register_key_callback(KeyMap.V, MouseMap.PRESS, on_v_press)
        input_handler.register_key_callback(KeyMap.F, MouseMap.PRESS, on_f_press)
        #input_handler.register_key_callback(KeyMap.H, MouseMap.PRESS, on_h_press)

        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.PRESS, on_mouse_left_press)

        # 添加鼠标左键释放的回调
        def on_mouse_left_release():
            # 清除左键按下标志和选定的标记
            self._mouse_left_pressed = False
            self._selected_marker = None

        input_handler.register_mouse_callback(MouseMap.LEFT, MouseMap.RELEASE, on_mouse_left_release)

        # 添加鼠标中键按下和释放的回调
        def on_mouse_middle_press():
            # 设置中键按下标志
            if hasattr(self.window, '_mouse_middle_pressed'):
                self.window._mouse_middle_pressed = True
            else:
                setattr(self.window, '_mouse_middle_pressed', True)

        def on_mouse_middle_release():
            # 清除中键按下标志
            if hasattr(self.window, '_mouse_middle_pressed'):
                self.window._mouse_middle_pressed = False

        # 注册鼠标中键回调
        input_handler.register_mouse_callback(MouseMap.MIDDLE, MouseMap.PRESS, on_mouse_middle_press)
        input_handler.register_mouse_callback(MouseMap.MIDDLE, MouseMap.RELEASE, on_mouse_middle_release)

    def process_mouse_drag(self):
        window = self.window
        # 处理鼠标左键持续按下，在标记位置添加向量
        if self._mouse_left_pressed:
            try:
                mx, my = window._mouse_x, window._mouse_y

                cam_x = self.app_core.state_manager.get("cam_x", 0.0)
                cam_y = self.app_core.state_manager.get("cam_y", 0.0)
                cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
                viewport_width = self.app_core.state_manager.get("viewport_width", window._width)
                viewport_height = self.app_core.state_manager.get("viewport_height", window._height)
                cell_size = self.app_core.config_manager.get("cell_size", 1.0)

                world_x = cam_x + (mx - (viewport_width / 2.0)) / cam_zoom
                world_y = cam_y + (my - (viewport_height / 2.0)) / cam_zoom

                gx = int(world_x / cell_size)
                gy = int(world_y / cell_size)

                h, w = self._grid.shape[:2]
                if gx >= 0 and gx < w and gy >= 0 and gy < h:
                    # 使用之前选定的标记，而不是重新查找
                    if self._selected_marker is not None:
                        # 计算从标记到鼠标位置的方向向量
                        vx = gx - self._selected_marker["x"]
                        vy = gy - self._selected_marker["y"]
                        '''
                        # 归一化向量
                        vec_len = (vx ** 2 + vy ** 2) ** 0.5
                        if vec_len > 0:
                            vx /= vec_len
                            vy /= vec_len
                        '''
                        # 使用微小向量创建函数
                        self.marker_system.create_tiny_vector(self._grid, x=self._selected_marker["x"], y=self._selected_marker["y"], vx=vx, vy=vy)

                        self.app_core.state_manager.update({"view_changed": True, "grid_updated": True})
            except Exception as e:
                print(f"[错误] 处理左键持续按下时发生异常: {e}")

        # 只在鼠标中键按下时才允许拖动视图
        if getattr(window, "_mouse_middle_pressed", False):
            x, y = window._mouse_x, window._mouse_y

            dx = x - (self._last_mouse_x if self._last_mouse_x is not None else x)
            dy = y - (self._last_mouse_y if self._last_mouse_y is not None else y)

            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)

            world_dx = dx / cam_zoom
            world_dy = dy / cam_zoom

            cam_x = self.app_core.state_manager.get("cam_x", 0.0) - world_dx
            cam_y = self.app_core.state_manager.get("cam_y", 0.0) - world_dy

            self.app_core.state_manager.update({
                "cam_x": cam_x,
                "cam_y": cam_y,
                "view_changed": True
            })

            self._last_mouse_x = x
            self._last_mouse_y = y
        else:
            # 清除上次位置，避免下次拖拽跳跃
            self._last_mouse_x = None
            self._last_mouse_y = None

    def process_scroll(self):
        window = self.window
        if hasattr(window, "_scroll_y") and window._scroll_y != 0:
            cam_zoom = self.app_core.state_manager.get("cam_zoom", 1.0)
            zoom_speed = 0.5
            cam_zoom += window._scroll_y * zoom_speed
            cam_zoom = max(0.1, min(10.0, cam_zoom))

            self.app_core.state_manager.update({
                "cam_zoom": cam_zoom,
                "view_changed": True
            })

            window._scroll_y = 0

    def update_markers(self, grid: np.ndarray, neighborhood: int = 2, move_factor: float = 1.0, clear_threshold: float = 1e-3):
        """使用标记系统更新标记位置
        
        Args:
            grid: 向量场网格
            neighborhood: 邻域大小
            move_factor: 移动因子
            clear_threshold: 清除阈值，低于此平均幅值的标记将被清除
        """
        self.marker_system.update_markers(grid, neighborhood, move_factor, clear_threshold)
