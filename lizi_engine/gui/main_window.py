"""
基于PyQt6的LiziEngine GUI主窗口
提供集成OpenGL渲染的主应用程序窗口
"""
import sys
import time
from typing import Optional, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QLabel, QPushButton, QSlider, QGroupBox,
    QGridLayout, QStatusBar, QMenuBar, QMenu, QToolBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup

from .opengl_widget import OpenGLWidget
from .control_panel import ControlPanel
from .event_manager import EventManager


class MainWindow(QMainWindow):
    """使用PyQt6的主应用程序窗口"""

    def __init__(self, controller=None, marker_system=None, config_manager=None,
                 state_manager=None, renderer=None):
        super().__init__()

        # 存储核心系统的引用
        self.controller = controller
        self.marker_system = marker_system
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.renderer = renderer

        # UI 组件
        self.opengl_widget = None
        self.control_panel = None
        self.event_manager = EventManager()

        # 实时更新状态
        self.realtime_updates_enabled = True

        # 重力强度
        self.gravity_strength = 0.1

        # 摩擦强度
        self.friction_strength = 0.05

        # 窗口属性
        self.setWindowTitle("粒子引擎 - PyQt6 GUI")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化 UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_menus()
        self._setup_status_bar()
        self._connect_signals()

        # 用于实时渲染的更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_loop)
        self.update_timer.start(16)  # ~60 FPS

    def _setup_ui(self):
        """设置主用户界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 创建可调整大小面板的分隔器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 控制面板（左侧）
        self.control_panel = ControlPanel(self.config_manager, self.state_manager)
        splitter.addWidget(self.control_panel)

        # OpenGL 渲染区域（右侧）
        self.opengl_widget = OpenGLWidget(
            self.renderer,
            self.state_manager,
            self.config_manager,
            self.marker_system,
            self.controller
        )
        splitter.addWidget(self.opengl_widget)

        # 设置分隔器比例（控制面板较窄）
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

    def _setup_toolbar(self):
        """设置用于鼠标模式控制的工具栏"""
        # 创建工具栏
        self.toolbar = self.addToolBar("Mouse Mode")

        # 创建用于互斥选择的动作组
        self.mouse_mode_group = QActionGroup(self)
        self.mouse_mode_group.setExclusive(True)

        # 拖动标记动作
        self.drag_marker_action = QAction("拖动标记", self)
        self.drag_marker_action.setCheckable(True)
        self.drag_marker_action.setChecked(True)  # 默认模式
        self.drag_marker_action.triggered.connect(self._set_drag_marker_mode)
        self.mouse_mode_group.addAction(self.drag_marker_action)
        self.toolbar.addAction(self.drag_marker_action)

        # 放置标记动作
        self.place_marker_action = QAction("放置标记", self)
        self.place_marker_action.setCheckable(True)
        self.place_marker_action.triggered.connect(self._set_place_marker_mode)
        self.mouse_mode_group.addAction(self.place_marker_action)
        self.toolbar.addAction(self.place_marker_action)

        # 在状态管理器中初始化默认模式
        if self.state_manager:
            self.state_manager.update({"mouse_mode": "drag"})

    def _set_drag_marker_mode(self):
        """设置鼠标模式为拖动标记"""
        if self.state_manager:
            self.state_manager.update({"mouse_mode": "drag"})

    def _set_place_marker_mode(self):
        """设置鼠标模式为放置标记"""
        if self.state_manager:
            self.state_manager.update({"mouse_mode": "place"})

    def _setup_menus(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        exit_action = QAction('退出', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu('视图')

        reset_view_action = QAction('重置视图', self)
        reset_view_action.setShortcut('R')
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

        toggle_grid_action = QAction('切换网格', self)
        toggle_grid_action.setShortcut('G')
        toggle_grid_action.triggered.connect(self._toggle_grid)
        view_menu.addAction(toggle_grid_action)

        # 编辑菜单
        edit_menu = menubar.addMenu('编辑')

        clear_grid_action = QAction('清除网格', self)
        clear_grid_action.setShortcut('C')
        clear_grid_action.triggered.connect(self._clear_grid)
        edit_menu.addAction(clear_grid_action)

        generate_tangential_action = QAction('生成切向图案', self)
        generate_tangential_action.setShortcut(QKeySequence('Space'))
        generate_tangential_action.triggered.connect(self._generate_tangential)
        edit_menu.addAction(generate_tangential_action)

    def _setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = self.statusBar()

        # 帧率标签
        self.fps_label = QLabel("FPS: 0")
        self.status_bar.addWidget(self.fps_label)

        # 网格大小标签
        self.grid_size_label = QLabel("Grid: 64x64")
        self.status_bar.addWidget(self.grid_size_label)

        # 标记数量标签
        self.marker_count_label = QLabel("Markers: 0")
        self.status_bar.addWidget(self.marker_count_label)

        # 相机位置标签
        self.camera_label = QLabel("Camera: (0.0, 0.0) Zoom: 1.0")
        self.status_bar.addWidget(self.camera_label)

    def _connect_signals(self):
        """Connect signals between components"""
        # Connect control panel signals
        if self.control_panel:
            self.control_panel.marker_add_requested.connect(self._add_marker)
            self.control_panel.marker_clear_requested.connect(self._clear_markers)
            self.control_panel.zoom_changed.connect(self._handle_zoom_change)
            self.control_panel.vector_scale_changed.connect(self._handle_vector_scale_change)
            self.control_panel.line_width_changed.connect(self._handle_line_width_change)
            self.control_panel.realtime_update_toggled.connect(self._handle_realtime_toggle)
            self.control_panel.show_vectors_toggled.connect(self._handle_show_vectors_toggle)
            self.control_panel.gravity_toggled.connect(self._handle_gravity_toggle)
            self.control_panel.gravity_strength_changed.connect(self._handle_gravity_strength_change)

        # Connect OpenGL widget signals
        if self.opengl_widget:
            self.opengl_widget.marker_selected.connect(self._handle_marker_selection)
            self.opengl_widget.zoom_changed.connect(self.control_panel.update_zoom_slider)

        # Connect event manager signals
        self.event_manager.grid_updated.connect(self._handle_grid_update)
        self.event_manager.view_changed.connect(self._handle_view_change)
        self.event_manager.marker_selected.connect(self._handle_marker_selection)
        self.event_manager.marker_added.connect(self._handle_marker_added)
        self.event_manager.markers_cleared.connect(self._handle_markers_cleared)
        self.event_manager.fps_updated.connect(self._handle_fps_update)
        self.event_manager.config_changed.connect(self._handle_config_change)

    def _update_loop(self):
        """主更新循环"""
        # 更新状态信息
        self._update_status_info()

        # 实时向量场更新
        if self.realtime_updates_enabled and self.controller and self.opengl_widget and self.opengl_widget.grid is not None:
            self.controller.vector_calculator.update_grid_with_adjacent_sum(self.opengl_widget.grid)
            if self.state_manager:
                self.state_manager.update({"grid_updated": True})

            # 在更新标记之前应用力到向量场
            if self.marker_system:
                # 如果启用重力
                if self.config_manager and self.config_manager.get("gravity_enabled", False):
                    for marker in self.marker_system.get_markers():
                        # 向向量场网格添加向下的重力力（类似于gravity_box.py）
                        self.marker_system.add_vector_at_position(self.opengl_widget.grid, x=marker["x"], y=marker["y"], vy=self.gravity_strength, vx=0.0)

                # 如果启用摩擦 - 反对粒子速度
                if self.config_manager and self.config_manager.get("friction_enabled", False):
                    for marker in self.marker_system.get_markers():
                        # 摩擦作用于速度方向相反
                        # 在标记位置添加一个向量，反对其当前速度
                        friction_vx = -marker["vx"] * self.friction_strength
                        friction_vy = -marker["vy"] * self.friction_strength
                        self.marker_system.add_vector_at_position(self.opengl_widget.grid,
                                                                x=marker["x"], y=marker["y"],
                                                                vx=friction_vx, vy=friction_vy)

                # 根据修改后的向量场更新标记
                self.marker_system.update_markers(self.opengl_widget.grid)

        # 更新OpenGL部件
        if self.opengl_widget:
            self.opengl_widget.update()

    def _update_status_info(self):
        """Update status bar information"""
        if not self.state_manager:
            return

        # Calculate actual FPS
        current_time = time.time()
        if not hasattr(self, '_last_fps_update'):
            self._last_fps_update = current_time
            self._frame_count = 0
            fps = 60  # Initial estimate
        else:
            self._frame_count += 1
            time_diff = current_time - self._last_fps_update
            if time_diff >= 0.5:  # Update FPS every 0.5 seconds
                fps = int(self._frame_count / time_diff)
                self._frame_count = 0
                self._last_fps_update = current_time
            else:
                fps = getattr(self, '_last_fps', 60)
        self._last_fps = fps
        self.fps_label.setText(f"帧率: {fps}")

        # Grid size
        grid_size = self.config_manager.get("grid_size", 64) if self.config_manager else 64
        self.grid_size_label.setText(f"网格: {grid_size}x{grid_size}")

        # Marker count
        marker_count = len(self.marker_system.get_markers()) if self.marker_system else 0
        self.marker_count_label.setText(f"标记: {marker_count}")

        # Camera info
        cam_x = self.state_manager.get("cam_x", 0.0)
        cam_y = self.state_manager.get("cam_y", 0.0)
        cam_zoom = self.state_manager.get("cam_zoom", 1.0)
        self.camera_label.setText(f"相机: ({cam_x:.2f}, {cam_y:.2f}) 缩放: {cam_zoom:.2f}")

    def _reset_view(self):
        """重置视图到默认"""
        if self.controller:
            self.controller.reset_view()
        else:
            # 当控制器不可用时的后备方案
            if self.state_manager:
                self.state_manager.update({
                    "cam_x": 0.0,
                    "cam_y": 0.0,
                    "cam_zoom": 1.0,
                    "view_changed": True
                })

    def _toggle_grid(self):
        """切换网格可见性"""
        if self.controller:
            self.controller.toggle_grid()

    def _clear_grid(self):
        """清除网格"""
        if self.controller:
            self.controller.clear_grid()

    def _generate_tangential(self):
        """生成切向图案"""
        if self.controller:
            self.controller.generate_tangential_pattern()

    def _add_marker(self):
        """添加随机标记"""
        if self.marker_system:
            import numpy as np
            grid_size = self.config_manager.get("grid_size", 64) if self.config_manager else 64
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            self.marker_system.add_marker(x, y)

    def _clear_markers(self):
        """清除所有标记"""
        if self.marker_system:
            self.marker_system.clear_markers()

    def _handle_zoom_change(self, zoom_value: float):
        """处理缩放滑块变化"""
        if self.state_manager:
            self.state_manager.update({
                "cam_zoom": zoom_value,
                "view_changed": True
            })
            # 更新控制面板滑块以匹配
            if self.control_panel:
                self.control_panel.update_zoom_slider(zoom_value)

    def _handle_vector_scale_change(self, scale_value: float):
        """处理向量缩放变化"""
        if self.config_manager:
            self.config_manager.set("vector_scale", scale_value)

    def _handle_line_width_change(self, width_value: float):
        """处理线宽变化"""
        if self.config_manager:
            self.config_manager.set("line_width", width_value)

    def _handle_realtime_toggle(self, enabled: bool):
        """处理实时更新切换"""
        self.realtime_updates_enabled = enabled

    def _handle_show_vectors_toggle(self, enabled: bool):
        """处理显示向量切换"""
        if self.config_manager:
            self.config_manager.set("show_vectors", enabled)

    def _handle_gravity_toggle(self, enabled: bool):
        """处理重力切换"""
        if self.config_manager:
            self.config_manager.set("gravity_enabled", enabled)

    def _handle_gravity_strength_change(self, strength_value: float):
        """处理重力强度变化"""
        self.gravity_strength = strength_value

    def _handle_friction_toggle(self, enabled: bool):
        """处理摩擦切换"""
        if self.config_manager:
            self.config_manager.set("friction_enabled", enabled)

    def _handle_friction_strength_change(self, strength_value: float):
        """处理摩擦强度变化"""
        self.friction_strength = strength_value

    def _handle_marker_selection(self, marker_id: int):
        """处理标记选择"""
        # 更新控制面板或执行其他操作
        pass

    def _handle_grid_update(self):
        """处理网格更新事件"""
        if self.opengl_widget:
            self.opengl_widget.update()

    def _handle_view_change(self, view_data: Dict[str, Any]):
        """处理视图变化事件"""
        if self.opengl_widget:
            self.opengl_widget.update()

    def _handle_marker_added(self, x: int, y: int):
        """处理标记添加事件"""
        self._update_status_info()

    def _handle_markers_cleared(self):
        """处理标记清除事件"""
        self._update_status_info()

    def _handle_fps_update(self, fps: int):
        """处理FPS更新事件"""
        self.fps_label.setText(f"FPS: {fps}")

    def _handle_config_change(self, key: str, value: Any):
        """处理配置变化事件"""
        # 根据配置变化更新UI组件
        if key == "grid_size":
            self._update_status_info()
        elif key == "show_grid":
            if self.opengl_widget:
                self.opengl_widget.update()

    def set_grid(self, grid):
        """设置向量场网格"""
        if self.opengl_widget:
            self.opengl_widget.set_grid(grid)

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        if self.update_timer:
            self.update_timer.stop()
        super().closeEvent(event)


def create_application():
    """创建并返回QApplication实例"""
    app = QApplication(sys.argv)
    app.setApplicationName("LiziEngine")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("LiziEngine")
    return app


def run_gui(controller=None, marker_system=None, config_manager=None,
            state_manager=None, renderer=None, grid=None):
    """运行PyQt6 GUI应用程序"""
    app = create_application()

    # 创建主窗口
    window = MainWindow(controller, marker_system, config_manager,
                       state_manager, renderer)

    # 如果提供网格，则设置网格
    if grid is not None:
        window.set_grid(grid)

    # 显示窗口
    window.show()

    # 运行事件循环
    return app.exec()
