"""
控制面板模块 - 提供用户界面控件
包含按钮、滑块等用于控制向量场和渲染参数
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QCheckBox, QGroupBox, QSpinBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlPanel(QWidget):
    """控制面板类"""

    # 信号定义
    view_reset_requested = pyqtSignal()
    grid_toggle_requested = pyqtSignal()
    grid_clear_requested = pyqtSignal()
    tangential_generate_requested = pyqtSignal()
    marker_add_requested = pyqtSignal()
    marker_clear_requested = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    vector_scale_changed = pyqtSignal(float)
    line_width_changed = pyqtSignal(float)
    realtime_update_toggled = pyqtSignal(bool)

    def __init__(self, config_manager, state_manager, parent=None):
        super().__init__(parent)

        self._config_manager = config_manager
        self._state_manager = state_manager

        # UI 组件
        self._view_reset_button = None
        self._grid_toggle_button = None
        self._grid_clear_button = None
        self._tangential_generate_button = None
        self._marker_add_button = None
        self._marker_clear_button = None
        self._zoom_slider = None
        self._vector_scale_slider = None
        self._line_width_slider = None
        self._realtime_update_checkbox = None

        # 状态标签
        self._fps_label = None
        self._grid_size_label = None
        self._marker_count_label = None
        self._camera_pos_label = None

        # 初始化UI
        self._init_ui()

        # 连接信号
        self._connect_signals()

    def _init_ui(self) -> None:
        """初始化用户界面"""
        layout = QVBoxLayout(self)

        # 视图控制组
        view_group = QGroupBox("视图控制")
        view_layout = QVBoxLayout(view_group)

        self._view_reset_button = QPushButton("重置视图")
        self._view_reset_button.setToolTip("重置相机视图到初始位置")
        view_layout.addWidget(self._view_reset_button)

        self._grid_toggle_button = QPushButton("切换网格")
        self._grid_toggle_button.setToolTip("显示/隐藏网格")
        view_layout.addWidget(self._grid_toggle_button)

        layout.addWidget(view_group)

        # 网格控制组
        grid_group = QGroupBox("网格控制")
        grid_layout = QVBoxLayout(grid_group)

        self._grid_clear_button = QPushButton("清空网格")
        self._grid_clear_button.setToolTip("清空所有向量")
        grid_layout.addWidget(self._grid_clear_button)

        self._tangential_generate_button = QPushButton("生成切线模式")
        self._tangential_generate_button.setToolTip("生成旋转向量场模式")
        grid_layout.addWidget(self._tangential_generate_button)

        layout.addWidget(grid_group)

        # 标记控制组
        marker_group = QGroupBox("标记控制")
        marker_layout = QVBoxLayout(marker_group)

        self._marker_add_button = QPushButton("添加标记")
        self._marker_add_button.setToolTip("在随机位置添加标记")
        marker_layout.addWidget(self._marker_add_button)

        self._marker_clear_button = QPushButton("清空标记")
        self._marker_clear_button.setToolTip("移除所有标记")
        marker_layout.addWidget(self._marker_clear_button)

        layout.addWidget(marker_group)

        # 渲染参数组
        render_group = QGroupBox("渲染参数")
        render_layout = QVBoxLayout(render_group)

        # 缩放滑块
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("缩放:"))
        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(1, 1000)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setToolTip("调整视图缩放级别")
        zoom_layout.addWidget(self._zoom_slider)
        render_layout.addLayout(zoom_layout)

        # 向量缩放滑块
        vector_scale_layout = QHBoxLayout()
        vector_scale_layout.addWidget(QLabel("向量缩放:"))
        self._vector_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._vector_scale_slider.setRange(1, 500)
        self._vector_scale_slider.setValue(100)
        self._vector_scale_slider.setToolTip("调整向量显示大小")
        vector_scale_layout.addWidget(self._vector_scale_slider)
        render_layout.addLayout(vector_scale_layout)

        # 线宽滑块
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("线宽:"))
        self._line_width_slider = QSlider(Qt.Orientation.Horizontal)
        self._line_width_slider.setRange(1, 50)
        self._line_width_slider.setValue(10)
        self._line_width_slider.setToolTip("调整向量线条宽度")
        line_width_layout.addWidget(self._line_width_slider)
        render_layout.addLayout(line_width_layout)

        # 实时更新复选框
        self._realtime_update_checkbox = QCheckBox("实时更新")
        self._realtime_update_checkbox.setChecked(True)
        self._realtime_update_checkbox.setToolTip("启用/禁用实时向量场更新")
        render_layout.addWidget(self._realtime_update_checkbox)

        layout.addWidget(render_group)

        # 状态信息组
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout(status_group)

        self._fps_label = QLabel("FPS: 0")
        status_layout.addWidget(self._fps_label)

        self._grid_size_label = QLabel("网格: 0x0")
        status_layout.addWidget(self._grid_size_label)

        self._marker_count_label = QLabel("标记: 0")
        status_layout.addWidget(self._marker_count_label)

        self._camera_pos_label = QLabel("相机: (0.0, 0.0)")
        status_layout.addWidget(self._camera_pos_label)

        layout.addWidget(status_group)

        # 设置布局属性
        layout.addStretch()

    def _connect_signals(self) -> None:
        """连接信号"""
        self._view_reset_button.clicked.connect(self.view_reset_requested.emit)
        self._grid_toggle_button.clicked.connect(self.grid_toggle_requested.emit)
        self._grid_clear_button.clicked.connect(self.grid_clear_requested.emit)
        self._tangential_generate_button.clicked.connect(self.tangential_generate_requested.emit)
        self._marker_add_button.clicked.connect(self.marker_add_requested.emit)
        self._marker_clear_button.clicked.connect(self.marker_clear_requested.emit)

        self._zoom_slider.valueChanged.connect(lambda v: self.zoom_changed.emit(v / 100.0))
        self._vector_scale_slider.valueChanged.connect(lambda v: self.vector_scale_changed.emit(v / 100.0))
        self._line_width_slider.valueChanged.connect(lambda v: self.line_width_changed.emit(v / 10.0))
        self._realtime_update_checkbox.toggled.connect(self.realtime_update_toggled.emit)

    def update_status_info(self, fps: int, grid_size: tuple, marker_count: int, camera_pos: tuple) -> None:
        """更新状态信息"""
        self._fps_label.setText(f"FPS: {fps}")
        self._grid_size_label.setText(f"网格: {grid_size[0]}x{grid_size[1]}")
        self._marker_count_label.setText(f"标记: {marker_count}")
        self._camera_pos_label.setText(f"相机: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f})")
