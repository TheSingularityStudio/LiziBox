"""
Control Panel for LiziEngine PyQt6 GUI
Provides UI controls for vector field manipulation and visualization settings
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSlider, QDoubleSpinBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class ControlPanel(QWidget):
    """Control panel with all GUI controls"""

    # Signals emitted when controls are activated
    marker_add_requested = pyqtSignal()
    marker_clear_requested = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    vector_scale_changed = pyqtSignal(float)
    line_width_changed = pyqtSignal(float)
    realtime_update_toggled = pyqtSignal(bool)
    show_vectors_toggled = pyqtSignal(bool)

    def __init__(self, config_manager=None, state_manager=None):
        super().__init__()

        self.config_manager = config_manager
        self.state_manager = state_manager

        self.setFixedWidth(280)
        self.setStyleSheet("""
            ControlPanel {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #404040;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4a9eff;
            }
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 3px;
                padding: 5px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
            QSlider::groove:horizontal {
                border: 1px solid #606060;
                height: 8px;
                background: #404040;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff;
                border: 1px solid #2a7fff;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QLabel {
                color: #cccccc;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel("粒子引擎控制面板")
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #4a9eff; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # View Controls
        view_group = self._create_view_controls()
        layout.addWidget(view_group)

        # Vector Field Controls
        vector_group = self._create_vector_field_controls()
        layout.addWidget(vector_group)

        # Marker Controls
        marker_group = self._create_marker_controls()
        layout.addWidget(marker_group)

        # Settings
        settings_group = self._create_settings_controls()
        layout.addWidget(settings_group)

        # Add stretch to push everything to the top
        layout.addStretch()

    def _create_view_controls(self):
        """Create view control group"""
        group = QGroupBox("视图控制")
        layout = QVBoxLayout(group)

        # Center View button
        center_btn = QPushButton("居中视图")
        center_btn.clicked.connect(self._center_view)
        layout.addWidget(center_btn)

        return group

    def _create_vector_field_controls(self):
        """Create vector field control group"""
        group = QGroupBox("矢量场")
        layout = QVBoxLayout(group)

        # Zoom control
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)  # 0.1 to 10.0
        self.zoom_slider.setValue(100)  # Default 1.0
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        self.zoom_value_label = QLabel("1.00")

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        layout.addLayout(zoom_layout)

        # Vector Scale control
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Vector Scale:")
        self.vector_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.vector_scale_slider.setRange(10, 500)  # 0.1 to 5.0
        self.vector_scale_slider.setValue(100)  # Default 1.0
        self.vector_scale_slider.valueChanged.connect(self._on_vector_scale_changed)
        self.vector_scale_value_label = QLabel("1.00")

        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.vector_scale_slider)
        scale_layout.addWidget(self.vector_scale_value_label)
        layout.addLayout(scale_layout)

        # Line Width control
        width_layout = QHBoxLayout()
        width_label = QLabel("Line Width:")
        self.line_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.line_width_slider.setRange(5, 30)  # 0.5 to 3.0
        self.line_width_slider.setValue(10)  # Default 1.0
        self.line_width_slider.valueChanged.connect(self._on_line_width_changed)
        self.line_width_value_label = QLabel("1.00")

        width_layout.addWidget(width_label)
        width_layout.addWidget(self.line_width_slider)
        width_layout.addWidget(self.line_width_value_label)
        layout.addLayout(width_layout)

        return group

    def _create_marker_controls(self):
        """Create marker control group"""
        group = QGroupBox("标记")
        layout = QVBoxLayout(group)

        # Add Marker button
        add_marker_btn = QPushButton("添加随机标记")
        add_marker_btn.clicked.connect(self.marker_add_requested.emit)
        layout.addWidget(add_marker_btn)

        # Clear Markers button
        clear_markers_btn = QPushButton("清空所有标记")
        clear_markers_btn.clicked.connect(self.marker_clear_requested.emit)
        layout.addWidget(clear_markers_btn)

        return group

    def _create_settings_controls(self):
        """Create settings control group"""
        group = QGroupBox("设置")
        layout = QVBoxLayout(group)

        # Real-time updates checkbox
        self.realtime_checkbox = QCheckBox("Real-time Updates")
        self.realtime_checkbox.setChecked(True)
        self.realtime_checkbox.stateChanged.connect(self._on_realtime_toggled)
        layout.addWidget(self.realtime_checkbox)

        # Show vectors checkbox
        self.show_vectors_checkbox = QCheckBox("Show Vectors")
        self.show_vectors_checkbox.setChecked(True)
        self.show_vectors_checkbox.stateChanged.connect(self._on_show_vectors_toggled)
        layout.addWidget(self.show_vectors_checkbox)

        return group

    def _center_view(self):
        """Center the view on the grid"""
        if self.state_manager and self.config_manager:
            # Get grid size to calculate center
            grid_size = self.config_manager.get("grid_size", 64)
            grid_center_x = grid_size / 2.0
            grid_center_y = grid_size / 2.0

            self.state_manager.update({
                "cam_x": grid_center_x,
                "cam_y": grid_center_y,
                "view_changed": True
            })

    def _on_zoom_changed(self, value):
        """Handle zoom slider change"""
        zoom_value = value / 100.0
        self.zoom_value_label.setText(f"{zoom_value:.2f}")
        self.zoom_changed.emit(zoom_value)

    def update_zoom_slider(self, zoom_value: float):
        """Update zoom slider to match current zoom value"""
        slider_value = int(zoom_value * 100.0)
        self.zoom_slider.blockSignals(True)  # Prevent recursive signal emission
        self.zoom_slider.setValue(slider_value)
        self.zoom_value_label.setText(f"{zoom_value:.2f}")
        self.zoom_slider.blockSignals(False)

    def _on_vector_scale_changed(self, value):
        """Handle vector scale slider change"""
        scale_value = value / 100.0
        self.vector_scale_value_label.setText(f"{scale_value:.2f}")
        self.vector_scale_changed.emit(scale_value)

    def _on_line_width_changed(self, value):
        """Handle line width slider change"""
        width_value = value / 10.0
        self.line_width_value_label.setText(f"{width_value:.2f}")
        self.line_width_changed.emit(width_value)

    def _on_realtime_toggled(self, state):
        """Handle real-time updates checkbox toggle"""
        enabled = state == Qt.CheckState.Checked.value
        self.realtime_update_toggled.emit(enabled)

    def _on_show_vectors_toggled(self, state):
        """Handle show vectors checkbox toggle"""
        enabled = state == Qt.CheckState.Checked.value
        self.show_vectors_toggled.emit(enabled)

    def update_status_info(self, fps=None, grid_size=None, marker_count=None,
                          camera_pos=None):
        """Update status information displays"""
        if fps is not None:
            self.fps_label.setText(f"帧率: {fps}")

        if grid_size is not None:
            self.grid_size_label.setText(f"网格尺寸: {grid_size}x{grid_size}")

        if marker_count is not None:
            self.marker_count_label.setText(f"标记: {marker_count}")

        if camera_pos is not None:
            cam_x, cam_y = camera_pos
            self.camera_pos_label.setText(f"({cam_x:.2f}, {cam_y:.2f})")

    def get_settings(self):
        """Get current settings"""
        return {
            'realtime_updates': self.realtime_checkbox.isChecked(),
            'show_vectors': self.show_vectors_checkbox.isChecked()
        }
