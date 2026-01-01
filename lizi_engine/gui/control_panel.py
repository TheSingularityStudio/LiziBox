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
    view_reset_requested = pyqtSignal()
    grid_toggle_requested = pyqtSignal()
    grid_clear_requested = pyqtSignal()
    tangential_generate_requested = pyqtSignal()
    marker_add_requested = pyqtSignal()
    marker_clear_requested = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    vector_scale_changed = pyqtSignal(float)
    line_width_changed = pyqtSignal(float)

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
        title_label = QLabel("LiziEngine Control Panel")
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

        # Status Information
        status_group = self._create_status_info()
        layout.addWidget(status_group)

        # Add stretch to push everything to the top
        layout.addStretch()

    def _create_view_controls(self):
        """Create view control group"""
        group = QGroupBox("View Controls")
        layout = QVBoxLayout(group)

        # Reset View button
        reset_btn = QPushButton("Reset View (R)")
        reset_btn.clicked.connect(self.view_reset_requested.emit)
        layout.addWidget(reset_btn)

        # Center View button
        center_btn = QPushButton("Center View")
        center_btn.clicked.connect(self._center_view)
        layout.addWidget(center_btn)

        # Toggle Grid button
        grid_btn = QPushButton("Toggle Grid (G)")
        grid_btn.clicked.connect(self.grid_toggle_requested.emit)
        layout.addWidget(grid_btn)

        # Clear Grid button
        clear_btn = QPushButton("Clear Grid (C)")
        clear_btn.clicked.connect(self.grid_clear_requested.emit)
        layout.addWidget(clear_btn)

        # Generate Tangential button
        tangential_btn = QPushButton("Generate Tangential (Space)")
        tangential_btn.clicked.connect(self.tangential_generate_requested.emit)
        layout.addWidget(tangential_btn)

        return group

    def _create_vector_field_controls(self):
        """Create vector field control group"""
        group = QGroupBox("Vector Field")
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
        group = QGroupBox("Markers")
        layout = QVBoxLayout(group)

        # Add Marker button
        add_marker_btn = QPushButton("Add Random Marker")
        add_marker_btn.clicked.connect(self.marker_add_requested.emit)
        layout.addWidget(add_marker_btn)

        # Clear Markers button
        clear_markers_btn = QPushButton("Clear All Markers")
        clear_markers_btn.clicked.connect(self.marker_clear_requested.emit)
        layout.addWidget(clear_markers_btn)

        return group

    def _create_settings_controls(self):
        """Create settings control group"""
        group = QGroupBox("Settings")
        layout = QVBoxLayout(group)

        # Real-time updates checkbox
        self.realtime_checkbox = QCheckBox("Real-time Updates")
        self.realtime_checkbox.setChecked(True)
        layout.addWidget(self.realtime_checkbox)

        # Show grid checkbox
        self.show_grid_checkbox = QCheckBox("Show Grid")
        self.show_grid_checkbox.setChecked(True)
        layout.addWidget(self.show_grid_checkbox)

        # Show vectors checkbox
        self.show_vectors_checkbox = QCheckBox("Show Vectors")
        self.show_vectors_checkbox.setChecked(True)
        layout.addWidget(self.show_vectors_checkbox)

        return group

    def _create_status_info(self):
        """Create status information group"""
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)

        # FPS display
        self.fps_label = QLabel("FPS: --")
        layout.addWidget(self.fps_label)

        # Grid size display
        self.grid_size_label = QLabel("Grid Size: --")
        layout.addWidget(self.grid_size_label)

        # Marker count display
        self.marker_count_label = QLabel("Markers: --")
        layout.addWidget(self.marker_count_label)

        # Camera position display
        self.camera_pos_label = QLabel("Camera: (--, --)")
        layout.addWidget(self.camera_pos_label)

        return group

    def _center_view(self):
        """Center the view on the grid"""
        if self.state_manager:
            # This would need grid information to center properly
            # For now, just reset to origin
            self.state_manager.update({
                "cam_x": 0.0,
                "cam_y": 0.0,
                "view_changed": True
            })

    def _on_zoom_changed(self, value):
        """Handle zoom slider change"""
        zoom_value = value / 100.0
        self.zoom_value_label.setText(".2f")
        self.zoom_changed.emit(zoom_value)

    def _on_vector_scale_changed(self, value):
        """Handle vector scale slider change"""
        scale_value = value / 100.0
        self.vector_scale_value_label.setText(".2f")
        self.vector_scale_changed.emit(scale_value)

    def _on_line_width_changed(self, value):
        """Handle line width slider change"""
        width_value = value / 10.0
        self.line_width_value_label.setText(".2f")
        self.line_width_changed.emit(width_value)

    def update_status_info(self, fps=None, grid_size=None, marker_count=None,
                          camera_pos=None):
        """Update status information displays"""
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps}")

        if grid_size is not None:
            self.grid_size_label.setText(f"Grid Size: {grid_size}x{grid_size}")

        if marker_count is not None:
            self.marker_count_label.setText(f"Markers: {marker_count}")

        if camera_pos is not None:
            cam_x, cam_y = camera_pos
            self.camera_pos_label.setText(".2f")

    def get_settings(self):
        """Get current settings"""
        return {
            'realtime_updates': self.realtime_checkbox.isChecked(),
            'show_grid': self.show_grid_checkbox.isChecked(),
            'show_vectors': self.show_vectors_checkbox.isChecked()
        }
