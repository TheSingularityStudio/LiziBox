"""
PyQt6-based Main Window for LiziEngine GUI
Provides the main application window with integrated OpenGL rendering
"""
import sys
import time
from typing import Optional, Dict, Any
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QLabel, QPushButton, QSlider, QGroupBox,
    QGridLayout, QStatusBar, QMenuBar, QMenu
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QKeySequence

from .opengl_widget import OpenGLWidget
from .control_panel import ControlPanel
from .event_manager import EventManager


class MainWindow(QMainWindow):
    """Main application window using PyQt6"""

    def __init__(self, controller=None, marker_system=None, config_manager=None,
                 state_manager=None, renderer=None):
        super().__init__()

        # Store references to core systems
        self.controller = controller
        self.marker_system = marker_system
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.renderer = renderer

        # UI components
        self.opengl_widget = None
        self.control_panel = None
        self.event_manager = EventManager()

        # Real-time updates state
        self.realtime_updates_enabled = True

        # Window properties
        self.setWindowTitle("粒子引擎 - PyQt6 GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize UI
        self._setup_ui()
        self._setup_menus()
        self._setup_status_bar()
        self._connect_signals()

        # Update timer for real-time rendering
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_loop)
        self.update_timer.start(16)  # ~60 FPS

    def _setup_ui(self):
        """Set up the main user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Control panel (left side)
        self.control_panel = ControlPanel(self.config_manager, self.state_manager)
        splitter.addWidget(self.control_panel)

        # OpenGL rendering area (right side)
        self.opengl_widget = OpenGLWidget(
            self.renderer,
            self.state_manager,
            self.config_manager,
            self.marker_system
        )
        splitter.addWidget(self.opengl_widget)

        # Set splitter proportions (control panel narrower)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

    def _setup_menus(self):
        """Set up menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('文件')

        exit_action = QAction('退出', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('视图')

        reset_view_action = QAction('重置视图', self)
        reset_view_action.setShortcut('R')
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

        toggle_grid_action = QAction('切换网格', self)
        toggle_grid_action.setShortcut('G')
        toggle_grid_action.triggered.connect(self._toggle_grid)
        view_menu.addAction(toggle_grid_action)

        # Edit menu
        edit_menu = menubar.addMenu('Edit')

        clear_grid_action = QAction('Clear Grid', self)
        clear_grid_action.setShortcut('C')
        clear_grid_action.triggered.connect(self._clear_grid)
        edit_menu.addAction(clear_grid_action)

        generate_tangential_action = QAction('Generate Tangential Pattern', self)
        generate_tangential_action.setShortcut(QKeySequence('Space'))
        generate_tangential_action.triggered.connect(self._generate_tangential)
        edit_menu.addAction(generate_tangential_action)

    def _setup_status_bar(self):
        """Set up status bar"""
        self.status_bar = self.statusBar()

        # FPS label
        self.fps_label = QLabel("FPS: 0")
        self.status_bar.addWidget(self.fps_label)

        # Grid size label
        self.grid_size_label = QLabel("Grid: 64x64")
        self.status_bar.addWidget(self.grid_size_label)

        # Marker count label
        self.marker_count_label = QLabel("Markers: 0")
        self.status_bar.addWidget(self.marker_count_label)

        # Camera position label
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
        """Main update loop"""
        # Update status information
        self._update_status_info()

        # Real-time vector field updates
        if self.realtime_updates_enabled and self.controller and self.opengl_widget and self.opengl_widget.grid is not None:
            self.controller.vector_calculator.update_grid_with_adjacent_sum(self.opengl_widget.grid)
            if self.state_manager:
                self.state_manager.update({"grid_updated": True})

            # Update markers based on the vector field
            if self.marker_system:
                self.marker_system.update_markers(self.opengl_widget.grid)

        # Update OpenGL widget
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
        """Reset view to default"""
        if self.controller:
            self.controller.reset_view()
        else:
            # Fallback when no controller available
            if self.state_manager:
                self.state_manager.update({
                    "cam_x": 0.0,
                    "cam_y": 0.0,
                    "cam_zoom": 1.0,
                    "view_changed": True
                })

    def _toggle_grid(self):
        """Toggle grid visibility"""
        if self.controller:
            self.controller.toggle_grid()

    def _clear_grid(self):
        """Clear the grid"""
        if self.controller:
            self.controller.clear_grid()

    def _generate_tangential(self):
        """Generate tangential pattern"""
        if self.controller:
            self.controller.generate_tangential_pattern()

    def _add_marker(self):
        """Add a random marker"""
        if self.marker_system:
            import numpy as np
            grid_size = self.config_manager.get("grid_size", 64) if self.config_manager else 64
            x = np.random.randint(0, grid_size)
            y = np.random.randint(0, grid_size)
            self.marker_system.add_marker(x, y)

    def _clear_markers(self):
        """Clear all markers"""
        if self.marker_system:
            self.marker_system.clear_markers()

    def _handle_zoom_change(self, zoom_value: float):
        """Handle zoom slider change"""
        if self.state_manager:
            self.state_manager.update({
                "cam_zoom": zoom_value,
                "view_changed": True
            })
            # Update control panel slider to match
            if self.control_panel:
                self.control_panel.update_zoom_slider(zoom_value)

    def _handle_vector_scale_change(self, scale_value: float):
        """Handle vector scale change"""
        if self.config_manager:
            self.config_manager.set("vector_scale", scale_value)

    def _handle_line_width_change(self, width_value: float):
        """Handle line width change"""
        if self.config_manager:
            self.config_manager.set("line_width", width_value)

    def _handle_realtime_toggle(self, enabled: bool):
        """Handle real-time updates toggle"""
        self.realtime_updates_enabled = enabled

    def _handle_show_vectors_toggle(self, enabled: bool):
        """Handle show vectors toggle"""
        if self.config_manager:
            self.config_manager.set("show_vectors", enabled)

    def _handle_marker_selection(self, marker_id: int):
        """Handle marker selection"""
        # Update control panel or perform other actions
        pass

    def _handle_grid_update(self):
        """Handle grid update event"""
        if self.opengl_widget:
            self.opengl_widget.update()

    def _handle_view_change(self, view_data: Dict[str, Any]):
        """Handle view change event"""
        if self.opengl_widget:
            self.opengl_widget.update()

    def _handle_marker_added(self, x: int, y: int):
        """Handle marker added event"""
        self._update_status_info()

    def _handle_markers_cleared(self):
        """Handle markers cleared event"""
        self._update_status_info()

    def _handle_fps_update(self, fps: int):
        """Handle FPS update event"""
        self.fps_label.setText(f"FPS: {fps}")

    def _handle_config_change(self, key: str, value: Any):
        """Handle configuration change event"""
        # Update UI components based on config changes
        if key == "grid_size":
            self._update_status_info()
        elif key == "show_grid":
            if self.opengl_widget:
                self.opengl_widget.update()

    def set_grid(self, grid):
        """Set the vector field grid"""
        if self.opengl_widget:
            self.opengl_widget.set_grid(grid)

    def closeEvent(self, event):
        """Handle window close event"""
        if self.update_timer:
            self.update_timer.stop()
        super().closeEvent(event)


def create_application():
    """Create and return QApplication instance"""
    app = QApplication(sys.argv)
    app.setApplicationName("LiziEngine")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("LiziEngine")
    return app


def run_gui(controller=None, marker_system=None, config_manager=None,
            state_manager=None, renderer=None, grid=None):
    """Run the PyQt6 GUI application"""
    app = create_application()

    # Create main window
    window = MainWindow(controller, marker_system, config_manager,
                       state_manager, renderer)

    # Set grid if provided
    if grid is not None:
        window.set_grid(grid)

    # Show window
    window.show()

    # Run event loop
    return app.exec()
