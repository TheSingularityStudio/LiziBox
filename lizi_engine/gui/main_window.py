"""
PyQt6-based Main Window for LiziEngine GUI
Provides the main application window with integrated OpenGL rendering
"""
import sys
from typing import Optional, Dict, Any
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

        # Window properties
        self.setWindowTitle("LiziEngine - PyQt6 GUI")
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
            self.config_manager
        )
        splitter.addWidget(self.opengl_widget)

        # Set splitter proportions (control panel narrower)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

    def _setup_menus(self):
        """Set up menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('R')
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

        toggle_grid_action = QAction('Toggle Grid', self)
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
            self.control_panel.view_reset_requested.connect(self._reset_view)
            self.control_panel.grid_toggle_requested.connect(self._toggle_grid)
            self.control_panel.grid_clear_requested.connect(self._clear_grid)
            self.control_panel.tangential_generate_requested.connect(self._generate_tangential)
            self.control_panel.marker_add_requested.connect(self._add_marker)
            self.control_panel.marker_clear_requested.connect(self._clear_markers)
            self.control_panel.zoom_changed.connect(self._handle_zoom_change)
            self.control_panel.vector_scale_changed.connect(self._handle_vector_scale_change)
            self.control_panel.line_width_changed.connect(self._handle_line_width_change)

        # Connect OpenGL widget signals
        if self.opengl_widget:
            self.opengl_widget.marker_selected.connect(self._handle_marker_selection)

        # Connect event manager
        self.event_manager.grid_updated.connect(self._handle_grid_update)
        self.event_manager.view_changed.connect(self._handle_view_change)

    def _update_loop(self):
        """Main update loop"""
        # Update status information
        self._update_status_info()

        # Update OpenGL widget
        if self.opengl_widget:
            self.opengl_widget.update()

    def _update_status_info(self):
        """Update status bar information"""
        if not self.state_manager:
            return

        # FPS (simplified)
        fps = 60  # In a real implementation, calculate actual FPS
        self.fps_label.setText(f"FPS: {fps}")

        # Grid size
        grid_size = self.config_manager.get("grid_size", 64) if self.config_manager else 64
        self.grid_size_label.setText(f"Grid: {grid_size}x{grid_size}")

        # Marker count
        marker_count = len(self.marker_system.get_markers()) if self.marker_system else 0
        self.marker_count_label.setText(f"Markers: {marker_count}")

        # Camera info
        cam_x = self.state_manager.get("cam_x", 0.0)
        cam_y = self.state_manager.get("cam_y", 0.0)
        cam_zoom = self.state_manager.get("cam_zoom", 1.0)
        self.camera_label.setText(".2f")

    def _reset_view(self):
        """Reset view to default"""
        if self.controller:
            self.controller.reset_view()
        elif self.state_manager:
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

    def _handle_vector_scale_change(self, scale_value: float):
        """Handle vector scale change"""
        if self.config_manager:
            self.config_manager.set("vector_scale", scale_value)

    def _handle_line_width_change(self, width_value: float):
        """Handle line width change"""
        if self.config_manager:
            self.config_manager.set("line_width", width_value)

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
