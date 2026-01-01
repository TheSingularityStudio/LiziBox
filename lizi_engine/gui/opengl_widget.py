"""
OpenGL Widget for LiziEngine PyQt6 GUI
Provides OpenGL rendering integration within Qt interface
"""
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeyEvent
from OpenGL.GL import *


class OpenGLWidget(QOpenGLWidget):
    """OpenGL rendering widget integrated with PyQt6"""

    # Signals
    marker_selected = pyqtSignal(int)  # marker_id

    def __init__(self, renderer=None, state_manager=None, config_manager=None):
        super().__init__()

        # Core system references
        self.renderer = renderer
        self.state_manager = state_manager
        self.config_manager = config_manager

        # Rendering data
        self.grid = None

        # Mouse interaction state
        self.selected_marker = None
        self.mouse_dragging = False
        self.last_mouse_pos = QPoint(0, 0)

        # OpenGL settings
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_grid(self, grid: np.ndarray):
        """Set the vector field grid data"""
        self.grid = grid
        self.update()

    def initializeGL(self):
        """Initialize OpenGL context"""
        try:
            # Set clear color (dark background)
            glClearColor(0.1, 0.1, 0.1, 1.0)

            # Enable depth testing
            glEnable(GL_DEPTH_TEST)

            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Enable line smoothing
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

            print("[OpenGL Widget] OpenGL initialized successfully")

        except Exception as e:
            print(f"[OpenGL Widget] OpenGL initialization failed: {e}")

    def resizeGL(self, width: int, height: int):
        """Handle widget resize"""
        glViewport(0, 0, width, height)

        # Update viewport in state manager
        if self.state_manager:
            self.state_manager.update({
                "viewport_width": width,
                "viewport_height": height
            })

    def paintGL(self):
        """Render the OpenGL scene"""
        if not self.renderer or self.grid is None:
            # Clear screen if no renderer or grid
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return

        try:
            # Get camera parameters
            cam_x = self.state_manager.get("cam_x", 0.0) if self.state_manager else 0.0
            cam_y = self.state_manager.get("cam_y", 0.0) if self.state_manager else 0.0
            cam_zoom = self.state_manager.get("cam_zoom", 1.0) if self.state_manager else 1.0

            # Get viewport dimensions
            viewport_width = self.width()
            viewport_height = self.height()

            # Get cell size from config
            cell_size = self.config_manager.get("cell_size", 1.0) if self.config_manager else 1.0

            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render background
            self.renderer.render_background()

            # Render vector field
            self.renderer.render_vector_field(
                self.grid,
                cell_size=cell_size,
                cam_x=cam_x,
                cam_y=cam_y,
                cam_zoom=cam_zoom,
                viewport_width=viewport_width,
                viewport_height=viewport_height
            )

            # Render grid
            self.renderer.render_grid(
                self.grid,
                cell_size=cell_size,
                cam_x=cam_x,
                cam_y=cam_y,
                cam_zoom=cam_zoom,
                viewport_width=viewport_width,
                viewport_height=viewport_height
            )

            # Render markers
            self._render_markers(cam_x, cam_y, cam_zoom, viewport_width, viewport_height, cell_size)

        except Exception as e:
            print(f"[OpenGL Widget] Rendering error: {e}")
            # Clear screen on error
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def _render_markers(self, cam_x: float, cam_y: float, cam_zoom: float,
                        viewport_width: int, viewport_height: int, cell_size: float):
        """Render markers if renderer supports it"""
        try:
            if hasattr(self.renderer, 'render_markers'):
                self.renderer.render_markers(
                    cell_size=cell_size,
                    cam_x=cam_x,
                    cam_y=cam_y,
                    cam_zoom=cam_zoom,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height
                )
        except Exception as e:
            # Markers are not critical, just log and continue
            print(f"[OpenGL Widget] Marker rendering error: {e}")

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self._handle_left_mouse_press(event.pos())
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._handle_middle_mouse_press(event.pos())

        self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_dragging = False
            self.selected_marker = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            pass  # Middle button release handled elsewhere

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self.mouse_dragging and self.selected_marker is not None:
            self._handle_mouse_drag(event.pos())

        self.last_mouse_pos = event.pos()

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel events"""
        if not self.state_manager:
            return

        # Get current zoom
        cam_zoom = self.state_manager.get("cam_zoom", 1.0)

        # Calculate zoom change
        zoom_speed = 0.1
        delta = event.angleDelta().y()
        if delta > 0:
            cam_zoom *= (1.0 + zoom_speed)
        else:
            cam_zoom *= (1.0 - zoom_speed)

        # Clamp zoom
        cam_zoom = max(0.1, min(10.0, cam_zoom))

        # Update state
        self.state_manager.update({
            "cam_zoom": cam_zoom,
            "view_changed": True
        })

        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events"""
        key = event.key()

        # Handle specific keys
        if key == Qt.Key.Key_R:
            self._reset_view()
        elif key == Qt.Key.Key_G:
            self._toggle_grid()
        elif key == Qt.Key.Key_C:
            self._clear_grid()
        elif key == Qt.Key.Key_Space:
            self._generate_tangential()

        # Call parent for other keys
        super().keyPressEvent(event)

    def _handle_left_mouse_press(self, pos: QPoint):
        """Handle left mouse button press"""
        # Convert screen coordinates to world coordinates
        world_pos = self._screen_to_world(pos)

        if world_pos:
            world_x, world_y = world_pos

            # Try to select a marker (this would need marker system integration)
            # For now, just emit a generic marker selected signal
            # In full implementation, this would check marker positions

            self.mouse_dragging = True
            # self.marker_selected.emit(marker_id)  # Would emit actual marker ID

    def _handle_middle_mouse_press(self, pos: QPoint):
        """Handle middle mouse button press for panning"""
        # Middle button press starts panning
        pass

    def _handle_mouse_drag(self, pos: QPoint):
        """Handle mouse drag for marker movement"""
        if not self.selected_marker:
            return

        # Convert to world coordinates and update marker position
        world_pos = self._screen_to_world(pos)
        if world_pos:
            world_x, world_y = world_pos
            # Update marker position through marker system
            # marker_system.update_marker_position(self.selected_marker, world_x, world_y)

    def _screen_to_world(self, screen_pos: QPoint):
        """Convert screen coordinates to world coordinates"""
        if not self.state_manager:
            return None

        cam_x = self.state_manager.get("cam_x", 0.0)
        cam_y = self.state_manager.get("cam_y", 0.0)
        cam_zoom = self.state_manager.get("cam_zoom", 1.0)

        # Get viewport dimensions
        viewport_width = self.width()
        viewport_height = self.height()

        # Convert screen to normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * screen_pos.x() / viewport_width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_pos.y() / viewport_height)

        # Convert to world coordinates
        world_x = cam_x + (ndc_x * viewport_width / (2.0 * cam_zoom))
        world_y = cam_y + (ndc_y * viewport_height / (2.0 * cam_zoom))

        return world_x, world_y

    def _reset_view(self):
        """Reset view to default"""
        if self.state_manager:
            self.state_manager.update({
                "cam_x": 0.0,
                "cam_y": 0.0,
                "cam_zoom": 1.0,
                "view_changed": True
            })
            self.update()

    def _toggle_grid(self):
        """Toggle grid visibility"""
        if self.config_manager:
            current = self.config_manager.get("show_grid", True)
            self.config_manager.set("show_grid", not current)
            self.update()

    def _clear_grid(self):
        """Clear the grid"""
        if self.grid is not None:
            self.grid.fill(0)
            if self.state_manager:
                self.state_manager.update({"grid_updated": True})
            self.update()

    def _generate_tangential(self):
        """Generate tangential pattern"""
        if self.grid is not None:
            # This would integrate with vector_calculator
            # For now, just update the grid
            if self.state_manager:
                self.state_manager.update({"grid_updated": True})
            self.update()

    def cleanup(self):
        """Clean up OpenGL resources"""
        # OpenGL cleanup is handled automatically by QOpenGLWidget
        pass
