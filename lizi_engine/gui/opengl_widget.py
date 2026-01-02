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
    zoom_changed = pyqtSignal(float)  # zoom_value

    def __init__(self, renderer=None, state_manager=None, config_manager=None, marker_system=None, controller=None):
        super().__init__()

        # Core system references
        self.renderer = renderer
        self.state_manager = state_manager
        self.config_manager = config_manager
        self.marker_system = marker_system
        self.controller = controller

        # Rendering data
        self.grid = None

        # Mouse interaction state
        self.selected_marker = None
        self.mouse_dragging = False
        self.middle_mouse_dragging = False
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
            if hasattr(self.renderer, 'render_markers') and self.marker_system:
                # Get markers from marker system
                markers = self.marker_system.get_markers()
                # Update state manager with markers
                if self.state_manager:
                    self.state_manager.update({"markers": markers})
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
            self.middle_mouse_dragging = False

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self.mouse_dragging and self.selected_marker is not None:
            self._handle_mouse_drag(event.pos())
        elif self.middle_mouse_dragging:
            self._handle_middle_mouse_drag(event.pos())

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

        # Emit zoom changed signal to update control panel
        if hasattr(self, 'zoom_changed') and self.zoom_changed:
            self.zoom_changed.emit(cam_zoom)

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
        if not self.controller:
            return

        # Check mouse mode from state manager
        mouse_mode = self.state_manager.get("mouse_mode", "drag") if self.state_manager else "drag"

        if mouse_mode == "place":
            # Place a new marker at the click position
            world_pos = self._screen_to_world(pos)
            if world_pos:
                world_x, world_y = world_pos
                # Convert world coordinates to grid coordinates
                cell_size = self.config_manager.get("cell_size", 1.0) if self.config_manager else 1.0
                gx = world_x / cell_size
                gy = world_y / cell_size
                if self.marker_system:
                    self.marker_system.add_marker(gx, gy)
                self.selected_marker = None  # No marker selected for dragging
        else:  # drag mode (default)
            # Select marker for dragging using controller
            self.selected_marker = self.controller.handle_mouse_left_press(pos.x(), pos.y())

        self.mouse_dragging = True

    def _handle_middle_mouse_press(self, pos: QPoint):
        """Handle middle mouse button press for panning"""
        # Middle button press starts panning
        self.middle_mouse_dragging = True

    def _handle_middle_mouse_drag(self, pos: QPoint):
        """Handle middle mouse drag for panning"""
        if not self.state_manager:
            return

        # Calculate movement in screen space
        delta_x = pos.x() - self.last_mouse_pos.x()
        delta_y = pos.y() - self.last_mouse_pos.y()

        # Get current camera position and zoom
        cam_x = self.state_manager.get("cam_x", 0.0)
        cam_y = self.state_manager.get("cam_y", 0.0)
        cam_zoom = self.state_manager.get("cam_zoom", 1.0)

        # Convert screen movement to world movement
        # Movement should be inversely proportional to zoom (higher zoom = slower panning)
        pan_speed = 1.0 / cam_zoom
        world_delta_x = -delta_x * pan_speed
        world_delta_y = -delta_y * pan_speed  # Y is inverted in screen coordinates

        # Update camera position
        new_cam_x = cam_x + world_delta_x
        new_cam_y = cam_y + world_delta_y

        # Update state
        self.state_manager.update({
            "cam_x": new_cam_x,
            "cam_y": new_cam_y,
            "view_changed": True
        })

        self.update()

    def _handle_mouse_drag(self, pos: QPoint):
        """Handle mouse drag for marker movement"""
        if self.selected_marker is None or not self.controller:
            return

        # Use controller to handle mouse drag
        self.controller.handle_mouse_drag(pos.x(), pos.y(), self.selected_marker)

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

        # Convert screen coordinates to world coordinates (same as controller.py)
        world_x = cam_x + (screen_pos.x() - (viewport_width / 2.0)) / cam_zoom
        world_y = cam_y + (screen_pos.y() - (viewport_height / 2.0)) / cam_zoom

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

    def _find_marker_at_position(self, world_x: float, world_y: float):
        """Find marker at the given world position"""
        if not self.marker_system:
            return None

        markers = self.marker_system.get_markers()
        marker_size = self.config_manager.get("marker_size", 8) if self.config_manager else 8

        # Convert marker size to world units (rough approximation)
        cam_zoom = self.state_manager.get("cam_zoom", 1.0) if self.state_manager else 1.0
        world_marker_size = marker_size / cam_zoom

        for i, marker in enumerate(markers):
            marker_x = marker.get("x", 0.0)
            marker_y = marker.get("y", 0.0)

            # Check if click is within marker bounds
            if (abs(world_x - marker_x) <= world_marker_size and
                abs(world_y - marker_y) <= world_marker_size):
                return i

        return None

    def _update_marker_position(self, marker_index: int, world_x: float, world_y: float):
        """Update marker position in the marker system"""
        if not self.marker_system:
            return

        markers = self.marker_system.markers
        if 0 <= marker_index < len(markers):
            markers[marker_index]["x"] = world_x
            markers[marker_index]["y"] = world_y
            self.marker_system._sync_to_state_manager()
            self.update()

    def cleanup(self):
        """Clean up OpenGL resources"""
        # OpenGL cleanup is handled automatically by QOpenGLWidget
        pass
