"""
Event Manager for LiziEngine PyQt6 GUI
Centralized event handling using Qt signals and slots
"""
from typing import Dict, Any, Callable
from PyQt6.QtCore import QObject, pyqtSignal


class EventManager(QObject):
    """Centralized event manager using Qt signals"""

    # Application lifecycle events
    app_initialized = pyqtSignal()
    app_closing = pyqtSignal()

    # View events
    view_changed = pyqtSignal(dict)  # view_data
    camera_moved = pyqtSignal(float, float)  # cam_x, cam_y
    zoom_changed = pyqtSignal(float)  # zoom_level

    # Grid events
    grid_updated = pyqtSignal()  # Grid data changed
    grid_cleared = pyqtSignal()  # Grid cleared
    grid_size_changed = pyqtSignal(int, int)  # width, height

    # Marker events
    marker_added = pyqtSignal(int, int)  # x, y
    marker_removed = pyqtSignal(int)  # marker_id
    marker_selected = pyqtSignal(int)  # marker_id
    markers_cleared = pyqtSignal()  # All markers cleared
    marker_moved = pyqtSignal(int, float, float)  # marker_id, x, y

    # Input events
    key_pressed = pyqtSignal(str)  # key_name
    mouse_clicked = pyqtSignal(int, int, int)  # x, y, button
    mouse_dragged = pyqtSignal(int, int)  # x, y
    mouse_wheel = pyqtSignal(int)  # delta

    # Rendering events
    render_frame = pyqtSignal()  # New frame should be rendered
    fps_updated = pyqtSignal(int)  # current_fps

    # Configuration events
    config_changed = pyqtSignal(str, Any)  # key, value
    settings_updated = pyqtSignal(dict)  # settings_dict

    def __init__(self):
        super().__init__()
        self._event_handlers = {}

    def emit_view_changed(self, view_data: Dict[str, Any]):
        """Emit view changed event"""
        self.view_changed.emit(view_data)

    def emit_camera_moved(self, cam_x: float, cam_y: float):
        """Emit camera moved event"""
        self.camera_moved.emit(cam_x, cam_y)

    def emit_zoom_changed(self, zoom_level: float):
        """Emit zoom changed event"""
        self.zoom_changed.emit(zoom_level)

    def emit_grid_updated(self):
        """Emit grid updated event"""
        self.grid_updated.emit()

    def emit_grid_cleared(self):
        """Emit grid cleared event"""
        self.grid_cleared.emit()

    def emit_marker_added(self, x: int, y: int):
        """Emit marker added event"""
        self.marker_added.emit(x, y)

    def emit_marker_selected(self, marker_id: int):
        """Emit marker selected event"""
        self.marker_selected.emit(marker_id)

    def emit_markers_cleared(self):
        """Emit markers cleared event"""
        self.markers_cleared.emit()

    def emit_key_pressed(self, key_name: str):
        """Emit key pressed event"""
        self.key_pressed.emit(key_name)

    def emit_mouse_clicked(self, x: int, y: int, button: int):
        """Emit mouse clicked event"""
        self.mouse_clicked.emit(x, y, button)

    def emit_mouse_wheel(self, delta: int):
        """Emit mouse wheel event"""
        self.mouse_wheel.emit(delta)

    def emit_render_frame(self):
        """Emit render frame event"""
        self.render_frame.emit()

    def emit_fps_updated(self, fps: int):
        """Emit FPS updated event"""
        self.fps_updated.emit(fps)

    def emit_config_changed(self, key: str, value: Any):
        """Emit configuration changed event"""
        self.config_changed.emit(key, value)

    def connect_event_handler(self, signal_name: str, handler: Callable):
        """Connect a custom event handler to a signal"""
        if hasattr(self, signal_name):
            signal = getattr(self, signal_name)
            signal.connect(handler)
        else:
            print(f"Warning: Signal '{signal_name}' not found in EventManager")

    def disconnect_event_handler(self, signal_name: str, handler: Callable):
        """Disconnect a custom event handler from a signal"""
        if hasattr(self, signal_name):
            signal = getattr(self, signal_name)
            signal.disconnect(handler)
        else:
            print(f"Warning: Signal '{signal_name}' not found in EventManager")

    def register_custom_event(self, event_name: str):
        """Register a custom event signal"""
        if not hasattr(self, event_name):
            setattr(self, event_name, pyqtSignal())
            print(f"Registered custom event: {event_name}")
        else:
            print(f"Event '{event_name}' already exists")

    def emit_custom_event(self, event_name: str, *args):
        """Emit a custom event"""
        if hasattr(self, event_name):
            signal = getattr(self, event_name)
            signal.emit(*args)
        else:
            print(f"Warning: Custom event '{event_name}' not registered")


# Global event manager instance
event_manager = EventManager()
