"""
LiziEngine GUI Package - PyQt6-based User Interface
Provides modern, reliable GUI components for vector field visualization
"""

from .main_window import MainWindow, run_gui, create_application
from .control_panel import ControlPanel
from .opengl_widget import OpenGLWidget
from .event_manager import EventManager, event_manager

__all__ = [
    'MainWindow',
    'ControlPanel',
    'OpenGLWidget',
    'EventManager',
    'event_manager',
    'run_gui',
    'create_application'
]

__version__ = "2.0.0"
