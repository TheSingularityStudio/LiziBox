"""
PyQt6 GUI Demo for LiziEngine
Demonstrates the new PyQt6-based GUI system
"""
import sys
import os
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.app import AppCore
from lizi_engine.core.config import config_manager
from lizi_engine.core.state import state_manager
from lizi_engine.compute.vector_field import vector_calculator
from plugins.controller import Controller
from plugins.marker_system import MarkerSystem
from lizi_engine.gui import run_gui


def main():
    """Main function for PyQt6 GUI demo"""
    print("Starting LiziEngine PyQt6 GUI Demo...")

    # Initialize application core
    app_core = AppCore()

    # Load configuration
    config_manager.load_from_file("config.json")

    # Create grid
    grid_size = config_manager.get("grid_size", 64)
    grid = np.zeros((grid_size, grid_size, 2), dtype=np.float32)

    # Initialize systems
    marker_system = MarkerSystem(app_core)
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # Get renderer (this would need proper dependency injection in full implementation)
    renderer = None
    try:
        from lizi_engine.graphics.renderer import VectorFieldRenderer
        renderer = VectorFieldRenderer()
    except Exception as e:
        print(f"Warning: Could not initialize renderer: {e}")

    print("Launching PyQt6 GUI...")

    # Run the PyQt6 GUI
    try:
        exit_code = run_gui(
            controller=controller,
            marker_system=marker_system,
            config_manager=config_manager,
            state_manager=state_manager,
            renderer=renderer,
            grid=grid
        )
        print("PyQt6 GUI demo completed")
        return exit_code

    except Exception as e:
        print(f"Error running PyQt6 GUI: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
