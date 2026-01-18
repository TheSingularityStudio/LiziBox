"""
LiziEngine 基本使用示例
演示如何使用LiziEngine创建和渲染向量场
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lizi_engine.core.container import container
from lizi_engine.core.app import AppCore, GUI_AVAILABLE
from lizi_engine.core.state import state_manager
from lizi_engine.compute.vector_field import vector_calculator
from lizi_engine.compute.marker_system import MarkerSystem
from lizi_engine.core.plugin import UIManager, Controller, add_inward_edge_vectors
from PyQt6.QtCore import QTimer

def main():
    """主函数"""
    print("[示例] 启动LiziEngine基本使用示例...")

    if not GUI_AVAILABLE:
        print("[示例] GUI不可用，程序退出")
        return

    # 初始化应用核心（GUI会在AppCore初始化时自动创建）
    app_core = container.resolve(AppCore)
    if app_core is None:
        # 如果容器中没有 AppCore 实例，创建并注册实例
        app_core = AppCore()
        container.register_singleton(AppCore, app_core)
    else:
        # 如果获取到的是类而不是实例，创建新实例
        if isinstance(app_core, type):
            app_core = AppCore()
            container.register_singleton(AppCore, app_core)

    # 获取网格
    grid = app_core.grid_manager.init_grid(64, 64)

    # 设置示例向量场 - 创建旋转模式
    vector_calculator.create_tangential_pattern(grid, magnitude=1.0)

    # 初始化视图
    try:
        app_core.view_manager.reset_view(grid.shape[1], grid.shape[0])
    except Exception:
        pass

    # 初始化标记系统
    marker_system = MarkerSystem(app_core)

    # 初始化控制器
    controller = Controller(app_core, vector_calculator, marker_system, grid)

    # 初始化 UI 管理器（适配GUI）
    ui_manager = UIManager(app_core, None, controller, marker_system)  # window参数设为None

    # 注册回调函数
    ui_manager.register_callbacks(grid)

    # 设置定时器用于更新逻辑
    
    update_timer = QTimer()
    update_timer.timeout.connect(lambda: update_logic(grid, ui_manager, marker_system, vector_calculator))
    update_timer.start(16)  # ~60 FPS

    # 运行GUI应用
    print("[示例] 开始GUI主循环...")
    print("[示例] 按空格键重新生成切线模式，按G键切换网格显示，按C键清空网格")
    print("[示例] 按U键切换实时更新；用鼠标拖动视图并滚轮缩放")

    return app_core.run_gui()

def update_logic(grid, ui_manager, marker_system, vector_calculator):
    """更新逻辑（在GUI定时器中调用）"""
    # 清空网格
    grid.fill(0.0)

    # 处理鼠标拖动与滚轮（GUI中通过事件处理）
    try:
        ui_manager.process_mouse_drag()
    except Exception as e:
        print(f"[错误] 鼠标拖动处理异常: {e}")

    ui_manager.process_scroll()

    # 实时更新向量场（如果启用）
    if state_manager.get("enable_update" , True):
        # 创建边缘向内向量
        add_inward_edge_vectors(grid, magnitude=0.5)

        # 更新标记位置（可选）
        try:
            #给每个标记添加摩擦力
            for marker in marker_system.markers:
                marker['vx'] *= 0.99
                marker['vy'] *= 0.99
            # 更新向量场和标记
            marker_system.update_field_and_markers(grid)
        except Exception as e:
            print(f"[错误] 更新标记异常: {e}")

if __name__ == "__main__":
    main()
