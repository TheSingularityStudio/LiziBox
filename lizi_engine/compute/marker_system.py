
"""
标记系统插件：管理向量场中的标记点
提供标记点的创建、更新和渲染功能。
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from lizi_engine.compute.vector_field import vector_calculator

class MarkerSystem:
    """标记系统，用于管理向量场中的标记点"""

    def __init__(self, app_core):
        self.app_core = app_core
        self.vector_calculator = vector_calculator
        # 标记列表，存储浮点网格坐标 {'id':int, 'x':float,'y':float,'mag':float,'vx':float,'vy':float}
        self.markers = []
        self.marker_id_counter = 0
        # 弹簧连接列表 [{'id1':int, 'id2':int, 'rest_length':float, 'strength':float, 'damping':float}]
        self.springs = []

    def add_marker(self, x: float, y: float, mag: float = 1.0, vx: float = 0.0, vy: float = 0.0) -> None:
        """添加一个新标记

        Args:
            x: 标记的x坐标（浮点）
            y: 标记的y坐标（浮点）
            mag: 标记的初始幅值（可选）
            vx: 标记的x方向速度（可选）
            vy: 标记的y方向速度（可选）
        """
        marker = {"id": self.marker_id_counter, "x": float(x), "y": float(y), "mag": float(mag), "vx": float(vx), "vy": float(vy)}
        self.markers.append(marker)
        self.marker_id_counter += 1
        self._sync_to_state_manager()

    def clear_markers(self) -> None:
        """清除所有标记"""
        self.markers = []
        self.springs = []
        self._sync_to_state_manager()

    def connect_spring(self, id1: int, id2: int, rest_length: float = None, strength: float = 0.01, damping: float = 0.1) -> None:
        """连接两个标记为弹簧

        Args:
            id1: 第一个标记的ID
            id2: 第二个标记的ID
            rest_length: 弹簧的自然长度（可选，默认使用当前距离）
            strength: 弹簧强度（可选，默认1.0）
            damping: 弹簧阻尼系数（可选，默认0.1）
        """
        # 检查标记是否存在
        marker1 = next((m for m in self.markers if m["id"] == id1), None)
        marker2 = next((m for m in self.markers if m["id"] == id2), None)
        if marker1 is None or marker2 is None:
            print(f"Error: Marker with id {id1} or {id2} not found")
            return

        # 如果未提供rest_length，使用当前距离
        if rest_length is None:
            dx = marker1["x"] - marker2["x"]
            dy = marker1["y"] - marker2["y"]
            rest_length = (dx**2 + dy**2)**0.5

        # 检查是否已存在连接
        existing = next((s for s in self.springs if (s["id1"] == id1 and s["id2"] == id2) or (s["id1"] == id2 and s["id2"] == id1)), None)
        if existing:
            print(f"Spring connection between {id1} and {id2} already exists")
            return

        spring = {"id1": id1, "id2": id2, "rest_length": rest_length, "strength": strength, "damping": damping}
        self.springs.append(spring)
        print(f"Spring connected between marker {id1} and {id2}")
        self._sync_to_state_manager()

    def get_markers(self) -> List[Dict[str, float]]:
        """获取所有标记

        Returns:
            标记列表
        """
        return list(self.markers)

    def update_markers(self, grid: np.ndarray, dt: float = 1.0, clear_threshold: float = 1e-3) -> None:
        """根据浮点坐标处拟合向量移动标记。

        算法：在标记的浮点坐标处使用双线性插值拟合向量值，将标记按 fitted_v * dt 偏移。

        Args:
            grid: 向量场网格
            dt: 时间步长
            clear_threshold: 清除阈值，低于此拟合向量幅值的标记将被清除
        """
        if not hasattr(grid, "ndim"):
            return

        # 优先从全局状态同步标记（如果其他模块在放置向量场时添加了标记）
        try:
            stored = self.app_core.state_manager.get("markers", None)
            if stored is not None:
                self.markers = list(stored)
        except Exception:
            pass

        # 检查网格维度是否有效
        if grid.ndim < 3 or grid.shape[2] < 2:
            return

        h, w = grid.shape[0], grid.shape[1]
        cell_size = self.app_core.config_manager.get("cell_size", 1.0)

        # 期望 grid 最后一维至少 2，代表 vx, vy
        new_markers = []

        for m in self.markers:
            x = m["x"]
            y = m["y"]
            mag = m["mag"]
            vx = m["vx"]
            vy = m["vy"]
            try:
                # 在浮点坐标处拟合向量值
                fitted_vx, fitted_vy = self.fit_vector_at_position(grid, x, y)

                # 设置标记的速度属性
                if fitted_vx ** 2 + fitted_vy ** 2 > 0.001 ** 2:
                    vx += fitted_vx * 1/mag
                    vy += fitted_vy * 1/mag

                # 应用弹簧向量到网格
                self._apply_spring_vectors_to_grid(grid, m["id"], x, y)

                # 应用重力向量到网格（如果启用）
                if self.app_core.state_manager.get("gravity_enabled", False):
                    self.add_vector_at_position(grid, m["x"], m["y"], 0.0, 0.1)

                # 限制速度不超过单元格大小
                if (vx ** 2 + vy ** 2) ** 0.5 > cell_size:  # 限制速度不超过单元格大小
                    vx = vx / (vx ** 2 + vy ** 2) ** 0.5 * cell_size
                    vy = vy / (vx ** 2 + vy ** 2) ** 0.5 * cell_size

                # 使用速度更新浮点位置（带反弹后的速度）
                new_x = max(0.0, min(w - 1.0, x + vx * dt))
                new_y = max(0.0, min(h - 1.0, y + vy * dt))

                # 创建微小向量影响
                self.create_tiny_vector(grid, new_x, new_y, mag)

                m["x"] = new_x
                m["y"] = new_y
                # 应用摩擦力
                m["vx"] = vx * 0.98
                m["vy"] = vy * 0.98
                new_markers.append(m)

            except Exception as e:
                # 添加更详细的错误日志
                print(f"Error updating marker at ({x}, {y}): {str(e)}")
                # 保留标记以便后续检查
                new_markers.append(m)
                continue

        # 更新内部标记列表并写回 state_manager 以便界面绘制或外部使用
        self.markers = new_markers
        self._sync_to_state_manager()

    def create_tiny_vector(self, grid: np.ndarray, x: float, y: float, mag: float = 1.0) -> None:
        # 在指定位置创建一个微小的向量场影响,只影响位置本身及上下左右四个邻居
        self.vector_calculator.create_tiny_vector(grid, x, y, mag)

    def add_vector_at_position(self, grid: np.ndarray, x: float, y: float, vx: float, vy: float) -> None:
        # 在指定位置添加一个向量
        self.vector_calculator.add_vector_at_position(grid, x, y, vx, vy)
        

    def fit_vector_at_position(self, grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        # 在指定位置拟合一个向量
        return self.vector_calculator.fit_vector_at_position(grid, x, y)
    
    def update_field_and_markers(self, grid: np.ndarray) -> None:
        # 更新向量场和标记
        self.update_markers(grid)
        self.vector_calculator.update_grid_with_adjacent_sum(grid)
        # 再次更新标记
        self.update_markers(grid)

        # 将更新后的网格设置回GridManager，确保及时通知渲染器
        self.app_core.grid_manager.set_grid(grid)

    def _apply_spring_vectors_to_grid(self, grid: np.ndarray, marker_id: int, x: float, y: float) -> None:
        """在标记位置添加弹簧向量到网格"""
        current_marker = next((m for m in self.markers if m["id"] == marker_id), None)
        if current_marker is None:
            return

        for spring in self.springs:
            if spring["id1"] == marker_id or spring["id2"] == marker_id:
                # 找到另一个标记
                other_id = spring["id2"] if spring["id1"] == marker_id else spring["id1"]
                other_marker = next((m for m in self.markers if m["id"] == other_id), None)
                if other_marker is None:
                    continue

                # 计算距离和方向
                dx = other_marker["x"] - x
                dy = other_marker["y"] - y
                distance = (dx**2 + dy**2)**0.5

                if distance == 0:
                    continue  # 避免除零

                # 归一化方向向量
                nx = dx / distance
                ny = dy / distance

                # 计算弹簧力（胡克定律）
                rest_length = spring["rest_length"]
                strength = spring["strength"]
                spring_force = strength * (distance - rest_length)

                # 总力
                total_force_x = spring_force * nx
                total_force_y = spring_force * ny

                # 在当前标记位置添加力向量
                self.add_vector_at_position(grid, x, y, total_force_x, total_force_y)
                # 在另一个标记位置添加相反的力向量
                self.add_vector_at_position(grid, other_marker["x"], other_marker["y"], -total_force_x, -total_force_y)

    def _sync_to_state_manager(self) -> None:
        """将标记列表和弹簧连接同步到状态管理器"""
        try:
            self.app_core.state_manager.set("markers", list(self.markers))
            self.app_core.state_manager.set("springs", list(self.springs))
        except Exception:
            pass

