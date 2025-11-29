"""
向量场计算模块 - 提供向量场计算的核心功能
"""
import numpy as np
from typing import Tuple, Union, List
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager

class VectorFieldCalculator:
    """向量场计算器"""
    def __init__(self):
        self._event_bus = EventBus()

    def sum_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
        """
        读取目标 (x,y) 的上下左右四个相邻格子的向量并相加（越界安全）。
        返回 (sum_x, sum_y) 的 tuple。
        """
        if include_self is None:
            include_self = config_manager.get("vector_field.include_self", True)

        h, w = grid.shape[:2]
        sum_x = sum_y = 0.0

        # 上
        if y > 0:
            sum_x += float(grid[y-1, x, 0])
            sum_y += float(grid[y-1, x, 1])
        # 下
        if y < h - 1:
            sum_x += float(grid[y+1, x, 0])
            sum_y += float(grid[y+1, x, 1])
        # 左
        if x > 0:
            sum_x += float(grid[y, x-1, 0])
            sum_y += float(grid[y, x-1, 1])
        # 右
        if x < w - 1:
            sum_x += float(grid[y, x+1, 0])
            sum_y += float(grid[y, x+1, 1])

        # 中心点自身
        if include_self:
            sum_x += float(grid[y, x, 0])
            sum_y += float(grid[y, x, 1])

        return sum_x, sum_y

    def average_adjacent_vectors(self, grid: np.ndarray, x: int, y: int, include_self: bool = None) -> Tuple[float, float]:
        """
        读取目标 (x,y) 的上下左右四个相邻格子的向量并求平均（越界安全）。
        返回 (avg_x, avg_y) 的 tuple。
        """
        sum_x, sum_y = self.sum_adjacent_vectors(grid, x, y, include_self)

        # 计算相邻格子数量
        h, w = grid.shape[:2]
        count = 0

        # 上
        if y > 0:
            count += 1
        # 下
        if y < h - 1:
            count += 1
        # 左
        if x > 0:
            count += 1
        # 右
        if x < w - 1:
            count += 1

        # 中心点自身
        if include_self is None:
            include_self = config_manager.get("vector_field.include_self", True)
        if include_self:
            count += 1

        if count > 0:
            return sum_x / count, sum_y / count
        else:
            return 0.0, 0.0

    def apply_vector_field(self, grid: np.ndarray, x: int, y: int, magnitude: float = None, brush_size: int = None) -> None:
        """
        在指定位置应用向量场
        """
        if magnitude is None:
            magnitude = config_manager.get("vector_field.default_vector_length", 1.0)
        if brush_size is None:
            brush_size = config_manager.get("vector_field.default_brush_size", 20)

        h, w = grid.shape[:2]
        reverse = config_manager.get("vector_field.reverse_vector", False)

        # 确保在网格范围内
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # 计算向量方向和强度
        avg_x, avg_y = self.average_adjacent_vectors(grid, x, y)
        avg_magnitude = np.sqrt(avg_x**2 + avg_y**2)

        if avg_magnitude > 0:
            # 归一化平均向量
            norm_x = avg_x / avg_magnitude
            norm_y = avg_y / avg_magnitude

            # 根据配置决定是否反转向量
            if reverse:
                norm_x = -norm_x
                norm_y = -norm_y

            # 应用向量场
            for dy in range(-brush_size, brush_size + 1):
                for dx in range(-brush_size, brush_size + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        # 计算距离权重
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= brush_size:
                            weight = 1.0 - (dist / brush_size)
                            grid[ny, nx, 0] = float(grid[ny, nx, 0]) * (1 - weight) + norm_x * magnitude * weight
                            grid[ny, nx, 1] = float(grid[ny, nx, 1]) * (1 - weight) + norm_y * magnitude * weight

    def create_tangential_pattern(self, grid: np.ndarray, magnitude: float = 0.2, radius_ratio: float = 0.3) -> None:
        """
        在网格上创建切线向量模式（围绕中心点的旋转模式）
        
        参数:
            grid: 向量网格
            magnitude: 向量强度（默认值减小为0.2）
            radius_ratio: 切线模式的半径比例，相对于网格最小边长的比例（默认0.3）
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return
            
        h, w = grid.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 计算切线模式的实际半径
        min_dimension = min(h, w)
        pattern_radius = min_dimension * radius_ratio
        
        for y in range(h):
            for x in range(w):
                # 计算从中心到当前点的向量
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                
                # 只在指定半径内创建切线向量
                if 0 < dist <= pattern_radius:
                    # 计算切线方向（垂直于径向）
                    # 切线方向可以通过交换x和y并取反一个分量得到
                    
                    # 根据距离调整向量强度，距离中心越远，向量越小
                    distance_factor = 1.0 - (dist / pattern_radius) * 0.5  # 距离中心最远处的向量强度为50%
                    adjusted_magnitude = magnitude * distance_factor
                    
                    tangent_x = -dy / dist * adjusted_magnitude
                    tangent_y = dx / dist * adjusted_magnitude
                    
                    # 设置向量
                    grid[y, x, 0] = tangent_x
                    grid[y, x, 1] = tangent_y
                else:
                    # 中心点和半径外的点没有切线方向，设为0
                    grid[y, x, 0] = 0
                    grid[y, x, 1] = 0

    def correct_vector_centers(self, grid: np.ndarray, threshold: float = 0.5, min_distance: int = 10) -> List[Tuple[int, int]]:
        """
        获取已记录的向量场中心点，并在每帧不断修正其位置

        参数:
            grid: 向量网格
            threshold: 用于判断是否更新中心点的向量强度阈值
            min_distance: 中心点之间的最小距离，避免重复识别

        返回:
            修正后的中心点坐标列表 [(x, y), ...] - 支持多个中心点
        """
        if grid is None or not isinstance(grid, np.ndarray):
            return []

        # 获取已记录的向量场中心点列表
        centers = state_manager.get("vector_field_centers", [])

        # 如果没有记录的中心点，返回空列表
        if not centers:
            return []

        # 确保在网格范围内
        h, w = grid.shape[:2]
        valid_centers = []
        updated_centers = []

        # 获取自动修正开关
        auto_correct = config_manager.get("vector_field.auto_correct_centers", True)

        for center in centers:
            if len(center) >= 2:  # 确保中心点有x,y坐标
                center_x, center_y = center[0], center[1]

                # 确保在网格范围内
                if 0 <= center_x < w and 0 <= center_y < h:
                    # 如果启用自动修正，则根据向量场调整中心点位置
                    if auto_correct:
                        # 获取搜索范围（基于配置）
                        search_radius = config_manager.get("vector_field.center_search_radius", 10)

                        # 确保搜索范围不超出网格边界
                        min_x = max(0, center_x - search_radius)
                        max_x = min(w - 1, center_x + search_radius)
                        min_y = max(0, center_y - search_radius)
                        max_y = min(h - 1, center_y + search_radius)

                        # 使用加权平均方法计算新的中心点位置
                        total_weight = 0.0
                        weighted_x = 0.0
                        weighted_y = 0.0

                        # 设置最大向量强度阈值，防止过大值影响修正
                        max_magnitude = 5.0

                        for y in range(min_y, max_y + 1):
                            for x in range(min_x, max_x + 1):
                                # 计算到原始中心的距离
                                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                                # 计算当前位置的向量
                                vec_x = float(grid[y, x, 0])
                                vec_y = float(grid[y, x, 1])
                                magnitude = np.sqrt(vec_x**2 + vec_y**2)

                                # 对向量强度进行限制，防止过大值影响修正
                                magnitude = min(magnitude, max_magnitude)

                                # 计算权重：向量强度越高，距离原始中心越近，权重越大
                                # 使用高斯函数计算距离权重，使中心附近的点权重更大
                                distance_weight = np.exp(-(dist**2) / (2 * (search_radius/2)**2))
                                weight = magnitude * distance_weight

                                # 累加权重和加权位置
                                total_weight += weight
                                weighted_x += x * weight
                                weighted_y += y * weight

                        # 计算加权平均位置
                        if total_weight > 0:
                            best_x = int(round(weighted_x / total_weight))
                            best_y = int(round(weighted_y / total_weight))

                            # 确保结果在搜索范围内
                            best_x = max(min_x, min(max_x, best_x))
                            best_y = max(min_y, min(max_y, best_y))

                            # 计算位置变化
                            position_change = np.sqrt((best_x - center_x)**2 + (best_y - center_y)**2)

                            # 如果位置变化足够大，则更新中心点
                            if (best_x != center_x or best_y != center_y) and position_change > 0.5:
                                # 更新中心点位置
                                center_x, center_y = best_x, best_y
                                updated_centers.append([center_x, center_y])

                    # 添加到有效中心点列表
                    valid_centers.append((center_x, center_y))

        # 如果有中心点位置被更新，则保存更新后的位置
        if updated_centers:
            state_manager.set("vector_field_centers", valid_centers)

        return valid_centers

# 创建全局向量场计算器实例
vector_calculator = VectorFieldCalculator()

def correct_vector_centers(grid: np.ndarray, threshold: float = 0.5, min_distance: int = 10) -> List[Tuple[int, int]]:
    """便捷函数：修正向量中心点位置"""
    return vector_calculator.correct_vector_centers(grid, threshold, min_distance)
