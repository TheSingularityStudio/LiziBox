
"""
渲染器模块 - 提供向量场的渲染功能
"""
import numpy as np
import ctypes
from typing import Optional, Dict, Any, List, Tuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from core.config import config_manager
from core.events import EventBus, Event, EventType
from core.state import state_manager

class ShaderProgram:
    """着色器程序管理器"""
    def __init__(self, vertex_src: str, fragment_src: str):
        self._program = None
        self._uniform_locations = {}
        self._attribute_locations = {}
        self._vertex_src = vertex_src
        self._fragment_src = fragment_src

    def compile(self) -> None:
        """编译着色器程序"""
        try:
            # 编译顶点着色器
            vertex_shader = shaders.compileShader(self._vertex_src, GL_VERTEX_SHADER)

            # 编译片段着色器
            fragment_shader = shaders.compileShader(self._fragment_src, GL_FRAGMENT_SHADER)

            # 链接着色器程序
            self._program = shaders.compileProgram(vertex_shader, fragment_shader)

            print("[渲染器] 着色器程序编译成功")
        except Exception as e:
            print(f"[渲染器] 着色器编译错误: {e}")
            raise

    def use(self) -> None:
        """使用着色器程序"""
        if self._program is not None:
            glUseProgram(self._program)

    def get_uniform_location(self, name: str) -> int:
        """获取uniform变量位置"""
        if name not in self._uniform_locations:
            self._uniform_locations[name] = glGetUniformLocation(self._program, name)
        return self._uniform_locations[name]

    def get_attribute_location(self, name: str) -> int:
        """获取attribute变量位置"""
        if name not in self._attribute_locations:
            self._attribute_locations[name] = glGetAttribLocation(self._program, name)
        return self._attribute_locations[name]

    def set_uniform_float(self, name: str, value: float) -> None:
        """设置float类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform1f(loc, value)

    def set_uniform_vec2(self, name: str, value: Tuple[float, float]) -> None:
        """设置vec2类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform2f(loc, value[0], value[1])

    def set_uniform_vec3(self, name: str, value: Tuple[float, float, float]) -> None:
        """设置vec3类型uniform变量"""
        loc = self.get_uniform_location(name)
        if loc >= 0:
            glUniform3f(loc, value[0], value[1], value[2])

    def cleanup(self) -> None:
        """清理着色器程序"""
        if self._program is not None:
            glDeleteProgram(self._program)
            self._program = None
            self._uniform_locations.clear()
            self._attribute_locations.clear()

class VectorFieldRenderer:
    """向量场渲染器"""
    def __init__(self):
        self._event_bus = EventBus()
        self._state_manager = state_manager

        # 着色器源代码
        self._vertex_shader_src = """
#version 120
attribute vec2 a_pos;
attribute vec3 a_col;
varying vec3 v_col;
uniform vec2 u_center;
uniform vec2 u_half;
void main() {
    vec2 ndc = (a_pos - u_center) / u_half;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_col = a_col;
}
"""
        self._fragment_shader_src = """
#version 120
varying vec3 v_col;
void main() {
    gl_FragColor = vec4(v_col, 1.0);
}
"""

        # 着色器程序
        self._shader_program = ShaderProgram(self._vertex_shader_src, self._fragment_shader_src)

        # OpenGL 对象
        self._vao = None
        self._vbo = None
        self._grid_vao = None
        self._grid_vbo = None

        # 渲染状态
        self._initialized = False

    def initialize(self) -> None:
        """初始化渲染器"""
        if self._initialized:
            return

        try:
            # 编译着色器
            self._shader_program.compile()

            # 创建顶点数组对象和顶点缓冲对象
            self._vao = glGenVertexArrays(1)
            self._vbo = glGenBuffers(1)

            # 创建网格顶点数组对象和顶点缓冲对象
            self._grid_vao = glGenVertexArrays(1)
            self._grid_vbo = glGenBuffers(1)

            self._initialized = True
            print("[渲染器] 初始化成功")
        except Exception as e:
            print(f"[渲染器] 初始化失败: {e}")
            raise

    def render_vector_field(self, grid: np.ndarray, cell_size: float = 1.0, 
                           cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                           viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染向量场"""
        if not self._initialized:
            self.initialize()

        if grid is None:
            return

        # 获取配置
        vector_color = config_manager.get("vector_color", [0.2, 0.6, 1.0])

        # 准备顶点数据
        h, w = grid.shape[:2]
        vertices = []

        # 为每个向量创建线段顶点
        for y in range(h):
            for x in range(w):
                vx, vy = grid[y, x]
                if abs(vx) < 0.001 and abs(vy) < 0.001:
                    continue  # 跳过零向量

                # 起点
                px = x * cell_size
                py = y * cell_size
                vertices.extend([px, py, vector_color[0], vector_color[1], vector_color[2]])

                # 终点
                px2 = px + vx
                py2 = py + vy
                vertices.extend([px2, py2, vector_color[0], vector_color[1], vector_color[2]])

        # 如果没有向量，直接返回
        if not vertices:
            return

        # 绑定VAO和VBO
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # 上传顶点数据
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, np.array(vertices, dtype=np.float32), GL_DYNAMIC_DRAW)

        # 设置顶点属性
        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")

        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))

        # 使用着色器程序
        self._shader_program.use()

        # 设置uniform变量
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))

        # 绘制向量线
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)

        # 解绑
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def render_grid(self, grid: np.ndarray, cell_size: float = 1.0,
                   cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                   viewport_width: int = 800, viewport_height: int = 600) -> None:
        """渲染网格线"""
        if not self._initialized:
            self.initialize()

        if grid is None:
            return

        # 获取配置
        grid_color = config_manager.get("grid_color", [0.3, 0.3, 0.3])
        show_grid = self._state_manager.get("show_grid", True)

        if not show_grid:
            return

        h, w = grid.shape[:2]
        vertices = []

        # 水平线
        for y in range(h + 1):
            py = y * cell_size
            vertices.extend([0, py, grid_color[0], grid_color[1], grid_color[2]])
            vertices.extend([w * cell_size, py, grid_color[0], grid_color[1], grid_color[2]])

        # 垂直线
        for x in range(w + 1):
            px = x * cell_size
            vertices.extend([px, 0, grid_color[0], grid_color[1], grid_color[2]])
            vertices.extend([px, h * cell_size, grid_color[0], grid_color[1], grid_color[2]])

        # 绑定VAO和VBO
        glBindVertexArray(self._grid_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._grid_vbo)

        # 上传顶点数据
        glBufferData(GL_ARRAY_BUFFER, len(vertices) * 4, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)

        # 设置顶点属性
        pos_loc = self._shader_program.get_attribute_location("a_pos")
        col_loc = self._shader_program.get_attribute_location("a_col")

        if pos_loc >= 0:
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))

        if col_loc >= 0:
            glEnableVertexAttribArray(col_loc)
            glVertexAttribPointer(col_loc, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(2 * 4))

        # 使用着色器程序
        self._shader_program.use()

        # 设置uniform变量
        half_w = (viewport_width / 2.0) / cam_zoom
        half_h = (viewport_height / 2.0) / cam_zoom
        self._shader_program.set_uniform_vec2("u_center", (cam_x, cam_y))
        self._shader_program.set_uniform_vec2("u_half", (half_w, half_h))

        # 绘制网格线
        glDrawArrays(GL_LINES, 0, len(vertices) // 5)

        # 解绑
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glUseProgram(0)

    def render_background(self) -> None:
        """渲染背景"""
        # 获取配置
        bg_color = config_manager.get("background_color", [0.1, 0.1, 0.1])

        # 设置清屏颜色
        glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)

        # 清除颜色和深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def cleanup(self) -> None:
        """清理渲染器资源"""
        if not self._initialized:
            return

        try:
            # 删除着色器程序
            self._shader_program.cleanup()

            # 删除VAO和VBO
            if self._vao is not None:
                glDeleteVertexArrays(1, [self._vao])
                self._vao = None

            if self._vbo is not None:
                glDeleteBuffers(1, [self._vbo])
                self._vbo = None

            if self._grid_vao is not None:
                glDeleteVertexArrays(1, [self._grid_vao])
                self._grid_vao = None

            if self._grid_vbo is not None:
                glDeleteBuffers(1, [self._grid_vbo])
                self._grid_vbo = None

            self._initialized = False
            print("[渲染器] 资源清理完成")
        except Exception as e:
            print(f"[渲染器] 清理资源时出错: {e}")

# 全局向量场渲染器实例
vector_field_renderer = VectorFieldRenderer()

# 便捷函数
def render_vector_field(grid: np.ndarray, cell_size: float = 1.0, 
                       cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
                       viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染向量场"""
    vector_field_renderer.render_vector_field(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_grid(grid: np.ndarray, cell_size: float = 1.0,
               cam_x: float = 0.0, cam_y: float = 0.0, cam_zoom: float = 1.0,
               viewport_width: int = 800, viewport_height: int = 600) -> None:
    """便捷函数：渲染网格"""
    vector_field_renderer.render_grid(grid, cell_size, cam_x, cam_y, cam_zoom, viewport_width, viewport_height)

def render_background() -> None:
    """便捷函数：渲染背景"""
    vector_field_renderer.render_background()
