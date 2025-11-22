from LiziLib import GRID_640x480
import viewLib
import numpy as np
from OpenGL.GL import *
import ctypes
import glfw
from config import config

def draw_toolbar(window, cam_x, cam_y, cam_zoom, sx, sy, toolbar_vao, toolbar_vbo,
                 a_pos_loc, a_col_loc, stride, toolbar_enabled,
                 brush_size, magnitude, reverse_vector,
                 mouse_x=None, mouse_y=None, click=False, grid=None):
    """
    绘制精致的工具栏并返回工具栏按钮的状态
    """
    if not toolbar_enabled:
        return brush_size, magnitude, reverse_vector

    # 从配置文件读取工具栏参数
    toolbar_px_w = config.get("toolbar_width", 320)
    toolbar_px_h = config.get("toolbar_height", 40)
    toolbar_padding = config.get("toolbar_padding", 10)

    # 将工具栏像素尺寸转换为 world 单位（随 cam_zoom 缩放）
    half_w = (sx/2.0) / cam_zoom
    half_h = (sy/2.0) / cam_zoom
    world_left = cam_x - half_w
    world_top = cam_y - half_h
    x0 = world_left + (toolbar_padding) / cam_zoom
    y0 = world_top + (toolbar_padding) / cam_zoom
    x1 = x0 + (toolbar_px_w) / cam_zoom
    y1 = y0 + (toolbar_px_h) / cam_zoom

    # 处理点击事件
    if click and mouse_x is not None and mouse_y is not None:
        if x0 <= mouse_x <= x1 and y0 <= mouse_y <= y1:
            # 从配置文件读取按钮参数
            btn_size_px = config.get("button_size", 24)
            btn_gap_px = config.get("button_gap", 10)
            btn_start_px = config.get("button_start", 15)

            # 转换到 world 单位
            bx = lambda px: x0 + (px) / cam_zoom
            by = lambda px: y0 + (px) / cam_zoom

            # 检查每个按钮
            for i in range(7):  # 增加到7个按钮
                px0 = btn_start_px + i * (btn_size_px + btn_gap_px)
                px1 = px0 + btn_size_px
                ix0 = bx(px0)
                ix1 = bx(px1)
                iy0 = by(8)  # 垂直内边距
                iy1 = iy0 + (btn_size_px) / cam_zoom

                if ix0 <= mouse_x <= ix1 and iy0 <= mouse_y <= iy1:
                    # 第一个按钮：增加笔刷大小
                    if i == 0:
                        max_brush_size = config.get("max_brush_size", 15)
                        brush_size = min(brush_size + 1, max_brush_size)
                        print(f"笔刷大小增加到: {brush_size}")
                    # 第二个按钮：减小笔刷大小
                    elif i == 1:
                        min_brush_size = config.get("min_brush_size", 0)
                        brush_size = max(brush_size - 1, min_brush_size)
                        print(f"笔刷大小减小到: {brush_size}")
                    # 第三个按钮：设置向量模值（循环）
                    elif i == 2:
                        # 在0.25, 0.5, 1.0, 1.5之间循环
                        if magnitude < 0.5:
                            magnitude = 0.5
                        elif magnitude < 1.0:
                            magnitude = 1.0
                        elif magnitude < 1.5:
                            magnitude = 1.5
                        else:
                            magnitude = 0.25
                        print(f"向量模值设置为: {magnitude}")
                    # 第四个按钮：反转向量方向
                    elif i == 3:
                        reverse_vector = not reverse_vector
                        print(f"向量方向反转: {'开启' if reverse_vector else '关闭'}")
                    # 第五个按钮：清空网格
                    elif i == 4:
                        if grid is not None:
                            grid[:] = np.zeros_like(grid)
                            print("网格已清空")
                    # 第六个按钮：重置视图
                    elif i == 5:
                        # 通过窗口用户指针获取viewLib中的变量
                        import viewLib
                        # 获取viewLib中的相机参数
                        view_ptr = glfw.get_window_user_pointer(window)
                        if view_ptr:
                            view_data = view_ptr.value
                            view_data['cam_x'] = (grid.shape[1] * 1) / 2.0
                            view_data['cam_y'] = (grid.shape[0] * 1) / 2.0
                            view_data['cam_zoom'] = 1.0
                        print("视图已重置")
                    # 第七个按钮：切换网格显示
                    elif i == 6:
                        # 通过窗口用户指针获取viewLib中的变量
                        view_ptr = glfw.get_window_user_pointer(window)
                        if view_ptr:
                            view_data = view_ptr.value
                            view_data['show_grid'] = not view_data.get('show_grid', True)
                            print(f"网格显示: {'开启' if view_data['show_grid'] else '关闭'}")

                    # 返回更新后的值
                    return brush_size, magnitude, reverse_vector

    # 背景矩形（深色）和 7 个按钮（明显颜色）
    bg_color = (0.08, 0.08, 0.08)  # 稍微亮一点的背景
    border_color = (0.25, 0.25, 0.25)  # 更明显的边框

    # 按钮参数（像素）
    btn_count = 7  # 增加到7个按钮
    btn_size_px = 24
    btn_gap_px = 10
    # 按钮起始位置（从左向右）
    btn_start_px = 15

    # 转换到 world 单位
    bx = lambda px: x0 + (px) / cam_zoom
    by = lambda px: y0 + (px) / cam_zoom

    # 预分配顶点：背景 (6) + 边框 (6) + 每按钮 6 顶点 * 7
    verts_ui = np.empty((6 + 6 + btn_count * 6, 5), dtype=np.float32)

    # 背景 (6 顶点)
    verts_ui[0] = (x0, y0, *bg_color)
    verts_ui[1] = (x1, y0, *bg_color)
    verts_ui[2] = (x1, y1, *bg_color)
    verts_ui[3] = (x0, y0, *bg_color)
    verts_ui[4] = (x1, y1, *bg_color)
    verts_ui[5] = (x0, y1, *bg_color)

    # 边框 (6 顶点，稍大一点)
    border_thick = 2 / cam_zoom  # 边框厚度
    verts_ui[6] = (x0 - border_thick, y0 - border_thick, *border_color)
    verts_ui[7] = (x1 + border_thick, y0 - border_thick, *border_color)
    verts_ui[8] = (x1 + border_thick, y1 + border_thick, *border_color)
    verts_ui[9] = (x0 - border_thick, y0 - border_thick, *border_color)
    verts_ui[10] = (x1 + border_thick, y1 + border_thick, *border_color)
    verts_ui[11] = (x0 - border_thick, y1 + border_thick, *border_color)

    # 七个按钮（每个 6 顶点）
    # 从配置文件读取按钮颜色
    btn_colors = [
        config.get("button_increase_brush_color", (0.9, 0.3, 0.3)),    # 红色
        config.get("button_decrease_brush_color", (0.3, 0.9, 0.3)),    # 绿色
        config.get("button_magnitude_color", (0.3, 0.5, 0.95)),      # 蓝色
        config.get("button_reverse_vector_color", (0.95, 0.8, 0.2)), # 黄色
        config.get("button_clear_grid_color", (0.8, 0.3, 0.9)),     # 紫色
        config.get("button_reset_view_color", (0.3, 0.9, 0.9)),     # 青色
        config.get("button_toggle_grid_color", (0.95, 0.6, 0.2))    # 橙色
    ]

    for i in range(btn_count):
        px0 = btn_start_px + i * (btn_size_px + btn_gap_px)
        px1 = px0 + btn_size_px
        ix0 = bx(px0)
        iy0 = by(8)  # 垂直内边距
        ix1 = bx(px1)
        iy1 = iy0 + (btn_size_px) / cam_zoom
        base = 12 + i * 6
        c = btn_colors[i]
        verts_ui[base + 0] = (ix0, iy0, *c)
        verts_ui[base + 1] = (ix1, iy0, *c)
        verts_ui[base + 2] = (ix1, iy1, *c)
        verts_ui[base + 3] = (ix0, iy0, *c)
        verts_ui[base + 4] = (ix1, iy1, *c)
        verts_ui[base + 5] = (ix0, iy1, *c)

    verts_ui_flat = verts_ui.ravel()
    ui_nbytes = verts_ui_flat.nbytes

    # 上传到 ui_vbo
    glBindBuffer(GL_ARRAY_BUFFER, toolbar_vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, ui_nbytes, verts_ui_flat)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # 绘制工具栏
    glBindVertexArray(toolbar_vao)
    glBindBuffer(GL_ARRAY_BUFFER, toolbar_vbo)
    glEnableVertexAttribArray(a_pos_loc)
    glVertexAttribPointer(a_pos_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(a_col_loc)
    glVertexAttribPointer(a_col_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
    # 背景 + 边框 + 7 按钮 (6 + 6 + 6*7 = 54 顶点)
    glDrawArrays(GL_TRIANGLES, 0, 54)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # 返回当前值
    return brush_size, magnitude, reverse_vector

def main():
    # 初始化网格
    grid = GRID_640x480.copy()

    # 设置一些初始值作为示例
    # 在中心创建一个旋转模式
    h, w = grid.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 4

    for y in range(h):
        for x in range(w):
            # 计算到中心的距离和角度
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx**2 + dy**2)

            if dist < radius and dist > 0:
                # 创建切线向量（旋转）
                angle = np.arctan2(dy, dx) + np.pi/2
                magnitude = 1.0 - (dist / radius)  # 距离中心越远，向量越小
                grid[y, x] = (magnitude * np.cos(angle), magnitude * np.sin(angle))

    # 运行 OpenGL 视图，启用工具栏
    show_grid = config.get("show_grid", True)
    viewLib.run_opengl_view(grid, cell_size=1, show_grid=show_grid, toolbar_draw_func=draw_toolbar)

if __name__ == "__main__":
    main()
