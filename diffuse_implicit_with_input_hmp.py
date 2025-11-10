
import taichi as ti
import mmap
import struct
import numpy as np
import time
import threading
from threading import Lock
import math

ti.init(arch=ti.cpu)

# control
paused = False
save_images = False
show_gradient_lines = True
show_isotherms = True

# problem setting
n_x = 78
n_y = 52
scatter = 8
res_x = n_x * scatter  # 624
res_y = n_y * scatter  # 416

# 共享内存参数
SHARED_MEMORY_NAME = "shared_touch_image"
SHARED_MEMORY_SIZE = 4068
INPUT_WIDTH = 78
INPUT_HEIGHT = 52
INPUT_FRAME_SIZE = INPUT_WIDTH * INPUT_HEIGHT

# physical parameters
h = 2e-3
substep = 1
dx = 1
t_max = 300
t_min = 0
k = 850.0
t_ambient = 20
cooling_rate = 0.005

# 实时输入参数
input_update_rate = 30
input_scale_factor = 0.8
input_threshold = 160
heat_intensity_scale = 1.0

# 梯度线参数
gradient_line_length = 40
gradient_line_min_magnitude = 5
gradient_line_width = 1

# 等温线参数
isotherm_levels = 10  # 等温线数量
isotherm_color = 0.2  # 等温线颜色（灰度值）
isotherm_thickness = 1  # 等温线粗细

# 插值参数
use_interpolation = True

# visualization
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

# 梯度场存储
gradient_magnitude = ti.field(ti.f32, shape=(n_y, n_x))
gradient_x = ti.field(ti.f32, shape=(n_y, n_x))
gradient_y = ti.field(ti.f32, shape=(n_y, n_x))

# diffuse matrix
n_total = n_x * n_y
D_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total*5)
I_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total)

# 预计算变量
precomputed_solver = None
precomputed_matrix = None

# temperature fields
t_n = ti.field(ti.f32, shape=n_total)
t_np1 = ti.field(ti.f32, shape=n_total)

# 实时输入数据存储
input_data = ti.field(ti.f32, shape=(INPUT_HEIGHT, INPUT_WIDTH))
processed_input = ti.field(ti.f32, shape=(n_y, n_x))

# 线程同步
data_lock = Lock()
new_data_available = False
last_update_time = 0

@ti.func
def ind(i, j):
    """将2D坐标转换为1D索引"""
    return i * n_x + j

@ti.kernel
def fillDiffusionMatrixBuilder(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(n_y, n_x):
        count = 0
        if i-1 >= 0:
            A[ind(i,j), ind(i-1,j)] += 1
            count += 1
        if i+1 < n_y:
            A[ind(i,j), ind(i+1,j)] += 1
            count += 1
        if j-1 >= 0:
            A[ind(i,j), ind(i,j-1)] += 1
            count += 1
        if j+1 < n_x:
            A[ind(i,j), ind(i,j+1)] += 1
            count += 1
        A[ind(i,j), ind(i,j)] += -count

@ti.kernel
def fillEyeMatrixBuilder(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(n_y, n_x):
        A[ind(i,j), ind(i,j)] += 1

def buildMatrices():
    global precomputed_solver, precomputed_matrix
    
    fillDiffusionMatrixBuilder(D_builder)
    fillEyeMatrixBuilder(I_builder)
    D = D_builder.build()
    I = I_builder.build()
    
    c = h * k / dx**2
    ImcD = I - c * D 
    
    precomputed_solver = ti.linalg.SparseSolver(solver_type="LLT")
    precomputed_solver.analyze_pattern(ImcD)
    precomputed_solver.factorize(ImcD)
    precomputed_matrix = ImcD
    
    print(f"矩阵预计算完成 - 网格大小: {n_x}×{n_y}, 系数c: {c:.6f}")
    
    return D, I

@ti.kernel
def init():
    for i, j in ti.ndrange(n_y, n_x):
        t_n[ind(i, j)] = t_min
        t_np1[ind(i, j)] = t_min
        processed_input[i, j] = 0.0
        gradient_magnitude[i, j] = 0.0
        gradient_x[i, j] = 0.0
        gradient_y[i, j] = 0.0

@ti.kernel
def compute_gradients():
    """计算温度梯度"""
    for i, j in ti.ndrange(n_y, n_x):
        gx = 0.0
        if j > 0 and j < n_x - 1:
            gx = (t_np1[ind(i, j+1)] - t_np1[ind(i, j-1)]) / 2.0
        elif j == 0:
            gx = t_np1[ind(i, j+1)] - t_np1[ind(i, j)]
        else:
            gx = t_np1[ind(i, j)] - t_np1[ind(i, j-1)]
        
        gy = 0.0
        if i > 0 and i < n_y - 1:
            gy = (t_np1[ind(i+1, j)] - t_np1[ind(i-1, j)]) / 2.0
        elif i == 0:
            gy = t_np1[ind(i+1, j)] - t_np1[ind(i, j)]
        else:
            gy = t_np1[ind(i, j)] - t_np1[ind(i-1, j)]
        
        gradient_x[i, j] = gx
        gradient_y[i, j] = gy
        gradient_magnitude[i, j] = ti.sqrt(gx*gx + gy*gy)

@ti.kernel
def process_input_data_kernel(threshold: ti.f32, intensity_scale: ti.f32):
    """处理输入数据"""
    for i, j in ti.ndrange(n_y, n_x):
        processed_input[i, j] = 0.0
    
    for i, j in ti.ndrange(n_y, n_x):
        flipped_i = n_y - 1 - i
        input_value = input_data[flipped_i, j]
        
        if input_value < threshold:
            processed_input[i, j] = (threshold - input_value) * intensity_scale

@ti.kernel
def update_temperature_with_input_and_cooling_kernel(ambient_temp: ti.f32, cooling: ti.f32):
    """更新温度场"""
    for i, j in ti.ndrange(n_y, n_x):
        current_temp = t_np1[ind(i, j)]
        cooled_temp = current_temp + (ambient_temp - current_temp) * cooling
        final_temp = cooled_temp
        
        heat_input = processed_input[i, j]
        if heat_input > 0:
            target_temp = 225 + heat_input * (t_max - t_min)
            calculated_temp = cooled_temp * 0.7 + target_temp * 0.3
            final_temp = ti.max(calculated_temp, t_ambient)
        else:
            final_temp = cooled_temp
        
        t_np1[ind(i, j)] = final_temp

def process_input_data(threshold: ti.f32, intensity_scale: ti.f32):
    process_input_data_kernel(threshold, intensity_scale)

def update_temperature_with_input_and_cooling(ambient_temp: ti.f32, cooling: ti.f32):
    update_temperature_with_input_and_cooling_kernel(ambient_temp, cooling)

@ti.func
def get_color(v, vmin, vmax):
    c = ti.Vector([1.0, 1.0, 1.0])
    
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    
    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v-vmin) / dv
    elif v < (vmin + 0.5 * dv):
        c[0] = 0
        c[2] = 1 + 4 * (vmin + 0.25*dv -v) / dv
    elif v < (vmin + 0.75*dv):
        c[0] = 4 * (v - vmin -0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0
    
    return c

@ti.func
def get_temperature_clamped(i, j):
    clamped_i = ti.max(0, ti.min(n_y - 1, i))
    clamped_j = ti.max(0, ti.min(n_x - 1, j))
    return t_np1[ind(clamped_i, clamped_j)]

@ti.kernel 
def temperature_to_color_original(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        for k, l in ti.ndrange(scatter, scatter):
            color[j*scatter+l, i*scatter+k] = get_color(t[ind(i,j)], tmin, tmax)

@ti.kernel
def temperature_to_color_bilinear_simple(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    for pixel_x, pixel_y in color:
        grid_j = pixel_x / scatter
        grid_i = pixel_y / scatter
        
        i0 = int(grid_i)
        j0 = int(grid_j)
        i1 = i0 + 1
        j1 = j0 + 1
        
        fx = grid_j - j0
        fy = grid_i - i0
        
        t00 = get_temperature_clamped(i0, j0)
        t01 = get_temperature_clamped(i0, j1)
        t10 = get_temperature_clamped(i1, j0)
        t11 = get_temperature_clamped(i1, j1)
        
        temp_interpolated = (1-fx) * (1-fy) * t00 + fx * (1-fy) * t01 + (1-fx) * fy * t10 + fx * fy * t11
        
        color[pixel_x, pixel_y] = get_color(temp_interpolated, tmin, tmax)

def draw_line_bresenham(image_array, x0, y0, x1, y1, r, g, b):
    """Bresenham 线条算法"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        if 0 <= x < image_array.shape[0] and 0 <= y < image_array.shape[1]:
            image_array[x, y, 0] = image_array[x, y, 0] * 0.7 + r * 0.3
            image_array[x, y, 1] = image_array[x, y, 1] * 0.7 + g * 0.3
            image_array[x, y, 2] = image_array[x, y, 2] * 0.7 + b * 0.3
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def draw_gradient_streamlines(line_length, min_magnitude, line_color):
    """
    绘制梯度流线（参考 Euler Fluid 的 streamlines 算法）
    从稀疏的网格点出发，沿着梯度方向迭代追踪，绘制连续曲线
    """
    grad_mag_np = gradient_magnitude.to_numpy()
    grad_x_np = gradient_x.to_numpy()
    grad_y_np = gradient_y.to_numpy()
    
    pixels_np = pixels.to_numpy()
    
    seg_len = scatter * 0.2
    num_segs = 20
    step_size = 3
    
    for i in range(1, n_y - 1, step_size):
        for j in range(1, n_x - 1, step_size):
            mag = grad_mag_np[i, j]
            
            if mag > min_magnitude:
                x = (j + 0.5) * scatter
                y = (i + 0.5) * scatter
                
                line_points = [(x, y)]
                
                # 正向追踪
                for seg in range(num_segs):
                    xi = int(x / scatter)
                    yi = int(y / scatter)
                    
                    xi = max(0, min(n_x - 1, xi))
                    yi = max(0, min(n_y - 1, yi))
                    
                    gx = grad_x_np[yi, xi]
                    gy = grad_y_np[yi, xi]
                    mag_local = np.sqrt(gx*gx + gy*gy)
                    
                    if mag_local > min_magnitude:
                        inv_mag_local = 1.0 / (mag_local + 1e-6)
                        ux = gx * inv_mag_local
                        uy = gy * inv_mag_local
                        
                        x += ux * seg_len
                        y += uy * seg_len
                    else:
                        break
                    
                    line_points.append((x, y))
                    
                    if x < 0 or x >= res_x or y < 0 or y >= res_y:
                        break
                
                # 反向追踪
                x = (j + 0.5) * scatter
                y = (i + 0.5) * scatter
                
                for seg in range(num_segs):
                    xi = int(x / scatter)
                    yi = int(y / scatter)
                    
                    xi = max(0, min(n_x - 1, xi))
                    yi = max(0, min(n_y - 1, yi))
                    
                    gx = grad_x_np[yi, xi]
                    gy = grad_y_np[yi, xi]
                    mag_local = np.sqrt(gx*gx + gy*gy)
                    
                    if mag_local > min_magnitude:
                        inv_mag_local = 1.0 / (mag_local + 1e-6)
                        ux = gx * inv_mag_local
                        uy = gy * inv_mag_local
                        
                        x -= ux * seg_len
                        y -= uy * seg_len
                    else:
                        break
                    
                    line_points.insert(0, (x, y))
                    
                    if x < 0 or x >= res_x or y < 0 or y >= res_y:
                        break
                
                if len(line_points) > 1:
                    for pt_idx in range(len(line_points) - 1):
                        x0, y0 = line_points[pt_idx]
                        x1, y1 = line_points[pt_idx + 1]
                        draw_line_bresenham(pixels_np, int(x0), int(y0), int(x1), int(y1),
                                          line_color, line_color, line_color)
    
    pixels.from_numpy(pixels_np)


def linear_interp(x, x0, x1, y0, y1):
    """线性插值"""
    if abs(x1 - x0) < 1e-10:
        return 0.5
    return (x - x0) / (x1 - x0)


def draw_isotherms(num_levels, line_color):
    """
    使用 Marching Squares 算法绘制等温线
    """
    # 获取温度场数据 - 直接转换为numpy数组并reshape
    t_np = t_np1.to_numpy().reshape(n_y, n_x)
    
    pixels_np = pixels.to_numpy()
    
    # 计算温度范围
    temp_min = np.min(t_np)
    temp_max = np.max(t_np)
    temp_range = temp_max - temp_min
    
    # 如果温度范围太小，不绘制
    if temp_range < 1.0:
        return
    
    # 生成等温线的温度值
    iso_temps = []
    for level in range(1, num_levels + 1):
        iso_temp = temp_min + (temp_range * level) / (num_levels + 1)
        iso_temps.append(iso_temp)
    
    # 对每个等温线温度值使用 marching squares
    for iso_temp in iso_temps:
        # 遍历每个网格单元
        for i in range(n_y - 1):
            for j in range(n_x - 1):
                # 获取单元四个角的温度值
                # v0: 左下 (i, j)
                # v1: 右下 (i, j+1)
                # v2: 右上 (i+1, j+1)
                # v3: 左上 (i+1, j)
                v0 = t_np[i, j]
                v1 = t_np[i, j+1]
                v2 = t_np[i+1, j+1]
                v3 = t_np[i+1, j]
                
                # 计算 marching squares 的 case index
                case = 0
                if v0 > iso_temp:
                    case |= 1
                if v1 > iso_temp:
                    case |= 2
                if v2 > iso_temp:
                    case |= 4
                if v3 > iso_temp:
                    case |= 8
                
                # 根据 case 绘制线段
                # 计算单元格的像素坐标
                x0 = j * scatter
                y0 = i * scatter
                x1 = (j + 1) * scatter
                y1 = (i + 1) * scatter
                
                # 计算边的中点（使用线性插值）
                # 边的顺序：bottom, right, top, left
                edges = []
                
                # Bottom edge (v0 to v1)
                if (v0 <= iso_temp < v1) or (v1 <= iso_temp < v0):
                    t = linear_interp(iso_temp, v0, v1, 0, 1)
                    edges.append((x0 + t * scatter, y0))
                
                # Right edge (v1 to v2)
                if (v1 <= iso_temp < v2) or (v2 <= iso_temp < v1):
                    t = linear_interp(iso_temp, v1, v2, 0, 1)
                    edges.append((x1, y0 + t * scatter))
                
                # Top edge (v3 to v2)
                if (v3 <= iso_temp < v2) or (v2 <= iso_temp < v3):
                    t = linear_interp(iso_temp, v3, v2, 0, 1)
                    edges.append((x0 + t * scatter, y1))
                
                # Left edge (v0 to v3)
                if (v0 <= iso_temp < v3) or (v3 <= iso_temp < v0):
                    t = linear_interp(iso_temp, v0, v3, 0, 1)
                    edges.append((x0, y0 + t * scatter))
                
                # 绘制线段（根据 case 连接边）
                if case == 1 or case == 14:  # bottom-left
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 2 or case == 13:  # bottom-right
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 3 or case == 12:  # bottom
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 4 or case == 11:  # top-right
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 5:  # ambiguous case
                    if len(edges) >= 4:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                        draw_line_bresenham(pixels_np, int(edges[2][0]), int(edges[2][1]),
                                          int(edges[3][0]), int(edges[3][1]), line_color, line_color, line_color)
                elif case == 6 or case == 9:  # right
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 7 or case == 8:  # top-left
                    if len(edges) >= 2:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 10:  # ambiguous case
                    if len(edges) >= 4:
                        draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                          int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                        draw_line_bresenham(pixels_np, int(edges[2][0]), int(edges[2][1]),
                                          int(edges[3][0]), int(edges[3][1]), line_color, line_color, line_color)
    
    pixels.from_numpy(pixels_np)


def diffuse():
    global precomputed_solver
    if precomputed_solver is None:
        print("错误：求解器未预计算！")
        return
    
    t_np1.from_numpy(precomputed_solver.solve(t_n))

def update_and_commit():
    t_n.copy_from(t_np1)

def read_shared_memory():
    """从共享内存读取数据的线程函数"""
    global new_data_available, last_update_time
    
    HEADER_SIZE = 12
    
    mmf = None
    try:
        import mmap
        import os
        
        if os.name == 'nt':  # Windows
            try:
                mmf = mmap.mmap(0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
                print("成功连接到共享内存")
            except Exception as e:
                print(f"无法打开Windows共享内存: {e}")
                return
        else:  # Linux/Mac
            try:
                temp_file = f"/tmp/{SHARED_MEMORY_NAME}"
                if not os.path.exists(temp_file):
                    print(f"共享内存文件不存在: {temp_file}")
                    return
                
                with open(temp_file, 'r+b') as f:
                    mmf = mmap.mmap(f.fileno(), SHARED_MEMORY_SIZE, access=mmap.ACCESS_READ)
                print("成功连接到共享内存")
            except Exception as e:
                print(f"无法打开Unix共享内存: {e}")
                return
        
        while True:
            try:
                mmf.seek(HEADER_SIZE)
                raw_data = mmf.read(INPUT_FRAME_SIZE)
                
                if len(raw_data) >= INPUT_FRAME_SIZE:
                    np_data = np.frombuffer(raw_data[:INPUT_FRAME_SIZE], dtype=np.uint8).copy()
                    np_data = np_data.reshape((INPUT_HEIGHT, INPUT_WIDTH))
                    normalized_data = np_data.astype(np.float32) / 255.0
                    
                    # 在data_lock外部准备好数据，然后快速更新
                    with data_lock:
                        try:
                            input_data.from_numpy(normalized_data)
                            new_data_available = True
                            last_update_time = time.time()
                        except Exception as e:
                            # 静默处理错误，避免日志刷屏
                            pass
                
                time.sleep(1.0 / input_update_rate)
                
            except Exception as e:
                # 静默处理读取错误
                time.sleep(0.1)
                
    except Exception as e:
        print(f"共享内存初始化失败: {e}")
    finally:
        if mmf:
            mmf.close()

my_gui = ti.GUI("Heat Diffusion with Isotherms & Streamlines", (res_x, res_y))

init()
D, I = buildMatrices()

memory_thread = threading.Thread(target=read_shared_memory, daemon=True)
memory_thread.start()

i = 0
frame_count = 0
last_fps_time = time.time()

print("控制说明:")
print("- 空格键: 暂停/继续")
print("- R键: 重置")
print("- I键: 开启/关闭图像保存")
print("- S键: 切换插值模式")
print("- G键: 开启/关闭梯度流线显示")
print("- T键: 开启/关闭等温线显示")
print("- +/-键: 调整输入强度")
print("- [/]键: 调整等温线数量")
print("- 上/下方向键: 调整阈值")
print("- ESC: 退出")
print(f"仿真网格: {n_x}×{n_y}, 显示分辨率: {res_x}×{res_y}")
print(f"梯度流线显示: {'开启' if show_gradient_lines else '关闭'}")
print(f"等温线显示: {'开启' if show_isotherms else '关闭'}, 数量: {isotherm_levels}")

while my_gui.running:
    current_time = time.time()
    
    for e in my_gui.get_events():
        if e.type == ti.GUI.PRESS:
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print(f"仿真 {'暂停' if paused else '继续'}")
            elif e.key == 'i':
                save_images = not save_images
                print(f"图像保存: {'开启' if save_images else '关闭'}")
            elif e.key == 'r':
                init()
                i = 0
                print("重置仿真")
            elif e.key == 's':
                use_interpolation = not use_interpolation
                print(f"插值模式: {'简化双线性插值' if use_interpolation else '无插值(马赛克)'}")
            elif e.key == 'g':
                show_gradient_lines = not show_gradient_lines
                print(f"梯度流线显示: {'开启' if show_gradient_lines else '关闭'}")
            elif e.key == 't':
                show_isotherms = not show_isotherms
                print(f"等温线显示: {'开启' if show_isotherms else '关闭'}")
            elif e.key == '=':
                heat_intensity_scale = min(2.0, heat_intensity_scale + 0.1)
                print(f"热强度: {heat_intensity_scale:.1f}")
            elif e.key == '-':
                heat_intensity_scale = max(0.1, heat_intensity_scale - 0.1)
                print(f"热强度: {heat_intensity_scale:.1f}")
            elif e.key == ']':
                isotherm_levels = min(20, isotherm_levels + 1)
                print(f"等温线数量: {isotherm_levels}")
            elif e.key == '[':
                isotherm_levels = max(1, isotherm_levels - 1)
                print(f"等温线数量: {isotherm_levels}")
            elif e.key == ti.GUI.UP:
                input_threshold = min(255, input_threshold + 5)
                print(f"输入阈值: {input_threshold}")
            elif e.key == ti.GUI.DOWN:
                input_threshold = max(0, input_threshold - 5)
                print(f"输入阈值: {input_threshold}")
    
    # 处理输入数据（在仿真更新之前）
    should_process = False
    with data_lock:
        if new_data_available:
            should_process = True
            new_data_available = False
    
    if should_process:
        process_input_data(input_threshold / 255.0, heat_intensity_scale)
    
    if not paused:
        for sub in range(substep):
            diffuse()
            update_temperature_with_input_and_cooling(t_ambient, cooling_rate)
            update_and_commit()
        
        compute_gradients()
    
    # 渲染背景（温度场）
    if use_interpolation:
        temperature_to_color_bilinear_simple(t_np1, pixels, t_min, t_max)
    else:
        temperature_to_color_original(t_np1, pixels, t_min, t_max)
    
    # 绘制等温线
    if show_isotherms:
        draw_isotherms(isotherm_levels, isotherm_color)
    
    # 绘制梯度流线
    if show_gradient_lines:
        draw_gradient_streamlines(gradient_line_length, gradient_line_min_magnitude, 0.0)
    
    my_gui.set_image(pixels)
    
    frame_count += 1
    if current_time - last_fps_time > 1.0:
        fps = frame_count / (current_time - last_fps_time)
        data_age = current_time - last_update_time if last_update_time > 0 else 0
        interpolation_text = "双线性插值" if use_interpolation else "马赛克"
        streamline_text = "流线ON" if show_gradient_lines else "流线OFF"
        isotherm_text = f"等温线ON({isotherm_levels})" if show_isotherms else "等温线OFF"
        my_gui.text(f"FPS: {fps:.1f} | {interpolation_text} | {streamline_text} | {isotherm_text}", 
                   (10, 10), color=0xFFFFFF)
        frame_count = 0
        last_fps_time = current_time
    
    if save_images and not paused:
        my_gui.show(f"images\\output_{i:05}.png")
        i += 1
    else:
        my_gui.show()