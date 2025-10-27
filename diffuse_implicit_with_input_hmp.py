import taichi as ti
import mmap
import struct
import numpy as np
import time
import threading
from threading import Lock

ti.init(arch=ti.cpu)

# control
paused = False
save_images = False

# problem setting - 改为矩形网格以匹配输入比例
n_x = 78  # 宽度，匹配输入宽度
n_y = 52  # 高度，匹配输入高度
scatter = 8
res_x = n_x * scatter  # 624
res_y = n_y * scatter  # 416

# 共享内存参数
SHARED_MEMORY_NAME = "shared_touch_image"
SHARED_MEMORY_SIZE = 4068  # 78*52*1 + 头部信息
INPUT_WIDTH = 78
INPUT_HEIGHT = 52
INPUT_FRAME_SIZE = INPUT_WIDTH * INPUT_HEIGHT

# physical parameters
h = 2e-3    # 稍微增大时间步长以适应实时输入
substep = 1
dx = 1
t_max = 300
t_min = 0
k = 700.0  # 稍微降低扩散系数，让热源更明显
t_ambient = 20  # 环境温度
cooling_rate = 0.02  # 环境冷却速率，每帧冷却2%

# 实时输入参数
input_update_rate = 30  # Hz，从共享内存读取数据的频率
input_scale_factor = 0.8  # 输入数据的缩放因子 (0-1)
input_threshold = 160  # 输入数据阈值，低于此值才认为是有效热源（反向逻辑）
heat_intensity_scale = 1.0  # 热强度缩放

# 插值参数
use_interpolation = True  # 是否使用插值平滑

# visualization - 改为矩形
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

# diffuse matrix
n_total = n_x * n_y  # 总网格点数
D_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total*5) #taichi的稀疏矩阵构建器
I_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total)

# 添加预计算变量
precomputed_solver = None
precomputed_matrix = None

# temperature fields
t_n = ti.field(ti.f32, shape=n_total)
t_np1 = ti.field(ti.f32, shape=n_total)

# 实时输入数据存储 - 与原始代码完全一致
input_data = ti.field(ti.f32, shape=(INPUT_HEIGHT, INPUT_WIDTH))
processed_input = ti.field(ti.f32, shape=(n_y, n_x))  # 注意顺序：高度×宽度

# 线程同步
data_lock = Lock()
new_data_available = False
last_update_time = 0

@ti.func
def ind(i, j):
    """将2D坐标转换为1D索引，i是行（y方向），j是列（x方向）"""
    return i * n_x + j

@ti.kernel
def fillDiffusionMatrixBuilder(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(n_y, n_x):  # i是行（y），j是列（x）
        count = 0
        # 上邻居
        if i-1 >= 0:
            A[ind(i,j), ind(i-1,j)] += 1
            count += 1
        # 下邻居
        if i+1 < n_y:
            A[ind(i,j), ind(i+1,j)] += 1
            count += 1
        # 左邻居
        if j-1 >= 0:
            A[ind(i,j), ind(i,j-1)] += 1
            count += 1
        # 右邻居
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
    
    # 预计算系数矩阵 - 使用固定的时间步长
    c = h * k / dx**2  # 注意：这里使用全局的 h，不再需要 dt 参数
    ImcD = I - c * D 
    
    # 预计算求解器
    precomputed_solver = ti.linalg.SparseSolver(solver_type="LLT") #taichi的稀疏矩阵求解器
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

@ti.kernel
def process_input_data(threshold: ti.f32, intensity_scale: ti.f32):
    """将输入数据处理并映射到仿真网格 - 与原始代码完全一致"""
    # 清除上一帧的处理数据
    for i, j in ti.ndrange(n_y, n_x):
        processed_input[i, j] = 0.0
    
    # 直接1:1映射，但修正上下翻转 - 与原始代码完全一致
    for i, j in ti.ndrange(n_y, n_x):
        # 修正上下翻转：input_data的第0行对应processed_input的最后一行
        flipped_i = n_y - 1 - i
        input_value = input_data[flipped_i, j]
        
        # 应用反向阈值逻辑：低于阈值才是有效数据
        if input_value < threshold:
            # 计算反向强度：阈值越低，热强度越高
            processed_input[i, j] = (threshold - input_value) * intensity_scale

@ti.kernel
def update_temperature_with_input_and_cooling(ambient_temp: ti.f32, cooling: ti.f32):
    """根据处理后的输入数据更新温度场，并添加环境冷却 - 与原始代码逻辑一致"""
    for i, j in ti.ndrange(n_y, n_x):
        current_temp = t_np1[ind(i, j)]
        
        # 环境冷却：温度向环境温度衰减
        cooled_temp = current_temp + (ambient_temp - current_temp) * cooling
        
        # 初始化最终温度为冷却后的温度
        final_temp = cooled_temp
        
        # 热输入加热 - 使用与原始代码相同的逻辑
        heat_input = processed_input[i, j]
        if heat_input > 0:
            # 计算加热后的温度 - 使用与原始代码相同的逻辑
            target_temp = 225 + heat_input * (t_max - t_min)
            calculated_temp = cooled_temp * 0.7 + target_temp * 0.3
            # 确保有热输入时温度不低于环境温度
            final_temp = ti.max(calculated_temp, t_ambient)
        else:
            # 没有热输入时，只有环境冷却
            final_temp = cooled_temp
        
        t_np1[ind(i, j)] = final_temp

def diffuse():
    """简化的扩散函数，使用预计算的求解器"""
    global precomputed_solver
    if precomputed_solver is None:
        print("错误：求解器未预计算！请先调用 buildMatrices()")
        return
    
    # 现在只需要求解线性方程组，这是唯一的计算开销
    t_np1.from_numpy(precomputed_solver.solve(t_n))

def update_and_commit():
    t_n.copy_from(t_np1)

# 共享内存读取函数
def read_shared_memory():
    """从共享内存读取数据的线程函数"""
    global new_data_available, last_update_time
    
    # 定义头部大小
    HEADER_SIZE = 12  # 4068 - 4056 = 12字节头部
    
    mmf = None
    try:
        # 尝试打开共享内存
        import mmap
        import os
        import tempfile
        
        if os.name == 'nt':  # Windows
            try:
                # Windows使用命名共享内存
                mmf = mmap.mmap(0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
                print("成功连接到共享内存")
            except Exception as e:
                print(f"无法打开Windows共享内存: {e}")
                return
        else:  # Linux/Mac
            try:
                # Unix系统使用文件映射
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
                with data_lock:
                    # 跳过头部，从正确位置读取帧数据
                    mmf.seek(HEADER_SIZE)  # 跳过12字节头部
                    raw_data = mmf.read(INPUT_FRAME_SIZE)
                    
                    if len(raw_data) >= INPUT_FRAME_SIZE:
                        # 将数据转换为numpy数组
                        np_data = np.frombuffer(raw_data[:INPUT_FRAME_SIZE], dtype=np.uint8)
                        np_data = np_data.reshape((INPUT_HEIGHT, INPUT_WIDTH))
                        
                        # 转换为float并归一化到0-1范围
                        normalized_data = np_data.astype(np.float32) / 255.0
                        
                        # 更新Taichi字段
                        input_data.from_numpy(normalized_data)
                        
                        new_data_available = True
                        last_update_time = time.time()
                
                time.sleep(1.0 / input_update_rate)  # 控制读取频率
                
            except Exception as e:
                print(f"读取共享内存时出错: {e}")
                time.sleep(0.1)
                
    except Exception as e:
        print(f"共享内存初始化失败: {e}")
    finally:
        if mmf:
            mmf.close()

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
    """获取温度值，越界时使用边界值（夹紧）"""
    clamped_i = ti.max(0, ti.min(n_y - 1, i))
    clamped_j = ti.max(0, ti.min(n_x - 1, j))
    return t_np1[ind(clamped_i, clamped_j)]

@ti.kernel 
def temperature_to_color_original(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    """原始的温度到颜色转换（简单重复）- 与原始代码完全一致"""
    for i, j in ti.ndrange(n_y, n_x):  # 遍历温度网格
        for k, l in ti.ndrange(scatter, scatter):  # 每个网格点对应scatter×scatter个像素
            # 注意坐标映射：i对应y方向，j对应x方向
            # 这与原始代码完全一致：color[j*scatter+l, i*scatter+k]
            color[j*scatter+l, i*scatter+k] = get_color(t[ind(i,j)], tmin, tmax)

@ti.kernel
def temperature_to_color_bilinear_simple(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    """简化的双线性插值 - 使用夹紧边界条件"""
    for pixel_x, pixel_y in color:  # 遍历所有像素
        # 将像素坐标转换为网格坐标（浮点数）
        grid_j = pixel_x / scatter  # 对应温度网格的 j (列，x方向)
        grid_i = pixel_y / scatter  # 对应温度网格的 i (行，y方向)
        
        # 获取四个最近的网格点
        i0 = int(grid_i)
        j0 = int(grid_j)
        i1 = i0 + 1
        j1 = j0 + 1
        
        # 计算插值权重
        fx = grid_j - j0
        fy = grid_i - i0
        
        # 获取四个角点的温度值 - 使用夹紧边界条件
        t00 = get_temperature_clamped(i0, j0)
        t01 = get_temperature_clamped(i0, j1)
        t10 = get_temperature_clamped(i1, j0)
        t11 = get_temperature_clamped(i1, j1)
        
        # 双线性插值
        temp_interpolated = (1-fx) * (1-fy) * t00 + fx * (1-fy) * t01 + (1-fx) * fy * t10 + fx * fy * t11
        
        color[pixel_x, pixel_y] = get_color(temp_interpolated, tmin, tmax)

# GUI - 使用矩形窗口
my_gui = ti.GUI("Heat Diffusion with Simple Bilinear Interpolation", (res_x, res_y))

init()
D, I = buildMatrices()

# 启动共享内存读取线程
memory_thread = threading.Thread(target=read_shared_memory, daemon=True)
memory_thread.start()

i = 0
frame_count = 0
last_fps_time = time.time()

print("控制说明:")
print("- 空格键: 暂停/继续")
print("- R键: 重置")
print("- I键: 开启/关闭图像保存")
print("- S键: 切换插值模式 (关闭/简化双线性)")
print("- +/-键: 调整输入强度")
print("- 上/下方向键: 调整阈值（低于此值才算有效热源）")
print("- ESC: 退出")
print(f"仿真网格: {n_x}×{n_y}, 显示分辨率: {res_x}×{res_y}")
print(f"环境冷却速率: {cooling_rate:.3f}")
print(f"插值模式: {'简化双线性插值' if use_interpolation else '无插值(马赛克)'}")

while my_gui.running: # main loop
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
            elif e.key == '=':  # +键
                heat_intensity_scale = min(2.0, heat_intensity_scale + 0.1)
                print(f"热强度: {heat_intensity_scale:.1f}")
            elif e.key == '-':
                heat_intensity_scale = max(0.1, heat_intensity_scale - 0.1)
                print(f"热强度: {heat_intensity_scale:.1f}")
            elif e.key == ti.GUI.UP:
                input_threshold = min(255, input_threshold + 5)
                print(f"输入阈值: {input_threshold} (低于此值才是有效热源)")
            elif e.key == ti.GUI.DOWN:
                input_threshold = max(0, input_threshold - 5)
                print(f"输入阈值: {input_threshold} (低于此值才是有效热源)")
    
    # 处理新的输入数据
    if new_data_available:
        with data_lock:
            process_input_data(input_threshold / 255.0, heat_intensity_scale)
            new_data_available = False
    
    if not paused:
        for sub in range(substep):
            diffuse()
            update_temperature_with_input_and_cooling(t_ambient, cooling_rate)  # 新函数，包含环境冷却
            update_and_commit()
    
    # 使用插值或原始方法渲染
    if use_interpolation:
        temperature_to_color_bilinear_simple(t_np1, pixels, t_min, t_max)
    else:
        temperature_to_color_original(t_np1, pixels, t_min, t_max)
    
    my_gui.set_image(pixels)
    
    # FPS计算和显示
    frame_count += 1
    if current_time - last_fps_time > 1.0:
        fps = frame_count / (current_time - last_fps_time)
        data_age = current_time - last_update_time if last_update_time > 0 else 0
        interpolation_text = "简化双线性插值" if use_interpolation else "马赛克模式"
        my_gui.text(f"FPS: {fps:.1f} | Data age: {data_age:.1f}s | {interpolation_text}", 
                   (10, 10), color=0xFFFFFF)
        frame_count = 0
        last_fps_time = current_time
    
    if save_images and not paused:
        my_gui.show(f"images\\output_{i:05}.png")
        i += 1
    else:
        my_gui.show()