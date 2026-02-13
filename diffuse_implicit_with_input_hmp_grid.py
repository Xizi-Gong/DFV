import taichi as ti
import mmap
import struct
import numpy as np
import time
import threading
from threading import Lock
import math
import random

ti.init(arch=ti.cpu)

# control
paused = False
save_images = False
show_gradient_lines = True
show_isotherms = False

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
t_min = -40
t_ambient = 20
cooling_rate = 0.008

# ============ 材料预设 (压缩k值) ============
MATERIAL_PRESETS = {
    '1': ("泡沫塑料 Foam",       10),
    '2': ("木材 Wood",           30),
    '3': ("混凝土 Concrete",      80),
    '4': ("玻璃 Glass",          120),
    '5': ("钢铁 Steel",          250),
    '6': ("铝合金 Aluminum",     400),
    '7': ("铜 Copper",           600),
}

# ============ 环境温度预设 ============
AMBIENT_PRESETS = {
    'z': ("寒冬 -30°C",    -30),
    'x': ("冰点 0°C",        0),
    'c': ("室温 20°C",       20),
    'v': ("酷暑 45°C",       45),
    'm': ("沙漠 60°C",       60),
}

# ============ 冷却速度预设 ============
COOLING_PRESETS = {
    'q': ("静止空气 Still Air",       0.001),
    'w': ("微弱气流 Weak Airflow",    0.003),
    'e': ("自然对流 Natural Conv.",    0.008),
    'd': ("台扇微风 Desk Fan",        0.020),
    'f': ("强力风扇 Strong Fan",      0.050),
}

# 当前选中的预设名（用于显示）
current_material_name = "钢铁 Steel"
current_ambient_name = "室温 20°C"
current_cooling_name = "自然对流 Natural Conv."

# 初始 k 值
k = 250.0

# 实时输入参数
input_update_rate = 30
input_scale_factor = 0.8
input_threshold = 160
heat_intensity_scale = 1.0

# 粒子系统参数
max_particles = 500
particle_spawn_rate = 50
particle_min_life = 15
particle_max_life = 15
particle_speed = 1.5
particle_trail_length = 8
gradient_line_min_magnitude = 0.1

# 等温线参数
isotherm_levels = 10
isotherm_color = 0.2
isotherm_thickness = 1

# 插值参数
use_interpolation = True

# visualization
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

# Spectral 调色板
spectral_hex_full = ["#9e0142","#a00343","#a20643","#a40844","#a70b44","#a90d45","#ab0f45","#ad1245","#af1446","#b11646","#b31947","#b51b47","#b71d48","#ba2048","#bc2248","#be2449","#c02749","#c12949","#c32b4a","#c52d4a","#c7304a","#c9324a","#cb344b","#cd364b","#ce384b","#d03b4b","#d23d4b","#d33f4b","#d5414b","#d7434b","#d8454b","#da474a","#db494a","#dd4b4a","#de4d4a","#df4f4a","#e1514a","#e2534a","#e35549","#e45749","#e65949","#e75b49","#e85d49","#e95f49","#ea6149","#eb6349","#ec6549","#ed6749","#ee6a49","#ef6c49","#f06e4a","#f0704a","#f1724a","#f2744b","#f3774b","#f3794c","#f47b4d","#f47e4d","#f5804e","#f6824f","#f68550","#f78750","#f78951","#f88c52","#f88e53","#f89154","#f99356","#f99557","#f99858","#fa9a59","#fa9c5a","#fa9f5c","#fba15d","#fba35e","#fba660","#fba861","#fcaa62","#fcad64","#fcaf65","#fcb167","#fcb368","#fcb56a","#fdb86b","#fdba6d","#fdbc6e","#fdbe70","#fdc071","#fdc273","#fdc474","#fdc676","#fdc878","#fdca79","#fecc7b","#fecd7d","#fecf7e","#fed180","#fed382","#fed584","#fed685","#fed887","#feda89","#fedb8b","#fedd8d","#fede8f","#fee090","#fee192","#fee394","#fee496","#fee698","#fee79a","#fee89b","#feea9d","#feeb9f","#feeca1","#feeda2","#feefa4","#fef0a5","#fef1a7","#fef2a8","#fdf3a9","#fdf3aa","#fdf4ab","#fdf5ac","#fcf6ad","#fcf6ae","#fcf7af","#fbf7af","#fbf8b0","#faf8b0","#faf9b0","#f9f9b0","#f9f9b0","#f8f9b0","#f7faaf","#f7faaf","#f6faae","#f5faae","#f4f9ad","#f3f9ac","#f2f9ac","#f2f9ab","#f0f9aa","#eff8a9","#eef8a8","#edf8a7","#ecf7a7","#ebf7a6","#e9f6a5","#e8f6a4","#e7f5a3","#e5f5a2","#e4f4a2","#e2f3a1","#e0f3a1","#dff2a0","#ddf1a0","#dbf19f","#d9f09f","#d7ef9f","#d6ee9f","#d4ee9f","#d2ed9e","#d0ec9e","#cdeb9f","#cbea9f","#c9e99f","#c7e89f","#c5e89f","#c3e79f","#c0e6a0","#bee5a0","#bce4a0","#b9e3a0","#b7e2a1","#b4e1a1","#b2e0a1","#b0dfa1","#addea2","#abdda2","#a8dca2","#a6dba3","#a3daa3","#a0d9a3","#9ed8a3","#9bd7a3","#99d6a4","#96d5a4","#94d4a4","#91d3a4","#8ed1a4","#8cd0a4","#89cfa5","#87cea5","#84cda5","#82cba5","#7fcaa6","#7dc9a6","#7ac7a6","#77c6a6","#75c5a7","#73c3a7","#70c2a8","#6ec0a8","#6bbea8","#69bda9","#66bba9","#64b9aa","#62b8aa","#60b6ab","#5db4ac","#5bb2ac","#59b0ad","#57aeae","#55acae","#53aaaf","#51a8af","#50a6b0","#4ea4b1","#4ca2b1","#4ba0b2","#499db2","#489bb3","#4799b3","#4697b3","#4595b4","#4492b4","#4390b4","#438eb4","#428cb5","#4289b5","#4287b4","#4285b4","#4283b4","#4280b4","#437eb3","#437cb3","#447ab3","#4577b2","#4575b1","#4673b1","#4771b0","#486eaf","#4a6caf","#4b6aae","#4c68ad","#4e65ac","#4f63ab","#5161aa","#525fa9","#545ca8","#555aa7","#5758a6","#5956a5","#5b53a4","#5c51a3","#5e4fa2"]

spectral_hex = spectral_hex_full[30:231]

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

spectral_colors_list = [hex_to_rgb(c) for c in spectral_hex]
n_colors = len(spectral_colors_list)

spectral_lut = ti.Vector.field(3, ti.f32, shape=n_colors)

def init_spectral_lut():
    for i, rgb in enumerate(spectral_colors_list):
        spectral_lut[i] = ti.Vector(rgb)

init_spectral_lut()

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


# ============ 梯度插值函数 ============
def bilinear_interpolate_gradient(grad_x_np, grad_y_np, grid_x, grid_y):
    x0 = int(np.floor(grid_x))
    y0 = int(np.floor(grid_y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    x0 = max(0, min(x0, n_x - 1))
    x1 = max(0, min(x1, n_x - 1))
    y0 = max(0, min(y0, n_y - 1))
    y1 = max(0, min(y1, n_y - 1))
    
    fx = grid_x - int(np.floor(grid_x))
    fy = grid_y - int(np.floor(grid_y))
    
    gx00 = grad_x_np[y0, x0]
    gx01 = grad_x_np[y0, x1]
    gx10 = grad_x_np[y1, x0]
    gx11 = grad_x_np[y1, x1]
    gx = (1-fx)*(1-fy)*gx00 + fx*(1-fy)*gx01 + (1-fx)*fy*gx10 + fx*fy*gx11
    
    gy00 = grad_y_np[y0, x0]
    gy01 = grad_y_np[y0, x1]
    gy10 = grad_y_np[y1, x0]
    gy11 = grad_y_np[y1, x1]
    gy = (1-fx)*(1-fy)*gy00 + fx*(1-fy)*gy01 + (1-fx)*fy*gy10 + fx*fy*gy11
    
    return gx, gy


# ============ 粒子系统 ============
class FlowParticle:
    def __init__(self, x, y, life):
        self.x = x
        self.y = y
        self.life = life
        self.max_life = life
        self.trail = [(x, y)]
        self.dying = False
        self.fade_alpha = 1.0
        self.fade_speed = 0.15
    
    def update(self, grad_x_np, grad_y_np, min_mag):
        if self.dying:
            self.fade_alpha -= self.fade_speed
            if len(self.trail) > 1 and self.fade_alpha < 0.7:
                self.trail.pop(0)
            return
        
        grid_x = self.x / scatter
        grid_y = self.y / scatter
        
        if not (0 <= grid_x < n_x and 0 <= grid_y < n_y):
            self.dying = True
            return
        
        gx, gy = bilinear_interpolate_gradient(grad_x_np, grad_y_np, grid_x, grid_y)
        
        mag = np.sqrt(gx*gx + gy*gy)
        
        if mag > min_mag:
            max_speed = particle_speed * 3.0
            actual_speed = min(mag * particle_speed, max_speed)
            
            gx_norm = gx / mag
            gy_norm = gy / mag
            
            self.x -= gx_norm * actual_speed
            self.y -= gy_norm * actual_speed
            
            self.trail.append((self.x, self.y))
            if len(self.trail) > particle_trail_length:
                self.trail.pop(0)
        else:
            self.dying = True
        
        if not (0 <= self.x < res_x and 0 <= self.y < res_y):
            self.dying = True
        
        self.life -= 1
        if self.life <= 0:
            self.dying = True
    
    def is_alive(self):
        if self.dying:
            return self.fade_alpha > 0 and len(self.trail) > 1
        return 0 <= self.x < res_x and 0 <= self.y < res_y
    
    def get_segment_alpha(self, segment_index, total_segments):
        if total_segments <= 1:
            return self.fade_alpha
        trail_alpha = 0.2 + 0.8 * (segment_index / (total_segments - 1))
        return trail_alpha * self.fade_alpha


class ParticleSystem:
    def __init__(self):
        self.particles = []
        self.warmup_frames = 0
        self.warmup_duration = 90
        self.warmup_delay = 15
    
    def reset(self):
        self.particles = []
        self.warmup_frames = 0
    
    def spawn_particles(self, grad_mag_np, num_particles, max_count):
        if len(self.particles) >= max_count:
            self.warmup_frames = self.warmup_duration
            return
        
        if self.warmup_frames < self.warmup_delay:
            self.warmup_frames += 1
            return
        
        if self.warmup_frames < self.warmup_duration:
            progress = (self.warmup_frames - self.warmup_delay) / (self.warmup_duration - self.warmup_delay)
            spawn_probability = progress ** 2
            self.warmup_frames += 1
        else:
            spawn_probability = 1.0
        
        high_grad_indices = np.argwhere(grad_mag_np > gradient_line_min_magnitude)
        
        if len(high_grad_indices) > 0:
            for _ in range(num_particles):
                if len(self.particles) >= max_count:
                    break
                
                if random.random() < spawn_probability:
                    idx = random.choice(high_grad_indices)
                    i, j = idx[0], idx[1]
                    x = (j + random.random()) * scatter
                    y = (i + random.random()) * scatter
                    
                    base_life = random.randint(particle_min_life, particle_max_life)
                    if spawn_probability < 1.0:
                        consumed = random.randint(0, base_life // 2)
                        life = base_life - consumed
                    else:
                        life = base_life
                    
                    self.particles.append(FlowParticle(x, y, life))
    
    def update(self, grad_x_np, grad_y_np, grad_mag_np, spawn_rate, max_count):
        self.particles = [p for p in self.particles if p.is_alive()]
        
        for p in self.particles:
            p.update(grad_x_np, grad_y_np, gradient_line_min_magnitude)
        
        self.spawn_particles(grad_mag_np, spawn_rate, max_count)
    
    def draw(self, pixels_np):
        for p in self.particles:
            total_segments = len(p.trail) - 1
            
            if total_segments < 1:
                continue
            
            for i in range(total_segments):
                x0, y0 = p.trail[i]
                x1, y1 = p.trail[i + 1]
                alpha = p.get_segment_alpha(i, total_segments)
                draw_line_with_alpha(pixels_np, int(x0), int(y0), 
                                     int(x1), int(y1), 1.0, alpha)
            
            if total_segments >= 1:
                x0, y0 = p.trail[-2]
                x1, y1 = p.trail[-1]
                
                ddx = x1 - x0
                ddy = y1 - y0
                length = np.sqrt(ddx*ddx + ddy*ddy)
                
                if length > 0.1:
                    ddx /= length
                    ddy /= length
                    
                    arrow_size = 4
                    arrow_angle = 0.5
                    
                    left_x = x1 - arrow_size * (ddx * np.cos(arrow_angle) - ddy * np.sin(arrow_angle))
                    left_y = y1 - arrow_size * (ddx * np.sin(arrow_angle) + ddy * np.cos(arrow_angle))
                    
                    right_x = x1 - arrow_size * (ddx * np.cos(-arrow_angle) - ddy * np.sin(-arrow_angle))
                    right_y = y1 - arrow_size * (ddx * np.sin(-arrow_angle) + ddy * np.cos(-arrow_angle))
                    
                    head_alpha = p.get_segment_alpha(total_segments - 1, total_segments)
                    
                    draw_line_with_alpha(pixels_np, int(x1), int(y1), 
                                         int(left_x), int(left_y), 1.0, head_alpha)
                    draw_line_with_alpha(pixels_np, int(x1), int(y1), 
                                         int(right_x), int(right_y), 1.0, head_alpha)


particle_system = ParticleSystem()


# ============ 绘制函数 ============
def draw_line_with_alpha(image_array, x0, y0, x1, y1, color_value, alpha):
    ddx = abs(x1 - x0)
    ddy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = ddx - ddy
    
    x, y = x0, y0
    
    blend = alpha * 0.6
    keep = 1.0 - blend
    
    while True:
        if 0 <= x < image_array.shape[0] and 0 <= y < image_array.shape[1]:
            image_array[x, y, 0] = image_array[x, y, 0] * keep + color_value * blend
            image_array[x, y, 1] = image_array[x, y, 1] * keep + color_value * blend
            image_array[x, y, 2] = image_array[x, y, 2] * keep + color_value * blend
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -ddy:
            err -= ddy
            x += sx
        if e2 < ddx:
            err += ddx
            y += sy


def draw_line_bresenham(image_array, x0, y0, x1, y1, r, g, b):
    ddx = abs(x1 - x0)
    ddy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = ddx - ddy
    
    x, y = x0, y0
    
    while True:
        if 0 <= x < image_array.shape[0] and 0 <= y < image_array.shape[1]:
            image_array[x, y, 0] = image_array[x, y, 0] * 0.7 + r * 0.3
            image_array[x, y, 1] = image_array[x, y, 1] * 0.7 + g * 0.3
            image_array[x, y, 2] = image_array[x, y, 2] * 0.7 + b * 0.3
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -ddy:
            err -= ddy
            x += sx
        if e2 < ddx:
            err += ddx
            y += sy


# ============ Taichi 核函数 ============
@ti.func
def ind(i, j):
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
    """构建并预计算扩散矩阵 - 使用当前全局 k 值"""
    global precomputed_solver, precomputed_matrix
    global D_builder, I_builder
    
    # 每次重建需要新的 builder（Taichi 要求）
    D_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total*5)
    I_builder = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total)
    
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
    
    print(f"矩阵预计算完成 - k={k:.0f}, c={c:.4f}")
    
    return D, I


def switch_material(key):
    """切换材料类型，重建矩阵"""
    global k, current_material_name
    
    if key in MATERIAL_PRESETS:
        name, new_k = MATERIAL_PRESETS[key]
        k = new_k
        current_material_name = name
        buildMatrices()
        particle_system.reset()
        print(f"材料切换: {name} (k={new_k})")


def switch_ambient(key):
    """切换环境温度 - 重置温度场到新环境温度"""
    global t_ambient, current_ambient_name
    
    if key in AMBIENT_PRESETS:
        name, new_temp = AMBIENT_PRESETS[key]
        t_ambient = new_temp
        current_ambient_name = name
        # 重置温度场到新的环境温度，让背景色立刻变化
        reset_temperature_to_ambient(float(t_ambient))
        particle_system.reset()
        print(f"环境温度切换: {name}")


def switch_cooling(key):
    """切换冷却速度"""
    global cooling_rate, current_cooling_name
    
    if key in COOLING_PRESETS:
        name, new_rate = COOLING_PRESETS[key]
        cooling_rate = new_rate
        current_cooling_name = name
        print(f"冷却速度切换: {name} (rate={new_rate})")


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
def reset_temperature_to_ambient(ambient: ti.f32):
    """将温度场重置到环境温度"""
    for i, j in ti.ndrange(n_y, n_x):
        t_n[ind(i, j)] = ambient
        t_np1[ind(i, j)] = ambient


@ti.kernel
def compute_gradients():
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
    for i, j in ti.ndrange(n_y, n_x):
        processed_input[i, j] = 0.0
    
    for i, j in ti.ndrange(n_y, n_x):
        flipped_i = n_y - 1 - i
        input_value = input_data[flipped_i, j]
        
        if input_value < threshold:
            processed_input[i, j] = (threshold - input_value) * intensity_scale


@ti.kernel
def update_temperature_with_input_and_cooling_kernel(ambient_temp: ti.f32, cool: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        current_temp = t_np1[ind(i, j)]
        cooled_temp = current_temp + (ambient_temp - current_temp) * cool
        final_temp = cooled_temp
        
        heat_input = processed_input[i, j]
        if heat_input > 0:
            target_temp = 200 + heat_input * (t_max - t_min)
            calculated_temp = cooled_temp * 0.7 + target_temp * 0.3
            final_temp = ti.max(calculated_temp, ambient_temp)
        else:
            final_temp = cooled_temp
        
        t_np1[ind(i, j)] = final_temp


def process_input_data(threshold, intensity_scale):
    process_input_data_kernel(threshold, intensity_scale)


def update_temperature_with_input_and_cooling(ambient_temp, cool):
    update_temperature_with_input_and_cooling_kernel(ambient_temp, cool)


# 调暗系数
brightness_scale = 0.9

@ti.func
def get_color(v, vmin, vmax, brightness: ti.f32):
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    
    t = (v - vmin) / (vmax - vmin + 1e-10)
    t = 1.0 - t
    
    idx_f = t * (n_colors - 1)
    idx0 = ti.cast(ti.floor(idx_f), ti.i32)
    idx1 = idx0 + 1
    
    if idx0 < 0:
        idx0 = 0
    if idx0 >= n_colors - 1:
        idx0 = n_colors - 2
    if idx1 >= n_colors:
        idx1 = n_colors - 1
    
    frac = idx_f - ti.cast(idx0, ti.f32)
    
    c0 = spectral_lut[idx0]
    c1 = spectral_lut[idx1]
    
    return (c0 * (1.0 - frac) + c1 * frac) * brightness


@ti.func
def get_temperature_clamped(i, j):
    clamped_i = ti.max(0, ti.min(n_y - 1, i))
    clamped_j = ti.max(0, ti.min(n_x - 1, j))
    return t_np1[ind(clamped_i, clamped_j)]


@ti.kernel 
def temperature_to_color_original(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32, brightness: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        for kk, l in ti.ndrange(scatter, scatter):
            color[j*scatter+l, i*scatter+kk] = get_color(t[ind(i,j)], tmin, tmax, brightness)


@ti.kernel
def temperature_to_color_bilinear_simple(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32, brightness: ti.f32):
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
        
        color[pixel_x, pixel_y] = get_color(temp_interpolated, tmin, tmax, brightness)


def linear_interp(x, x0, x1, y0, y1):
    if abs(x1 - x0) < 1e-10:
        return 0.5
    return (x - x0) / (x1 - x0)


def draw_isotherms(num_levels, line_color):
    t_np = t_np1.to_numpy().reshape(n_y, n_x)
    pixels_np = pixels.to_numpy()
    
    temp_min = np.min(t_np)
    temp_max = np.max(t_np)
    temp_range = temp_max - temp_min
    
    if temp_range < 1.0:
        return
    
    iso_temps = []
    for level in range(1, num_levels + 1):
        iso_temp = temp_min + (temp_range * level) / (num_levels + 1)
        iso_temps.append(iso_temp)
    
    for iso_temp in iso_temps:
        for i in range(n_y - 1):
            for j in range(n_x - 1):
                v0 = t_np[i, j]
                v1 = t_np[i, j+1]
                v2 = t_np[i+1, j+1]
                v3 = t_np[i+1, j]
                
                case = 0
                if v0 > iso_temp:
                    case |= 1
                if v1 > iso_temp:
                    case |= 2
                if v2 > iso_temp:
                    case |= 4
                if v3 > iso_temp:
                    case |= 8
                
                x0 = j * scatter
                y0 = i * scatter
                x1 = (j + 1) * scatter
                y1 = (i + 1) * scatter
                
                edges = []
                
                if (v0 <= iso_temp < v1) or (v1 <= iso_temp < v0):
                    tt = linear_interp(iso_temp, v0, v1, 0, 1)
                    edges.append((x0 + tt * scatter, y0))
                
                if (v1 <= iso_temp < v2) or (v2 <= iso_temp < v1):
                    tt = linear_interp(iso_temp, v1, v2, 0, 1)
                    edges.append((x1, y0 + tt * scatter))
                
                if (v3 <= iso_temp < v2) or (v2 <= iso_temp < v3):
                    tt = linear_interp(iso_temp, v3, v2, 0, 1)
                    edges.append((x0 + tt * scatter, y1))
                
                if (v0 <= iso_temp < v3) or (v3 <= iso_temp < v0):
                    tt = linear_interp(iso_temp, v0, v3, 0, 1)
                    edges.append((x0, y0 + tt * scatter))
                
                if case in [1, 14] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case in [2, 13] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case in [3, 12] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case in [4, 11] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 5 and len(edges) >= 4:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                    draw_line_bresenham(pixels_np, int(edges[2][0]), int(edges[2][1]),
                                      int(edges[3][0]), int(edges[3][1]), line_color, line_color, line_color)
                elif case in [6, 9] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case in [7, 8] and len(edges) >= 2:
                    draw_line_bresenham(pixels_np, int(edges[0][0]), int(edges[0][1]),
                                      int(edges[1][0]), int(edges[1][1]), line_color, line_color, line_color)
                elif case == 10 and len(edges) >= 4:
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
    global new_data_available, last_update_time
    
    HEADER_SIZE = 12
    
    mmf = None
    try:
        import mmap
        import os
        
        if os.name == 'nt':
            try:
                mmf = mmap.mmap(0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
                print("成功连接到共享内存")
            except Exception as e:
                print(f"无法打开Windows共享内存: {e}")
                return
        else:
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
                    
                    with data_lock:
                        try:
                            input_data.from_numpy(normalized_data)
                            new_data_available = True
                            last_update_time = time.time()
                        except Exception as e:
                            pass
                
                time.sleep(1.0 / input_update_rate)
                
            except Exception as e:
                time.sleep(0.1)
                
    except Exception as e:
        print(f"共享内存初始化失败: {e}")
    finally:
        if mmf:
            mmf.close()


# ============ 主循环 ============
my_gui = ti.GUI("Heat Diffusion with Gradient Particle Flow", (res_x, res_y))

init()
D, I = buildMatrices()

memory_thread = threading.Thread(target=read_shared_memory, daemon=True)
memory_thread.start()

i = 0
frame_count = 0
last_fps_time = time.time()

print("\n" + "="*60)
print("控制说明:")
print("="*60)
print("\n--- 材料 (数字键 1-7, 需重建矩阵) ---")
for key, (name, val) in sorted(MATERIAL_PRESETS.items()):
    marker = " ◀" if name == current_material_name else ""
    print(f"  {key}: {name} (k={val}){marker}")

print("\n--- 环境温度 (Z/X/C/V/M) ---")
for key in ['z','x','c','v','m']:
    name, val = AMBIENT_PRESETS[key]
    marker = " ◀" if name == current_ambient_name else ""
    print(f"  {key.upper()}: {name}{marker}")

print("\n--- 冷却速度 (Q/W/E/D/F) ---")
for key in ['q','w','e','d','f']:
    name, val = COOLING_PRESETS[key]
    marker = " ◀" if name == current_cooling_name else ""
    print(f"  {key.upper()}: {name} (rate={val}){marker}")

print("\n--- 显示控制 ---")
print("  G: 粒子流线开关")
print("  T: 等温线开关")
print("  S: 插值模式切换")
print("  B/N: 调暗/调亮背景")
print("\n--- 其他 ---")
print("  空格: 暂停/继续  R: 重置  I: 保存图像")
print("  +/-: 热强度  [/]: 等温线数量  上/下: 阈值")
print("  ,/.: 粒子生成速率  O/P: 最大粒子数")
print("  ESC: 退出")
print("="*60)
print(f"当前: {current_material_name} | {current_ambient_name} | {current_cooling_name}")
print("="*60 + "\n")

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
                particle_system.reset()
                i = 0
                print("重置仿真")
            elif e.key == 's':
                use_interpolation = not use_interpolation
                print(f"插值模式: {'双线性插值' if use_interpolation else '马赛克'}")
            elif e.key == 'g':
                show_gradient_lines = not show_gradient_lines
                print(f"粒子流线: {'开启' if show_gradient_lines else '关闭'}")
            elif e.key == 't':
                show_isotherms = not show_isotherms
                print(f"等温线: {'开启' if show_isotherms else '关闭'}")
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
            elif e.key == '.':
                particle_spawn_rate = min(100, particle_spawn_rate + 5)
                print(f"粒子生成速率: {particle_spawn_rate}/帧")
            elif e.key == ',':
                particle_spawn_rate = max(0, particle_spawn_rate - 5)
                print(f"粒子生成速率: {particle_spawn_rate}/帧")
            elif e.key == 'p':
                max_particles = min(2000, max_particles + 100)
                print(f"最大粒子数: {max_particles}")
            elif e.key == 'o':
                max_particles = max(100, max_particles - 100)
                print(f"最大粒子数: {max_particles}")
            elif e.key == 'b':
                brightness_scale = max(0.1, brightness_scale - 0.1)
                print(f"背景亮度: {brightness_scale:.1f}")
            elif e.key == 'n':
                brightness_scale = min(1.0, brightness_scale + 0.1)
                print(f"背景亮度: {brightness_scale:.1f}")
            # --- 材料切换 (1-7) ---
            elif e.key in MATERIAL_PRESETS:
                switch_material(e.key)
            # --- 环境温度切换 (Z/X/C/V/M) ---
            elif e.key in AMBIENT_PRESETS:
                switch_ambient(e.key)
            # --- 冷却速度切换 (Q/W/E/D/F) ---
            elif e.key in COOLING_PRESETS:
                switch_cooling(e.key)
    
    # 处理输入数据
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
        temperature_to_color_bilinear_simple(t_np1, pixels, t_min, t_max, brightness_scale)
    else:
        temperature_to_color_original(t_np1, pixels, t_min, t_max, brightness_scale)
    
    # 绘制等温线
    if show_isotherms:
        draw_isotherms(isotherm_levels, isotherm_color)
    
    # 更新并绘制粒子系统
    if show_gradient_lines and not paused:
        grad_mag_np = gradient_magnitude.to_numpy()
        grad_x_np = gradient_x.to_numpy()
        grad_y_np = gradient_y.to_numpy()
        
        particle_system.update(grad_x_np, grad_y_np, grad_mag_np, particle_spawn_rate, max_particles)
        
        pixels_np = pixels.to_numpy()
        particle_system.draw(pixels_np)
        pixels.from_numpy(pixels_np)
    
    my_gui.set_image(pixels)
    
    frame_count += 1
    if current_time - last_fps_time > 1.0:
        fps = frame_count / (current_time - last_fps_time)
        interpolation_text = "双线性" if use_interpolation else "马赛克"
        particle_text = f"粒子({len(particle_system.particles)})" if show_gradient_lines else "粒子OFF"
        isotherm_text = f"等温线({isotherm_levels})" if show_isotherms else ""
        
        # 状态栏显示当前预设
        status = f"FPS:{fps:.0f} | {current_material_name} | {current_ambient_name} | {current_cooling_name}"
        my_gui.text(status, (10, 10), color=0xFFFFFF)
        
        frame_count = 0
        last_fps_time = current_time
    
    if save_images and not paused:
        my_gui.show(f"images\\output_{i:05}.png")
        i += 1
    else:
        my_gui.show()
        