"""
Heat Diffusion — WebSocket 帧流（WebGL 版）
- 不再编码 JPEG，改为发送裸 float32 温度数组 + 粒子线段
- 浏览器用 WebGL fragment shader 完成着色，GPU 双线性插值
- 帧尺寸: 8B header + 78×52×4B temp + n_segs×20B particles ≈ 16~26 KB
"""

import taichi as ti
import mmap
import numpy as np
import struct
import time
import threading
from threading import Lock
import random
import asyncio
import websockets
import json
import io

ti.init(arch=ti.cpu)

# ──────────────────────────────────────────────
#  仿真参数
# ──────────────────────────────────────────────
paused      = False
freeze_input = False
show_gradient_lines = False

# ── 长按冻结配置 ──
LONGPRESS_DURATION  = 1.5          # 按住多少秒触发冻结
longpress_start     = None         # 当前按压开始时间
freeze_progress     = 0.0          # 冻结进度 0.0-1.0
show_isotherms      = True
use_interpolation   = True
brightness_scale    = 0.9

n_x, n_y = 78, 52
scatter   = 8
res_x     = n_x * scatter   # 624（粒子坐标仍用此空间）
res_y     = n_y * scatter   # 416

SHARED_MEMORY_NAME = "shared_touch_image"
SHARED_MEMORY_SIZE = 4068
INPUT_WIDTH        = 78
INPUT_HEIGHT       = 52
INPUT_FRAME_SIZE   = INPUT_WIDTH * INPUT_HEIGHT

h   = 2e-3
substep = 1
dx  = 1
t_max = 50                    # ← 从 300 改为 50（物理温度上限）
t_min = -40
t_ambient   = 0.0
cooling_rate = 0.008

MATERIAL_PRESETS = {
    'Foam Plastic': 10,
    'Wood':         30,
    'Concrete':     80,
    'Glass':       120,
    'Steel':       250,
    'Alum':        400,
    'Copper':      600,
}
AMBIENT_PRESETS = {                # ← 改为真实物理温度
    'Cold Winter Outdoors': -20,
    'Freezing Point':         0,
    'Comfortable Room':      20,
    'Hot Summer':            35,
    'Desert Heat':           50,
}
COOLING_PRESETS = {
    'Still Air':                 0.001,
    'Weak Airflow':              0.003,
    'Indoor Natural Convection': 0.008,
    'Strong Fan':                0.050,
}

current_material_name = "Glass"
current_ambient_name  = "Freezing Point"
current_cooling_name  = "Indoor Natural Convection"
k = 120.0

input_update_rate    = 30
input_threshold      = 160
heat_intensity_scale = 1.0

max_particles          = 500
particle_spawn_rate    = 50
particle_min_life      = 15
particle_max_life      = 15
particle_speed         = 1.5
particle_trail_length  = 8
gradient_line_min_magnitude = 0.1

isotherm_levels = 10
isotherm_color  = 0.2

# ──────────────────────────────────────────────
#  WebSocket 配置
# ──────────────────────────────────────────────
WS_HOST = "localhost"
WS_PORT = 8765

latest_frame_bytes = None
latest_frame_lock  = Lock()
connected_clients  = set()
clients_lock       = Lock()

rebuild_matrix_flag  = False
rebuild_matrix_lock  = Lock()
reset_ambient_flag   = False
reset_ambient_value  = 20.0
reset_ambient_lock   = Lock()
reset_sim_flag       = False
reset_sim_lock       = Lock()


def handle_ws_message(msg_str):
    global k, t_ambient, cooling_rate
    global show_gradient_lines, show_isotherms, paused, freeze_input
    global current_material_name, current_ambient_name, current_cooling_name
    global rebuild_matrix_flag, reset_ambient_flag, reset_ambient_value
    global reset_sim_flag, heat_intensity_scale, isotherm_levels, brightness_scale
    global longpress_start, freeze_progress, peak_touch_area

    try:
        msg    = json.loads(msg_str)
        action = msg.get("action")
        value  = msg.get("value")

        if action == "set_material" and value in MATERIAL_PRESETS:
            k = float(MATERIAL_PRESETS[value])
            current_material_name = value
            with rebuild_matrix_lock:
                rebuild_matrix_flag = True

        elif action == "set_ambient" and value in AMBIENT_PRESETS:
            new_temp = float(AMBIENT_PRESETS[value])
            t_ambient = new_temp
            current_ambient_name = value
            with reset_ambient_lock:
                reset_ambient_flag  = True
                reset_ambient_value = new_temp

        elif action == "set_airflow" and value in COOLING_PRESETS:
            cooling_rate = float(COOLING_PRESETS[value])
            current_cooling_name = value

        elif action == "toggle_streamlines":
            show_gradient_lines = bool(value)

        elif action == "toggle_isotherms":
            show_isotherms = bool(value)

        elif action == "toggle_pause":
            paused = not paused

        elif action == "set_pause":
            paused = bool(value)

        elif action == "set_freeze_input":
            freeze_input = bool(value)
            if not freeze_input:
                longpress_start = None
                freeze_progress = 0.0
                peak_touch_area = 0

        elif action == "reset":
            with reset_sim_lock:
                reset_sim_flag = True

        elif action == "set_heat_intensity":
            heat_intensity_scale = max(0.1, min(2.0, float(value)))

        elif action == "set_brightness":
            brightness_scale = max(0.3, min(1.0, float(value)))

    except Exception as e:
        print(f"[WS] 解析错误: {e}")


# ──────────────────────────────────────────────
#  Spectral 调色板
# ──────────────────────────────────────────────
spectral_hex_full = ["#9e0142","#a00343","#a20643","#a40844","#a70b44","#a90d45","#ab0f45","#ad1245","#af1446","#b11646","#b31947","#b51b47","#b71d48","#ba2048","#bc2248","#be2449","#c02749","#c12949","#c32b4a","#c52d4a","#c7304a","#c9324a","#cb344b","#cd364b","#ce384b","#d03b4b","#d23d4b","#d33f4b","#d5414b","#d7434b","#d8454b","#da474a","#db494a","#dd4b4a","#de4d4a","#df4f4a","#e1514a","#e2534a","#e35549","#e45749","#e65949","#e75b49","#e85d49","#e95f49","#ea6149","#eb6349","#ec6549","#ed6749","#ee6a49","#ef6c49","#f06e4a","#f0704a","#f1724a","#f2744b","#f3774b","#f3794c","#f47b4d","#f47e4d","#f5804e","#f6824f","#f68550","#f78750","#f78951","#f88c52","#f88e53","#f89154","#f99356","#f99557","#f99858","#fa9a59","#fa9c5a","#fa9f5c","#fba15d","#fba35e","#fba660","#fba861","#fcaa62","#fcad64","#fcaf65","#fcb167","#fcb368","#fcb56a","#fdb86b","#fdba6d","#fdbc6e","#fdbe70","#fdc071","#fdc273","#fdc474","#fdc676","#fdc878","#fdca79","#fecc7b","#fecd7d","#fecf7e","#fed180","#fed382","#fed584","#fed685","#fed887","#feda89","#fedb8b","#fedd8d","#fede8f","#fee090","#fee192","#fee394","#fee496","#fee698","#fee79a","#fee89b","#feea9d","#feeb9f","#feeca1","#feeda2","#feefa4","#fef0a5","#fef1a7","#fef2a8","#fdf3a9","#fdf3aa","#fdf4ab","#fdf5ac","#fcf6ad","#fcf6ae","#fcf7af","#fbf7af","#fbf8b0","#faf8b0","#faf9b0","#f9f9b0","#f9f9b0","#f8f9b0","#f7faaf","#f7faaf","#f6faae","#f5faae","#f4f9ad","#f3f9ac","#f2f9ac","#f2f9ab","#f0f9aa","#eff8a9","#eef8a8","#edf8a7","#ecf7a7","#ebf7a6","#e9f6a5","#e8f6a4","#e7f5a3","#e5f5a2","#e4f4a2","#e2f3a1","#e0f3a1","#dff29f","#ddf19f","#dbf19f","#d9f09f","#d7ef9f","#d6ee9f","#d4ee9f","#d2ed9e","#d0ec9e","#cdeb9f","#cbea9f","#c9e99f","#c7e89f","#c5e89f","#c3e79f","#c0e6a0","#bee5a0","#bce4a0","#b9e3a0","#b7e2a1","#b4e1a1","#b2e0a1","#b0dfa1","#addea2","#abdda2","#a8dca2","#a6dba3","#a3daa3","#a0d9a3","#9ed8a3","#9bd7a3","#99d6a4","#96d5a4","#94d4a4","#91d3a4","#8ed1a4","#8cd0a4","#89cfa5","#87cea5","#84cda5","#82cba5","#7fcaa6","#7dc9a6","#7ac7a6","#77c6a6","#75c5a7","#73c3a7","#70c2a8","#6ec0a8","#6bbea8","#69bda9","#66bba9","#64b9aa","#62b8aa","#60b6ab","#5db4ac","#5bb2ac","#59b0ad","#57aeae","#55acae","#53aaaf","#51a8af","#50a6b0","#4ea4b1","#4ca2b1","#4ba0b2","#499db2","#489bb3","#4799b3","#4697b3","#4595b4","#4492b4","#4390b4","#438eb4","#428cb5","#4289b5","#4287b4","#4285b4","#4283b4","#4280b4","#437eb3","#437cb3","#447ab3","#4577b2","#4575b1","#4673b1","#4771b0","#486eaf","#4a6caf","#4b6aae","#4c68ad","#4e65ac","#4f63ab","#5161aa","#525fa9","#545ca8","#555aa7","#5758a6","#5956a5","#5b53a4","#5c51a3","#5e4fa2"]
spectral_hex = spectral_hex_full[30:231]

def hex_to_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

spectral_colors_list = [hex_to_rgb(c) for c in spectral_hex]
n_colors = len(spectral_colors_list)
spectral_lut = ti.Vector.field(3, ti.f32, shape=n_colors)
for i, rgb in enumerate(spectral_colors_list):
    spectral_lut[i] = ti.Vector(rgb)

# ── 预构建 LUT 初始化消息（连接时一次性发送给浏览器）──
_lut_flat = []
for _rgb in spectral_colors_list:
    _lut_flat.extend([round(_rgb[0]*255), round(_rgb[1]*255), round(_rgb[2]*255)])
LUT_INIT_MSG = json.dumps({
    "type": "lut_init",
    "grid_w": n_x,
    "grid_h": n_y,
    "tmin":   float(t_min),
    "tmax":   float(t_max),
    "lut":    _lut_flat          # flat uint8 RGB array, len = n_colors * 3
})
print(f"[Init] LUT message size: {len(LUT_INIT_MSG)//1024} KB, {n_colors} colors")


async def ws_handler(websocket):
    # 先发 LUT（浏览器需要它才能初始化 WebGL 纹理）
    try:
        await websocket.send(LUT_INIT_MSG)
    except Exception as e:
        print(f"[WS] LUT send failed: {e}")
        return

    with clients_lock:
        connected_clients.add(websocket)
    print(f"[WS] 客户端连接: {websocket.remote_address}  在线: {len(connected_clients)}")
    try:
        async for message in websocket:
            if isinstance(message, str):
                handle_ws_message(message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with clients_lock:
            connected_clients.discard(websocket)
        print(f"[WS] 客户端断开  在线: {len(connected_clients)}")


async def frame_broadcaster():
    while True:
        await asyncio.sleep(1 / 30)
        frame = None
        with latest_frame_lock:
            frame = latest_frame_bytes
        if frame is None:
            continue
        with clients_lock:
            targets = list(connected_clients)
        if not targets:
            continue
        dead = []
        for ws in targets:
            try:
                await ws.send(frame)
            except Exception:
                dead.append(ws)
        if dead:
            with clients_lock:
                for ws in dead:
                    connected_clients.discard(ws)


async def ws_server_main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] 服务器启动  ws://{WS_HOST}:{WS_PORT}")
        await frame_broadcaster()


def ws_thread_func():
    asyncio.run(ws_server_main())


# ──────────────────────────────────────────────
#  Taichi 场
# ──────────────────────────────────────────────
gradient_magnitude = ti.field(ti.f32, shape=(n_y, n_x))
gradient_x         = ti.field(ti.f32, shape=(n_y, n_x))
gradient_y         = ti.field(ti.f32, shape=(n_y, n_x))

n_total = n_x * n_y
t_n     = ti.field(ti.f32, shape=n_total)
t_np1   = ti.field(ti.f32, shape=n_total)

input_data      = ti.field(ti.f32, shape=(INPUT_HEIGHT, INPUT_WIDTH))
processed_input = ti.field(ti.f32, shape=(n_y, n_x))

data_lock          = Lock()
new_data_available = False
has_touch_current  = False
last_touch_nd      = None          # 保存最后一帧有触摸的输入数据
had_touch_prev     = False         # 用于后端检测抬手瞬间
peak_touch_nd      = None          # 保存接触面积最大时的输入数据
peak_touch_area    = 0             # 当前触摸周期的峰值面积

precomputed_solver = None

# ──────────────────────────────────────────────
#  梯度插值 + 粒子系统
# ──────────────────────────────────────────────
def bilinear_interpolate_gradient(grad_x_np, grad_y_np, grid_x, grid_y):
    x0 = int(np.floor(grid_x)); y0 = int(np.floor(grid_y))
    x1 = x0 + 1;                y1 = y0 + 1
    x0 = max(0, min(x0, n_x-1)); x1 = max(0, min(x1, n_x-1))
    y0 = max(0, min(y0, n_y-1)); y1 = max(0, min(y1, n_y-1))
    fx = grid_x - int(np.floor(grid_x))
    fy = grid_y - int(np.floor(grid_y))
    gx = ((1-fx)*(1-fy)*grad_x_np[y0,x0] + fx*(1-fy)*grad_x_np[y0,x1]
         +(1-fx)*fy*grad_x_np[y1,x0]      + fx*fy*grad_x_np[y1,x1])
    gy = ((1-fx)*(1-fy)*grad_y_np[y0,x0] + fx*(1-fy)*grad_y_np[y0,x1]
         +(1-fx)*fy*grad_y_np[y1,x0]      + fx*fy*grad_y_np[y1,x1])
    return gx, gy


class FlowParticle:
    def __init__(self, x, y, life):
        self.x = x; self.y = y
        self.life = life; self.max_life = life
        self.trail = [(x, y)]
        self.dying = False; self.fade_alpha = 1.0; self.fade_speed = 0.15

    def update(self, grad_x_np, grad_y_np, min_mag):
        if self.dying:
            self.fade_alpha -= self.fade_speed
            if len(self.trail) > 1 and self.fade_alpha < 0.7:
                self.trail.pop(0)
            return
        gx, gy = bilinear_interpolate_gradient(grad_x_np, grad_y_np, self.x/scatter, self.y/scatter)
        mag = np.sqrt(gx*gx + gy*gy)
        if mag > min_mag:
            speed = min(mag * particle_speed, particle_speed * 3.0)
            self.x -= (gx/mag)*speed; self.y -= (gy/mag)*speed
            self.trail.append((self.x, self.y))
            if len(self.trail) > particle_trail_length: self.trail.pop(0)
        else:
            self.dying = True
        if not (0 <= self.x < res_x and 0 <= self.y < res_y): self.dying = True
        self.life -= 1
        if self.life <= 0: self.dying = True

    def is_alive(self):
        return (self.fade_alpha > 0 and len(self.trail) > 1) if self.dying else (0 <= self.x < res_x and 0 <= self.y < res_y)

    def seg_alpha(self, idx, total):
        return self.fade_alpha if total <= 1 else (0.2 + 0.8*(idx/(total-1)))*self.fade_alpha


class ParticleSystem:
    def __init__(self):
        self.particles=[]; self.warmup_frames=0
        self.warmup_duration=90; self.warmup_delay=15

    def reset(self): self.particles=[]; self.warmup_frames=0

    def spawn(self, grad_mag_np, num, max_count):
        if len(self.particles) >= max_count: self.warmup_frames = self.warmup_duration; return
        if self.warmup_frames < self.warmup_delay: self.warmup_frames += 1; return
        prob = ((self.warmup_frames-self.warmup_delay)/(self.warmup_duration-self.warmup_delay))**2 \
               if self.warmup_frames < self.warmup_duration else 1.0
        if self.warmup_frames < self.warmup_duration: self.warmup_frames += 1
        idx_list = np.argwhere(grad_mag_np > gradient_line_min_magnitude)
        if len(idx_list) == 0: return
        for _ in range(num):
            if len(self.particles) >= max_count: break
            if random.random() < prob:
                i,j = random.choice(idx_list)
                x = (j + random.random())*scatter; y = (i + random.random())*scatter
                life = random.randint(particle_min_life, particle_max_life)
                if prob < 1.0: life -= random.randint(0, life//2)
                self.particles.append(FlowParticle(x, y, life))

    def update(self, gx_np, gy_np, gm_np):
        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles: p.update(gx_np, gy_np, gradient_line_min_magnitude)
        self.spawn(gm_np, particle_spawn_rate, max_particles)


particle_system = ParticleSystem()


# ──────────────────────────────────────────────
#  Taichi 核函数
# ──────────────────────────────────────────────
@ti.func
def ind(i, j): return i * n_x + j


@ti.kernel
def fillD(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(n_y, n_x):
        cnt = 0
        if i-1>=0:   A[ind(i,j),ind(i-1,j)]+=1; cnt+=1
        if i+1<n_y:  A[ind(i,j),ind(i+1,j)]+=1; cnt+=1
        if j-1>=0:   A[ind(i,j),ind(i,j-1)]+=1; cnt+=1
        if j+1<n_x:  A[ind(i,j),ind(i,j+1)]+=1; cnt+=1
        A[ind(i,j),ind(i,j)] += -cnt


@ti.kernel
def fillI(A: ti.types.sparse_matrix_builder()):
    for i, j in ti.ndrange(n_y, n_x):
        A[ind(i,j),ind(i,j)] += 1


def buildMatrices():
    global precomputed_solver
    Db = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total*5)
    Ib = ti.linalg.SparseMatrixBuilder(n_total, n_total, max_num_triplets=n_total)
    fillD(Db); fillI(Ib)
    D = Db.build(); I = Ib.build()
    c = h * k / dx**2
    M = I - c * D
    precomputed_solver = ti.linalg.SparseSolver(solver_type="LLT")
    precomputed_solver.analyze_pattern(M)
    precomputed_solver.factorize(M)
    print(f"矩阵预计算完成  k={k:.0f}  c={c:.4f}")


@ti.kernel
def init_fields():
    for i, j in ti.ndrange(n_y, n_x):
        t_n[ind(i,j)] = t_min; t_np1[ind(i,j)] = t_min
        processed_input[i,j] = 0.0
        gradient_magnitude[i,j] = 0.0
        gradient_x[i,j] = 0.0; gradient_y[i,j] = 0.0


@ti.kernel
def reset_temperature_to_ambient(ambient: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        t_n[ind(i,j)] = ambient; t_np1[ind(i,j)] = ambient


@ti.kernel
def compute_gradients():
    for i, j in ti.ndrange(n_y, n_x):
        gx = (t_np1[ind(i,j+1)]-t_np1[ind(i,j-1)])/2.0 if 0<j<n_x-1 else \
             (t_np1[ind(i,j+1)]-t_np1[ind(i,j)] if j==0 else t_np1[ind(i,j)]-t_np1[ind(i,j-1)])
        gy = (t_np1[ind(i+1,j)]-t_np1[ind(i-1,j)])/2.0 if 0<i<n_y-1 else \
             (t_np1[ind(i+1,j)]-t_np1[ind(i,j)] if i==0 else t_np1[ind(i,j)]-t_np1[ind(i-1,j)])
        gradient_x[i,j]=gx; gradient_y[i,j]=gy
        gradient_magnitude[i,j]=ti.sqrt(gx*gx+gy*gy)


@ti.kernel
def process_input_kernel(threshold: ti.f32, scale: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        processed_input[i,j] = 0.0
    for i, j in ti.ndrange(n_y, n_x):
        v = input_data[n_y-1-i, j]
        if v < threshold:
            processed_input[i,j] = (threshold - v) * scale


@ti.kernel
def update_temp_kernel(ambient: ti.f32, cool: ti.f32):
    for i, j in ti.ndrange(n_y, n_x):
        cooled = t_np1[ind(i,j)] + (ambient - t_np1[ind(i,j)]) * cool
        hi = processed_input[i,j]
        if hi > 0:
            # 压力线性映射：轻触≈环境温度，按实→体温37°C
            blend = ti.min(hi * 10.0, 1.0)
            target = ambient + blend * (37.0 - ambient)
            t_np1[ind(i,j)] = cooled * 0.7 + target * 0.3
        else:
            t_np1[ind(i,j)] = cooled


# ──────────────────────────────────────────────
#  帧编码（核心改动）：裸温度 + 粒子线段，无 JPEG
# ──────────────────────────────────────────────
def encode_temp_frame():
    """
    Binary layout:
      [4B]  n_segs     uint32  粒子线段数量
      [4B]  brightness float32
      [1B]  has_touch  uint8   当前帧是否有触摸 (0/1)
      [1B]  progress   uint8   长按冻结进度 (0-255)
      [n_y * n_x * 4B] temperature float32 row-major (row0 = i=0 = Taichi bottom)
      [n_segs * 20B]   segments: x0,y0,x1,y1,alpha float32 (normalized 0-1, y: 0=bottom)
    Total typical: 10 + 16224 + ~500*20 ≈ 26 KB
    """
    temp_data = t_np1.to_numpy().reshape(n_y, n_x).astype(np.float32)

    segs = []
    if show_gradient_lines:
        for p in particle_system.particles:
            trail = p.trail
            n_pts = len(trail)
            for idx in range(n_pts - 1):
                x0, y0 = trail[idx]
                x1, y1 = trail[idx + 1]
                a = float(p.seg_alpha(idx, n_pts - 1))
                # normalize: x→[0,1] left-right, y→[0,1] bottom-top
                segs.append((float(x0)/res_x, float(y0)/res_y,
                             float(x1)/res_x, float(y1)/res_y, a))

    n_s = len(segs)
    with data_lock:
        touch_byte = 1 if has_touch_current else 0
        progress_byte = int(freeze_progress * 255)
    header   = struct.pack('<IfBB', n_s, brightness_scale, touch_byte, progress_byte)
    t_bytes  = temp_data.tobytes()
    s_bytes  = np.array(segs, dtype=np.float32).tobytes() if n_s > 0 else b''
    return header + t_bytes + s_bytes


def read_shared_memory():
    global new_data_available, has_touch_current, last_touch_nd
    global had_touch_prev, freeze_input
    global peak_touch_nd, peak_touch_area
    global longpress_start, freeze_progress
    HEADER_SIZE = 12
    mmf = None
    try:
        import os
        if os.name == 'nt':
            try: mmf = mmap.mmap(0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
            except Exception as e: print(f"共享内存失败: {e}"); return
        else:
            fp = f"/tmp/{SHARED_MEMORY_NAME}"
            if not os.path.exists(fp): print(f"共享内存文件不存在: {fp}"); return
            with open(fp,'r+b') as f: mmf = mmap.mmap(f.fileno(), SHARED_MEMORY_SIZE, access=mmap.ACCESS_READ)
        print("成功连接到共享内存")
        while True:
            try:
                mmf.seek(HEADER_SIZE)
                raw = mmf.read(INPUT_FRAME_SIZE)
                if len(raw) >= INPUT_FRAME_SIZE:
                    raw_uint8 = np.frombuffer(raw[:INPUT_FRAME_SIZE], dtype=np.uint8).copy()
                    contact_area = int(np.sum(raw_uint8 < input_threshold))
                    touch_detected = contact_area > 5
                    nd = raw_uint8.reshape(INPUT_HEIGHT,INPUT_WIDTH).astype(np.float32)/255.0
                    with data_lock:
                        has_touch_current = touch_detected

                        if freeze_input:
                            # 已冻结：只更新触摸状态（前端用来检测恢复）
                            pass
                        elif touch_detected:
                            # 正常更新输入
                            try:
                                input_data.from_numpy(nd); new_data_available = True
                            except: pass
                            # 追踪峰值接触面积
                            if contact_area >= peak_touch_area * 0.9:
                                peak_touch_nd = nd.copy()
                                peak_touch_area = contact_area
                            # 长按计时
                            if longpress_start is None:
                                longpress_start = time.time()
                            elapsed = time.time() - longpress_start
                            freeze_progress = min(1.0, elapsed / LONGPRESS_DURATION)
                            # 达到时长：冻结，恢复峰值输入
                            if freeze_progress >= 1.0:
                                freeze_input = True
                                freeze_progress = 1.0
                                if peak_touch_nd is not None:
                                    try:
                                        input_data.from_numpy(peak_touch_nd)
                                        new_data_available = True
                                    except: pass
                        else:
                            # 触摸消失但未冻结：重置计时
                            try:
                                input_data.from_numpy(nd); new_data_available = True
                            except: pass
                            longpress_start = None
                            freeze_progress = 0.0
                            peak_touch_area = 0

                        had_touch_prev = touch_detected
                time.sleep(1/input_update_rate)
            except: time.sleep(0.1)
    except Exception as e: print(f"共享内存初始化失败: {e}")
    finally:
        if mmf: mmf.close()


# ──────────────────────────────────────────────
#  启动
# ──────────────────────────────────────────────
init_fields()
buildMatrices()

threading.Thread(target=read_shared_memory, daemon=True).start()
threading.Thread(target=ws_thread_func, daemon=True).start()

print("\n" + "="*60)
print(f"WebSocket 帧流服务器: ws://{WS_HOST}:{WS_PORT}")
print("在浏览器中打开 ui_stream.html")
print("按 Ctrl+C 退出")
print("="*60 + "\n")

frame_count    = 0
last_fps_time  = time.time()
last_frame_enc = time.time()
FPS_ENCODE_CAP = 60

try:
    while True:
        # ── flag 处理 ──
        with rebuild_matrix_lock:
            if rebuild_matrix_flag:
                rebuild_matrix_flag = False
                particle_system.reset(); buildMatrices()

        with reset_ambient_lock:
            if reset_ambient_flag:
                reset_ambient_flag = False
                reset_temperature_to_ambient(float(reset_ambient_value))
                particle_system.reset()

        with reset_sim_lock:
            if reset_sim_flag:
                reset_sim_flag = False
                init_fields(); particle_system.reset()

        # ── 处理触摸输入 ──
        with data_lock:
            do_input = new_data_available
            if do_input: new_data_available = False
        if do_input:
            process_input_kernel(input_threshold/255.0, heat_intensity_scale)

        # ── 仿真步进 ──
        if not paused:
            for _ in range(substep):
                t_np1.from_numpy(precomputed_solver.solve(t_n))
                update_temp_kernel(t_ambient, cooling_rate)
                t_n.copy_from(t_np1)
            compute_gradients()

        # ── 粒子物理更新（不再 draw，只更新位置供编码）──
        if show_gradient_lines and not paused:
            gm = gradient_magnitude.to_numpy()
            gx = gradient_x.to_numpy()
            gy = gradient_y.to_numpy()
            particle_system.update(gx, gy, gm)

        # ── 编码温度帧（无 JPEG，直接打包 float32）──
        now = time.time()
        if now - last_frame_enc >= 1/FPS_ENCODE_CAP:
            frame_bytes = encode_temp_frame()
            with latest_frame_lock:
                latest_frame_bytes = frame_bytes
            last_frame_enc = now

        # ── FPS 统计 ──
        frame_count += 1
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            with clients_lock:
                nc = len(connected_clients)
            t_arr = t_np1.to_numpy()
            print(f"FPS:{fps:.0f}  clients:{nc}  T:[{t_arr.min():.1f}, {t_arr.max():.1f}]  ambient={t_ambient:.0f}"
                  f"  {current_material_name} | {current_ambient_name} | {current_cooling_name}")
            frame_count = 0; last_fps_time = now

except KeyboardInterrupt:
    print("\n退出")