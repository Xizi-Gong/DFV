import os
import io
import json
import math
import mmap
import time
import queue
import asyncio
import threading
import argparse
from threading import Lock

import numpy as np
import taichi as ti
import seaborn as sns
import cv2
import perlin_noise
import websockets
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
#  全局配置与常量
# ──────────────────────────────────────────────
res_x = 585
res_y = 390
dt = 0.01

fluid_properties = {
    "water": {"rho": 1000.0, "viscosity": 1.0e-6, "u_in_default": 1000.0},
    "air":   {"rho": 1.2,    "viscosity": 1.5e-5,  "u_in_default": 1000.0},
    "honey": {"rho": 1420.0, "viscosity": 7.0e-3,  "u_in_default": 1000.0},
}

# 速度扩散（隐式粘性）迭代次数，按材质区分。
# 空气接近无粘 → 旋涡不被涂抹，卡门涡街清晰；蜂蜜高粘 → 强阻尼。
# 参考 Matthias Müller 的无粘 Stable-Fluids 设计：低粘度才能保留涡街结构。
VELOCITY_DIFFUSION_ITERS = {"air": 2, "water": 6, "honey": 20}

p_iters_map = {"water": 500, "air": 200, "honey": 100}
PIXEL_SIZE_MAP = {
    "air":   1.0e-3,
    "water": 3.0e-3,
    "honey": 1.0e-3,
}

P_SOR_CONFIG = {
    "air":   {"omega": 1.6, "n_iter": 50},
    "water": {"omega": 1.5, "n_iter": 80},
    "honey": {"omega": 1.7, "n_iter": 30},
}

fluid_material = "air"
PIXEL_SIZE = PIXEL_SIZE_MAP[fluid_material]
U_IN       = fluid_properties[fluid_material]["u_in_default"]
RHO        = fluid_properties[fluid_material]["rho"]
VISCOSITY  = fluid_properties[fluid_material]["viscosity"]

# 可视化与求解器参数
streamline_step   = 20
STREAMLINE_LENGTH = 0.03
STREAMLINE_START  = streamline_step // 2
STREAMLINE_COLS   = ((res_x - 1 - STREAMLINE_START) // streamline_step) + 1
STREAMLINE_ROWS   = ((res_y - 1 - STREAMLINE_START) // streamline_step) + 1
STREAMLINE_COUNT  = STREAMLINE_COLS * STREAMLINE_ROWS
# 密集细烟丝(streakline)，模拟风洞烟线法：等距细线，靠近圆柱的几条被卷进剪切层→脱涡，
# 涡街的滚卷结构以"多根细线卷进涡心"的方式清晰呈现（比少数粗带清楚得多）。
smoke_positions   = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90]
smoke_width       = [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
SMOKE_PULSE_PERIOD = 0.36
SMOKE_PULSE_AMPLITUDE = 0.05   # 原0.18脉冲周期≈30帧，与脱涡周期接近会叠出"循环"错觉，调小
p_jacobi_iters    = 200
curl_strength     = 3.5
time_c            = 5.0
maxfps            = 60
dye_decay         = 1 - 1 / (maxfps * time_c)
debug             = False
# 物理自洽：求解器速度 = 真实 m/s × (1/PIXEL_SIZE)，使 slider 直接为真实速度、Re 自洽。
# 1000 = 1/PIXEL_SIZE(air=1e-3)；air/honey 精确，water(3e-3)偏 3×(其 Re 本就极高、无层流街，可忽略)。
SOLVER_SPEED_PER_MPS = 1000.0
WIND_SPEED_MIN_MPS   = 0.01   # 真实 m/s：0.01→Re~27(稳态)
WIND_SPEED_MAX_MPS   = 2.0    # 视觉速度峰值在 ~1.0–1.5 m/s(穿屏~1.7s)；>1.5 fps 崩、尾流变乱
# 涡量增强后的速度安全钳制。必须显著高于最大入流速度(=WIND_SPEED_MAX*SOLVER_SPEED)，
# 否则会把高速流场削平 → 滑块失效、烟雾不随速度加快。给 2.5x 余量兜住真正的数值发散。
VELOCITY_CLAMP = WIND_SPEED_MAX_MPS * SOLVER_SPEED_PER_MPS * 2.5
WIND_SPEED_STEP_MPS  = 0.005
MAX_ADVECT_CFL_PX    = 2.5    # BFECC(低耗散)下每子步~2.5px 仍保清晰涡街（半拉格朗日需~1.5px）
MAX_FLUID_SUBSTEPS   = 16     # → 高速时子步更少 = 更快，且烟雾标记更锐利（BFECC）
MIN_SOR_ITERS_PER_SUBSTEP = 12
CYLINDER_RADIUS_PX = 20.0   # 直径↓→涡街波长(λ≈5.3D)↓→尾流区能容纳2~3个涡，才看得出"街"而非单涡循环
MAX_SMOKE_CFL_PX = 8.0
MAX_SMOKE_SUBSTEPS_PER_FLUID = 3

# 共享内存配置
SHARED_MEMORY_NAME = "shared_touch_image"
SHARED_MEMORY_SIZE = 4068
INPUT_WIDTH        = 78
INPUT_HEIGHT       = 52
INPUT_FRAME_SIZE   = INPUT_WIDTH * INPUT_HEIGHT
HEADER_SIZE        = 12

LONGPRESS_DURATION = 5.0
LONGPRESS_PRESSURE_THRESHOLD = 0.35

ti.init(arch=ti.gpu)

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt
    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


# ──────────────────────────────────────────────
#  Taichi 场变量声明
# ──────────────────────────────────────────────
_velocities       = ti.Vector.field(2, float, shape=(res_x, res_y))
_new_velocities   = ti.Vector.field(2, float, shape=(res_x, res_y))
velocity_divs     = ti.field(float, shape=(res_x, res_y))
velocity_curls    = ti.field(float, shape=(res_x, res_y))
_pressures        = ti.field(float, shape=(res_x, res_y))
_new_pressures    = ti.field(float, shape=(res_x, res_y))
_dye_buffer       = ti.Vector.field(2, float, shape=(res_x, res_y))
_new_dye_buffer   = ti.Vector.field(2, float, shape=(res_x, res_y))
_smoke            = ti.field(float, shape=(res_x, res_y))
_new_smoke        = ti.field(float, shape=(res_x, res_y))
_solid            = ti.field(float, shape=(res_x, res_y))   # 1=solid, 0=fluid
color_field       = ti.Vector.field(3, float, shape=(res_x, res_y))
streamline_start_field = ti.Vector.field(2, float, shape=STREAMLINE_COUNT)
streamline_dir_field   = ti.Vector.field(2, float, shape=STREAMLINE_COUNT)
streamline_mag_field   = ti.field(float, shape=STREAMLINE_COUNT)

# 渲染专用场
_rgb_dye  = ti.Vector.field(3, float, shape=(res_x, res_y))
_nrgb_dye = ti.Vector.field(3, float, shape=(res_x, res_y))

# BFECC 平流的临时缓冲（每种类型两个：前向 / 反向结果）。低耗散 → 高 CFL 仍清晰。
_bfecc_v_a = ti.Vector.field(2, float, shape=(res_x, res_y))
_bfecc_v_b = ti.Vector.field(2, float, shape=(res_x, res_y))
_bfecc_s_a = ti.field(float, shape=(res_x, res_y))
_bfecc_s_b = ti.field(float, shape=(res_x, res_y))
_bfecc_c_a = ti.Vector.field(3, float, shape=(res_x, res_y))
_bfecc_c_b = ti.Vector.field(3, float, shape=(res_x, res_y))

# 运行时物理标量
_rho_field        = ti.field(dtype=float, shape=())
_nu_field         = ti.field(dtype=float, shape=())
_pixel_size_field = ti.field(dtype=float, shape=())
noise_time_field  = ti.field(dtype=float, shape=())

_pixel_size_field[None] = PIXEL_SIZE
_rho_field[None]        = RHO
_nu_field[None]         = VISCOSITY
noise_time_field[None]  = 0.0

velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair  = TexPair(_pressures,  _new_pressures)
dyes_pair       = TexPair(_dye_buffer, _new_dye_buffer)
smoke_pair      = TexPair(_smoke,      _new_smoke)
rgb_dye_pair    = TexPair(_rgb_dye,    _nrgb_dye)


# ──────────────────────────────────────────────
#  WebSocket 状态与控制
# ──────────────────────────────────────────────
_ws_clients        = set()
_ws_clients_lock   = threading.Lock()
_latest_frame      = None          
_latest_frame_lock = threading.Lock()
_latest_state      = None
_latest_state_lock = threading.Lock()
_ws_cmd_queue      = queue.Queue()

async def _ws_handler(websocket):
    with _ws_clients_lock:
        _ws_clients.add(websocket)
    try:
        await websocket.send(json.dumps({
            "type": "init", "width": res_x, "height": res_y,
            "materials": list(fluid_properties.keys())
        }))
        with _latest_state_lock:
            state = _latest_state
        if state:
            await websocket.send(state)
        with _latest_frame_lock:
            frame = _latest_frame
        if frame:
            await websocket.send(frame)
        async for raw in websocket:
            try:
                _ws_cmd_queue.put_nowait(json.loads(raw))
            except Exception:
                pass
    except Exception:
        pass
    finally:
        with _ws_clients_lock:
            _ws_clients.discard(websocket)

async def _ws_broadcast(data):
    with _ws_clients_lock:
        clients = list(_ws_clients)
    if clients:
        await asyncio.gather(*[c.send(data) for c in clients], return_exceptions=True)

async def _ws_main():
    async with websockets.serve(_ws_handler, "localhost", 8765):
        print("[WS] 服务器启动 ws://localhost:8765")
        await asyncio.Future()

_ws_loop   = asyncio.new_event_loop()
_ws_thread = threading.Thread(target=lambda: _ws_loop.run_until_complete(_ws_main()), daemon=True)

def _broadcast_sync(data):
    if _ws_clients:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(data), _ws_loop)


# ──────────────────────────────────────────────
#  共享内存提取
# ──────────────────────────────────────────────
shared_mask_lock   = Lock()
shared_mask_np     = None
new_mask_available = False
last_lowres_mask_np = None

freeze_input = False
longpress_start = None
freeze_progress = 0.0
has_touch_current = False
frozen_mask_np = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=bool)
longpress_latched = False

def release_frozen_input():
    global freeze_input, longpress_start, freeze_progress, longpress_latched
    with shared_mask_lock:
        frozen_mask_np.fill(False)
        freeze_input = False
        longpress_start = None
        freeze_progress = 0.0
        longpress_latched = False

def read_shared_memory():
    """后台线程读取 MMap 中的触摸数据"""
    global shared_mask_np, new_mask_available, freeze_input
    global longpress_start, freeze_progress, has_touch_current
    global frozen_mask_np, longpress_latched
    mmf = None
    try:
        if os.name == 'nt':
            mmf = mmap.mmap(0, SHARED_MEMORY_SIZE, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
        else:
            fp = f"/tmp/{SHARED_MEMORY_NAME}"
            if not os.path.exists(fp):
                print(f"[SharedMem] 共享内存文件不存在: {fp}")
                return
            with open(fp, 'r+b') as f:
                mmf = mmap.mmap(f.fileno(), SHARED_MEMORY_SIZE, access=mmap.ACCESS_READ)
        
        print("[SharedMem] 成功连接到共享内存")
        while True:
            try:
                mmf.seek(HEADER_SIZE)
                raw = mmf.read(INPUT_FRAME_SIZE)
                if len(raw) >= INPUT_FRAME_SIZE:
                    nd = np.frombuffer(raw[:INPUT_FRAME_SIZE], dtype=np.uint8).copy().reshape(INPUT_HEIGHT, INPUT_WIDTH)
                    flat = nd.reshape(-1)
                    contact = flat < 160
                    contact_grid = contact.reshape(INPUT_HEIGHT, INPUT_WIDTH)
                    contact_area = int(np.sum(contact))
                    touching = contact_area > 5
                    if contact_area:
                        depth = 160.0 - flat[contact].astype(np.float32)
                        pressure = float(np.percentile(depth, 90)) / 160.0
                    else:
                        pressure = 0.0
                    with shared_mask_lock:
                        has_touch_current = touching
                        if longpress_latched and np.count_nonzero(contact_grid & ~frozen_mask_np) > 5:
                            longpress_latched = False
                        if touching and pressure >= LONGPRESS_PRESSURE_THRESHOLD and not longpress_latched:
                            if longpress_start is None:
                                longpress_start = time.time()
                            freeze_progress = min(1.0, (time.time() - longpress_start) / LONGPRESS_DURATION)
                            if freeze_progress >= 1.0:
                                frozen_mask_np |= contact_grid
                                freeze_input = True
                                longpress_latched = True
                                longpress_start = None
                                freeze_progress = 0.0
                        elif not touching or pressure < LONGPRESS_PRESSURE_THRESHOLD:
                            longpress_start = None
                            freeze_progress = 0.0
                            longpress_latched = False
                        combined = nd.copy()
                        combined[frozen_mask_np] = 0
                        shared_mask_np = combined
                        new_mask_available = True
                time.sleep(1 / 30)
            except Exception:
                time.sleep(0.1)
    except Exception as e:
        print(f"[SharedMem] 初始化失败: {e}")
    finally:
        if mmf: mmf.close()

def reset_mask_change_cache():
    global last_lowres_mask_np
    last_lowres_mask_np = None

def process_mask_update(threshold=200.0, mode="threshold"):
    """主线程调用，处理触摸数据并映射到障碍物 Taichi field"""
    global new_mask_available, shared_mask_np, last_lowres_mask_np
    with shared_mask_lock:
        if not new_mask_available or shared_mask_np is None:
            return False
        arr = shared_mask_np.copy()
        new_mask_available = False

    if mode == "normalize":
        amin, amax = float(arr.min()), float(arr.max())
        arr_norm_low = (arr.astype(np.float32) - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=np.float32)
        thr = threshold if threshold <= 1.0 else threshold / 255.0
        lowres_mask = arr_norm_low > thr
    else:
        lowres_mask = arr < threshold

    if last_lowres_mask_np is not None and np.array_equal(lowres_mask, last_lowres_mask_np):
        return False
    last_lowres_mask_np = lowres_mask.copy()

    img = Image.fromarray(arr).resize((res_x, res_y), resample=Image.BICUBIC)
    arr_resized = np.array(img, copy=False)

    if mode == "normalize":
        amin, amax = float(arr_resized.min()), float(arr_resized.max())
        arr_norm = (arr_resized.astype(np.float32) - amin) / (amax - amin) if amax > amin else np.zeros_like(arr_resized, dtype=np.float32)
        thr = threshold if threshold <= 1.0 else threshold / 255.0
        mask_np = (arr_norm > thr).astype(np.float32)
    else:
        mask_np = (arr_resized.astype(np.float32) < threshold).astype(np.float32)

    _solid.from_numpy(np.fliplr(mask_np.T))
    return True


# ──────────────────────────────────────────────
#  辅助函数
# ──────────────────────────────────────────────
def rgb_array_to_hex(rgba_array):
    r = (rgba_array[:, 0] * 255).astype(np.uint32)
    g = (rgba_array[:, 1] * 255).astype(np.uint32)
    b = (rgba_array[:, 2] * 255).astype(np.uint32)
    return (r << 16) | (g << 8) | b

def get_mask_span_x(mask):
    nz = np.nonzero(mask)
    return float(nz[0].max() - nz[0].min()) if len(nz[0]) > 0 else 0.0

@ti.kernel
def sample_streamline_arrows(vf: ti.template(), solidf: ti.template()):
    for idx in streamline_start_field:
        col = idx // STREAMLINE_ROWS
        row = idx - col * STREAMLINE_ROWS
        i = STREAMLINE_START + col * streamline_step
        j = STREAMLINE_START + row * streamline_step

        direction = ti.Vector([0.0, 0.0])
        mag = 0.0
        if solidf[i, j] == 0:
            vel = vf[i, j]
            mag = vel.norm()
            if mag > 1e-6:
                direction = vel
                direction.x /= ti.cast(res_x, ti.f32) / STREAMLINE_LENGTH
                direction.y /= ti.cast(res_y, ti.f32) / STREAMLINE_LENGTH
                length = direction.norm()
                max_len = STREAMLINE_LENGTH * 0.8
                if length > max_len:
                    direction *= max_len / length

        streamline_start_field[idx] = ti.Vector([
            ti.cast(i, ti.f32) / ti.cast(res_x, ti.f32),
            ti.cast(j, ti.f32) / ti.cast(res_y, ti.f32),
        ])
        streamline_dir_field[idx] = direction
        streamline_mag_field[idx] = mag

def calculate_streamline(vf, solidf):
    sample_streamline_arrows(vf, solidf)
    return (
        streamline_dir_field.to_numpy(),
        streamline_start_field.to_numpy(),
        streamline_mag_field.to_numpy(),
    )


# ──────────────────────────────────────────────
#  Taichi 核函数：求解器与采样
# ──────────────────────────────────────────────
@ti.func
def solid_at(i, j):
    is_solid = 0.0
    if   i < 0 or i >= res_x: is_solid = 0.0
    elif j < 0 or j >= res_y: is_solid = 1.0
    else:                     is_solid = _solid[i, j]
    return is_solid

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I.x = ti.max(0, ti.min(res_x - 1, I.x))
    I.y = ti.max(0, ti.min(res_y - 1, I.y))
    return qf[I]

@ti.func
def sample_fluid(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I.x = ti.max(0, ti.min(res_x - 1, I.x))
    I.y = ti.max(0, ti.min(res_y - 1, I.y))
    return qf[I] * (1.0 - ti.cast(_solid[I], ti.f32))

@ti.func
def lerp(vl, vr, frac): return vl + frac * (vr - vl)

@ti.func
def bilerp(vf, p):
    u, v = p; s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    return lerp(lerp(sample(vf, iu, iv), sample(vf, iu+1, iv), fu),
                lerp(sample(vf, iu, iv+1), sample(vf, iu+1, iv+1), fu), fv)

@ti.func
def bilerp_fluid(vf, p):
    u, v = p; s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    return lerp(lerp(sample_fluid(vf, iu, iv), sample_fluid(vf, iu+1, iv), fu),
                lerp(sample_fluid(vf, iu, iv+1), sample_fluid(vf, iu+1, iv+1), fu), fv)

@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp_fluid(vf, p);  p1 = p - 0.5  * dt_ * v1
    v2 = bilerp_fluid(vf, p1); p2 = p - 0.75 * dt_ * v2
    v3 = bilerp_fluid(vf, p2)
    p -= dt_ * ((2/9)*v1 + (1/3)*v2 + (4/9)*v3)
    p.x = ti.max(0.5, ti.min(res_x - 1.5, p.x))
    p.y = ti.max(0.5, ti.min(res_y - 1.5, p.y))
    return p

@ti.kernel
def diffuse_velocity(vf: ti.template(), new_vf: ti.template(), dt_step: ti.f32):
    px = _pixel_size_field[None]
    alpha = (px * px) / (_nu_field[None] * dt_step)
    for i, j in vf:
        if _solid[i, j] == 1:
            new_vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            vl = sample(vf, i-1, j); vr = sample(vf, i+1, j)
            vb = sample(vf, i, j-1); vt = sample(vf, i, j+1)
            new_vf[i, j] = (vf[i,j]*alpha + vl + vr + vb + vt) / (4.0 + alpha)

@ti.kernel
def advect_velocity(vf: ti.template(), new_vf: ti.template(), dt_step: ti.f32):
    for i, j in vf:
        if _solid[i, j] == 1:
            new_vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            p = ti.Vector([i, j]) + 0.5
            new_vf[i, j] = bilerp_fluid(vf, backtrace(vf, p, dt_step))

@ti.kernel
def advect_passive(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), dt_step: ti.f32, decay: ti.f32):
    for i, j in vf:
        if _solid[i, j] == 1:
            new_qf[i, j] = qf[i, j] * 0
        else:
            p = ti.Vector([i, j]) + 0.5
            new_qf[i, j] = bilerp_fluid(qf, backtrace(vf, p, dt_step)) * decay

@ti.kernel
def advect_uv(vf: ti.template(), qf: ti.template(), new_qf: ti.template(), dt_step: ti.f32, decay: ti.f32):
    for i, j in vf:
        if _solid[i, j] == 1:
            new_qf[i, j] = qf[i, j] * 0
        else:
            p = ti.Vector([i, j]) + 0.5
            new_qf[i, j] = bilerp(qf, backtrace(vf, p, dt_step)) * decay

# ── BFECC 平流（二阶、低数值耗散）─────────────────────────────────
# 比双线性半拉格朗日耗散低得多 → 高 CFL 下仍保持参考级清晰涡街，
# 因此可以减少子步数 → 更快 + 保持清晰，且不改物理标定（自洽）。
@ti.kernel
def _bfecc_sl(vf: ti.template(), qf: ti.template(), out: ti.template(), dt_step: ti.f32):
    # 单步半拉格朗日前向平流（无衰减、无限幅），bilerp_fluid 尊重固体边界
    for i, j in qf:
        if _solid[i, j] == 1:
            out[i, j] = qf[i, j] * 0
        else:
            p = ti.Vector([i, j]) + 0.5
            out[i, j] = bilerp_fluid(qf, backtrace(vf, p, dt_step))

@ti.kernel
def _bfecc_combine(cur: ti.template(), q2: ti.template(), qhat: ti.template()):
    # 误差补偿：qhat = cur + 0.5*(cur - q2)
    for i, j in cur:
        qhat[i, j] = cur[i, j] + 0.5 * (cur[i, j] - q2[i, j])

@ti.kernel
def _bfecc_final(vf: ti.template(), qhat: ti.template(), q_orig: ti.template(),
                 out: ti.template(), dt_step: ti.f32, decay: ti.f32, do_clamp: ti.template()):
    # 用补偿后的 qhat 再前向平流。do_clamp=True 时限幅到原场局部 min/max
    # （标量烟雾需要，防过冲/负值）；速度场 do_clamp=False（限幅会在涡心削掉锐度）。
    for i, j in qhat:
        if _solid[i, j] == 1:
            out[i, j] = qhat[i, j] * 0
        else:
            p = ti.Vector([i, j]) + 0.5
            src = backtrace(vf, p, dt_step)
            val = bilerp_fluid(qhat, src)
            if ti.static(do_clamp):
                s = src[0] - 0.5
                t = src[1] - 0.5
                iu = ti.floor(s)
                iv = ti.floor(t)
                c00 = sample_fluid(q_orig, iu,     iv)
                c10 = sample_fluid(q_orig, iu + 1, iv)
                c01 = sample_fluid(q_orig, iu,     iv + 1)
                c11 = sample_fluid(q_orig, iu + 1, iv + 1)
                lo = ti.min(ti.min(c00, c10), ti.min(c01, c11))
                hi = ti.max(ti.max(c00, c10), ti.max(c01, c11))
                val = ti.max(lo, ti.min(hi, val))
            out[i, j] = val * decay

def bfecc_advect(vf, cur, nxt, sa, sb, dt_step, decay, clamp=True):
    """BFECC：cur --(vf, dt)--> nxt。sa/sb 为与 cur 同类型的临时缓冲。
    clamp=False 用于速度场（保留涡心锐度）；clamp=True 用于烟雾/染料（防过冲负值）。"""
    _bfecc_sl(vf, cur, sa, dt_step)                         # q1 = 前向平流
    _bfecc_sl(vf, sa, sb, -dt_step)                         # q2 = 把 q1 反向平流
    _bfecc_combine(cur, sb, sa)                             # qhat = cur + 0.5*(cur - q2) → 存 sa
    _bfecc_final(vf, sa, cur, nxt, dt_step, decay, clamp)   # 最终前向 (+限幅) + 衰减 → nxt

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            velocity_divs[i, j] = 0.0
        else:
            vl = sample(vf, i-1, j); vr = sample(vf, i+1, j)
            vb = sample(vf, i, j-1); vt = sample(vf, i, j+1)
            if solid_at(i-1, j) == 1: vl.x = 0.0
            if solid_at(i+1, j) == 1: vr.x = 0.0
            if solid_at(i, j-1) == 1: vb.y = 0.0
            if solid_at(i, j+1) == 1: vt.y = 0.0
            velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i-1, j); vr = sample(vf, i+1, j)
        vb = sample(vf, i, j-1); vt = sample(vf, i, j+1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5

@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    rho = _rho_field[None]
    for i, j in pf:
        if i == res_x - 1 or _solid[i, j] == 1:
            new_pf[i, j] = 0.0
        else:
            pc = pf[i, j]
            pl = sample(pf, i-1, j) if solid_at(i-1, j) == 0 else pc
            pr = sample(pf, i+1, j) if solid_at(i+1, j) == 0 else pc
            pb = sample(pf, i, j-1) if solid_at(i, j-1) == 0 else pc
            pt = sample(pf, i, j+1) if solid_at(i, j+1) == 0 else pc
            new_pf[i, j] = (pl + pr + pb + pt - velocity_divs[i, j] * rho / dt) * 0.25

@ti.kernel
def pressure_sor_pass(pf: ti.template(), parity: int, omega: float, dt_step: ti.f32):
    rho_val = _rho_field[None]
    for i, j in pf:
        if (i + j) % 2 != parity:
            continue
        if i == res_x - 1 or _solid[i, j] == 1:
            pf[i, j] = 0.0
            continue
        pc = pf[i, j]
        pl = sample(pf, i-1, j) if solid_at(i-1, j) == 0 else pc
        pr = sample(pf, i+1, j) if solid_at(i+1, j) == 0 else pc
        pb = sample(pf, i, j-1) if solid_at(i, j-1) == 0 else pc
        pt = sample(pf, i, j+1) if solid_at(i, j+1) == 0 else pc
        p_gs = (pl + pr + pb + pt - velocity_divs[i, j] * rho_val / dt_step) * 0.25
        pf[i, j] = (1.0 - omega) * pc + omega * p_gs

def solve_pressure_sor(dt_step, substeps=1, stability_blend=0.0):
    cfg = P_SOR_CONFIG.get(fluid_material, {"omega": 1.6, "n_iter": 50}) 
    if substeps <= 1:
        n_iter = cfg["n_iter"]
    else:
        base_iters = max(MIN_SOR_ITERS_PER_SUBSTEP, int(math.ceil(cfg["n_iter"] / substeps)))
        stable_fraction = 0.60 if substeps == 2 else 0.70
        stable_iters = max(base_iters, int(math.ceil(cfg["n_iter"] * stable_fraction)))
        n_iter = int(math.ceil(base_iters + (stable_iters - base_iters) * stability_blend))
    for _ in range(n_iter):
        pressure_sor_pass(pressures_pair.cur, 0, cfg["omega"], dt_step)
        pressure_sor_pass(pressures_pair.cur, 1, cfg["omega"], dt_step)

@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template(), dt_step: ti.f32):
    rho = _rho_field[None]
    for i, j in vf:
        if _solid[i, j] == 1:
            vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            pc = pf[i, j]
            pl = sample(pf, i-1, j) if solid_at(i-1, j) == 0 else pc
            pr = sample(pf, i+1, j) if solid_at(i+1, j) == 0 else pc
            pb = sample(pf, i, j-1) if solid_at(i, j-1) == 0 else pc
            pt = sample(pf, i, j+1) if solid_at(i, j+1) == 0 else pc
            vf[i, j] -= 0.5 * (dt_step / rho) * ti.Vector([pr - pl, pt - pb])

@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template(), dt_step: ti.f32, curl_scale: ti.f32):
    for i, j in vf:
        cl = sample(cf, i-1, j); cr = sample(cf, i+1, j)
        cb = sample(cf, i, j-1); ct = sample(cf, i, j+1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb), abs(cl) - abs(cr)]).normalized(1e-3) * curl_strength * curl_scale * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt_step, -VELOCITY_CLAMP), VELOCITY_CLAMP)


# ──────────────────────────────────────────────
#  Taichi 核函数：物理边界与渲染
# ──────────────────────────────────────────────
@ti.kernel
def refine_solid_mask(solidf: ti.template()):
    for i, j in solidf:
        if i == 0 or i == res_x - 1 or j == 0 or j == res_y - 1:
            solidf[i, j] = 0.0

@ti.kernel
def apply_wind_bc(vf: ti.template(), pf: ti.template(), u_in: ti.f32):
    for j in range(res_y):
        vf[0,       j] = ti.Vector([u_in, 0.0])
        vf[res_x-1, j] = vf[res_x-2, j]
        pf[0,       j] = pf[1, j]
        pf[res_x-1, j] = 0.0
    for i in range(res_x):
        vf[i, 0].y       = 0.0
        vf[i, res_y-1].y = 0.0
        vf[i, 0].x       = vf[i, 1].x
        vf[i, res_y-1].x = vf[i, res_y-2].x
        pf[i, 0]         = pf[i, 1]
        pf[i, res_y-1]   = pf[i, res_y-2]

@ti.kernel
def apply_obstacle_bc(vf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            if solid_at(i-1, j) == 1: vf[i, j].x = ti.max(vf[i, j].x, 0.0)
            if solid_at(i+1, j) == 1: vf[i, j].x = ti.min(vf[i, j].x, 0.0)
            if solid_at(i, j-1) == 1: vf[i, j].y = ti.max(vf[i, j].y, 0.0)
            if solid_at(i, j+1) == 1: vf[i, j].y = ti.min(vf[i, j].y, 0.0)

@ti.kernel
def init_cylinder_obstacle():
    _solid.fill(0)
    # cy 故意偏离正中心 ~6px：打破完美对称，触发卡门涡街周期性脱涡
    # （圆柱在网格中心线上时尾流是稳态对称的，永远不会脱涡）
    # cx 左移到 0.15：延长下游尾流区，给涡街更多发展空间
    cx = res_x * 0.15; cy = res_y * 0.5 + 6.0; r2 = CYLINDER_RADIUS_PX ** 2
    for i, j in _solid:
        if (i - cx)**2 + (j - cy)**2 < r2:
            _solid[i, j] = 1.0

@ti.kernel
def init_uv_field(dyef: ti.template()):
    for i, j in dyef:
        dyef[i, j] = ti.Vector([i / float(res_x), j / float(res_y)])

def enable_karman_obstacle():
    init_cylinder_obstacle()
    pressures_pair.cur.fill(0); pressures_pair.nxt.fill(0)
    smoke_pair.cur.fill(0); smoke_pair.nxt.fill(0)
    rgb_dye_pair.cur.fill(0); rgb_dye_pair.nxt.fill(0)
    color_field.fill(0)
    init_uv_field(dyes_pair.cur); init_uv_field(dyes_pair.nxt)
    apply_obstacle_bc(velocities_pair.cur)
    apply_obstacle_bc(velocities_pair.nxt)

@ti.kernel
def advance_noise_time():
    noise_time_field[None] += 0.012

@ti.kernel
def inject_smokes(smokef: ti.template(), dyef: ti.template(),
                  rgb_dyef: ti.template(), smoke_info: ti.types.ndarray(),
                  frame_cfl_px: ti.f32):
    speed_blend = ti.max(0.0, ti.min(1.0, (frame_cfl_px - 16.0) / 28.0))
    stripe_weight = 1.0 - 0.45 * speed_blend
    pulse = 1.0 + SMOKE_PULSE_AMPLITUDE * ti.sin(
        noise_time_field[None] * (2.0 * math.pi / SMOKE_PULSE_PERIOD)
    )
    for i, j in smokef:
        if i < 4:
            dyef[i, j] = ti.Vector([i / float(res_x), j / float(res_y)])
            smokef[i, j] = 0.0
            rgb_dyef[i, j] = ti.Vector([0.0, 0.0, 0.0])

            for idx in range(smoke_info.shape[0]):
                y0 = float(smoke_info[idx, 0])
                y1 = float(smoke_info[idx, 1])

                if y0 <= float(j) <= y1:
                    band_h = ti.max(1.0, y1 - y0)
                    yc = 0.5 * (y0 + y1)
                    dy = float(j) - yc

                    sigma = 0.19 * band_h + 1e-5
                    gauss = ti.exp(-(dy * dy) / (2.0 * sigma * sigma))
                    smoke_inj = gauss * (0.78 - 0.06 * speed_blend)
                    smokef[i, j] = ti.max(smokef[i, j], ti.min(1.0, smoke_inj))

                    rel = (float(j) - y0) / band_h
                    stripe_a = 0.5 + 0.5 * ti.sin(rel * 30.0)
                    stripe_b = 0.5 + 0.5 * ti.sin(rel * 55.0 + float(i) * 0.55)
                    stripe = ti.pow(0.35 + 0.65 * stripe_a * stripe_b, 1.15)

                    detail_seed = 0.18 + 1.20 * stripe
                    smooth_seed = 0.72
                    history_seed = ti.min(1.0, gauss * (detail_seed * stripe_weight + smooth_seed * (1.0 - stripe_weight)) * pulse)
                    rgb_dyef[i, j] = ti.Vector([history_seed, history_seed, history_seed])

@ti.kernel
def paint_color_field(smokef: ti.template(), dyef: ti.template(),
                      solidf: ti.template(), rgb_dyef: ti.template(),
                      colorf: ti.template(), frame_cfl_px: ti.f32):
    t = noise_time_field[None]
    speed_blend = ti.max(0.0, ti.min(1.0, (frame_cfl_px - 16.0) / 28.0))
    fine_weight = 1.0 - 0.55 * speed_blend
    exposure_compensation = 1.0 + 1.25 * speed_blend
    wind_speed_mps = frame_cfl_px / (SOLVER_SPEED_PER_MPS * dt)
    luminance_blend = ti.max(0.0, ti.min(1.0, (wind_speed_mps - 20.0) / 80.0))
    luminance_compensation = 1.0 + 1.45 * luminance_blend

    for i, j in smokef:
        if solidf[i, j] == 1:
            colorf[i, j] = ti.Vector([198/255.0, 134/255.0, 66/255.0])
            continue

        density_raw = smokef[i, j]
        density_blur = (
            density_raw * 4.0
            + sample(smokef, i - 1, j)
            + sample(smokef, i + 1, j)
            + sample(smokef, i, j - 1)
            + sample(smokef, i, j + 1)
        ) * 0.125
        density = ti.max(0.0, ti.min(1.0, density_raw * (1.0 - 0.45 * speed_blend) + density_blur * (0.45 * speed_blend)))

        if density < 8.0e-5:
            colorf[i, j] = ti.Vector([0.0, 0.0, 0.0])
            continue

        uv = dyef[i, j]
        streak_raw = rgb_dyef[i, j][0]
        streak_blur = (
            streak_raw * 4.0
            + sample(rgb_dyef, i - 1, j)[0]
            + sample(rgb_dyef, i + 1, j)[0]
            + sample(rgb_dyef, i, j - 1)[0]
            + sample(rgb_dyef, i, j + 1)[0]
        ) * 0.125
        streak_mem = ti.max(0.0, ti.min(1.0, streak_raw * (1.0 - 0.45 * speed_blend) + streak_blur * (0.45 * speed_blend)))
        streak_struct = ti.pow(streak_mem, 0.50)

        uv_coarse = ti.Vector([uv.x * 3.5 - t * 0.10, uv.y * 5.5 + t * 0.035])
        uv_fine   = ti.Vector([uv.x * 13.0 - t * 0.32, uv.y * 21.0 + t * 0.070])

        n_coarse = perlin_noise.fbm(uv_coarse, 3)
        n_fine   = perlin_noise.fbm(uv_fine, 3)

        noise_struct = ti.max(0.0, ti.min(1.0, 0.50 + 0.22 * n_coarse + (0.12 * fine_weight) * n_fine))
        noise_centered = (noise_struct - 0.5) * 2.0
        noise_mod = 1.0 + (0.25 * fine_weight) * noise_centered

        body = ti.pow(density, 1.05)
        stripe = ti.max(0.0, ti.min(1.0, streak_struct * noise_mod * (1.0 - 0.35 * speed_blend)))
        internal = ti.max(0.0, ti.min(1.0, 0.14 * body + (1.08 - 0.35 * speed_blend) * stripe))

        d_soft = ti.pow(density, 0.95)
        val_linear = d_soft * internal * 0.95 * exposure_compensation
        val = 1.0 - ti.exp(-0.85 * val_linear)

        cool_smoke = ti.Vector([0.70, 0.80, 0.92])
        warm_core  = ti.Vector([0.93, 0.89, 0.80])
        tone_mix = ti.max(0.0, ti.min(1.0, 0.18 + 0.34 * density + 0.06 * noise_centered))
        base_color = cool_smoke * (1.0 - tone_mix) + warm_core * tone_mix

        core = ti.max(0.0, ti.min(1.0, density * 1.35))
        self_shadow = 1.0 - 0.26 * core * (1.0 - 0.45 * stripe)
        fine_grain = 0.90 + (0.18 * fine_weight) * noise_struct + (0.08 * fine_weight) * stripe
        thin_edge = ti.max(0.0, 1.0 - density * 1.7)
        rim_light = 0.045 * (1.0 + 0.30 * speed_blend) * thin_edge * noise_struct * noise_struct

        col = (base_color * val * self_shadow * fine_grain + cool_smoke * rim_light) * luminance_compensation
        colorf[i, j] = ti.Vector([
            ti.max(0.0, ti.min(1.0, col.x)),
            ti.max(0.0, ti.min(1.0, col.y)),
            ti.max(0.0, ti.min(1.0, col.z)),
        ])

@ti.kernel
def decay_pressure(pf: ti.template(), decay_factor: ti.f32):
    for i, j in pf:
        pf[i, j] *= decay_factor

# ──────────────────────────────────────────────
#  主仿真逻辑步进与重置
# ──────────────────────────────────────────────
def fluid_substep_count(u_in):
    displacement_px = abs(float(u_in)) * dt
    return max(1, min(MAX_FLUID_SUBSTEPS, int(math.ceil(displacement_px / MAX_ADVECT_CFL_PX))))

def step(u_in, smoke_info):
    substeps = fluid_substep_count(u_in)
    dt_step = dt / substeps
    smoke_substeps = max(1, min(
        MAX_SMOKE_SUBSTEPS_PER_FLUID,
        int(math.ceil(abs(float(u_in)) * dt_step / MAX_SMOKE_CFL_PX)),
    ))
    smoke_dt_step = dt_step / smoke_substeps
    total_smoke_substeps = substeps * smoke_substeps
    frame_cfl_px = abs(float(u_in)) * dt
    speed_blend = max(0.0, min(1.0, (frame_cfl_px - 16.0) / 28.0))
    pressure_stability_blend = max(0.0, min(1.0, (frame_cfl_px - 28.8) / 9.6))
    curl_scale = max(0.70, 1.0 - 0.30 * speed_blend)
    smoke_decay = 0.996 ** (1.0 / total_smoke_substeps)   # 每帧 −0.4%（原 0.992），远处烟雾更持久
    rgb_decay = 0.998 ** (1.0 / total_smoke_substeps)     # 每帧 −0.2%（原 0.996），条纹纹理更持久
    pressure_decay = 0.90 ** (1.0 / substeps)
    diffusion_iters = VELOCITY_DIFFUSION_ITERS.get(fluid_material, 20)

    for _ in range(substeps):
        # 速度自平流：BFECC（低耗散、不限幅）→ 涡心保持锐利
        bfecc_advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt,
                     _bfecc_v_a, _bfecc_v_b, dt_step, 1.0, clamp=False)
        velocities_pair.swap()
        for _ in range(diffusion_iters):
            diffuse_velocity(velocities_pair.cur, velocities_pair.nxt, dt_step)
            velocities_pair.swap()

        apply_wind_bc(velocities_pair.cur, pressures_pair.cur, u_in)
        apply_obstacle_bc(velocities_pair.cur)

        for _ in range(smoke_substeps):
            advect_uv(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt, smoke_dt_step, 1.0)
            # 烟雾密度与条纹纹理也用 BFECC → 涡街标记不被抹糊
            bfecc_advect(velocities_pair.cur, smoke_pair.cur, smoke_pair.nxt,
                         _bfecc_s_a, _bfecc_s_b, smoke_dt_step, smoke_decay)
            bfecc_advect(velocities_pair.cur, rgb_dye_pair.cur, rgb_dye_pair.nxt,
                         _bfecc_c_a, _bfecc_c_b, smoke_dt_step, rgb_decay)
            dyes_pair.swap(); smoke_pair.swap(); rgb_dye_pair.swap()
            inject_smokes(smoke_pair.cur, dyes_pair.cur, rgb_dye_pair.cur, smoke_info, frame_cfl_px)

        if curl_strength:
            vorticity(velocities_pair.cur)
            enhance_vorticity(velocities_pair.cur, velocity_curls, dt_step, curl_scale)
        divergence(velocities_pair.cur)

        decay_pressure(pressures_pair.cur, pressure_decay)
        solve_pressure_sor(dt_step, substeps, pressure_stability_blend)
        subtract_gradient(velocities_pair.cur, pressures_pair.cur, dt_step)

        apply_wind_bc(velocities_pair.cur, pressures_pair.cur, u_in)
        apply_obstacle_bc(velocities_pair.cur)

    advance_noise_time()
    paint_color_field(smoke_pair.cur, dyes_pair.cur, _solid, rgb_dye_pair.cur, color_field, frame_cfl_px)

    if debug:
        divergence(velocities_pair.cur)
        div_np = velocity_divs.to_numpy()
        print(
            f"div_sum={np.sum(div_np):.4f}, div_abs_max={np.max(np.abs(div_np)):.6f}, "
            f"substeps={substeps}, smoke_substeps={smoke_substeps}"
        )

def reset():
    reset_mask_change_cache()
    velocities_pair.cur.fill(0);  velocities_pair.nxt.fill(0)
    pressures_pair.cur.fill(0);   pressures_pair.nxt.fill(0)
    smoke_pair.cur.fill(0);       smoke_pair.nxt.fill(0)
    rgb_dye_pair.cur.fill(0);     rgb_dye_pair.nxt.fill(0)
    color_field.fill(0)
    noise_time_field[None] = 0.0
    _solid.fill(0)
    init_uv_field(dyes_pair.cur)
    init_uv_field(dyes_pair.nxt)


# ──────────────────────────────────────────────
#  主循环
# ──────────────────────────────────────────────
def main(web_ui_only=False, max_frames=None, stats_json=None, initial_streamline=False):
    global RHO, VISCOSITY, fluid_material, curl_strength, debug, PIXEL_SIZE, _latest_frame, _latest_state

    # ── 可视化状态 ──
    visualize_d          = True   
    visualize_v          = False  
    visualize_c          = False  
    vis_pressure         = False  
    vis_start_stream_line = initial_streamline
    visualize_karman     = True   
    paused               = False
    cylinder_obstacle_initialized = False
    inlet_speed_mps      = U_IN / SOLVER_SPEED_PER_MPS

    streamline_color_map = sns.color_palette("viridis", as_cmap=True)
    cmap = plt.cm.coolwarm

    # ── 注入区信息 ──
    smoke_info = np.zeros((len(smoke_positions), 2), dtype=np.int16)
    for i in range(len(smoke_positions)):
        smoke_info[i, 0] = int(smoke_positions[i] * res_y - smoke_width[i] / 2)
        smoke_info[i, 1] = int(smoke_positions[i] * res_y + smoke_width[i] / 2)

    gui = None if web_ui_only else ti.GUI("Fluid Field Simulation", (res_x, res_y))
    init_uv_field(dyes_pair.cur)

    inlet_velocity_slider = None
    # 记录上一帧的滑块值：仅当用户"真正拖动"滑块（值发生变化）时才把它当作输入，
    # 避免把程序化同步 / 量化误差误判为用户操作而每帧回拽 web 端滑块。
    last_slider_value = inlet_speed_mps
    if gui is not None:
        inlet_velocity_slider = gui.slider("Inlet Velocity (m/s)", WIND_SPEED_MIN_MPS, WIND_SPEED_MAX_MPS, step=WIND_SPEED_STEP_MPS)
        inlet_velocity_slider.value = inlet_speed_mps
        last_slider_value = float(inlet_velocity_slider.value)

    def current_viz_name():
        if visualize_v:
            return "Velocity"
        if visualize_c:
            return "Vortex"
        if vis_pressure:
            return "Pressure"
        return "Smoke"

    def publish_ui_state():
        global _latest_state
        state = json.dumps({
            "type": "state",
            "material": fluid_material,
            "speed_mps": inlet_speed_mps,
            "viz": current_viz_name(),
            "streamline": vis_start_stream_line,
            "karman": visualize_karman,
            "paused": paused,
            "wind_speed_min_mps": WIND_SPEED_MIN_MPS,
            "wind_speed_max_mps": WIND_SPEED_MAX_MPS,
            "wind_speed_step_mps": WIND_SPEED_STEP_MPS,
            "solver_speed_per_mps": SOLVER_SPEED_PER_MPS,
        })
        with _latest_state_lock:
            _latest_state = state
        _broadcast_sync(state)

    def set_karman_enabled(enabled):
        nonlocal visualize_karman, cylinder_obstacle_initialized
        was_enabled = visualize_karman
        visualize_karman = bool(enabled)
        reset_mask_change_cache()
        if visualize_karman and not was_enabled:
            enable_karman_obstacle()
            cylinder_obstacle_initialized = True
        elif not visualize_karman:
            _solid.fill(0)
            cylinder_obstacle_initialized = False

    # ── 启动后台线程 ──
    threading.Thread(target=read_shared_memory, daemon=True).start()
    _ws_thread.start()
    publish_ui_state()

    last_frame_enc = time.time()
    reynolds_number = 0.0
    last_re_broadcast = 0.0
    last_longpress_broadcast = 0.0

    _ws_dirs   = None   
    _ws_starts = None   
    _ws_mags   = None   

    frame_count = 0
    stats = {
        "mode": "web_ui" if web_ui_only else "local_gui_plus_websocket",
        "frame_ms": [],
        "sim_ms": [],
        "compose_ms": [],
        "gui_ms": [],
        "encode_ms": [],
        "arrows_ms": [],
        "encoded_frames": 0,
    }

    while web_ui_only or gui.running:
        frame_start = time.perf_counter()

        # ── ① WS 控制指令 ──
        while not _ws_cmd_queue.empty():
            try:
                cmd    = _ws_cmd_queue.get_nowait()
                action = cmd.get("action")
                value  = cmd.get("value")

                if action == "set_material" and value in fluid_properties:
                    fluid_material = value
                    RHO = fluid_properties[value]["rho"]
                    VISCOSITY = fluid_properties[value]["viscosity"]
                    PIXEL_SIZE = PIXEL_SIZE_MAP[value]
                    _pixel_size_field[None] = PIXEL_SIZE
                    inlet_speed_mps = min(max(fluid_properties[value].get("u_in_default", U_IN) / SOLVER_SPEED_PER_MPS, WIND_SPEED_MIN_MPS), WIND_SPEED_MAX_MPS)
                    if inlet_velocity_slider is not None:
                        inlet_velocity_slider.value = inlet_speed_mps
                        last_slider_value = inlet_speed_mps
                    _rho_field[None] = RHO
                    _nu_field[None] = VISCOSITY
                    reset()
                    cylinder_obstacle_initialized = False
                    publish_ui_state()
                    print(f"[WS] Material → {value}  RHO={RHO}  ν={VISCOSITY}")
                elif action == "set_speed":
                    inlet_speed_mps = min(max(float(value), WIND_SPEED_MIN_MPS), WIND_SPEED_MAX_MPS)
                    if inlet_velocity_slider is not None:
                        inlet_velocity_slider.value = inlet_speed_mps
                        last_slider_value = inlet_speed_mps
                    publish_ui_state()
                elif action == "set_viz":
                    visualize_d = value == "smoke"
                    visualize_v = value == "velocity"
                    visualize_c = value == "vortex"
                    vis_pressure = value == "pressure"
                    publish_ui_state()
                elif action == "toggle_streamline":
                    vis_start_stream_line = bool(value)
                    publish_ui_state()
                elif action == "toggle_karman":
                    set_karman_enabled(value)
                    publish_ui_state()
                elif action == "reset":
                    reset(); paused = False
                    cylinder_obstacle_initialized = False
                    publish_ui_state()
                elif action == "set_freeze_input" and not bool(value):
                    release_frozen_input()
            except Exception:
                pass

        # ── 长按冻结状态同步 ──
        lp_now = time.time()
        if lp_now - last_longpress_broadcast >= 1 / 15:
            with shared_mask_lock:
                lp_progress = freeze_progress
                lp_frozen = freeze_input
                lp_touching = has_touch_current
            _broadcast_sync(json.dumps({
                "type": "longpress", "progress": lp_progress,
                "frozen": lp_frozen, "has_touch": lp_touching,
            }))
            last_longpress_broadcast = lp_now

        # ── ② GUI 键盘控制 ──
        if gui is not None and gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if   e.key == ti.GUI.ESCAPE: break
            elif e.key == "r": paused = False; reset()
            elif e.key == "v": visualize_v=True;  visualize_c=False; visualize_d=False; vis_pressure=False
            elif e.key == "d": visualize_d=True;  visualize_v=False; visualize_c=False; vis_pressure=False
            elif e.key == "c": visualize_c=True;  visualize_d=False; visualize_v=False; vis_pressure=False
            elif e.key == "i": vis_pressure=True; visualize_d=False; visualize_v=False; visualize_c=False
            elif e.key == "l": vis_start_stream_line = not vis_start_stream_line
            elif e.key == "k": set_karman_enabled(not visualize_karman)
            elif e.key == "p": paused = not paused
            elif e.key == "s": curl_strength = 3.5 if not curl_strength else 0
            publish_ui_state()

        if inlet_velocity_slider is not None:
            raw_slider = float(inlet_velocity_slider.value)
            # 只有用户实际拖动滑块（与上一帧记录值不同）才视为输入；
            # 程序化写入 .value 时会同步 last_slider_value，因此不会在这里触发回拽。
            if abs(raw_slider - last_slider_value) > 1e-4:
                last_slider_value = raw_slider
                inlet_speed_mps = min(max(raw_slider, WIND_SPEED_MIN_MPS), WIND_SPEED_MAX_MPS)
                publish_ui_state()

        if not paused:
            # ── ③ 共享内存输入与障碍物更新 ──
            if visualize_karman:
                if not cylinder_obstacle_initialized:
                    init_cylinder_obstacle()
                    cylinder_obstacle_initialized = True
            else:
                if process_mask_update(threshold=160.0, mode="threshold"):
                    refine_solid_mask(_solid)

            mask     = np.array(_solid.to_numpy()[1:-2, 1:-2], dtype=bool)
            bounds_x = get_mask_span_x(mask)
            active_wind_speed_mps = inlet_velocity_slider.value if inlet_velocity_slider is not None else inlet_speed_mps
            active_solver_speed = active_wind_speed_mps * SOLVER_SPEED_PER_MPS
            U_phys = active_wind_speed_mps

            if bounds_x > 1.0:
                L_char = bounds_x * PIXEL_SIZE
                re_label = "Re_obs"
            else:
                L_char = res_y * PIXEL_SIZE
                re_label = "Re_ch"

            reynolds_number = U_phys * L_char / max(VISCOSITY, 1e-12)
            
            # ── ④ 仿真步进 ──
            sim_start = time.perf_counter()
            step(active_solver_speed, smoke_info)
            stats["sim_ms"].append((time.perf_counter() - sim_start) * 1000.0)

            # ── ⑤ RGB 场构造 ──
            now = time.time()
            should_encode_frame = now - last_frame_enc >= 1/30
            should_render_frame = gui is not None or should_encode_frame

            compose_start = time.perf_counter()
            frame_rgb = None
            if should_render_frame:
                if visualize_c:
                    vorticity(velocities_pair.cur)
                    gray = velocity_curls.to_numpy() * 0.03 + 0.5
                    frame_rgb = np.stack([gray, gray, gray], axis=-1)
                elif visualize_d:
                    frame_rgb = color_field.to_numpy()
                    lum = (0.2126 * frame_rgb[..., 0] + 0.7152 * frame_rgb[..., 1] + 0.0722 * frame_rgb[..., 2])[..., None]
                    bright_mask = np.maximum(lum - 0.82, 0) * frame_rgb / (lum + 1e-6)

                    h, w = bright_mask.shape[:2]
                    small_w = max(1, w // 4)
                    small_h = max(1, h // 4)
                    small = cv2.resize(bright_mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
                    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=1.2)
                    bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

                    frame_rgb = np.clip(frame_rgb + bloom * 0.05, 0.0, 1.0)
                    frame_rgb = np.power(np.clip(frame_rgb, 0, 1), 0.90)
                    frame_rgb = np.clip((frame_rgb - 0.02) * 1.04, 0, 1)
                elif visualize_v:
                    vel_np = velocities_pair.cur.to_numpy() * 0.01 + 0.5
                    frame_rgb = np.concatenate([vel_np, np.zeros_like(vel_np[..., :1])], axis=-1)
                elif vis_pressure:
                    p = pressures_pair.cur.to_numpy()
                    p_base = np.mean(p)
                    P_RANGE = 40000.0
                    p_norm = np.clip((p - p_base + P_RANGE) / (2 * P_RANGE), 0.0, 1.0)
                    frame_rgb = cmap(p_norm)[..., :3].astype(np.float32)
                else:
                    frame_rgb = color_field.to_numpy()
            stats["compose_ms"].append((time.perf_counter() - compose_start) * 1000.0)

            # ── ⑥ 渲染到 GUI ──
            gui_start = time.perf_counter()
            if gui is not None and visualize_c:
                gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
            elif gui is not None and visualize_d:
                gui.set_image(color_field)
            elif gui is not None and visualize_v:
                gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
            elif gui is not None and vis_pressure and frame_rgb is not None:
                gui.set_image(frame_rgb)

            arrows_start = time.perf_counter()
            if should_render_frame and vis_start_stream_line:
                directions, streamline_starts, streamline_mag = calculate_streamline(velocities_pair.cur, _solid)
                _ws_dirs   = directions
                _ws_starts = streamline_starts
                _ws_mags   = streamline_mag
                colors = rgb_array_to_hex(streamline_color_map(streamline_mag / 500.0))
                if gui is not None:
                    gui.arrows(orig=streamline_starts, direction=directions, radius=2, color=colors)
                flat = []
                for (ox, oy), (dx, dy), mag in zip(streamline_starts, directions, streamline_mag):
                    flat.extend([round(float(ox), 4), round(float(oy), 4),
                                 round(float(dx), 5), round(float(dy), 5),
                                 round(float(mag), 3)])
                _broadcast_sync(json.dumps({"type": "arrows", "data": flat}))
            elif should_render_frame:
                _ws_dirs = _ws_starts = _ws_mags = None
                _broadcast_sync(json.dumps({"type": "arrows", "data": []}))
            stats["arrows_ms"].append((time.perf_counter() - arrows_start) * 1000.0)

            if gui is not None:
                gui.text(content=f'{re_label}: {reynolds_number:.1f}', pos=[0.75, 0.9], font_size=25, color=0xFFFFFF)
                gui.show()
            stats["gui_ms"].append((time.perf_counter() - gui_start) * 1000.0)

            # ── ⑦ 广播流编码数据 (限速 30 FPS) ──
            encode_ms = 0.0
            if should_encode_frame and frame_rgb is not None:
                encode_start = time.perf_counter()
                img_u8 = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
                img_u8 = np.flipud(np.transpose(img_u8, (1, 0, 2)))

                buf = io.BytesIO()
                Image.fromarray(img_u8).save(buf, format="JPEG", quality=92)
                frame_bytes = buf.getvalue()
                with _latest_frame_lock:
                    _latest_frame = frame_bytes
                _broadcast_sync(frame_bytes)
                last_frame_enc = now
                encode_ms = (time.perf_counter() - encode_start) * 1000.0
                stats["encoded_frames"] += 1
            stats["encode_ms"].append(encode_ms)

            # ── ⑧ WebSocket 每秒同步 Reynolds 数值 ──
            if now - last_re_broadcast >= 1.0:
                _broadcast_sync(json.dumps({"type": "reynolds", "value": reynolds_number}))
                last_re_broadcast = now

        stats["frame_ms"].append((time.perf_counter() - frame_start) * 1000.0)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    if stats_json:
        def summarize(values):
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0:
                return {"avg": 0.0, "p95": 0.0}
            return {"avg": float(arr.mean()), "p95": float(np.percentile(arr, 95))}

        total_s = sum(stats["frame_ms"]) / 1000.0
        summary = {
            "simulator": "fluid",
            "mode": stats["mode"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frames": frame_count,
            "duration_s": total_s,
            "fps_avg": (frame_count / total_s) if total_s > 0 else 0.0,
            "encoded_fps": (stats["encoded_frames"] / total_s) if total_s > 0 else 0.0,
            "frame_ms": summarize(stats["frame_ms"]),
            "sim_ms": summarize(stats["sim_ms"]),
            "compose_ms": summarize(stats["compose_ms"]),
            "gui_ms": summarize(stats["gui_ms"]),
            "encode_ms": summarize(stats["encode_ms"]),
            "arrows_ms": summarize(stats["arrows_ms"]),
            "encoded_frames": stats["encoded_frames"],
            "web_ui_only": web_ui_only,
            "streamline": vis_start_stream_line,
            "notes": "Main-loop UI path benchmark; no browser decode/canvas timing included.",
        }
        os.makedirs(os.path.dirname(stats_json) or ".", exist_ok=True)
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-ui", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--streamline", action="store_true")
    args = parser.parse_args()
    main(
        web_ui_only=args.web_ui,
        max_frames=args.max_frames,
        stats_json=args.stats_json,
        initial_streamline=args.streamline,
    )
