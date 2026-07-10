import os
import io
import math
import mmap
import time
import json
import struct
import asyncio
import threading
import argparse
from threading import Lock

import numpy as np
import taichi as ti
import websockets
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ──────────────────────────────────────────────
#  全局配置与常量
# ──────────────────────────────────────────────
RESOLUTION_SCALE = 640.0 / 850.0
sim_size = round(850 * RESOLUTION_SCALE)
sim_res_x = sim_size
sim_res_y = sim_size

disp_res_x = round(780 * RESOLUTION_SCALE)
disp_res_y = round(520 * RESOLUTION_SCALE)

offset_x = (sim_res_x - disp_res_x) // 2
offset_y = (sim_res_y - disp_res_y) // 2

alpha_blend = 0.8
STREAMLINE_LENGTH = 10.0 * RESOLUTION_SCALE
FILING_DENSITY_SCALE = 0.30
num_particles = round(5000 * RESOLUTION_SCALE * RESOLUTION_SCALE * FILING_DENSITY_SCALE)
DIRECTION_ARROW_STEP = round(20 * RESOLUTION_SCALE)
DIRECTION_ARROW_START = DIRECTION_ARROW_STEP // 2
DIRECTION_ARROW_COLS = ((disp_res_x - 1 - DIRECTION_ARROW_START) // DIRECTION_ARROW_STEP) + 1
DIRECTION_ARROW_ROWS = ((disp_res_y - 1 - DIRECTION_ARROW_START) // DIRECTION_ARROW_STEP) + 1
DIRECTION_ARROW_COUNT = DIRECTION_ARROW_COLS * DIRECTION_ARROW_ROWS
TARGET_RENDER_FPS = 60
DIRECTION_UPDATE_FPS = 30
DIRECTION_QUANTIZATION = 10000
FILING_RASTER_STEPS = 12

# Interactive solves are advanced in short chunks so a material/mask update cannot
# monopolize a display frame. The same PCG state continues on following frames,
# so a stationary input still converges to the requested tolerance.
INTERACTIVE_SOLVE_ITERS = 12
SETTLE_SOLVE_ITERS = 12
SCENE_CHANGE_SOLVE_ITERS = 24
INTERACTIVE_SOLVE_TOL = 1e-4
MASK_DIFF_THRESH = 100.0
LOWRES_MASK_DIFF_THRESH = 1
MASK_CHANGED_SOLVE_ITERS = INTERACTIVE_SOLVE_ITERS
MASK_STEADY_SOLVE_ITERS = SETTLE_SOLVE_ITERS

# 共享内存配置
SHARED_MEMORY_NAME = "shared_touch_image"
SHARED_MEMORY_SIZE = 4068
INPUT_WIDTH        = 78
INPUT_HEIGHT       = 52
INPUT_FRAME_SIZE   = INPUT_WIDTH * INPUT_HEIGHT
HEADER_SIZE        = 12
LONGPRESS_DURATION = 5.0
LONGPRESS_PRESSURE_THRESHOLD = 0.55


interactive_materials = {
    'air':     {'mu': 1.0,  'sigma': 0.0},
    'flesh':   {'mu': 2.5,  'sigma': 10.0},
    'ceramic': {'mu': 8.0,  'sigma': 100.0},
    'iron':    {'mu': 25.0, 'sigma': 1e5},
    'repel':   {'mu': 0.05, 'sigma': 1e7},
}

SCENE_NAMES = [
    "Single Wire", "Dipole", "Uniform Field",
    "Quadrupole", "Solenoid", "Horseshoe", "Two Magnets"
]

ti.init(arch=ti.gpu)


# ──────────────────────────────────────────────
#  Taichi 场变量声明
# ──────────────────────────────────────────────
A_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
J_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
B_field = ti.Vector.field(2, dtype=ti.f32, shape=(sim_res_x, sim_res_y))

fixed_A_mask  = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
fixed_A_value = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
rhs_eff_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

r_field  = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
p_field  = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
Ap_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

magnetic_intensity_field = ti.field(dtype=ti.f32, shape=(disp_res_x, disp_res_y))
initial_mask    = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
input_mask      = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
last_input_mask = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
color_field     = ti.Vector.field(3, float, shape=(disp_res_x, disp_res_y))
viridis_lut     = ti.Vector.field(3, dtype=ti.f32, shape=256)

mu_field         = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
inv_mu_field     = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_field      = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
mu_base_field    = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_base_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

filing_density = ti.field(ti.f32, shape=(sim_res_x, sim_res_y))
filing_outer_overlay = ti.field(ti.f32, shape=(disp_res_x, disp_res_y))
filing_inner_overlay = ti.field(ti.f32, shape=(disp_res_x, disp_res_y))

# PCG Solver 中间量
_cg_rsold  = ti.field(ti.f32, shape=())
_cg_rsnew  = ti.field(ti.f32, shape=())
_cg_pAp    = ti.field(ti.f32, shape=())
_cg_alpha  = ti.field(ti.f32, shape=())
_cg_thresh = ti.field(ti.f32, shape=())
_cg_res2   = ti.field(ti.f32, shape=())   
_cg_status = ti.field(ti.i32, shape=())
precond_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  
z_field       = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  

# 渲染控制参数
line_freq_field      = ti.field(dtype=ti.f32, shape=())
line_thickness_field = ti.field(dtype=ti.f32, shape=())

# 粒子系统
p_pos   = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
p_vel   = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
p_mass  = ti.field(dtype=ti.f32, shape=num_particles)
p_start = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
p_end   = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)

direction_arrow_dir = ti.Vector.field(2, dtype=ti.f32, shape=DIRECTION_ARROW_COUNT)

_direction_arrow_starts_np = np.empty((DIRECTION_ARROW_COUNT, 2), dtype=np.float32)
for _idx in range(DIRECTION_ARROW_COUNT):
    _col = _idx // DIRECTION_ARROW_ROWS
    _row = _idx - _col * DIRECTION_ARROW_ROWS
    _direction_arrow_starts_np[_idx] = (
        (DIRECTION_ARROW_START + _col * DIRECTION_ARROW_STEP) / disp_res_x,
        (DIRECTION_ARROW_START + _row * DIRECTION_ARROW_STEP) / disp_res_y,
    )

_viridis_initialized = False


# ──────────────────────────────────────────────
#  WebSocket 状态与控制
# ──────────────────────────────────────────────
WS_HOST = "0.0.0.0"
WS_PORT = 8765

_latest_frame_bytes = None
_latest_frame_seq = 0
_latest_frame_lock  = Lock()          
_latest_arrows_json = None
_latest_arrows_seq = 0
_latest_longpress_json = None
_latest_arrows_lock = Lock()          
_connected_clients  = set()
_clients_lock       = Lock()           
_pending_command    = None
_cmd_lock           = Lock()             

_WS_INIT_MSG = json.dumps({
    "type":      "init",
    "width":     disp_res_x,
    "height":    disp_res_y,
    "scenes":    SCENE_NAMES,
    "materials": list(interactive_materials.keys()),
})

def _handle_ws_message(msg_str: str):
    global _pending_command
    try:
        msg = json.loads(msg_str)
        with _cmd_lock:
            _pending_command = msg
    except Exception as e:
        print(f"[WS] 解析失败: {e}")

def _set_latest_arrows_json(payload):
    global _latest_arrows_json, _latest_arrows_seq
    with _latest_arrows_lock:
        if payload == _latest_arrows_json:
            return False
        _latest_arrows_json = payload
        _latest_arrows_seq += 1
    return True

async def _ws_handler(websocket):
    try:
        await websocket.send(_WS_INIT_MSG)
        with _latest_arrows_lock:
            arrows_json = _latest_arrows_json
        if arrows_json is not None:
            await websocket.send(arrows_json)
    except Exception as e:
        print(f"[WS] 初始化失败: {e}")
        return

    with _clients_lock:
        _connected_clients.add(websocket)
    print(f"[WS] 客户端已连接 在线: {len(_connected_clients)}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                _handle_ws_message(message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        with _clients_lock:
            _connected_clients.discard(websocket)
        print(f"[WS] 客户端断开 在线: {len(_connected_clients)}")

async def _frame_broadcaster():
    last_sent_seq = -1
    last_sent_arrows_seq = -1
    while True:
        await asyncio.sleep(1 / TARGET_RENDER_FPS)
        with _latest_frame_lock:
            frame = _latest_frame_bytes
            frame_seq = _latest_frame_seq
        if frame is None or frame_seq == last_sent_seq:
            continue
        with _clients_lock:
            targets = list(_connected_clients)
        if not targets:
            continue
        with _latest_arrows_lock:
            longpress_json = _latest_longpress_json
            arrows_json = _latest_arrows_json
            arrows_seq = _latest_arrows_seq
        arrows_changed = arrows_seq != last_sent_arrows_seq
        dead = []
        for ws in targets:
            try:
                if longpress_json is not None:
                    await ws.send(longpress_json)
                await ws.send(frame)
                if arrows_changed and arrows_json is not None:
                    await ws.send(arrows_json)
            except Exception:
                dead.append(ws)
        last_sent_seq = frame_seq
        if arrows_changed:
            last_sent_arrows_seq = arrows_seq
        if dead:
            with _clients_lock:
                for ws in dead:
                    _connected_clients.discard(ws)

async def _ws_server_main():
    async with websockets.serve(_ws_handler, WS_HOST, WS_PORT):
        print(f"[WS] 服务器启动 ws://{WS_HOST}:{WS_PORT}")
        await _frame_broadcaster()

def _ws_thread_func(): asyncio.run(_ws_server_main())


# ──────────────────────────────────────────────
#  共享内存提取
# ──────────────────────────────────────────────
shared_mask_lock = Lock()
shared_mask_np = None
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
    """后台线程读取 MMap 中 78x52 的触控数据"""
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
                print(f"[SharedMem] 内存文件不存在: {fp}")
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

def process_mask_update(threshold=179.0):
    """主线程中调用，重置形状并更新至 Taichi field"""
    global new_mask_available, shared_mask_np, last_lowres_mask_np
    with shared_mask_lock:
        if not new_mask_available or shared_mask_np is None:
            return False
        arr = shared_mask_np.copy()
        new_mask_available = False

    lowres_mask = arr < threshold
    if last_lowres_mask_np is not None and np.array_equal(lowres_mask, last_lowres_mask_np):
        return False
    last_lowres_mask_np = lowres_mask.copy()

    img = Image.fromarray(arr).resize((disp_res_x, disp_res_y), resample=Image.BICUBIC)
    arr_resized = np.array(img, copy=False).astype(np.float32)

    mask_np = (arr_resized < threshold).astype(np.float32)
    #mask_np_flipped = np.flipud(mask_np.T)
    mask_np_flipped = np.flip(mask_np)
    mask_np_flipped = mask_np_flipped.T
    mask_np_flipped = np.flipud(mask_np_flipped)

    input_mask_full = np.zeros((sim_res_x, sim_res_y), dtype=np.float32)
    input_mask_full[offset_x:offset_x+disp_res_x, offset_y:offset_y+disp_res_y] = mask_np_flipped
    input_mask.from_numpy(input_mask_full)
    return True

# ──────────────────────────────────────────────
#  Taichi 核函数：求解器与数学操作
# ──────────────────────────────────────────────
@ti.func
def sample_A_free(field: ti.template(), i: int, j: int) -> ti.f32:
    val = 0.0
    if 0 <= i < sim_res_x and 0 <= j < sim_res_y:
        if fixed_A_mask[i, j] < 0.5:
            val = field[i, j]
    return val

@ti.func
def sample_inv_mu(i: int, j: int) -> float:
    val = 1.0
    if 0 <= i < sim_res_x and 0 <= j < sim_res_y: val = inv_mu_field[i, j]
    return val

@ti.func
def harm(a, b): return 2.0 * a * b / (a + b + 1e-12)

@ti.kernel
def update_inv_mu():
    for i, j in mu_field: inv_mu_field[i, j] = 1.0 / mu_field[i, j]

@ti.kernel
def compute_Ap(p: ti.template(), Ap: ti.template()):
    for i, j in p:
        if fixed_A_mask[i, j] > 0.5:
            Ap[i, j] = p[i, j]
        else:
            p_c = p[i, j]
            p_L = sample_A_free(p, i-1, j); p_R = sample_A_free(p, i+1, j)
            p_D = sample_A_free(p, i, j-1); p_U = sample_A_free(p, i, j+1)
            imc = sample_inv_mu(i, j)
            wL = harm(imc, sample_inv_mu(i-1, j))
            wR = harm(imc, sample_inv_mu(i+1, j))
            wD = harm(imc, sample_inv_mu(i, j-1))
            wU = harm(imc, sample_inv_mu(i, j+1))
            Ap[i, j] = (wL + wR + wD + wU) * p_c - (wL * p_L + wR * p_R + wD * p_D + wU * p_U)

@ti.kernel
def compute_r(x: ti.template(), r: ti.template(), b: ti.template(), Ap: ti.template()):
    for I in ti.grouped(x): r[I] = b[I] - Ap[I]

@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src): dst[I] = src[I]

@ti.kernel
def _cg_dot_to_field(v1: ti.template(), v2: ti.template(), out: ti.template()):
    s = 0.0
    for I in ti.grouped(v1):
        s += v1[I] * v2[I]
    out[()] = s

@ti.kernel
def _cg_compute_thresh(b: ti.template(), tol: ti.f32):
    # Fixed-value cells are equations we enforce exactly and must not inflate the
    # relative residual tolerance (especially in the Uniform Field scene).
    s = 0.0
    for I in ti.grouped(b):
        if fixed_A_mask[I] < 0.5:
            s += b[I] * b[I]
    _cg_thresh[()] = s * tol * tol + 1e-10

@ti.kernel
def compute_preconditioner():
    for i, j in precond_field:
        if fixed_A_mask[i, j] > 0.5:
            precond_field[i, j] = 1.0
        else:
            imc = sample_inv_mu(i, j)
            wL  = harm(imc, sample_inv_mu(i-1, j))
            wR  = harm(imc, sample_inv_mu(i+1, j))
            wD  = harm(imc, sample_inv_mu(i, j-1))
            wU  = harm(imc, sample_inv_mu(i, j+1))
            precond_field[i, j] = 1.0 / (wL + wR + wD + wU + 1e-10)

@ti.kernel
def apply_fixed_A(x: ti.template()):
    for I in ti.grouped(x):
        if fixed_A_mask[I] > 0.5:
            x[I] = fixed_A_value[I]

@ti.kernel
def build_effective_rhs():
    for i, j in J_field:
        if fixed_A_mask[i, j] > 0.5:
            rhs_eff_field[i, j] = fixed_A_value[i, j]
        else:
            imc = sample_inv_mu(i, j)
            wL  = harm(imc, sample_inv_mu(i-1, j))
            wR  = harm(imc, sample_inv_mu(i+1, j))
            wD  = harm(imc, sample_inv_mu(i, j-1))
            wU  = harm(imc, sample_inv_mu(i, j+1))
            b = J_field[i, j]
            if i - 1 >= 0 and fixed_A_mask[i-1, j] > 0.5:
                b += wL * fixed_A_value[i-1, j]
            if i + 1 < sim_res_x and fixed_A_mask[i+1, j] > 0.5:
                b += wR * fixed_A_value[i+1, j]
            if j - 1 >= 0 and fixed_A_mask[i, j-1] > 0.5:
                b += wD * fixed_A_value[i, j-1]
            if j + 1 < sim_res_y and fixed_A_mask[i, j+1] > 0.5:
                b += wU * fixed_A_value[i, j+1]
            rhs_eff_field[i, j] = b

@ti.kernel
def apply_preconditioner(r: ti.template(), z: ti.template()):
    for I in ti.grouped(r):
        z[I] = r[I] * precond_field[I]

@ti.kernel
def _cg_apply_preconditioner_and_dot(
    r: ti.template(), z: ti.template(), out: ti.template()
):
    s = 0.0
    for I in ti.grouped(r):
        value = r[I] * precond_field[I]
        z[I] = value
        s += r[I] * value
    out[()] = s

@ti.kernel
def _cg_compute_Ap_and_dot(p: ti.template(), Ap: ti.template()):
    s = 0.0
    for i, j in p:
        value = 0.0
        if fixed_A_mask[i, j] > 0.5:
            value = p[i, j]
        else:
            p_c = p[i, j]
            p_L = sample_A_free(p, i-1, j)
            p_R = sample_A_free(p, i+1, j)
            p_D = sample_A_free(p, i, j-1)
            p_U = sample_A_free(p, i, j+1)
            imc = sample_inv_mu(i, j)
            wL = harm(imc, sample_inv_mu(i-1, j))
            wR = harm(imc, sample_inv_mu(i+1, j))
            wD = harm(imc, sample_inv_mu(i, j-1))
            wU = harm(imc, sample_inv_mu(i, j+1))
            value = (
                (wL + wR + wD + wU) * p_c
                - (wL * p_L + wR * p_R + wD * p_D + wU * p_U)
            )
        Ap[i, j] = value
        s += p[i, j] * value
    _cg_pAp[()] = s

@ti.kernel
def _cg_prepare_alpha_checked():
    if _cg_status[()] == 0:
        pAp = _cg_pAp[()]
        rsold = _cg_rsold[()]
        alpha = 0.0
        ok = (pAp == pAp) and (rsold == rsold) and pAp > 1e-20 and rsold > 1e-30
        if ok:
            alpha = rsold / pAp
            ok = (alpha == alpha) and ti.abs(alpha) <= 1e8
        if ok:
            _cg_alpha[()] = alpha
        else:
            _cg_alpha[()] = 0.0
            _cg_status[()] = 1

@ti.kernel
def _cg_update_xr_checked(
    x: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template()
):
    alpha = _cg_alpha[()]
    for I in ti.grouped(x):
        if _cg_status[()] == 0:
            x[I] += alpha * p[I]
            r[I] -= alpha * Ap[I]

@ti.kernel
def _cg_prepare_beta_checked():
    if _cg_status[()] == 0:
        rsnew = _cg_rsnew[()]
        rsold = _cg_rsold[()]
        beta = 0.0
        ok = (rsnew == rsnew) and (rsold == rsold) and rsnew >= 0.0 and rsold > 1e-30
        if ok:
            beta = rsnew / rsold
            ok = (beta == beta) and ti.abs(beta) <= 1e8
        if ok:
            _cg_alpha[()] = beta
            _cg_rsold[()] = rsnew
        else:
            _cg_alpha[()] = 0.0
            _cg_status[()] = 1

@ti.kernel
def _cg_update_p_checked(z: ti.template(), p: ti.template()):
    beta = _cg_alpha[()]
    for I in ti.grouped(p):
        if _cg_status[()] == 0:
            p[I] = z[I] + beta * p[I]

@ti.kernel
def _cg_compute_res2(r: ti.template()):
    s = 0.0
    for I in ti.grouped(r):
        s += r[I] * r[I]
    _cg_res2[()] = s


_pcg_active = False
_pcg_converged = True
_pcg_last_res2 = 0.0
_pcg_last_thresh = 0.0
_pcg_last_status = 0


def _update_pcg_host_state():
    global _pcg_active, _pcg_converged
    global _pcg_last_res2, _pcg_last_thresh, _pcg_last_status

    _pcg_last_res2 = float(_cg_res2[()])
    _pcg_last_thresh = float(_cg_thresh[()])
    _pcg_last_status = int(_cg_status[()])
    finite = math.isfinite(_pcg_last_res2) and math.isfinite(_pcg_last_thresh)
    _pcg_converged = (
        _pcg_last_status == 0
        and finite
        and _pcg_last_res2 <= _pcg_last_thresh
    )
    _pcg_active = _pcg_last_status == 0 and finite and not _pcg_converged
    return _pcg_converged


def start_poisson_pcg(x_field, b_field, tol=1e-4):
    """Initialize a PCG solve while preserving x_field as the warm start."""
    global _pcg_active

    _cg_status[()] = 0
    compute_Ap(x_field, Ap_field)
    compute_r(x_field, r_field, b_field, Ap_field)
    _cg_compute_thresh(b_field, tol)
    _cg_compute_res2(r_field)
    if _update_pcg_host_state():
        return True
    if not _pcg_active:
        return False

    _cg_apply_preconditioner_and_dot(r_field, z_field, _cg_rsold)
    copy_field(z_field, p_field)
    return False


def advance_poisson_pcg(x_field, max_iters=12, residual_check_interval=6):
    """Advance the current PCG state without restarting the conjugate direction."""
    global _pcg_active

    if not _pcg_active or max_iters <= 0:
        return 0

    executed = 0
    check_every = max(1, int(residual_check_interval))
    for it in range(max_iters):
        _cg_compute_Ap_and_dot(p_field, Ap_field)
        _cg_prepare_alpha_checked()
        _cg_update_xr_checked(x_field, p_field, r_field, Ap_field)
        executed = it + 1

        should_check = (executed % check_every == 0) or (executed == max_iters)
        if should_check:
            _cg_compute_res2(r_field)
            if _update_pcg_host_state() or not _pcg_active:
                break

        _cg_apply_preconditioner_and_dot(r_field, z_field, _cg_rsnew)
        _cg_prepare_beta_checked()
        _cg_update_p_checked(z_field, p_field)

    return executed


def solve_poisson_pcg_safe(x_field, b_field, max_iters=100, tol=1e-4, verbose=False):
    start_poisson_pcg(x_field, b_field, tol=tol)
    return advance_poisson_pcg(x_field, max_iters=max_iters)


def start_current_system(tol=1e-3, rebuild_rhs=True):
    if rebuild_rhs:
        build_effective_rhs()
    apply_fixed_A(A_field)
    return start_poisson_pcg(A_field, rhs_eff_field, tol=tol)


def continue_current_system(max_iters=12):
    iters = advance_poisson_pcg(A_field, max_iters=max_iters)
    apply_fixed_A(A_field)
    return iters


def current_solver_active():
    return _pcg_active


def current_solver_converged():
    return _pcg_converged


def current_solver_metrics():
    return {
        "active": _pcg_active,
        "converged": _pcg_converged,
        "residual": math.sqrt(max(_pcg_last_res2, 0.0)),
        "threshold": math.sqrt(max(_pcg_last_thresh, 0.0)),
        "status": _pcg_last_status,
    }


def solve_current_system(max_iters=100, tol=1e-3, verbose=False):
    start_current_system(tol=tol, rebuild_rhs=True)
    iters = continue_current_system(max_iters=max_iters)
    if verbose:
        metrics = current_solver_metrics()
        print(
            f"[Solver] iters={iters} residual={metrics['residual']:.6e} "
            f"target={metrics['threshold']:.6e} status={metrics['status']} "
            f"converged={metrics['converged']}"
        )
    return iters


@ti.kernel
def field_sum(f: ti.template()) -> ti.f32:
    s = 0.0
    for I in ti.grouped(f): s += f[I]
    return s

@ti.kernel
def mask_l1_diff(a: ti.template(), b: ti.template()) -> ti.f32:
    s = 0.0
    for I in ti.grouped(a): s += ti.abs(a[I] - b[I])
    return s

@ti.kernel
def copy_mask(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src): dst[I] = src[I]


# ──────────────────────────────────────────────
#  Taichi 核函数：物理状态计算与粒子系统
# ──────────────────────────────────────────────
@ti.kernel
def compute_B_field(A: ti.template(), B: ti.template()):
    for i, j in B:
        if 0 < i < sim_res_x-1 and 0 < j < sim_res_y-1:
            B[i,j] = ti.Vector([(A[i,j+1]-A[i,j-1])*0.5, -(A[i+1,j]-A[i-1,j])*0.5])
        else:
            B[i,j] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_magnetic_intensity(B_field: ti.template(), magnetic_intensity_field: ti.template()):
    for i,j in magnetic_intensity_field: 
        magnetic_intensity_field[i,j] = B_field[offset_x+i,offset_y+j].norm()

@ti.kernel
def compute_density_grid():
    for I in ti.grouped(filing_density): filing_density[I] = 0.0
    for i in p_pos:
        ix = ti.cast(p_pos[i].x, ti.i32); iy = ti.cast(p_pos[i].y, ti.i32)
        if 1 <= ix < sim_res_x-1 and 1 <= iy < sim_res_y-1:
            filing_density[ix, iy] += 1.0

@ti.kernel
def init_particles():
    for i in p_pos:
        valid=False; fp=ti.Vector([0.0,0.0]); att=0
        while not valid and att < 100:
            rx=offset_x+ti.random()*disp_res_x; ry=offset_y+ti.random()*disp_res_y
            ix=ti.max(1,ti.min(ti.cast(rx,ti.i32),sim_res_x-2))
            iy=ti.max(1,ti.min(ti.cast(ry,ti.i32),sim_res_y-2))
            if initial_mask[ix,iy] < 0.5: fp=ti.Vector([rx,ry]); valid=True
            att+=1
        p_pos[i]=fp; p_vel[i]=ti.Vector([0.0,0.0]); p_mass[i]=0.5+ti.random()*1.5

@ti.kernel
def prepare_particle_lines():
    for i in p_pos:
        pos=p_pos[i]; ix=ti.max(1,ti.min(ti.cast(pos.x,ti.i32),sim_res_x-2))
        iy=ti.max(1,ti.min(ti.cast(pos.y,ti.i32),sim_res_y-2))
        Bv=B_field[ix,iy]; Bm=Bv.norm()
        if pos.x<offset_x or pos.x>offset_x+disp_res_x or pos.y<offset_y or pos.y>offset_y+disp_res_y or Bm<0.1:
            p_start[i]=ti.Vector([-1.0,-1.0]); p_end[i]=ti.Vector([-1.0,-1.0])
        else:
            dp=Bv/(Bm+1e-5)
            grain_scale=0.75+0.25*p_mass[i]
            fl=ti.max(2.0,ti.min(Bm*2.2+2.0,9.0))*grain_scale
            cx=(pos.x-offset_x)/disp_res_x; cy=(pos.y-offset_y)/disp_res_y
            dx=(dp.x*fl)/disp_res_x * 0.5; dy=(dp.y*fl)/disp_res_y * 0.5
            p_start[i]=ti.Vector([cx-dx,cy-dy]); p_end[i]=ti.Vector([cx+dx,cy+dy])

@ti.kernel
def update_particles_fast():
    dt=0.8
    for i in p_pos:
        pos=p_pos[i]; ix=ti.max(1,ti.min(ti.cast(pos.x,ti.i32),sim_res_x-2))
        iy=ti.max(1,ti.min(ti.cast(pos.y,ti.i32),sim_res_y-2))
        Bv=B_field[ix,iy]; Bm=Bv.norm()+1e-7; dB=Bv/Bm
        gB=ti.Vector([B_field[ix+1,iy].norm()-B_field[ix-1,iy].norm(),
                      B_field[ix,iy+1].norm()-B_field[ix,iy-1].norm()])*0.5
        gBm=gB.norm(); fg=ti.Vector([0.0,0.0])
        if gBm>1e-5: fg=(gB/gBm)*ti.min(gBm*0.08,0.4)
        dR=filing_density[ix+1,iy]; dL=filing_density[ix-1,iy]
        dU=filing_density[ix,iy+1]; dD=filing_density[ix,iy-1]
        gd=ti.Vector([(dR-dL)*0.5,(dU-dD)*0.5]); dp=ti.Vector([-dB.y,dB.x])
        tf=fg-dB*gd.dot(dB)*3.0-dp*gd.dot(dp)*8.0
        fn=tf.norm(); sf=0.10+ti.min(Bm,3.0)*0.04; drive=ti.Vector([0.0,0.0])
        if fn>sf and Bm>0.05: drive=(tf/fn)*ti.min((fn-sf)*0.22,1.0)
        tv=p_vel[i]*0.55+drive
        if tv.norm()<0.015: tv=ti.Vector([0.0,0.0])
        np2=p_pos[i]+tv*dt
        nx=ti.max(1,ti.min(ti.cast(np2.x,ti.i32),sim_res_x-2))
        ny=ti.max(1,ti.min(ti.cast(np2.y,ti.i32),sim_res_y-2))
        outside=np2.x<offset_x or np2.x>offset_x+disp_res_x or np2.y<offset_y or np2.y>offset_y+disp_res_y
        if outside or initial_mask[nx,ny]>0.5: tv=ti.Vector([0.0,0.0])
        p_vel[i]=tv; p_pos[i]+=tv*dt

@ti.kernel
def clear_filing_overlay():
    for I in ti.grouped(filing_outer_overlay):
        filing_outer_overlay[I] = 0.0
        filing_inner_overlay[I] = 0.0


@ti.kernel
def rasterize_filing_overlay(B: ti.template()):
    for i in p_pos:
        pos = p_pos[i]
        ix = ti.max(1, ti.min(ti.cast(pos.x, ti.i32), sim_res_x - 2))
        iy = ti.max(1, ti.min(ti.cast(pos.y, ti.i32), sim_res_y - 2))
        Bv = B[ix, iy]
        Bm = Bv.norm()
        outside = (
            pos.x < offset_x
            or pos.x > offset_x + disp_res_x
            or pos.y < offset_y
            or pos.y > offset_y + disp_res_y
        )
        if not outside and Bm >= 0.1:
            direction = Bv / (Bm + 1e-5)
            grain_scale = 0.75 + 0.25 * p_mass[i]
            filing_length = ti.max(
                2.0, ti.min(Bm * 2.2 + 2.0, 9.0)
            ) * grain_scale
            center_x = (pos.x - offset_x) / disp_res_x
            center_y = (pos.y - offset_y) / disp_res_y
            half_dx = direction.x * filing_length / disp_res_x * 0.5
            half_dy = direction.y * filing_length / disp_res_y * 0.5
            x0 = (center_x - half_dx) * (disp_res_x - 1)
            y0 = (center_y - half_dy) * (disp_res_y - 1)
            x1 = (center_x + half_dx) * (disp_res_x - 1)
            y1 = (center_y + half_dy) * (disp_res_y - 1)
            line_dx = x1 - x0
            line_dy = y1 - y0
            pixel_length = ti.sqrt(line_dx * line_dx + line_dy * line_dy)
            steps = ti.max(
                1,
                ti.min(
                    FILING_RASTER_STEPS - 1,
                    ti.cast(ti.ceil(pixel_length), ti.i32),
                ),
            )
            shade = ti.cast(150 + (i * 29) % 66, ti.f32) / 255.0
            normal_x = -line_dy / (pixel_length + 1e-5)
            normal_y = line_dx / (pixel_length + 1e-5)
            normal_sign = 1.0
            if i % 2 != 0:
                normal_sign = -1.0

            for sample in ti.static(range(FILING_RASTER_STEPS)):
                if sample <= steps:
                    t = ti.cast(sample, ti.f32) / ti.cast(steps, ti.f32)
                    px = ti.cast(ti.round(x0 + line_dx * t), ti.i32)
                    py = ti.cast(ti.round(y0 + line_dy * t), ti.i32)
                    if 0 <= px < disp_res_x and 0 <= py < disp_res_y:
                        ti.atomic_max(filing_inner_overlay[px, py], shade)
                    for side_index in ti.static(range(2)):
                        side = ti.cast(side_index, ti.f32) * normal_sign
                        qx = px + ti.cast(ti.round(normal_x * side), ti.i32)
                        qy = py + ti.cast(ti.round(normal_y * side), ti.i32)
                        if 0 <= qx < disp_res_x and 0 <= qy < disp_res_y:
                            ti.atomic_max(filing_outer_overlay[qx, qy], 1.0)

            if i % 3 == 0:
                for marker in ti.static(range(3)):
                    t = ti.cast(marker, ti.f32) * 0.5
                    px = ti.cast(ti.round(x0 + line_dx * t), ti.i32)
                    py = ti.cast(ti.round(y0 + line_dy * t), ti.i32)
                    if 0 <= px < disp_res_x and 0 <= py < disp_res_y:
                        ti.atomic_max(
                            filing_inner_overlay[px, py], 220.0 / 255.0
                        )


@ti.kernel
def blend_filing_overlay(colorf: ti.template()):
    for di, dj in colorf:
        outer = filing_outer_overlay[di, dj]
        inner = filing_inner_overlay[di, dj]
        if outer > 0.0:
            colorf[di, dj] = ti.Vector([
                68.0 / 255.0,
                74.0 / 255.0,
                76.0 / 255.0,
            ])
        if inner > 0.0:
            colorf[di, dj] = ti.Vector([
                inner,
                ti.min(inner + 4.0 / 255.0, 1.0),
                ti.min(inner + 5.0 / 255.0, 1.0),
            ])


def render_filings_to_color(B, colorf):
    clear_filing_overlay()
    rasterize_filing_overlay(B)
    blend_filing_overlay(colorf)


@ti.kernel
def sample_direction_arrows(B: ti.template()):
    for idx in direction_arrow_dir:
        col = idx // DIRECTION_ARROW_ROWS
        row = idx - col * DIRECTION_ARROW_ROWS
        di = DIRECTION_ARROW_START + col * DIRECTION_ARROW_STEP
        dj = DIRECTION_ARROW_START + row * DIRECTION_ARROW_STEP
        Bv = B[di + offset_x, dj + offset_y]
        mag = Bv.norm()
        direction = ti.Vector([0.0, 0.0])
        if mag > 1e-12:
            direction = Bv / mag
        direction_arrow_dir[idx] = ti.Vector([
            direction.x / ti.cast(disp_res_x, ti.f32) * STREAMLINE_LENGTH,
            direction.y / ti.cast(disp_res_y, ti.f32) * STREAMLINE_LENGTH,
        ])


def calculate_streamline(B_field):
    sample_direction_arrows(B_field)
    return direction_arrow_dir.to_numpy(), _direction_arrow_starts_np

@ti.func
def _phase(v: ti.f32) -> ti.f32:
    return v * line_freq_field[()]

@ti.kernel
def compute_and_render(A: ti.template(), colorf: ti.template(), mu_f: ti.template(),
                       highlight_A_field: ti.template(), highlight_count: ti.i32):
    gf = line_freq_field[()]
    gt = line_thickness_field[()]
    for di, dj in colorf:
        i = di + offset_x; j = dj + offset_y
        il = ti.max(i-1, 0); ir = ti.min(i+1, sim_res_x-1)
        jd = ti.max(j-1, 0); ju = ti.min(j+1, sim_res_y-1)
        phase_c = _phase(A[i, j])
        pl = _phase(A[il, j]); pr = _phase(A[ir, j])
        pd = _phase(A[i, jd]); pu = _phase(A[i, ju])
        dfx = (pr - pl) * 0.5; dfy = (pu - pd) * 0.5
        gn = ti.sqrt(dfx**2 + dfy**2) + 1e-7; lpp = 1.0 / gn
        frac = phase_c - ti.floor(phase_c); dtc = ti.min(frac, 1.0 - frac)
        is_line = 1.0 - ti.math.smoothstep(gt - 0.5, gt + 0.5, dtc / gn)
        coverage = ti.min((gt * 1.5) / lpp, 1.0)
        blend = 1.0 - ti.math.smoothstep(1.5, 3.0, lpp)
        bg_vis = is_line * (1.0 - blend) + coverage * blend
        is_entity = mu_f[i, j] > 1.1
        is_magnet_body = mu_base_field[i, j] > 1.1

        bg_col = ti.Vector([0.10, 0.10, 0.10]) if is_entity else ti.Vector([0.0, 0.0, 0.0])
        line_col = ti.Vector([0.55, 0.55, 0.55]) if is_magnet_body else ti.Vector([1.0, 1.0, 1.0])
        pixel = bg_col * (1.0 - bg_vis) + line_col * bg_vis

        max_rv = 0.0
        mac = A[i, j]
        for k in range(highlight_count):
            dp = (ti.abs(mac - highlight_A_field[k]) * gf) / gn; pt = 2.5
            max_rv = ti.max(max_rv, 1.0 - ti.math.smoothstep(pt*0.5-0.5, pt*0.5+0.5, dp))
        red_vis = ti.min(max_rv * bg_vis * 1.6, 1.0)
        if red_vis > 0.0:
            # Negative green/blue cancels the intensity heatmap during additive composition.
            pixel = pixel * (1.0 - red_vis) + ti.Vector([1.0, -0.75, -0.55]) * red_vis
        colorf[di, dj] = pixel

def initialize_render_lut():
    global _viridis_initialized
    if _viridis_initialized:
        return
    lut = plt.cm.viridis(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    viridis_lut.from_numpy(lut)
    _viridis_initialized = True


@ti.func
def _viridis_color(value: ti.f32):
    t = ti.min(1.0, ti.max(0.0, value))
    idx = ti.min(255, ti.cast(t * 256.0, ti.i32))
    return viridis_lut[idx]


@ti.kernel
def compose_magnetic_frame(
    A: ti.template(),
    B: ti.template(),
    colorf: ti.template(),
    mu_f: ti.template(),
    highlight_A_field: ti.template(),
    highlight_count: ti.i32,
    show_intensity: ti.i32,
    show_fieldline: ti.i32,
):
    """Compose heatmap and field lines on the GPU with one host transfer."""
    gf = line_freq_field[()]
    gt = line_thickness_field[()]
    for di, dj in colorf:
        i = di + offset_x
        j = dj + offset_y

        heat = ti.Vector([0.0, 0.0, 0.0])
        if show_intensity != 0:
            be = B[i, j].norm() * 0.02
            heat = _viridis_color(be / (be + 1.0))

        overlay = ti.Vector([0.0, 0.0, 0.0])
        if show_fieldline != 0:
            il = ti.max(i-1, 0)
            ir = ti.min(i+1, sim_res_x-1)
            jd = ti.max(j-1, 0)
            ju = ti.min(j+1, sim_res_y-1)
            phase_c = _phase(A[i, j])
            pl = _phase(A[il, j])
            pr = _phase(A[ir, j])
            pd = _phase(A[i, jd])
            pu = _phase(A[i, ju])
            dfx = (pr - pl) * 0.5
            dfy = (pu - pd) * 0.5
            gn = ti.sqrt(dfx**2 + dfy**2) + 1e-7
            lpp = 1.0 / gn
            frac = phase_c - ti.floor(phase_c)
            dtc = ti.min(frac, 1.0 - frac)
            is_line = 1.0 - ti.math.smoothstep(gt - 0.5, gt + 0.5, dtc / gn)
            coverage = ti.min((gt * 1.5) / lpp, 1.0)
            blend = 1.0 - ti.math.smoothstep(1.5, 3.0, lpp)
            bg_vis = is_line * (1.0 - blend) + coverage * blend
            is_entity = mu_f[i, j] > 1.1
            is_magnet_body = mu_base_field[i, j] > 1.1

            bg_col = (
                ti.Vector([0.10, 0.10, 0.10])
                if is_entity
                else ti.Vector([0.0, 0.0, 0.0])
            )
            line_col = (
                ti.Vector([0.55, 0.55, 0.55])
                if is_magnet_body
                else ti.Vector([1.0, 1.0, 1.0])
            )
            overlay = bg_col * (1.0 - bg_vis) + line_col * bg_vis

            max_rv = 0.0
            mac = A[i, j]
            for k in range(highlight_count):
                dp = (ti.abs(mac - highlight_A_field[k]) * gf) / gn
                pt = 2.5
                max_rv = ti.max(
                    max_rv,
                    1.0 - ti.math.smoothstep(pt*0.5-0.5, pt*0.5+0.5, dp),
                )
            red_vis = ti.min(max_rv * bg_vis * 1.6, 1.0)
            if red_vis > 0.0:
                overlay = (
                    overlay * (1.0 - red_vis)
                    + ti.Vector([1.0, -0.75, -0.55]) * red_vis
                )

        composed = heat + overlay
        for channel in ti.static(range(3)):
            composed[channel] = ti.min(1.0, ti.max(0.0, composed[channel]))
        colorf[di, dj] = composed

def warm_optional_visualization_kernels(highlight_field, highlight_count):
    """Compile optional Direction/Filings kernels before their first toggle."""
    compute_B_field(A_field, B_field)
    compose_magnetic_frame(
        A_field,
        B_field,
        color_field,
        mu_field,
        highlight_field,
        highlight_count,
        1,
        1,
    )
    init_particles()
    compute_density_grid()
    update_particles_fast()
    render_filings_to_color(B_field, color_field)
    # Restore the normal frame after the filing overlay warm-up.
    compose_magnetic_frame(
        A_field,
        B_field,
        color_field,
        mu_field,
        highlight_field,
        highlight_count,
        1,
        1,
    )
    sample_direction_arrows(B_field)
    direction_arrow_dir.to_numpy()
    color_field.to_numpy()

def update_auto_highlights(highlight_field, highlight_count):
    crop = A_field.to_numpy()[offset_x:offset_x+disp_res_x, offset_y:offset_y+disp_res_y]
    phase = crop * float(line_freq_field[()])
    finite = np.isfinite(phase)
    if not finite.any() or float(np.ptp(phase[finite])) < 1e-8:
        highlight_field.fill(0.0)
        highlight_count[()] = 0
        return 0

    nearest = np.rint(phase[finite]).astype(np.int32)
    distance = np.abs(phase[finite] - nearest)
    line_ids, counts = np.unique(nearest[distance < 0.08], return_counts=True)
    min_visible_pixels = max(20, int(phase.size * 0.0002))
    line_ids = line_ids[counts >= min_visible_pixels]
    if line_ids.size < 3:
        line_ids = np.unique(nearest)

    sample_positions = np.rint(np.array([0.2, 0.5, 0.8]) * (line_ids.size - 1)).astype(np.int32)
    selected_ids = line_ids[sample_positions]
    levels = (selected_ids.astype(np.float32) / float(line_freq_field[()])).astype(np.float32)
    highlight_field.from_numpy(levels)
    highlight_count[()] = 3
    return 3


# ──────────────────────────────────────────────
#  场景设置内核
# ──────────────────────────────────────────────
def initialize_materials():
    mu_base_field.fill(1.0); sigma_base_field.fill(0.0); initial_mask.fill(0)
    mu_field.fill(1.0); sigma_field.fill(0.0); inv_mu_field.fill(1.0)

def reset_scene():
    A_field.fill(0.0); J_field.fill(0.0); rhs_eff_field.fill(0.0)
    fixed_A_mask.fill(0.0); fixed_A_value.fill(0.0)
    mu_base_field.fill(1.0); sigma_base_field.fill(0.0)

@ti.kernel
def setup_single_wire():
    for I in ti.grouped(J_field): J_field[I]=0.0
    cx,cy=sim_res_x//2,sim_res_y//2; J_field[cx,cy]=1000.0; initial_mask[cx,cy]=1.0

@ti.kernel
def setup_dipole():
    for i,j in J_field: J_field[i,j]=0.0
    cx,cy=sim_res_x//2,sim_res_y//2; off=75
    J_field[cx-off,cy]=2000.0; initial_mask[cx-off,cy]=1.0
    J_field[cx+off,cy]=-2000.0; initial_mask[cx+off,cy]=1.0

@ti.kernel
def setup_uniform_field():
    cy = sim_res_y // 2
    B0 = 20.0
    margin = 2
    for i, j in J_field:
        J_field[i, j] = 0.0
        initial_mask[i, j] = 0.0
        fixed_A_mask[i, j] = 0.0
        fixed_A_value[i, j] = 0.0
        A_field[i, j] = B0 * (j - cy)
        if i < margin or i >= sim_res_x - margin or j < margin or j >= sim_res_y - margin:
            fixed_A_mask[i, j] = 1.0
            fixed_A_value[i, j] = B0 * (j - cy)
            A_field[i, j] = fixed_A_value[i, j]
            initial_mask[i, j] = 1.0

@ti.kernel
def setup_quadrupole():
    for I in ti.grouped(J_field): J_field[I]=0.0
    cx,cy=sim_res_x//2,sim_res_y//2; d=30; s=2000.0
    J_field[cx-d,cy+d]=s;  initial_mask[cx-d,cy+d]=1.0
    J_field[cx+d,cy+d]=-s; initial_mask[cx+d,cy+d]=1.0
    J_field[cx-d,cy-d]=-s; initial_mask[cx-d,cy-d]=1.0
    J_field[cx+d,cy-d]=s;  initial_mask[cx+d,cy-d]=1.0

@ti.kernel
def setup_solenoid():
    for I in ti.grouped(J_field): J_field[I]=0.0
    cx,cy=sim_res_x//2,sim_res_y//2; cl=300; cr=75; th=15; cd=0.5
    for i,j in J_field:
        if cx-cl//2<i<cx+cl//2:
            if cy+cr<j<cy+cr+th:   J_field[i,j]=cd;  initial_mask[i,j]=1.0
            elif cy-cr-th<j<cy-cr: J_field[i,j]=-cd; initial_mask[i,j]=1.0

@ti.kernel
def setup_horseshoe():
    for I in ti.grouped(J_field): J_field[I]=0.0
    for I in ti.grouped(mu_base_field): mu_base_field[I]=1.0
    mw=60; gh=53; ah=150; imu=8.0; mc=20.0; et=6
    cx=sim_res_x//2-(ah-gh-mw)//2; cy=sim_res_y//2
    for i,j in mu_base_field:
        dx=i-cx; dy=j-cy
        # Inverse-map world coordinates through a 90-degree clockwise rotation.
        hx=-dy; hy=dx
        ii=False; inn=False; out=False
        if (-gh-mw<hx<-gh) and (0<hy<ah):
            ii=True
            if hx>-gh-et: inn=True
            if hx<-gh-mw+et: out=True
        elif (gh<hx<gh+mw) and (0<hy<ah):
            ii=True
            if hx<gh+et: inn=True
            if hx>gh+mw-et: out=True
        elif hy<=0:
            dist=ti.sqrt(float(hx)**2+float(hy)**2)
            if gh<dist<gh+mw:
                ii=True
                if dist<gh+et: inn=True
                if dist>gh+mw-et: out=True
        if ii:
            mu_field[i,j]=imu; mu_base_field[i,j]=imu; initial_mask[i,j]=1.0
            if inn: J_field[i,j]=mc
            elif out: J_field[i,j]=-mc

@ti.kernel
def setup_two_magnets():
    for I in ti.grouped(J_field): J_field[I]=0.0
    for I in ti.grouped(mu_base_field): mu_base_field[I]=1.0
    cx,cy=sim_res_x//2,sim_res_y//2; mw=150; mh=60; gh=60; et=6; imu=8.0; cv=80.0
    for i,j in mu_base_field:
        dx=i-cx; dy=j-cy; ii=False; it=False; ib=False
        if (-gh-mw<dx<-gh) and (-mh//2<dy<mh//2):
            ii=True; (it, ib) = (True, False) if dy>mh//2-et else ((False, True) if dy<-mh//2+et else (False, False))
        elif (gh<dx<gh+mw) and (-mh//2<dy<mh//2):
            ii=True; (it, ib) = (True, False) if dy>mh//2-et else ((False, True) if dy<-mh//2+et else (False, False))
        if ii:
            mu_field[i,j]=imu; mu_base_field[i,j]=imu; initial_mask[i,j]=1.0
            if it: J_field[i,j]=cv
            elif ib: J_field[i,j]=-cv

@ti.kernel
def update_materials_with_mask(input_mask: ti.template(), mu_field: ti.template(),
                               sigma_field: ti.template(), initial_mask: ti.template(),
                               mu: ti.f32, sigma: ti.f32):
    for i,j in input_mask:
        if initial_mask[i,j]==0.0:
            if input_mask[i,j]>0.9: mu_field[i,j]=mu; sigma_field[i,j]=sigma
            else: mu_field[i,j]=mu_base_field[i,j]; sigma_field[i,j]=sigma_base_field[i,j]
        else:
            mu_field[i,j]=mu_base_field[i,j]; sigma_field[i,j]=sigma_base_field[i,j]

def set_up_scene(n):
    reset_scene()
    initialize_materials()
    reset_mask_change_cache()

    if n == 0:      # Single Wire
        line_freq_field[()] = 0.015; line_thickness_field[()] = 1.00
    elif n == 1:    # Dipole
        line_freq_field[()] = 0.010; line_thickness_field[()] = 0.95
    elif n == 2:    # Uniform Field
        line_freq_field[()] = 0.003; line_thickness_field[()] = 1.00
    elif n == 3:    # Quadrupole
        line_freq_field[()] = 0.007; line_thickness_field[()] = 0.90
    elif n == 4:    # Solenoid
        line_freq_field[()] = 0.004; line_thickness_field[()] = 0.85
    elif n == 5:    # Horseshoe
        line_freq_field[()] = 0.00030; line_thickness_field[()] = 0.42
    else:           # Two Magnets
        line_freq_field[()] = 0.00018; line_thickness_field[()] = 0.40

    SCENE_FUNCTIONS = [
        setup_single_wire, setup_dipole, setup_uniform_field,
        setup_quadrupole, setup_solenoid, setup_horseshoe, setup_two_magnets
    ]
    SCENE_FUNCTIONS[n]()


# ──────────────────────────────────────────────
#  主循环
# ──────────────────────────────────────────────
def main(web_ui_only=False, max_frames=None, stats_json=None, initial_direction=False):
    global _pending_command, _latest_frame_bytes, _latest_frame_seq
    global _latest_arrows_json, _latest_longpress_json

    vis_magnetic_intensity = True
    vis_magnetic_field_line = True
    vis_magnetic_field_direction = initial_direction
    vis_iron_filings_method = False
    particle_initialized = False

    initialize_render_lut()
    highlight_A_field_gpu = ti.field(ti.f32, shape=3)
    highlight_count_gpu = ti.field(ti.i32, shape=())
    highlight_count_gpu[()] = 0

    initial_scene_number = 0
    set_up_scene(initial_scene_number)

    print("Pre-computing...")
    update_inv_mu()
    compute_preconditioner()
    solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
    highlight_count_value = update_auto_highlights(
        highlight_A_field_gpu, highlight_count_gpu
    )
    # Compile optional Direction and Iron Filings paths before their first toggle.
    warm_optional_visualization_kernels(
        highlight_A_field_gpu, highlight_count_value
    )
    print("Warm up done.")

    input_material = interactive_materials["iron"]
    gui = None if web_ui_only else ti.GUI(
        "Magnetic Field Simulation", (disp_res_x, disp_res_y)
    )

    threading.Thread(target=read_shared_memory, daemon=True).start()
    threading.Thread(target=_ws_thread_func, daemon=True).start()

    last_input_mask.fill(0.0)
    first_mask_frame = True
    accepted_lowres_mask = None
    last_frame_enc = time.perf_counter()
    cached_arrow_dirs = cached_arrow_starts = None
    cached_arrow_quantized = None
    last_direction_update = -1e9
    arrows_dirty = True
    field_dirty = False
    b_initialized = True
    highlights_pending = current_solver_active()
    frame_count = 0
    stats = {
        "mode": "web_ui" if web_ui_only else "local_gui_plus_websocket",
        "frame_ms": [],
        "sim_ms": [],
        "compose_ms": [],
        "gui_ms": [],
        "encode_ms": [],
        "arrows_ms": [],
        "filings_ms": [],
        "encoded_frames": 0,
    }
    run_start = time.perf_counter()

    def start_scene_transition(scene_number):
        set_up_scene(scene_number)
        update_inv_mu()
        compute_preconditioner()
        start_current_system(tol=1e-6, rebuild_rhs=True)
        return continue_current_system(max_iters=SCENE_CHANGE_SOLVE_ITERS)

    while web_ui_only or gui.running:
        frame_start = time.perf_counter()

        # ── ① Consume the newest WebSocket control command ──
        with _cmd_lock:
            cmd = _pending_command
            _pending_command = None

        if cmd:
            action = cmd.get("action")
            value = cmd.get("value")
            if action == "set_scene" and isinstance(value, int) and 0 <= value < 7:
                initial_scene_number = value
                start_scene_transition(initial_scene_number)
                highlight_count_value = update_auto_highlights(
                    highlight_A_field_gpu, highlight_count_gpu
                )
                highlights_pending = current_solver_active()
                field_dirty = True
                arrows_dirty = True
                particle_initialized = False
                last_input_mask.fill(0.0)
                first_mask_frame = True
                print(f"[WS] Scene -> {SCENE_NAMES[initial_scene_number]}")
            elif action == "set_material" and value in interactive_materials:
                input_material = interactive_materials[value]
                update_materials_with_mask(
                    input_mask,
                    mu_field,
                    sigma_field,
                    initial_mask,
                    input_material["mu"],
                    input_material["sigma"],
                )
                update_inv_mu()
                compute_preconditioner()
                start_current_system(tol=INTERACTIVE_SOLVE_TOL, rebuild_rhs=True)
                continue_current_system(max_iters=INTERACTIVE_SOLVE_ITERS)
                highlights_pending = True
                field_dirty = True
                arrows_dirty = True
                print(f"[WS] Material -> {value}")
            elif action == "toggle_intensity":
                vis_magnetic_intensity = not vis_magnetic_intensity
            elif action == "toggle_fieldline":
                vis_magnetic_field_line = not vis_magnetic_field_line
            elif action == "toggle_direction":
                vis_magnetic_field_direction = not vis_magnetic_field_direction
                arrows_dirty = True
            elif action == "toggle_filings":
                vis_iron_filings_method = not vis_iron_filings_method
                particle_initialized = False
            elif action == "reset":
                start_scene_transition(initial_scene_number)
                highlight_count_value = update_auto_highlights(
                    highlight_A_field_gpu, highlight_count_gpu
                )
                highlights_pending = current_solver_active()
                field_dirty = True
                arrows_dirty = True
                particle_initialized = False
                last_input_mask.fill(0.0)
                first_mask_frame = True
                print("[WS] Scene reset")
            elif action == "set_freeze_input" and not bool(value):
                release_frozen_input()

        # ── ② Long-press state and local keyboard controls ──
        with shared_mask_lock:
            lp_progress = freeze_progress
            lp_frozen = freeze_input
            lp_touching = has_touch_current
        with _latest_arrows_lock:
            _latest_longpress_json = json.dumps(
                {
                    "type": "longpress",
                    "progress": lp_progress,
                    "frozen": lp_frozen,
                    "has_touch": lp_touching,
                },
                separators=(",", ":"),
            )

        if gui is not None and gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                start_scene_transition(initial_scene_number)
                highlight_count_value = update_auto_highlights(
                    highlight_A_field_gpu, highlight_count_gpu
                )
                highlights_pending = current_solver_active()
                field_dirty = True
                arrows_dirty = True
                particle_initialized = False
                last_input_mask.fill(0.0)
                first_mask_frame = True
            elif e.key in [ti.GUI.LEFT, ti.GUI.RIGHT]:
                delta = 1 if e.key == ti.GUI.RIGHT else -1
                initial_scene_number = (initial_scene_number + delta) % 7
                start_scene_transition(initial_scene_number)
                highlight_count_value = update_auto_highlights(
                    highlight_A_field_gpu, highlight_count_gpu
                )
                highlights_pending = current_solver_active()
                field_dirty = True
                arrows_dirty = True
                particle_initialized = False
                last_input_mask.fill(0.0)
                first_mask_frame = True
            elif e.key == "i":
                vis_magnetic_intensity = not vis_magnetic_intensity
            elif e.key == "l":
                vis_magnetic_field_line = not vis_magnetic_field_line
            elif e.key == "d":
                vis_magnetic_field_direction = not vis_magnetic_field_direction
                arrows_dirty = True
            elif e.key == "f":
                vis_iron_filings_method = not vis_iron_filings_method
                particle_initialized = False

        # ── ③ Input update and time-sliced PCG solve ──
        sim_start = time.perf_counter()
        got_mask = process_mask_update(threshold=160.0)
        mask_changed = False
        if got_mask and last_lowres_mask_np is not None:
            if accepted_lowres_mask is None:
                changed_cells = int(np.count_nonzero(last_lowres_mask_np))
            else:
                changed_cells = int(
                    np.count_nonzero(last_lowres_mask_np != accepted_lowres_mask)
                )
            mask_changed = (
                first_mask_frame or changed_cells > LOWRES_MASK_DIFF_THRESH
            )

        if mask_changed:
            update_materials_with_mask(
                input_mask,
                mu_field,
                sigma_field,
                initial_mask,
                input_material["mu"],
                input_material["sigma"],
            )
            update_inv_mu()
            compute_preconditioner()
            start_current_system(tol=INTERACTIVE_SOLVE_TOL, rebuild_rhs=True)
            continue_current_system(max_iters=INTERACTIVE_SOLVE_ITERS)
            copy_mask(input_mask, last_input_mask)
            accepted_lowres_mask = last_lowres_mask_np.copy()
            first_mask_frame = False
            highlights_pending = True
            field_dirty = True
            arrows_dirty = True
        elif current_solver_active():
            iters = continue_current_system(max_iters=SETTLE_SOLVE_ITERS)
            if iters > 0:
                field_dirty = True
                arrows_dirty = True

        if highlights_pending and not current_solver_active():
            highlight_count_value = update_auto_highlights(
                highlight_A_field_gpu, highlight_count_gpu
            )
            highlights_pending = False
        stats["sim_ms"].append((time.perf_counter() - sim_start) * 1000.0)

        # ── ④ Render only at display cadence; B is cached until A changes ──
        now = time.perf_counter()
        should_encode_frame = now - last_frame_enc >= 1 / TARGET_RENDER_FPS
        should_render_frame = gui is not None or should_encode_frame

        compose_start = time.perf_counter()
        crop_img = None
        filings_elapsed_ms = 0.0
        filings_start = None
        if should_render_frame:
            if field_dirty or not b_initialized:
                compute_B_field(A_field, B_field)
                field_dirty = False
                b_initialized = True
                arrows_dirty = True

            compose_magnetic_frame(
                A_field,
                B_field,
                color_field,
                mu_field,
                highlight_A_field_gpu,
                highlight_count_value,
                int(vis_magnetic_intensity),
                int(vis_magnetic_field_line),
            )

            if vis_iron_filings_method:
                filings_start = time.perf_counter()
                if not particle_initialized:
                    init_particles()
                    for _ in range(10):
                        compute_density_grid()
                        update_particles_fast()
                    particle_initialized = True
                else:
                    compute_density_grid()
                    update_particles_fast()
                render_filings_to_color(B_field, color_field)

            crop_img = color_field.to_numpy()
            if filings_start is not None:
                filings_elapsed_ms = (
                    time.perf_counter() - filings_start
                ) * 1000.0
        stats["compose_ms"].append((time.perf_counter() - compose_start) * 1000.0)
        stats["filings_ms"].append(filings_elapsed_ms)

        gui_start = time.perf_counter()
        if gui is not None and crop_img is not None:
            gui.set_image(crop_img)

        arrows_start = time.perf_counter()
        if should_render_frame and vis_magnetic_field_direction:
            update_due = (
                now - last_direction_update >= 1 / DIRECTION_UPDATE_FPS
            )
            if (
                (arrows_dirty and update_due)
                or cached_arrow_dirs is None
                or cached_arrow_quantized is None
            ):
                cached_arrow_dirs, cached_arrow_starts = calculate_streamline(B_field)
                arrow_rows = np.column_stack(
                    (cached_arrow_starts, cached_arrow_dirs)
                )
                quantized = np.rint(
                    arrow_rows * DIRECTION_QUANTIZATION
                ).astype(np.int16)
                if (
                    cached_arrow_quantized is None
                    or not np.array_equal(quantized, cached_arrow_quantized)
                ):
                    flat = quantized.astype(np.int32).reshape(-1).tolist()
                    _set_latest_arrows_json(
                        json.dumps(
                            {
                                "type": "arrows",
                                "scale": DIRECTION_QUANTIZATION,
                                "data": flat,
                            },
                            separators=(",", ":"),
                        )
                    )
                    cached_arrow_quantized = quantized
                last_direction_update = now
                arrows_dirty = False
            if gui is not None and cached_arrow_dirs is not None:
                gui.arrows(
                    orig=cached_arrow_starts,
                    direction=cached_arrow_dirs,
                    radius=2,
                    color=0xFFFFFF,
                )
        elif should_render_frame:
            _set_latest_arrows_json('{"type":"arrows","data":[]}')
            cached_arrow_quantized = None
        stats["arrows_ms"].append((time.perf_counter() - arrows_start) * 1000.0)

        if gui is not None:
            gui.show()
        stats["gui_ms"].append((time.perf_counter() - gui_start) * 1000.0)

        # ── ⑤ Encode the already composited RGB frame ──
        encode_ms = 0.0
        if should_encode_frame and crop_img is not None:
            encode_start = time.perf_counter()
            img_u8 = (crop_img * 255).clip(0, 255).astype(np.uint8)
            img_u8 = np.ascontiguousarray(
                np.flipud(np.transpose(img_u8, (1, 0, 2)))
            )
            bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            if ok:
                frame_bytes = encoded.tobytes()
            else:
                buf = io.BytesIO()
                Image.fromarray(img_u8).save(buf, format="JPEG", quality=85)
                frame_bytes = buf.getvalue()

            with _latest_frame_lock:
                _latest_frame_bytes = frame_bytes
                _latest_frame_seq += 1
            last_frame_enc = now
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            stats["encoded_frames"] += 1
        stats["encode_ms"].append(encode_ms)

        stats["frame_ms"].append((time.perf_counter() - frame_start) * 1000.0)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

        # Avoid a busy-spin once a static field is fully converged.
        if web_ui_only and not should_encode_frame and not current_solver_active():
            remaining = (1 / TARGET_RENDER_FPS) - (time.perf_counter() - last_frame_enc)
            if remaining > 0.0:
                # Sleep(0) yields the CPU without Windows' coarse timer rounding,
                # which otherwise drops a nominal display cadence below its configured target.
                time.sleep(0)

    if stats_json:
        def summarize(values):
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0:
                return {"avg": 0.0, "p95": 0.0}
            return {
                "avg": float(arr.mean()),
                "p95": float(np.percentile(arr, 95)),
            }

        total_s = time.perf_counter() - run_start
        summary = {
            "simulator": "magnetic",
            "mode": stats["mode"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frames": frame_count,
            "duration_s": total_s,
            "fps_avg": (frame_count / total_s) if total_s > 0 else 0.0,
            "encoded_fps": (
                stats["encoded_frames"] / total_s if total_s > 0 else 0.0
            ),
            "frame_ms": summarize(stats["frame_ms"]),
            "sim_ms": summarize(stats["sim_ms"]),
            "compose_ms": summarize(stats["compose_ms"]),
            "gui_ms": summarize(stats["gui_ms"]),
            "encode_ms": summarize(stats["encode_ms"]),
            "arrows_ms": summarize(stats["arrows_ms"]),
            "filings_ms": summarize(stats["filings_ms"]),            "encoded_frames": stats["encoded_frames"],
            "web_ui_only": web_ui_only,
            "notes": (
                "Time-sliced persistent PCG with cached B field and fused GPU "
                "composition; browser decode/canvas timing not included."
            ),
        }
        os.makedirs(os.path.dirname(stats_json) or ".", exist_ok=True)
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="serve frames to the HTML UI without opening the local Taichi GUI",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--direction", action="store_true")
    args = parser.parse_args()
    main(
        web_ui_only=args.web_ui,
        max_frames=args.max_frames,
        stats_json=args.stats_json,
        initial_direction=args.direction,
    )
