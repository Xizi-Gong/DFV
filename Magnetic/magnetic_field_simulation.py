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
from PIL import Image, ImageDraw
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

mu_field         = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
inv_mu_field     = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_field      = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
mu_base_field    = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_base_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

filing_density = ti.field(ti.f32, shape=(sim_res_x, sim_res_y))

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

direction_arrow_start = ti.Vector.field(2, dtype=ti.f32, shape=DIRECTION_ARROW_COUNT)
direction_arrow_dir   = ti.Vector.field(2, dtype=ti.f32, shape=DIRECTION_ARROW_COUNT)


# ──────────────────────────────────────────────
#  WebSocket 状态与控制
# ──────────────────────────────────────────────
WS_HOST = "0.0.0.0"
WS_PORT = 8765

_latest_frame_bytes = None
_latest_frame_lock  = Lock()          
_latest_arrows_json = None
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

async def _ws_handler(websocket):
    try: await websocket.send(_WS_INIT_MSG)
    except Exception as e:
        print(f"[WS] 初始化失败: {e}")
        return

    with _clients_lock: _connected_clients.add(websocket)
    print(f"[WS] 客户端已连接 在线: {len(_connected_clients)}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                _handle_ws_message(message)
    except websockets.exceptions.ConnectionClosed: pass
    finally:
        with _clients_lock: _connected_clients.discard(websocket)
        print(f"[WS] 客户端断开 在线: {len(_connected_clients)}")

async def _frame_broadcaster():
    while True:
        await asyncio.sleep(1 / 30)
        with _latest_frame_lock: frame = _latest_frame_bytes
        if frame is None: continue
        with _latest_arrows_lock: longpress_json = _latest_longpress_json
        with _clients_lock: targets = list(_connected_clients)
        if not targets: continue
        with _latest_arrows_lock: arrows_json = _latest_arrows_json
        dead = []
        for ws in targets:
            try:
                if longpress_json is not None: await ws.send(longpress_json)
                await ws.send(frame)
                if arrows_json is not None: await ws.send(arrows_json)
            except Exception: dead.append(ws)
        if dead:
            with _clients_lock:
                for ws in dead: _connected_clients.discard(ws)

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
    for I in ti.grouped(v1): s += v1[I] * v2[I]
    out[()] = s

@ti.kernel
def _cg_compute_thresh(b: ti.template(), tol: ti.f32):
    s = 0.0
    for I in ti.grouped(b): s += b[I] * b[I]
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
        if fixed_A_mask[I] > 0.5: x[I] = fixed_A_value[I]

@ti.kernel
def build_effective_rhs():
    for i, j in J_field:
        if fixed_A_mask[i, j] > 0.5:
            rhs_eff_field[i, j] = fixed_A_value[i, j]
        else:
            imc = sample_inv_mu(i, j)
            wL  = harm(imc, sample_inv_mu(i-1, j)); wR  = harm(imc, sample_inv_mu(i+1, j))
            wD  = harm(imc, sample_inv_mu(i, j-1)); wU  = harm(imc, sample_inv_mu(i, j+1))
            b = J_field[i, j]
            if i - 1 >= 0 and fixed_A_mask[i-1, j] > 0.5: b += wL * fixed_A_value[i-1, j]
            if i + 1 < sim_res_x and fixed_A_mask[i+1, j] > 0.5: b += wR * fixed_A_value[i+1, j]
            if j - 1 >= 0 and fixed_A_mask[i, j-1] > 0.5: b += wD * fixed_A_value[i, j-1]
            if j + 1 < sim_res_y and fixed_A_mask[i, j+1] > 0.5: b += wU * fixed_A_value[i, j+1]
            rhs_eff_field[i, j] = b

@ti.kernel
def apply_preconditioner(r: ti.template(), z: ti.template()):
    for I in ti.grouped(r): z[I] = r[I] * precond_field[I]

@ti.kernel
def _cg_update_xr_async(x: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template()):
    alpha = _cg_rsold[()] / (_cg_pAp[()] + 1e-30)
    for I in ti.grouped(x):
        x[I] += alpha * p[I]
        r[I] -= alpha * Ap[I]

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
def _cg_update_xr_checked(x: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template()):
    alpha = _cg_alpha[()]
    for I in ti.grouped(x):
        if _cg_status[()] == 0:
            x[I] += alpha * p[I]
            r[I] -= alpha * Ap[I]

@ti.kernel
def _cg_update_p_pcg(z: ti.template(), p: ti.template()):
    beta = _cg_rsnew[()] / (_cg_rsold[()] + 1e-30)
    _cg_rsold[()] = _cg_rsnew[()]
    for I in ti.grouped(p):
        p[I] = z[I] + beta * p[I]

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
    for I in ti.grouped(r): s += r[I] * r[I]
    _cg_res2[()] = s

def solve_poisson_pcg_safe(x_field, b_field, max_iters=100, tol=1e-4, verbose=False):
    residual_check_interval = 5
    _cg_status[()] = 0
    compute_Ap(x_field, Ap_field)
    compute_r(x_field, r_field, b_field, Ap_field)
    _cg_compute_thresh(b_field, tol)
    _cg_compute_res2(r_field)

    res2 = float(_cg_res2[()])
    thresh = float(_cg_thresh[()])

    if not math.isfinite(res2) or res2 <= thresh:
        return 0

    apply_preconditioner(r_field, z_field)
    copy_field(z_field, p_field)
    _cg_dot_to_field(r_field, z_field, _cg_rsold)

    for it in range(max_iters):
        compute_Ap(p_field, Ap_field)
        _cg_dot_to_field(p_field, Ap_field, _cg_pAp)
        _cg_prepare_alpha_checked()
        _cg_update_xr_checked(x_field, p_field, r_field, Ap_field)
        _cg_compute_res2(r_field)

        should_check = ((it + 1) % residual_check_interval == 0) or (it + 1 == max_iters)
        if should_check:
            if int(_cg_status[()]) != 0:
                return it
            res2 = float(_cg_res2[()])
            if not math.isfinite(res2) or res2 <= thresh:
                return it + 1

        apply_preconditioner(r_field, z_field)
        _cg_dot_to_field(r_field, z_field, _cg_rsnew)
        _cg_prepare_beta_checked()
        _cg_update_p_checked(z_field, p_field)
    return max_iters

def solve_current_system(max_iters=100, tol=1e-3, verbose=False):
    build_effective_rhs()
    apply_fixed_A(A_field)
    iters = solve_poisson_pcg_safe(A_field, rhs_eff_field, max_iters=max_iters, tol=tol, verbose=verbose)
    print(f"[Solver] 迭代完成 iters={iters} res={math.sqrt(float(_cg_res2[()])):.6e} status={_cg_status[()]}")
    apply_fixed_A(A_field)
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
def sample_direction_arrows(B: ti.template()):
    for idx in direction_arrow_start:
        col = idx // DIRECTION_ARROW_ROWS
        row = idx - col * DIRECTION_ARROW_ROWS
        di = DIRECTION_ARROW_START + col * DIRECTION_ARROW_STEP
        dj = DIRECTION_ARROW_START + row * DIRECTION_ARROW_STEP
        ix = di + offset_x
        iy = dj + offset_y
        Bv = B[ix, iy]
        mag = Bv.norm()
        direction = ti.Vector([0.0, 0.0])
        if mag > 1e-12:
            direction = Bv / mag
        direction_arrow_start[idx] = ti.Vector([
            ti.cast(di, ti.f32) / ti.cast(disp_res_x, ti.f32),
            ti.cast(dj, ti.f32) / ti.cast(disp_res_y, ti.f32),
        ])
        direction_arrow_dir[idx] = ti.Vector([
            direction.x / ti.cast(disp_res_x, ti.f32) * STREAMLINE_LENGTH,
            direction.y / ti.cast(disp_res_y, ti.f32) * STREAMLINE_LENGTH,
        ])

def calculate_streamline(B_field):
    sample_direction_arrows(B_field)
    return direction_arrow_dir.to_numpy(), direction_arrow_start.to_numpy()

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

def update_auto_highlights(highlight_field, highlight_count):
    crop = A_field.to_numpy()[offset_x:offset_x+disp_res_x, offset_y:offset_y+disp_res_y]
    phase = crop * float(line_freq_field[()])
    finite = np.isfinite(phase)
    if not finite.any() or float(np.ptp(phase[finite])) < 1e-8:
        highlight_field.fill(0.0)
        highlight_count[()] = 0
        return

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
    global _pending_command, _latest_frame_bytes, _latest_arrows_json, _latest_longpress_json

    vis_magnetic_intensity       = True
    vis_magnetic_field_line      = True
    vis_magnetic_field_direction = initial_direction
    vis_iron_filings_method      = False
    particle_initialized         = False

    highlight_A_field_gpu = ti.field(ti.f32, shape=3)
    highlight_count_gpu = ti.field(ti.i32, shape=())
    highlight_count_gpu[()] = 0

    initial_scene_number = 0
    set_up_scene(initial_scene_number)

    print("Pre-computing...")
    update_inv_mu()
    compute_preconditioner()
    solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
    update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
    print("Warm up done.")

    input_material = interactive_materials["iron"]
    cmap = plt.cm.viridis
    gui = None if web_ui_only else ti.GUI("Magnetic Field Simulation", (disp_res_x, disp_res_y))

    threading.Thread(target=read_shared_memory, daemon=True).start()
    threading.Thread(target=_ws_thread_func, daemon=True).start()

    last_mask_sum  = -1.0
    last_input_mask.fill(0.0)
    first_mask_frame = True
    last_frame_enc = time.time()
    _ws_filings_start = _ws_filings_end = None
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

        # ── ① 消费 WebSocket 控制指令 ──
        with _cmd_lock:
            cmd = _pending_command
            _pending_command = None

        if cmd:
            action = cmd.get("action"); value = cmd.get("value")
            if action == "set_scene" and isinstance(value, int) and 0 <= value < 7:
                initial_scene_number = value
                set_up_scene(initial_scene_number)
                update_inv_mu()
                compute_preconditioner()
                solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
                update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
                particle_initialized = False
                print(f"[WS] 切换场景 → {SCENE_NAMES[initial_scene_number]}")
            elif action == "set_material" and value in interactive_materials:
                input_material = interactive_materials[value]
                update_materials_with_mask(input_mask, mu_field, sigma_field, initial_mask,
                                           input_material["mu"], input_material["sigma"])
                update_inv_mu()
                compute_preconditioner()
                solve_current_system(max_iters=400, tol=1e-3, verbose=True)
                update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
                last_mask_sum = -1.0   
                print(f"[WS] 切换材质 → {value}")
            elif action == "toggle_intensity":   vis_magnetic_intensity      = not vis_magnetic_intensity
            elif action == "toggle_fieldline":   vis_magnetic_field_line     = not vis_magnetic_field_line
            elif action == "toggle_direction":   vis_magnetic_field_direction = not vis_magnetic_field_direction
            elif action == "toggle_filings":     vis_iron_filings_method     = not vis_iron_filings_method
            elif action == "reset":
                set_up_scene(initial_scene_number) 
                update_inv_mu()
                compute_preconditioner()
                solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
                update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
                particle_initialized = False
                print("[WS] 场景已重置")

            elif action == "set_freeze_input" and not bool(value):
                release_frozen_input()
        # ── ② 键盘事件 ──
        with shared_mask_lock:
            lp_progress = freeze_progress
            lp_frozen = freeze_input
            lp_touching = has_touch_current
        with _latest_arrows_lock:
            _latest_longpress_json = json.dumps({
                "type": "longpress", "progress": lp_progress,
                "frozen": lp_frozen, "has_touch": lp_touching,
            })
        if gui is not None and gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE: break
            elif e.key == "r":
                set_up_scene(initial_scene_number)
                update_inv_mu()
                compute_preconditioner()
                solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
                compute_B_field(A_field, B_field)
                update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
                particle_initialized = False
            elif e.key in [ti.GUI.LEFT, ti.GUI.RIGHT]:
                initial_scene_number = (initial_scene_number + (1 if e.key==ti.GUI.RIGHT else -1)) % 7
                set_up_scene(initial_scene_number)
                update_inv_mu()
                compute_preconditioner()
                solve_current_system(max_iters=2000, tol=1e-6, verbose=True)
                update_auto_highlights(highlight_A_field_gpu, highlight_count_gpu)
                particle_initialized = False
            elif e.key == "i": vis_magnetic_intensity      = not vis_magnetic_intensity
            elif e.key == "l": vis_magnetic_field_line     = not vis_magnetic_field_line
            elif e.key == "d": vis_magnetic_field_direction = not vis_magnetic_field_direction
            elif e.key == "f": vis_iron_filings_method     = not vis_iron_filings_method

        # ── ④ 内存掩码输入 + 仿真步进 ──
        sim_start = time.perf_counter()
        got_mask = process_mask_update(threshold=160.0)
        mask_changed = False
        current_mask_sum = last_mask_sum
        if got_mask:
            current_mask_sum = field_sum(input_mask)
            mask_diff = mask_l1_diff(input_mask, last_input_mask)

            MASK_DIFF_THRESH = 100.0
            mask_changed = first_mask_frame or (mask_diff > MASK_DIFF_THRESH)

        if mask_changed:
            update_materials_with_mask(input_mask, mu_field, sigma_field, initial_mask,
                                       input_material["mu"], input_material["sigma"])
            update_inv_mu()
            compute_preconditioner()
            solve_current_system(max_iters=150, tol=1e-3, verbose=True)
            copy_mask(input_mask, last_input_mask)
            last_mask_sum = current_mask_sum
            first_mask_frame = False
        else:
            solve_current_system(max_iters=50, tol=1e-3, verbose=False)

        compute_B_field(A_field, B_field)
        stats["sim_ms"].append((time.perf_counter() - sim_start) * 1000.0)

        # ── ⑤ 渲染合成 ──
        now = time.time()
        should_encode_frame = now - last_frame_enc >= 1 / 30
        should_render_frame = gui is not None or should_encode_frame

        compose_start = time.perf_counter()
        crop_img = None
        if should_render_frame:
            if vis_magnetic_intensity:
                compute_magnetic_intensity(B_field, magnetic_intensity_field)
                be = magnetic_intensity_field.to_numpy() * 0.02
                heatmap_layer = cmap(be / (be + 1.0))[:, :, :3]
            else:
                heatmap_layer = np.zeros((disp_res_x, disp_res_y, 3), dtype=np.float32)

            if vis_magnetic_field_line:
                compute_and_render(A_field, color_field, mu_field,
                                   highlight_A_field_gpu, highlight_count_gpu[()])
                overlay_layer = color_field.to_numpy()
            else:
                overlay_layer = np.zeros((disp_res_x, disp_res_y, 3), dtype=np.float32)

            crop_img = np.clip(heatmap_layer + overlay_layer, 0.0, 1.0)
        stats["compose_ms"].append((time.perf_counter() - compose_start) * 1000.0)

        gui_start = time.perf_counter()
        if gui is not None and crop_img is not None:
            gui.set_image(crop_img)

        arrows_start = time.perf_counter()
        if should_render_frame and vis_magnetic_field_direction:
            dirs, starts = calculate_streamline(B_field)
            if gui is not None:
                gui.arrows(orig=starts, direction=dirs, radius=2, color=0xFFFFFF)
            flat = []
            for (ox, oy), (dx, dy) in zip(starts, dirs):
                flat.extend([round(float(ox),4), round(float(oy),4),
                             round(float(dx),4), round(float(dy),4)])
            with _latest_arrows_lock: _latest_arrows_json = json.dumps({"type":"arrows","data":flat})
        elif should_render_frame:
            with _latest_arrows_lock: _latest_arrows_json = json.dumps({"type":"arrows","data":[]})
        stats["arrows_ms"].append((time.perf_counter() - arrows_start) * 1000.0)

        if should_render_frame and vis_iron_filings_method:
            if not particle_initialized:
                init_particles()
                for _ in range(10): compute_density_grid(); update_particles_fast()
                particle_initialized = True
            else:
                compute_density_grid(); update_particles_fast()
            prepare_particle_lines()
            _ws_filings_start = p_start.to_numpy()
            _ws_filings_end   = p_end.to_numpy()
            if gui is not None:
                gui.lines(begin=_ws_filings_start, end=_ws_filings_end, radius=0.7, color=0xB8C0C2)
        elif should_render_frame:
            _ws_filings_start = _ws_filings_end = None

        if gui is not None:
            gui.show()
        stats["gui_ms"].append((time.perf_counter() - gui_start) * 1000.0)

        # ── ⑥ 编码 JPEG → WebSocket 广播 ──
        encode_ms = 0.0
        if should_encode_frame and crop_img is not None:
            encode_start = time.perf_counter()
            img_u8 = (crop_img * 255).clip(0, 255).astype(np.uint8)
            img_u8 = np.flipud(np.transpose(img_u8, (1, 0, 2)))

            if _ws_filings_start is not None:
                pil_img = Image.fromarray(img_u8)
                draw    = ImageDraw.Draw(pil_img)
                IW, IH  = pil_img.size  

                def n2p(nx, ny): return int(nx * IW), int((1.0 - ny) * IH)

                for idx, ((x0n, y0n), (x1n, y1n)) in enumerate(zip(_ws_filings_start, _ws_filings_end)):
                    if x0n < 0: continue
                    p0 = n2p(x0n, y0n)
                    p1 = n2p(x1n, y1n)
                    shade = 150 + (idx * 29) % 66
                    draw.line([p0, p1], fill=(68, 74, 76), width=2)
                    draw.line([p0, p1], fill=(shade, shade + 4, shade + 5), width=1)
                    if idx % 3 == 0:
                        pm = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
                        draw.point([p0, pm, p1], fill=(220, 224, 225))
                img_u8 = np.array(pil_img)

            buf = io.BytesIO()
            Image.fromarray(img_u8).save(buf, format="JPEG", quality=85)
            with _latest_frame_lock: _latest_frame_bytes = buf.getvalue()
            last_frame_enc = now
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            stats["encoded_frames"] += 1
        stats["encode_ms"].append(encode_ms)

        stats["frame_ms"].append((time.perf_counter() - frame_start) * 1000.0)
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break

    if stats_json:
        def summarize(values):
            arr = np.array(values, dtype=np.float64)
            if arr.size == 0:
                return {"avg": 0.0, "p95": 0.0}
            return {
                "avg": float(arr.mean()),
                "p95": float(np.percentile(arr, 95)),
            }

        total_s = sum(stats["frame_ms"]) / 1000.0
        summary = {
            "simulator": "magnetic",
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
            "notes": "Main-loop UI path benchmark; no browser decode/canvas timing included.",
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
