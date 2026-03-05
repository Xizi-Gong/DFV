import numpy as np

import taichi as ti

import zmq
import threading
import time
import json

from PIL import Image
import matplotlib.pyplot as plt


sim_size = 1200
sim_res_x = sim_size
sim_res_y = sim_size

disp_res_x = 710
disp_res_y = 410

# 计算裁剪的偏移量 (为了把画面居中)
offset_x = (sim_res_x - disp_res_x) // 2
offset_y = (sim_res_y - disp_res_y) // 2

alpha_blend = 0.8

ti.init(arch=ti.gpu)

# Fields
A_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y)) 
J_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
B_field = ti.Vector.field(2, dtype=ti.f32, shape=(sim_res_x, sim_res_y)) 

# === 新增：共轭梯度法(CG)的高维工作区 ===
r_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  # 残差 (Residual)
p_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  # 搜索方向 (Search direction)
Ap_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y)) # 算子投影 (Operator projection)

magnetic_intensity_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

initial_mask = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  # inital mask to define initial materials, if it is 1, this cannot be overridden by user input

input_mask = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))  # 来自 ZMQ 的输入遮罩
color_field = ti.Vector.field(3, float, shape=(sim_res_x, sim_res_y))

# Material properties
mu_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))    
inv_mu_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y)) 

mu_base_field    = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))
sigma_base_field = ti.field(dtype=ti.f32, shape=(sim_res_x, sim_res_y))

STREAMLINE_LENGTH = 10.0

# particle system
# === 新增：铁屑粒子系统 ===
num_particles = 8000  # 1万5千颗铁屑，足够产生致密的场域感
p_pos = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
p_vel = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)

# Define material combinations for easy switching; these can be expanded or modified as needed
materials = {
    'air':    {'mu': 1.0,    'sigma': 0.0},
    'iron':   {'mu': 1000.0, 'sigma': 0.0},
    'copper': {'mu': 1.0,    'sigma': 500.0},
    'steel':  {'mu': 500.0,  'sigma': 200.0},
    'shield': {'mu': 0.01,   'sigma': 0.0}
}

class ZmqMaskReceiver:
    def __init__(self, url="tcp://127.0.0.1:5556", topic="frame", hwm=10, conflated=False, poll_timeout_ms=5):
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(url)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.sub.setsockopt(zmq.RCVHWM, hwm)
        if conflated:
            self.sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest in socket
        self.poller = zmq.Poller()
        self.poller.register(self.sub, zmq.POLLIN)
        self.poll_timeout_ms = poll_timeout_ms

        self._lock = threading.Lock()
        self._latest = None  # latest numpy array (h, w)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.poller.unregister(self.sub)
        except Exception:
            pass
        self.sub.close(0)

    def _loop(self):
        # Non-blocking poll + recv; drop old frames and keep only the latest
        while not self._stop.is_set():
            socks = dict(self.poller.poll(self.poll_timeout_ms))
            if self.sub in socks and socks[self.sub] & zmq.POLLIN:
                try:
                    topic_b, header_b, raw_b = self.sub.recv_multipart(flags=zmq.NOBLOCK)
                    hdr = json.loads(header_b.decode("utf-8"))
                    h, w = hdr["shape"]
                    arr = np.frombuffer(raw_b, dtype=np.int16).reshape(h, w)

                    with self._lock:
                        self._latest = arr  # overwrite to keep only the newest
                except zmq.Again:
                    pass
            else:
                # tiny sleep to avoid hot loop
                time.sleep(0.001)

    def get_latest(self):
        # Return and keep; caller decides whether to reuse or not
        with self._lock:
            return self._latest

def update_mask_from_stream(receiver: ZmqMaskReceiver, mode="normalize", threshold=200.0):
    """
    Pull latest mask frame from receiver (if any) and write into obstacle_mask.
    mode: "normalize" (min-max to [0,1] then threshold) or "threshold" on raw values.
    If no frame is available, clear the mask.
    """
    if receiver is None:
        input_mask.fill(0)
        print("No receiver!")
        return False
    arr = receiver.get_latest()
    if arr is None:
        input_mask.fill(0)
        print("No frame received!")
        return False

    h, w = arr.shape
    if (w, h) != (disp_res_x, disp_res_y):
        img = Image.fromarray(arr)
        img = img.resize((disp_res_x, disp_res_y), resample=Image.BICUBIC)
        arr = np.array(img, copy=False)

    amin = float(arr.min())
    amax = float(arr.max())
    if mode == "normalize":
        if amax > amin:
            arr_norm = (arr.astype(np.float32) - amin) / (amax - amin)
        else:
            arr_norm = np.zeros_like(arr, dtype=np.float32)
        mask_np = (arr_norm > threshold).astype(np.float32)
    else:
        mask_np = (arr.astype(np.float32) > threshold).astype(np.float32)

    mask_np_flipped = np.flipud(mask_np.T)
    input_mask_np = np.zeros((sim_res_x, sim_res_y), dtype=np.float32)
    input_mask_np[offset_x:offset_x + disp_res_x, offset_y:offset_y + disp_res_y] = mask_np_flipped
    input_mask.from_numpy(input_mask_np)  # Taichi fields are (res_x, res_y)
    return True

def calculate_streamline(B_field, ii_flat, jj_flat):
    """Calculate streamline data from Taichi fields using vectorized NumPy."""
    # Convert to numpy once
    B_np = B_field.to_numpy()
    B_np_cropped = B_np[offset_x:offset_x + disp_res_x, offset_y:offset_y + disp_res_y]
    
    # Filter out solid cells
    # mask = solidf_np[ii_flat, jj_flat] == 0
    # ii_fluid = ii_flat[mask]
    # jj_fluid = jj_flat[mask]
    ii_fluid = ii_flat
    jj_fluid = jj_flat
    
    # Extract velocities at fluid cells
    velocities = B_np_cropped[ii_fluid, jj_fluid]  # Shape: (N, 2)
    
    # Compute magnitudes
    v_mag = np.linalg.norm(velocities, axis=1)
    # print(v_mag.min(), v_mag.max(), v_mag.mean())
    
    # Normalize directions (avoid division by zero)
    directions = np.where(v_mag[:, None] > 0, 
                          velocities / v_mag[:, None],
                          np.zeros_like(velocities))

    # # do not normalize, just use raw velocities for better length representation
    # directions = velocities.copy() / 300.0
    
    directions[:,0] = directions[:,0] / disp_res_x * STREAMLINE_LENGTH
    directions[:,1] = directions[:,1] / disp_res_y * STREAMLINE_LENGTH

    # clamp long streams
    lengths = np.linalg.norm(directions, axis=1)
    long_mask = lengths > STREAMLINE_LENGTH * 0.8
    directions[long_mask] = (directions[long_mask].T * (STREAMLINE_LENGTH * 0.8 / lengths[long_mask])).T

    # Compute normalized positions
    orig = np.stack([ii_fluid / float(disp_res_x), 
                     jj_fluid / float(disp_res_y)], axis=1)
    
    return directions, orig

@ti.func
def sample_A(field: ti.template(), i: int, j: int) -> float:
    # 磁矢势的 Dirichlet 边界：越界返回 0.0，让磁场自然消散
    val = 0.0
    if 0 <= i < sim_res_x and 0 <= j < sim_res_y:
        val = field[i, j]
    return val

@ti.func
def sample_mu(field: ti.template(), i: int, j: int) -> float:
    # 磁导率的边界：越界返回 1.0 (空气的磁导率)，防止除以 0 导致系统崩溃
    val = 1.0
    if 0 <= i < sim_res_x and 0 <= j < sim_res_y:
        val = field[i, j]
    return val

@ti.kernel
def update_inv_mu():
    for i, j in mu_field:
        inv_mu_field[i, j] = 1.0 / mu_field[i, j]

@ti.func
def sample_inv_mu(i: int, j: int) -> float:
    # 直接返回倒数，越界则返回空气的倒数 (1.0/1.0 = 1.0)
    val = 1.0 
    if 0 <= i < sim_res_x and 0 <= j < sim_res_y:
        val = inv_mu_field[i, j]
    return val


@ti.func
def harm(a, b):
    return 2.0 * a * b / (a + b + 1e-12)

# 1. 核心算子：计算稀疏矩阵的向量乘法 (A * p)
@ti.kernel
def compute_Ap(p: ti.template(), Ap: ti.template()):
    for i, j in p:
        p_c = p[i, j]
        p_L = sample_A(p, i-1, j)
        p_R = sample_A(p, i+1, j)
        p_D = sample_A(p, i, j-1)
        p_U = sample_A(p, i, j+1)
        
        # 读取预计算的磁导率倒数
        inv_mu_c = sample_inv_mu(i, j)
        inv_mu_L = sample_inv_mu(i-1, j)
        inv_mu_R = sample_inv_mu(i+1, j)
        inv_mu_D = sample_inv_mu(i, j-1)
        inv_mu_U = sample_inv_mu(i, j+1)

        # w_L = 0.5 * (inv_mu_c + inv_mu_L)
        # w_R = 0.5 * (inv_mu_c + inv_mu_R)
        # w_D = 0.5 * (inv_mu_c + inv_mu_D)
        # w_U = 0.5 * (inv_mu_c + inv_mu_U)
        w_L = harm(inv_mu_c, inv_mu_L)
        w_R = harm(inv_mu_c, inv_mu_R)
        w_D = harm(inv_mu_c, inv_mu_D)
        w_U = harm(inv_mu_c, inv_mu_U)
        
        total_weight = w_L + w_R + w_D + w_U
        
        # 隐式方程的左值
        Ap[i, j] = total_weight * p_c - (w_L * p_L + w_R * p_R + w_D * p_D + w_U * p_U)

# 2. 向量操作基石
@ti.kernel
def compute_r(x: ti.template(), r: ti.template(), b: ti.template(), Ap: ti.template()):
    for I in ti.grouped(x):
        r[I] = b[I] - Ap[I]

@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    for I in ti.grouped(src):
        dst[I] = src[I]

@ti.kernel
def dot_product(v1: ti.template(), v2: ti.template()) -> ti.f32:
    res = 0.0
    for I in ti.grouped(v1):
        res += v1[I] * v2[I]
    return res

@ti.kernel
def update_x_and_r(x: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template(), alpha: ti.f32):
    for I in ti.grouped(x):
        x[I] += alpha * p[I]
        r[I] -= alpha * Ap[I]

@ti.kernel
def update_p(p: ti.template(), r: ti.template(), beta: ti.f32):
    for I in ti.grouped(p):
        p[I] = r[I] + beta * p[I]

# 3. 共轭梯度主循环
def solve_poisson_cg(x_field, b_field, max_iters=40, tol=1e-4, atol=0.0):
    """
    Conjugate Gradient solve for A x = b (A is implicit via compute_Ap).
    tol: relative L2 residual tolerance, i.e. ||r||/||r0|| < tol
    atol: absolute L2 residual tolerance, i.e. ||r|| < atol  (useful when b is tiny)
    """

    # r0 = b - A x0
    compute_Ap(x_field, Ap_field)
    compute_r(x_field, r_field, b_field, Ap_field)
    copy_field(r_field, p_field)

    rsold = dot_product(r_field, r_field)          # ||r||^2
    rs0 = rsold + 1e-30                            # avoid zero-div
    abs_thresh = (atol * atol)                     # compare on squared norm
    rel_thresh = rs0 * (tol * tol)                 # ||r||^2 < ||r0||^2 * tol^2
    thresh = max(abs_thresh, rel_thresh)

    # already good enough
    if rsold <= thresh:
        return

    eps_denom = 1e-20  # safe-guard for pAp

    for _ in range(max_iters):
        compute_Ap(p_field, Ap_field)
        pAp = dot_product(p_field, Ap_field)

        # A should be SPD; if not (numerical/BC issues), avoid explosion
        if ti.abs(pAp) < eps_denom:
            break

        alpha = rsold / pAp
        update_x_and_r(x_field, p_field, r_field, Ap_field, alpha)

        rsnew = dot_product(r_field, r_field)

        # convergence check (relative + absolute)
        if rsnew <= thresh:
            break

        beta = rsnew / (rsold + 1e-30)
        update_p(p_field, r_field, beta)
        rsold = rsnew

@ti.kernel
def compute_B_field(A: ti.template(), B: ti.template()):
    for i, j in B:
        if 0 < i < sim_res_x - 1 and 0 < j < sim_res_y - 1:
            dA_dy = (A[i, j+1] - A[i, j-1]) * 0.5
            dA_dx = (A[i+1, j] - A[i-1, j]) * 0.5
            B[i, j] = ti.Vector([dA_dy, -dA_dx])
        else:
            B[i, j] = ti.Vector([0.0, 0.0])

@ti.kernel
def compute_and_render_v0(A: ti.template(), B_field: ti.template(), colorf: ti.template(), input_mask: ti.template()):
    freq = 0.005 
    # 线条的像素半宽（可以自由调节粗细）
    line_thickness = 1.2 

    for i, j in colorf:
        val = A[i, j]
        
        # 1. 核心映射：连续的周期正弦波
        phase = val * freq * 2.0 * 3.14159265
        wave = ti.sin(phase)
        
        # 2. 获取该点的磁场强度 (即 A 的梯度大小)
        grad_norm = B_field[i, j].norm()
        
        # 3. 链式法则求导：估算 wave 在一个像素内的局部变化率
        # 加上 1e-5 是为了绝对防止中心点或零场区域除以零导致画面崩溃
        pixel_grad = ti.abs(ti.cos(phase)) * 2.0 * 3.14159265 * freq * grad_norm + 1e-5
        
        # 4. 计算当前点到最近等值线 (wave=0) 的真实像素距离
        pixel_dist = ti.abs(wave) / pixel_grad
        
        # 5. 平滑抗锯齿 (Anti-aliasing)
        is_line = 0.0
        if pixel_dist < line_thickness:
            is_line = 1.0 - ti.math.smoothstep(pixel_dist, line_thickness - 1.0, line_thickness)

        # 6. 图层合成逻辑
        mask_val = input_mask[i, j]
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        if mask_val > 0.5:
            # 实体区域
            pixel_color = ti.Vector([0.5, 0.5, 0.5])
            if is_line > 0.0:
                pixel_color = pixel_color * (1.0 - is_line) + ti.Vector([1.0, 1.0, 1.0]) * is_line
        else:
            # 空气区域
            if is_line > 0.0:
                pixel_color = ti.Vector([1.0, 1.0, 1.0]) * is_line

        colorf[i, j] = pixel_color

@ti.kernel
def compute_and_render_v1(A: ti.template(), B_field: ti.template(), colorf: ti.template(), input_mask: ti.template()):
    # 【舒缓】：频率调至 0.015，适应降温后的 A 场
    freq = 0.005 # 0.015
    line_thickness = 1.2 

    for i, j in colorf:
        val = A[i, j]
        
        phase = val * freq * 2.0 * 3.14159265
        wave = ti.sin(phase)
        
        grad_norm = B_field[i, j].norm()
        
        pixel_grad = ti.abs(ti.cos(phase)) * 2.0 * 3.14159265 * freq * grad_norm + 1e-5
        pixel_dist = ti.abs(wave) / pixel_grad
        
        local_period_pixels = 1.0 / (freq * grad_norm + 1e-5)
        
        fade = 1.0
        if local_period_pixels < 4.0:
            fade = ti.math.smoothstep(local_period_pixels, 1.0, 4.0)

        is_line = 0.0
        if pixel_dist < line_thickness:
            is_line = 1.0 - ti.math.smoothstep(pixel_dist, line_thickness - 1.0, line_thickness)
        
        is_line = is_line * fade 

        mask_val = input_mask[i, j]
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        if mask_val > 0.5:
            pixel_color = ti.Vector([0.5, 0.5, 0.5])
            if is_line > 0.0:
                pixel_color = pixel_color * (1.0 - is_line) + ti.Vector([1.0, 1.0, 1.0]) * is_line
        else:
            if is_line > 0.0:
                pixel_color = ti.Vector([1.0, 1.0, 1.0]) * is_line

        colorf[i, j] = pixel_color

@ti.kernel
def init_particles():
    print("Initializing particle")
    for i in p_pos:    
        p_pos[i] = ti.Vector([offset_x + ti.random() * disp_res_x, offset_y + ti.random() * disp_res_y])
        p_vel[i] = ti.Vector([0.0, 0.0])

@ti.kernel
def update_particles():
    dt = 0.5
    for i in p_pos:
        pos = p_pos[i]
        ix = ti.cast(pos.x, ti.i32)
        iy = ti.cast(pos.y, ti.i32)

        # 防止越界
        ix = ti.max(1, ti.min(ix, sim_res_x - 2))
        iy = ti.max(1, ti.min(iy, sim_res_y - 2))

        B_vec = B_field[ix, iy]
        B_mag = B_vec.norm()

        # 1. 探测深渊：计算梯度拉扯力
        B_mag_R = B_field[ix+1, iy].norm()
        B_mag_L = B_field[ix-1, iy].norm()
        B_mag_U = B_field[ix, iy+1].norm()
        B_mag_D = B_field[ix, iy-1].norm()
        
        grad_B = ti.Vector([B_mag_R - B_mag_L, B_mag_U - B_mag_D]) * 0.5
        
        # 赋予适度的拉力
        f_grad = grad_B * 15.0 
        pull_force = f_grad.norm()

        # === 核心重构：动态防御机制 (磁拥堵) ===
        
        # 基础的纸面静摩擦
        base_friction = 0.2
        
        # 磁拥堵阻力：越靠近中心，铁屑互相咬合的阻力随着磁场的平方呈指数级暴增
        # 这就是对抗奇点塌陷的最强防御
        jamming_friction = base_friction + (B_mag ** 2) * 1.5
        
        target_vel = ti.Vector([0.0, 0.0])
        
        # 只有当拉扯力大过这道极其厚重的动态摩擦力时，才允许滑动
        if pull_force > jamming_friction:
            # 留下一部分力用来对抗阻力，剩下的力转化为游走
            effective_force = pull_force - jamming_friction
            
            # 严格限制最大滑动速度，让系统保持极度的克制，防止产生巨大的空洞
            speed = ti.min(effective_force * 0.1, 1.2)
            target_vel = (f_grad / pull_force) * speed
                
        # 极速过阻尼定格：剥夺任何多余的惯性
        p_vel[i] = p_vel[i] * 0.1 + target_vel * 0.9

        p_pos[i] += p_vel[i] * dt

# def initialize_materials():
#     mu_field.fill(1.0)    
#     sigma_field.fill(0.0) 

def initialize_materials():
    mu_base_field.fill(1.0)
    sigma_base_field.fill(0.0)
    initial_mask.fill(0)

    # derived fields 也给个初始值（可选）
    mu_field.fill(1.0)
    sigma_field.fill(0.0)
    inv_mu_field.fill(1.0)

def reset_scene():
    A_field.fill(0.0)
    J_field.fill(0.0)
    mu_base_field.fill(1.0)
    sigma_base_field.fill(0.0)

# single wire magnetic field
@ti.kernel
def setup_single_wire():
    for I in ti.grouped(J_field): J_field[I] = 0.0
    cx, cy = sim_res_x // 2, sim_res_y // 2
    J_field[cx, cy] = 1000.0 
    initial_mask[cx, cy] = 1.0  # 标记这个点为初始材料，禁止用户覆盖

# dipole magnetic field
@ti.kernel
def setup_dipole():
    for i, j in J_field:
            J_field[i, j] = 0.0
    
    cx, cy = sim_res_x // 2, sim_res_y // 2
    offset = 100  # 两个“磁极”的距离
    
    # 左边：流出 (+I)，相当于 N 极一侧
    J_field[cx - offset, cy] = 2000.0
    initial_mask[cx - offset, cy] = 1.0  # 标记这个点为初始材料，禁止用户覆盖
    # 右边：流入 (-I)，相当于 S 极一侧
    J_field[cx + offset, cy] = -2000.0
    initial_mask[cx + offset, cy] = 1.0  # 标记这个点为初始材料，禁止用户覆盖

# uniform magnetic field
@ti.kernel
def setup_uniform_field():
    for I in ti.grouped(J_field): J_field[I] = 0.0
    
    # 在左右两侧 1/4 处各放一根强电流
    # 注意：电流方向必须相同！这样中间的场才会叠加增强而非抵消
    
    margin = sim_res_x // 4
    cy = sim_res_y // 2
    
    strength = 1000.0 # 需要足够强才能覆盖整个中间区域
    
    J_field[margin, cy] = strength         # 左源
    J_field[sim_res_x - margin, cy] = -strength  # 右源 (设为相反数)
    initial_mask[margin, cy] = 1.0  # 标记这个点为初始材料，禁止用户覆盖
    initial_mask[sim_res_x - margin, cy] = 1.0  # 标记这个点为初始材料，禁止用户覆盖

# quadrupole magnetic field
@ti.kernel
def setup_quadrupole():
    for I in ti.grouped(J_field): J_field[I] = 0.0
    
    cx, cy = sim_res_x // 2, sim_res_y // 2
    d = 40 # 间距
    s = 2000.0
    
    # 左上 (+)   右上 (-)
    # 左下 (-)   右下 (+)
    
    J_field[cx - d, cy + d] =  s
    J_field[cx + d, cy + d] = -s
    J_field[cx - d, cy - d] = -s
    J_field[cx + d, cy - d] =  s

    initial_mask[cx - d, cy + d] = 1.0
    initial_mask[cx + d, cy + d] = 1.0
    initial_mask[cx - d, cy - d] = 1.0
    initial_mask[cx + d, cy - d] = 1.0

# solenoid magnetic field
@ti.kernel
def setup_solenoid():
    # 1. 清空电流场
    for I in ti.grouped(J_field): J_field[I] = 0.0
    
    # 2. 定义线圈几何参数 (在 sim_res 正方形坐标系中)
    cx, cy = sim_res_x // 2, sim_res_y // 2
    
    coil_len = 400       # 线圈总长度
    coil_radius = 100    # 线圈半径 (决定了上下两排的间距)
    thickness = 20       # 导线层的厚度，越厚场越强
    current_density = 0.5 
    
    # 3. 填充电流
    # 遍历整个网格，找到属于"上排"和"下排"的区域
    for i, j in J_field:
        # X 轴范围限制 (线圈长度)
        if cx - coil_len//2 < i < cx + coil_len//2:
            
            # 上排导线 (流出 +)
            if cy + coil_radius < j < cy + coil_radius + thickness:
                J_field[i, j] = current_density
                initial_mask[i, j] = 1.0  # 标记为初始材料，禁止用户覆盖
            
            # 下排导线 (流入 -)
            elif cy - coil_radius - thickness < j < cy - coil_radius:
                J_field[i, j] = -current_density
                initial_mask[i, j] = 1.0  # 标记为初始材料，禁止用户覆盖

    # 注意：螺线管内部是空气，所以不需要修改 mu_field (保持为 1.0)

@ti.kernel
def setup_horseshoe():
    for I in ti.grouped(J_field): J_field[I] = 0.0
    for I in ti.grouped(mu_base_field): mu_base_field[I] = 1.0 
    
    cx, cy = sim_res_x // 2, sim_res_y // 2
    
    magnet_width = 80
    gap_half = 70
    arm_height = 100
    
    iron_mu = 1000.0
    M_current = 0.5  
    
    for i, j in mu_base_field:
        dx = i - cx
        dy = j - cy
        
        is_iron = False
        is_inner_edge = False
        is_outer_edge = False
        edge_thickness = 8 
        
        # 1. 左直臂
        if (-gap_half - magnet_width < dx < -gap_half) and (0 < dy < arm_height):
            is_iron = True
            if dx > -gap_half - edge_thickness: is_inner_edge = True
            if dx < -gap_half - magnet_width + edge_thickness: is_outer_edge = True
            
        # 2. 右直臂
        elif (gap_half < dx < gap_half + magnet_width) and (0 < dy < arm_height):
            is_iron = True
            if dx < gap_half + edge_thickness: is_inner_edge = True
            if dx > gap_half + magnet_width - edge_thickness: is_outer_edge = True
            
        # 3. 底部半圆弯曲
        elif dy <= 0:
            dist = ti.sqrt(dx**2 + dy**2)
            r_inner = gap_half
            r_outer = gap_half + magnet_width
            if r_inner < dist < r_outer:
                is_iron = True
                if dist < r_inner + edge_thickness: is_inner_edge = True
                if dist > r_outer - edge_thickness: is_outer_edge = True

        if is_iron:
            mu_field[i, j] = iron_mu
            mu_base_field[i, j] = iron_mu
            initial_mask[i, j] = 1.0
            
        # 注入边界电流
        if is_inner_edge:
            J_field[i, j] = M_current
        elif is_outer_edge:  
            J_field[i, j] = -M_current

@ti.kernel
def update_materials_with_mask(input_mask: ti.template(), mu_field: ti.template(), sigma_field: ti.template(), initial_mask: ti.template(), mu: ti.f32, sigma: ti.f32):
    for i, j in input_mask:
        if initial_mask[i, j] == 0.0:
            if input_mask[i, j] > 0.9:
                mu_field[i, j] = mu
                sigma_field[i, j] = sigma
            else:
                mu_field[i, j] = mu_base_field[i, j]
                sigma_field[i, j] = sigma_base_field[i, j]
        else:
            # continue
            mu_field[i, j] = mu_base_field[i, j]
            sigma_field[i, j] = sigma_base_field[i, j]

@ti.kernel
def compose_material_fields(input_mask: ti.template(), mu_field: ti.template(), sigma_field: ti.template(), inv_mu_field: ti.template(), mu_base_field: ti.template(), sigma_base_field: ti.template(), locked_mask: ti.template(), mu_mat: ti.f32, sigma_mat: ti.f32):
    for i, j in mu_field:
        mu = mu_base_field[i, j]
        sigma = sigma_base_field[i, j]

        if locked_mask[i, j] == 0:
            if input_mask[i, j] > 0.9:
                mu = mu_mat
                sigma = sigma_mat

        mu_field[i, j] = mu
        sigma_field[i, j] = sigma
        inv_mu_field[i, j] = 1.0 / (mu + 1e-12)

@ti.kernel
def compute_magnetic_intensity(B_field: ti.template(), magnetic_intensity_field: ti.template()):
    for i, j in magnetic_intensity_field:
        B = B_field[i, j]
        magnetic_intensity_field[i, j] = B.norm()

def set_up_scene(scene_number):
    reset_scene()
    initialize_materials()
    if scene_number == 0:
        setup_single_wire()
    elif scene_number == 1:
        setup_dipole()
    elif scene_number == 2:
        setup_uniform_field()
    elif scene_number == 3:
        setup_quadrupole()
    elif scene_number == 4:
        setup_solenoid()
    elif scene_number == 5:
        setup_horseshoe()

def main():
    vis_magnetic_intensity = True
    vis_magnetic_field_line = True
    vis_magnetic_field_direction = False
    vis_iron_filings_method =False
    particle_initialized = False

    if vis_iron_filings_method and not particle_initialized:
        init_particles()
        particle_initialized = True

    initial_scene_number = 0
    set_up_scene(initial_scene_number)

    streamline_step = 20
    start_step = streamline_step // 2

    i_indices = np.arange(start_step, disp_res_x, streamline_step)
    j_indices = np.arange(start_step, disp_res_y, streamline_step)
    ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Flatten to 1D arrays
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()

    print("Pre-computing...")
    update_inv_mu() 
    
    # 用足够深的迭代深度，一眼看穿并一次性到达绝对稳态
    # 2000 次对 GPU 上的 CG 来说只是一瞬间，但足以让百万级网格的误差彻底归零
    solve_poisson_cg(A_field, J_field, max_iters=2000, tol=1e-6)
    
    print("Warm up done.")    

    # iron, copper, steel, shield
    input_material = materials['iron']

    cmap = plt.cm.viridis                    # or coolwarm, viridis, etc.
    
    gui = ti.GUI("Magnetic Field Simulation", (disp_res_x, disp_res_y))

    mask_receiver = ZmqMaskReceiver(url="tcp://127.0.0.1:5556", topic="frame", conflated=True)
    mask_receiver.start()
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                set_up_scene(initial_scene_number)  # reset to initial scene
            elif e.key in [ti.GUI.LEFT, ti.GUI.RIGHT]:
                if e.key == ti.GUI.LEFT:
                    initial_scene_number = (initial_scene_number - 1) % 6
                else:
                    initial_scene_number = (initial_scene_number + 1) % 6
                set_up_scene(initial_scene_number)
                update_inv_mu() 
                solve_poisson_cg(A_field, J_field, max_iters=2000, tol=1e-6)
                particle_initialized = False  # 场景切换后重置粒子状态
            elif e.key == "i":
                vis_magnetic_intensity = not vis_magnetic_intensity
            elif e.key == "l":
                vis_magnetic_field_line = not vis_magnetic_field_line
            elif e.key == "d":
                vis_magnetic_field_direction = not vis_magnetic_field_direction
            elif e.key == "f":
                vis_iron_filings_method = not vis_iron_filings_method


        update_mask_from_stream(mask_receiver, mode="threshold", threshold=200.0)
        update_materials_with_mask(input_mask, mu_field, sigma_field, initial_mask, input_material['mu'], input_material['sigma'])
        
        
        update_inv_mu() 
        # compose_material_fields(input_mask, mu_field, sigma_field, inv_mu_field, mu_base_field, sigma_base_field, locked_mask, input_material['mu'], input_material['sigma'])
        
        solve_poisson_cg(A_field, J_field, max_iters=40, tol=1e-4)
        compute_B_field(A_field, B_field)

        if vis_magnetic_intensity:
            compute_magnetic_intensity(B_field, magnetic_intensity_field)
            magnetic_intensity_np = magnetic_intensity_field.to_numpy()
            B_VISUAL_MAX = 50.0 
            
            b_clipped = np.clip(magnetic_intensity_np, 0.0, B_VISUAL_MAX)
            
            b_norm = b_clipped / B_VISUAL_MAX
            
            b_mapped = b_norm ** 0.6  
            
            rgba = cmap(b_mapped)
            heatmap_layer = rgba[:, :, :3]
        else:
            heatmap_layer = np.zeros((sim_res_x, sim_res_y, 3), dtype=np.float32)
        
        if vis_magnetic_field_line:
            compute_and_render_v0(A_field, B_field, color_field, input_mask)
            overlay_layer = color_field.to_numpy()
        else:
            overlay_layer = np.zeros((sim_res_x, sim_res_y, 3), dtype=np.float32)


        img_np = np.clip(heatmap_layer + overlay_layer, 0.0, 1.0)
        #img_np = np.clip(heatmap_layer, 0.0, 1.0)
        crop_img = img_np[offset_x : offset_x + disp_res_x, offset_y : offset_y + disp_res_y]
        gui.set_image(crop_img)
        if vis_magnetic_field_direction:
            directions, streamline_starts = calculate_streamline(B_field, ii_flat, jj_flat)
            gui.arrows(orig=streamline_starts, direction=directions, radius=2, color=0xFFFFFF)
        
        if vis_iron_filings_method:
            if not particle_initialized:
                init_particles()
                particle_initialized = True
            else:
                # 激活粒子物理引擎
                update_particles()
                
                # === 铁屑渲染逻辑 ===
                pos_np = p_pos.to_numpy()
                
                # 只提取在当前裁剪视口内的铁屑
                valid_mask = (pos_np[:, 0] > offset_x) & (pos_np[:, 0] < offset_x + disp_res_x) & \
                            (pos_np[:, 1] > offset_y) & (pos_np[:, 1] < offset_y + disp_res_y)
                valid_pos = pos_np[valid_mask]
                
                if len(valid_pos) > 0:
                    print(valid_pos.shape[0], "particles in view")
                    ix_np = np.clip(valid_pos[:, 0].astype(int), 0, sim_res_x - 1)
                    iy_np = np.clip(valid_pos[:, 1].astype(int), 0, sim_res_y - 1)
                    
                    # 读取底层的 B 场，决定每根铁屑的偏转角度
                    B_np_full = B_field.to_numpy()
                    B_particles = B_np_full[ix_np, iy_np]
                    B_mag_p = np.linalg.norm(B_particles, axis=1, keepdims=True) + 1e-5
                    dir_particles = B_particles / B_mag_p
                    
                    # 映射到 [0, 1] 显示空间
                    centers_x = (valid_pos[:, 0] - offset_x) / disp_res_x
                    centers_y = (valid_pos[:, 1] - offset_y) / disp_res_y
                    centers = np.stack([centers_x, centers_y], axis=1)
                    
                    # 控制单根铁屑的视觉长度 (像素)
                    filing_len = 3.0 
                    dx = (dir_particles[:, 0] * filing_len) / disp_res_x
                    dy = (dir_particles[:, 1] * filing_len) / disp_res_y
                    delta = np.stack([dx, dy], axis=1)
                    
                    starts = centers - delta
                    ends = centers + delta
                    
                    # 绘制黑色的铁屑阵列 (0x111111)
                    gui.lines(begin=starts, end=ends, radius=1.0, color=0xFFFFFF)
            
        gui.show()

if __name__ == "__main__":
    main()