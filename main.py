# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01
# https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py

import numpy as np

import taichi as ti

import seaborn as sns
import cv2
import perlin_noise
from PIL import Image
import matplotlib.pyplot as plt
import zmq
from utils import ZmqMaskReceiver, flat_flow_to_rgb, rgb_array_to_hex


# field parameters
res_x = 710
res_y = 410
dt = 0.01
U_IN = 300.0  # tuneable inflow velocity 100- 1000
RHO = 1000.0        # 1.2 for air, 1000.0 for water
VISCOSITY = 0.01    # Kinematic viscosity (nu). Water ~0.01, Honey ~10.0+

# visualization parameters
streamline_step = 20         # for streamline density
STREAMLINE_LENGTH = 0.03

# pick smoke positions
smoke_positions = [0.20, 0.50, 0.80]
# pick smoke width
smoke_width = [30, 30, 30]

# solver parameters
p_jacobi_iters = 200  # 40 for a quicker but less accurate result
f_strength = 1000.0
curl_strength = 0.0  # set to 0 to disable vorticity confinement


time_c = 5.0
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
#dye_decay = 1.0
force_radius = res_y / 2.0
debug = False

use_sparse_matrix = False
arch = "gpu"
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda)
else:
    raise ValueError("Only CPU and CUDA backends are supported for now.")

if use_sparse_matrix:
    print("Using sparse matrix")
else:
    print("Using jacobi iteration")

_velocities = ti.Vector.field(2, float, shape=(res_x, res_y))
_new_velocities = ti.Vector.field(2, float, shape=(res_x, res_y))
velocity_divs = ti.field(float, shape=(res_x, res_y))
velocity_curls = ti.field(float, shape=(res_x, res_y))
_pressures = ti.field(float, shape=(res_x, res_y))
_new_pressures = ti.field(float, shape=(res_x, res_y))
_dye_buffer = ti.Vector.field(2, float, shape=(res_x, res_y))
_new_dye_buffer = ti.Vector.field(2, float, shape=(res_x, res_y))
_smoke = ti.field(float, shape=(res_x, res_y))
_new_smoke = ti.field(float, shape=(res_x, res_y))
_solid = ti.field(float, shape=(res_x, res_y))              # 1=solid, 0=fluid
_new_solid_buffer = ti.field(float, shape=(res_x, res_y))   # for detecting newly-added solids

color_field = ti.Vector.field(3, float, shape=(res_x, res_y))

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)
smoke_pair = TexPair(_smoke, _new_smoke)

if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(A: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(res_x, res_y):
            row = i * res_y + j
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0
                center += 1.0
            if j != res_y - 1:
                A[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                A[row, row - res_y] += -1.0
                center += 1.0
            if i != res_x - 1:
                A[row, row + res_y] += -1.0
                center += 1.0
            A[row, row] += center

    N = res_x * res_y
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)

# @ti.func
# def solid_at(i, j):
#     inside = (0 <= i) & (i < res_x) & (0 <= j) & (j < res_y)
#     return ti.select(inside, _solid[i, j], ti.cast(1, float))
def update_obstacle_from_stream(receiver: ZmqMaskReceiver, mode="normalize", threshold=200.0):
    """
    Pull latest mask frame from receiver (if any) and write into obstacle_mask.
    mode: "normalize" (min-max to [0,1] then threshold) or "threshold" on raw values.
    If no frame is available, clear the mask.
    """
    if receiver is None:
        _solid.fill(0)
        print("No receiver!")
        return False
    arr = receiver.get_latest()
    if arr is None:
        _solid.fill(0)
        print("No frame received!")
        return False

    h, w = arr.shape
    if (w, h) != (res_x, res_y):
        img = Image.fromarray(arr)
        img = img.resize((res_x, res_y), resample=Image.BICUBIC)
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
    _solid.from_numpy(mask_np_flipped)  # Taichi fields are (res_x, res_y)
    return True

@ti.func
def solid_at(i, j):
    # If we go off the Left/Right edge, it's OPEN (0)
    # If we go off the Top/Bottom edge, it's SOLID (1)
    is_solid = 0.0
    
    if i < 0 or i >= res_x:
        is_solid = 0.0  # Open Tunnel Ends
    elif j < 0 or j >= res_y:
        is_solid = 1.0  # Solid Tunnel Walls
    else:
        is_solid = _solid[i, j] # Check internal obstacles
        
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
    s = 1.0 - ti.cast(_solid[I], ti.f32)  # 1 in fluid, 0 in solid
    return qf[I] * s

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

@ti.func
def bilerp_fluid(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    iu, iv = ti.floor(s), ti.floor(t)
    fu, fv = s - iu, t - iv
    a = sample_fluid(vf, iu, iv)
    b = sample_fluid(vf, iu + 1, iv)
    c = sample_fluid(vf, iu, iv + 1)
    d = sample_fluid(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

# 3rd order Runge-Kutta
# @ti.func
# def backtrace(vf: ti.template(), p, dt_: ti.template()):
#     v1 = bilerp(vf, p)
#     p1 = p - 0.5 * dt_ * v1
#     v2 = bilerp(vf, p1)
#     p2 = p - 0.75 * dt_ * v2
#     v3 = bilerp(vf, p2)
#     p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
#     return p

@ti.kernel
def diffuse_velocity(vf: ti.template(), new_vf: ti.template()):
    # Implicit Viscosity Solver using Jacobi iteration
    # alpha = dx^2 / (nu * dt)
    alpha = 1.0 / (VISCOSITY * dt) 
    for i, j in vf:
        if _solid[i, j] == 1:
            new_vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            vl = sample(vf, i - 1, j)
            vr = sample(vf, i + 1, j)
            vb = sample(vf, i, j - 1)
            vt = sample(vf, i, j + 1)
            vc = vf[i, j]
            # Solve: (1 + 4/alpha) * u_next = u_prev + (1/alpha) * sum(neighbors)
            new_vf[i, j] = (vc * alpha + vl + vr + vb + vt) / (4.0 + alpha)

@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp_fluid(vf, p)
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp_fluid(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp_fluid(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)

    # (optional but recommended) keep sampling away from boundary to avoid clamping artifacts
    p.x = ti.max(0.5, ti.min(res_x - 1.5, p.x))
    p.y = ti.max(0.5, ti.min(res_y - 1.5, p.y))
    return p

@ti.kernel
def paint_color_field(smokef: ti.template(), dyef: ti.template(), solidf: ti.template(), colorf: ti.template()):
    for i, j in smokef:
        
        if solidf[i, j] == 1:
            colorf[i, j] = ti.Vector([198 / 255.0, 134 / 255.0, 66 / 255.0])
        else:
            # if smokef[i, j] > 0.2:
            #     noise_val = perlin_noise.fbm(dyef[i,j] * 5.0, 1)
            #     structure = ti.pow(noise_val, 3)
            #     colorf[i, j] = ti.Vector([1.0, 1.0, 1.0]) * (0.2 + 0.8 * structure)
            # else:
            #     colorf[i, j] = ti.Vector([0.0, 0.0, 0.0])
            noise_val = perlin_noise.fbm(dyef[i,j] * 5.0, 5)
            structure = ti.pow(noise_val, 1)
            colorf[i, j] = ti.Vector([1.0, 1.0, 1.0]) * (0.5 + 0.5 * structure) * smokef[i,j]
            #smokef[i,j] = smokef[i,j] * ti.max(0.0, structure - 0.2)

# @ti.kernel
# def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
#     for i, j in vf:
#         p = ti.Vector([i, j]) + 0.5
#         p = backtrace(vf, p, dt)
#         new_qf[i, j] = bilerp(qf, p) * dye_decay

@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            new_qf[i, j] = qf[i,j] * 0  # velocity=(0,0) or dye=(0,0,0)
        else:
            p = ti.Vector([i, j]) + 0.5
            p = backtrace(vf, p, dt)
            new_qf[i, j] = bilerp(qf, p) * dye_decay

@ti.kernel
def advect_smoke(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            new_qf[i, j] = qf[i,j] * 0  # velocity=(0,0) or dye=(0,0,0)
        else:
            p = ti.Vector([i, j]) + 0.5
            p = backtrace(vf, p, dt)
            new_qf[i, j] = bilerp(qf, p)  # no decay for smoke

# @ti.kernel
# def divergence(vf: ti.template()):
#     for i, j in vf:
#         vl = sample(vf, i - 1, j)
#         vr = sample(vf, i + 1, j)
#         vb = sample(vf, i, j - 1)
#         vt = sample(vf, i, j + 1)
#         vc = sample(vf, i, j)
#         # if i == 0:
#         #     vl.x = -vc.x
#         # if i == res_x - 1:
#         #     vr.x = -vc.x
#         if j == 0:
#             vb.y = -vc.y
#         if j == res_y - 1:
#             vt.y = -vc.y
#         velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5

@ti.kernel
def refine_solid_mask(solidf:ti.template()):
    for i, j in solidf:
        if i == 0 or i == res_x - 1 or j == 0 or j == res_y - 1:
            solidf[i, j] = 0.0

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            velocity_divs[i, j] = 0.0
        else:
            vl = sample(vf, i - 1, j)
            vr = sample(vf, i + 1, j)
            vb = sample(vf, i, j - 1)
            vt = sample(vf, i, j + 1)
            vc = sample(vf, i, j)

            # top/bottom tunnel walls (keep your original)
            if j == 0:
                vb.y = -vc.y
            if j == res_y - 1:
                vt.y = -vc.y

            # obstacle blocks: treat neighbor velocity as 0 (normal component effectively blocked)
            if solid_at(i - 1, j) == 1:
                vl.x = 0.0
            if solid_at(i + 1, j) == 1:
                vr.x = 0.0
            if solid_at(i, j - 1) == 1:
                vb.y = 0.0
            if solid_at(i, j + 1) == 1:
                vt.y = 0.0

            velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


# @ti.kernel
# def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
#     for i, j in pf:
#         pl = sample(pf, i - 1, j)
#         pr = sample(pf, i + 1, j)
#         pb = sample(pf, i, j - 1)
#         pt = sample(pf, i, j + 1)
#         div = velocity_divs[i, j]
#         new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25
@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        # Enforce Outlet Dirichlet BC: Pressure is always 0 at the outlet
        if i == res_x - 1:
            new_pf[i, j] = 0.0
        if _solid[i, j] == 1:
            new_pf[i, j] = 0.0
        else:
            pc = pf[i, j]

            pl = sample(pf, i - 1, j) if solid_at(i - 1, j) == 0 else pc
            pr = sample(pf, i + 1, j) if solid_at(i + 1, j) == 0 else pc
            pb = sample(pf, i, j - 1) if solid_at(i, j - 1) == 0 else pc
            pt = sample(pf, i, j + 1) if solid_at(i, j + 1) == 0 else pc

            div = velocity_divs[i, j]
            # Multiply divergence by (RHO / dt) to get the correct pressure magnitude for liquids
            new_pf[i, j] = (pl + pr + pb + pt - (div * RHO / dt)) * 0.25
            # new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


# @ti.kernel
# def subtract_gradient(vf: ti.template(), pf: ti.template()):
#     for i, j in vf:
#         pl = sample(pf, i - 1, j)
#         pr = sample(pf, i + 1, j)
#         pb = sample(pf, i, j - 1)
#         pt = sample(pf, i, j + 1)
#         vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])
@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            vf[i, j] = ti.Vector([0.0, 0.0])
        else:
            pc = pf[i, j]
            pl = sample(pf, i - 1, j) if solid_at(i - 1, j) == 0 else pc
            pr = sample(pf, i + 1, j) if solid_at(i + 1, j) == 0 else pc
            pb = sample(pf, i, j - 1) if solid_at(i, j - 1) == 0 else pc
            pt = sample(pf, i, j + 1) if solid_at(i, j + 1) == 0 else pc

            #vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])
            vf[i, j] -= 0.5 * (dt / RHO) * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb), abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.types.ndarray()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res_y + I[1]] = -div_in[I]



@ti.kernel
def apply_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res_y + I[1]]

@ti.kernel
def inject_smokes(smokef: ti.template(), dyef: ti.template(), smoke_info: ti.types.ndarray()):
    for i, j in smokef:
        # only inject near the left side for efficiency
        if i < 1:
            random_noise_x = ti.random() - 0.5
            random_noise_y = ti.random() - 0.5
            dyef[i, j] = ti.Vector([i / float(res_x) + random_noise_x * 0.1, j / float(res_y) + random_noise_y * 0.1])
            for idx in range(smoke_info.shape[0]):
                if smoke_info[idx, 0] <= j <= smoke_info[idx, 1]:
                    smokef[i, j] = 1.0

@ti.kernel
def apply_wind_bc(vf: ti.template(), pf: ti.template(), u_in: ti.f32):
    # Inlet/Outlet in x
    for j in range(res_y):
        # inlet: fixed inflow to +x
        vf[0, j] = ti.Vector([u_in, 0.0])
        # outlet: zero-gradient outflow (copy from interior)
        vf[res_x - 1, j] = vf[res_x - 2, j]

        # pressure: set outlet pressure reference (common simple choice)
        # pf[res_x - 1, j] = 0.0
        # pf[0, j] = pf[1, j]

        pf[0, j] = pf[1, j]
        #pf[res_x - 1, j] = pf[res_x - 2, j]
        pf[res_x - 1, j] = 0.0

        # optional: keep dye from sticking at inlet
        # dyef[0, j] *= 0.0
        # dyef[res_x - 1, j] = dyef[res_x - 2, j]

    # Top/bottom: usually walls in a tunnel
    for i in range(res_x):
        # free-slip wall: no normal flow
        vf[i, 0].y = 0.0
        vf[i, res_y - 1].y = 0.0
        # copy tangential component
        vf[i, 0].x = vf[i, 1].x
        vf[i, res_y - 1].x = vf[i, res_y - 2].x
        # pressure Neumann at walls
        pf[i, 0] = pf[i, 1]
        pf[i, res_y - 1] = pf[i, res_y - 2]


@ti.kernel
def apply_obstacle_bc(vf: ti.template(), dyef: ti.template()):
    for i, j in vf:
        if _solid[i, j] == 1:
            vf[i, j] = ti.Vector([0.0, 0.0])
            # dyef[i, j] = ti.Vector([0.0, 0.0, 0.0])
        else:
            # if neighbor is solid, prevent flow into it
            if solid_at(i - 1, j) == 1:
                vf[i, j].x = ti.max(vf[i, j].x, 0.0)   # don't go left
            if solid_at(i + 1, j) == 1:
                vf[i, j].x = ti.min(vf[i, j].x, 0.0)   # don't go right
            if solid_at(i, j - 1) == 1:
                vf[i, j].y = ti.max(vf[i, j].y, 0.0)   # don't go down
            if solid_at(i, j + 1) == 1:
                vf[i, j].y = ti.min(vf[i, j].y, 0.0)   # don't go up


@ti.kernel
def redistribute_new_solid_dye(dyef: ti.template()):
    for i, j in dyef:
        if _solid[i, j] == 1 and _new_solid_buffer[i, j] == 0:
            d = dyef[i, j]
            dyef[i, j] = ti.Vector([0.0, 0.0, 0.0])

            wsum = 0.0
            # 4-neighbors
            for di, dj in ti.static([(-1,0),(1,0),(0,-1),(0,1)]):
                ni, nj = i + di, j + dj
                if 0 <= ni < res_x and 0 <= nj < res_y and _solid[ni, nj] == 0:
                    wsum += 1.0

            if wsum > 0:
                for di, dj in ti.static([(-1,0),(1,0),(0,-1),(0,1)]):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < res_x and 0 <= nj < res_y and _solid[ni, nj] == 0:
                        dyef[ni, nj] += d / wsum

@ti.kernel
def init_cylinder_obstacle():
    # clear previous solids
    _solid.fill(0)

    # Circle parameters
    # Position: Closer to the left inlet to allow space for the wake
    cx = res_x * 0.2
    cy = res_y * 0.5
    radius = 40.0  # Adjust size (30-50 is usually good for this grid size)

    for i, j in _solid:
        # Distance calculation (x-cx)^2 + (y-cy)^2 < r^2
        dist_sq = (i - cx)**2 + (j - cy)**2
        if dist_sq < radius**2:
            _solid[i, j] = 1.0

@ti.kernel
def count_solid_pixels(_solidf: ti.template()) -> int:
    count = 0
    for i, j in _solidf:
        # Check if pixel is solid
        if _solidf[i, j] > 0.0:
            count += 1
    return count

# @ti.kernel
# def outlet_sponge(vf: ti.template()):
#     start = int(res_x * 0.9)
#     for i, j in vf:
#         if i >= start:
#             t = (i - start) / max(1, (res_x - 1 - start))
#             vf[i, j] *= (1.0 - 0.2 * t)  # tune 0.2

@ti.kernel
def extrapolate_dye(dyef: ti.template(), solidf: ti.template()):
    for i, j in dyef:
        if solidf[i, j] == 1:
            # Initialize with existing color (or black)
            neighbor_sum = ti.Vector([0.0, 0.0, 0.0])
            neighbor_count = 0.0
            
            # Check 4 neighbors (Up, Down, Left, Right)
            for dx, dy in ti.static([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                ni, nj = i + dx, j + dy
                if 0 <= ni < res_x and 0 <= nj < res_y:
                    # If neighbor is FLUID, add its color to the average
                    if solidf[ni, nj] == 0:
                        neighbor_sum += dyef[ni, nj]
                        neighbor_count += 1.0
            
            # If we found valid fluid neighbors, copy their average color
            if neighbor_count > 0:
                dyef[i, j] = neighbor_sum / neighbor_count

# @ti.kernel
# def calculate_streamline(vf: ti.template(), solidf: ti.template(), start_point: int, step: int):
#     directions = []
#     orig = []
#     v_mag = []
#     for i, j in ti.ndrange((start_point, res_x), (start_point, res_y)):
#         if (i - start_point) % step != 0 or (j - start_point) % step != 0:
#             continue
#         if solidf[i, j] == 1:
#             continue
#         else:
#             v_np = vf[i, j].to_numpy()
#             speed = np.linalg.norm(v_np)
#             directions.append(v_np / (speed + 1e-5))
#             orig.append((float(i) / float(res_x), float(j) / float(res_y)))
#             v_mag.append(speed)
#     return directions, orig, v_mag

@ti.kernel
def init_uv_field(dyef: ti.template()):
    for i, j in dyef:
        dyef[i, j] = ti.Vector([i / float(res_x), j / float(res_y)])

def calculate_streamline(vf, solidf, ii_flat, jj_flat):
    """Calculate streamline data from Taichi fields using vectorized NumPy."""
    # Convert to numpy once
    vf_np = vf.to_numpy()
    solidf_np = solidf.to_numpy()
    
    # Filter out solid cells
    mask = solidf_np[ii_flat, jj_flat] == 0
    ii_fluid = ii_flat[mask]
    jj_fluid = jj_flat[mask]
    
    # Extract velocities at fluid cells
    velocities = vf_np[ii_fluid, jj_fluid]  # Shape: (N, 2)
    
    # Compute magnitudes
    v_mag = np.linalg.norm(velocities, axis=1)
    # print(v_mag.min(), v_mag.max(), v_mag.mean())
    
    # Normalize directions (avoid division by zero)
    directions = np.where(v_mag[:, None] > 1e-6, 
                          velocities, 
                          np.zeros_like(velocities))

    # # do not normalize, just use raw velocities for better length representation
    # directions = velocities.copy() / 300.0
    
    directions[:,0] = directions[:,0] / res_x * STREAMLINE_LENGTH
    directions[:,1] = directions[:,1] / res_y * STREAMLINE_LENGTH

    # clamp long streams
    lengths = np.linalg.norm(directions, axis=1)
    long_mask = lengths > STREAMLINE_LENGTH * 0.8
    directions[long_mask] = (directions[long_mask].T * (STREAMLINE_LENGTH * 0.8 / lengths[long_mask])).T

    # Compute normalized positions
    orig = np.stack([ii_fluid / float(res_x), 
                     jj_fluid / float(res_y)], axis=1)
    
    return directions, orig, v_mag

def get_mask_bounds(mask):
    nonzero = np.nonzero(mask)
    
    if len(nonzero[0]) == 0:
        return 0.0  # Empty mask
    
    y_min = nonzero[1].min()
    y_max = nonzero[1].max()
    return y_max - y_min

def solve_pressure_sp_mat():
    copy_divergence(velocity_divs, F_b)
    x = solver.solve(F_b)
    apply_pressure(x, pressures_pair.cur)

def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()


def step(u_in, smoke_info):
    # if solid_count == 0:
    #     pressures_pair.cur.fill(0)

    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    advect(velocities_pair.cur, smoke_pair.cur, smoke_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()
    smoke_pair.swap()

    # 2. NEW: Diffusion (Applying Viscosity)
    # For honey, you might run this iteration 20-40 times like the pressure solver
    for _ in range(20): 
        diffuse_velocity(velocities_pair.cur, velocities_pair.nxt)
        velocities_pair.swap()

    # apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    # apply_wind_bc(velocities_pair.cur, pressures_pair.cur, dyes_pair.cur)

    apply_wind_bc(velocities_pair.cur, pressures_pair.cur, u_in)
    # apply_obstacle_bc(velocities_pair.cur, dyes_pair.cur)
    # apply_obstacle_bc(velocities_pair.cur, smoke_pair.cur)

    divergence(velocities_pair.cur)

    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    # if use_sparse_matrix:
    #     solve_pressure_sp_mat()
    # else:
    solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    # outlet_sponge(velocities_pair.cur)

    # apply_wind_bc(velocities_pair.cur, pressures_pair.cur, dyes_pair.cur)
    apply_wind_bc(velocities_pair.cur, pressures_pair.cur, u_in)
    apply_obstacle_bc(velocities_pair.cur, dyes_pair.cur)
    apply_obstacle_bc(velocities_pair.cur, smoke_pair.cur)

    # inject 3 smoke streams at left
    inject_smokes(smoke_pair.cur, dyes_pair.cur, smoke_info)

    paint_color_field(smoke_pair.cur, dyes_pair.cur, _solid, color_field)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f"divergence={div_s}")

def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)
    color_field.fill(0)
    _solid.fill(0)


def main():
    global debug, curl_strength
    visualize_d = True  # visualize dye (default)
    visualize_v = False  # visualize velocity
    visualize_c = False  # visualize curl

    paused = False

    visualize_karman = True
    cylinder_obstacle_initialized = False

    vis_start_stream_line  = False
    start_step = streamline_step // 2
    streamline_color_map = sns.color_palette("viridis", as_cmap=True)

    vis_pressure = False

    # Create grid of indices
    i_indices = np.arange(start_step, res_x, streamline_step)
    j_indices = np.arange(start_step, res_y, streamline_step)
    ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Flatten to 1D arrays
    ii_flat = ii.flatten()
    jj_flat = jj.flatten()

    # initialize smoke info
    smoke_info = np.zeros((len(smoke_positions), 2), dtype=np.int16)
    for i in range(len(smoke_positions)):
        smoke_info[i, 0] = int(smoke_positions[i] * res_y - smoke_width[i] / 2)
        smoke_info[i, 1] = int(smoke_positions[i] * res_y + smoke_width[i] / 2)

    cmap = plt.cm.coolwarm                     # or coolwarm, viridis, etc.

    gui = ti.GUI("Fluid Field Simulation", (res_x, res_y))

    #md_gen = MouseDataGen()
    mask_receiver = ZmqMaskReceiver(url="tcp://127.0.0.1:5556", topic="frame", conflated=True)
    mask_receiver.start()

    init_uv_field(dyes_pair.cur)
    inlet_velocity_slider = gui.slider("Inlet Velocity", 100.0, 1000.0, step=50.0)
    inlet_velocity_slider.value = 900.0  # Set initial value to match your U_IN constant


    while gui.running:
        
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                reset()
            elif e.key == "s":
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == "v":
                visualize_v = True
                visualize_c = False
                visualize_d = False
            elif e.key == "d":
                visualize_d = True
                visualize_v = False
                visualize_c = False
            elif e.key == "c":
                visualize_c = True
                visualize_d = False
                visualize_v = False
            elif e.key == "p":
                paused = not paused
            elif e.key == "d":
                debug = not debug
            elif e.key == "k":
                visualize_karman = not visualize_karman
                cylinder_obstacle_initialized = False
            elif e.key == "l":
                vis_start_stream_line = not vis_start_stream_line
            elif e.key == "i":
                vis_pressure = not vis_pressure
                visualize_d = not visualize_d

        # Debug divergence:
        # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

        if not paused:
            #mouse_data = md_gen(gui)
            if visualize_karman:
                if not cylinder_obstacle_initialized:
                    init_cylinder_obstacle()
                    cylinder_obstacle_initialized = True
            else:
                update_obstacle_from_stream(mask_receiver, mode="threshold", threshold=200.0)
            refine_solid_mask(_solid)
            mask = np.array(_solid.to_numpy()[1:-2, 1:-2], dtype=bool)
            bounds_x = get_mask_bounds(mask)
            reynolds_number = (inlet_velocity_slider.value * bounds_x) / (VISCOSITY + 1e-5)
            step(inlet_velocity_slider.value, smoke_info)
        if visualize_c:
            vorticity(velocities_pair.cur)
            gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
        elif visualize_d:
            #gui.set_image(dyes_pair.cur)
            gui.set_image(color_field)
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        elif vis_pressure:
            p = pressures_pair.cur.to_numpy()          # shape (H, W)
            p_min, p_max = p.min(), p.max()
            p_norm = (p - p_min) / (p_max - p_min + 1e-10)  # → [0, 1]
            rgba = cmap(p_norm) 
            rgb  = (rgba[..., :3]).astype(np.float32)
            gui.set_image(rgb) 
            
        if vis_start_stream_line:
            directions, streamline_starts, streamline_mag = calculate_streamline(velocities_pair.cur, _solid, ii_flat, jj_flat)

            # map to colors
            streamline_colors = rgb_array_to_hex(streamline_color_map(np.array(streamline_mag/500.0)))
            gui.arrows(orig=streamline_starts, direction=directions, radius=2, color=streamline_colors)

            # map to length
            # gui.arrows(orig=streamline_starts, direction=directions, radius=2, color=0xED553B)
        gui.text(content=f'Re: {reynolds_number:.2f}', pos=[0.75, 0.9], font_size=25, color=0xFFFFFF)
        # has_nan = np.isnan(velocities_pair.cur.to_numpy()).any()
        # print(has_nan)
        gui.show()


if __name__ == "__main__":
    main()