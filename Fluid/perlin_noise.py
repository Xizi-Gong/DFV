import taichi as ti
import math

# --- perlin noise basic functions ---

@ti.func
def fade(t: float) -> float:
    # 使用 Smootherstep: 6t^5 - 15t^4 + 10t^3
    # 比传统的 Hermite 插值 (3t^2 - 2t^3) 更平滑，减少网格伪影
    return t * t * t * (t * (t * 6 - 15) + 10)

@ti.func
def lerp(t: float, a: float, b: float) -> float:
    # 线性插值
    return a + t * (b - a)

@ti.func
def random_gradient(ix: int, iy: int):
    # Shader 中经典的伪随机 Hash 算法
    # 这里不需要返回一个归一化向量，只需要返回一个随机角度的单位向量即可
    # 这里的常数 (127.1, 311.7) 是经验值，能打散由于网格规律造成的波纹
    random_val = ti.sin(ix * 127.1 + iy * 311.7) * 43758.5453123
    angle = (random_val - ti.floor(random_val)) * 2.0 * math.pi
    return ti.Vector([ti.cos(angle), ti.sin(angle)])

@ti.func
def dot_grid_gradient(ix: int, iy: int, x: float, y: float) -> float:
    # 计算网格点梯度与距离向量的点积
    gradient = random_gradient(ix, iy)
    dx = x - float(ix)
    dy = y - float(iy)
    return dx * gradient[0] + dy * gradient[1]

# --- Perlin Noise 主函数 ---

@ti.func
def perlin_noise(uv: ti.template()) -> float:
    """
    输入: uv (ti.Vector), 建议范围是 [0, 10] 甚至更大，取决于你想要多密的噪点
    输出: float, 范围大约在 -1.0 到 1.0 之间
    """
    x0 = int(ti.floor(uv.x))
    x1 = x0 + 1
    y0 = int(ti.floor(uv.y))
    y1 = y0 + 1

    # 计算平滑插值权重
    sx = fade(uv.x - float(x0))
    sy = fade(uv.y - float(y0))

    # 计算四个角落的影响值
    n0 = dot_grid_gradient(x0, y0, uv.x, uv.y)
    n1 = dot_grid_gradient(x1, y0, uv.x, uv.y)
    n2 = dot_grid_gradient(x0, y1, uv.x, uv.y)
    n3 = dot_grid_gradient(x1, y1, uv.x, uv.y)

    # 在 X 轴上插值
    ix0 = lerp(sx, n0, n1)
    ix1 = lerp(sx, n2, n3)

    # 在 Y 轴上插值，得到最终结果
    return lerp(sy, ix0, ix1)

# --- 进阶：FBM (分形布朗运动) ---
# 单层 Perlin Noise 看起来像一团团棉花。
# FBM 叠加多层频率不同、振幅递减的噪声，看起来才像“烟雾”或“云层”。

@ti.func
def fbm(uv: ti.template(), octaves: int) -> float:
    value = 0.0
    amplitude = 0.5
    frequency = 1.0
    
    # 循环叠加多层
    for _ in range(octaves):
        # 叠加当前频率的噪声
        value += amplitude * perlin_noise(uv * frequency)
        
        # 频率翻倍 (更细碎)
        frequency *= 2.0
        # 振幅减半 (细节影响更小)
        amplitude *= 0.5
        
    # 将结果大致归一化到 0.0 ~ 1.0 范围 (Perlin 原本是 -1~1)
    # return value + 0.5
    return ti.max(0.0, ti.min(1.0, value * 0.9 + 0.5)) 