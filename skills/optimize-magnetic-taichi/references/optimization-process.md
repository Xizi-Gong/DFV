# Magnetic Taichi 优化过程

## 目录

1. [原始架构](#原始架构)
2. [瓶颈定位](#瓶颈定位)
3. [优化阶段](#优化阶段)
4. [关键参数](#关键参数)
5. [实测结果](#实测结果)
6. [没有采用的方案](#没有采用的方案)
7. [后续升级方向](#后续升级方向)

## 原始架构

本项目使用 640×640 Taichi 仿真网格，在裁剪区域中显示磁场。后端求解二维磁矢势 A，再由 A 计算 B 场，叠加强度热力图、磁感线、方向箭头和铁粉。Python 后端通过 JPEG/WebSocket 把画面送到 HTML Canvas。

主要路径：

~~~text
触摸或 baseline 掩码
  → 更新 mu/sigma
  → 更新 inv_mu 和 Jacobi 预条件器
  → PCG 求解 A
  → 计算 B
  → 强度图 + 磁感线
  → GPU→CPU
  → JPEG
  → WebSocket
  → 浏览器解码与 Canvas
~~~

Uniform Field 使用解析初值 A = B0 × (y - cy)，外圈为固定 Dirichlet 边界。其他场景由电流源或高磁导率磁体产生场。

## 瓶颈定位

### 1. 静止状态仍重复求解

旧主循环在掩码没有变化时仍调用最多 50 次 PCG。磁静态方程在系数、源项和边界均不变化时，解不会随时间演化，因此这些求解全部无效。

旧 baseline 更严重：在编码间隔尚未到达时仍反复执行 50 次求解，然后跳过渲染。求解耗时反而决定了画面发送节奏。

### 2. 交互变化把 150 次迭代塞进一帧

掩码变化后旧实现同步执行最多 150 次 PCG。密集 sweep 时单帧仿真约 252 ms，画面 FPS 下降到约 3.84，并产生明显的输入滞后。

简单把 150 改成 12 会损失准确性，因为下一帧会重新初始化 PCG，共轭方向无法延续，静止后也没有可靠的最终收敛保证。

### 3. 单次 PCG 迭代 kernel 过多

旧迭代大致包含：

1. 计算 Ap；
2. 计算 p·Ap；
3. 标量安全检查；
4. 更新 x/r；
5. 每次计算 r·r；
6. 应用预条件器；
7. 计算 r·z；
8. 更新 beta；
9. 更新 p。

其中 Ap 与 p·Ap、预条件器与 r·z 都可以在同一次网格遍历中完成。完整残差只需要在收敛检查点计算。

### 4. Uniform Field 的阈值被固定边界放大

旧阈值使用整个 RHS 的二范数。Uniform Field 边界值可达数千，固定边界单元会显著放大 ||b||，使材料加入后产生的残差在尚未真正改变场时就被判为满足 tol=1e-3。

固定单元的方程由 apply_fixed_A 精确执行，残差为零。收敛分母应只累计自由变量方程。

### 5. 渲染包含双重回传和 CPU colormap

旧路径分别回传：

- magnetic_intensity_field.to_numpy()；
- color_field.to_numpy()。

然后由 Matplotlib viridis 和 NumPy 在 CPU 合成。这造成两次 GPU→CPU 同步、一个 CPU colormap 和额外临时数组。

### 6. 帧会在传输链路中积压

旧 broadcaster 每 1/30 秒发送当前 latest frame，即使该帧没有更新也会重复发送。浏览器对每个二进制消息直接启动异步 createImageBitmap，解码慢于到达速度时，旧帧会排队，视觉上出现“输入结束后画面仍追赶”的延迟。

## 优化阶段

### 阶段 A：持久化分帧 PCG

把求解拆为两个状态机操作：

~~~text
start_current_system(tol):
    构建 RHS
    应用固定 A
    r = b - operator(x)
    z = inverse(M) × r
    p = z
    保存 rsold、threshold、status

continue_current_system(max_iters):
    使用保存的 r/p/z/rsold 继续 PCG
    达到阈值或 breakdown 时停止
~~~

交互变化时重新 start，立即推进一小批迭代；如果输入未再次变化，就在之后的循环中继续相同共轭序列。这样首段响应短，而静止后的最终准确性仍能恢复。

本次采用：

- 交互首段：12 iterations；
- 后续 settle：每段 12 iterations；
- 场景切换首段：24 iterations；
- 交互最终 tolerance：1e-4；
- 基础场高精度 tolerance：1e-6。

### 阶段 B：融合 PCG kernel

新增两个融合 kernel：

- cg_compute_Ap_and_dot：一次遍历写 Ap 并归约 p·Ap；
- cg_apply_preconditioner_and_dot：一次遍历写 z 并归约 r·z。

完整 r·r 从每次迭代改为每 6 次及最后一次检查。保留：

- pAp > 0 检查；
- NaN/Inf 检查；
- alpha/beta 上界；
- breakdown 状态；
- 每段末尾残差检查。

融合后高精度 Two Magnets 基础场求解从约 2872 ms 降到约 1192 ms；最终残差仍满足同一阈值。

### 阶段 C：变化驱动与缓存

主循环维护以下状态：

- current_solver_active：是否仍需继续收敛；
- field_dirty：A 是否变化；
- b_initialized：B 缓存是否有效；
- arrows_dirty：方向箭头是否需要重新采样；
- highlights_pending：收敛后是否需要重算高亮磁感线。

规则：

~~~text
掩码/材料/场景变化 → 重建系统并启动 PCG
PCG 推进一步      → A dirty
A dirty 且需要显示 → 重算 B
B 未变化           → 复用 B 和箭头
PCG 已收敛且无变化 → 不再求解
~~~

### 阶段 D：低分辨率掩码差异

旧路径上传完整掩码后再用 GPU 计算 L1 difference，并把标量读回 CPU。新的主循环保留已接受的 78×52 布尔掩码，用 NumPy 统计变化单元。

必须比较“最新观测掩码”和“最近一次已接受掩码”，不能只比较连续观测帧。否则两个连续的单像素变化可能永远达不到阈值。

本项目使用 LOWRES_MASK_DIFF_THRESH = 1，即累计至少两个低分辨率单元变化后接受，与旧完整网格阈值约等价。

### 阶段 E：修正 Uniform 收敛判据

收敛阈值只累计 fixed_A_mask < 0.5 的 RHS：

~~~text
for every equation:
    if this is a free cell:
        norm2 += b × b
threshold = norm2 × tol² + absolute_floor
~~~

配合 tol=1e-4，Uniform 圆形 iron 掩码：

- 首段 12 iterations：约 10–14 ms；
- 首段方向平均余弦：约 0.99499；
- 后台 33 个小段后收敛；
- 相对高精度参考 B 场 L2 误差：约 0.59%；
- 最终方向平均余弦：约 0.999995。

若只用 tol=5e-4，最终相对误差约 9.1%，因此没有采用。

### 阶段 F：GPU 单次画面合成

把 Matplotlib viridis 预采样为 256 项 GPU LUT。新的 compose_magnetic_frame 在一个 kernel 中完成：

1. 从 B 计算强度参数；
2. 查询 viridis LUT；
3. 计算磁感线覆盖率；
4. 处理磁体/材料颜色；
5. 处理红色高亮线；
6. 合成并 clamp 最终 RGB。

每帧只执行一次 color_field.to_numpy()。

对 Single Wire、Uniform、Two Magnets 比较旧/新合成：

- 平均绝对差通常约 1e-9 到 4e-8；
- 最大差通常约一个 float32 舍入单位；
- Two Magnets 极少量像素最大差约 0.00696，来自 LUT 边界量化；
- 无 NaN。

合成平均耗时从约 8.29 ms 降到约 1.33 ms，约 6.23×。

### 阶段 G：编码、广播与帧率

没有铁粉时使用 OpenCV JPEG 编码；有铁粉时保留 PIL 线段绘制路径。每次生成帧时增加 sequence，只广播新 sequence，避免重复发送。

显示目标从 30 提升到 60 FPS。触摸共享内存仍按 30 Hz 读取，不改变输入语义；两次输入之间可显示 PCG 收敛和铁粉动画的中间帧。

在 Windows 上，短 time.sleep(0.002) 可能被粗粒度计时器放大，导致 30 Hz 实际只有约 23 Hz。使用 time.sleep(0) 让出 CPU，不引入固定粗粒度延迟。渲染 kernel 在正式循环前热身，避免首次 JIT 影响第一帧。

静止 Web UI 实测约 59.66 FPS，接近 60 FPS 上限。

### 阶段 H：浏览器最新帧策略

浏览器维护一个正在解码状态和一个 pending frame：

~~~javascript
let pendingFrame = null;
let decoding = false;

function queueFrame(buffer) {
  pendingFrame = buffer;
  drainFrames();
}
~~~

drainFrames 同一时间只解码一个 bitmap。解码期间到达的多帧只保留最新一帧，从而把视觉延迟限制在一个正在解码帧加一个最新等待帧。

### 阶段 I：共享给 baseline

draw/shapes baseline 使用相同 PCG 状态机、GPU 合成、B 缓存、60 FPS 节奏和最新帧浏览器策略。实时 draw 发送的连续掩码先在命令队列中合并，只应用最新的连续 mask command，同时保留场景/材质命令顺序。

### 阶段 J：Direction 独立更新与批绘

Direction 的主要问题不是磁场采样本身，而是 1014 个箭头的重复序列化、广播和 Canvas draw call：

- 旧采样 + JSON：约 2.33 ms；
- 旧 payload：约 82 KB；
- 60 FPS 重复广播：约 4.95 MB/s；
- 浏览器每个箭头分别 stroke 和 fill，约两千次提交。

优化方式：

1. 箭头起点网格在 NumPy 中预计算，GPU 只回传方向。
2. 坐标乘以 10000 后量化为整数；端点最大量化误差小于约 0.06 屏幕像素。
3. payload 降至约 18 KB，后端生成约 0.91 ms。
4. Direction 使用独立 30 Hz 上限；B 未变化或量化结果相同则不生成新消息。
5. broadcaster 用 arrow sequence 只发送真正变化的消息；新客户端连接时直接获得当前状态。
6. 浏览器把所有 shaft 合并成一个 Path2D、所有 arrow head 合并成另一个 Path2D，每次更新只执行一次 stroke 和一次 fill。

静止场打开 Direction 后不再以 60 FPS 重发和重绘相同箭头。

### 阶段 K：Iron Fillings GPU 光栅化

旧路径每帧执行 p_start/p_end 两次 GPU 回传，再在 PIL 中逐条绘制约 850 个双层线段，最后重新 JPEG。粒子物理更新约 1 ms，CPU 绘制和编码才是主要开销。

新路径：

1. 保持原粒子位置、密度排斥和 B 场方向算法。
2. 在 GPU 根据 p_pos、p_mass 和 B 直接计算每条 filings 的端点。
3. 用两个标量 overlay 表示深色外轮廓和亮色内芯。
4. 沿线段采样，并沿法线光栅化两像素宽度；交替选择法线侧，避免整体偏移。
5. 用原来的 shade 公式和亮点规则覆盖最终 color field。
6. 只回传最终 RGB，并统一使用 OpenCV JPEG；不再回传线段数组或逐条调用 PIL。

同一粒子状态下，GPU 版本的覆盖像素为旧 PIL 版本的 98.95%，重心偏差小于 1 像素。端到端每帧从约 12.50 ms 降至 7.44 ms，约 1.68×；不同源规模场景约 6.7–7.1 ms，未出现 NaN 或异常场值。

Direction 与 Iron Fillings 的可选 kernel 在启动 warm-up 阶段预编译，避免第一次点击功能时出现 JIT 卡顿。

## 关键参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| TARGET_RENDER_FPS | 60 | JPEG/广播显示上限 |
| INTERACTIVE_SOLVE_ITERS | 12 | 输入变化后的首段 PCG 预算 |
| SETTLE_SOLVE_ITERS | 12 | 后续每段收敛预算 |
| SCENE_CHANGE_SOLVE_ITERS | 24 | 场景切换首段预算 |
| INTERACTIVE_SOLVE_TOL | 1e-4 | 交互最终相对残差阈值 |
| LOWRES_MASK_DIFF_THRESH | 1 | 接受输入变化的低分辨率阈值 |
| residual check interval | 6 | 完整残差归约间隔 |

调参原则：

- 首段太慢：先降低每段 iterations，不要放宽最终 tolerance。
- 停止输入后收敛太慢：改善预条件器或增加 settle iterations；确认仍满足帧预算。
- Uniform 最终强度误差大：降低 tolerance。
- GPU 启动开销高：增加融合，避免只增加迭代批量。

## 实测结果

同一套小场景 backend benchmark：

| 条件 | 优化前 FPS | 优化后 FPS | 提升 |
|---|---:|---:|---:|
| idle static | 10.76 | 21.28 | 1.98× |
| interactive sparse | 11.14 | 42.52 | 3.82× |
| interactive dense | 3.84 | 33.81 | 8.80× |
| idle recovery | 10.83 | 34.90 | 3.22× |

注意：旧 benchmark 的 idle 分支会强制重启 50 次求解，因此不能代表优化后主循环的静止路径。实际主循环收敛后完全停止求解，静止 Web UI 约 59.66 FPS。

规模测试结果：

| 场景 | sparse FPS | dense FPS | recovery FPS |
|---|---:|---:|---:|
| Single Wire | 56.33 | 32.58 | 31.57 |
| Quadrupole | 32.61 | 31.81 | 37.18 |
| Two Magnets | 30.54 | 26.79 | 33.08 |

所有记录均无 NaN、无发散标记。

## 没有采用的方案

### 降低仿真分辨率

可直接降低 stencil 成本，但会改变边界几何、细小材料区域和磁感线精度。本次目标要求保持准确性，因此不采用。

### 只降低最大迭代数

首帧会更快，但每帧重启 PCG 会丢失共轭方向，最终解可能永远不收敛。使用持久化分帧 PCG 替代。

### 只降低 JPEG quality

只能减少编码和网络开销，不能解决 150 次同步求解造成的 250 ms 仿真延迟。

### 立即实现多重网格

多重网格预条件器可能进一步降低 640² Poisson 系统的迭代次数，但变量磁导率、内部固定边界和 Taichi 多层字段会显著增加实现与验证成本。本次通过消除重复求解、分帧延续和 kernel 融合已达到 60 FPS 静止上限，因此暂不承担该风险。

## 后续升级方向

满足以下任一条件时再评估多重网格或更强预条件器：

- 显示分辨率提高到 1024² 或更高；
- 大型动态材料场要求稳定 60 FPS；
- 单次 12-iteration chunk 在目标 GPU 上超过 16 ms；
- 最终 1e-4 收敛时间不可接受。

下一阶段可探索：

1. geometric multigrid V-cycle 作为 PCG 预条件器；
2. 分层系数 restriction/prolongation；
3. 对 constant-mu 场景使用 FFT 或解析缓存；
4. 为 GPU filings overlay 加入亚像素抗锯齿和更精细的线宽控制；
5. WebCodecs 或原始纹理流，减少 JPEG 解码开销；
6. 依据实际 GPU 时间自适应每段迭代数，而不是固定 12。
