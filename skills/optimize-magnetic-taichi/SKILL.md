---
name: optimize-magnetic-taichi
description: Diagnose and optimize real-time magnetic-field simulations built with Python, Taichi GPU kernels, iterative PCG solvers, WebSocket/JPEG streaming, and browser canvas UIs. Use for low FPS, high touch latency, Uniform Field stutter, slow Direction arrows or Iron Fillings, scene or material switching stalls, solver convergence problems, excessive GPU-to-CPU transfers, browser frame backlog, or performance work on Magnetic main/draw/shapes baselines while preserving field accuracy, stability, units, direction conventions, particle behavior, and visual density.
---

# Optimize Magnetic Taichi

## 目标

把实时磁场系统优化为低延迟、稳定、可持续收敛的实现。优先消除无效计算和排队延迟；不要通过静默降低网格分辨率、改变材料参数或牺牲磁场方向正确性来制造虚假 FPS。

## 读取参考资料

- 执行实际优化前，读取 [references/optimization-process.md](references/optimization-process.md)。其中记录本项目的完整瓶颈定位、算法改造顺序、权衡和实测结果。
- 建立 benchmark、比较准确性或准备交付前，读取 [references/benchmark-and-correctness.md](references/benchmark-and-correctness.md)。

## 工作流

### 1. 确认边界和现状

1. 读取目标目录内的 AGENTS.md、Agents.md 或项目说明。
2. 检查工作区状态，保护用户已有改动。
3. 找到主后端、共享 baseline 后端、主 UI、draw/shapes UI 和 benchmark。
4. 记录以下不可静默改变的量：
   - 仿真分辨率和显示裁剪范围；
   - 场方向、坐标翻转和边界条件；
   - 电流源、磁导率、材料参数和场景定义；
   - UI 的功能开关和输出尺寸。

在 Touch Beyond Tap 项目中优先检查：

- Magnetic/magnetic_field_simulation.py
- Magnetic/magnetic_field_simulation_baseline_common.py
- Magnetic/magnetic_ui.html
- Magnetic/magnetic_ui_draw.html
- Magnetic/magnetic_ui_shapes.html
- benchmarks/backend_benchmark.py

### 2. 建立可重复基线

同时测量静止、稀疏交互、密集移动和移除输入后的恢复阶段。至少覆盖小、中、大三个源复杂度场景，并单独覆盖 Uniform Field。

分开记录：

- 输入处理和材料更新；
- 线性系统准备与求解；
- B 场更新；
- 热力图/磁感线/箭头/铁粉渲染；
- GPU→CPU 传输；
- JPEG 编码与 WebSocket 广播；
- 浏览器解码和 Canvas 绘制。

先热身 Taichi kernel，再计时。不要把首次 JIT 编译误判为稳态帧耗时。

### 3. 按收益顺序定位瓶颈

依次检查：

1. 系统没有变化时是否仍重复求解。
2. 一次触摸更新是否把大量迭代全部塞进单帧。
3. 每次 PCG 迭代是否启动过多 kernel 或重复 reduction。
4. 收敛阈值是否被固定边界值放大，导致 Uniform Field 过早停止。
5. A 未变化时是否仍重算 B。
6. 一帧是否发生多次 GPU→CPU 传输和 CPU colormap 合成。
7. JPEG 是否重复编码或 WebSocket 是否重复发送同一帧。
8. 浏览器是否并发解码所有旧帧并积累延迟。

先解决复杂度和无效工作，再做微优化。

### 4. 选择算法策略

优先采用以下组合：

- **状态变化驱动**：仅在场景、材料或输入掩码变化时重建系统；收敛后停止求解。
- **持久化分帧 PCG**：把初始化与推进拆开，保留 r/p/z 和共轭方向；每帧只推进固定小批迭代，静止后继续收敛。
- **kernel 融合**：合并 Ap 与 pAp，合并预条件器与残差点积；只在检查点计算完整残差。
- **正确的自由变量残差**：固定 Dirichlet 单元必须精确执行，但不要让它们放大相对误差分母。
- **派生场缓存**：只有 A 改变时才更新 B；只有 B 改变时才更新方向箭头数据。
- **GPU 单次合成**：用 LUT 在 GPU 合成强度图与磁感线，只回传最终 RGB。
- **最新帧优先**：后端给帧编号并避免重复广播；浏览器最多保留一个尚未解码的新帧。
- **可选视觉解耦**：Direction 只在 B 变化且达到独立更新节奏时量化发送，并用批量 Path2D 绘制；Iron Fillings 在 GPU 直接光栅化进最终 RGB，避免粒子线段回传和逐条 CPU 绘制。

不要只降低 max_iters。若没有保留并继续求解状态，FPS 会提高但静止后的准确性无法恢复。

### 5. 小步实现

按以下顺序提交可验证改动：

1. 修复静止时重复求解和无条件日志同步。
2. 引入 start/advance 持久化 PCG。
3. 融合 PCG kernel，并保持 breakdown/NaN 检查。
4. 修正收敛范数和 Uniform Field。
5. 缓存 B、箭头和未变化的派生数据。
6. 合并 GPU 渲染与传输。
7. 优化 JPEG、广播节奏和浏览器解码队列。
8. 把共享优化接入 draw/shapes baseline。

每一步后运行最小性能测试和数值检查，发现回归立即缩小范围。

### 6. 验证并交付

必须验证：

- Python 和 UI JavaScript 语法；
- 小/中/大场景的 FPS、平均和 p95 延迟；
- Uniform Field 首段响应时间和最终收敛误差；
- NaN、发散、异常峰值和方向翻转；
- GPU 合成与旧画面的数值差异；
- draw/shapes baseline 的实时更新、擦除和控制按钮；
- 浏览器持续运行时不会累计旧帧。

交付时说明测试环境、显示 FPS 上限、动态首帧延迟、最终误差和仍存在的权衡。提醒用户重启 Python 后端并刷新浏览器。
## 可选可视化专项流程

### Direction

1. 分别测量方向采样、序列化、广播和 Canvas 绘制，避免把前端 draw call 开销误算为求解器问题。
2. 预计算固定箭头起点，只在 B 变化时采样方向；将坐标按明确 scale 量化为整数，并验证端点误差。
3. 给 Direction 设置独立更新上限和 sequence；静止场或量化结果未变化时不重复发送。
4. 前端将箭身合并为一个 `Path2D`、箭头合并为另一个 `Path2D`，每次更新各提交一次。
5. 启动时预热可选 Direction kernel，检查第一次打开开关没有 JIT 卡顿。

### Iron Fillings

1. 先区分粒子物理更新、线段生成、GPU→CPU 回传、CPU 绘制和 JPEG 编码的耗时。
2. 保留粒子位置、质量、排斥和 B 场取向算法；只替换可视化实现，防止性能优化改变物理行为。
3. 在 GPU 直接生成线段端点并光栅化双层 overlay，再合成最终 RGB；不要回传 `p_start`/`p_end` 或用 PIL 逐条绘线。
4. 用覆盖像素密度比、重心偏差、方向一致性和多场景 NaN 检查比较新旧实现。
5. 启动时预热 Iron Fillings 的更新、光栅化和合成 kernel，单独报告启用后的 mean、p95 与首帧延迟。

## 决策原则

- 若静止慢：优先处理 dirty state、重复求解、重复广播。
- 若移动时卡顿：优先处理分帧迭代、输入合并和 kernel 启动数量。
- 若 Uniform 不响应：检查自由变量残差阈值，不要直接增加迭代上限。
- 若后端快但画面滞后：检查 JPEG、WebSocket 和浏览器解码队列。
- 若方向正确但强度误差大：降低最终 tolerance，让后台继续收敛，不增加单帧预算。
- 若全部优化后仍达不到目标：评估多重网格预条件器；把降低分辨率作为显式、可测量、需用户接受的最后选项。

## 完成标准

只有同时满足以下条件才标记完成：

- 交互帧不再被一次完整求解阻塞；
- 静止系统不再重复求解；
- 最终数值误差达到项目要求；
- 没有 NaN、发散或方向语义变化；
- 浏览器不会因旧帧积压产生持续延迟；
- 主程序和所有共享 baseline 都通过回归检查。
