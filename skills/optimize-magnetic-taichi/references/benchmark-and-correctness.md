# Magnetic 性能与正确性验证

## 目录

1. [验证目标](#验证目标)
2. [环境与热身](#环境与热身)
3. [性能矩阵](#性能矩阵)
4. [准确性指标](#准确性指标)
5. [稳定性检查](#稳定性检查)
6. [渲染等价性](#渲染等价性)
7. [端到端延迟](#端到端延迟)
8. [回归清单](#回归清单)
9. [结果解读陷阱](#结果解读陷阱)

## 验证目标

同时回答四个问题：

1. FPS 是否提高？
2. 输入到可见变化的 latency 是否降低？
3. 停止输入后是否继续收敛到准确结果？
4. 是否保持场方向、边界、材料和视觉语义？

只报告 FPS 不足以证明优化有效。

## 环境与热身

在本项目中使用 tbt conda 环境和 CUDA Taichi。记录：

- GPU 型号和驱动；
- Taichi 版本与 backend；
- Python 版本；
- 仿真/显示分辨率；
- 是否打开 intensity、field line、direction、filings；
- JPEG quality 和目标显示 FPS。

第一次调用每个 Taichi kernel 会 JIT 编译。正式计时前：

1. 设置场景；
2. 更新 inverse mu 和预条件器；
3. 完成一次基础求解；
4. 计算 B；
5. 运行一次最终合成；
6. 执行 ti.sync() 或 to_numpy()。

不要把 JIT 时间混入稳态平均值。另行报告首次启动时间。

## 性能矩阵

### 输入条件

至少包含：

- idle_static：无输入，系统稳定；
- interactive_sparse：较小按压区域保持或缓慢移动；
- interactive_dense：大面积 sweep，每帧系数变化；
- idle_recovery：先 sweep，再移除输入并观察收敛；
- Uniform circle：Uniform Field 中移动或放置圆形高磁导率材料。

### 场景复杂度

至少包含：

- Single Wire：小源数量；
- Quadrupole：中等源数量；
- Two Magnets：大面积材料和多源；
- Uniform Field：固定边界特殊场景。

### 时间分解

每帧至少记录：

~~~text
input_ms
coefficient_ms
solver_ms
B_ms
compose_ms
readback_ms
encode_ms
send_ms
frame_ms
~~~

报告 mean、p95、最大值和有效样本数。实时系统优先关注 p95；平均值可能掩盖场景切换或密集输入的尖峰。

### 项目 benchmark

运行现有 backend benchmark，并保持优化前后参数一致：

~~~powershell
$env:CONDA_DEFAULT_ENV='tbt'
& "$env:USERPROFILE/.conda/envs/tbt/python.exe" benchmarks/backend_benchmark.py --sim magnetic --frames 20 --warmup 3 --gate-conditions
~~~

该 benchmark 的旧 idle_static 路径可能显式调用求解器，不能证明主循环已正确跳过静止求解。必须另外测试真实主循环。

## 准确性指标

### 残差

对自由变量方程计算：

~~~text
r = b - operator(x)
relative_residual = norm(r_free) / max(norm(b_free), epsilon)
~~~

固定 Dirichlet 单元必须满足 x = fixed_value，但不要把固定单元值加入自由系统阈值分母。

记录：

- residual；
- threshold；
- status/breakdown；
- executed iterations；
- 是否 active/converged。

### B 场相对误差

用高精度参考解 B_ref：

~~~python
relative_l2 = norm(B_test - B_ref) / (norm(B_ref) + eps)
~~~

在显示裁剪区域内比较，避免未显示边界主导指标。

建议门槛：

- 最终 relative_l2 不超过 1%；
- 首段可以更高，但必须报告；
- 首段后继续输入不应产生不连续爆炸。

### 方向准确性

在 B_ref 模长高于低分位阈值的像素计算：

~~~python
cosine = dot(B_test, B_ref) / (norm(B_test) * norm(B_ref) + eps)
~~~

避免在近零场像素判断方向。建议最终平均方向余弦至少 0.999。

### 固定边界

检查：

- Uniform 外圈 A 等于解析固定值；
- fixed mask 内残差为零或数值误差范围；
- 场景切换、材料切换、reset 后边界不会继承旧场景。

## 稳定性检查

### 数值有限性

每类场景检查：

~~~python
np.isfinite(A).all()
np.isfinite(B).all()
np.isfinite(render).all()
~~~

标记：

- NaN/Inf；
- alpha/beta 非有限；
- pAp <= 0；
- residual 突然增加多个数量级；
- B 最大值超过场景历史合理范围。

### 动态连续性

对移动掩码序列记录连续帧：

- norm(B_t - B_previous)；
- 高强度区域方向翻转数量；
- 每次新掩码后的首段 residual；
- 停止移动后的 residual 单调趋势或总体下降趋势。

CG 的欧氏残差不必每步严格单调，但不应持续发散。

### 状态机重启

以下事件必须重新初始化 PCG：

- mu/inv_mu 改变；
- fixed mask/value 改变；
- J/source 改变；
- RHS 或场景改变。

只在 A 被当前 PCG 正常推进、系统矩阵未变化时继续保存的共轭状态。

## 渲染等价性

比较旧 CPU 合成与新 GPU 合成：

~~~python
delta = abs(new_rgb - old_rgb)
mean_abs = delta.mean()
p99 = percentile(delta, 99)
max_abs = delta.max()
~~~

同时检查：

- intensity on/off；
- field line on/off；
- magnet body 灰色；
- interactive material 背景；
- 红色高亮线抵消 heatmap 的行为；
- Uniform、Single Wire、Two Magnets。

若 LUT 使用与 Matplotlib 相同的 256 项表，绝大多数像素应只存在 float32 舍入差异。少量 LUT 边界像素可单独解释，但不能出现结构性颜色偏移。

## 端到端延迟

后端 FPS 高不代表浏览器延迟低。检查完整链路：

~~~text
输入采样
→ 掩码接受
→ 系数更新
→ 首段 PCG
→ 合成
→ 编码
→ 广播
→ createImageBitmap
→ drawImage
~~~

### 浏览器队列检查

用计数器或时间戳确认：

- 同时最多一个 decode；
- pending frame 只保存最新 buffer；
- 解码落后时旧 frame 被覆盖；
- WebSocket bufferedAmount 不持续增长；
- 输入停止后画面不会继续回放旧轨迹。

### 帧率检查

使用 wall-clock duration：

~~~text
encoded_fps = encoded_frames / wall_duration
~~~

不要用只累加 active processing time 的 duration；sleep/yield 被排除后会夸大 FPS。

Windows 上验证短 sleep 的实际行为。若 2 ms 请求被放大到十几毫秒，会让 30 Hz 降为约 23 Hz。可改用 sleep(0) yield 或高分辨率 timer，并重新实测。

## 回归清单

### 后端

- [ ] Python py_compile 通过。
- [ ] 初始场景高精度求解通过。
- [ ] 场景切换首段不出现长阻塞。
- [ ] 材料切换重新构建 coefficients/preconditioner。
- [ ] 掩码静止后继续收敛并最终停止。
- [ ] reset 清除旧场景求解状态。
- [ ] Uniform 固定边界正确。
- [ ] 无 NaN/Inf/breakdown。

### 性能

- [ ] 静止主循环没有 PCG kernel。
- [ ] 动态首段在目标帧预算内。
- [ ] 小/中/大场景 FPS 均有记录。
- [ ] compose 只发生一次最终 readback。
- [ ] JPEG 和 broadcaster 不重复处理同一 sequence。
- [ ] 实际发送 FPS 接近目标上限。
- [ ] Direction 静止时不重复发送，动态更新不超过独立节奏。
- [ ] 箭头量化后的端点误差小于视觉像素容差。
- [ ] Iron Fillings 不再回传 p_start/p_end 或进入逐条 PIL 绘制路径。
- [ ] GPU filings 覆盖密度、重心和方向与参考实现接近。

### UI

- [ ] 主 UI JavaScript 语法通过。
- [ ] draw/shapes UI JavaScript 语法通过。
- [ ] Canvas 尺寸和方向不变。
- [ ] intensity/field line/direction/filings 开关正常。
- [ ] 最新帧队列不会积压。
- [ ] draw 实时跟手、eraser 范围可见。
- [ ] shapes 放置、移动、缩放和擦除正常。

### 工作区

- [ ] git diff --check 通过。
- [ ] benchmark 临时结果没有意外留在仓库。
- [ ] 没有覆盖用户无关改动。
- [ ] 告知用户重启后端并刷新浏览器。

## 结果解读陷阱

### benchmark 强制求解

若 benchmark 在 idle_static 中主动调用 solve_current_system(50)，它测的是求解器吞吐，不是状态变化驱动主循环。两者都要测，但不要混为一谈。

### GPU 异步计时

只在 kernel 调用前后读 perf_counter 会低估 GPU 时间。使用 ti.sync()、标量回读或 to_numpy() 建立同步点。

### 首次 JIT

首次 kernel 编译可能数百毫秒。把它移到明确的 warm-up，并同时报告冷启动和稳态。

### 只看方向或只看残差

方向余弦很高时，场强仍可能有明显误差。残差很小也需要确认算子、边界和 RHS 构建正确。至少联合使用 residual、B relative L2、direction cosine 和固定边界检查。

### 输入采样上限

触摸源为 30 Hz 时，60 FPS 显示不会创造新的输入样本，但可以更快展示求解中间状态、动画和清除浏览器队列。分别说明 input Hz 与 render FPS。
