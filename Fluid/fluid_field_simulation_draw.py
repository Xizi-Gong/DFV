"""Baseline 1 — DRAW mode backend.

Reuses the validated solver from fluid_field_simulation.py (imported, NOT modified).
Obstacles come from freehand strokes drawn in the web UI (no raw heatmap, no karman cylinder).
WebSocket on ws://localhost:8770. Run:  python fluid_field_simulation_draw.py
"""
import io
import json
import time
import queue
import asyncio
import threading

import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import websockets

import fluid_field_simulation as fs

PORT   = 8770
res_x  = fs.res_x
res_y  = fs.res_y
dt     = fs.dt
cmap   = plt.cm.coolwarm

# ──────────────────────────────────────────────
#  WebSocket（独立端口，复用 fs 的求解器）
# ──────────────────────────────────────────────
_clients      = set()
_clients_lock = threading.Lock()
_cmd_queue    = queue.Queue()
_latest_state = None
_latest_frame = None

async def _handler(ws):
    with _clients_lock:
        _clients.add(ws)
    try:
        await ws.send(json.dumps({"type": "init", "width": res_x, "height": res_y,
                                  "materials": list(fs.fluid_properties.keys())}))
        if _latest_state:
            await ws.send(_latest_state)
        if _latest_frame:
            await ws.send(_latest_frame)
        async for raw in ws:
            try:
                _cmd_queue.put_nowait(json.loads(raw))
            except Exception:
                pass
    except Exception:
        pass
    finally:
        with _clients_lock:
            _clients.discard(ws)

async def _serve():
    async with websockets.serve(_handler, "localhost", PORT):
        print(f"[WS] draw baseline ws://localhost:{PORT}")
        await asyncio.Future()

_loop   = asyncio.new_event_loop()
_thread = threading.Thread(target=lambda: _loop.run_until_complete(_serve()), daemon=True)

async def _bcast(data):
    with _clients_lock:
        cs = list(_clients)
    if cs:
        await asyncio.gather(*[c.send(data) for c in cs], return_exceptions=True)

def broadcast(data):
    if _clients:
        asyncio.run_coroutine_threadsafe(_bcast(data), _loop)


# ──────────────────────────────────────────────
#  材质 / 障碍物栅格化
# ──────────────────────────────────────────────
def apply_material(name):
    fs.fluid_material = name
    fs.RHO = fs.fluid_properties[name]["rho"]
    fs.VISCOSITY = fs.fluid_properties[name]["viscosity"]
    fs.PIXEL_SIZE = fs.PIXEL_SIZE_MAP[name]
    fs._pixel_size_field[None] = fs.PIXEL_SIZE
    fs._rho_field[None] = fs.RHO
    fs._nu_field[None] = fs.VISCOSITY

def rasterize_strokes(strokes):
    """strokes: [{"pts":[[nx,ny],...], "r": norm_radius, "erase": bool}]，归一化坐标。
    返回 (res_x,res_y) float32，方向与 _solid 一致。"""
    img = Image.new("L", (res_x, res_y), 0)
    d = ImageDraw.Draw(img)
    for s in strokes:
        pts = [(float(p[0]) * res_x, float(p[1]) * res_y) for p in s.get("pts", [])]
        if not pts:
            continue
        w = max(1, int(2.0 * float(s.get("r", 0.03)) * res_y))
        r = w / 2.0
        fill = 0 if bool(s.get("erase", False)) else 255
        if len(pts) >= 2:
            d.line(pts, fill=fill, width=w, joint="curve")
        for (x, y) in (pts[0], pts[-1]):                 # 圆头
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        for (x, y) in pts:                                # 路径每点也补圆，避免细缝
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
    arr = (np.array(img, dtype=np.float32) > 127.0).astype(np.float32)   # (res_y,res_x)
    return np.fliplr(arr.T)                                              # → (res_x,res_y)

def set_obstacle_mask(mask_np):
    fs._solid.from_numpy(mask_np)
    fs.refine_solid_mask(fs._solid)


# ──────────────────────────────────────────────
#  渲染（复刻 fs.main 的可视化分支）
# ──────────────────────────────────────────────
def compose(viz):
    if viz == "vortex":
        fs.vorticity(fs.velocities_pair.cur)
        gray = fs.velocity_curls.to_numpy() * 0.03 + 0.5
        return np.stack([gray, gray, gray], axis=-1)
    if viz == "velocity":
        vel = fs.velocities_pair.cur.to_numpy() * 0.01 + 0.5
        return np.concatenate([vel, np.zeros_like(vel[..., :1])], axis=-1)
    if viz == "pressure":
        p = fs.pressures_pair.cur.to_numpy()
        P_RANGE = 40000.0
        p_norm = np.clip((p - np.mean(p) + P_RANGE) / (2 * P_RANGE), 0.0, 1.0)
        return cmap(p_norm)[..., :3].astype(np.float32)
    # smoke (默认) + bloom
    frame = fs.color_field.to_numpy()
    lum = (0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2])[..., None]
    bright = np.maximum(lum - 0.82, 0) * frame / (lum + 1e-6)
    h, w = bright.shape[:2]
    small = cv2.resize(bright, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=1.2)
    bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    frame = np.clip(frame + bloom * 0.05, 0.0, 1.0)
    frame = np.power(np.clip(frame, 0, 1), 0.90)
    return np.clip((frame - 0.02) * 1.04, 0, 1)


# ──────────────────────────────────────────────
#  主循环
# ──────────────────────────────────────────────
def run():
    global _latest_state, _latest_frame

    material = "air"
    apply_material(material)
    inlet_speed_mps = fs.fluid_properties[material]["u_in_default"] / fs.SOLVER_SPEED_PER_MPS
    viz = "smoke"
    streamline = False

    smoke_info = np.zeros((len(fs.smoke_positions), 2), dtype=np.int16)
    for i in range(len(fs.smoke_positions)):
        smoke_info[i, 0] = int(fs.smoke_positions[i] * res_y - fs.smoke_width[i] / 2)
        smoke_info[i, 1] = int(fs.smoke_positions[i] * res_y + fs.smoke_width[i] / 2)

    fs.reset()
    fs._solid.fill(0)

    def publish_state():
        global _latest_state
        st = json.dumps({
            "type": "state", "mode": "draw",
            "material": material, "speed_mps": inlet_speed_mps,
            "viz": viz.capitalize(), "streamline": streamline,
            "wind_speed_min_mps": fs.WIND_SPEED_MIN_MPS,
            "wind_speed_max_mps": fs.WIND_SPEED_MAX_MPS,
            "wind_speed_step_mps": fs.WIND_SPEED_STEP_MPS,
            "solver_speed_per_mps": fs.SOLVER_SPEED_PER_MPS,
        })
        _latest_state = st
        broadcast(st)

    _thread.start()
    publish_state()

    last_enc = time.time()
    last_re = 0.0
    reynolds = 0.0

    while True:
        while not _cmd_queue.empty():
            try:
                cmd = _cmd_queue.get_nowait()
                action = cmd.get("action")
                value = cmd.get("value")
                if action == "set_material" and value in fs.fluid_properties:
                    material = value
                    apply_material(material)
                    inlet_speed_mps = min(max(fs.fluid_properties[material]["u_in_default"] / fs.SOLVER_SPEED_PER_MPS,
                                              fs.WIND_SPEED_MIN_MPS), fs.WIND_SPEED_MAX_MPS)
                    saved = fs._solid.to_numpy()
                    fs.reset()
                    fs._solid.from_numpy(saved)        # reset 会清掉障碍，恢复之
                    publish_state()
                elif action == "set_speed":
                    inlet_speed_mps = min(max(float(value), fs.WIND_SPEED_MIN_MPS), fs.WIND_SPEED_MAX_MPS)
                    publish_state()
                elif action == "set_viz":
                    viz = str(value)
                    publish_state()
                elif action == "toggle_streamline":
                    streamline = bool(value)
                    publish_state()
                elif action == "set_obstacles":
                    set_obstacle_mask(rasterize_strokes(cmd.get("strokes", [])))
                elif action == "clear_obstacles":
                    fs._solid.fill(0)
                elif action == "reset":
                    saved = fs._solid.to_numpy()
                    fs.reset()
                    fs._solid.from_numpy(saved)        # 保留已画障碍，只清流场
                    publish_state()
            except Exception:
                pass

        active_solver_speed = inlet_speed_mps * fs.SOLVER_SPEED_PER_MPS

        mask = np.array(fs._solid.to_numpy()[1:-2, 1:-2], dtype=bool)
        bounds_x = fs.get_mask_span_x(mask)
        L_char = (bounds_x * fs.PIXEL_SIZE) if bounds_x > 1.0 else (res_y * fs.PIXEL_SIZE)
        reynolds = inlet_speed_mps * L_char / max(fs.VISCOSITY, 1e-12)

        fs.step(active_solver_speed, smoke_info)

        now = time.time()
        if now - last_enc >= 1 / 30:
            frame_rgb = compose(viz)
            img = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
            img = np.flipud(np.transpose(img, (1, 0, 2)))
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=92)
            _latest_frame = buf.getvalue()
            broadcast(_latest_frame)
            last_enc = now

            if streamline:
                directions, starts, mags = fs.calculate_streamline(fs.velocities_pair.cur, fs._solid)
                flat = []
                for (ox, oy), (dx, dy), mag in zip(starts, directions, mags):
                    flat.extend([round(float(ox), 4), round(float(oy), 4),
                                 round(float(dx), 5), round(float(dy), 5), round(float(mag), 3)])
                broadcast(json.dumps({"type": "arrows", "data": flat}))
            else:
                broadcast(json.dumps({"type": "arrows", "data": []}))

        if now - last_re >= 1.0:
            broadcast(json.dumps({"type": "reynolds", "value": reynolds}))
            last_re = now


if __name__ == "__main__":
    run()
