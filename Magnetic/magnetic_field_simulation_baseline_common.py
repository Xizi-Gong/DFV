"""Shared Magnetic Field baseline runner.

The two baseline entry points mirror the Fluid baselines:

- draw: a freehand mask is sent by the web UI
- shapes: primitive shape masks are sent by the web UI

The solver/rendering path is reused from magnetic_field_simulation.py.  Only the
input source changes: instead of reading the touch shared-memory mask, the
baseline receives a persistent mask from WebSocket commands.
"""
import io
import json
import math
import time
import queue
import asyncio
import threading

import numpy as np
from PIL import Image, ImageDraw
import websockets

import magnetic_field_simulation as mf


res_x = mf.disp_res_x
res_y = mf.disp_res_y

_clients = set()
_clients_lock = threading.Lock()
_cmd_queue = queue.Queue()
_latest_state = None
_latest_frame = None
_latest_arrows = None


def _display_mask_to_full(mask_2d):
    """Convert a top-left-origin display mask into the Taichi sim field layout."""
    display_field = np.fliplr(mask_2d.T).astype(np.float32)
    full = np.zeros((mf.sim_res_x, mf.sim_res_y), dtype=np.float32)
    full[
        mf.offset_x:mf.offset_x + mf.disp_res_x,
        mf.offset_y:mf.offset_y + mf.disp_res_y,
    ] = display_field
    return full


def rasterize_strokes(strokes):
    """strokes: [{"pts":[[nx,ny],...], "r": norm_radius, "erase": bool}]"""
    img = Image.new("L", (res_x, res_y), 0)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        pts = [
            (float(p[0]) * res_x, float(p[1]) * res_y)
            for p in stroke.get("pts", [])
        ]
        if not pts:
            continue
        width = max(1, int(2.0 * float(stroke.get("r", 0.03)) * res_y))
        radius = width / 2.0
        fill = 0 if bool(stroke.get("erase", False)) else 255
        if len(pts) >= 2:
            draw.line(pts, fill=fill, width=width, joint="curve")
        for x, y in pts:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)

    mask = (np.array(img, dtype=np.float32) > 127.0).astype(np.float32)
    return _display_mask_to_full(mask)


def _poly(cx, cy, size, angle, unit_pts):
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for ux, uy in unit_pts:
        dx, dy = ux * size, uy * size
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


_UNIT_SHAPES = {
    "square": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    "triangle": [(0, -1), (0.866, 0.5), (-0.866, 0.5)],
    "trapezoid": [(-0.55, -0.7), (0.55, -0.7), (1.0, 0.7), (-1.0, 0.7)],
}


def rasterize_shapes(shapes):
    """shapes: [{type,cx,cy,size,angle}] in normalized display coordinates."""
    img = Image.new("L", (res_x, res_y), 0)
    draw = ImageDraw.Draw(img)
    for shape in shapes:
        cx = float(shape.get("cx", 0.5)) * res_x
        cy = float(shape.get("cy", 0.5)) * res_y
        size = max(2.0, float(shape.get("size", 0.08)) * res_y)
        angle = math.radians(float(shape.get("angle", 0.0)))
        kind = shape.get("type", "circle")
        if kind == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=255)
        elif kind in _UNIT_SHAPES:
            draw.polygon(_poly(cx, cy, size, angle, _UNIT_SHAPES[kind]), fill=255)

    mask = (np.array(img, dtype=np.float32) > 127.0).astype(np.float32)
    return _display_mask_to_full(mask)


async def _handler(ws, mode):
    global _latest_state, _latest_frame, _latest_arrows
    with _clients_lock:
        _clients.add(ws)
    try:
        await ws.send(json.dumps({
            "type": "init",
            "mode": mode,
            "width": res_x,
            "height": res_y,
            "scenes": mf.SCENE_NAMES,
            "materials": list(mf.interactive_materials.keys()),
        }))
        if _latest_state:
            await ws.send(_latest_state)
        if _latest_frame:
            await ws.send(_latest_frame)
        if _latest_arrows:
            await ws.send(_latest_arrows)
        async for raw in ws:
            if isinstance(raw, str):
                try:
                    _cmd_queue.put_nowait(json.loads(raw))
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        with _clients_lock:
            _clients.discard(ws)


async def _serve(port, mode):
    async with websockets.serve(lambda ws: _handler(ws, mode), "localhost", port):
        print(f"[WS] magnetic {mode} baseline ws://localhost:{port}")
        await asyncio.Future()


def _start_server(port, mode):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=lambda: loop.run_until_complete(_serve(port, mode)),
        daemon=True,
    )
    thread.start()
    return loop


async def _bcast(data):
    with _clients_lock:
        clients = list(_clients)
    if clients:
        await asyncio.gather(*[c.send(data) for c in clients], return_exceptions=True)


def _broadcast(loop, data):
    if _clients:
        asyncio.run_coroutine_threadsafe(_bcast(data), loop)


def _apply_mask(mask_np):
    mf.input_mask.from_numpy(mask_np.astype(np.float32))


def _rebuild_scene_with_mask(scene_number, mask_np, input_material, solve_iters=700):
    mf.set_up_scene(scene_number)
    _apply_mask(mask_np)
    mf.update_materials_with_mask(
        mf.input_mask,
        mf.mu_field,
        mf.sigma_field,
        mf.initial_mask,
        input_material["mu"],
        input_material["sigma"],
    )
    mf.update_inv_mu()
    mf.compute_preconditioner()
    mf.solve_current_system(max_iters=solve_iters, tol=1e-3, verbose=True)


def run_baseline(mode, port, mask_action, rasterizer):
    """Run a Magnetic baseline with a specific mask command/rasterizer."""
    global _latest_state, _latest_frame, _latest_arrows

    loop = _start_server(port, mode)
    mf.initialize_render_lut()

    vis_magnetic_intensity = True
    vis_magnetic_field_line = True
    vis_magnetic_field_direction = False
    vis_iron_filings_method = False
    particle_initialized = False

    scene_number = 0
    material_name = "iron"
    input_material = mf.interactive_materials[material_name]
    mask_np = np.zeros((mf.sim_res_x, mf.sim_res_y), dtype=np.float32)

    highlight_A_field_gpu = mf.ti.field(mf.ti.f32, shape=3)
    highlight_count_gpu = mf.ti.field(mf.ti.i32, shape=())
    highlight_count_gpu[()] = 0

    print("Pre-computing magnetic baseline...")
    _rebuild_scene_with_mask(scene_number, mask_np, input_material, solve_iters=2000)
    highlight_count_value = mf.update_auto_highlights(
        highlight_A_field_gpu, highlight_count_gpu
    )
    mf.warm_optional_visualization_kernels(
        highlight_A_field_gpu, highlight_count_value
    )
    print("Warm up done.")

    def publish_state():
        global _latest_state
        state = json.dumps(
            {
                "type": "state",
                "mode": mode,
                "scene": scene_number,
                "scene_name": mf.SCENE_NAMES[scene_number],
                "material": material_name,
                "intensity": vis_magnetic_intensity,
                "fieldline": vis_magnetic_field_line,
                "direction": vis_magnetic_field_direction,
                "filings": vis_iron_filings_method,
            },
            separators=(",", ":"),
        )
        _latest_state = state
        _broadcast(loop, state)

    publish_state()
    last_enc = time.perf_counter()
    mask_dirty = False
    field_dirty = True
    b_initialized = False
    highlights_pending = mf.current_solver_active()
    arrows_dirty = True
    cached_arrow_dirs = cached_arrow_starts = None
    cached_arrow_quantized = None
    last_direction_update = -1e9

    while True:
        latest_mask_cmd = None

        def apply_latest_mask_cmd():
            nonlocal latest_mask_cmd, mask_np, mask_dirty
            if latest_mask_cmd is None:
                return
            payload = latest_mask_cmd.get(
                "strokes", latest_mask_cmd.get("shapes", [])
            )
            new_mask = rasterizer(payload)
            if not np.array_equal(new_mask, mask_np):
                mask_np = new_mask
                mask_dirty = True
            latest_mask_cmd = None

        while not _cmd_queue.empty():
            try:
                cmd = _cmd_queue.get_nowait()
                action = cmd.get("action")
                value = cmd.get("value")

                if action == mask_action:
                    latest_mask_cmd = cmd
                    continue

                apply_latest_mask_cmd()

                if (
                    action == "set_scene"
                    and isinstance(value, int)
                    and 0 <= value < len(mf.SCENE_NAMES)
                ):
                    scene_number = value
                    _rebuild_scene_with_mask(
                        scene_number,
                        mask_np,
                        input_material,
                        solve_iters=mf.SCENE_CHANGE_SOLVE_ITERS,
                    )
                    highlight_count_value = mf.update_auto_highlights(
                        highlight_A_field_gpu, highlight_count_gpu
                    )
                    highlights_pending = mf.current_solver_active()
                    particle_initialized = False
                    mask_dirty = False
                    field_dirty = True
                    arrows_dirty = True
                    publish_state()
                elif action == "set_material" and value in mf.interactive_materials:
                    material_name = value
                    input_material = mf.interactive_materials[material_name]
                    mask_dirty = True
                    publish_state()
                elif action == "toggle_intensity":
                    vis_magnetic_intensity = bool(value)
                    publish_state()
                elif action == "toggle_fieldline":
                    vis_magnetic_field_line = bool(value)
                    publish_state()
                elif action == "toggle_direction":
                    vis_magnetic_field_direction = bool(value)
                    arrows_dirty = True
                    publish_state()
                elif action == "toggle_filings":
                    vis_iron_filings_method = bool(value)
                    particle_initialized = False
                    publish_state()
                elif action == "clear_mask":
                    mask_np = np.zeros(
                        (mf.sim_res_x, mf.sim_res_y), dtype=np.float32
                    )
                    mask_dirty = True
                elif action == "reset":
                    _rebuild_scene_with_mask(
                        scene_number,
                        mask_np,
                        input_material,
                        solve_iters=mf.SCENE_CHANGE_SOLVE_ITERS,
                    )
                    highlight_count_value = mf.update_auto_highlights(
                        highlight_A_field_gpu, highlight_count_gpu
                    )
                    highlights_pending = mf.current_solver_active()
                    particle_initialized = False
                    mask_dirty = False
                    field_dirty = True
                    arrows_dirty = True
                    publish_state()
            except Exception as exc:
                print(f"[WS] command failed: {exc}")

        apply_latest_mask_cmd()

        if mask_dirty:
            _apply_mask(mask_np)
            mf.update_materials_with_mask(
                mf.input_mask,
                mf.mu_field,
                mf.sigma_field,
                mf.initial_mask,
                input_material["mu"],
                input_material["sigma"],
            )
            mf.update_inv_mu()
            mf.compute_preconditioner()
            mf.start_current_system(
                tol=mf.INTERACTIVE_SOLVE_TOL, rebuild_rhs=True
            )
            mf.continue_current_system(max_iters=mf.INTERACTIVE_SOLVE_ITERS)
            highlights_pending = True
            particle_initialized = False
            mask_dirty = False
            field_dirty = True
            arrows_dirty = True
        elif mf.current_solver_active():
            iters = mf.continue_current_system(
                max_iters=mf.SETTLE_SOLVE_ITERS
            )
            if iters > 0:
                field_dirty = True
                arrows_dirty = True

        if highlights_pending and not mf.current_solver_active():
            highlight_count_value = mf.update_auto_highlights(
                highlight_A_field_gpu, highlight_count_gpu
            )
            highlights_pending = False

        now = time.perf_counter()
        if now - last_enc < 1 / mf.TARGET_RENDER_FPS:
            if not mf.current_solver_active():
                time.sleep(0)
            continue

        if field_dirty or not b_initialized:
            mf.compute_B_field(mf.A_field, mf.B_field)
            field_dirty = False
            b_initialized = True
            arrows_dirty = True

        mf.compose_magnetic_frame(
            mf.A_field,
            mf.B_field,
            mf.color_field,
            mf.mu_field,
            highlight_A_field_gpu,
            highlight_count_value,
            int(vis_magnetic_intensity),
            int(vis_magnetic_field_line),
        )

        if vis_iron_filings_method:
            if not particle_initialized:
                mf.init_particles()
                for _ in range(10):
                    mf.compute_density_grid()
                    mf.update_particles_fast()
                particle_initialized = True
            else:
                mf.compute_density_grid()
                mf.update_particles_fast()
            mf.render_filings_to_color(mf.B_field, mf.color_field)

        crop_img = mf.color_field.to_numpy()

        if vis_magnetic_field_direction:
            update_due = (
                now - last_direction_update >= 1 / mf.DIRECTION_UPDATE_FPS
            )
            if (
                (arrows_dirty and update_due)
                or cached_arrow_dirs is None
                or cached_arrow_quantized is None
            ):
                cached_arrow_dirs, cached_arrow_starts = mf.calculate_streamline(
                    mf.B_field
                )
                arrow_rows = np.column_stack(
                    (cached_arrow_starts, cached_arrow_dirs)
                )
                quantized = np.rint(
                    arrow_rows * mf.DIRECTION_QUANTIZATION
                ).astype(np.int16)
                if (
                    cached_arrow_quantized is None
                    or not np.array_equal(quantized, cached_arrow_quantized)
                ):
                    flat = quantized.astype(np.int32).reshape(-1).tolist()
                    payload = json.dumps(
                        {
                            "type": "arrows",
                            "scale": mf.DIRECTION_QUANTIZATION,
                            "data": flat,
                        },
                        separators=(",", ":"),
                    )
                    if payload != _latest_arrows:
                        _latest_arrows = payload
                        _broadcast(loop, _latest_arrows)
                    cached_arrow_quantized = quantized
                last_direction_update = now
                arrows_dirty = False
        else:
            empty_arrows = '{"type":"arrows","data":[]}'
            if _latest_arrows != empty_arrows:
                _latest_arrows = empty_arrows
                _broadcast(loop, _latest_arrows)
            cached_arrow_quantized = None

        img_u8 = (crop_img * 255).clip(0, 255).astype(np.uint8)
        img_u8 = np.ascontiguousarray(
            np.flipud(np.transpose(img_u8, (1, 0, 2)))
        )
        bgr = mf.cv2.cvtColor(img_u8, mf.cv2.COLOR_RGB2BGR)
        ok, encoded = mf.cv2.imencode(
            ".jpg",
            bgr,
            [int(mf.cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if ok:
            frame_bytes = encoded.tobytes()
        else:
            buf = io.BytesIO()
            Image.fromarray(img_u8).save(buf, format="JPEG", quality=85)
            frame_bytes = buf.getvalue()

        _latest_frame = frame_bytes
        _broadcast(loop, _latest_frame)
        last_enc = now
