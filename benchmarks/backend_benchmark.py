import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark_results"


MAGNETIC_SCENES = [
    {"name": "small_single_wire", "scene_index": 0},
    {"name": "medium_quadrupole", "scene_index": 3},
    {"name": "large_two_magnets", "scene_index": 6},
]

FLUID_SCENES = [
    {"name": "small_honey", "material": "honey"},
    {"name": "medium_air", "material": "air"},
    {"name": "large_water", "material": "water"},
]

HEAT_SCENES = [
    {"name": "small_foam_plastic", "material": "Foam Plastic"},
    {"name": "medium_steel", "material": "Steel"},
    {"name": "large_copper", "material": "Copper"},
]

INPUT_CONDITIONS = [
    {"condition": "no_mask_stable", "mode": "none", "phase": "no_mask"},
    {"condition": "press_hold_dynamic", "mode": "press_hold", "phase": "dynamic"},
    {"condition": "sweep_dynamic", "mode": "sweep", "phase": "dynamic"},
    {"condition": "pulsing_contact_dynamic", "mode": "pulsing_contact", "phase": "dynamic"},
    {"condition": "recovery_after_sweep", "mode": "sweep", "phase": "recovery"},
]

GATE_CONDITIONS = [
    {"condition": "idle_static", "mode": "none", "phase": "no_mask"},
    {"condition": "interactive_sparse", "mode": "press_hold", "phase": "dynamic"},
    {"condition": "interactive_dense", "mode": "sweep", "phase": "dynamic"},
    {"condition": "idle_recovery", "mode": "sweep", "phase": "recovery"},
]


def import_from_path(name, path, extra_path):
    sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_heat_without_loop():
    heat_dir = ROOT / "Heat"
    path = heat_dir / "heat_diffusion_ws.py"
    sys.path.insert(0, str(heat_dir))
    source = path.read_text(encoding="utf-8")
    marker = "# ──────────────────────────────────────────────\n#  启动"
    if marker not in source:
        raise RuntimeError("Heat startup marker not found; refusing to guess benchmark split.")
    prefix = source.split(marker, 1)[0]
    ns = {
        "__name__": "heat_diffusion_ws_benchmark",
        "__file__": str(path),
        "__package__": None,
    }
    exec(compile(prefix, str(path), "exec"), ns)
    return ns


def sync_taichi(module_or_ns):
    ti = module_or_ns["ti"] if isinstance(module_or_ns, dict) else module_or_ns.ti
    ti.sync()


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def query_nvidia():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        if not out:
            return None, None
        first = out.splitlines()[0]
        util, mem = [float(part.strip()) for part in first.split(",")[:2]]
        return util, mem
    except Exception:
        return None, None


def synthetic_hand_mask(mode, frame_idx, total_frames, width=78, height=52):
    arr = np.full((height, width), 255, dtype=np.uint8)
    if mode == "none":
        return arr

    yy, xx = np.mgrid[0:height, 0:width]
    if mode == "press_hold":
        cx, cy = width * 0.50, height * 0.54
        rx, ry = width * 0.14, height * 0.18
        contact = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    elif mode == "sweep":
        denom = max(total_frames - 1, 1)
        t = frame_idx / denom
        cx = width * (0.18 + 0.64 * t)
        cy = height * (0.50 + 0.18 * math.sin(2.0 * math.pi * t))
        rx, ry = width * 0.12, height * 0.16
        contact = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    elif mode == "pulsing_contact":
        pulse = 0.5 + 0.5 * math.sin(2.0 * math.pi * frame_idx / 6.0)
        cx, cy = width * 0.42, height * 0.48
        rx = width * (0.08 + 0.08 * pulse)
        ry = height * (0.10 + 0.10 * pulse)
        contact = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        if frame_idx % 8 in (6, 7):
            contact[:] = False
    else:
        raise ValueError(f"Unknown synthetic hand mask mode: {mode}")

    arr[contact] = 0
    return arr


def result_record(simulator, scene, input_condition, input_mode, input_phase, mask_active_frames,
                  frame_times, sim_times, render_times, nan_detected, diverged, crashed, notes):
    proc = psutil.Process(os.getpid())
    gpu_util, vram = query_nvidia()
    total_s = sum(frame_times)
    fps = (len(frame_times) / total_s) if total_s > 0 else 0.0
    return {
        "simulator": simulator,
        "scene": scene,
        "input_condition": input_condition,
        "input_mode": input_mode,
        "input_phase": input_phase,
        "mask_active_frames": int(mask_active_frames),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fps_avg": fps,
        "frame_time_p95_ms": percentile(frame_times, 95) * 1000.0 if frame_times else None,
        "sim_step_ms": statistics.mean(sim_times) * 1000.0 if sim_times else None,
        "render_ms": statistics.mean(render_times) * 1000.0 if render_times else None,
        "gpu_util_avg": gpu_util,
        "vram_mb": vram,
        "rss_mb": proc.memory_info().rss / (1024 * 1024),
        "nan_detected": bool(nan_detected),
        "diverged": bool(diverged),
        "crashed": bool(crashed),
        "notes": notes,
    }


def save_result(record):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{record['simulator']}_{record['scene']}_{record['input_condition']}_{ts}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def set_fluid_material(m, material):
    m.fluid_material = material
    m.RHO = m.fluid_properties[material]["rho"]
    m.VISCOSITY = m.fluid_properties[material]["viscosity"]
    m.PIXEL_SIZE = m.PIXEL_SIZE_MAP[material]
    m.U_IN = m.fluid_properties[material]["u_in_default"]
    m._rho_field[None] = m.RHO
    m._nu_field[None] = m.VISCOSITY
    m._pixel_size_field[None] = m.PIXEL_SIZE


def feed_shared_mask(module, mask):
    with module.shared_mask_lock:
        module.shared_mask_np = mask
        module.new_mask_available = True


def render_fluid_smoke(m):
    frame_rgb = m.color_field.to_numpy()
    lum = (0.2126 * frame_rgb[..., 0] + 0.7152 * frame_rgb[..., 1] + 0.0722 * frame_rgb[..., 2])[..., None]
    bright_mask = np.maximum(lum - 0.82, 0) * frame_rgb / (lum + 1e-6)
    h, w = bright_mask.shape[:2]
    small = m.cv2.resize(bright_mask, (w // 2, h // 2))
    blurred = m.cv2.GaussianBlur(small, (0, 0), sigmaX=2.2)
    bloom = m.cv2.resize(blurred, (w, h)).astype(np.float32)
    frame_rgb = np.clip(frame_rgb + bloom * 0.05, 0.0, 1.0)
    frame_rgb = np.power(np.clip(frame_rgb, 0, 1), 0.90)
    return np.clip((frame_rgb - 0.02) * 1.04, 0, 1)


def apply_fluid_mask(m, mask):
    feed_shared_mask(m, mask)
    if m.process_mask_update(threshold=160.0, mode="threshold"):
        m.refine_solid_mask(m._solid)


def apply_magnetic_mask(m, mask, input_material, state):
    feed_shared_mask(m, mask)
    got_mask = m.process_mask_update(threshold=160.0)
    mask_changed = False
    if got_mask:
        mask_diff = m.mask_l1_diff(m.input_mask, m.last_input_mask)
        mask_changed = state["first_mask_frame"] or (mask_diff > getattr(m, "MASK_DIFF_THRESH", 100.0))
    if mask_changed:
        m.update_materials_with_mask(
            m.input_mask,
            m.mu_field,
            m.sigma_field,
            m.initial_mask,
            input_material["mu"],
            input_material["sigma"],
        )
        m.update_inv_mu()
        m.compute_preconditioner()
        m.solve_current_system(max_iters=getattr(m, "MASK_CHANGED_SOLVE_ITERS", 150), tol=1e-3, verbose=False)
        m.copy_mask(m.input_mask, m.last_input_mask)
        state["first_mask_frame"] = False
    else:
        m.solve_current_system(max_iters=getattr(m, "MASK_STEADY_SOLVE_ITERS", 50), tol=1e-3, verbose=False)


def apply_heat_mask(h, mask):
    data = mask.astype(np.float32) / 255.0
    h["input_data"].from_numpy(data)
    h["process_input_kernel"](h["input_threshold"] / 255.0, h["heat_intensity_scale"])


def measured_mask_for_condition(condition, frame_idx, total_frames):
    if condition["phase"] == "recovery":
        return synthetic_hand_mask("none", frame_idx, total_frames)
    return synthetic_hand_mask(condition["mode"], frame_idx, total_frames)


def run_fluid(args):
    m = import_from_path(
        "fluid_field_simulation_bench",
        ROOT / "Fluid" / "fluid_field_simulation.py",
        ROOT / "Fluid",
    )
    results = []
    smoke_info = np.zeros((len(m.smoke_positions), 2), dtype=np.int16)
    for i in range(len(m.smoke_positions)):
        smoke_info[i, 0] = int(m.smoke_positions[i] * m.res_y - m.smoke_width[i] / 2)
        smoke_info[i, 1] = int(m.smoke_positions[i] * m.res_y + m.smoke_width[i] / 2)

    conditions = GATE_CONDITIONS if args.gate_conditions else INPUT_CONDITIONS
    scenes = [FLUID_SCENES[args.scene_index]] if args.scene_index is not None else FLUID_SCENES
    for scene in scenes:
        for condition in conditions:
            crashed = False
            nan_detected = False
            diverged = False
            frame_times = []
            sim_times = []
            render_times = []
            mask_active_frames = 0
            notes = (
                "Backend-only benchmark. GUI and browser WebSocket end-to-end FPS are not measured. "
                "Fluid no_mask_stable uses the existing default Karman cylinder path; dynamic/recovery "
                "conditions use the existing shared hand-mask obstacle path."
            )
            try:
                set_fluid_material(m, scene["material"])
                m.reset()
                if condition["phase"] == "no_mask":
                    m.init_cylinder_obstacle()
                sync_taichi(m)
                if condition["phase"] == "recovery":
                    for i in range(args.recovery_prep):
                        apply_fluid_mask(m, synthetic_hand_mask(condition["mode"], i, args.recovery_prep))
                        m.step(m.U_IN, smoke_info)
                    sync_taichi(m)
                for i in range(args.warmup):
                    mask = measured_mask_for_condition(condition, i, args.warmup)
                    if condition["phase"] != "no_mask":
                        apply_fluid_mask(m, mask)
                    m.step(m.U_IN, smoke_info)
                    sync_taichi(m)
                    _ = render_fluid_smoke(m)
                for i in range(args.frames):
                    mask = measured_mask_for_condition(condition, i, args.frames)
                    if mask.min() < 160:
                        mask_active_frames += 1
                    t0 = time.perf_counter()
                    s0 = time.perf_counter()
                    if condition["phase"] != "no_mask":
                        apply_fluid_mask(m, mask)
                    m.step(m.U_IN, smoke_info)
                    sync_taichi(m)
                    s1 = time.perf_counter()
                    frame_rgb = render_fluid_smoke(m)
                    r1 = time.perf_counter()
                    nan_detected = nan_detected or bool(np.isnan(frame_rgb).any())
                    diverged = diverged or bool(np.max(np.abs(frame_rgb)) > 10.0)
                    sim_times.append(s1 - s0)
                    render_times.append(r1 - s1)
                    frame_times.append(r1 - t0)
            except Exception as exc:
                crashed = True
                notes += f" Exception: {type(exc).__name__}: {exc}"
            record = result_record(
                "fluid", scene["name"], condition["condition"], condition["mode"],
                condition["phase"], mask_active_frames, frame_times, sim_times, render_times,
                nan_detected, diverged, crashed, notes
            )
            save_result(record)
            results.append(record)
    return results


def run_magnetic(args):
    m = import_from_path(
        "magnetic_field_simulation_bench",
        ROOT / "Magnetic" / "magnetic_field_simulation.py",
        ROOT / "Magnetic",
    )
    results = []
    highlight_field = m.ti.field(m.ti.f32, shape=3)
    highlight_count = m.ti.field(m.ti.i32, shape=())
    input_material = m.interactive_materials["iron"]
    conditions = GATE_CONDITIONS if args.gate_conditions else INPUT_CONDITIONS
    scenes = [MAGNETIC_SCENES[args.scene_index]] if args.scene_index is not None else MAGNETIC_SCENES
    for scene in scenes:
        for condition in conditions:
            crashed = False
            nan_detected = False
            diverged = False
            frame_times = []
            sim_times = []
            render_times = []
            mask_active_frames = 0
            notes = (
                "Backend-only benchmark. GUI and browser WebSocket end-to-end FPS are not measured. "
                "Dynamic/recovery conditions use the existing shared hand-mask material update path with iron material."
            )
            try:
                state = {"first_mask_frame": True}
                m.set_up_scene(scene["scene_index"])
                m.last_input_mask.fill(0.0)
                m.input_mask.fill(0.0)
                m.update_inv_mu()
                m.compute_preconditioner()
                m.solve_current_system(max_iters=50, tol=1e-3, verbose=False)
                m.update_auto_highlights(highlight_field, highlight_count)
                sync_taichi(m)
                if condition["phase"] == "recovery":
                    for i in range(args.recovery_prep):
                        apply_magnetic_mask(m, synthetic_hand_mask(condition["mode"], i, args.recovery_prep), input_material, state)
                        m.compute_B_field(m.A_field, m.B_field)
                    sync_taichi(m)
                for i in range(args.warmup):
                    mask = measured_mask_for_condition(condition, i, args.warmup)
                    if condition["phase"] == "no_mask":
                        m.solve_current_system(max_iters=50, tol=1e-3, verbose=False)
                    else:
                        apply_magnetic_mask(m, mask, input_material, state)
                    m.compute_B_field(m.A_field, m.B_field)
                    m.compute_magnetic_intensity(m.B_field, m.magnetic_intensity_field)
                    m.compute_and_render(m.A_field, m.color_field, m.mu_field,
                                         highlight_field, highlight_count[()])
                    sync_taichi(m)
                    _ = m.magnetic_intensity_field.to_numpy()
                    _ = m.color_field.to_numpy()
                for i in range(args.frames):
                    mask = measured_mask_for_condition(condition, i, args.frames)
                    if mask.min() < 160:
                        mask_active_frames += 1
                    t0 = time.perf_counter()
                    s0 = time.perf_counter()
                    if condition["phase"] == "no_mask":
                        m.solve_current_system(max_iters=50, tol=1e-3, verbose=False)
                    else:
                        apply_magnetic_mask(m, mask, input_material, state)
                    m.compute_B_field(m.A_field, m.B_field)
                    sync_taichi(m)
                    s1 = time.perf_counter()
                    m.compute_magnetic_intensity(m.B_field, m.magnetic_intensity_field)
                    m.compute_and_render(m.A_field, m.color_field, m.mu_field,
                                         highlight_field, highlight_count[()])
                    sync_taichi(m)
                    intensity = m.magnetic_intensity_field.to_numpy()
                    color = m.color_field.to_numpy()
                    r1 = time.perf_counter()
                    nan_detected = nan_detected or bool(np.isnan(intensity).any() or np.isnan(color).any())
                    diverged = diverged or bool(np.max(np.abs(intensity)) > 1e12)
                    sim_times.append(s1 - s0)
                    render_times.append(r1 - s1)
                    frame_times.append(r1 - t0)
            except Exception as exc:
                crashed = True
                notes += f" Exception: {type(exc).__name__}: {exc}"
            record = result_record(
                "magnetic", scene["name"], condition["condition"], condition["mode"],
                condition["phase"], mask_active_frames, frame_times, sim_times, render_times,
                nan_detected, diverged, crashed, notes
            )
            save_result(record)
            results.append(record)
    return results


def run_heat(args):
    h = load_heat_without_loop()
    results = []
    conditions = GATE_CONDITIONS if args.gate_conditions else INPUT_CONDITIONS
    scenes = [HEAT_SCENES[args.scene_index]] if args.scene_index is not None else HEAT_SCENES
    for scene in scenes:
        for condition in conditions:
            crashed = False
            nan_detected = False
            diverged = False
            frame_times = []
            sim_times = []
            render_times = []
            mask_active_frames = 0
            notes = (
                "Backend-only benchmark. Heat has no import-safe main guard; benchmark executes the existing "
                "module code before the startup loop, then reuses init_fields/buildMatrices/solver/update/encode hooks. "
                "Browser WebGL end-to-end FPS requires frontend automation conditions not yet defined."
            )
            try:
                random.seed(1234)
                h["random"].seed(1234)
                h["k"] = float(h["MATERIAL_PRESETS"][scene["material"]])
                h["current_material_name"] = scene["material"]
                h["init_fields"]()
                h["buildMatrices"]()
                h["particle_system"].reset()
                sync_taichi(h)
                if condition["phase"] == "recovery":
                    for i in range(args.recovery_prep):
                        apply_heat_mask(h, synthetic_hand_mask(condition["mode"], i, args.recovery_prep))
                        h["t_np1"].from_numpy(h["precomputed_solver"].solve(h["t_n"]))
                        h["update_temp_kernel"](h["t_ambient"], h["cooling_rate"])
                        h["t_n"].copy_from(h["t_np1"])
                    sync_taichi(h)
                for i in range(args.warmup):
                    mask = measured_mask_for_condition(condition, i, args.warmup)
                    apply_heat_mask(h, mask)
                    h["t_np1"].from_numpy(h["precomputed_solver"].solve(h["t_n"]))
                    h["update_temp_kernel"](h["t_ambient"], h["cooling_rate"])
                    h["t_n"].copy_from(h["t_np1"])
                    h["compute_gradients"]()
                    sync_taichi(h)
                    gm = h["gradient_magnitude"].to_numpy()
                    gx = h["gradient_x"].to_numpy()
                    gy = h["gradient_y"].to_numpy()
                    if h["show_gradient_lines"]:
                        h["particle_system"].update(gx, gy, gm)
                    _ = h["encode_temp_frame"]()
                for i in range(args.frames):
                    mask = measured_mask_for_condition(condition, i, args.frames)
                    if mask.min() < 160:
                        mask_active_frames += 1
                    t0 = time.perf_counter()
                    s0 = time.perf_counter()
                    apply_heat_mask(h, mask)
                    for _ in range(h["substep"]):
                        h["t_np1"].from_numpy(h["precomputed_solver"].solve(h["t_n"]))
                        h["update_temp_kernel"](h["t_ambient"], h["cooling_rate"])
                        h["t_n"].copy_from(h["t_np1"])
                    h["compute_gradients"]()
                    sync_taichi(h)
                    s1 = time.perf_counter()
                    gm = h["gradient_magnitude"].to_numpy()
                    gx = h["gradient_x"].to_numpy()
                    gy = h["gradient_y"].to_numpy()
                    if h["show_gradient_lines"]:
                        h["particle_system"].update(gx, gy, gm)
                    frame_bytes = h["encode_temp_frame"]()
                    r1 = time.perf_counter()
                    t_arr = h["t_np1"].to_numpy()
                    nan_detected = nan_detected or bool(np.isnan(t_arr).any())
                    diverged = diverged or bool(np.min(t_arr) < -1e4 or np.max(t_arr) > 1e4 or len(frame_bytes) == 0)
                    sim_times.append(s1 - s0)
                    render_times.append(r1 - s1)
                    frame_times.append(r1 - t0)
            except Exception as exc:
                crashed = True
                notes += f" Exception: {type(exc).__name__}: {exc}"
            record = result_record(
                "heat", scene["name"], condition["condition"], condition["mode"],
                condition["phase"], mask_active_frames, frame_times, sim_times, render_times,
                nan_detected, diverged, crashed, notes
            )
            save_result(record)
            results.append(record)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", choices=["magnetic", "fluid", "heat"], required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--recovery-prep", type=int, default=8)
    parser.add_argument("--gate-conditions", action="store_true")
    parser.add_argument("--scene-index", type=int, default=None)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "tbt":
        raise SystemExit("Benchmark must run inside conda env 'tbt'.")

    runners = {
        "magnetic": run_magnetic,
        "fluid": run_fluid,
        "heat": run_heat,
    }
    results = runners[args.sim](args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
