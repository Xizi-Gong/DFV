import argparse
import importlib.util
import json
import math
import os
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


CASES = [
    {"name": "clear_20mps", "speed_mps": 20.0, "karman": False},
    {"name": "clear_45mps", "speed_mps": 45.0, "karman": False},
    {"name": "clear_50mps", "speed_mps": 50.0, "karman": False},
    {"name": "clear_52mps", "speed_mps": 52.0, "karman": False},
    {"name": "clear_58mps", "speed_mps": 58.0, "karman": False},
    {"name": "clear_55mps", "speed_mps": 55.0, "karman": False},
    {"name": "clear_64mps", "speed_mps": 64.0, "karman": False},
    {"name": "clear_64mps_no_curl", "speed_mps": 64.0, "karman": False, "curl_strength": 0.0},
    {"name": "clear_76mps", "speed_mps": 76.0, "karman": False},
    {"name": "karman_45mps", "speed_mps": 45.0, "karman": True},
    {"name": "karman_50mps", "speed_mps": 50.0, "karman": True},
    {"name": "karman_52mps", "speed_mps": 52.0, "karman": True},
    {"name": "karman_64mps", "speed_mps": 64.0, "karman": True},
    {"name": "clear_100mps", "speed_mps": 100.0, "karman": False},
    {"name": "karman_20mps", "speed_mps": 20.0, "karman": True},
    {"name": "karman_58mps", "speed_mps": 58.0, "karman": True},
    {"name": "karman_76mps", "speed_mps": 76.0, "karman": True},
    {"name": "karman_toggle_58mps", "speed_mps": 58.0, "karman": True, "toggle": True},
    {"name": "karman_100mps", "speed_mps": 100.0, "karman": True},
]


def import_fluid():
    fluid_dir = ROOT / "Fluid"
    sys.path.insert(0, str(fluid_dir))
    spec = importlib.util.spec_from_file_location(
        "fluid_field_simulation_high_speed_bench",
        fluid_dir / "fluid_field_simulation.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        util, mem = [float(part.strip()) for part in out.splitlines()[0].split(",")[:2]]
        return util, mem
    except Exception:
        return None, None


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def set_air(m):
    m.fluid_material = "air"
    m.RHO = m.fluid_properties["air"]["rho"]
    m.VISCOSITY = m.fluid_properties["air"]["viscosity"]
    m.PIXEL_SIZE = m.PIXEL_SIZE_MAP["air"]
    m._rho_field[None] = m.RHO
    m._nu_field[None] = m.VISCOSITY
    m._pixel_size_field[None] = m.PIXEL_SIZE


def smoke_luma(m):
    frame = m.color_field.to_numpy().astype(np.float32)
    return 0.2126 * frame[..., 0] + 0.7152 * frame[..., 1] + 0.0722 * frame[..., 2]


def divergence_metrics(m):
    m.divergence(m.velocities_pair.cur)
    div = m.velocity_divs.to_numpy()
    solid = m._solid.to_numpy() > 0.5
    interior = np.zeros_like(solid, dtype=bool)
    interior[3:-3, 3:-3] = True
    fluid = interior & ~solid
    div_abs = np.abs(div[fluid])
    velocity = m.velocities_pair.cur.to_numpy()
    vy_abs = np.abs(velocity[..., 1][fluid])
    pressure = m.pressures_pair.cur.to_numpy()[fluid]
    return {
        "div_abs_max": float(np.max(np.abs(div))),
        "div_abs_mean": float(np.mean(np.abs(div))),
        "div_interior_abs_mean": float(np.mean(div_abs)),
        "div_interior_abs_p95": float(np.percentile(div_abs, 95)),
        "div_interior_abs_max": float(np.max(div_abs)),
        "velocity_y_rms": float(np.sqrt(np.mean(vy_abs * vy_abs))),
        "velocity_y_abs_p95": float(np.percentile(vy_abs, 95)),
        "pressure_rms": float(np.sqrt(np.mean(pressure * pressure))),
        "pressure_abs_max": float(np.max(np.abs(pressure))),
    }


def spatial_roughness(luma):
    dx2 = np.abs(luma[2:, 1:-1] - 2.0 * luma[1:-1, 1:-1] + luma[:-2, 1:-1])
    dy2 = np.abs(luma[1:-1, 2:] - 2.0 * luma[1:-1, 1:-1] + luma[1:-1, :-2])
    rough = dx2 + dy2
    active = luma[1:-1, 1:-1] > 1.0e-4
    values = rough[active] if np.any(active) else rough.reshape(-1)
    return float(np.mean(values)), float(np.percentile(values, 95))


def run_case(m, case, frames, warmup):
    smoke_info = np.zeros((len(m.smoke_positions), 2), dtype=np.int16)
    for i in range(len(m.smoke_positions)):
        smoke_info[i, 0] = int(m.smoke_positions[i] * m.res_y - m.smoke_width[i] / 2)
        smoke_info[i, 1] = int(m.smoke_positions[i] * m.res_y + m.smoke_width[i] / 2)

    set_air(m)
    m.curl_strength = case.get("curl_strength", 2.5)
    m.reset()
    is_toggle = case.get("toggle", False)
    if case["karman"] and not is_toggle:
        m.init_cylinder_obstacle()
    m.ti.sync()

    u_solver = case["speed_mps"] * m.SOLVER_SPEED_PER_MPS
    for _ in range(warmup):
        m.step(u_solver, smoke_info)
    m.ti.sync()

    toggle_pre_step_wake_luma = 0.0
    if is_toggle:
        if hasattr(m, "enable_karman_obstacle"):
            m.enable_karman_obstacle()
        else:
            m.init_cylinder_obstacle()
        m.ti.sync()
        luma = smoke_luma(m)
        cx = int(m.res_x * 0.2)
        radius = int(m.CYLINDER_RADIUS_PX)
        y0 = max(0, int(m.res_y * 0.5) - radius)
        y1 = min(m.res_y, int(m.res_y * 0.5) + radius + 1)
        x0 = min(m.res_x, cx + radius)
        x1 = min(m.res_x, cx + radius * 4)
        wake = luma[x0:x1, y0:y1]
        toggle_pre_step_wake_luma = float(np.mean(wake)) if wake.size else 0.0

    frame_times = []
    sim_times = []
    render_times = []
    temporal_deltas = []
    high_freq_deltas = []
    spatial_roughness_means = []
    spatial_roughness_p95s = []
    active_luma_means = []
    active_luma_p95s = []
    frame_luma_means = []
    smoke_coverage_ratios = []
    prev_luma = None
    nan_detected = False
    diverged = False

    for _ in range(frames):
        t0 = time.perf_counter()
        s0 = time.perf_counter()
        m.step(u_solver, smoke_info)
        m.ti.sync()
        s1 = time.perf_counter()
        luma = smoke_luma(m)
        solid = m._solid.to_numpy() > 0.5
        fluid_pixels = ~solid
        active_pixels = (luma > 1.0e-4) & fluid_pixels
        active_luma = luma[active_pixels]
        frame_luma_means.append(float(np.mean(luma[fluid_pixels])))
        smoke_coverage_ratios.append(float(np.mean(active_pixels[fluid_pixels])))
        if active_luma.size:
            active_luma_means.append(float(np.mean(active_luma)))
            active_luma_p95s.append(float(np.percentile(active_luma, 95)))
        rough_mean, rough_p95 = spatial_roughness(luma)
        spatial_roughness_means.append(rough_mean)
        spatial_roughness_p95s.append(rough_p95)
        r1 = time.perf_counter()

        if prev_luma is not None:
            delta = np.abs(luma - prev_luma)
            temporal_deltas.append(float(np.mean(delta)))
            high_freq_deltas.append(float(np.percentile(delta, 95)))
        prev_luma = luma

        nan_detected = nan_detected or bool(np.isnan(luma).any())
        diverged = diverged or bool(np.max(np.abs(luma)) > 10.0)
        sim_times.append(s1 - s0)
        render_times.append(r1 - s1)
        frame_times.append(r1 - t0)

    projection = divergence_metrics(m)
    proc = psutil.Process(os.getpid())
    gpu_util, vram = query_nvidia()
    total_s = sum(frame_times)
    record = {
        "simulator": "fluid",
        "scene": case["name"],
        "input_condition": "high_speed_karman" if case["karman"] else "high_speed_clear",
        "input_mode": "karman_toggle" if is_toggle else ("karman_cylinder" if case["karman"] else "none"),
        "input_phase": "no_mask",
        "mask_active_frames": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fps_avg": (len(frame_times) / total_s) if total_s > 0 else 0.0,
        "frame_time_p95_ms": percentile(frame_times, 95) * 1000.0 if frame_times else None,
        "sim_step_ms": statistics.mean(sim_times) * 1000.0 if sim_times else None,
        "render_ms": statistics.mean(render_times) * 1000.0 if render_times else None,
        "gpu_util_avg": gpu_util,
        "vram_mb": vram,
        "rss_mb": proc.memory_info().rss / (1024 * 1024),
        "nan_detected": nan_detected,
        "diverged": diverged,
        "crashed": False,
        "notes": "Backend-only high-speed smoke stability benchmark; browser/UI timing is not measured.",
        "speed_mps": case["speed_mps"],
        "karman": case["karman"],
        "curl_strength": m.curl_strength,
        "smoke_temporal_delta_mean": statistics.mean(temporal_deltas) if temporal_deltas else 0.0,
        "smoke_temporal_delta_p95": percentile(temporal_deltas, 95) if temporal_deltas else 0.0,
        "smoke_high_freq_delta_p95": percentile(high_freq_deltas, 95) if high_freq_deltas else 0.0,
        "smoke_spatial_roughness_mean": statistics.mean(spatial_roughness_means) if spatial_roughness_means else 0.0,
        "smoke_spatial_roughness_p95": percentile(spatial_roughness_p95s, 95) if spatial_roughness_p95s else 0.0,
        "smoke_active_luma_mean": statistics.mean(active_luma_means) if active_luma_means else 0.0,
        "smoke_active_luma_p95": percentile(active_luma_p95s, 95) if active_luma_p95s else 0.0,
        "smoke_frame_luma_mean": statistics.mean(frame_luma_means) if frame_luma_means else 0.0,
        "smoke_coverage_ratio": statistics.mean(smoke_coverage_ratios) if smoke_coverage_ratios else 0.0,
        "karman_toggle_pre_step_wake_luma": toggle_pre_step_wake_luma,
        **projection,
    }
    return record


def save_record(record):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"fluid_{record['scene']}_{record['input_condition']}_{ts}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--case", choices=[case["name"] for case in CASES], default=None)
    args = parser.parse_args()

    if os.environ.get("CONDA_DEFAULT_ENV") != "tbt":
        raise SystemExit("Benchmark must run inside conda env 'tbt'.")

    m = import_fluid()
    cases = [case for case in CASES if args.case in (None, case["name"])]
    results = []
    for case in cases:
        try:
            record = run_case(m, case, args.frames, args.warmup)
        except Exception as exc:
            record = {
                "simulator": "fluid",
                "scene": case["name"],
                "input_condition": "high_speed_karman" if case["karman"] else "high_speed_clear",
                "input_mode": "karman_cylinder" if case["karman"] else "none",
                "input_phase": "no_mask",
                "mask_active_frames": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fps_avg": 0.0,
                "frame_time_p95_ms": None,
                "sim_step_ms": None,
                "render_ms": None,
                "gpu_util_avg": None,
                "vram_mb": None,
                "rss_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
                "nan_detected": False,
                "diverged": True,
                "crashed": True,
                "notes": f"Exception: {type(exc).__name__}: {exc}",
                "speed_mps": case["speed_mps"],
                "karman": case["karman"],
            }
        save_record(record)
        results.append(record)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
