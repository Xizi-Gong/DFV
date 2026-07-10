"""Baseline 1 — DRAW mode backend for Magnetic Field.

Run:
    python Magnetic/magnetic_field_simulation_draw.py

Open:
    Magnetic/magnetic_ui_draw.html
"""
from magnetic_field_simulation_baseline_common import (
    rasterize_strokes,
    run_baseline,
)


PORT = 8772


if __name__ == "__main__":
    run_baseline(
        mode="draw",
        port=PORT,
        mask_action="set_strokes",
        rasterizer=rasterize_strokes,
    )
