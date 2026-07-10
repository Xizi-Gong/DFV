"""Baseline 2 — SHAPE-PALETTE mode backend for Magnetic Field.

Run:
    python Magnetic/magnetic_field_simulation_shapes.py

Open:
    Magnetic/magnetic_ui_shapes.html
"""
from magnetic_field_simulation_baseline_common import (
    rasterize_shapes,
    run_baseline,
)


PORT = 8773


if __name__ == "__main__":
    run_baseline(
        mode="shapes",
        port=PORT,
        mask_action="set_shapes",
        rasterizer=rasterize_shapes,
    )
