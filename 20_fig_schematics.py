#!/usr/bin/env python3
"""
20_fig_schematics.py — Export standalone schematic deliverables from data/.

Copies the two bundled schematic images (verbatim) into the figures directory:
  fig2a_flowchart.png    ->  fig2a_yrep_definition.png
  fig3a_architecture.png ->  fig3_arch_schematic.png
"""
import json
from pathlib import Path

from PIL import Image

import config

SRC = config.SCHEMATICS
OUTDIR = Path(config.FIGDIR)
DPI = 600

SOURCES = [  # (bundled image, deliverable)
    ("fig2a_flowchart.png", "fig2a_yrep_definition.png"),
    ("fig3a_architecture.png", "fig3_arch_schematic.png"),
]


def export(png_path, deliverable):
    """Copy the bundled image into the figures directory with the audit JSON."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    target = OUTDIR / deliverable
    with Image.open(png_path) as im:
        im.save(target, dpi=(DPI, DPI))
        size = im.size
    report = {"figure": Path(deliverable).stem, "inches": [round(size[0] / DPI, 2),
              round(size[1] / DPI, 2)], "dpi": DPI, "text_sizes_pt": [],
              "title_plot_collisions": [],
              "note": "user-supplied artwork, rendered verbatim; "
                      "panel titles live in fig2/fig3 scripts"}
    (OUTDIR / (Path(deliverable).stem + ".json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[optimized] {target}", flush=True)


if __name__ == "__main__":
    for asset, deliverable in SOURCES:
        export(SRC / asset, deliverable)
