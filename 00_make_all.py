#!/usr/bin/env python3
"""Run the full analysis pipeline and render all figures."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PIPELINE = [
    "01_prepare_precip.py",
    "02_prepare_reanalysis.py",
    "03_define_extremes.py",
    "03b_sensitivity_allday.py",
    "04_train_cnn.py",
    "04b_cnn_ensemble.py",
    "05_explain_cnn.py",
    "06_composite_trends.py",
    "07_moisture_budget.py",
    "08_spatial_footprint.py",
    "01b_fig1_climatology.py",
    "decomposition.py --real",
    "decomposition.py --bootfull",
    "decomposition.py --sensfull",
    "10_reach_decomposition.py",
    "13_figS4_threshold_sweep.py",
    "15_fig_temporal_evolution.py",
    "19_fig_window_dependence.py",
]

FIGURES = [
    "09a_fig1.py",
    "09a_fig2.py",
    "09a_fig3.py",
    "09a_fig4.py",
    "09a_fig5.py",
    "09a_fig6.py",
    "09a_fig7.py",
    "09a_figS1.py",
    "09a_figS2.py",
    "09a_figS3.py",
    "09a_figS4.py",
    "09a_figS5.py",
    "09a_figS6.py",
    "09a_figS7.py",
    "09a_figS8.py",
    "12_fig8_conceptual.py",
    "16_graphical_abstract.py",
    "20_fig_schematics.py",
]


def run(script):
    print(f"\n{'='*60}\n  {script}\n{'='*60}", flush=True)
    result = subprocess.run(
        [sys.executable, str(HERE / script.split()[0]), *script.split()[1:]],
        cwd=str(HERE),
    )
    if result.returncode != 0:
        print(f"  [FAILED] {script} (exit {result.returncode})", flush=True)
        return False
    return True


def main():
    ok = True
    for s in PIPELINE:
        ok = run(s) and ok
    print("\n--- Rendering figures ---", flush=True)
    for f in FIGURES:
        ok = run(f) and ok
    print("\n  DONE" if ok else "\n  COMPLETED WITH ERRORS", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
