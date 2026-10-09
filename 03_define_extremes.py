#!/usr/bin/env python3
"""03_define_extremes.py — Formal YREP-day table and wet-day sensitivity grid.

Consumes script 01's cache (precip_daily_region_1960_2024.nc), which holds daily
Pbar, daily coverage F at Q90/Q95/Q99, and per-grid thresholds.

Primary definition: local wet-day Q95 + area coverage F95 >= 10% + area-mean
wet-day P90. Sensitivity grid: coverage {5,10,15%} × intensity {Q90,Q95,Q99}
× area-mean {P90,P95} = 18 combos.

Outputs:
  YREP_day_table_<y0>_<y1>.csv       — formal day table
  sensitivity_grid_<y0>_<y1>.csv     — Class-1 fraction per threshold combo
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import xarray as xr

import config

DEFAULT_CACHE = os.path.join(config.OUT, "precip_daily_region_1960_2024.nc")
OUT_DIR = config.OUT

WET_DAY_MM = 1.0
# primary (locked)
PRIMARY_Q, PRIMARY_COV, PRIMARY_P = 0.95, 0.10, 0.90
# sensitivity grid
SENS_Q = (0.90, 0.95, 0.99)
SENS_COV = (0.05, 0.10, 0.15)
SENS_P = (0.90, 0.95)


def run(cache=DEFAULT_CACHE, outdir=OUT_DIR):
    ds = xr.open_dataset(cache)
    y0 = int(pd.to_datetime(ds.time.values[0]).year)
    y1 = int(pd.to_datetime(ds.time.values[-1]).year)
    Pbar = ds["Pbar"].values  # (time,)
    F = {q: ds[f"F{int(q*100):02d}"].values for q in SENS_Q}
    dates = pd.to_datetime(ds.time.values)
    n = Pbar.size

    # area-mean wet-day percentile thresholds
    wet = Pbar[Pbar >= WET_DAY_MM]
    Pp = {p: float(np.quantile(wet, p)) for p in SENS_P}
    print(f"[load] {cache}\n       years {y0}-{y1}, n_days={n}, wet region-days={wet.size}")
    for p, v in Pp.items():
        print(f"       area-mean wet-day P{int(p*100):02d} = {v:.3f} mm/d")

    # ---- primary YREP table (locked definition) ----
    Fp = F[PRIMARY_Q]
    yrep = ((Fp >= PRIMARY_COV) & (Pbar >= Pp[PRIMARY_P])).astype(int)
    n_yrep = int(yrep.sum())
    print(f"[primary] Q95 & F>=10% & Pbar>=P90 -> YREP = {n_yrep}/{n} "
          f"({100*n_yrep/n:.2f}%)")

    df = pd.DataFrame(dict(
        date=dates, year=dates.year, month=dates.month,
        area_mean_precip_mm_d=np.round(Pbar, 3),
        extreme_area_fraction=np.round(Fp, 4),
        YREP_label=yrep,
    ))
    csv = os.path.join(outdir, f"YREP_day_table_{y0}_{y1}.csv")
    df.to_csv(csv, index=False)
    print(f"[save] {csv}")

    # ---- sensitivity grid (Class-1 fraction for every threshold combo) ----
    rows = []
    for q, c, p in itertools.product(SENS_Q, SENS_COV, SENS_P):
        lab = (F[q] >= c) & (Pbar >= Pp[p])
        rows.append(dict(intensity_q=int(q * 100), coverage_pct=int(c * 100),
                         areamean_p=int(p * 100),
                         n_class1=int(lab.sum()), class1_pct=round(100 * lab.mean(), 3)))
    sens = pd.DataFrame(rows)
    sens_csv = os.path.join(outdir, f"sensitivity_grid_{y0}_{y1}.csv")
    sens.to_csv(sens_csv, index=False)
    print(f"[save] {sens_csv}")
    print("[sensitivity] Class-1 day fraction (%) by threshold combo "
          "(intensity Q x coverage% | area-mean P):")
    piv = sens.pivot(index=["intensity_q", "coverage_pct"], columns="areamean_p",
                     values="class1_pct")
    print(piv.to_string())
    print(f"\n[sensitivity] spread: min={sens.class1_pct.min():.2f}%  "
          f"median={sens.class1_pct.median():.2f}%  max={sens.class1_pct.max():.2f}%  "
          f"(primary Q95/10%/P90 = {100*n_yrep/n:.2f}%)")
    ds.close()
    return csv, sens_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--outdir", default=OUT_DIR)
    a = ap.parse_args()
    run(a.cache, a.outdir)
