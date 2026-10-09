#!/usr/bin/env python3
"""06_composite_trends.py — Yearly trends and composite analysis.

Yearly inter-annual trends (Theil-Sen slope, TFPW Mann-Kendall p, bootstrap 95% CI)
for YREP freq, YR-EPCMP freq/mean-P/non-mean-P/VIMFC/IVT/ext-area/event-total.
Composite analysis for 4 categories (Early/Late × YR-EPCMP/non) of Z500, SLP,
IVT_x/y/mag, VIMFC, PW, omega500, q850, u850, v850, plus 3 differences.

Early = 1960-1991, Late = 1992-2024.

Inputs: diagnostics_native_1960_2024.nc, prediction_probability_full.csv,
        YREP_day_table_1960_2024.csv, precip_daily_region_1960_2024.nc
Outputs: composite_fields_1960_2024.nc, trend_statistics.csv, yearly_metrics_1960_2024.csv
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import config

OUT = config.OUT
TAG = "1960_2024"
EARLY_END = 1991  # Early 1960-1991, Late 1992-2024
Y0, Y1 = 1960, 2024
THRESH = 0.5  # P(YR-EPCMP) > 0.5
RNG = np.random.default_rng(20260712)


def tfpw_mk(y):
    """Trend-free pre-whitened Mann-Kendall: returns (theil-sen slope, MK p, lag-1 resid)."""
    y = np.asarray(y, float); t = np.arange(len(y))
    ts = stats.theilslopes(y, t); slope, intercept = ts.slope, ts.intercept
    det = y - (intercept + slope * t)
    r1 = np.corrcoef(det[:-1], det[1:])[0, 1] if det.var() > 0 else 0.0
    if not np.isfinite(r1):
        r1 = 0.0
    y_pw = y if r1 <= 0 else np.concatenate([[y[0]], y[1:] - r1 * y[:-1]])
    n = len(y_pw); S = 0
    for i in range(n - 1):
        S += np.sum(np.sign(y_pw[i + 1:] - y_pw[i]))
    varS = n * (n - 1) * (2 * n + 5) / 18.0
    Z = (S - 1) / np.sqrt(varS) if S > 0 else (S + 1) / np.sqrt(varS) if S < 0 else 0.0
    return float(slope), float(2 * (1 - stats.norm.cdf(abs(Z)))), float(r1)


def ts_ci_bs(y, n_boot=2000):
    """Theil-Sen slope with bootstrap 95% CI (block: residual resampling)."""
    y = np.asarray(y, float); t = np.arange(len(y))
    ts = stats.theilslopes(y, t); slope, intercept = ts.slope, ts.intercept
    resid = y - (intercept + slope * t); boot = []
    for _ in range(n_boot):
        yb = intercept + slope * t + RNG.choice(resid, size=len(resid), replace=True)
        try:
            boot.append(stats.theilslopes(yb, t).slope)
        except Exception:
            pass
    if len(boot) < 100:
        return slope, np.nan, np.nan
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(slope), float(lo), float(hi)


def w_mean(da, lat_name="lat", lon_name="lon"):
    w = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(w).mean(dim=(lat_name, lon_name))


def main():
    diag = xr.open_dataset(os.path.join(OUT, f"diagnostics_native_{TAG}.nc"))
    prob = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    ytbl = pd.read_csv(os.path.join(OUT, f"YREP_day_table_{TAG}.csv"), parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["yrecmp"] = (prob.y_prob >= THRESH).astype(int)
    prob["year"] = prob.date.dt.year
    df = ytbl[["date", "year", "area_mean_precip_mm_d", "extreme_area_fraction", "YREP_label"]].merge(
        prob[["date", "y_prob", "yrecmp"]], on="date", how="inner")
    diag_dates = pd.to_datetime(diag.time.values).normalize()
    assert len(df) == diag_dates.size == (Y1 - Y0 + 1) * 365 + 17, (len(df), diag_dates.size)

    # daily area-mean VIMFC (mm/d) + IVT over the CNN region (Tibet-fringe raw
    # inflation noted; standardized composites in figures will mask Tibet).
    region = diag.sel(lat=slice(70, -10), lon=slice(60, 180))
    df["VIMFC_am_mmday"] = (w_mean(region["VIMFC"]).values * 86400.0)
    df["IVT_am"] = w_mean(region["IVT_mag"]).values

    # ---------- yearly metrics + trends ----------
    rows = []
    metrics = [
        ("YREP_freq",         df[df.YREP_label == 1], "year", "count"),
        ("YREPCMP_freq",      df[df.yrecmp == 1],     "year", "count"),
        ("YREPCMP_mean_P",    df[df.yrecmp == 1],     "area_mean_precip_mm_d", "mean"),
        ("nonYREPCMP_mean_P", df[df.yrecmp == 0],     "area_mean_precip_mm_d", "mean"),
        ("YREPCMP_VIMFC",     df[df.yrecmp == 1],     "VIMFC_am_mmday",        "mean"),
        ("YREPCMP_IVT",       df[df.yrecmp == 1],     "IVT_am",                "mean"),
        ("YREPCMP_extarea",   df[df.yrecmp == 1],     "extreme_area_fraction", "mean"),
        ("YREPCMP_eventP",    df[df.yrecmp == 1],     "area_mean_precip_mm_d", "mean"),
    ]
    yr_table = {}
    for name, sub, col, agg in metrics:
        if col == "year":
            s = sub.groupby("year").size()
        else:
            s = sub.groupby("year")[col].mean() if agg == "mean" else sub.groupby("year")[col].sum()
        s = s.reindex(range(Y0, Y1 + 1)).fillna(0)
        yr_table[name] = s.values
        slope, pmk, r1 = tfpw_mk(s.values)
        slope2, lo, hi = ts_ci_bs(s.values)
        rows.append(dict(metric=name, theilsen_per_yr=round(slope, 5), mk_p=round(pmk, 4),
                         lag1_resid=round(r1, 3), boot95=[round(lo, 5), round(hi, 5)],
                         early_mean=round(float(np.mean(s.loc[Y0:EARLY_END])), 4),
                         late_mean=round(float(np.mean(s.loc[EARLY_END + 1:Y1])), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "trend_statistics.csv"), index=False)
    pd.DataFrame(yr_table, index=range(Y0, Y1 + 1)).to_csv(
        os.path.join(OUT, f"yearly_metrics_{TAG}.csv"))
    print(f"[save] trend_statistics.csv + yearly_metrics_{TAG}.csv")
    for r in rows:
        print(f"  {r['metric']:20s} slope={r['theilsen_per_yr']:+.5f}/yr  MK p={r['mk_p']:.3f}  "
              f"early={r['early_mean']:.3f} late={r['late_mean']:.3f}")

    # ---------- composites ----------
    early = df.year <= EARLY_END
    late = df.year >= EARLY_END + 1
    cats = {
        "early_YREPCMP": early & (df.yrecmp == 1),
        "late_YREPCMP":  late & (df.yrecmp == 1),
        "early_non":     early & (df.yrecmp == 0),
        "late_non":      late & (df.yrecmp == 0),
    }
    print("\n[composite] category sizes:")
    for k, m in cats.items():
        print(f"  {k:16s} n={int(m.sum())}")
    fields = ["Z500", "SLP", "IVT_x", "IVT_y", "IVT_mag", "VIMFC", "PW",
              "omega500", "q850", "u850", "v850"]
    comp = {}
    for cat, mask in cats.items():
        idx = np.where(mask.values)[0]
        for f in fields:
            comp[f"{cat}__{f}"] = diag[f].isel(time=idx).mean("time")
    diffs = {
        "LateMinusEarly_YREPCMP": ("late_YREPCMP", "early_YREPCMP"),
        "LateMinusEarly_non":     ("late_non", "early_non"),
        "YREPCMPMinusNon_all":    (None, None),
    }
    comp_ds = dict(comp)
    for diffname, (a, b) in diffs.items():
        for f in fields:
            if a is None:
                ia = np.where(df.yrecmp.values == 1)[0]
                ib = np.where(df.yrecmp.values == 0)[0]
                comp_ds[f"{diffname}__{f}"] = diag[f].isel(time=ia).mean("time") - \
                                              diag[f].isel(time=ib).mean("time")
            else:
                comp_ds[f"{diffname}__{f}"] = comp[f"{a}__{f}"] - comp[f"{b}__{f}"]
    xr.Dataset(comp_ds).assign_attrs(
        early_period=f"{Y0}-{EARLY_END}", late_period=f"{EARLY_END+1}-{Y1}",
        YREPCMP_threshold=f"P(combined) >= {THRESH}",
        note="Composite means + differences on native NCEP/NCAR R1 2.5° grid; "
             "Tibet-fringe raw moisture inflation present — mask in figure display").to_netcdf(
        os.path.join(OUT, f"composite_fields_{TAG}.nc"))
    print(f"[save] composite_fields_{TAG}.nc ({len(comp_ds)} fields)")


if __name__ == "__main__":
    main()
