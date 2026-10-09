#!/usr/bin/env python3
"""10_reach_decomposition.py — Middle vs lower reach decomposition.

Applies the frequency/intensity decomposition separately to each reach, to
test whether the robust thermodynamic finding holds per-reach and whether
mechanisms differ (middle = Loess-Plateau orographic; lower = North-China-Plain
moisture-transport).

Design:
  - Partition = basin-wide YR-EPCMP CNN pattern labels (P>=0.5, combined CNN).
  - X (the quantity decomposed) becomes reach-specific:
      ΔP̄_reach   : X = reach area-weighted mean precipitation
      ΔPr(E)_reach: X = reach regional-extreme indicator (F_reach >= 10%)
  - Same early/late split (1960-1991 vs 1992-2024) and annual+JAS cuts.
  - year-block bootstrap CIs on each reach's intensity term.

Reuses: 01_prepare_precip.{load_chmpre, build_reach_masks}; decomposition.decompose.

Outputs:
  reach_Pbar_1960_2024.csv
  reach_decomposition_1960_2024.csv
  reach_boot_1960_2024.csv
"""
import argparse, os, importlib.util
import numpy as np
import pandas as pd
import xarray as xr

import config

HERE = os.path.dirname(__file__)
OUT = config.OUT
EARLY_MAIN, LATE_MAIN = 1991, 1992


def _import01():
    spec = importlib.util.spec_from_file_location("prep01", os.path.join(HERE, "01_prepare_precip.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _import_decomp():
    spec = importlib.util.spec_from_file_location("decomp", os.path.join(HERE, "decomposition.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def compute_reach_series(year0, year1):
    """Load CHM_PRE once, build reach masks, compute reach P̄ + reach F (coverage vs basin Q95)."""
    prep = _import01()
    da, _ = prep.load_chmpre(year0, year1)  # (time,lat,lon), ~5GB
    reaches = prep.build_reach_masks(da)  # {middle,lower}: bool (lat,lon)
    # cached basin per-grid Q95 on the SAME grid
    cache = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    Q95 = cache["Q95"].values  # (lat,lon)
    basin_mask = cache["mask"].values.astype(bool)
    cache.close()
    # alignment sanity: da grid == cache grid
    assert da.lat.size == Q95.shape[0] and da.lon.size == Q95.shape[1]

    wlat = np.cos(np.deg2rad(da.lat.values))
    P = da.values.astype(np.float32)  # (time,lat,lon)
    P = np.where(np.isfinite(P), P, 0.0)  # NaN outside data domain -> 0
    # (reach cells are all finite; NaN are ocean/non-China only, where W=0 anyway,
    #  so zeroing them avoids NaN×0 contaminating the sum)
    dates = pd.to_datetime(da.time.values).normalize()
    out = {"date": dates}
    union = reaches["middle"] | reaches["lower"]
    print(f"[reach] middle={reaches['middle'].sum()} lower={reaches['lower'].sum()} "
          f"union={union.sum()} basin_mask={basin_mask.sum()} (overlap={int((reaches['middle']&reaches['lower']).sum())})")
    for reach, m in reaches.items():
        W = (wlat[:, None] * m).astype(np.float64)
        Wsum = W.sum()
        # reach area-weighted mean P̄
        Pbar_r = (P * W).sum(axis=(1, 2)) / Wsum
        # reach coverage F = area-weighted fraction of reach cells with P >= basin Q95
        exceed = (P >= Q95[None, :, :]) & m[None, :, :]
        F_r = (exceed * W[None, :, :]).sum(axis=(1, 2)) / Wsum
        out[f"Pbar_{reach}"] = Pbar_r
        out[f"F_{reach}"] = F_r
        out[f"reach_extreme_{reach}"] = (F_r >= 0.10).astype(float)
        print(f"[reach] {reach}: mean P̄={np.nanmean(Pbar_r):.2f} mm/d  "
              f"mean F(Q95)={np.nanmean(F_r)*100:.2f}%  extreme-days={int((F_r>=0.10).sum())}")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, f"reach_Pbar_{year0}_{year1}.csv"), index=False)
    print(f"[save] out/reach_Pbar_{year0}_{year1}.csv  ({len(df)} days)")
    return df


def load_basin_labels():
    """Basin CNN pattern labels (combined, P>=0.5) + dates, aligned to the cache time axis."""
    cache = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    dates = pd.to_datetime(cache.time.values).normalize()
    cache.close()
    prob = pd.read_csv(os.path.join(OUT, "prediction_probability_full.csv"), parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["date"] = prob["date"].dt.normalize()
    lab = pd.DataFrame({"date": dates}).reset_index().rename(columns={"index": "i"})
    lab = lab.merge(prob, on="date", how="inner")
    assert len(lab) == len(dates)
    labels = (lab["y_prob"].values >= 0.5).astype(int)
    return dates, labels


def main(year0, year1, smoke=False):
    decomp = _import_decomp()
    df = compute_reach_series(year0, year1) if not smoke else compute_reach_series(year0, year1)
    dates, labels = load_basin_labels()
    # align df dates to labels dates (same axis)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    assert len(df) == len(labels), (len(df), len(labels))
    yr = dates.year.values; mo = dates.month.values
    jas = np.isin(mo, [7, 8, 9])
    early = yr <= EARLY_MAIN; late = yr >= LATE_MAIN

    rows = []
    print("\n=== REACH DECOMPOSITION (basin CNN labels as partition) ===")
    for reach in ("middle", "lower"):
        for xname, X, sel in [(f"dPbar_{reach}", df[f"Pbar_{reach}"].values, np.ones(len(yr), bool)),
                              (f"dPbar_{reach}_JAS", df[f"Pbar_{reach}"].values, jas),
                              (f"dPrE_{reach}", df[f"reach_extreme_{reach}"].values, np.ones(len(yr), bool)),
                              (f"dPrE_{reach}_JAS", df[f"reach_extreme_{reach}"].values, jas)]:
            r = decomp.decompose(X, labels, early & sel, late & sel)
            tot = r["term_freq"] + r["term_intensity"] + r["term_interaction"]
            for term, val in [("freq", r["term_freq"]), ("inten", r["term_intensity"]),
                              ("inter", r["term_interaction"]),
                              ("thermo_share_pct", 100 * r["term_intensity"] / tot if abs(tot) > 1e-12 else np.nan),
                              ("delta", r["delta"])]:
                rows.append(dict(reach=reach, case=xname, term=term, value=val))
            sh = 100 * r["term_intensity"] / tot if abs(tot) > 1e-12 else float("nan")
            print(f"  {xname:18s}: Δ={r['delta']:+.4f}  freq {r['term_freq']:+.4f} "
                  f"inten {r['term_intensity']:+.4f}  (thermo {sh:+.0f}%)")
    rd = pd.DataFrame(rows)
    rd.to_csv(os.path.join(OUT, f"reach_decomposition_{year0}_{year1}.csv"), index=False)
    print(f"[save] out/reach_decomposition_{year0}_{year1}.csv")

    # ---- year-block bootstrap on the reach INTENSITY terms (honest bound) ----
    rng = np.random.RandomState(20260712)
    yrs_e = np.unique(yr[early]); yrs_l = np.unique(yr[late])
    boot_rows = []
    NBOOT = 2000
    for reach in ("middle", "lower"):
        for xname, X, sel in [(f"dPbar_{reach}", df[f"Pbar_{reach}"].values, np.ones(len(yr), bool)),
                              (f"dPbar_{reach}_JAS", df[f"Pbar_{reach}"].values, jas),
                              (f"dPrE_{reach}", df[f"reach_extreme_{reach}"].values, np.ones(len(yr), bool)),
                              (f"dPrE_{reach}_JAS", df[f"reach_extreme_{reach}"].values, jas)]:
            inten_b = []
            for _ in range(NBOOT):
                ye = rng.choice(yrs_e, size=len(yrs_e), replace=True)
                yl = rng.choice(yrs_l, size=len(yrs_l), replace=True)
                ie = np.isin(yr, ye) & sel; il = np.isin(yr, yl) & sel
                r = decomp.decompose(X, labels, ie, il)
                inten_b.append(r["term_intensity"])
            c = [float(np.percentile(inten_b, 2.5)), float(np.percentile(inten_b, 97.5))]
            boot_rows.append(dict(reach=reach, case=xname, term="inten",
                                  ci_low=c[0], ci_high=c[1], ci_excludes_0=(c[0] > 0 or c[1] < 0)))
            print(f"[boot] {xname}: intensity year-block CI [{c[0]:+.4f},{c[1]:+.4f}]  "
                  f"{'ROBUST' if (c[0]>0 or c[1]<0) else 'spans-0'}")
    pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, f"reach_boot_{year0}_{year1}.csv"), index=False)
    print(f"[save] out/reach_boot_{year0}_{year1}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[1960, 2024])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.years[0], a.years[1], smoke=a.smoke)
