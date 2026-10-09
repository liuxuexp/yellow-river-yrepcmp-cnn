#!/usr/bin/env python3
"""decomposition.py — Frequency/intensity decomposition of precipitation change.

Splits the observed change in regional mean precipitation (ΔP̄, early -> late)
AND the change in extreme-event frequency (ΔPr(E)) into a DYNAMIC
(pattern-frequency) part, a THERMODYNAMIC (pattern-conditioned) part, and
an interaction.

Exact algebraic identity (binary pattern partition of days):
  Let f_p = fraction of pattern-positive days,
      μ_p = mean of X on pattern-positive days,  μ_n = mean on pattern-negative days.
  Period mean ⟨X⟩ = f_p·μ_p + (1−f_p)·μ_n  (mixture identity).

  Δ⟨X⟩ = Δf_p·(μ_p0 − μ_n0)              … (1) frequency / DYNAMIC
       + f_p0·Δμ_p + (1−f_p0)·Δμ_n       … (2) conditional / THERMODYNAMIC
       + Δf_p·(Δμ_p − Δμ_n)              … (3) interaction

Two headline decompositions: ΔP̄ (X = daily area-mean precip) and
ΔPr(E) (X = YREP indicator). Each reported annual and JAS-only.
Main split: 1960-1991 vs 1992-2024 (32/33 yr balanced).

Self-verifies (1)+(2)+(3) == Δ⟨X⟩ to machine precision.
"""
import argparse, os
import numpy as np
import pandas as pd
import xarray as xr

import config

OUT = config.OUT
CACHE = os.path.join(OUT, "precip_daily_region_1960_2024.nc")
PROB_CSV = os.path.join(OUT, "prediction_probability_full.csv")
THRESH = 0.5
EARLY_MAIN, LATE_MAIN = 1991, 1992  # 1960-1991 vs 1992-2024 (32/33 yr)


def _moments(X, labels):
    """f_p, mu_p, mu_n for one period (X, labels both length-N). mu_* = mean of X in class."""
    f_p = float(np.mean(labels))
    if 0 < f_p < 1:
        mu_p = float(np.mean(X[labels == 1]))
        mu_n = float(np.mean(X[labels == 0]))
    elif f_p == 1:
        mu_p, mu_n = float(np.mean(X)), float("nan")
    elif f_p == 0:
        mu_p, mu_n = float("nan"), float(np.mean(X))
    else:
        mu_p, mu_n = float("nan"), float("nan")
    mean = f_p * (0.0 if np.isnan(mu_p) else mu_p) + (1 - f_p) * (0.0 if np.isnan(mu_n) else mu_n)
    return dict(f_p=f_p, mu_p=mu_p, mu_n=mu_n, mean=mean)


def decompose(X, labels, early_mask, late_mask):
    """4-term decomposition of Δ⟨X⟩ (early -> late). X general (P̄ or YREP indicator)."""
    e = _moments(X[early_mask], labels[early_mask])
    l = _moments(X[late_mask], labels[late_mask])
    df = l["f_p"] - e["f_p"]
    dmu_p = l["mu_p"] - e["mu_p"]
    dmu_n = l["mu_n"] - e["mu_n"]
    freq = df * (e["mu_p"] - e["mu_n"])                # (1) dynamic
    inten = e["f_p"] * dmu_p + (1 - e["f_p"]) * dmu_n  # (2) thermodynamic
    inter = df * (dmu_p - dmu_n)                        # (3) interaction
    dX = l["mean"] - e["mean"]
    return dict(early=e, late=l, delta_f_p=df, delta_mu_p=dmu_p, delta_mu_n=dmu_n,
                term_freq=freq, term_intensity=inten, term_interaction=inter,
                delta=dX, residual=dX - (freq + inten + inter))


def fmt(name, r, xunit):
    tot = r["term_freq"] + r["term_intensity"] + r["term_interaction"]
    sh = lambda t: f"{100*t/tot:+.1f}%" if abs(tot) > 1e-12 else "n/a"
    print(f"\n[{name}]  early f_p={r['early']['f_p']:.4f} mu_p={r['early']['mu_p']:.3f} "
          f"mu_n={r['early']['mu_n']:.3f}  | late f_p={r['late']['f_p']:.4f} "
          f"mu_p={r['late']['mu_p']:.3f} mu_n={r['late']['mu_n']:.3f}")
    print(f"   Δ⟨X⟩={r['delta']:+.4f} {xunit}  =  freq {r['term_freq']:+.4f} [dyn] "
          f"+ inten {r['term_intensity']:+.4f} [thermo] + inter {r['term_interaction']:+.4f}")
    print(f"   shares (of summed terms): freq {sh(r['term_freq'])}  "
          f"thermo {sh(r['term_intensity'])}  inter {sh(r['term_interaction'])}  "
          f"| residual {r['residual']:+.2e}")
    return r


def load_labels():
    """P̄, YREP indicator, dates, and CNN Combined YR-EPCMP pattern labels (P>=0.5)."""
    ds = xr.open_dataset(CACHE)
    pbar = ds["Pbar"].values.astype(float)
    yrep = ds["YREP"].values.astype(float)  # 0/1 extreme-event indicator
    dates = pd.to_datetime(ds.time.values)
    ds.close()
    prob = pd.read_csv(PROB_CSV, parse_dates=["date"])
    prob = prob[prob.model == "combined"][["date", "y_prob"]].copy()
    prob["date"] = prob["date"].dt.normalize()
    lab = pd.DataFrame({"date": pd.to_datetime(dates).normalize()}).reset_index().rename(columns={"index": "i"})
    lab = lab.merge(prob, on="date", how="inner")
    assert len(lab) == len(dates), (len(lab), len(dates))
    labels = (lab["y_prob"].values >= THRESH).astype(int)
    return pbar, yrep, dates, labels


def selftest():
    rng = np.random.RandomState(0)
    max_err = 0.0
    for _ in range(2000):
        n = 4000
        X = rng.randn(n) * 5 + 8
        lab = (rng.rand(n) < rng.uniform(0.05, 0.5)).astype(int)
        early = rng.rand(n) < 0.5; late = ~early
        max_err = max(max_err, abs(decompose(X, lab, early, late)["residual"]))
    print(f"[selftest] 2000 trials; max |residual| = {max_err:.2e}  (expect ~1e-10)")
    assert max_err < 1e-8, "decomposition identity failed to close!"
    print("[selftest] PASS — (1)+(2)+(3) == Δ⟨X⟩ exactly for both ΔP̄ and ΔPr(E).")


def real():
    """HEADLINE: ΔP̄ and ΔPr(E), annual + JAS, main split 1960-1991 vs 1992-2024."""
    pbar, yrep, dates, labels = load_labels()
    yr = dates.year.values
    mo = dates.month.values
    early = yr <= EARLY_MAIN
    late = yr >= LATE_MAIN
    jas = np.isin(mo, [7, 8, 9])
    rows = []
    for name, X, xunit, sel in [
        ("ΔP̄   annual",  pbar, "mm/d",  np.ones(len(yr), bool)),
        ("ΔP̄   JAS",     pbar, "mm/d",  jas),
        ("ΔPr(E) annual", yrep, "prob",  np.ones(len(yr), bool)),
        ("ΔPr(E) JAS",    yrep, "prob",  jas),
    ]:
        # labels = PATTERN partition (YR-EPCMP); X = quantity being decomposed
        r = decompose(X, labels, early & sel, late & sel)
        fmt(name, r, xunit)
        rows.append(dict(decomposition=name, delta_X=r["delta"], term_freq_dynamic=r["term_freq"],
                         term_intensity_thermo=r["term_intensity"], term_interaction=r["term_interaction"],
                         residual=r["residual"], f_p_early=r["early"]["f_p"], f_p_late=r["late"]["f_p"],
                         mu_p_early=r["early"]["mu_p"], mu_p_late=r["late"]["mu_p"],
                         mu_n_early=r["early"]["mu_n"], mu_n_late=r["late"]["mu_n"]))
    out = os.path.join(OUT, "decomposition_headline_1960_2024.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[save] {out}")


def sensitivity():
    """Breakpoint {1980,1990,1991,2000} + 1979-2024 window, annual ΔP̄."""
    pbar, yrep, dates, labels = load_labels()
    yr = dates.year.values
    rows = []
    for B in (1980, 1990, 1991, 2000):
        e = yr <= B; l = yr > B
        r = decompose(pbar, labels, e, l)
        rows.append(dict(window="1960-2024", breakpoint=B, delta_Pbar=r["delta"],
                         term_freq=r["term_freq"], term_intensity=r["term_intensity"],
                         term_interaction=r["term_interaction"]))
        print(f"[bkpt {B}] ΔP̄={r['delta']:+.4f}  freq {r['term_freq']:+.4f}  "
              f"inten {r['term_intensity']:+.4f}  inter {r['term_interaction']:+.4f}")
    # 1979-2024 window
    sel79 = yr >= 1979
    for B in (2000,):
        e = (yr <= B) & sel79; l = (yr > B) & sel79
        r = decompose(pbar[sel79], labels[sel79], (yr[sel79] <= B), (yr[sel79] > B))
        rows.append(dict(window="1979-2024", breakpoint=B, delta_Pbar=r["delta"],
                         term_freq=r["term_freq"], term_intensity=r["term_intensity"],
                         term_interaction=r["term_interaction"]))
        print(f"[1979-2024 bkpt {B}] ΔP̄={r['delta']:+.4f}  freq {r['term_freq']:+.4f}  "
              f"inten {r['term_intensity']:+.4f}  inter {r['term_interaction']:+.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "decomposition_sensitivity_1960_2024.csv"), index=False)
    print(f"[save] out/decomposition_sensitivity_1960_2024.csv")


def boot(n_boot=1000, seed=20260712):
    """Bootstrap CIs (day-resampling within early/late) for annual ΔP̄ terms + thermo share."""
    rng = np.random.RandomState(seed)
    pbar, yrep, dates, labels = load_labels()
    yr = dates.year.values
    ie = np.where(yr <= EARLY_MAIN)[0]; il = np.where(yr >= LATE_MAIN)[0]
    freq_b, inten_b, inter_b, share_b = [], [], [], []
    for _ in range(n_boot):
        e = rng.choice(ie, size=len(ie), replace=True)
        l = rng.choice(il, size=len(il), replace=True)
        idx_e = np.zeros(len(pbar), bool); idx_e[e] = True
        idx_l = np.zeros(len(pbar), bool); idx_l[l] = True
        r = decompose(pbar, labels, idx_e, idx_l)
        freq_b.append(r["term_freq"]); inten_b.append(r["term_intensity"]); inter_b.append(r["term_interaction"])
        tot = r["term_freq"] + r["term_intensity"] + r["term_interaction"]
        share_b.append(100 * r["term_intensity"] / tot if abs(tot) > 1e-12 else np.nan)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    rows = [
        dict(term="frequency_dynamic", ci_low=ci(freq_b)[0], ci_high=ci(freq_b)[1]),
        dict(term="intensity_thermodynamic", ci_low=ci(inten_b)[0], ci_high=ci(inten_b)[1]),
        dict(term="interaction", ci_low=ci(inter_b)[0], ci_high=ci(inter_b)[1]),
        dict(term="thermo_share_pct", ci_low=ci(share_b)[0], ci_high=ci(share_b)[1]),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "decomposition_boot_1960_2024.csv"), index=False)
    print(f"[boot] {n_boot} resamples; thermo share 95% CI = [{ci(share_b)[0]:.1f}%, {ci(share_b)[1]:.1f}%]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true", help="headline ΔP̄+ΔPr(E), annual+JAS")
    ap.add_argument("--sensitivity", action="store_true", help="breakpoint + 1979-2024")
    ap.add_argument("--boot", action="store_true", help="bootstrap CIs")
    a = ap.parse_args()
    if a.selftest or not (a.real or a.sensitivity or a.boot):
        selftest()
    if a.real:
        real()
    if a.sensitivity:
        sensitivity()
    if a.boot:
        boot()
