#!/usr/bin/env python3
"""15_fig_temporal_evolution.py — Temporal evolution of decomposition in a rolling 30-yr window.

Shows WHEN the thermodynamic intensification emerged:
  (a) pattern frequency f_p (dynamic) — expect ~flat
  (b) pattern-day mean precip μ_p — expect rising (thermodynamic)
  (c) pattern-conditioned Pr(YREP|pattern) — expect rising (the robust ΔPr(E) channel)
  (d) within-window ΔPr(E) intensity (first-half vs second-half of each window)

Uses the canonical CNN labels (decomposition.load_labels) and a rolling window. PNG only.
"""
import os, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

HERE = os.path.dirname(__file__)
OUT = config.OUT
FIG = config.FIGDIR
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 10})
WIN = 30  # rolling window (years)


def main():
    spec = importlib.util.spec_from_file_location("decomp", os.path.join(HERE, "decomposition.py"))
    dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)
    pbar, yrep, dates, labels = dc.load_labels()
    yr = dates.year.values
    rows = []
    for c in range(yr.min() + WIN // 2, yr.max() - WIN // 2 + 1):  # window-center years
        m = (yr >= c - WIN // 2) & (yr <= c + WIN // 2 - 1)
        lab = labels[m]; Xp = pbar[m]; Xe = yrep[m]
        fp = lab.mean()
        mu_p = Xp[lab == 1].mean() if lab.sum() else np.nan          # pattern-day mean precip
        pext = Xe[lab == 1].mean() if lab.sum() else np.nan          # Pr(YREP|pattern)
        # within-window early/late split -> ΔPr(E) intensity
        yrm = yr[m]; half = (yrm >= c)  # second half = center year onward
        if lab[~half].sum() and lab[half].sum() and (~half).any() and half.any():
            r = dc.decompose(Xe, lab, ~half, half)
            inten = r["term_intensity"]; freq = r["term_freq"]
        else:
            inten = freq = np.nan
        rows.append(dict(center=c, f_p=fp, mu_p=mu_p, p_ext=pext, inten=inten, freq=freq))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "temporal_evolution_1960_2024.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # (a) f_p + mu_p (dual axis): frequency flat, intensity rising
    ax = axes[0]
    ax.plot(d.center, d.f_p * 100, color="#e67e22", lw=1.8, label="pattern freq f_p (%)")
    ax.set_xlabel("window center year (30-yr rolling)"); ax.set_ylabel("f_p (%)", color="#e67e22")
    ax.tick_params(axis="y", labelcolor="#e67e22")
    ax2 = ax.twinx()
    ax2.plot(d.center, d.mu_p, color="#2c3e9a", lw=1.8, label=r"$\mu_p$ (pattern-day mean P, mm/d)")
    ax2.set_ylabel(r"$\mu_p$ (mm/d)", color="#2c3e9a"); ax2.tick_params(axis="y", labelcolor="#2c3e9a")
    ax.set_title("(a) Pattern frequency ~flat; pattern-day intensity rises", fontsize=9.5)
    # (b) Pr(YREP|pattern) — the robust thermo channel
    ax = axes[1]
    ax.plot(d.center, d.p_ext * 100, color="#1a7d1a", lw=2.0)
    ax.set_xlabel("window center year"); ax.set_ylabel("Pr(YREP | pattern day) (%)")
    ax.set_title("(b) Pattern-conditioned extreme probability rises\n(the robust ΔPr(E) channel)",
                 fontsize=9.5)
    # (c) within-window ΔPr(E) intensity (thermo) vs frequency (dynamic)
    ax = axes[2]
    ax.plot(d.center, d.inten, color="#1a7d1a", lw=1.8, label="intensity (thermo)")
    ax.plot(d.center, d.freq, color="#c0392b", lw=1.8, label="frequency (dynamic)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("window center year"); ax.set_ylabel("within-window ΔPr(E) term (prob)")
    ax.set_title("(c) Rolling 30-yr ΔPr(E): thermo positive, dynamic near zero", fontsize=9.5)
    ax.legend(fontsize=8)
    fig.suptitle("Fig 5c — Temporal evolution (30-yr rolling): the thermodynamic intensification "
                 "is gradual; pattern frequency stays flat (1960–2024)", fontsize=11.5, y=1.02)
    plt.tight_layout()
    p = os.path.join(FIG, "fig5c_temporal_evolution.png")
    plt.savefig(p, bbox_inches="tight"); plt.close(); print(f"[save] {p}")
    # quick numeric verdict
    print(f"[verdict] f_p: {d.f_p.iloc[0]:.3f} -> {d.f_p.iloc[-1]:.3f} | "
          f"mu_p: {d.mu_p.iloc[0]:.2f} -> {d.mu_p.iloc[-1]:.2f} mm/d | "
          f"Pr(YREP|pat): {d.p_ext.iloc[0]:.3f} -> {d.p_ext.iloc[-1]:.3f}")
    print(f"[verdict] within-window ΔPr(E) intensity positive in "
          f"{(d.inten > 0).sum()}/{len(d)} windows (mean {d.inten.mean():+.4f}); "
          f"freq mean {d.freq.mean():+.4f}")


if __name__ == "__main__":
    main()
