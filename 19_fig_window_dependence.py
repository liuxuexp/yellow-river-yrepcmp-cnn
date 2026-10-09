#!/usr/bin/env python3
"""19_fig_window_dependence.py — Window-dependence of the robust attribution.

The robust dynamic/thermodynamic attribution of Yellow-River extreme-precipitation
intensification is time-window-dependent:
  - 1960-2024 (1960-91 vs 1992-24): ΔPr(E) annual INTENSITY/thermodynamic robust
    (year-block CI [+0.0006,+0.0054]); frequency spans 0.
  - 1979-2024 (1979-2000 vs 2001-24): ΔPr(E) annual FREQUENCY/dynamic robust
    (year-block CI [+0.0007,+0.0051]); intensity spans 0.

Figure: grouped bars of the ΔPr(E) annual frequency (dynamic) vs intensity
(thermodynamic) term for each window, with year-block bootstrap 95% CI error bars
and a robustness annotation (CI excludes 0). Self-contained: recomputes both
windows with decomposition.decompose() + the year-block bootstrap.

PNG only. Output: figures/fig5d_window_dependence.png
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

sys.path.insert(0, os.path.dirname(__file__))
import decomposition as D

OUT = config.OUT
FIG = config.FIGDIR
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 11})
RNG = np.random.RandomState(20260713); NBOOT = 4000

pbar, yrep, dates, labels = D.load_labels()
yr = dates.year.values
X = yrep.astype(float)   # ΔPr(E)

def window_terms(e0, e1, l0, l1, nb=NBOOT):
    """Point terms + year-block bootstrap CIs for one window's ΔPr(E) annual split."""
    ye_all = np.unique(yr[(yr >= e0) & (yr <= e1)])
    yl_all = np.unique(yr[(yr >= l0) & (yr <= l1)])
    ie = (yr >= e0) & (yr <= e1); il = (yr >= l0) & (yr <= l1)
    r0 = D.decompose(X, labels, ie, il)
    fB, iB = [], []
    for _ in range(nb):
        ye = RNG.choice(ye_all, size=len(ye_all), replace=True)
        yl = RNG.choice(yl_all, size=len(yl_all), replace=True)
        r = D.decompose(X, labels, np.isin(yr, ye), np.isin(yr, yl))
        fB.append(r["term_freq"]); iB.append(r["term_intensity"])
    def ci(a): lo, hi = np.percentile(a, [2.5, 97.5]); return float(lo), float(hi)
    return dict(freq=r0["term_freq"], inten=r0["term_intensity"],
                freq_ci=ci(fB), inten_ci=ci(iB),
                f_p_e=r0["early"]["f_p"], f_p_l=r0["late"]["f_p"])

W = {
    "1960–2024\n(1960–91 vs 1992–24)": window_terms(1960, 1991, 1992, 2024),
    "1979–2024\n(1979–2000 vs 2001–24)": window_terms(1979, 2000, 2001, 2024),
}
# save provenance
rows = []
for w, r in W.items():
    rows.append(dict(window=w.replace("\n", " "), term="freq_dynamic", point=r["freq"],
                     ci_low=r["freq_ci"][0], ci_high=r["freq_ci"][1],
                     robust=bool(r["freq_ci"][0] > 0 or r["freq_ci"][1] < 0)))
    rows.append(dict(window=w.replace("\n", " "), term="inten_thermo", point=r["inten"],
                     ci_low=r["inten_ci"][0], ci_high=r["inten_ci"][1],
                     robust=bool(r["inten_ci"][0] > 0 or r["inten_ci"][1] < 0)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig5d_window_dependence_data.csv"), index=False)

# ---- plot ----
DYN, THM = "#c0392b", "#2980b9"   # dynamic=freq=red, thermodynamic=inten=blue
fig, ax = plt.subplots(figsize=(8.8, 5.4))
xs = np.arange(len(W)); width = 0.35
for i, (wname, r) in enumerate(W.items()):
    # frequency / dynamic bar
    flo, fhi = r["freq_ci"]; ferr_f = [[r["freq"] - flo], [fhi - r["freq"]]]
    frob = (flo > 0 or fhi < 0)
    ax.bar(i - width/2, r["freq"], width, color=DYN, edgecolor="white",
           label="frequency / dynamic" if i == 0 else None, zorder=3)
    ax.errorbar(i - width/2, r["freq"], yerr=ferr_f, fmt="none", ecolor="k",
                capsize=5, lw=1.4, zorder=4)
    ax.text(i - width/2, fhi + 0.0006, "★ robust" if frob else "spans 0",
            ha="center", fontsize=8.5, color=DYN if frob else "#888", fontweight="bold" if frob else "normal")
    # intensity / thermodynamic bar
    ilo, ihi = r["inten_ci"]; ferr_i = [[r["inten"] - ilo], [ihi - r["inten"]]]
    irob = (ilo > 0 or ihi < 0)
    ax.bar(i + width/2, r["inten"], width, color=THM, edgecolor="white",
           label="intensity / thermodynamic" if i == 0 else None, zorder=3)
    ax.errorbar(i + width/2, r["inten"], yerr=ferr_i, fmt="none", ecolor="k",
                capsize=5, lw=1.4, zorder=4)
    ax.text(i + width/2, ihi + 0.0006, "★ robust" if irob else "spans 0",
            ha="center", fontsize=8.5, color=THM if irob else "#888", fontweight="bold" if irob else "normal")
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(xs); ax.set_xticklabels(list(W.keys()), fontsize=10)
ax.set_ylabel(r"$\Delta\mathrm{Pr}(E)$ annual term contribution (per day)")
ax.set_title("Fig 5d — the robust attribution is time-window-dependent\n"
             "(year-block bootstrap 95% CI; ★ = CI excludes 0)",
             fontsize=12)
ax.legend(loc="upper left", framealpha=0.95)
# annotate the headline contrast
ax.annotate("thermodynamic\n(intensity robust)", xy=(0 + width/2, W["1960–2024\n(1960–91 vs 1992–24)"]["inten"]),
            xytext=(0.42, 0.0042), fontsize=9, color=THM, ha="left",
            arrowprops=dict(arrowstyle="->", color=THM, lw=0.9))
ax.annotate("dynamic\n(frequency robust)", xy=(1 - width/2, W["1979–2024\n(1979–2000 vs 2001–24)"]["freq"]),
            xytext=(0.55, 0.0055), fontsize=9, color=DYN, ha="left",
            arrowprops=dict(arrowstyle="->", color=DYN, lw=0.9))
ax.text(0.5, -0.0009, f"pattern frequency f$_p$: 1960–2024 {W['1960–2024\n(1960–91 vs 1992–24)']['f_p_e']:.3f}→{W['1960–2024\n(1960–91 vs 1992–24)']['f_p_l']:.3f} (flat)   |   "
         f"1979–2024 {W['1979–2024\n(1979–2000 vs 2001–24)']['f_p_e']:.3f}→{W['1979–2024\n(1979–2000 vs 2001–24)']['f_p_l']:.3f} (rises)",
         transform=ax.transData, fontsize=8.2, color="#555", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5d_window_dependence.png"), bbox_inches="tight"); plt.close()
print(f"[save] {FIG}/fig5d_window_dependence.png")
for r in rows:
    print(f"  {r['window']:24s} {r['term']:13s} point={r['point']:+.4f} CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] {'ROBUST' if r['robust'] else 'spans0'}")
