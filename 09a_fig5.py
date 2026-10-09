#!/usr/bin/env python3
"""09a_fig5.py — Fig 5: frequency/intensity trends + decomposition + PDF.

Layout (6 panels):
  (a) YR-EPCMP pattern frequency f_p        — annual + Theil-Sen + MK p
  (b) Conditional intensity μ_p (+ μ_n)    — annual + Theil-Sen + MK p
  (c) ΔP̄ closed-form decomposition          — grouped bars freq/inten/inter, annual vs
      JAS, + year-block bootstrap 95% CI; robust terms (CI excludes 0) starred.
  (d) ΔPr(E) decomposition                   — same (the doubly-robust thermodynamic headline)
  (e) early/late PDF of pattern-day P̄       — the visual Δμ_p tail-shift
  (f) Sampling-bootstrap robustness forest   — only ΔPr(E) annual intensity excludes 0

Decomposition identity (closes to machine precision, residual ≈1e-16):
  ΔP̄   = Δf_p·(μ_p0−μ_n0) + f_p0·Δμ_p + (1−f_p0)·Δμ_n + Δf_p·(Δμ_p−Δμ_n)
  ΔPr(E)= Δf_p·(Pr0_p−Pr0_n)+ f_p0·ΔPr_p + (1−f_p0)·ΔPr_n + Δf_p·(ΔPr_p−ΔPr_n)
  four terms = pattern-frequency (dynamic) / conditional-intensity (thermo) /
               background / interaction.

Reads: out_legacy46x82/yearly_metrics_1960_2024.csv,
       out_legacy46x82/trend_statistics.csv,
       out_legacy46x82/decomposition_headline_1960_2024.csv,
       out_legacy46x82/decomposition_boot_full_1960_2024.csv,
       out_legacy46x82/prediction_probability_full.csv,
       out/precip_daily_region_1960_2024.nc
Output: figures/fig5.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.lines import Line2D
from scipy.stats import theilslopes, kendalltau, gaussian_kde
import config

FIGDIR = config.FIGDIR
OUT    = config.OUT
OUT_LEG = config.OUT_LEGACY

# Okabe-Ito palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")
C_FREQ  = OI["vermillion"]     # frequency / dynamic
C_INTEN = OI["blue"]           # intensity / thermodynamic
C_INTER = OI["gray"]           # interaction
C_ROB   = OI["green"]          # robust (CI excludes 0)
C_DATA  = OI["blue"]           # time-series data line
C_TREND = OI["black"]          # Theil-Sen line
C_EARLY = OI["blue"]           # early period
C_LATE  = OI["vermillion"]     # late period

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.8, fontweight="bold", loc="left", pad=3)

EARLY_Y, LATE_Y = (1960, 1991), (1992, 2024)   # canonical early/late split
BRK = 1991                                       # canonical breakpoint


# --------------------------------------------------------------------------- #
def _mk_p(x, y):
    return float(kendalltau(x, y).pvalue)


# --------------------------------------------------------------------------- #
# Panel (a) — pattern frequency f_p trend (dynamic channel)
# --------------------------------------------------------------------------- #
def _panel_a(ax, ym, yr):
    ndays = np.array([366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365 for y in yr])
    fp = 100.0 * ym["YREPCMP_freq"].values / ndays      # pattern-day fraction (%)
    good = ~np.isnan(fp)
    ax.axvspan(EARLY_Y[0] - 0.5, EARLY_Y[1] + 0.5, color=OI["sky"], alpha=0.10, zorder=0)
    ax.axvspan(LATE_Y[0] - 0.5, LATE_Y[1] + 0.5, color=OI["yellow"], alpha=0.16, zorder=0)
    ax.plot(yr[good], fp[good], "o-", color=C_DATA, ms=2.0, lw=0.8, mfc=C_DATA, mec=C_DATA)
    tsl = theilslopes(fp[good], yr[good])
    ax.plot(yr, tsl.intercept + tsl.slope * yr, "--", color=C_TREND, lw=1.0)
    p = _mk_p(yr[good], fp[good])
    ax.text(0.5 * (EARLY_Y[0] + EARLY_Y[1]), 0.955, "Early 1960–91",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#555")
    ax.text(0.5 * (LATE_Y[0] + LATE_Y[1]), 0.955, "Late 1992–24",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#555")
    ax.plot([], [], "-", color=C_DATA, lw=0.8, marker="o", ms=2.5,
            label=r"$f_p$ (pattern-day fraction)")
    ax.plot([], [], "--", color=C_TREND, lw=1.0,
            label=f"Theil–Sen {tsl.slope:+.4f} %/yr (MK p={p:.2f})")
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.995, 0.89),
              framealpha=0.9, handlelength=1.8, borderpad=0.3, handletextpad=0.4)
    ax.set_xlabel("Year", fontsize=8.6, labelpad=1.5); ax.set_ylabel("Pattern frequency  $f_p$  (%)", fontsize=8.6)
    ax.set_title("(a)  Pattern frequency", **TTL)
    ax.tick_params(labelsize=7.8)
    ax.tick_params(axis="x", pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Panel (b) — conditional intensity μ_p (+ μ_n background) trend (thermo channel)
# --------------------------------------------------------------------------- #
def _panel_b(ax, ym, yr, ts):
    mup = ym["YREPCMP_mean_P"].values        # pattern-day mean P̄  (mm/d)
    mun = ym["nonYREPCMP_mean_P"].values     # non-pattern mean P̄  (mm/d)
    g = ~np.isnan(mup)
    # authoritative μ_p trend (trend_statistics.csv: YREPCMP_mean_P)
    rp = ts[ts.metric == "YREPCMP_mean_P"].iloc[0]
    sp, pp = float(rp.theilsen_per_yr), float(rp.mk_p)
    ci = [float(v) for v in str(rp.boot95).strip("[]").split(",")]
    ax.axvspan(EARLY_Y[0] - 0.5, EARLY_Y[1] + 0.5, color=OI["sky"], alpha=0.10, zorder=0)
    ax.axvspan(LATE_Y[0] - 0.5, LATE_Y[1] + 0.5, color=OI["yellow"], alpha=0.16, zorder=0)
    ax.plot(yr[g], mup[g], "o-", color=C_INTEN, ms=2.0, lw=0.9)
    ax.plot(yr, rp.early_mean + sp * (yr - EARLY_Y[0]), "--", color=C_TREND, lw=1.0)
    # μ_n background (faint)
    gn = ~np.isnan(mun)
    ax.plot(yr[gn], mun[gn], "-", color=OI["gray"], lw=0.7, alpha=0.8)
    rn = ts[ts.metric == "nonYREPCMP_mean_P"].iloc[0]
    ax.plot(yr, rn.early_mean + float(rn.theilsen_per_yr) * (yr - EARLY_Y[0]),
            ":", color=OI["gray"], lw=0.7)
    ax.text(0.5 * (EARLY_Y[0] + EARLY_Y[1]), 1.01, "Early 1960–91",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#555")
    ax.text(0.5 * (LATE_Y[0] + LATE_Y[1]), 1.01, "Late 1992–24",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="#555")
    ax.plot([], [], "-", color=C_INTEN, lw=0.9, marker="o", ms=2.5,
            label=r"$\mu_p$ (pattern-day $\bar P$)")
    ax.plot([], [], "-", color=OI["gray"], lw=0.7,
            label=r"$\mu_n$ (non-pattern background)")
    ax.plot([], [], "--", color=C_TREND, lw=1.0, label=r"Theil–Sen trend")
    ax.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(0.985, 0.37),
              framealpha=0.9, handlelength=1.8, borderpad=0.3, handletextpad=0.4)
    ax.text(0.015, 0.205, r"$\mu_p$ trend %+.4f mm d$^{-1}$ yr$^{-1}$" % sp +
            "\n(MK p=%.2f, boot95 [%+.3f,%+.3f])" % (pp, ci[0], ci[1]),
            transform=ax.transAxes, ha="left", va="center", fontsize=6.5, color="#222",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    ax.set_xlabel("Year", fontsize=8.6, labelpad=1.5); ax.set_ylabel(r"Conditional mean $\bar P$  (mm d$^{-1}$)", fontsize=8.6)
    ax.set_title("(b)  Conditional intensity (pattern-conditioned)", **TTL)
    ax.tick_params(labelsize=7.8)
    ax.tick_params(axis="x", pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Panels (c)/(d) — closed-form decomposition grouped bars (annual vs JAS) + CI
# --------------------------------------------------------------------------- #
def _decomp_panel(ax, head, boot, season_keys, boot_cases, title, unit, vfmt="%+.3f", ylim=None, xgap=0.5):
    """Grouped bars: freq/inten/inter at x=annual, x=JAS; year-block CI; robust star.

    Offsets/ylim are scale-relative (work for both mm/d and probability units): M = max
    |value| incl. CI over the panel; ylim = [-0.40M, 2.00M] unless overridden. Every
    labelled bar carries its value ABOVE the CI whisker cap; adjacent labels in a
    cluster that would land within 9 pt of each other are staircased one line apart.
    Above each bar sits, bottom to top: the value label (skipped for near-zero bars,
    |v|<=0.04M), the green robust star (CI excludes 0), and the bold thermo-% /
    "net≈0" tag anchored above the cluster's max upper extent + 6 pt so nothing
    collides. vfmt formats the value labels (mm/d -> %+.3f, probability -> %+.4f)."""
    head = head.assign(decomp=head["decomposition"].str.replace(r"\s+", " ", regex=True).str.strip())
    terms = [("freq", C_FREQ, "frequency"),
             ("inten", C_INTEN, "intensity\n(pattern-conditioned)"),
             ("inter", C_INTER, "interaction")]
    w = 0.115
    M = 0.0
    bars, clusters = [], []
    for ix, (skey, bcase) in enumerate(zip(season_keys, boot_cases)):
        xc = ix * xgap
        row = head[head.decomp == skey].iloc[0]
        f, i, it = (row.term_freq_dynamic, row.term_intensity_thermo, row.term_interaction)
        tot = f + i + it
        # thermo share only when the net change is a meaningful fraction of the terms;
        # when terms largely cancel the % is undefined -> mark NaN.
        meaningful = abs(tot) > 0.30 * max(abs(f), abs(i), abs(it), 1e-9)
        share = 100.0 * i / tot if meaningful else float("nan")
        cl_hi = 0.0
        for ti, (tk, col, _lbl) in enumerate(terms):
            off = (ti - 1) * w
            v = {"freq": f, "inten": i, "inter": it}[tk]
            sub = boot[(boot.case == bcase) & (boot.term == tk) & (boot.bootstrap == "year")]
            if len(sub):
                r = sub.iloc[0]
                lo, hi, robust = float(r.ci_low), float(r.ci_high), bool(r.ci_excludes_0)
            else:
                lo = hi = v; robust = False
            ax.bar(xc + off, v, w, color=col, alpha=0.90, edgecolor="white", lw=0.4)
            ax.errorbar(xc + off, v, yerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor=C_TREND, elinewidth=0.8, capsize=2.2, zorder=5)
            bars.append((ix, xc + off, v, hi, robust))
            M = max(M, abs(v), abs(lo), abs(hi))
            cl_hi = max(cl_hi, hi, v)
        clusters.append((xc, share, cl_hi, not meaningful))
    M = M if M > 0 else 1.0
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ylim = (-0.40 * M, 2.00 * M)
        ax.set_ylim(*ylim)
    # pt->data scale: label/star/tag offsets are specified in points so the mm/d panel (c)
    # and the probability panel (d) share identical geometry regardless of unit scale.
    pt2d = (ylim[1] - ylim[0]) / (ax.get_position().height
                                  * ax.figure.get_size_inches()[1] * 72.0)
    GAP, LF, LH = 2.5, 7.2, 1.15    # label gap above whisker (pt) / fontsize / line-height factor
    cl_top = [c[2] for c in clusters]
    placed = []
    for ix_c, x, v, hi, robust in bars:
        if abs(v) <= 0.04 * M:
            continue
        y = hi + GAP * pt2d
        prev = [p for p in placed if p[0] == ix_c]
        if prev and abs(y - prev[-1][1]) < 9.0 * pt2d:
            y = prev[-1][1] + 9.0 * pt2d
        top = y + LF * LH * pt2d
        ax.text(x, y, vfmt % v, ha="center", va="bottom", fontsize=LF, color="#333", zorder=6)
        if robust:
            sy = top + 4.0 * pt2d
            ax.plot(x, sy, marker="*", color=C_ROB, ms=8.0, mec="k", mew=0.3,
                    zorder=6, clip_on=False)
            top = sy + 4.0 * pt2d
        placed.append((ix_c, top))
        cl_top[ix_c] = max(cl_top[ix_c], top)
    for (xc, share, cl_hi, cancelled), top in zip(clusters, cl_top):
        if cancelled:
            ax.annotate("net ≈ 0 (terms cancel)", (xc, top + 6.0 * pt2d),
                        ha="center", va="bottom", fontsize=7.5, color="#888", style="italic")
        else:
            ax.annotate("intensity %+.0f%%" % share, (xc, top + 6.0 * pt2d),
                        ha="center", va="bottom", fontsize=7, color=C_INTEN, fontweight="bold")
    ax.axhline(0, color=C_TREND, lw=0.8, xmin=0.06, xmax=0.94)
    ax.set_xticks([0, xgap]); ax.set_xticklabels(["Annual", "JAS"], fontsize=8.1)
    ax.set_xlim(-2.0 * w, xgap + 2.0 * w)
    ax.set_ylabel(unit, fontsize=8.6)
    ax.set_title(title, **TTL)
    ax.tick_params(labelsize=7.8)
    ax.tick_params(axis="x", pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# Panel (e) — early/late PDF of pattern-day P̄
# --------------------------------------------------------------------------- #
def _panel_e(ax, df):
    df = df.copy(); df["year"] = df.date.dt.year
    pat = df["y_prob"] >= 0.5
    e = pat & (df.year >= EARLY_Y[0]) & (df.year <= EARLY_Y[1])
    l = pat & (df.year >= LATE_Y[0]) & (df.year <= LATE_Y[1])
    xe = df.loc[e, "Pbar"].values; xl = df.loc[l, "Pbar"].values
    grid = np.linspace(0, max(xe.max(), xl.max()) * 1.02, 400)
    pe = gaussian_kde(xe)(grid); pl = gaussian_kde(xl)(grid)
    ymax = float(max(pe.max(), pl.max()))
    ax.fill_between(grid, pe, color=C_EARLY, alpha=0.30)
    ax.fill_between(grid, pl, color=C_LATE, alpha=0.30)
    ax.plot(grid, pe, color=C_EARLY, lw=1.3)
    ax.plot(grid, pl, color=C_LATE, lw=1.3)
    ax.axvline(xe.mean(), color=C_EARLY, ls="--", lw=0.9)
    ax.axvline(xl.mean(), color=C_LATE, ls="--", lw=0.9)
    dmu = xl.mean() - xe.mean()
    bx = 0.985
    by, dy = 0.972, 0.112
    ax.text(bx, by,        "early 1960–91    μ=%.2f  (n=%d)" % (xe.mean(), int(e.sum())),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.9, color=C_EARLY,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_EARLY, lw=0.5, alpha=0.9))
    ax.text(bx, by - dy,   "late  1992–24    μ=%.2f  (n=%d)" % (xl.mean(), int(l.sum())),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.9, color=C_LATE,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_LATE, lw=0.5, alpha=0.9))
    ax.text(bx, by - 2*dy, r"$\Delta\mu_p = %+.2f$  mm d$^{-1}$" % dmu,
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color="#222",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333", lw=0.5, alpha=0.9))
    ax.set_xlim(0, grid[-1]); ax.set_ylim(0, ymax * 1.18)
    ax.set_xlabel(r"Pattern-day area-mean precipitation  $\bar P$  (mm d$^{-1}$)", fontsize=8.6)
    ax.set_ylabel("Density", fontsize=8.6)
    ax.set_title("(e)  Early/Late PDF", **TTL)
    ax.tick_params(labelsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return dmu, int(e.sum()), int(l.sum())


# --------------------------------------------------------------------------- #
# Panel (f) — sampling-bootstrap robustness forest (4 cases × freq/inten)
# --------------------------------------------------------------------------- #
def _panel_f(ax, boot):
    cases = ["dPrE_annual", "dPrE_JAS", "dPbar_annual", "dPbar_JAS"]   # top -> bottom
    case_lbl = {"dPbar_annual": r"$\Delta\bar P$" + "\nAnnual",
                "dPbar_JAS":    r"$\Delta\bar P$" + "\nJAS",
                "dPrE_annual":  r"$\Delta\Pr(E)$" + "\nAnnual",
                "dPrE_JAS":     r"$\Delta\Pr(E)$" + "\nJAS"}
    y = 0
    yticks, ylbls, glbls, gseps = [], [], [], []
    for ci, case in enumerate(cases):
        for term, mk in [("inten", "o"), ("freq", "s")]:
            r = boot[(boot.case == case) & (boot.term == term) & (boot.bootstrap == "year")].iloc[0]
            robust = bool(r.ci_excludes_0)
            col = C_ROB if robust else OI["gray"]
            ax.errorbar(float(r["point"]), y, yerr=[[float(r["point"]) - float(r.ci_low)],
                       [float(r.ci_high) - float(r["point"])]], fmt=mk, color=col, ecolor=col,
                       capsize=2.6, ms=5.2, lw=1.6, zorder=3,
                       markeredgecolor="k" if robust else col, mew=0.4 if robust else 0)
            yticks.append(y); ylbls.append(term.capitalize())   # Freq / Inten
            y += 1
        glbls.append((y - 1.5, case_lbl[case]))
        if ci < len(cases) - 1:
            gseps.append(y + 0.275)
        y += 0.55
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylbls, fontsize=7.3, rotation=0, va="center", ha="right")
    for ys in gseps:
        ax.axhline(ys, color="#dddddd", lw=0.6, zorder=0)
    for yy, lab in glbls:
        ax.text(0.015, yy, lab, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.8, fontweight="bold", color="#333",
                zorder=5)
    ax.axvline(0, color=C_TREND, lw=0.9)
    ax.set_xlim(-0.04, 0.10)
    ax.set_xlabel("Term value (point + year-block 95% CI)", fontsize=8.6)
    ax.set_title("(f)  Robustness forest",
                 **TTL)
    ax.tick_params(labelsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend([Line2D([0], [0], color=C_ROB, lw=1.6, marker="o", ms=5),
               Line2D([0], [0], color=OI["gray"], lw=1.6, marker="s", ms=5)],
              ["robust (CI excludes 0)", "not robust"], fontsize=7, loc="upper right",
              framealpha=0.9)


# --------------------------------------------------------------------------- #
def _load_pdf_data():
    ds = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    dates = pd.to_datetime(ds.time.values).normalize()
    pbar = ds["Pbar"].values.astype(float)
    ds.close()
    pp = pd.read_csv(os.path.join(OUT_LEG, "prediction_probability_full.csv"), parse_dates=["date"])
    pp = pp[pp.model == "combined"][["date", "y_prob"]].copy()
    pp["date"] = pp["date"].dt.normalize()
    df = pd.DataFrame({"date": dates, "Pbar": pbar}).merge(pp, on="date", how="inner").dropna()
    return df


def main():
    ym = pd.read_csv(os.path.join(OUT_LEG, "yearly_metrics_1960_2024.csv"))
    ym = ym.rename(columns={ym.columns[0]: "year"})
    yr = ym["year"].values.astype(int)
    ts = pd.read_csv(os.path.join(OUT_LEG, "trend_statistics.csv"))
    head = pd.read_csv(os.path.join(OUT_LEG, "decomposition_headline_1960_2024.csv"))
    boot = pd.read_csv(os.path.join(OUT_LEG, "decomposition_boot_full_1960_2024.csv"))
    df = _load_pdf_data()

    fig = plt.figure(figsize=(7.48, 7.4))
    # row 1 — trends
    ax_a = fig.add_axes([0.030, 0.7395, 0.430, 0.2092]); _panel_a(ax_a, ym, yr)
    ax_b = fig.add_axes([0.515, 0.7395, 0.430, 0.2092]); _panel_b(ax_b, ym, yr, ts)
    # row 2 — decomposition bars
    ax_c = fig.add_axes([0.030, 0.4516, 0.430, 0.2092])
    _decomp_panel(ax_c, head, boot,
                  ["ΔP̄ annual", "ΔP̄ JAS"], ["dPbar_annual", "dPbar_JAS"],
                  "(c)  $\\Delta\\bar P$ decomposition",
                  "contribution (mm d$^{-1}$)", vfmt="%+.3f", ylim=(-0.18, 0.37))
    ax_d = fig.add_axes([0.515, 0.4516, 0.430, 0.2092])
    _decomp_panel(ax_d, head, boot,
                  ["ΔPr(E) annual", "ΔPr(E) JAS"], ["dPrE_annual", "dPrE_JAS"],
                  "(d)  $\\Delta\\Pr(E)$ decomposition",
                  "contribution (probability)", vfmt="%+.4f", ylim=(-0.018, 0.037))
    # c↔d share one vertical reference at 1/10 scale (d's probability units ≈ c's mm/d ÷10):
    # d's ylim is exactly c's ÷10 and its ticks are c's ÷10, so the two zero-lines and
    # every tick level sit at the same height across the two panels.
    ax_c.set_yticks([-0.1, 0.0, 0.1, 0.2])
    ax_d.set_yticks([-0.01, 0.0, 0.01, 0.02])
    ax_d.yaxis.set_label_coords(-0.11, 0.5)
    # row 3 — PDF + robustness
    ax_e = fig.add_axes([0.030, 0.1354, 0.430, 0.2627])
    dmu, ne, nl = _panel_e(ax_e, df)
    ax_f = fig.add_axes([0.515, 0.1354, 0.430, 0.2627])
    _panel_f(ax_f, boot)

    out = os.path.join(FIGDIR, "fig5.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig5] saved {out}")

    # ---- numeric echo (sanity: bars close, μ_p trend authoritative) ----
    h = head.assign(decomp=head.decomposition.str.replace(r"\s+", " ", regex=True).str.strip())
    for k in ["ΔP̄ annual", "ΔP̄ JAS", "ΔPr(E) annual", "ΔPr(E) JAS"]:
        r = h[h.decomp == k].iloc[0]
        s = r.term_freq_dynamic + r.term_intensity_thermo + r.term_interaction
        print(f"       {k:16s} delta={r.delta_X:+.5f}  sum3={s:+.5f}  resid={r.residual:.1e}  "
              f"thermo%={100*r.term_intensity_thermo/r.delta_X:+.0f}")
    rp = ts[ts.metric == "YREPCMP_mean_P"].iloc[0]
    print(f"       μ_p trend {float(rp.theilsen_per_yr):+.4f} mm/d/yr p={float(rp.mk_p):.2f} "
          f"boot95 {rp.boot95};  f_p(%) MK p={_mk_p(yr, 100*ym.YREPCMP_freq.values/np.array([366 if (y%4==0 and (y%100!=0 or y%400==0)) else 365 for y in yr])):.2f}")
    print(f"       PDF early μ vs late μ  Δμ_p={dmu:+.3f} mm/d  (n {ne}→{nl})")


if __name__ == "__main__":
    main()
