#!/usr/bin/env python3
"""09a_fig3.py — Fig 3: CNN architecture + skill + probability relations.

Layout (6 panels, 3 rows; labelled a-f in reading order):
  (a) CNN architecture schematic — artwork, 4/5 width         [row 1 L]
  (b) Confusion matrix (combined, held-out test, P>=0.5) — 1/5 [row 1 R]
  (c) Precision-Recall curves (4 models, held-out test)        [row 2 L]
  (d) PR-AUC bar — three-model comparison + month-only baseline [row 2 R]
  (e) P(YR-EPCMP) vs area-mean P — probability tracks intensity [row 3 L]
  (f) P(YR-EPCMP) vs coverage F(d) — probability tracks extent  [row 3 R]

Tensor shapes in (a) are the 46x82 classifier's (model_combined.pt:
conv1 (16,5,3,3), conv2 (32,16,3,3), fc1 (64,6080)). Artwork:
data/schematics/fig3a_architecture.png.

Reads: out_legacy46x82/prediction_probability.csv,
       out_legacy46x82/prediction_probability_full.csv,
       out/precip_daily_region_1960_2024.nc
Output: figures/fig3.png
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import save_figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LogNorm
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix)
from scipy.ndimage import gaussian_filter1d
import config

FIGDIR  = config.FIGDIR
OUT     = config.OUT
OUT_LEG = config.OUT_LEGACY

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", pink="#CC79A7",
          gray="#777777")
MOD_STYLE = {                       # model -> (color, linestyle, linewidth, name)
    "combined":   (OI["vermillion"], "-",  2.0, "Combined (5ch)"),
    "circulation":(OI["blue"],       "--", 1.4, "Circulation (SLP+Z500)"),
    "moisture":   (OI["green"],      "--", 1.4, "Moisture (VIMFC)"),
    "month_only": (OI["gray"],       ":",  1.4, "Month-only baseline"),
}
ORDER = ["combined", "circulation", "moisture", "month_only"]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "font.size": 9.1,
                     "font.sans-serif": ["Noto Sans CJK TC", "Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6})
TTL = dict(fontsize=9.8, fontweight="bold", loc="left", pad=3)


# --------------------------------------------------------------------------- #
# Panel (a) — CNN architecture schematic (shapes from weights)
# --------------------------------------------------------------------------- #
def _abox(ax, x, y, w, h, text, fc, ec="#333", fs=5.6, fw="normal", tc="#222", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, zorder=4, linespacing=1.2)


def _aarrow(ax, x0, x1, y, color="#444", lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw,
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


def _panel_a(ax):
    """Legacy 46x82 architecture artwork
    (data/schematics/fig3a_architecture.png, body 1579x672; the
    version with baked shapes 46x82x5 -> 44x80x16 -> 22x40x16 -> 20x38x32 ->
    10x19x32 -> 6080 -> 64 -> 2)."""
    img = plt.imread(str(config.SCHEMATICS / "fig3a_architecture.png"))
    # extent= maps the raster onto the 0-1 axes box; the axes rect itself carries the
    # artwork aspect, so aspect="auto" fills the box without distortion.
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto", interpolation="none", zorder=3)
    t = ax.set_title("(a)  YR-EPCMP CNN architecture", **TTL)
    t.set_x(0.0313)


# --------------------------------------------------------------------------- #
def _smooth_pr(rec, prec, n=500, sigma=4.0):
    """Visually smooth a (noisy, step-like) PR curve: resample onto a uniform
    recall grid (np.interp, sorted ascending) then a light Gaussian low-pass.
    AUC stays from the raw scores (passed in separately); only the line is smoothed."""
    order = np.argsort(rec)
    rs, ps = rec[order], prec[order]
    rg = np.linspace(rs.min(), rs.max(), n)
    pg = np.interp(rg, rs, ps)
    pg = gaussian_filter1d(pg, sigma=sigma)
    return rg, np.clip(pg, 0.0, 1.0)


def _panel_bc(pp_test, ax_b, ax_c):
    base = pp_test[pp_test.model == "combined"].y_true.mean()
    # (b) PR curves
    for mod in ORDER:
        s = pp_test[pp_test.model == mod]
        if s.y_true.nunique() < 2:
            continue
        prec, rec, _ = precision_recall_curve(s.y_true, s.y_prob)
        auc = average_precision_score(s.y_true, s.y_prob)
        c, ls, lw, name = MOD_STYLE[mod]
        rg, pg = _smooth_pr(rec, prec)
        ax_b.plot(rg, pg, color=c, ls=ls, lw=lw, label=f"{name} ({auc:.3f})")
    ax_b.axhline(base, color=OI["black"], ls=":", lw=0.8, label=f"Random ({base:.3f})")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1.02)
    ax_b.set_xlabel("Recall", fontsize=9.3, labelpad=1.0); ax_b.set_ylabel("Precision", fontsize=9.3)
    ax_b.set_title("(c)  Precision–Recall (held-out test)", **TTL)
    ax_b.legend(fontsize=8.4, loc="upper right", framealpha=0.9)
    ax_b.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax_b.spines[s].set_visible(False)
    # (c) PR-AUC bar
    vals = [average_precision_score(pp_test[pp_test.model == m].y_true,
                                    pp_test[pp_test.model == m].y_prob) for m in ORDER]
    cols = [MOD_STYLE[m][0] for m in ORDER]
    x = np.arange(len(ORDER))
    ax_c.bar(x, vals, color=cols, alpha=0.9, width=0.43)
    for xi, v in zip(x, vals):
        ax_c.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.0)
    ax_c.axhline(base, color=OI["black"], ls=":", lw=0.8, label=f"Random ({base:.3f})")
    _xt = [MOD_STYLE[m][3].split(" (")[0] for m in ORDER]
    _xt[ORDER.index("month_only")] = "Month-only\nbaseline"
    ax_c.set_xticks(x); ax_c.set_xticklabels(_xt, rotation=0, fontsize=8.6, ha="center")
    ax_c.set_ylim(0, max(vals) * 1.22)
    ax_c.set_ylabel("Test PR-AUC", fontsize=9.3)
    ax_c.set_title("(d)  Three-model skill vs month-only baseline", **TTL)
    ax_c.legend(fontsize=8.4, loc="upper right", framealpha=0.9)
    ax_c.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)


def _panel_d(pp_test, ax):
    tc = pp_test[pp_test.model == "combined"]
    cm = confusion_matrix(tc.y_true, (tc.y_prob >= 0.5).astype(int), labels=[1, 0])
    M = np.array([[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]])  # [[TP,FN],[FP,TN]]
    im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=M.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}\n({100 * M[i, j] / M.sum():.1f}%)", ha="center",
                    va="center", fontsize=9.5,
                    color="white" if M[i, j] > M.max() / 2 else "#222")
    ax.set_xticks([0, 1]); ax_c_xt = ["P≥0.5", "P<0.5"]
    ax.set_yticks([0, 1])
    ax.set_xticklabels(ax_c_xt, fontsize=8.6)
    ax.set_yticklabels(["YREP", "Non-YREP"], fontsize=9.0, rotation=90,
                       va="center", ha="right")
    ax.set_xlabel("CNN prediction", fontsize=9.3)
    ax.set_title("(b)  Confusion matrix", **TTL)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------- #
def _binned_mean(x, y, nbin=10, min_n=20):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbin + 1)))
    bx, bm, bs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if sel.sum() >= min_n:
            bx.append(0.5 * (lo + hi)); bm.append(y[sel].mean())
            bs.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()))
    return np.array(bx), np.array(bm), np.array(bs)


def _panel_scatter(ax, x, y, xlab, title):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    hb = ax.hexbin(x, y, gridsize=32, cmap="magma_r", mincnt=1, norm=LogNorm(), alpha=0.92)
    bx, bm, bs = _binned_mean(x, y)
    # SE band — white so it reads against the dense (dark) part of the hex density.
    ax.fill_between(bx, bm - bs, bm + bs, color="white", alpha=0.38, zorder=4)
    # Binned-mean line: white halo + dark core, so it stays visible on BOTH the dark
    # (high-count) and light (low-count) ends of the magma_r map.
    ax.plot(bx, bm, "-", color="white", lw=3.0, zorder=5, solid_capstyle="round")
    ax.plot(bx, bm, "-", color="#111", lw=1.1, zorder=6, solid_capstyle="round")
    ax.plot(bx, bm, "o", ms=3.6, color="white", markeredgecolor="k", markeredgewidth=0.7,
            zorder=7, label="Binned mean ± SE")
    ax.axhline(0.5, color="0.35", ls=":", lw=0.9, label="CNN threshold 0.5")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel(xlab, fontsize=9.3)
    ax.set_ylabel("P(YR-EPCMP)  [Combined]", fontsize=9.3)
    ax.set_title(title, **TTL)
    ax.legend(fontsize=7.6, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return hb


# --------------------------------------------------------------------------- #
def main():
    # ---- data ----
    pp_test = pd.read_csv(os.path.join(OUT_LEG, "prediction_probability.csv"))   # test split, 4 models
    ds = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    dates = pd.to_datetime(ds.time.values).normalize()
    pbar = ds["Pbar"].values.astype(float)
    F = ds["F"].values.astype(float)
    ds.close()
    pf = pd.read_csv(os.path.join(OUT_LEG, "prediction_probability_full.csv"), parse_dates=["date"])
    pf = pf[pf.model == "combined"][["date", "y_prob"]].copy()
    pf["date"] = pf["date"].dt.normalize()
    df = pd.DataFrame({"date": dates, "Pbar": pbar, "F": F}).merge(pf, on="date", how="inner").dropna()

    # ---- figure ----
    fig = plt.figure(figsize=(7.48, 8.60))
    # Row 1: (a) artwork [rect left 0.030; TITLE held at x=0.050, see _panel_a]
    ax_a = fig.add_axes([0.0300, 0.7405, 0.6396, 0.2367])   # (a) architecture artwork
    ax_b = fig.add_axes([0.7230, 0.7838, 0.2270, 0.1934])   # confusion matrix, top = (a) top
    _panel_a(ax_a)
    _panel_d(pp_test, ax_b)            # confusion matrix  -> panel (b)

    # Row 2: (c) PR curves + (d) PR-AUC bar
    ax_c = fig.add_axes([0.0500, 0.4377, 0.4220, 0.2540])
    ax_d = fig.add_axes([0.5280, 0.4377, 0.4220, 0.2540])
    _panel_bc(pp_test, ax_c, ax_d)     # PR curves -> (c), PR-AUC bar -> (d)

    # Row 3: (e) scatter P + (f) scatter F
    ax_e = fig.add_axes([0.0500, 0.1135, 0.4220, 0.2637])
    ax_f = fig.add_axes([0.5280, 0.1135, 0.4220, 0.2637])
    hb_e = _panel_scatter(ax_e, df.Pbar.values, df.y_prob.values,
                          "Area-mean precipitation  $\\bar P$  (mm d$^{-1}$)",
                          "(e)  P(YR-EPCMP) vs intensity")
    hb_f = _panel_scatter(ax_f, df["F"].values, df.y_prob.values,
                          "Extreme-area fraction  $F(d)$  (Q95 coverage)",
                          "(f)  P(YR-EPCMP) vs extent")
    # shared colorbar for (e) & (f): one common count scale, drawn once in the e-f gap.
    vmax = max(float(np.nanmax(hb_e.get_array())), float(np.nanmax(hb_f.get_array())))
    shared = LogNorm(vmin=1, vmax=vmax)
    hb_e.set_norm(shared); hb_f.set_norm(shared)
    cax = fig.add_axes([0.275, 0.0400, 0.450, 0.0176])  # horizontal, centered below (e)&(f)
    cb = fig.colorbar(hb_e, cax=cax, orientation="horizontal")
    cb.set_label("Count (log)", fontsize=8.0)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=7.6)

    out = os.path.join(FIGDIR, "fig3.png")
    save_figure(fig, out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig3] saved {out}")
    # quick echo of the verified numbers
    for mod in ORDER:
        s = pp_test[pp_test.model == mod]
        print(f"       {MOD_STYLE[mod][3]:24s} PR-AUC={average_precision_score(s.y_true, s.y_prob):.3f}")
    tc = pp_test[pp_test.model == "combined"]
    cm = confusion_matrix(tc.y_true, (tc.y_prob >= 0.5).astype(int), labels=[1, 0])
    print(f"       confusion TP={cm[0,0]} FN={cm[0,1]} FP={cm[1,0]} TN={cm[1,1]}  "
          f"recall={cm[0,0]/(cm[0,0]+cm[0,1]):.2f} prec={cm[0,0]/(cm[0,0]+cm[1,0]):.2f}")
    print(f"       scatter n={len(df)}  pattern(P>=0.5)={int((df.y_prob>=0.5).sum())} "
          f"= {100*(df.y_prob>=0.5).mean():.2f}% of full record")


if __name__ == "__main__":
    main()
