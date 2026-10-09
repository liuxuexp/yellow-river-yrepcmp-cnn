#!/usr/bin/env python3
"""04b_cnn_ensemble.py — Init-seed ensemble of the Combined-CNN for model uncertainty.

Reuses 04_train_cnn.train_one verbatim. make_splits is seed-0-locked internally
so train/val/test splits are identical across seeds; only weight init +
DataLoader shuffle vary (a clean init-ensemble). For each seed: train, save
model, compute full-record probs, binarize at 0.5, run decompose() on 4 cases.

Outputs:
  model_combined_seed{k}.pt                        (k=0..K-1)
  prediction_probability_combined_seeds.csv
  decomposition_ensemble_1960_2024.csv
  decomposition_ensemble_summary_1960_2024.csv
"""
import argparse, os, importlib.util, time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn.functional as F

import config

HERE = os.path.dirname(__file__)
OUT = config.OUT


def _load_04():
    """Import the digit-prefixed 04_train_cnn.py as a module."""
    spec = importlib.util.spec_from_file_location("cnn04", os.path.join(HERE, "04_train_cnn.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _load_pbar_yrep():
    """P̄ and YREP indicator aligned to the NCEP date axis (from the precip cache)."""
    ds = xr.open_dataset(os.path.join(OUT, "precip_daily_region_1960_2024.nc"))
    pbar = ds["Pbar"].values.astype(float)
    yrep = ds["YREP"].values.astype(float)
    dates_cache = pd.to_datetime(ds.time.values).normalize()
    ds.close()
    return pbar, yrep, dates_cache


def main(K, device):
    cnn04 = _load_04()
    hp = dict(cnn04.HP)  # canonical HP; oversample off (canonical)
    channels = cnn04.CHANNEL_SETS["combined"]
    CNN = os.path.join(OUT, "cnn_inputs_1deg_1960_2024.nc")
    LBL = os.path.join(OUT, "YREP_day_table_1960_2024.csv")
    print(f"[ensemble] K={K}  device={device}  channels=combined({len(channels)})")

    X, y, dates = cnn04.load_inputs_and_labels(CNN, LBL, channels)
    pbar, yrep, dates_cache = _load_pbar_yrep()
    era = pd.to_datetime(dates).normalize()
    assert np.array_equal(era.values, dates_cache.values), "NCEP & cache date axes differ!"
    yr = era.year.values; mo = era.month.values
    jas = np.isin(mo, [7, 8, 9])

    seeds = list(range(1000, 1000 + K))  # deterministic seed list
    prob_rows, decomp_rows, metric_rows = [], [], []
    t0 = time.time()
    for k, seed in enumerate(seeds):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        state, metrics, splits, _ = cnn04.train_one(
            f"combined_s{seed}", channels, X, y, dates, device, hp=hp)
        torch.save(state, os.path.join(OUT, f"model_combined_seed{k}.pt"))
        # full-record probs -> pattern labels (chunked: one-shot forward OOMs on
        # the 82x122 grid — same fix as 04_train_cnn.py)
        m = cnn04.YRCNN(len(channels)).to(device); m.load_state_dict(state); m.eval()
        with torch.no_grad():
            Xt_full = torch.from_numpy(X).float()
            p = torch.cat([
                F.softmax(m(Xt_full[i0:i0 + 2048].to(device)), dim=1)[:, 1].cpu()
                for i0 in range(0, Xt_full.shape[0], 2048)]).numpy()
        labels = (p >= hp["threshold"]).astype(int)
        for d, yi, pi in zip(era, y, p):
            prob_rows.append(dict(date=d.strftime("%Y-%m-%d"), seed=k, y_true=int(yi), y_prob=float(pi)))
        metric_rows.append(dict(seed=k, pr_auc_test=metrics["test"].get("pr_auc", np.nan),
                                f_p=float(labels.mean())))
        # per-seed decomposition (4 cases) — labels=pattern partition, X=P̄ or YREP indicator
        for name, Xd, sel in [("dPbar_annual", pbar, np.ones(len(yr), bool)),
                              ("dPbar_JAS",    pbar, jas),
                              ("dPrE_annual",  yrep, np.ones(len(yr), bool)),
                              ("dPrE_JAS",     yrep, jas)]:
            from decomposition import decompose, EARLY_MAIN, LATE_MAIN
            r = decompose(Xd, labels, (yr <= EARLY_MAIN) & sel, (yr >= LATE_MAIN) & sel)
            tot = r["term_freq"] + r["term_intensity"] + r["term_interaction"]
            for term, val in [("freq", r["term_freq"]), ("inten", r["term_intensity"]),
                              ("inter", r["term_interaction"]),
                              ("thermo_share_pct", 100 * r["term_intensity"] / tot if abs(tot) > 1e-12 else np.nan),
                              ("delta", r["delta"])]:
                decomp_rows.append(dict(seed=k, case=name, term=term, value=val))
        print(f"[ensemble] seed {k} ({seed}) done — test PR-AUC={metrics['test'].get('pr_auc',0):.3f} "
              f"f_p={labels.mean():.3f}  [{time.time()-t0:.0f}s elapsed]")

    pd.DataFrame(prob_rows).to_csv(
        os.path.join(OUT, "prediction_probability_combined_seeds.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(OUT, "ensemble_cnn_metrics.csv"), index=False)
    dd = pd.DataFrame(decomp_rows)
    dd.to_csv(os.path.join(OUT, "decomposition_ensemble_1960_2024.csv"), index=False)

    # summary: per (case, term) across seeds
    summ_rows = []
    print("\n=== SEED-ENSEMBLE DECOMPOSITION (model uncertainty) ===")
    for (case, term), g in dd.groupby(["case", "term"]):
        v = g.value.values
        summ_rows.append(dict(case=case, term=term, mean=float(np.mean(v)), std=float(np.std(v)),
                              min=float(np.min(v)), max=float(np.max(v)),
                              frac_positive=float(np.mean(v > 0)), all_positive=bool(np.all(v > 0)),
                              all_negative=bool(np.all(v < 0))))
        flag = ("ROBUST+" if np.all(v > 0) else "ROBUST-" if np.all(v < 0) else "spans-0")
        print(f"  {case:13s} {term:17s}: {np.mean(v):+.4f} ± {np.std(v):.4f}  "
              f"[{np.min(v):+.4f},{np.max(v):+.4f}]  {flag}")
    pd.DataFrame(summ_rows).to_csv(
        os.path.join(OUT, "decomposition_ensemble_summary_1960_2024.csv"), index=False)
    print(f"\n[save] out/decomposition_ensemble_1960_2024.csv (+summary), "
          f"prediction_probability_combined_seeds.csv, model_combined_seed*.pt")
    print(f"[PR-AUC] mean={np.mean([r['pr_auc_test'] for r in metric_rows]):.3f} "
          f"std={np.std([r['pr_auc_test'] for r in metric_rows]):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="K=2 smoke")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    K = 2 if a.quick else a.seeds
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    main(K, device)
