#!/usr/bin/env python3
"""04_train_cnn.py — Train CNN classifiers on NCEP standardized-anomaly channels.

Trains three CNN classifiers (circulation, moisture, combined) plus a
month-only climatology baseline. Precipitation never enters the input —
only generates the YREP label.

Architecture: Conv2D(16,3x3,ReLU)->MaxPool2->Dropout(0.2)->Conv2D(32,3x3,ReLU)->
MaxPool2->Dropout(0.2)->Flatten->Linear(64,ReLU)->Dropout(0.3)->Linear(2).
Input 82×122 (-11-70N, 59-180E at 1 degree).

Training: Adam lr=1e-4, L2 weight_decay=1e-4, batch=256, CE with class weight,
early stop on val PR-AUC (patience 10). Threshold P(YREP)>0.5.
Splits: stratified 70/15/15 + temporal-block (train<=2005/val 2006-2014/test>=2015).
"""
import argparse, os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_fscore_support, roc_auc_score)

import config

DEFAULT_OUT = config.OUT
DEFAULT_CNN = os.path.join(DEFAULT_OUT, "cnn_inputs_1deg_1960_2024.nc")
DEFAULT_LBL = os.path.join(DEFAULT_OUT, "YREP_day_table_1960_2024.csv")

# channel sets; names match script 02's `_anom` outputs
CHANNEL_SETS = {
    "circulation": ["SLP_anom", "Z500_anom"],
    "moisture":    ["VIMFC_anom"],
    "combined":    ["Z500_anom", "SLP_anom", "IVT_mag_anom", "VIMFC_anom", "PW_anom"],
}
HP = dict(lr=1e-4, weight_decay=1e-4, batch_size=256, max_epochs=80,
          patience=10, dropout1=0.2, dropout2=0.2, dropout3=0.3, threshold=0.5,
          oversample=False)  # oversampling toggle (ablation)


# --------------------------------------------------------------------------- #
def _conv_pool_orphans(D, p, k=3):
    """Structural orphan check for ONE axis (length D) under the conv stack
    conv(k=3,stride=1,padding=p) -> MaxPool2d(2) -> conv(k=3,padding=p) -> MaxPool2d(2).
    Returns the set of input indices that have NO forward path to the flatten layer
    (dropped by an odd-dimension maxpool tail). Such pixels can NEVER carry LRP
    relevance (always exactly 0) -> the 'blank band' artifact. Pure geometry, no weights.
    """
    def covers(j):  # input idx conv-output j reads (k=3, pad p)
        return [j - p, j - p + 1, j - p + 2]
    c1 = D + 2 * p - 2  # conv1 out dim
    p1 = c1 // 2  # pool1 out (tail of c1 dropped if c1 odd)
    c2 = p1 + 2 * p - 2  # conv2 out dim
    p2 = c2 // 2  # pool2 out (tail of c2 dropped if c2 odd)
    alive_c2 = set(range(2 * p2))  # conv2 outputs surviving pool2
    alive_pool1 = set()
    for kk in alive_c2:
        for m in covers(kk):
            if 0 <= m < p1:
                alive_pool1.add(m)
    alive_c1 = set()
    for m in alive_pool1:
        if 0 <= 2 * m < c1:
            alive_c1.add(2 * m)
        if 0 <= 2 * m + 1 < c1:
            alive_c1.add(2 * m + 1)
    covered = set()
    for j in alive_c1:
        for i in covers(j):
            if 0 <= i < D:
                covered.add(i)
    return set(range(D)) - covered


def assert_no_orphan(grid, padding=0):
    """Raise if the CNN input grid + conv padding leaves any input pixel without a forward
    path (orphan). Prevents the maxpool dead-band artifact."""
    H, W = grid
    oh = sorted(_conv_pool_orphans(H, padding))
    ow = sorted(_conv_pool_orphans(W, padding))
    if oh or ow:
        raise ValueError(
            f"CNN grid {grid} (conv padding={padding}) ORPHANS input pixels with no forward "
            f"path -> they carry zero LRP relevance (the 'blank band'). Orphan lat rows={oh}, "
            f"lon cols={ow}. Pick dims so no MaxPool2d(2) tail-drops an edge: for pad=0 use "
            f"dims ≡2 mod 4 (e.g. 42x78); for pad=1 use dims divisible by 4 (e.g. 44x76).")


# --------------------------------------------------------------------------- #
class YRCNN(nn.Module):
    """CNN classifier. Input (N, C, 82, 122) -> logits (N, 2).  (grid = (lat, lon))"""
    def __init__(self, n_channels, hp=HP, grid=(82, 122)):
        super().__init__()
        assert_no_orphan(grid, padding=0)  # conv layers below use padding=0 (valid)
        self.conv1 = nn.Conv2d(n_channels, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.drop1 = nn.Dropout2d(hp["dropout1"])
        self.drop2 = nn.Dropout2d(hp["dropout2"])
        # flatten size via a dummy pass (robust to grid/kernel changes):
        # 82x122 ->conv3-> 80x120 ->pool2-> 40x60 ->conv3-> 38x58 ->pool2-> 19x29
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, *grid)
            x = self.drop2(F.max_pool2d(F.relu(self.conv2(
                self.drop1(F.max_pool2d(F.relu(self.conv1(dummy)), 2)))), 2))
            flat = x.numel()  # 32 * 19 * 29 = 17632
        self.fc1 = nn.Linear(flat, 64)
        self.drop3 = nn.Dropout(hp["dropout3"])
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.drop1(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.drop2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = torch.flatten(x, 1)
        x = self.drop3(F.relu(self.fc1(x)))
        return self.fc2(x)


# --------------------------------------------------------------------------- #
def load_inputs_and_labels(nc_path, csv_path, channels):
    """Inner-join NCEP daily anomaly fields with YREP labels by date.
    Returns X (N, C, lat, lon) float32, y (N,) int, dates (N,) datetime64."""
    ds = xr.open_dataset(nc_path)
    fields = [ds[c].values.astype(np.float32) for c in channels]  # each (time,lat,lon)
    X = np.stack(fields, axis=1)  # (N, C, lat, lon)
    era_dates = pd.to_datetime(ds.time.values).normalize()
    ds.close()
    lab = pd.read_csv(csv_path, parse_dates=["date"])
    lab["date"] = lab["date"].dt.normalize()
    df = pd.DataFrame({"date": era_dates}).reset_index().rename(columns={"index": "i"})
    df = df.merge(lab[["date", "YREP_label"]], on="date", how="inner")
    idx = df["i"].values
    y = df["YREP_label"].values.astype(np.int64)
    dates = df["date"].values
    return X[idx], y, dates


def class_weights(y):
    """w_c = N / (2 * N_c)."""
    N = len(y)
    w = np.zeros(2, dtype=np.float32)
    for c in (0, 1):
        Nc = int((y == c).sum())
        w[c] = N / (2.0 * Nc) if Nc > 0 else 1.0
    return w


def make_splits(dates, y, seed=0):
    """Stratified 70/15/15 (main) + temporal-block (train<=2005/val2006-14/test>=2015)."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    def fracs(idx):
        n_tr = int(round(0.70 * len(idx))); n_va = int(round(0.15 * len(idx)))
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    p_tr, p_va, p_te = fracs(pos); n_tr, n_va, n_te = fracs(neg)
    splits = {
        "train": np.concatenate([p_tr, n_tr]),
        "val":   np.concatenate([p_va, n_va]),
        "test":  np.concatenate([p_te, n_te]),
    }
    yrs = pd.to_datetime(dates).year
    tb = {
        "train": np.where(yrs <= 2005)[0],
        "val":   np.where((yrs >= 2006) & (yrs <= 2014))[0],
        "test":  np.where(yrs >= 2015)[0],
    }
    return splits, tb


def score_binary(y_true, p, threshold=HP["threshold"]):
    """PR-AUC, ROC-AUC, Brier, precision/recall/F1 at threshold."""
    mets = {}
    if len(np.unique(y_true)) == 2:
        mets["pr_auc"] = float(average_precision_score(y_true, p))
        mets["roc_auc"] = float(roc_auc_score(y_true, p))
        mets["brier"] = float(brier_score_loss(y_true, p))
    yhat = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary",
                                                       zero_division=0)
    mets.update(precision=float(prec), recall=float(rec), f1=float(f1), threshold=threshold)
    return mets


def season_of(months):
    """Map month numbers (1-12) -> season label DJF/MAM/JJA/SON."""
    m = np.asarray(months)
    s = np.empty(m.shape, dtype="<U3")
    s[np.isin(m, [12, 1, 2])]  = "DJF"
    s[np.isin(m, [3, 4, 5])]   = "MAM"
    s[np.isin(m, [6, 7, 8])]   = "JJA"
    s[np.isin(m, [9, 10, 11])] = "SON"
    return s


def seasonal_metrics(y_sub, p_sub, dates_sub, threshold=HP["threshold"]):
    """Per-season PR-AUC/precision/recall/F1 on already-aligned subsets."""
    out = {}
    seas = season_of(pd.to_datetime(dates_sub).month)
    for s in ("DJF", "MAM", "JJA", "SON"):
        si = np.where(seas == s)[0]
        if len(si) == 0 or len(np.unique(y_sub[si])) < 2:
            continue
        mets = score_binary(y_sub[si], p_sub[si], threshold)
        mets.update(n=len(si), n_pos=int(y_sub[si].sum()))
        out[s] = mets
    return out


def evaluate(model, X, y, device, threshold=HP["threshold"]):
    if len(y) == 0:
        nan = float("nan")
        return (dict(n=0, n_pos=0, threshold=threshold, pr_auc=nan, roc_auc=nan,
                     brier=nan, precision=0.0, recall=0.0, f1=0.0), np.zeros(0))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    mets = score_binary(y, p, threshold)
    mets.update(n=len(y), n_pos=int(y.sum()))
    return mets, p


def train_one(name, channels, X, y, dates, device, hp=HP, log_fn=print):
    """Train one CNN; return best state_dict (val PR-AUC), per-split metrics, test probs, splits."""
    import random
    torch.manual_seed(0); np.random.seed(0); random.seed(0)  # reproducible weights + loader shuffle
    splits, tb = make_splits(dates, y)
    w = class_weights(y[splits["train"]])
    log_fn(f"[{name}] channels={channels}  train n={len(splits['train'])} "
           f"(pos={int(y[splits['train']].sum())})  class_weight={w}")

    Xt = torch.from_numpy(X[splits["train"]]).float().to(device)
    yt = torch.from_numpy(y[splits["train"]]).long().to(device)
    Xv = torch.from_numpy(X[splits["val"]]).float().to(device)
    if hp.get("oversample"):
        # WeightedRandomSampler over-samples the minority (YREP) class
        yc = y[splits["train"]]
        inv = 1.0 / np.bincount(yc)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=hp["batch_size"],
                        sampler=WeightedRandomSampler(inv[yc], len(yc), replacement=True))
    else:
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=hp["batch_size"], shuffle=True)

    model = YRCNN(len(channels), hp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    lossf = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))

    best_pr, best_state, wait = -1.0, None, 0
    for ep in range(hp["max_epochs"]):
        model.train(); total = 0.0
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            total += loss.item() * len(yb)
        m_val, _ = evaluate(model, X[splits["val"]], y[splits["val"]], device)
        if m_val.get("pr_auc", 0.0) > best_pr:
            best_pr, best_state, wait = m_val["pr_auc"], {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    out = {}
    for split_name, idx in [("val", splits["val"]), ("test", splits["test"]),
                            ("tb_val", tb["val"]), ("tb_test", tb["test"])]:
        m, _ = evaluate(model, X[idx], y[idx], device)
        out[split_name] = m
    _, p_test = evaluate(model, X[splits["test"]], y[splits["test"]], device)
    for s, mets in seasonal_metrics(y[splits["test"]], p_test, dates[splits["test"]]).items():
        out[f"test_{s}"] = mets  # DJF/MAM/JJA/SON
    log_fn(f"[{name}] BEST val PR-AUC={best_pr:.3f}  "
           f"test PR-AUC={out['test'].get('pr_auc',float('nan')):.3f} "
           f"recall={out['test']['recall']:.3f} precision={out['test']['precision']:.3f} "
           f"F1={out['test']['f1']:.3f} brier={out['test'].get('brier',float('nan')):.3f}")
    return best_state, out, splits, p_test


def monthonly_baseline(dates, y, hp=HP, log_fn=print):
    """P(YREP) from calendar-month climatology (train per-month YREP rate,
    Laplace-smoothed). Same stratified splits (seed 0, deterministic)."""
    splits, _ = make_splits(dates, y)
    months = pd.to_datetime(dates).month.values
    tr = splits["train"]
    rate = np.zeros(13, dtype=np.float32)  # 1..12
    for m in range(1, 13):
        sel = tr[months[tr] == m]
        rate[m] = (int(y[sel].sum()) + 0.5) / (len(sel) + 1.0)
    out = {}
    for split_name, idx in [("val", splits["val"]), ("test", splits["test"])]:
        if len(idx) == 0:
            continue
        p = rate[months[idx]]
        mets = score_binary(y[idx], p, hp["threshold"])
        mets.update(n=len(idx), n_pos=int(y[idx].sum()))
        out[split_name] = mets
    p_full = rate[months]
    ti = splits["test"]
    for s, mets in seasonal_metrics(y[ti], rate[months[ti]], dates[ti]).items():
        out[f"test_{s}"] = mets
    log_fn(f"[month_only] test PR-AUC={out.get('test',{}).get('pr_auc',float('nan')):.3f}  "
           f"(floor the CNN must beat)")
    return out, splits, p_full


# --------------------------------------------------------------------------- #
def main(nc_path, csv_path, outdir, device, oversample=False):
    os.makedirs(outdir, exist_ok=True)
    hp = dict(HP); hp["oversample"] = oversample
    rows, prob_rows, prob_full = [], [], []
    dates_ref, y_ref = None, None
    for name, channels in CHANNEL_SETS.items():
        X, y, dates = load_inputs_and_labels(nc_path, csv_path, channels)
        dates_ref, y_ref = dates, y  # identical across models (same time axis + labels)
        state, metrics, splits, p = train_one(name, channels, X, y, dates, device, hp=hp)
        torch.save(state, os.path.join(outdir, f"model_{name}.pt"))
        for split_name, m in metrics.items():
            rows.append(dict(model=name, split=split_name, **m))
        ti = splits["test"]
        for d, yi, pi in zip(pd.to_datetime(dates[ti]), y[ti], p):
            prob_rows.append(dict(date=d.strftime("%Y-%m-%d"), model=name, split="test",
                                  y_true=int(yi), y_prob=float(pi)))
        # full-record probs -> pattern-positive day labels
        model_full = YRCNN(len(channels)).to(device)
        model_full.load_state_dict(state); model_full.eval()
        # chunked: one-shot 23742x5x82x122 forward OOMs (conv1 act alone ~14.6 GB on
        # the 82x122 grid). Chunking is numerically identical.
        with torch.no_grad():
            Xt_full = torch.from_numpy(X).float()
            p_full = torch.cat([
                F.softmax(model_full(Xt_full[i0:i0 + 2048].to(device)), dim=1)[:, 1].cpu()
                for i0 in range(0, Xt_full.shape[0], 2048)]).numpy()
        for d, yi, pi in zip(pd.to_datetime(dates), y, p_full):
            prob_full.append(dict(date=d.strftime("%Y-%m-%d"), model=name,
                                  y_true=int(yi), y_prob=float(pi)))

    # ---- month-only baseline ----
    mo_metrics, mo_splits, mo_pfull = monthonly_baseline(dates_ref, y_ref)
    for split_name, m in mo_metrics.items():
        rows.append(dict(model="month_only", split=split_name, **m))
    for d, yi, pi in zip(pd.to_datetime(dates_ref), y_ref, mo_pfull):
        prob_full.append(dict(date=d.strftime("%Y-%m-%d"), model="month_only",
                              y_true=int(yi), y_prob=float(pi)))
    # month-only TEST-split probs -> prediction_probability.csv too.
    mo_ti = mo_splits["test"]
    for d, yi, pi in zip(pd.to_datetime(dates_ref[mo_ti]), y_ref[mo_ti], mo_pfull[mo_ti]):
        prob_rows.append(dict(date=d.strftime("%Y-%m-%d"), model="month_only", split="test",
                              y_true=int(yi), y_prob=float(pi)))

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    pd.DataFrame(prob_rows).to_csv(os.path.join(outdir, "prediction_probability.csv"), index=False)
    pd.DataFrame(prob_full).to_csv(os.path.join(outdir, "prediction_probability_full.csv"), index=False)
    summ = pd.DataFrame(rows)
    summ = summ[summ.split == "test"][["model", "pr_auc", "roc_auc", "brier",
                                       "precision", "recall", "f1", "n_pos"]]
    print("\n[summary] held-out TEST metrics (sorted by PR-AUC):")
    print(summ.sort_values("pr_auc", ascending=False).to_string(index=False))
    print(f"\n[save] {outdir}/model_metrics.csv + prediction_probability.csv "
          f"+ prediction_probability_full.csv + model_*.pt")


def smoke(outdir, device, seed=0):
    """Synthetic-input smoke test of the full code path (no inputs needed)."""
    rng = np.random.RandomState(seed)
    N, H, W = 600, 82, 122
    all_d = pd.date_range("1960-01-01", "2024-12-31", freq="D").values
    dates = all_d[np.sort(rng.choice(len(all_d), size=N, replace=False))]
    y = (rng.rand(N) < 0.10).astype(np.int64)
    X = rng.randn(N, 5, H, W).astype(np.float32)
    X[y == 1, 0] += 1.5
    print(f"[smoke] X={X.shape} y pos_rate={y.mean():.3f} "
          f"years {pd.to_datetime(dates).year.min()}-{pd.to_datetime(dates).year.max()}")
    for name, channels in CHANNEL_SETS.items():
        train_one(name, channels, X[:, :len(channels)], y, dates, device)
    monthonly_baseline(dates, y)
    print("[smoke] OK — CNN architecture + training loop + metrics + month-only baseline all run.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-in", default=DEFAULT_CNN)
    ap.add_argument("--labels", default=DEFAULT_LBL)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--oversample", action="store_true",
                    help="over-sample YREP minority class; default off (class-weight only)")
    a = ap.parse_args()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[device] {device}  (torch {torch.__version__})  oversample={a.oversample}")
    if a.smoke:
        smoke(a.outdir, device)
    else:
        main(a.cnn_in, a.labels, a.outdir, device, a.oversample)
