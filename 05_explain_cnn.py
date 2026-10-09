#!/usr/bin/env python3
"""05_explain_cnn.py — CNN interpretability via four XAI methods.

Applies LRP (alpha=1/beta=0, manual z+ rule), Integrated Gradients,
Grad-CAM (last conv layer), and SmoothGrad (NoiseTunnel over IG) to the
trained Combined-CNN. Per-method: per-channel relevance + spatial maps,
averaged over explained (positive) days.

Inputs: model_combined.pt, cnn_inputs_1deg_1960_2024.nc, YREP_day_table_1960_2024.csv
Outputs: relevance_<method>_combined.nc (+ LRP_map, GradCAM_map)
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

import config

# --------------------------------------------------------------------------- #
# Manual alpha=1/beta=0 LRP (z+ rule) for YRCNN.
# z+ for Conv2d/Linear, winner-take-all (max_unpool) for MaxPool2d,
# dropout = identity at eval, input layer uses a+. Keeps MaxPool2d (no AvgPool swap).
# --------------------------------------------------------------------------- #
def _fwd_pos_conv(conv, a):
    return F.conv2d(a, torch.clamp(conv.weight, min=0), None,
                    conv.stride, conv.padding, conv.dilation, conv.groups)

def _fwd_pos_lin(fc, a):
    return F.linear(a, torch.clamp(fc.weight, min=0))

def _zplus(forward_pos, a_in, R_out, eps=1e-9, clamp_pos=False):
    a = a_in.detach().requires_grad_(True)
    zp = forward_pos(a)
    s = R_out / (zp + eps)
    g = torch.autograd.grad((zp * s.detach()).sum(), a, retain_graph=False)[0]
    base = a.clamp(min=0) if clamp_pos else a  # input layer: a+ for strict a1b0
    return (base * g).detach()


def lrp_alpha1_beta0(model, X, target=1, eps=1e-9):
    """Manual alpha=1/beta=0 LRP through YRCNN. X (N,C,H,W) -> input relevance (N,C,H,W)."""
    model.eval()
    with torch.enable_grad():
        a1 = F.relu(model.conv1(X))
        p1, idx1 = F.max_pool2d(a1, 2, return_indices=True)
        a2 = F.relu(model.conv2(p1))
        p2, idx2 = F.max_pool2d(a2, 2, return_indices=True)
        flat = p2.flatten(1)
        a3 = F.relu(model.fc1(flat))
        logits = model.fc2(a3)
        Rt = logits[:, target]
        W2p = torch.clamp(model.fc2.weight, min=0)
        z2p_t = (a3 * W2p[target]).sum(1)
        s2 = Rt / (z2p_t + eps)
        R_a3 = a3 * (s2.unsqueeze(1) * W2p[target].unsqueeze(0))
        R_flat = _zplus(lambda a: _fwd_pos_lin(model.fc1, a), flat, R_a3, eps)
        R_p2 = R_flat.view_as(p2)
        R_a2 = F.max_unpool2d(R_p2, idx2, kernel_size=2, output_size=a2.shape)
        R_p1 = _zplus(lambda a: _fwd_pos_conv(model.conv2, a), p1, R_a2, eps)
        R_a1 = F.max_unpool2d(R_p1, idx1, kernel_size=2, output_size=a1.shape)
        R_X = _zplus(lambda a: _fwd_pos_conv(model.conv1, a), X, R_a1, eps, clamp_pos=True)
    return R_X


sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
m04 = import_module("04_train_cnn")
YRCNN, CHANNEL_SETS, HP = m04.YRCNN, m04.CHANNEL_SETS, m04.HP

DEFAULT_OUT = config.OUT


def relevance_per_channel(attr, X):
    """Raw |relevance| summed over spatial dims, averaged over batch -> (C,)."""
    return attr.abs().sum(dim=(2, 3)).mean(dim=0).detach().cpu().numpy()


def explain(model, X, y, device, target=1, n_smooth=25):
    """LRP, IG, Grad-CAM, SmoothGrad on positive examples; per-channel + spatial maps."""
    from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution, NoiseTunnel
    model.eval()
    pos = X[y == 1][:32]
    if len(pos) == 0:
        raise RuntimeError("no positive samples to explain")
    Xin = pos.clone().detach().requires_grad_(True).to(device)
    out = {}
    # 1) LRP alpha=1/beta=0 (manual z+)
    a = lrp_alpha1_beta0(model, Xin, target=target)
    out["LRP_a1b0"] = relevance_per_channel(a, Xin)
    out["LRP_map_pos"] = a.abs().mean(0).detach().cpu().numpy()  # (C,H,W)
    # 2) Integrated Gradients
    # internal_batch_size chunks the (batch x n_steps) expansion — on the 82x122
    # grid the one-shot 32x32 expand is a ~23 GB activation spike and OOMs even
    # an empty 32 GB GPU. Chunking is numerically identical.
    ig = IntegratedGradients(model)
    a = ig.attribute(Xin, target=target, baselines=Xin * 0.0, n_steps=32,
                     internal_batch_size=64)
    out["IG"] = relevance_per_channel(a, Xin)
    # 3) Grad-CAM on conv2
    gc = LayerGradCam(model, model.conv2)
    a = gc.attribute(Xin, target=target)
    out["GradCAM_map"] = a.mean(dim=0).squeeze().detach().cpu().numpy()
    up = LayerAttribution.interpolate(a, Xin.shape[2:])
    out["GradCAM_per_ch"] = ((up * Xin.abs()).sum(dim=(2, 3)) /
                             Xin.abs().sum(dim=(2, 3)).clamp(min=1e-6)).mean(dim=0).detach().cpu().numpy()
    out["GradCAM_map_interp"] = up.mean(dim=0).squeeze().detach().cpu().numpy()
    # 4) SmoothGrad — NoiseTunnel(smoothgrad) over IG
    sg = NoiseTunnel(ig)
    a = sg.attribute(Xin, target=target, nt_type="smoothgrad", nt_samples=n_smooth,
                     stdevs=0.15, internal_batch_size=64)
    out["SmoothGrad"] = relevance_per_channel(a, Xin)
    out["SmoothGrad_map_pos"] = a.abs().mean(0).detach().cpu().numpy()
    return out


def smoke(device, seed=0):
    rng = np.random.RandomState(seed)
    N, H, W = 600, 82, 122
    y = (rng.rand(N) < 0.12).astype(np.int64)
    X = rng.randn(N, 5, H, W).astype(np.float32)
    X[y == 1, 0] += 1.5  # signal ONLY on channel 0 (Z500)
    Xt = torch.from_numpy(X).to(device); yt = torch.from_numpy(y).to(device)
    model = YRCNN(5, HP).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(15):
        opt.zero_grad(); loss = lossf(model(Xt), yt); loss.backward(); opt.step()
    print(f"[smoke] trained 5-ch CNN; train acc={float((model(Xt).argmax(1)==yt).float().mean()):.3f}")
    res = explain(model, Xt.cpu(), yt.cpu(), device)
    ch = ["Z500", "SLP", "IVT_mag", "VIMFC", "PW"]
    print("[smoke] per-channel relevance (signal injected on Z500 = ch0):")
    for meth in ("LRP_a1b0", "IG", "GradCAM_per_ch", "SmoothGrad"):
        v = res.get(meth)
        if v is None:
            continue
        top = ch[int(np.argsort(-v)[0])]
        print(f"  {meth:16s}: " + "  ".join(f"{ch[i]}={v[i]:.3f}" for i in range(5))
              + f"   top={top} {'PASS' if top==ch[0] else '(not Z500)'}")
    print("[smoke] OK — LRP z+ + IG + Grad-CAM + SmoothGrad all ran.")


def main(model_pt, nc_path, csv_path, outdir, device):
    """Explain the Combined-CNN on YREP test days; save per-method relevance .nc."""
    X, y, dates = m04.load_inputs_and_labels(nc_path, csv_path, CHANNEL_SETS["combined"])
    model = YRCNN(len(CHANNEL_SETS["combined"]), HP).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    res = explain(model, torch.from_numpy(X), torch.from_numpy(y), device)
    ds_in = xr.open_dataset(nc_path)
    lat, lon = ds_in.lat.values, ds_in.lon.values
    chans = CHANNEL_SETS["combined"]
    for meth in ("LRP_a1b0", "IG", "GradCAM_per_ch", "SmoothGrad"):
        if res.get(meth) is None:
            continue
        xr.DataArray(res[meth], dims=("channel",), coords=dict(channel=chans),
                     name=f"relevance_per_channel_{meth}").to_netcdf(
            os.path.join(outdir, f"relevance_{meth}_combined.nc"))
    xr.DataArray(res["GradCAM_map_interp"], dims=("lat", "lon"),
                 coords=dict(lat=lat, lon=lon), name="gradcam_map").to_netcdf(
        os.path.join(outdir, "relevance_GradCAM_map_combined.nc"))
    if res.get("LRP_map_pos") is not None:
        xr.DataArray(res["LRP_map_pos"], dims=("channel", "lat", "lon"),
                     coords=dict(channel=chans, lat=lat, lon=lon),
                     name="lrp_relevance_a1b0").to_netcdf(
            os.path.join(outdir, "relevance_LRP_map_combined.nc"))
    print("[explain] per-channel relevance (Combined-CNN, YREP days), ranked:")
    for meth in ("LRP_a1b0", "IG", "GradCAM_per_ch", "SmoothGrad"):
        v = res.get(meth)
        if v is None:
            continue
        rank = np.argsort(-v)
        print(f"  {meth:16s}: " + ", ".join(f"{chans[i]}={v[i]:.3f}" for i in rank))
    print(f"[save] relevance_*.nc in {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(DEFAULT_OUT, "model_combined.pt"))
    ap.add_argument("--cnn-in", default=os.path.join(DEFAULT_OUT, "cnn_inputs_1deg_1960_2024.nc"))
    ap.add_argument("--labels", default=os.path.join(DEFAULT_OUT, "YREP_day_table_1960_2024.csv"))
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[device] {device}  (captum + manual z+ LRP)")
    if a.smoke:
        smoke(device)
    else:
        main(a.model, a.cnn_in, a.labels, a.outdir, device)
