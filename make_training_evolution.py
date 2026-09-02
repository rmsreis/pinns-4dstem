# -*- coding: utf-8 -*-
"""
Pedagogical animation: how the SIREN-PINN reconstruction of each strain
component evolves during training, at 1% / 10% / 50% probe sampling.

One MP4 + GIF per component (e_xx, e_yy, e_xy), laid out as
[Ground Truth | 1% | 10% | 50%] with the live R^2 in each panel title.

Run with:  /Users/robertoreis/anaconda3/envs/pinns/bin/python make_evolution_anim.py
(must be run from the project root so ./data/*.npy resolve)
"""
import os, time, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from pathlib import Path

# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print("device:", device)

OUT = Path("outputs/training_evolution"); OUT.mkdir(parents=True, exist_ok=True)

EPOCHS      = 600           # fixed budget (no early stop) so every fraction has the same frames
SNAP_EVERY  = 12            # capture a frame every N epochs  -> ~50 frames
HOLD_FRAMES = 10            # freeze on the converged result at the end
FRACS       = [0.01, 0.10, 0.50]
FPS         = 6

config = dict(
    architecture="siren", hidden_dim=128, num_layers=6, skip_connections=True, omega_0=30.0,
    nu=0.27, E=150e9,
    n_collocation=2000, alpha_eq=0.1, alpha_co=0.1, w_data=1.0,
    lambda_phys_max=1.0, ramp_tau=500.0,
    normalize_residuals=True, phys_norm_warmup=200,
    lr=1e-3, lr_decay_factor=0.95, lr_decay_interval=500,
    enable_rar=True, rar_trigger_epoch=200, rar_interval=100,
    rar_num_new_points_pct=0.10, rar_max_pool_mult=5,
)

# ============================ data (mirrors notebook Section 2) ============
root = Path.cwd()
e_xx = np.load(root/"data"/"strain_exx.npy")
e_xy = np.load(root/"data"/"strain_exy.npy")
e_yy = np.load(root/"data"/"strain_eyy.npy")
mask = np.isfinite(e_xx).astype(np.float32)
theta = np.arctan2(e_xy, (e_xx - e_yy)/2.0 + 1e-10)
H, W = e_xx.shape
valid = mask.astype(bool)

scalers = {}
for name, arr in {"e_xx": e_xx, "e_yy": e_yy, "e_xy": e_xy, "theta": theta}.items():
    v = arr[valid]
    scalers[name] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)) + 1e-8}

def scale(arr, name):
    s = scalers[name]; return (arr - s["mean"]) / s["std"]
def unscale_t(t, name):
    s = scalers[name]; return t * s["std"] + s["mean"]

x_lin = torch.linspace(0, 1, W, dtype=torch.float32)
y_lin = torch.linspace(0, 1, H, dtype=torch.float32)
X, Y = torch.meshgrid(x_lin, y_lin, indexing="xy")
x_flat = X.flatten().to(device); y_flat = Y.flatten().to(device)
coords_all = torch.stack([x_flat, y_flat], dim=1)

e_xx_s = torch.tensor(scale(e_xx, "e_xx").flatten(), dtype=torch.float32, device=device)
e_yy_s = torch.tensor(scale(e_yy, "e_yy").flatten(), dtype=torch.float32, device=device)
e_xy_s = torch.tensor(scale(e_xy, "e_xy").flatten(), dtype=torch.float32, device=device)
th_s   = torch.tensor(scale(theta, "theta").flatten(), dtype=torch.float32, device=device)
tgt_all = torch.stack([e_xx_s, e_yy_s, e_xy_s, th_s], dim=1)

# ============================ model (SIREN + skips) =======================
class SirenLayer(nn.Module):
    def __init__(self, i, o, is_first=False, omega_0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(i, o)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/i, 1/i)
            else:
                b = np.sqrt(6/i)/omega_0
                self.linear.weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class StrainPINN_SIREN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6, omega_0=30.0, skip_connections=True):
        super().__init__()
        self.skip_connections = skip_connections
        self.first = SirenLayer(2, hidden_dim, is_first=True, omega_0=omega_0)
        self.hidden_layers = nn.ModuleList([
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0)
            for _ in range(num_layers - 2)])
        if skip_connections:
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final = nn.Linear(hidden_dim, 4)
        with torch.no_grad():
            self.final.weight.uniform_(-np.sqrt(6/hidden_dim), np.sqrt(6/hidden_dim))
    def forward(self, x):
        h = self.first(x)
        for i, layer in enumerate(self.hidden_layers):
            if self.skip_connections and i % 2 == 0 and i > 0:
                h = layer(h) + self.skip_proj(h)
            else:
                h = layer(h)
        return self.final(h)

def make_model():
    m = StrainPINN_SIREN(config["hidden_dim"], config["num_layers"],
                         omega_0=config["omega_0"], skip_connections=config["skip_connections"])
    return m.to(device)

# ============================ physics loss (autodiff) ====================
def _grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True, retain_graph=True)[0]

def physics_residuals(model, n_coll, cfg, extra_pts=None):
    nu = cfg["nu"]
    pts = torch.rand(n_coll, 2, device=device)
    if extra_pts is not None and extra_pts.numel() > 0:
        pts = torch.cat([pts, extra_pts.to(device)], dim=0)
    pts = pts.detach().requires_grad_(True)
    out = model(pts)
    exx, eyy, exy = out[:, 0], out[:, 1], out[:, 2]
    g_xx = _grad(exx, pts); g_yy = _grad(eyy, pts); g_xy = _grad(exy, pts)
    exx_x, exx_y = g_xx[:, 0], g_xx[:, 1]
    eyy_x, eyy_y = g_yy[:, 0], g_yy[:, 1]
    exy_x, exy_y = g_xy[:, 0], g_xy[:, 1]
    sxx_x = exx_x + nu*eyy_x
    syy_y = eyy_y + nu*exx_y
    c_xy  = (1.0 - nu)*0.5
    sxy_x = c_xy*exy_x
    sxy_y = c_xy*exy_y
    L_eq = ((sxx_x + sxy_y)**2 + (sxy_x + syy_y)**2).mean()
    exx_yy = _grad(exx_y, pts)[:, 1]
    eyy_xx = _grad(eyy_x, pts)[:, 0]
    exy_xy = _grad(exy_x, pts)[:, 1]
    L_co = (exx_yy + eyy_xx - 2.0*exy_xy).pow(2).mean()
    return L_eq, L_co

def normalized_physics(L_eq, L_co, cfg, ema):
    ema["n"] = ema.get("n", 0) + 1
    if ema["n"] <= cfg["phys_norm_warmup"]:
        for k, L in (("eq", L_eq), ("co", L_co)):
            v = float(L.detach())
            ema[k] = v if k not in ema else 0.99*ema[k] + 0.01*v
    return (cfg["alpha_eq"]*L_eq/(ema["eq"] + 1e-12)
            + cfg["alpha_co"]*L_co/(ema["co"] + 1e-12))

def rar_new_points(model, cfg, n_candidates=5000):
    nu = cfg["nu"]
    pts = torch.rand(n_candidates, 2, device=device, requires_grad=True)
    out = model(pts)
    exx, eyy, exy = out[:, 0], out[:, 1], out[:, 2]
    g_xx = _grad(exx, pts); g_yy = _grad(eyy, pts); g_xy = _grad(exy, pts)
    sxx_x = g_xx[:, 0] + nu*g_yy[:, 0]
    syy_y = g_yy[:, 1] + nu*g_xx[:, 1]
    c_xy  = (1.0 - nu)*0.5
    resid = (sxx_x + c_xy*g_xy[:, 1])**2 + (c_xy*g_xy[:, 0] + syy_y)**2
    k = max(1, int(cfg["rar_num_new_points_pct"]*cfg["n_collocation"]))
    idx = torch.topk(resid.detach(), min(k, resid.numel())).indices
    return pts.detach()[idx]

# ============================ ground-truth references ====================
gt_np   = {"e_xx": e_xx, "e_yy": e_yy, "e_xy": e_xy}
gt_flat = {k: torch.tensor(v.flatten(), dtype=torch.float32, device=device) for k, v in gt_np.items()}
vmask_t = torch.tensor(mask.flatten() > 0.5, device=device)
sstot   = {k: float(((gt_flat[k][vmask_t] - gt_flat[k][vmask_t].mean())**2).sum()) for k in gt_np}

@torch.no_grad()
def snapshot(model):
    model.eval()
    out = model(coords_all)
    maps, r2 = {}, {}
    for j, k in enumerate(["e_xx", "e_yy", "e_xy"]):
        pred = unscale_t(out[:, j], k)
        ssres = float(((pred[vmask_t] - gt_flat[k][vmask_t])**2).sum())
        r2[k] = 1.0 - ssres/sstot[k]
        m = pred.detach().cpu().numpy().reshape(H, W)
        m[~valid] = np.nan
        maps[k] = m
    model.train()
    return maps, r2

# ============================ training with capture ======================
def train_capture(frac):
    torch.manual_seed(SEED)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=config["lr_decay_interval"],
                                            gamma=config["lr_decay_factor"])
    # sparse training mask (same RNG convention as the notebook)
    rng = np.random.RandomState(SEED + int(frac*1000))
    vidx = np.where(mask.flatten())[0]
    nsamp = max(1, int(np.ceil(frac*len(vidx))))
    sel = rng.choice(vidx, nsamp, replace=False)
    rng2 = np.random.RandomState(SEED); rng2.shuffle(sel)
    nval = max(1, int(0.10*len(sel)))
    tr_idx = torch.as_tensor(sel[nval:] if len(sel) - nval > 0 else sel, dtype=torch.long, device=device)
    tr_in  = coords_all[tr_idx]
    tr_tgt = tgt_all[tr_idx]

    ema, coll_pool = {}, torch.empty(0, 2, device=device)
    frames = []
    t0 = time.time()
    for epoch in range(EPOCHS + 1):
        if epoch % SNAP_EVERY == 0 or epoch == EPOCHS:
            maps, r2 = snapshot(model)
            frames.append((epoch, maps, r2))

        if epoch == EPOCHS:
            break

        if (config["enable_rar"] and epoch >= config["rar_trigger_epoch"]
                and epoch % config["rar_interval"] == 0):
            coll_pool = torch.cat([coll_pool, rar_new_points(model, config)], dim=0)
            cap = config["rar_max_pool_mult"]*config["n_collocation"]
            if coll_pool.shape[0] > cap:
                coll_pool = coll_pool[-cap:]

        model.train(); opt.zero_grad()
        pred = model(tr_in)
        loss_data = ((pred - tr_tgt)**2).mean()
        lam = config["lambda_phys_max"]*(1.0 - math.exp(-epoch/config["ramp_tau"]))
        L_eq, L_co = physics_residuals(model, config["n_collocation"], config,
                                       extra_pts=(coll_pool if coll_pool.numel() else None))
        loss_phys = normalized_physics(L_eq, L_co, config, ema)
        total = config["w_data"]*loss_data + lam*loss_phys
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()

        if epoch % 100 == 0:
            print(f"  [{frac*100:>4.0f}%] epoch {epoch:4d}  "
                  f"data={loss_data.item():.3e}  phys={float(loss_phys):.3e}  "
                  f"R2(exx)={frames[-1][2]['e_xx']:+.3f}")
    print(f"  [{frac*100:>4.0f}%] done  {time.time()-t0:.1f}s  {len(frames)} frames")
    return frames

CACHE = OUT/"snapshots.npz"
runs = {}
if CACHE.exists():
    print("loading cached snapshots:", CACHE)
    z = np.load(CACHE, allow_pickle=True)
    runs = z["runs"].item()
else:
    for f in FRACS:
        print(f"training {f*100:.0f}% ...")
        runs[f] = train_capture(f)
    np.savez_compressed(CACHE, runs=np.array(runs, dtype=object))
    print("cached ->", CACHE)

# frames are on the same epoch schedule for every fraction
epoch_list = [e for (e, _, _) in runs[FRACS[0]]]
nF = len(epoch_list)

# ============================ render one animation per component =========
PRETTY = {"e_xx": r"$\varepsilon_{xx}$", "e_yy": r"$\varepsilon_{yy}$", "e_xy": r"$\varepsilon_{xy}$"}

def render(comp):
    gt_disp = np.where(valid, gt_np[comp], np.nan)
    vmax = float(np.nanpercentile(np.abs(gt_np[comp][valid]), 98))
    vmin = -vmax
    kw = dict(cmap="RdBu", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), constrained_layout=True)
    axes[0].imshow(gt_disp, **kw)
    axes[0].set_title("Ground Truth", fontweight="bold")
    ims = []
    for j, frac in enumerate(FRACS):
        im = axes[j+1].imshow(runs[frac][0][1][comp], **kw)
        ims.append(im)
        axes[j+1].set_title(f"{frac*100:.0f}% data")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(ims[0], ax=axes, shrink=0.85, fraction=0.046, pad=0.02)
    cb.set_label(comp)
    sup = fig.suptitle("", fontsize=15, fontweight="bold")

    def update(k):
        kk = min(k, nF - 1)
        ep = epoch_list[kk]
        for j, frac in enumerate(FRACS):
            _, maps, r2 = runs[frac][kk]
            ims[j].set_data(maps[comp])
            axes[j+1].set_title(f"{frac*100:.0f}% data   $R^2$={r2[comp]:+.3f}")
        sup.set_text(f"{PRETTY[comp]}   —   training epoch {ep:d}"
                     + ("      (converged)" if k >= nF else ""))
        return ims

    total_frames = nF + HOLD_FRAMES
    mp4 = OUT/f"evolution_{comp}.mp4"
    try:
        w = FFMpegWriter(fps=FPS, bitrate=3000, metadata=dict(artist="pinns-4dstem"))
        with w.saving(fig, str(mp4), dpi=110):
            for k in range(total_frames):
                update(k); w.grab_frame()
        print("saved", mp4)
    except Exception as e:
        print("ffmpeg writer failed:", e)

    gif = OUT/f"evolution_{comp}.gif"
    gw = PillowWriter(fps=FPS)
    with gw.saving(fig, str(gif), dpi=75):
        for k in range(total_frames):
            update(k); gw.grab_frame()
    print("saved", gif)

    update(nF)  # final state
    fig.savefig(OUT/f"evolution_{comp}_final.png", dpi=130)
    plt.close(fig)

for comp in ["e_xx", "e_yy", "e_xy"]:
    print("rendering", comp)
    render(comp)

print("\nALL DONE ->", OUT.resolve())
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size/1e6:6.2f} MB")
