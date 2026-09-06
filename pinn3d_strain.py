# -*- coding: utf-8 -*-
"""
Path B : an elasticity-consistent 3D strain field from the 2D (depth-projected)
strain maps.

This is NOT a depth measurement -- the maps do not contain depth information.
It is a regularised *model*: the minimum-energy displacement field u(x,y,z) whose
in-plane strain, averaged over the foil thickness, reproduces the measured
eps_xx, eps_yy, eps_xy(x,y), and which additionally satisfies

  * 3D linear-elastic equilibrium            div(sigma) = 0            (bulk)
  * traction-free foil surfaces              sigma_iz = 0  at z=0, z=t
  * compatibility                            (automatic: strain = sym grad u)
  * a columnar-domain prior                  d eps_inplane / d z ~ 0    (interior)

What the 3D field then adds over a trivial z-extrusion of the 2D map is
(i) the surface-relaxation boundary layers near z=0,t forced by the free surface,
and (ii) the out-of-plane components eps_zz, eps_xz, eps_yz implied by equilibrium.

Backbone: a 3D SIREN on displacement (same activation / omega_0 as the 2D paper),
strain by autograd.  Runs on CPU (MPS autograd is unreliable on this box).

Run:  /Users/robertoreis/anaconda3/envs/pinns/bin/python pinn3d_strain.py
"""
import math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = torch.device("cpu")
OUT = Path("outputs/dynscatt"); OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------- config ------------------------------------- #
CFG = dict(
    px_nm=17.9,                 # real-space pixel size of the strain map
    thickness_nm=100.0,         # ASSUMED foil thickness (sets the z scale; unknown from maps)
    nu=0.27, E=1.0,             # E cancels under scale-normalised residuals
    hidden=160, layers=5, omega_0=30.0,
    epochs=1300, lr=6e-4,
    n_data=3072, n_coll=2048, n_surf=1024, n_zquad=8,
    w_eq=1.0, w_surf=1.0, w_col=0.3, w_amp=0.5, phys_cap=0.06, lam_max=0.5,
    ramp_tau=200.0, phys_start_frac=0.5,
    out_nx=180, out_ny=90, out_nz=24,     # export volume (y downsampled for size)
    proj_H=90, proj_W=200,                # grid for the data-consistency projection check
    dtype=torch.float32, ckpt_every=200,
)
DT = CFG["dtype"]

# ----------------------------- data ------------------------------------- #
def load_maps():
    exx = np.load("data/strain_exx.npy").astype(np.float64)
    eyy = np.load("data/strain_eyy.npy").astype(np.float64)
    exy = np.load("data/strain_exy.npy").astype(np.float64)
    H, W = exx.shape
    px = CFG["px_nm"]
    # physical extents (nm); centre at 0
    Lx, Ly = (W - 1) * px, (H - 1) * px
    maps = {"e_xx": exx, "e_yy": eyy, "e_xy": exy}
    # per-component standardisation (as in the 2D paper)
    sc = {k: (float(np.nanmean(v)), float(np.nanstd(v)) + 1e-12) for k, v in maps.items()}
    return maps, sc, (H, W), (Lx, Ly)


# ----------------------------- SIREN (3D displacement) ------------------- #
class SirenLayer(nn.Module):
    def __init__(self, i, o, first=False, w0=30.0):
        super().__init__()
        self.w0 = w0; self.lin = nn.Linear(i, o)
        with torch.no_grad():
            if first:
                self.lin.weight.uniform_(-1 / i, 1 / i)
            else:
                b = math.sqrt(6 / i) / w0
                self.lin.weight.uniform_(-b, b)
    def forward(self, x):
        return torch.sin(self.w0 * self.lin(x))

class DispSIREN(nn.Module):
    """(x,y,z) in [-1,1]^3  ->  (u_x, u_y, u_z)  (arbitrary units; strain is scale-free)."""
    def __init__(self, hidden=256, layers=5, w0=30.0):
        super().__init__()
        self.first = SirenLayer(3, hidden, first=True, w0=w0)
        self.hid = nn.ModuleList([SirenLayer(hidden, hidden, w0=w0) for _ in range(layers - 2)])
        self.out = nn.Linear(hidden, 3)
        with torch.no_grad():
            b = math.sqrt(6 / hidden) / w0
            self.out.weight.uniform_(-b, b)
    def forward(self, X):
        h = self.first(X)
        for L in self.hid:
            h = L(h)
        return self.out(h)


# ----------------------------- strain / stress via autograd ------------- #
def strain_from_disp(model, X, scale_vec, create_graph=True):
    """X: (N,3) in [-1,1]^3. Returns (eps (N,3,3), X_leaf, u).
    create_graph=True keeps the graph so 2nd derivatives (equilibrium) work;
    set False for a cheap forward-only strain evaluation."""
    X = X.clone().requires_grad_(True)
    u = model(X)                                     # (N,3)
    g = []
    for i in range(3):
        gi = torch.autograd.grad(u[:, i], X, torch.ones_like(u[:, i]),
                                 create_graph=create_graph, retain_graph=True)[0]
        g.append(gi * scale_vec)                     # -> d u_i / d x_j  (physical)
    G = torch.stack(g, dim=1)                        # (N,3,3)  G[:,i,j] = du_i/dx_j
    eps = 0.5 * (G + G.transpose(1, 2))
    return eps, X, u


def strain_eval(model, X, scale_vec, chunk=16384):
    """Forward-only strain (no 2nd-order graph), chunked. Returns eps (N,3,3) detached."""
    outs = []
    for s in range(0, X.shape[0], chunk):
        e, _, _ = strain_from_disp(model, X[s:s + chunk], scale_vec, create_graph=False)
        outs.append(e.detach())
    return torch.cat(outs, 0)

def stress(eps, nu, E=1.0):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
    sig = 2 * mu * eps
    idx = torch.arange(3)
    sig[:, idx, idx] = sig[:, idx, idx] + lam * tr[:, None]
    return sig

def divergence(model, X, scale_vec, nu):
    """div(sigma)_i  at X ; needs 2nd derivatives of u."""
    eps, Xr, _ = strain_from_disp(model, X, scale_vec)
    sig = stress(eps, nu)                            # (N,3,3)
    div = []
    for i in range(3):
        di = 0.0
        for j in range(3):
            dsig = torch.autograd.grad(sig[:, i, j], Xr, torch.ones_like(sig[:, i, j]),
                                       create_graph=True, retain_graph=True)[0][:, j]
            di = di + dsig * scale_vec[j]
        div.append(di)
    return torch.stack(div, dim=1)                   # (N,3)


# ----------------------------- training -------------------------------- #
def train():
    maps, sc, (H, W), (Lx, Ly) = load_maps()
    t = CFG["thickness_nm"]
    Lz = t
    scale_vec = torch.tensor([2.0 / Lx, 2.0 / Ly, 2.0 / Lz], dtype=DT, device=DEV)

    # measured maps as tensors, standardised (network strain is compared in std units)
    m = {k: torch.tensor((maps[k] - sc[k][0]) / sc[k][1], dtype=DT, device=DEV)
         for k in maps}
    # pixel-centre coords in [-1,1]
    ys = torch.linspace(-1, 1, H, dtype=DT, device=DEV)
    xs = torch.linspace(-1, 1, W, dtype=DT, device=DEV)

    model = DispSIREN(CFG["hidden"], CFG["layers"], CFG["omega_0"]).to(DT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])

    # z Gauss-Legendre quadrature nodes/weights on [-1,1]
    zq, wq = np.polynomial.legendre.leggauss(CFG["n_zquad"])
    zq = torch.tensor(zq, dtype=DT, device=DEV)
    wq = torch.tensor(wq / 2.0, dtype=DT, device=DEV)   # average over z

    ema = {}
    def norm_res(name, val):
        """live EMA scale-normalisation -> each physics residual is O(1)."""
        v = float(val.detach())
        ema[name] = v if name not in ema else 0.9 * ema[name] + 0.1 * v
        return val / (ema[name] + 1e-12)

    # component index in eps tensor
    CI = {"e_xx": (0, 0), "e_yy": (1, 1), "e_xy": (0, 1)}
    sc_std = torch.tensor([sc["e_xx"][1], sc["e_yy"][1], sc["e_xy"][1]],
                          dtype=DT, device=DEV)

    hist = {"data": [], "eq": [], "surf": [], "col": []}
    t0 = time.time()
    for ep in range(CFG["epochs"] + 1):
        ema["n"] = ep
        opt.zero_grad()

        # ---- data term: z-averaged in-plane strain == measured map ----
        ii = torch.randint(0, H, (CFG["n_data"],), device=DEV)
        jj = torch.randint(0, W, (CFG["n_data"],), device=DEV)
        xd, yd = xs[jj], ys[ii]
        # build (n_data, n_zquad, 3) query
        Xz = torch.stack([xd[:, None].expand(-1, CFG["n_zquad"]),
                          yd[:, None].expand(-1, CFG["n_zquad"]),
                          zq[None].expand(CFG["n_data"], -1)], dim=-1).reshape(-1, 3)
        epsz, _, _ = strain_from_disp(model, Xz, scale_vec)
        epsz = epsz.reshape(CFG["n_data"], CFG["n_zquad"], 3, 3)
        # standardise network strain the same way the maps were
        loss_data = 0.0
        for c, (a, b) in CI.items():
            avg = (epsz[:, :, a, b] * wq[None]).sum(1)           # z-average (physical units)
            avg_std = avg / sc_std[list(CI).index(c)]            # to std units
            tgt = m[c][ii, jj]
            loss_data = loss_data + ((avg_std - tgt) ** 2).mean()
        loss_data = loss_data / 3.0

        # curriculum: fit the measured maps first (lam = 0), then ramp physics
        # in as a *small* bounded refinement.  u = 0 zeroes every physics
        # residual, so physics must stay a gentle nudge (lam caps at lam_max,
        # combined weight further capped at phys_cap x data loss) or it drags
        # the network to the trivial zero-strain solution.
        e0 = CFG["phys_start_frac"] * CFG["epochs"]
        lam = 0.0 if ep < e0 else CFG["lam_max"] * (1.0 - math.exp(-(ep - e0) / CFG["ramp_tau"]))

        # ---- equilibrium in the bulk ----
        Xc = torch.rand(CFG["n_coll"], 3, dtype=DT, device=DEV) * 2 - 1
        div = divergence(model, Xc, scale_vec, CFG["nu"])
        loss_eq = (div ** 2).mean()

        # ---- traction-free surfaces z = +-1 ----
        Xs = torch.rand(CFG["n_surf"], 3, dtype=DT, device=DEV) * 2 - 1
        Xs[:, 2] = torch.where(torch.rand(CFG["n_surf"], device=DEV) < 0.5, -1.0, 1.0)
        eps_s, _, _ = strain_from_disp(model, Xs, scale_vec)
        sig_s = stress(eps_s, CFG["nu"])
        loss_surf = (sig_s[:, 2, 0] ** 2 + sig_s[:, 2, 1] ** 2 + sig_s[:, 2, 2] ** 2).mean()

        # ---- columnar-domain prior: d eps_inplane / dz ~ 0 in the interior ----
        Xk = torch.rand(CFG["n_coll"], 3, dtype=DT, device=DEV) * 2 - 1
        Xk[:, 2] *= 0.7                                          # keep away from surfaces
        eps_k, Xkr, _ = strain_from_disp(model, Xk, scale_vec)
        dcol = 0.0
        for (a, b) in CI.values():
            d = torch.autograd.grad(eps_k[:, a, b], Xkr, torch.ones_like(eps_k[:, a, b]),
                                    create_graph=True, retain_graph=True)[0][:, 2]
            dcol = dcol + (d * scale_vec[2]) ** 2
        loss_col = dcol.mean() / 3.0

        # out-of-plane components (eps_zz, eps_xz, eps_yz) have no data term at
        # all, so nothing bounds their amplitude -- equilibrium only constrains
        # their *derivatives*, not their scale, and they can wander to
        # unphysically large values. A small always-on L2 penalty (reusing the
        # eps_k collocation points) keeps them near zero unless equilibrium /
        # the free surface actually need them nonzero.
        loss_amp = (eps_k[:, 2, 2] ** 2 + eps_k[:, 0, 2] ** 2 + eps_k[:, 1, 2] ** 2).mean()

        # scale-normalise each physics residual to O(1) (live EMA), then cap the
        # combined physics contribution at `phys_cap` x the current data loss so
        # it can never drive the network to the trivial u = 0 solution.
        phys_n = (CFG["w_eq"] * norm_res("eq", loss_eq)
                  + CFG["w_surf"] * norm_res("surf", loss_surf)
                  + CFG["w_col"] * norm_res("col", loss_col))
        cap = CFG["phys_cap"] * loss_data.detach()
        pscale = (cap / (phys_n.detach() + 1e-30)).clamp(max=1.0)
        loss = loss_data + CFG["w_amp"] * loss_amp + lam * pscale * phys_n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        hist["data"].append(float(loss_data)); hist["eq"].append(float(loss_eq))
        hist["surf"].append(float(loss_surf)); hist["col"].append(float(loss_col))
        if ep % 100 == 0:
            print(f"  ep{ep:4d}  data={loss_data:.3e}  eq={loss_eq:.2e}  "
                  f"surf={loss_surf:.2e}  col={loss_col:.2e}  amp={float(loss_amp):.2e}  "
                  f"pscale={float(pscale):.2e}  lam={lam:.2f}  ({time.time()-t0:.0f}s)")
        if ep % CFG["ckpt_every"] == 0 and ep > 0:
            torch.save({"state": model.state_dict(), "cfg": CFG, "epoch": ep},
                       OUT / "pinn3d_model.pt")
    torch.save({"state": model.state_dict(), "cfg": CFG, "epoch": CFG["epochs"]},
               OUT / "pinn3d_model.pt")
    return model, maps, sc, (H, W), (Lx, Ly), scale_vec, hist


def load_model_for_export():
    """Rebuild the model + context from the saved checkpoint (for export-only)."""
    ck = torch.load(OUT / "pinn3d_model.pt", map_location="cpu")
    CFG.update({k: v for k, v in ck["cfg"].items() if k in CFG})
    maps, sc, (H, W), (Lx, Ly) = load_maps()
    model = DispSIREN(CFG["hidden"], CFG["layers"], CFG["omega_0"]).to(DT)
    model.load_state_dict(ck["state"]); model.eval()
    sv = torch.tensor([2.0 / Lx, 2.0 / Ly, 2.0 / CFG["thickness_nm"]], dtype=DT)
    print(f"loaded checkpoint @ epoch {ck.get('epoch','?')}")
    return model, maps, sc, (H, W), (Lx, Ly), sv, {}


# ----------------------------- export + figures ------------------------- #
def export(model, maps, sc, HW, LL, scale_vec, hist):
    H, W = HW; Lx, Ly = LL; t = CFG["thickness_nm"]
    nx, ny, nz = CFG["out_nx"], CFG["out_ny"], CFG["out_nz"]
    xs = torch.linspace(-1, 1, nx, dtype=DT)
    ys = torch.linspace(-1, 1, ny, dtype=DT)
    zs = torch.linspace(-1, 1, nz, dtype=DT)
    Xg, Yg, Zg = torch.meshgrid(xs, ys, zs, indexing="ij")
    P = torch.stack([Xg, Yg, Zg], -1).reshape(-1, 3)
    eps = strain_eval(model, P, scale_vec).reshape(nx, ny, nz, 3, 3).numpy()

    # network strain ~ (physical strain - per-component mean); restore the mean
    # on the three components that had a data target. out-of-plane components
    # carry no measured offset (kept mean-free).
    off = {"e_xx": sc["e_xx"][0], "e_yy": sc["e_yy"][0], "e_xy": sc["e_xy"][0],
           "e_zz": 0.0, "e_xz": 0.0, "e_yz": 0.0}
    comps = {"e_xx": eps[..., 0, 0] + off["e_xx"],
             "e_yy": eps[..., 1, 1] + off["e_yy"],
             "e_xy": eps[..., 0, 1] + off["e_xy"],
             "e_zz": eps[..., 2, 2], "e_xz": eps[..., 0, 2], "e_yz": eps[..., 1, 2]}
    np.savez_compressed(OUT / "pinn3d_strain_volume.npz",
                        thickness_nm=t, px_nm=CFG["px_nm"],
                        note="Path B regularised model, not a depth measurement; "
                             "z axis scale set by assumed thickness", **comps)

    # projected reconstruction vs measured map (data-consistency check), coarse grid
    pW, pH = CFG["proj_W"], CFG["proj_H"]
    zq, wq = np.polynomial.legendre.leggauss(12)
    zt = torch.tensor(zq, dtype=DT); wt = torch.tensor(wq / 2, dtype=DT)
    xs2 = torch.linspace(-1, 1, pW, dtype=DT)
    ys2 = torch.linspace(-1, 1, pH, dtype=DT)
    Xm, Ym = torch.meshgrid(xs2, ys2, indexing="ij")
    proj = {"e_xx": None, "e_yy": None, "e_xy": None}
    acc = {c: torch.zeros(pW, pH, dtype=DT) for c in proj}
    for zz, ww in zip(zt, wt):
        Q = torch.stack([Xm, Ym, torch.full_like(Xm, float(zz))], -1).reshape(-1, 3)
        e = strain_eval(model, Q, scale_vec)
        for c, (a, b) in {"e_xx": (0, 0), "e_yy": (1, 1), "e_xy": (0, 1)}.items():
            acc[c] += e[:, a, b].reshape(pW, pH) * ww
    for c in proj:
        proj[c] = acc[c].T.numpy() + off[c]        # restore physical mean
    # measured maps downsampled to the same coarse grid for a like-for-like compare
    maps_c = {c: torch.nn.functional.interpolate(
        torch.tensor(maps[c])[None, None], size=(pH, pW), mode="bilinear",
        align_corners=True)[0, 0].numpy() for c in proj}

    # ---- figure ----
    fig, ax = plt.subplots(4, 3, figsize=(15, 15))
    for j, c in enumerate(["e_xx", "e_yy", "e_xy"]):
        vv = float(np.nanpercentile(np.abs(maps[c]), 98))
        ax[0, j].imshow(maps_c[c], cmap="RdBu", vmin=-vv, vmax=vv, origin="lower")
        ax[0, j].set_title(f"measured  {c}  (coarse)")
        ax[1, j].imshow(proj[c], cmap="RdBu", vmin=-vv, vmax=vv, origin="lower")
        r = np.corrcoef(proj[c].ravel(), maps_c[c].ravel())[0, 1]
        ax[1, j].set_title(f"model z-projection  (r={r:.3f})")
        mid = comps[c][:, ny // 2, :]              # x-z slice at mid-y
        vv2 = float(np.nanpercentile(np.abs(mid), 99))
        im = ax[2, j].imshow(mid.T, cmap="RdBu", vmin=-vv2, vmax=vv2,
                             aspect="auto", extent=[-Lx/2, Lx/2, t, 0])
        ax[2, j].set_title(f"{c}(x,z) at mid-y"); ax[2, j].set_xlabel("x (nm)"); ax[2, j].set_ylabel("z (nm)")
        plt.colorbar(im, ax=ax[2, j], fraction=0.046)
    for j, c in enumerate(["e_zz", "e_xz", "e_yz"]):
        mid = comps[c][:, ny // 2, :]
        vv2 = float(np.nanpercentile(np.abs(mid), 99)) or 1e-6
        im = ax[3, j].imshow(mid.T, cmap="PuOr", vmin=-vv2, vmax=vv2,
                             aspect="auto", extent=[-Lx/2, Lx/2, t, 0])
        ax[3, j].set_title(f"out-of-plane {c}(x,z) at mid-y  (equilibrium-implied)")
        ax[3, j].set_xlabel("x (nm)"); ax[3, j].set_ylabel("z (nm)")
        plt.colorbar(im, ax=ax[3, j], fraction=0.046)
    fig.suptitle("Path B: elasticity-consistent 3D strain model from 2D maps "
                 f"(assumed t = {t:.0f} nm)  -- a regularised model, not a depth measurement",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "pinn3d_strain.png", dpi=120)
    plt.close(fig)
    print("saved", OUT / "pinn3d_strain.png")
    print("saved", OUT / "pinn3d_strain_volume.npz")


if __name__ == "__main__":
    import sys
    if "--export-only" in sys.argv:           # render figures from the last checkpoint
        export(*load_model_for_export())
    else:
        export(*train())
