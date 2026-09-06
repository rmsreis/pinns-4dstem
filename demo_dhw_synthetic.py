# -*- coding: utf-8 -*-
"""
End-to-end synthetic validation of the differentiable DHW forward model and the
regularised inversion (dynscatt/dhw.py).

Pipeline per test case:
  1. build a known strain field eps_true(x,z)
  2. forward-simulate the systematic-row qx-plot, add shot noise -> I_exp
  3. L-curve sweep over the Tikhonov weight lambda
  4. reconstruct eps(x,z) at the corner lambda from a zero start
  5. score reconstruction vs ground truth, accounting for the known
     mid-plane mirror ambiguity  I[eps(x,z)] = I[eps(x, t-z)]

Figures -> outputs/dynscatt/ .

Run:  /Users/robertoreis/anaconda3/envs/pinns/bin/python demo_dhw_synthetic.py
"""
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynscatt.dhw import Geometry, simulate, reconstruct, l_curve, lcurve_corner
from dynscatt.phantoms import (make_grid, inclined_layer, precipitate,
                               add_poisson_noise)

torch.manual_seed(0); np.random.seed(0)
DEV = "cpu"                       # small problem; CPU is deterministic and fast
OUT = Path("outputs/dynscatt"); OUT.mkdir(parents=True, exist_ok=True)


def pearson(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def midplane_mirrors(field):
    """The 4 strain fields eps(x,z) that give the same qx-plot (paper Fig. 3a):
    flip {neither, left half, right half, both} about z = t/2.  field: [x, z]."""
    nx = field.shape[0]
    mid = nx // 2
    flipz = torch.flip(field, dims=[1])
    variants = []
    for L, R in [(False, False), (True, False), (False, True), (True, True)]:
        v = field.clone()
        if L:
            v[:mid] = flipz[:mid]
        if R:
            v[mid:] = flipz[mid:]
        variants.append(v)
    return variants


def score(rec, true):
    """best of rec vs the 4 midplane-mirror equivalents of `true`.
    Returns (corr, rmse, variant_index in 0..3)."""
    best = (-2.0, None, 0)
    for k, tv in enumerate(midplane_mirrors(true)):
        c = pearson(rec, tv)
        if c > best[0]:
            best = (c, float((rec - tv).pow(2).mean().sqrt()), k)
    return best


def run_case(name, geom, eps_true, x_fine, z_fine,
             nx_probe=48, nv=26, nw=13, counts=2.0e3):
    t0 = time.time()
    x_probe = torch.linspace(float(x_fine[0]), float(x_fine[-1]), nx_probe,
                             dtype=torch.float64)

    # ---- forward: ground-truth qx-plot + shot noise ----
    with torch.no_grad():
        I_clean, q, W = simulate(eps_true, x_fine, z_fine, geom, x_probe)
    I_exp = add_poisson_noise(I_clean, W, counts_per_px=counts, seed=1)

    # ---- reconstruction grid (coarse, as in the paper) ----
    xg = torch.linspace(float(x_fine[0]), float(x_fine[-1]), nv, dtype=torch.float64)
    zg = torch.linspace(0.0, geom.thickness_nm, nw, dtype=torch.float64)

    SMOOTH = 0.05
    # ---- L-curve ----
    lams = np.logspace(0, 6, 11)
    lc = l_curve(I_exp, geom, x_probe, xg, zg, lams, n_iter=35, smooth=SMOOTH)
    R = [r for _, r, _, _ in lc]; S = [s for _, _, s, _ in lc]
    k = lcurve_corner(lams, R, S)
    lam_star = lams[k]
    print(f"[{name}] L-curve corner: lambda = {lam_star:.3e}")

    # ---- final reconstruction at corner lambda ----
    eps_rec, hist = reconstruct(I_exp, geom, x_probe, xg, zg, lam_star,
                                n_iter=140, verbose=True, smooth=SMOOTH)

    # score against ground truth sampled on the recon grid (best of 4 midplane mirrors)
    true_on_rec = torch.nn.functional.interpolate(
        eps_true[None, None], size=(nv, nw), mode="bilinear", align_corners=True)[0, 0]
    corr, rmse, vk = score(eps_rec, true_on_rec)
    vlabel = ["no flip", "left half flipped", "right half flipped", "both halves flipped"][vk]
    with torch.no_grad():
        I_rec, _, _ = simulate(eps_rec, xg, zg, geom, x_probe)
    print(f"[{name}] corr={corr:.3f}  rmse={rmse:.2e}  "
          f"best-match GT variant: {vlabel}  ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------------- figure
    gt_variant = midplane_mirrors(true_on_rec)[vk]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    ext_s = [float(x_fine[0]), float(x_fine[-1]), geom.thickness_nm, 0.0]
    vmax = float(eps_true.abs().max())
    for a, (arr, ttl) in zip(
        ax[0],
        [(eps_true, "true  $\\epsilon(x,z)$"),
         (gt_variant, f"true, {vlabel}\n(best-matching midplane mirror)"),
         (eps_rec, "reconstructed")],
    ):
        im = a.imshow(arr.T.cpu(), extent=ext_s, aspect="auto", cmap="RdBu",
                      vmin=-vmax, vmax=vmax)
        a.set_title(ttl); a.set_xlabel("x (nm)"); a.set_ylabel("z (nm)")
        plt.colorbar(im, ax=a, fraction=0.046)

    ext_q = [float(q[0]), float(q[-1]), float(x_probe[-1]), float(x_probe[0])]
    for a, (arr, ttl) in zip(
        ax[1],
        [(I_exp, "$I_{exp}(x,q)$  (noisy)"),
         (I_rec, "$I_{sim}$ for reconstruction"),
         (None, "L-curve")],
    ):
        if arr is None:
            a.loglog(R, S, "-o", ms=4)
            a.loglog(R[k], S[k], "s", ms=11, mfc="none", mec="r")
            a.set_xlabel("$\\|W(NI_{sim}-I_{exp})\\|^2$"); a.set_ylabel("$\\|\\epsilon\\|^2$")
            a.set_title("L-curve"); a.grid(alpha=0.3, which="both")
        else:
            im = a.imshow(arr.cpu(), extent=ext_q, aspect="auto", cmap="inferno")
            a.set_title(ttl); a.set_xlabel("q (1/nm)"); a.set_ylabel("x (nm)")
            plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{name}   corr={corr:.3f}  RMSE={rmse:.2e}  "
                 f"lambda*={lam_star:.2g}  GT mirror: {vlabel}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / f"synthetic_{name}.png", dpi=130)
    plt.close(fig)
    print(f"[{name}] saved {OUT / f'synthetic_{name}.png'}\n")
    return dict(name=name, corr=corr, rmse=rmse, lam=lam_star, mirror=vlabel)


if __name__ == "__main__":
    results = []

    # ----- case 1: inclined layer (GaN {01-12}-like row, 200 kV, t=115 nm) -----
    geom1 = Geometry(voltage_kV=200.0, g1_inv_nm=5.29, n_beams_half=3,   # GaN {01-12}
                     thickness_nm=115.0, theta_max_mrad=6.3, dz_nm=0.5,
                     xi_nm={1: 90.0, 2: 230.0, 3: 480.0}, absorption_ratio=0.06,
                     n_ktilt=27, dq_inv_nm=0.1)
    xf, zf = make_grid(x_half_nm=70.0, z_thickness_nm=geom1.thickness_nm, nx=140, nz=120)
    eps1 = inclined_layer(xf, zf, amp=-5e-3, width_nm=5.0, angle_deg=43.0,
                          x_at_top_nm=-25.0)
    results.append(run_case("inclined_layer", geom1, eps1, xf, zf,
                            nx_probe=48, nv=24, nw=13, counts=3.0e3))

    # ----- case 2: spherical precipitate (GaAs {220}, 300 kV, t=95 nm) -----
    geom2 = Geometry(voltage_kV=300.0, g1_inv_nm=5.0018, n_beams_half=3,  # GaAs {220}
                     thickness_nm=95.0, theta_max_mrad=4.3, dz_nm=0.5,
                     xi_nm={1: 76.0, 2: 210.0, 3: 460.0}, absorption_ratio=0.05,
                     n_ktilt=27, dq_inv_nm=0.1)
    xf2, zf2 = make_grid(x_half_nm=22.0, z_thickness_nm=geom2.thickness_nm, nx=120, nz=120)
    eps2 = precipitate(xf2, zf2, eps_star=1.2e-2, radius_nm=6.0, depth_nm=25.0)
    results.append(run_case("precipitate", geom2, eps2, xf2, zf2,
                            nx_probe=44, nv=26, nw=15, counts=4.0e3))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['name']:16s} corr={r['corr']:.3f}  rmse={r['rmse']:.2e}  "
              f"lambda*={r['lam']:.2g}  GT mirror: {r['mirror']}")
