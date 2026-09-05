# -*- coding: utf-8 -*-
"""Synthetic ground-truth strain fields eps(x,z) = d u_x / d z for validating
the DHW forward + inversion (mirrors the test cases in arXiv:2508.18897)."""
import numpy as np
import torch


def make_grid(x_half_nm, z_thickness_nm, nx, nz, device="cpu"):
    x = torch.linspace(-x_half_nm, x_half_nm, nx, dtype=torch.float64, device=device)
    z = torch.linspace(0.0, z_thickness_nm, nz, dtype=torch.float64, device=device)
    return x, z


def inclined_layer(x_nm, z_nm, amp=-5e-3, width_nm=5.0, angle_deg=43.0,
                   x_at_top_nm=-20.0, edge_nm=1.5):
    """A constant-strain slab of width `width_nm` whose centre line runs from
    x_at_top at z=0 to x_at_top + t*tan(angle) at z=t.  Smooth tanh edges."""
    X, Z = torch.meshgrid(x_nm, z_nm, indexing="ij")
    t = float(z_nm[-1])
    x_center = x_at_top_nm + Z * np.tan(np.deg2rad(angle_deg))
    d = (X - x_center).abs()
    prof = 0.5 * (torch.tanh((width_nm / 2 - d) / edge_nm) + 1.0)
    return amp * prof


def precipitate(x_nm, z_nm, eps_star=1.2e-2, radius_nm=6.0, depth_nm=25.0):
    """Hard-sphere (Eshelby-like) misfitting inclusion; slice through centre (y=0).
    Outside:  u_x = eps_star R^3 x / r^3      ->  eps = d u_x/dz = -3 eps_star R^3 x z / r^5
    Inside:   uniform dilatation, d u_x/dz = 0.
    Gives the four-lobe shear pattern of Fig. 1 in the paper."""
    X, Z = torch.meshgrid(x_nm, z_nm, indexing="ij")
    zc = Z - depth_nm
    r2 = X ** 2 + zc ** 2
    r = torch.sqrt(r2.clamp_min(1e-9))
    out = -3.0 * eps_star * radius_nm ** 3 * X * zc / r.clamp_min(1e-6) ** 5
    out = torch.where(r < radius_nm, torch.zeros_like(out), out)
    return out


def add_poisson_noise(I_xq, W_xq, counts_per_px=2.0e3, seed=0):
    """Scale the aperture-windowed qx-plot to a peak count level and resample
    with shot noise, then rescale back."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Iw = (I_xq * W_xq[None]).clamp_min(0.0)
    scale = counts_per_px / Iw.max().clamp_min(1e-30)
    noisy = torch.poisson(Iw.cpu() * float(scale), generator=g).to(I_xq) / scale
    return noisy
