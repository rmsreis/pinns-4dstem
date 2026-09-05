# -*- coding: utf-8 -*-
"""
Differentiable Darwin-Howie-Whelan (DHW) forward model + regularised inversion
for depth-resolved strain from systematic-row SCBED / qx-plots.

Implements the method of Niermann, Niermann, Song & Ophus,
"3D Strain Field Reconstruction by Inversion of Dynamical Scattering"
(arXiv:2508.18897v2), with the "simple elastic + constant absorption"
structure-factor model requested for the first bring-up:

    structure factors are set from extinction distances xi_n (nm) via
        U_n' = pi / (lambda * xi_n)          [1/nm^2]
        U_n" = absorption_ratio * U_n'       (uniform anomalous absorption)
    no mean-inner-potential refraction term (folded into k0).

Everything is torch and autograd-safe: the qx-plot is a differentiable
function of the strain grid, so the inversion is plain L-BFGS on
    L[eps] = R[eps] + lambda * S[eps]
with  R = sum_{x,q} [ W(q) (N * I_sim - I_exp) ]^2 ,  S = sum eps^2 .

Systematic-row, zeroth-order Laue zone, column approximation. strain is
eps(x, z) = d u_x / d z  (dimensionless).  Units: nm and 1/nm throughout.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  physical constants / helpers
# --------------------------------------------------------------------------- #
def electron_wavelength_nm(voltage_kV: float) -> float:
    """Relativistic electron wavelength in nm."""
    V = voltage_kV * 1e3
    h, m, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    lam_m = h / math.sqrt(2 * m * e * V * (1 + e * V / (2 * m * c * c)))
    return lam_m * 1e9


@dataclass
class Geometry:
    """Acquisition / crystal geometry for one systematic row."""
    voltage_kV: float = 300.0
    g1_inv_nm: float = 5.0018          # |g| of the first row reflection (1/nm). GaAs {220}: a/sqrt(8)
    n_beams_half: int = 4             # beams are n*g1 for n = -half .. +half
    thickness_nm: float = 95.0
    theta_max_mrad: float = 4.3       # semi-convergence half angle
    residual_tilt_mrad: float = 0.0
    dz_nm: float = 0.2               # RK integration / propagation step

    # structure factor model (simple)
    xi_nm: dict = field(default_factory=lambda: {1: 76.0})   # extinction dist per |n|
    xi_falloff_B: float = 2.0        # if xi for |n| missing: U_n = U_1 exp(-B (n^2-1) s1^2)
    absorption_ratio: float = 0.05

    # qx-plot sampling
    q_min_inv_nm: float | None = None
    q_max_inv_nm: float | None = None
    dq_inv_nm: float = 0.1
    n_ktilt: int = 41               # incident-tilt samples across the cone
    aperture_beta_nm: float = 20.0   # sigmoid edge softening for w(q)

    def __post_init__(self):
        self.lam = electron_wavelength_nm(self.voltage_kV)
        self.k0 = 1.0 / self.lam
        span = (self.n_beams_half + 1.2) * self.g1_inv_nm
        if self.q_min_inv_nm is None:
            self.q_min_inv_nm = -span
        if self.q_max_inv_nm is None:
            self.q_max_inv_nm = span
        # SCBED assumption: convergence angle below the Bragg angle so the
        # systematic-row disks do not overlap (the qx assembly places each disk
        # independently).
        theta_bragg_mrad = 0.5 * self.lam * self.g1_inv_nm * 1e3
        if self.theta_max_mrad >= theta_bragg_mrad:
            import warnings
            warnings.warn(
                f"theta_max ({self.theta_max_mrad:.2f} mrad) >= Bragg angle "
                f"({theta_bragg_mrad:.2f} mrad): row disks overlap, the qx-plot "
                f"assembly is only approximate.", stacklevel=2)

    # ------------------------------------------------------------------ #
    @property
    def n_index(self) -> np.ndarray:
        return np.arange(-self.n_beams_half, self.n_beams_half + 1)

    @property
    def g_vec_inv_nm(self) -> np.ndarray:
        return self.n_index * self.g1_inv_nm

    @property
    def ktilt_max_inv_nm(self) -> float:
        return self.k0 * self.theta_max_mrad * 1e-3

    def structure_factor_matrix(self, device=None, dtype=torch.complex128) -> torch.Tensor:
        """(B,B) complex Toeplitz U_{n-m} in 1/nm^2 (elastic + constant absorption)."""
        s1 = 0.5 * self.g1_inv_nm
        U1p = math.pi / (self.lam * self.xi_nm.get(1, 76.0))
        nmax = 2 * self.n_beams_half
        Un = {}
        for d in range(0, nmax + 1):
            if d == 0:
                Up = 0.0                                   # no MIP refraction term
            elif d in self.xi_nm:
                Up = math.pi / (self.lam * self.xi_nm[d])
            else:
                Up = U1p * math.exp(-self.xi_falloff_B * (d * d - 1) * s1 * s1)
            Un[d] = complex(Up, self.absorption_ratio * (Up if d != 0 else U1p))
        B = len(self.n_index)
        U = torch.zeros(B, B, dtype=dtype)
        idx = self.n_index
        for i in range(B):
            for j in range(B):
                d = abs(int(idx[i] - idx[j]))
                U[i, j] = torch.tensor(Un[d], dtype=dtype)
        if device is not None:
            U = U.to(device)
        return U


# --------------------------------------------------------------------------- #
#  strain-grid sampling
# --------------------------------------------------------------------------- #
def sample_strain(eps_grid: torch.Tensor, x_grid_nm: torch.Tensor, z_grid_nm: torch.Tensor,
                  x_query_nm: torch.Tensor, z_query_nm: torch.Tensor) -> torch.Tensor:
    """Bilinear-interpolate eps_grid[v,w] defined on (x_grid_nm, z_grid_nm)
    at the outer product of (x_query_nm, z_query_nm).  Returns (Nx, Nz)."""
    # normalise queries to [-1, 1] for grid_sample (align_corners=True)
    def to_unit(q, g):
        return 2.0 * (q - g[0]) / (g[-1] - g[0]) - 1.0
    xs = to_unit(x_query_nm, x_grid_nm)          # (Nx,)
    zs = to_unit(z_query_nm, z_grid_nm)          # (Nz,)
    gx, gz = torch.meshgrid(xs, zs, indexing="ij")           # (Nx,Nz)
    # grid_sample expects (N,C,Hin,Win) input and (N,Hout,Wout,2) grid with (x,y)=(w,h)
    inp = eps_grid.transpose(0, 1)[None, None]               # (1,1, Nz(v->h? ), Nx)  -> treat rows=z, cols=x
    # eps_grid is [v (x), w (z)]; make input H=z, W=x
    inp = eps_grid.t()[None, None]                           # (1,1,Nz,Nx)
    grid = torch.stack([gx, gz], dim=-1)[None]               # (1,Nx,Nz,2) with last=(x_norm, z_norm)->(W,H)
    out = torch.nn.functional.grid_sample(inp, grid, mode="bilinear",
                                          align_corners=True, padding_mode="border")
    return out[0, 0]                                          # (Nx, Nz)


# --------------------------------------------------------------------------- #
#  DHW propagation  (piecewise-constant matrix exponential)
# --------------------------------------------------------------------------- #
def propagate(eps_grid: torch.Tensor, x_grid_nm: torch.Tensor, z_grid_nm: torch.Tensor,
              geom: Geometry, x_probe_nm: torch.Tensor, taylor_order: int = 6):
    """Integrate the DHW equations 0 -> t for every probe position and every
    incident tilt.  Returns disk rocking curves I_n(x, ktilt): (Nx, Nk, B).

    Propagation over one slice uses  phi <- exp(A dz) phi  evaluated by a
    truncated Taylor series applied directly to the state vector (matvecs only;
    ||A dz|| ~ 0.1 here, so order 6 is exact to ~1e-10 and far cheaper than a
    batched dense matrix_exp)."""
    device = eps_grid.device
    cdt = torch.complex64
    g = torch.as_tensor(geom.g_vec_inv_nm, dtype=torch.float64, device=device)      # (B,)
    B = g.numel()
    U = geom.structure_factor_matrix(device=device, dtype=cdt)                       # (B,B)

    kt = torch.linspace(-geom.ktilt_max_inv_nm, geom.ktilt_max_inv_nm,
                        geom.n_ktilt, dtype=torch.float64, device=device)            # (Nk,)
    s = -(g[None, :] * (g[None, :] + 2.0 * kt[:, None])) / (2.0 * geom.k0)           # (Nk,B)

    nz = int(round(geom.thickness_nm / geom.dz_nm))
    dz = geom.thickness_nm / nz
    z_mid = (torch.arange(nz, dtype=torch.float64, device=device) + 0.5) * dz        # (nz,)

    eps_xz = sample_strain(eps_grid.double(), x_grid_nm.double(), z_grid_nm.double(),
                           x_probe_nm.double(), z_mid).to(torch.float32)             # (Nx, nz)

    Nx = x_probe_nm.numel()
    n0 = geom.n_beams_half
    phi = torch.zeros(Nx, geom.n_ktilt, B, dtype=cdt, device=device)
    phi[..., n0] = 1.0

    two_pi_i = torch.tensor(2j * math.pi, dtype=cdt, device=device)
    coupl_T = ((1j * math.pi / geom.k0) * U).T.contiguous()                          # (B,B) for v @ coupl_T
    g_c = g.to(cdt)                                                                  # (B,)
    s_c = s.to(cdt)                                                                  # (Nk,B)

    for iz in range(nz):
        # diagonal part of A for this slice: (Nx,Nk,B)
        diag = two_pi_i * (s_c[None] + g_c[None, None] * eps_xz[:, iz][:, None, None])
        term = phi
        for m in range(1, taylor_order + 1):
            # (A dz) @ term  =  dz * ( diag * term  +  term @ coupl^T )
            term = (dz / m) * (diag * term + torch.matmul(term, coupl_T))
            phi = phi + term

    return (phi.abs() ** 2).float(), kt                                             # (Nx,Nk,B), (Nk,)


# --------------------------------------------------------------------------- #
#  qx-plot assembly
# --------------------------------------------------------------------------- #
def _aperture_w(u: torch.Tensor, geom: Geometry) -> torch.Tensor:
    kmax = geom.ktilt_max_inv_nm
    b = 1.0 / max(geom.dq_inv_nm, 1e-6) if geom.aperture_beta_nm <= 0 else geom.aperture_beta_nm
    return torch.sigmoid(b * (kmax - u)) * torch.sigmoid(b * (kmax + u))


def qx_plot(I_disk: torch.Tensor, kt: torch.Tensor, geom: Geometry):
    """Assemble I(x,q) = sum_n w(q-g_n) * interp_kt[ I_n(x, .) ](q - g_n - k0*Theta).
    Returns  I_xq (Nx, Nq),  q_grid (Nq,),  W_xq aperture weight (Nq,)."""
    device = I_disk.device
    q = torch.arange(geom.q_min_inv_nm, geom.q_max_inv_nm + 1e-9, geom.dq_inv_nm,
                     dtype=torch.float64, device=device)                            # (Nq,)
    g = torch.as_tensor(geom.g_vec_inv_nm, dtype=torch.float64, device=device)
    tilt = geom.k0 * geom.residual_tilt_mrad * 1e-3
    Nx = I_disk.shape[0]
    I_xq = torch.zeros(Nx, q.numel(), dtype=I_disk.dtype, device=device)
    W = torch.zeros(q.numel(), dtype=I_disk.dtype, device=device)

    kt0, kt1, nk = kt[0], kt[-1], kt.numel()
    for n in range(g.numel()):
        u = q - g[n]                                                               # offset from disk centre
        w = _aperture_w(u, geom)                                                    # (Nq,)
        # linear interpolation of I_disk[:, :, n] over kt at position (u - tilt)
        pos = (u - tilt - kt0) / (kt1 - kt0) * (nk - 1)
        lo = torch.clamp(pos.floor(), 0, nk - 2).long()
        frac = torch.clamp(pos - lo, 0.0, 1.0)
        col_lo = I_disk[:, lo, n]                                                   # (Nx,Nq)
        col_hi = I_disk[:, lo + 1, n]
        val = col_lo * (1 - frac)[None] + col_hi * frac[None]
        inside = ((u >= -geom.ktilt_max_inv_nm) & (u <= geom.ktilt_max_inv_nm)).to(I_disk.dtype)
        I_xq = I_xq + w[None] * inside[None] * val
        W = W + w * inside
    return I_xq, q, W


def simulate(eps_grid, x_grid_nm, z_grid_nm, geom: Geometry, x_probe_nm):
    """strain grid -> qx-plot.  Returns (I_xq, q_grid, W_xq)."""
    I_disk, kt = propagate(eps_grid, x_grid_nm, z_grid_nm, geom, x_probe_nm)
    return qx_plot(I_disk, kt, geom)


# --------------------------------------------------------------------------- #
#  inversion
# --------------------------------------------------------------------------- #
def _best_scale(I_sim, I_exp, W):
    w2 = (W[None] ** 2)
    num = (w2 * I_sim * I_exp).sum()
    den = (w2 * I_sim * I_sim).sum().clamp_min(1e-30)
    return (num / den).detach()


def reconstruct(I_exp, geom: Geometry, x_probe_nm, x_grid_nm, z_grid_nm,
                lam: float, n_iter: int = 60, init=None, verbose: bool = True,
                device=None, smooth: float = 0.0):
    """L-BFGS minimisation of  R + lam * S  over the strain grid eps[v,w].
        S = sum eps^2  +  smooth * sum |grad eps|^2
    `smooth`=0 reproduces the paper's plain Tikhonov term; a small positive
    value adds a first-difference smoothness penalty that suppresses the
    checkerboard null-space of the qx forward map.
    Returns (eps_grid_detached, history dict)."""
    device = device or I_exp.device
    Nv, Nw = x_grid_nm.numel(), z_grid_nm.numel()
    eps = torch.zeros(Nv, Nw, dtype=torch.float64, device=device, requires_grad=True) \
        if init is None else init.clone().detach().to(device).double().requires_grad_(True)

    opt = torch.optim.LBFGS([eps], lr=1.0, max_iter=n_iter, history_size=25,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-12,
                            tolerance_change=1e-14)
    hist = {"R": [], "S": [], "L": []}

    def closure():
        opt.zero_grad()
        I_sim, q, W = simulate(eps, x_grid_nm, z_grid_nm, geom, x_probe_nm)
        N = _best_scale(I_sim, I_exp, W)
        resid = W[None] * (N * I_sim - I_exp)
        R = (resid ** 2).sum()
        S = (eps ** 2).sum()
        if smooth > 0:
            S = S + smooth * ((eps[1:] - eps[:-1]) ** 2).sum() \
                  + smooth * ((eps[:, 1:] - eps[:, :-1]) ** 2).sum()
        L = R + lam * S
        L.backward()
        hist["R"].append(float(R)); hist["S"].append(float(S)); hist["L"].append(float(L))
        return L

    opt.step(closure)
    if verbose:
        print(f"    lam={lam:.3e}  R={hist['R'][-1]:.4e}  S={hist['S'][-1]:.4e}  "
              f"iters={len(hist['R'])}")
    return eps.detach(), hist


def l_curve(I_exp, geom, x_probe_nm, x_grid_nm, z_grid_nm, lams, n_iter=40, smooth=0.0):
    """Sweep lambda; return list of (lam, ||resid||^2, ||eps||^2, eps_grid)."""
    out = []
    for lam in lams:
        eps, h = reconstruct(I_exp, geom, x_probe_nm, x_grid_nm, z_grid_nm,
                             lam, n_iter=n_iter, verbose=True, smooth=smooth)
        out.append((lam, h["R"][-1], h["S"][-1], eps))
    return out


def lcurve_corner(lams, R, S):
    """L-curve corner = point of the normalised log-log curve closest to the
    ideal (min-residual, min-norm) origin.  Robust to a near-vertical
    under-regularised tail that defeats a pure max-curvature criterion."""
    x = np.log10(np.asarray(R, float) + 1e-300)
    y = np.log10(np.asarray(S, float) + 1e-300)
    x = (x - x.min()) / (x.ptp() + 1e-12)
    y = (y - y.min()) / (y.ptp() + 1e-12)
    return int(np.argmin(x ** 2 + y ** 2))
