# dynscatt — depth-resolved strain by inversion of dynamical scattering

Differentiable implementation of the method of **Niermann, Niermann, Song & Ophus,
"3D Strain Field Reconstruction by Inversion of Dynamical Scattering"**
(arXiv:2508.18897v2), plus a Path-B bridge to the 2D PINN strain work in this repo.

## What is here

| file | purpose |
|---|---|
| `dhw.py` | systematic-row Darwin–Howie–Whelan forward model (`simulate`), qx-plot assembly, and regularised L-BFGS inversion (`reconstruct`, `l_curve`). Fully torch / autograd. |
| `phantoms.py` | synthetic ground-truth strain fields ε(x,z)=∂u_x/∂z: `inclined_layer`, `precipitate` (hard-sphere 4-lobe), `add_poisson_noise`. |
| `../demo_dhw_synthetic.py` | end-to-end validation: phantom → qx-plot + shot noise → L-curve → reconstruction → score (accounting for the mid-plane mirror ambiguity). Figures → `outputs/dynscatt/synthetic_*.png`. |
| `../pinn3d_strain.py` | **Path B**: elasticity-consistent 3D strain *model* built from the repo's 2D strain maps (see caveat). Figures + volume → `outputs/dynscatt/pinn3d_strain*`. |

## Physics model (first bring-up)

* systematic row, zeroth-order Laue zone, column approximation; beams `g = n·g1`.
* DHW ODE `∂φ_g/∂z = i2π[s_g + g·ε(x,z)]φ_g + i(π/k0)·Σ U_{g-g'} φ_g'`, integrated
  by a truncated-Taylor propagator (‖A·dz‖≈0.1, order 6 exact to ~1e-10).
* **structure factors** = "simple elastic + constant absorption": `U_n' = π/(λ·ξ_n)`
  from tabulated extinction distances, `U_n" = c·U_n'`. No mean-inner-potential term.
  Swap in Weickenmeier–Kohl / real electron scattering factors later.
* qx-plot `I(x,q) = Σ_n w(q-g_n)·I_n(x, q-g_n-k0Θ)`, sigmoid aperture `w`.
* inversion `L = R + λ·S`, `R = Σ[W(N·I_sim - I_exp)]²`, `S = Σε² + smooth·Σ|∇ε|²`,
  `N` closed-form, `λ` by L-curve corner (min normalised distance to the origin).

## Validation status (`demo_dhw_synthetic.py`, CPU)

| case | geometry | corr(ε_rec, ε_true) | note |
|---|---|---|---|
| spherical precipitate | GaAs {220}, 300 kV, t=95 nm | ~0.87 | 4-lobe shear recovered at the right depth; qx-plot match excellent |
| inclined layer | GaN {01-12}, 200 kV, t=115 nm | ~0.48 | qx-plot reproduced, but z-localisation lost — the **mid-plane mirror ambiguity** (paper Fig. 3); needs a crystallographic prior to resolve |

## Path B caveat (important)

`pinn3d_strain.py` does **not** recover depth information — the 2D shift-derived
strain maps do not contain it (that lives in the raw dynamical CBED intensities,
which are not in this repo). It produces the *minimum-energy* displacement field
whose thickness-averaged in-plane strain matches the maps and which satisfies 3D
equilibrium + traction-free surfaces + a columnar-domain prior. The z-axis scale
is set by an **assumed** foil thickness. Treat it as a physically self-consistent
model / hypothesis, not a measurement. The real depth reconstruction (Path A) needs
the raw 4D-STEM dataset acquired near a strong systematic-row condition.

Run with the `pinns` conda env python.
