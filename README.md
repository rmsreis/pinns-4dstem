# Physics-Informed Neural Networks (PINNs) for 4D-STEM Strain Mapping


## Overview

We develop a PINN architecture that embeds elastic equilibrium and Saint-Venant compatibility into the training loss via automatic differentiation. The backbone is a sine-activated residual network (SIREN) with residual-based adaptive collocation refinement (RAR). Two Bayesian variants (MC Dropout and mean-field variational inference) provide per-pixel epistemic uncertainty maps.

### Key results (180 × 400 px experimental 4D-STEM strain map)

Specimen: PbGeSnSe₁.₅Te₁.₅, a domain-structured IV–VI high-entropy thermoelectric (Liu et al.,
*J. Am. Chem. Soc.* 2024, 146, 12620–12635); the chevron strain bands are ferroelastic domains.

| Sampling | Train pts | R² (ε_xx) | RMSE (ε_xx) | MAE (ε_xx) | R² avg |
|----------|-----------|-----------|-------------|------------|--------|
|  1 %     |   720     |   0.44    |   3.9×10⁻² |  2.9×10⁻² |  0.46  |
|  5 %     | 3,600     |   0.72    |   2.8×10⁻² |  2.0×10⁻² |  0.71  |
| 10 %     | 7,200     |   0.80    |   2.3×10⁻² |  1.7×10⁻² |  0.78  |
| 25 %     | 18,000    |   0.84    |   2.1×10⁻² |  1.5×10⁻² |  0.83  |
| 50 %     | 36,000    |   0.85    |   2.0×10⁻² |  1.5×10⁻² |  0.84  |
| 75 %     | 54,000    |   0.86    |   2.0×10⁻² |  1.5×10⁻² |  0.85  |

## Quick Start

1. Create the conda environment:
   ```bash
   conda env create -f environment-pinns.yml
   conda activate pinns
   ```

2. (Optional) Register as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=pinns --display-name "pinns (Python 3.10)"
   ```

3. Run the main notebook:
   ```bash
   jupyter notebook pinns-strain-sota-adaptive-2.ipynb
   ```

   To export paper figures (runs Cell 24 after training):
   ```bash
   jupyter nbconvert --to notebook --execute pinns-strain-sota-adaptive-2.ipynb \
       --output pinns-strain-sota-adaptive-2-executed.ipynb
   ```

4. (Optional) Explore the trained models interactively in 3D with `napari` — layer stack, field surfaces for all strain/rotation components, first-layer gratings, and training evolution (both the output field and, jointly, per-layer activity + physics residuals) across sampling fractions (1–75%), each in its own napari window:
   ```bash
   jupyter notebook pinn-viz3d.ipynb
   ```

## Repository layout

```
pinns-4dstem/
├── pinns-strain-sota-adaptive-2.ipynb   ← main notebook (training + paper figures)
├── pinn-viz3d.ipynb                     ← interactive 3D explainability views (napari)
├── data/
│   ├── strain_exx.npy                   ← 180×400 experimental ε_xx map
│   ├── strain_eyy.npy
│   └── strain_exy.npy
├── outputs/
│   ├── sota_adaptive-2/                 ← generated figures and metric CSVs
│   └── viz3d/                           ← cached per-fraction checkpoints + training snapshots
├── environment-pinns.yml
└── requirements.txt
```

`data/` and `outputs/` are git-ignored (regenerated locally / too large for git). The manuscript
(`paper/`: `main.tex`, `reference.bib`, `Fig/`) is also kept local only and is not part of this
repository — it isn't tracked in git history and won't appear in a fresh clone.

## Architecture at a glance

- **Backbone**: SIREN (input sine layer + 4 hidden sine layers, width 128, ω₀=30, skip projection into the 3rd hidden layer) — 83,460 parameters. ω₀=1 underfits badly (R²≤0.05 at any sampling fraction); ω₀=30 is required.
- **Loss**: data MSE + physics loss (elastic equilibrium + Saint-Venant compatibility), each scale-normalised against a frozen EMA of its own residual (frozen after 200 steps) and combined with an exponential ramp on the physics weight, λ(t)=1−e^(−t/500)
- **Adaptive refinement (RAR)**: from epoch 2,000, every 500 epochs add the top-10% highest-residual collocation points (pool capped at 5×)
- **Bayesian variants**: MC Dropout (p=0.05 on hidden activations only, T=150) and mean-field variational inference (prior matched to the SIREN init scale, T=150)
- **Training**: Adam, lr=1×10⁻³, step decay 0.95/500 epochs, gradient clipping at 1.0, early stopping (patience 300), max 5,000 epochs

## Hardware

Tested on Apple Silicon (MPS) — ≈35 ms/epoch, 13–26 s total training per sampling fraction (early
stopping terminates most runs well under the 5,000-epoch budget). Inference over the full
72,000-pixel grid: < 1 s. The full six-fraction study, ablations, classical baselines and both
Bayesian variants complete in ~10 minutes on a laptop.

## Requirements

- Python 3.10
- PyTorch (MPS / CUDA / CPU)
- NumPy, Matplotlib, SciPy, tqdm, pandas, Jupyter

<!-- ## Citation

```bibtex
@article{dosreis2026pinns4dstem,
  title   = {Physics-Informed Neural Networks for Sparse Strain-Field Reconstruction in 4D-STEM},
  author  = {dos Reis, Roberto and dos Santos, Gabriel T. and Liu, Yukun and Hu, Xiaobing and Dravid, Vinayak P.},
  journal = {Microscopy and Microanalysis},
  year    = {2026},
  doi     = {DOI HERE}
}
``` -->
