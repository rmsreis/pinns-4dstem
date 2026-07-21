# Physics-Informed Neural Networks (PINNs) for 4D-STEM Strain Mapping


## Overview

We develop a PINN architecture that embeds elastic equilibrium and Saint-Venant compatibility into the training loss via automatic differentiation. The backbone is a sine-activated residual network (SIREN) with residual-based adaptive collocation refinement (RAR). Two Bayesian variants (MC Dropout and mean-field variational inference) provide per-pixel epistemic uncertainty maps.

### Key results (180 × 400 px experimental 4D-STEM strain map)

| Sampling | Train pts | R² (ε_xx) | RMSE (ε_xx) | MAE (ε_xx) | R² avg |
|----------|-----------|-----------|-------------|------------|--------|
|  1 %     |   720     |   0.36    |   4.2×10⁻² |  3.0×10⁻² |  0.38  |
|  5 %     | 3,600     |   0.74    |   2.7×10⁻² |  2.0×10⁻² |  0.73  |
| 10 %     | 7,200     |   0.75    |   2.6×10⁻² |  2.0×10⁻² |  0.73  |
| 25 %     | 18,000    |   0.81    |   2.3×10⁻² |  1.7×10⁻² |  0.79  |
| 50 %     | 36,000    |   0.78    |   2.4×10⁻² |  1.8×10⁻² |  0.77  |
| 75 %     | 54,000    |   0.85    |   2.1×10⁻² |  1.5×10⁻² |  0.83  |

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

4. (Optional) Explore the trained models interactively in 3D — layer stack, field surfaces for all strain/rotation components, first-layer gratings, and training evolution across sampling fractions (1–75%), each in a napari window:
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
├── outputs/sota_adaptive-2/             ← generated figures and metric CSVs
├── paper/
│   ├── main.tex                         ← manuscript (Microscopy & Microanalysis template)
│   ├── reference.bib
│   └── Fig/                             ← publication-ready figures (300 dpi)
├── environment-pinns.yml
└── requirements.txt
```

## Architecture at a glance

- **Backbone**: SIREN (6 hidden layers, width 128, ω₀=1.0, skip connections) — 29,396 parameters
- **Loss**: data MSE + physics loss (elastic equilibrium + Saint-Venant compatibility) with exponential ramp on physics weight
- **Adaptive refinement (RAR)**: from epoch 2,000, every 500 epochs add the top-10% highest-residual collocation points
- **Bayesian variants**: MC Dropout (p=0.1, T=150) and mean-field variational inference (T=150)
- **Training**: Adam, lr=1×10⁻³, step decay 0.95/500 epochs, early stopping (patience 400), max 5,000 epochs

## Hardware

Tested on Apple Silicon (MPS) — ~13 ms/epoch. Inference over the full 72,000-pixel grid: < 1 s.

## Requirements

- Python 3.10
- PyTorch (MPS / CUDA / CPU)
- NumPy, Matplotlib, SciPy, tqdm, pandas, Jupyter

## Citation

```bibtex
@article{dosreis2026pinns4dstem,
  title   = {Physics-Informed Neural Networks for Sparse Strain-Field Reconstruction in 4D-STEM},
  author  = {dos Reis, Roberto and dos Santos, Gabriel T. and Liu, Yukun and Dravid, Vinayak P.},
  journal = {Microscopy and Microanalysis},
  year    = {2026},
  doi     = {DOI HERE}
}
```

## Acknowledgements

Supported in part by NSF \#1636933 and \#1920920. Facilities at the Northwestern NUANCE Center.
