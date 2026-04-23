# Physics-Informed Neural Networks (PINNs) for 4D-STEM Strain Mapping

This repository contains code and Jupyter notebooks demonstrating the use of Physics-Informed Neural Networks (PINNs) applied to 4D-STEM strain mapping in materials science applications.

## Quick Start

1. Create the conda environment (recommended):
   ```bash
   conda env create -f environment-pinns.yml
   conda activate pinns
   ```

2. (Optional) Register the environment as a Jupyter kernel so notebooks can use it:
   ```bash
   python -m ipykernel install --user --name=pinns --display-name "pinns (Python 3.10)"
   ```

3. Run the main notebook (headless execution to HTML):
   ```bash
   jupyter nbconvert --to html pinns-strain-05.ipynb --output pinns-strain-05.html --execute
   ```

## Files of Interest

- `pinns-strain-05.ipynb` — Main notebook used for experiments and visualizations.
- `pinns_dpc_pn_sota.py` — State-of-the-art DPC PN-junction modeling.
- `environment-pinns.yml` — Conda environment template (recommended for development).
- `requirements.txt` — Pip-style dependency list.

*Legacy files (in `_legacy/`):*
- `environment-pinned.yml` — Pinned export for reproducible fallback.
- `requirements-locked.txt` — Locked pip requirements.

## Features

- Implementation of PINN models for 4D-STEM strain field prediction.
- Training processes with loss visualization.
- Comparison of PINN predictions with ground truth.
- Metrics calculation for model evaluation.
- Application in specific physical representations (e.g., PN-junctions).

## Requirements

- Python 3.9+ / 3.10
- PyTorch (Platform-specific build recommended: https://pytorch.org/get-started/locally/)
- NumPy, Matplotlib, SciPy, Jupyter

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rmsreis/pinns-4dstem.git
   ```

2. Set up the environment as detailed in the Quick Start section.

3. Open and run the Jupyter notebooks:
   ```bash
   jupyter notebook
   ```

## Reproducibility Notes

- For PyTorch, prefer installing the platform-specific build depending on if using CUDA or Apple Metal (`mps`).
- Make sure to activate the specified `pinns` conda environment before running experiments.
