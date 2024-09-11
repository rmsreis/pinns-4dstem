# Physics-Informed Neural Networks (PINNs) for Strain Field Prediction

This repository contains Jupyter notebooks demonstrating the use of Physics-Informed Neural Networks (PINNs) for predicting strain fields in materials science applications.

## Contents

1. `pinns-strain-simple.ipynb`: A simplified implementation of PINNs for strain field prediction.
2. `pinns-strain-complex.ipynb`: A more complex implementation with additional features and visualizations.

## Features

- Implementation of a PINN model for strain field prediction
- Training process with loss visualization
- Comparison of PINN predictions with ground truth
- Animation of the training process
- Demonstration of PINN advantage with sparse data
- Metrics calculation for model evaluation

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- SciPy

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pinns-strain-prediction.git
   ```

2. Install the required dependencies:
   ```
   pip install torch numpy matplotlib scipy
   ```

3. Open and run the Jupyter notebooks:
   ```
   jupyter notebook
   ```

## Key Functions

- `StrainPINN`: Neural network model for strain prediction
- `strain_components`: Calculates strain components from the PINN output
- `pinn_loss`: Defines the loss function for training the PINN
- `train_pinn`: Trains the PINN model
- `visualize_results`: Creates visualizations of the PINN predictions
- `create_animation`: Generates an animation of the training process
- `demonstrate_sparse_data_advantage`: Shows the advantage of PINNs with limited data

## Results

The notebooks demonstrate the ability of PINNs to accurately predict strain fields, even with limited data. Visualizations and metrics are provided to evaluate the model's performance.

## Contributing

Contributions to improve the implementation or extend the functionality are welcome. Please feel free to submit pull requests or open issues for discussion.

## License

[MIT License](LICENSE)