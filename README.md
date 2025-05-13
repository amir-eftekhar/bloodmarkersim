# Wrist Blood Flow Simulation

This project implements a comprehensive simulation of blood flow in the wrist, incorporating Computational Fluid Dynamics (CFD), optical transport modeling, and machine learning for biomarker prediction.

## Components

1. **Computational Fluid Dynamics (CFD)**: Simulates blood flow in the wrist's arterial network using the Navier-Stokes equations.
2. **Optical Transport Modeling**: Uses Monte Carlo methods to simulate light propagation through tissue and blood.
3. **Machine Learning**: Predicts biomarkers from simulated optical signals.

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyTorch for machine learning components
- FEniCS or OpenFOAM for CFD (via Python wrappers)
- VTK for visualization
- scikit-learn for traditional ML algorithms

## Setup

```bash
# Clone the repository
git clone https://github.com/username/wristbandsim.git
cd wristbandsim

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete simulation pipeline
python main.py

# Run individual components
python src/cfd/blood_flow.py
python src/optical/monte_carlo.py
python src/ml/biomarker_prediction.py
```

## Project Structure

```
wristbandsim/
├── data/                   # Data files and simulation results
│   ├── anatomy/            # Anatomical models
│   ├── optical_properties/ # Optical properties of tissues
│   └── results/            # Simulation results
├── src/                    # Source code
│   ├── cfd/                # Computational Fluid Dynamics module
│   ├── optical/            # Optical transport module
│   ├── ml/                 # Machine Learning module
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks for analysis and visualization
├── tests/                  # Test cases
├── main.py                 # Main entry point
└── README.md               # Project documentation
```
