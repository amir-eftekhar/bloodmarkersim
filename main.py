#!/usr/bin/env python3
"""
Main script for running the wrist blood flow simulation.
Integrates CFD, optical simulation, and machine learning components.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import h5py
import pandas as pd
import time

# Import simulation modules
from src.cfd.blood_flow import run_simulation as run_cfd_simulation
from src.utils.geometry import create_wrist_mesh, visualize_wrist_model
from src.optical.monte_carlo import run_simulation as run_optical_simulation
from src.optical.waveforms import generate_waveforms
from src.ml.data_generation import generate_training_dataset
from src.ml.biomarker_prediction import (
    train_glucose_predictor, train_spo2_predictor, 
    train_lipids_predictor, train_inflammation_predictor
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Wrist Blood Flow Simulation")
    
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "cfd", "optical", "waveform", "ml"],
                       help="Simulation mode (default: full)")
    
    parser.add_argument("--output_dir", type=str, default="data/results",
                       help="Output directory for simulation results")
    
    parser.add_argument("--wavelength", type=str, default="all",
                       choices=["all", "green", "red", "nir"],
                       help="Wavelength for optical simulation (default: all)")
    
    parser.add_argument("--num_photons", type=int, default=10000,
                       help="Number of photons for Monte Carlo simulation")
    
    parser.add_argument("--biomarker", type=str, default="all",
                       choices=["all", "glucose", "spo2", "lipids", "inflammation"],
                       help="Biomarker for ML prediction (default: all)")
    
    parser.add_argument("--artery_radius", type=float, default=0.15,
                       help="Radius of radial artery in cm (default: 0.15)")
    
    parser.add_argument("--artery_depth", type=float, default=0.2,
                       help="Depth of radial artery in cm (default: 0.2)")
    
    parser.add_argument("--visualize", action="store_true",
                       help="Enable visualization of results")
    
    parser.add_argument("--ml_samples", type=int, default=1000,
                       help="Number of samples for ML training (default: 1000)")
    
    return parser.parse_args()


def run_cfd_component(args):
    """Run the CFD simulation component."""
    print("\n===== Running CFD Simulation =====")
    
    # Configuration for CFD
    cfd_config = {
        'artery_radius': args.artery_radius,
        'artery_depth': args.artery_depth,
        'artery_length': 5.0,
        'blood_density': 1.06,
        'blood_viscosity': 0.04,
        'num_x': 50,
        'num_r': 20,
        'dt': 0.005,
        'cardiac_period': 0.8,
        'num_cycles': 3,
        'womersley_number': 2.5
    }
    
    # Output directory
    output_dir = os.path.join(args.output_dir, 'cfd')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run CFD simulation
    start_time = time.time()
    cfd_simulation = run_cfd_simulation(cfd_config, output_dir)
    elapsed_time = time.time() - start_time
    
    print(f"CFD Simulation completed in {elapsed_time:.2f} seconds")
    
    # Visualize results if requested
    if args.visualize:
        cfd_simulation.visualize_results(os.path.join(output_dir, 'figures'))
    
    # Return results for next component
    return os.path.join(output_dir, 'cfd_results.h5')


def run_optical_component(args, cfd_results_file=None):
    """Run the optical simulation component."""
    print("\n===== Running Optical Simulation =====")
    
    # Create wrist model
    print("Creating wrist model...")
    wrist_config = {
        'width': 5.0,
        'height': 2.5,
        'length': 4.0,
        'grid_resolution': 0.05,
        'include_vessels': True,
        'include_bone': True
    }
    
    wrist_model = create_wrist_mesh(wrist_config)
    
    # Visualize wrist model if requested
    if args.visualize:
        visualize_dir = os.path.join(args.output_dir, 'geometry')
        os.makedirs(visualize_dir, exist_ok=True)
        visualize_wrist_model(wrist_model, os.path.join(visualize_dir, 'wrist_model.png'))
    
    # Determine wavelengths to simulate
    if args.wavelength == "all":
        wavelengths = ["green", "red", "nir"]
    else:
        wavelengths = [args.wavelength]
    
    optical_results_files = {}
    
    # Run optical simulation for each wavelength
    for wavelength in wavelengths:
        print(f"\nSimulating {wavelength} wavelength...")
        
        # Optical simulation configuration
        optical_config = {
            'num_photons': args.num_photons,
            'wavelength': wavelength,
            'source_position': [2.0, 0.0, 0.0],
            'source_direction': [0, 0, 1],
            'source_radius': 0.1,
            'detector_radius': 0.3,
            'detector_position': [2.5, 0.0, 0.0],
            'max_weight': 1e-4,
            'record_paths': True
        }
        
        # Output directory
        output_dir = os.path.join(args.output_dir, 'optical', wavelength)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run optical simulation
        start_time = time.time()
        optical_simulation = run_optical_simulation(wrist_model, optical_config, output_dir)
        elapsed_time = time.time() - start_time
        
        print(f"{wavelength} optical simulation completed in {elapsed_time:.2f} seconds")
        
        # Visualize results if requested
        if args.visualize:
            optical_simulation.visualize_results(os.path.join(output_dir, 'figures'))
        
        # Store output file for next component
        optical_results_files[wavelength] = os.path.join(output_dir, f'optical_results_{wavelength}.h5')
    
    return optical_results_files


def run_waveform_component(args, cfd_results_file=None, optical_results_files=None):
    """Run the waveform generation component."""
    print("\n===== Generating PPG Waveforms =====")
    
    if optical_results_files is None:
        print("Error: Optical simulation results required for waveform generation.")
        return None
    
    # Load optical results
    optical_results = {}
    for wavelength, file_path in optical_results_files.items():
        try:
            with h5py.File(file_path, 'r') as f:
                optical_results[wavelength] = {
                    'detection_rate': f['optical_results'].attrs['detection_rate'],
                    'wavelength': wavelength
                }
        except Exception as e:
            print(f"Error loading optical results for {wavelength}: {e}")
            optical_results[wavelength] = {
                'detection_rate': 0.15,
                'wavelength': wavelength
            }
    
    # Load CFD results if available
    cfd_results = None
    if cfd_results_file and os.path.exists(cfd_results_file):
        try:
            with h5py.File(cfd_results_file, 'r') as f:
                # Extract time data
                time = f['cfd_results/time'][:]
                
                # Extract velocity data
                velocity_data = f['cfd_results/velocity_data'][:]
                time_selected = f['cfd_results/time_selected'][:]
                
                # Reshape for waveform generation
                velocity_profiles = []
                for i in range(len(time_selected)):
                    velocity_profiles.append(velocity_data[i])
                
                cfd_results = {
                    'time': time_selected,
                    'velocity_profiles': velocity_profiles
                }
        except Exception as e:
            print(f"Error loading CFD results: {e}")
            cfd_results = None
    
    # Waveform configuration
    waveform_config = {
        'sampling_rate': 100,
        'duration': 10,
        'heart_rate': 75,
        'heart_rate_variability': 3,
        'respiratory_rate': 0.2,
        'noise_level': 0.02,
        'motion_artifacts': True,
        'baseline_drift': True,
        'use_cfd_data': cfd_results is not None,
        'oxygen_saturation': 0.98,
        'glucose_level': 120,
        'lipid_levels': {
            'total_cholesterol': 220,
            'hdl': 45,
            'ldl': 150,
            'triglycerides': 180
        },
        'blood_volumes': {
            'green': 0.05,
            'red': 0.05,
            'nir': 0.05
        }
    }
    
    # Output directory
    output_dir = os.path.join(args.output_dir, 'waveforms')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate waveforms
    start_time = time.time()
    signals = generate_waveforms(optical_results, cfd_results, waveform_config, output_dir)
    elapsed_time = time.time() - start_time
    
    print(f"Waveform generation completed in {elapsed_time:.2f} seconds")
    
    return os.path.join(output_dir, 'ppg_waveforms.h5')


def run_ml_component(args, waveform_file=None):
    """Run the machine learning component."""
    print("\n===== Running Machine Learning Component =====")
    
    # Generate synthetic training dataset
    print("Generating synthetic training data...")
    
    ml_config = {
        'n_samples': args.ml_samples,
        'window_size': 5.0,
        'sampling_rate': 100,
        'random_seed': 42
    }
    
    # Output directory for synthetic data
    synthetic_dir = os.path.join(args.output_dir, 'synthetic')
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Generate synthetic dataset
    start_time = time.time()
    synthetic_data = generate_training_dataset(ml_config, synthetic_dir)
    synthetic_time = time.time() - start_time
    
    print(f"Synthetic data generation completed in {synthetic_time:.2f} seconds")
    
    # Output directory for models
    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Train models for different biomarkers
    biomarkers_to_train = []
    if args.biomarker == "all":
        biomarkers_to_train = ["glucose", "spo2", "lipids", "inflammation"]
    else:
        biomarkers_to_train = [args.biomarker]
    
    trained_models = {}
    
    # Training configuration
    train_config = {
        'feature_selection': True,
        'n_features': 30,
        'test_size': 0.2,
        'random_state': 42,
        'model_type': 'auto',
        'hyperparameter_tuning': True,
        'n_folds': 3,
        'verbose': 1
    }
    
    for biomarker in biomarkers_to_train:
        print(f"\nTraining {biomarker} prediction model...")
        
        # Get features and target
        X = synthetic_data['features']
        
        if biomarker == "glucose":
            y = synthetic_data['biomarkers']['glucose']
            predictor_func = train_glucose_predictor
        elif biomarker == "spo2":
            y = synthetic_data['biomarkers']['spo2']
            predictor_func = train_spo2_predictor
        elif biomarker == "lipids":
            y = synthetic_data['biomarkers']['total_cholesterol']
            predictor_func = train_lipids_predictor
        elif biomarker == "inflammation":
            y = synthetic_data['biomarkers']['inflammation']
            predictor_func = train_inflammation_predictor
        
        # Train model
        start_time = time.time()
        predictor = predictor_func(X, y, train_config)
        train_time = time.time() - start_time
        
        print(f"{biomarker} model training completed in {train_time:.2f} seconds")
        
        # Save model
        model_file = os.path.join(models_dir, f'{biomarker}_predictor.joblib')
        predictor.save_model(model_file)
        
        # Evaluate model
        evaluation = predictor.evaluate(X, y)
        print(f"Training set metrics: MAE={evaluation['mae']:.4f}, RMSE={evaluation['rmse']:.4f}, R²={evaluation['r2']:.4f}")
        
        # Visualize feature importance if requested
        if args.visualize:
            fig_dir = os.path.join(models_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            predictor.plot_feature_importance(save_path=os.path.join(fig_dir, f'{biomarker}_feature_importance.png'))
            predictor.plot_predictions(
                evaluation['true_values'], 
                evaluation['predictions'],
                save_path=os.path.join(fig_dir, f'{biomarker}_predictions.png')
            )
        
        trained_models[biomarker] = predictor
    
    # Return trained models
    return trained_models


def run_full_simulation(args):
    """Run the full simulation pipeline."""
    print("\n===== Running Full Wrist Blood Flow Simulation =====")
    
    # Step 1: Run CFD simulation
    cfd_results_file = run_cfd_component(args)
    
    # Step 2: Run optical simulation
    optical_results_files = run_optical_component(args, cfd_results_file)
    
    # Step 3: Generate waveforms
    waveform_file = run_waveform_component(args, cfd_results_file, optical_results_files)
    
    # Step 4: Run machine learning component
    trained_models = run_ml_component(args, waveform_file)
    
    print("\n===== Simulation Pipeline Completed =====")
    return {
        'cfd_results': cfd_results_file,
        'optical_results': optical_results_files,
        'waveform_file': waveform_file,
        'trained_models': trained_models
    }


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the requested simulation mode
    if args.mode == "full":
        results = run_full_simulation(args)
    elif args.mode == "cfd":
        results = run_cfd_component(args)
    elif args.mode == "optical":
        results = run_optical_component(args)
    elif args.mode == "waveform":
        # For standalone waveform generation, we need to mock optical results
        mock_optical_results = {
            'green': {'detection_rate': 0.15, 'wavelength': 'green'},
            'red': {'detection_rate': 0.15, 'wavelength': 'red'},
            'nir': {'detection_rate': 0.15, 'wavelength': 'nir'}
        }
        results = run_waveform_component(args, None, mock_optical_results)
    elif args.mode == "ml":
        results = run_ml_component(args)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
