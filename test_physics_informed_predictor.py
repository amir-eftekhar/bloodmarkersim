#!/usr/bin/env python3
"""
Test script for the Physics-Informed Biomarker Prediction Framework.
Tests advanced cholesterol prediction capabilities with the updated model.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import the Physics-Informed Biomarker Predictor
from src.ml.biomarker_prediction import PhysicsInformedBiomarkerPredictor
from test_model import generate_test_ppg, extract_features

def generate_synthetic_dataset(n_samples=300, biomarker='total_cholesterol'):
    """
    Generate a synthetic dataset for training and evaluating the physics-informed model.
    
    Args:
        n_samples (int): Number of samples to generate
        biomarker (str): Target biomarker - focused on cholesterol types
        
    Returns:
        tuple: (X, y) where X is features dataframe and y is target values
    """
    print(f"Generating {n_samples} synthetic samples for {biomarker} prediction...")
    
    # Biomarker ranges with more detailed lipid profile
    biomarker_ranges = {
        'total_cholesterol': (120, 300),  # 120-300 mg/dL
        'ldl': (40, 200),                 # 40-200 mg/dL
        'hdl': (25, 100),                 # 25-100 mg/dL
        'triglycerides': (50, 300),       # 50-300 mg/dL
        'glucose': (60, 300),             # 60-300 mg/dL (for comparison)
        'spo2': (0.90, 1.0),              # 90-100% (for baseline)
        'hemoglobin': (8, 18)             # 8-18 g/dL (for comparison)
    }
    
    # Initialize data structures
    features_list = []
    targets = []
    
    # Generate samples with correlations between lipid values
    for _ in tqdm(range(n_samples)):
        # Generate base values for primary biomarkers
        total_cholesterol_base = np.random.uniform(*biomarker_ranges['total_cholesterol'])
        
        # Create correlated lipid values (HDL, LDL, triglycerides should sum close to total_cholesterol)
        hdl = np.random.uniform(*biomarker_ranges['hdl'])
        # LDL correlates with total cholesterol
        ldl = np.clip(total_cholesterol_base * 0.6 + np.random.normal(0, 15), 
                      biomarker_ranges['ldl'][0], biomarker_ranges['ldl'][1])
        # Triglycerides with some correlation to total cholesterol
        triglycerides = np.clip(total_cholesterol_base * 0.3 + np.random.normal(0, 25),
                               biomarker_ranges['triglycerides'][0], biomarker_ranges['triglycerides'][1])
        
        # Add other biomarkers
        glucose = np.random.uniform(*biomarker_ranges['glucose'])
        spo2 = np.random.uniform(*biomarker_ranges['spo2'])
        hemoglobin = np.random.uniform(*biomarker_ranges['hemoglobin'])
        
        # Set target value based on requested biomarker
        if biomarker in biomarker_ranges:
            if biomarker == 'total_cholesterol':
                target = total_cholesterol_base
            elif biomarker == 'ldl':
                target = ldl
            elif biomarker == 'hdl':
                target = hdl
            elif biomarker == 'triglycerides':
                target = triglycerides
            elif biomarker == 'glucose':
                target = glucose
            elif biomarker == 'spo2':
                target = spo2
            elif biomarker == 'hemoglobin':
                target = hemoglobin
        else:
            raise ValueError(f"Unsupported biomarker: {biomarker}")
        
        # Bundle all biomarkers for PPG signal generation
        blood_content = {
            'total_cholesterol': total_cholesterol_base,
            'ldl': ldl,
            'hdl': hdl,
            'triglycerides': triglycerides,
            'glucose': glucose,
            'spo2': spo2,
            'hemoglobin': hemoglobin
        }
        
        # Generate heart rate with physiological variations
        heart_rate = np.random.uniform(60, 100)
        
        # Generate multi-wavelength PPG signals
        ppg_signals = {
            'green': generate_test_ppg('green', blood_content, heart_rate),
            'red': generate_test_ppg('red', blood_content, heart_rate),
            'nir': generate_test_ppg('nir', blood_content, heart_rate)
        }
        
        # Extract comprehensive features from PPG signals
        features = extract_features(ppg_signals)
        features_list.append(features)
        targets.append(target)
    
    # Convert to DataFrame/array
    X = pd.DataFrame(features_list)
    y = np.array(targets)
    
    return X, y

def train_physics_informed_model(X, y, biomarker_type, config=None):
    """
    Train and evaluate the physics-informed biomarker predictor.
    
    Args:
        X (DataFrame): Feature dataframe
        y (array): Target values
        biomarker_type (str): Type of biomarker
        config (dict, optional): Configuration parameters
        
    Returns:
        tuple: (predictor, results) - trained model and evaluation results
    """
    print(f"\nTraining Physics-Informed {biomarker_type.upper()} predictor...")
    
    # Split data with stratification if possible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Default configuration if not provided
    if config is None:
        config = {
            'model_type': 'physics_hybrid', # Use physics-informed hybrid model
            'hyperparameter_tuning_method': 'randomizedsearch',
            'n_tuning_iter': 25,  # Reasonable for testing
            'cv_folds': 3,        # K-fold cross-validation
            'enable_physics_constraints': True, # Enable physics-based constraints
            'apply_feature_engineering': True,  # Advanced feature engineering
            'verbose': 1,
            'random_state': 42
        }
    
    # Initialize and train the physics-informed model
    predictor = PhysicsInformedBiomarkerPredictor(biomarker_type, config=config)
    predictor.fit(X_train, y_train)
    
    # Evaluate on test set
    test_results = predictor.evaluate(X_test, y_test)
    print(f"Test set R² score: {test_results['r2']:.4f}")
    print(f"Test set MAE: {test_results['mae']:.4f}")
    
    # Create results directory
    os.makedirs('data/results', exist_ok=True)
    
    # Plot predictions vs actual values
    predictor.plot_predictions(y_test, predictor.predict(X_test), 
                              save_path=f'data/results/{biomarker_type}_physics_informed_predictions.png')
    
    # Plot feature importance if available
    if hasattr(predictor, 'feature_importances_') and predictor.feature_importances_ is not None:
        predictor.plot_feature_importances(
            top_n=15, 
            save_path=f'data/results/{biomarker_type}_physics_informed_features.png')
    
    return predictor, test_results

def test_cholesterol_prediction(n_samples=300):
    """
    Test cholesterol prediction with the physics-informed model.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        dict: Test results for all lipid biomarkers
    """
    # Test different lipid biomarkers
    biomarkers = ['total_cholesterol', 'ldl', 'hdl', 'triglycerides']
    results = {}
    saved_models = {}
    
    # Advanced physics-hybrid configuration for lipid prediction
    lipid_config = {
        'model_type': 'physics_hybrid',
        'estimator_type': 'ensemble',  
        'hyperparameter_tuning_method': 'randomizedsearch',
        'n_tuning_iter': 30,
        'cv_folds': 3,
        'enable_physics_constraints': True,
        'apply_feature_engineering': True,
        'feature_extraction_method': 'advanced',
        'apply_wavelength_specific_processing': True,  # Process different wavelengths with domain knowledge
        'use_absorption_physics': True,  # Use Beer-Lambert law principles
        'random_state': 42,
        'verbose': 1
    }
    
    for biomarker in biomarkers:
        print(f"\n{'='*80}")
        print(f"Testing PHYSICS-INFORMED {biomarker.upper()} prediction")
        print(f"{'='*80}")
        
        # Generate dataset
        X, y = generate_synthetic_dataset(n_samples, biomarker)
        
        # Train and evaluate model
        predictor, eval_results = train_physics_informed_model(X, y, biomarker, lipid_config)
        
        # Save model with timestamp
        model_path = predictor.save()
        print(f"Model saved to: {model_path}")
        
        # Store results
        results[biomarker] = {
            'r2': eval_results['r2'],
            'mae': eval_results['mae'],
            'rmse': eval_results['rmse'],
            'model_path': model_path
        }
        saved_models[biomarker] = predictor
    
    # Print summary
    print("\n" + "="*80)
    print("PHYSICS-INFORMED CHOLESTEROL PREDICTION SUMMARY")
    print("="*80)
    for biomarker, res in results.items():
        print(f"{biomarker.upper()}:")
        print(f"  R² score: {res['r2']:.4f}")
        print(f"  MAE: {res['mae']:.4f} mg/dL")
        print(f"  RMSE: {res['rmse']:.4f} mg/dL")
        print(f"  Model saved to: {res['model_path']}")
    
    return results, saved_models

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create necessary directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Run cholesterol prediction tests
    print("\nTesting Physics-Informed Biomarker Prediction Framework")
    print("Focus: Advanced Cholesterol Prediction\n")
    
    results, models = test_cholesterol_prediction(n_samples=300)
