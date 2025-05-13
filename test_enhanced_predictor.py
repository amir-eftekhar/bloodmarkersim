#!/usr/bin/env python3
"""
Test script to evaluate the enhanced BiomarkerPredictor implementation with
biomarker-specific transformers, advanced pipeline construction, and
state-of-the-art ensemble models.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import necessary modules
from src.ml.biomarker_prediction import BiomarkerPredictor
from test_model import generate_test_ppg, extract_features

def generate_synthetic_dataset(n_samples=500, biomarker='spo2'):
    """
    Generate a synthetic dataset for training and evaluating biomarker prediction models.
    
    Args:
        n_samples (int): Number of samples to generate
        biomarker (str): Target biomarker ('spo2', 'glucose', 'lipids', or 'hemoglobin')
        
    Returns:
        tuple: (X, y) where X is features dataframe and y is target values
    """
    print(f"Generating {n_samples} synthetic samples for {biomarker} prediction...")
    
    # Define biomarker ranges and parameters
    biomarker_ranges = {
        'spo2': (0.75, 1.0),  # 75-100%
        'glucose': (60, 300),  # 60-300 mg/dL
        'total_cholesterol': (120, 300),  # 120-300 mg/dL (for lipids)
        'hemoglobin': (8, 18)   # 8-18 g/dL
    }
    
    # Get range for target biomarker
    if biomarker == 'lipids':
        target_min, target_max = biomarker_ranges['total_cholesterol']
    else:
        target_min, target_max = biomarker_ranges[biomarker]
    
    # Initialize data structures
    features_list = []
    targets = []
    
    # Generate samples
    for _ in tqdm(range(n_samples)):
        # Generate random biomarker values
        spo2 = np.random.uniform(*biomarker_ranges['spo2'])
        glucose = np.random.uniform(*biomarker_ranges['glucose'])
        total_cholesterol = np.random.uniform(*biomarker_ranges['total_cholesterol'])
        hemoglobin = np.random.uniform(*biomarker_ranges['hemoglobin'])
        
        # Set target value based on biomarker type
        if biomarker == 'spo2':
            target = spo2
        elif biomarker == 'glucose':
            target = glucose
        elif biomarker == 'lipids':
            target = total_cholesterol
        elif biomarker == 'hemoglobin':
            target = hemoglobin
        else:
            raise ValueError(f"Unsupported biomarker: {biomarker}")
        
        # Create blood content dictionary
        blood_content = {
            'spo2': spo2,
            'glucose': glucose,
            'total_cholesterol': total_cholesterol,
            'hemoglobin': hemoglobin
        }
        
        # Generate heart rate variations for more diverse samples
        heart_rate = np.random.uniform(60, 100)
        
        # Generate PPG signals for different wavelengths
        ppg_green = generate_test_ppg('green', blood_content, heart_rate)
        ppg_red = generate_test_ppg('red', blood_content, heart_rate)
        ppg_nir = generate_test_ppg('nir', blood_content, heart_rate)
        
        # Bundle signals
        ppg_signals = {
            'green': ppg_green,
            'red': ppg_red,
            'nir': ppg_nir
        }
        
        # Extract features
        features = extract_features(ppg_signals)
        features_list.append(features)
        targets.append(target)
    
    # Convert to DataFrame/array
    X = pd.DataFrame(features_list)
    y = np.array(targets)
    
    return X, y

def train_advanced_model(X, y, biomarker_type, advanced_config=None):
    """
    Train and evaluate the advanced biomarker predictor model.
    
    Args:
        X (DataFrame): Feature dataframe
        y (array): Target values
        biomarker_type (str): Type of biomarker
        advanced_config (dict, optional): Advanced configuration parameters
        
    Returns:
        tuple: (predictor, results) - trained model and evaluation results
    """
    print(f"\nTraining enhanced {biomarker_type} predictor model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Define advanced configuration if not provided
    if advanced_config is None:
        advanced_config = {
            'hyperparameter_tuning_method': 'randomizedsearch',
            'n_tuning_iter': 30,  # Reduced for faster testing
            'cv_folds': 3,        # Reduced for faster testing
            'verbose': 1,
            'random_state': 42
        }
    
    # Initialize and train model
    predictor = BiomarkerPredictor(biomarker_type, config=advanced_config)
    predictor.fit(X_train, y_train)
    
    # Evaluate on test set
    test_results = predictor.evaluate(X_test, y_test)
    print(f"Test set R² score: {test_results['r2']:.4f}")
    print(f"Test set MAE: {test_results['mae']:.4f}")
    
    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    if len(y_test) <= 1000:  # Only plot if results are included
        plt.scatter(test_results['true_values'], test_results['predictions'], alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Enhanced {biomarker_type.capitalize()} Predictor\nR² = {test_results["r2"]:.4f}, MAE = {test_results["mae"]:.4f}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('data/results', exist_ok=True)
        plt.savefig(f'data/results/{biomarker_type}_predictions.png', dpi=300)
        plt.close()
    
    # Print feature importance if available
    if hasattr(predictor, 'important_features_'):
        print("\nTop 10 important features:")
        for i, feature in enumerate(predictor.important_features_[:10]):
            print(f"{i+1}. {feature}")
    
    return predictor, test_results

def test_all_biomarkers(n_samples=500):
    """
    Test the enhanced BiomarkerPredictor on all biomarker types.
    
    Args:
        n_samples (int): Number of samples to generate for each biomarker
        
    Returns:
        dict: Results for all biomarker types
    """
    results = {}
    biomarkers = ['spo2', 'glucose', 'lipids', 'hemoglobin']
    
    for biomarker in biomarkers:
        print(f"\n{'='*80}")
        print(f"Testing {biomarker.upper()} prediction with enhanced model")
        print(f"{'='*80}")
        
        # Generate dataset
        X, y = generate_synthetic_dataset(n_samples, biomarker)
        
        # Train and evaluate model
        predictor, eval_results = train_advanced_model(X, y, biomarker)
        
        # Save model
        model_path = predictor.save()
        print(f"Model saved to {model_path}")
        
        # Store results
        results[biomarker] = {
            'r2': eval_results['r2'],
            'mae': eval_results['mae'],
            'model_path': model_path
        }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    for biomarker, res in results.items():
        print(f"{biomarker.upper()}: R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}")
    
    return results

def compare_tuning_methods(biomarker='spo2', n_samples=300):
    """
    Compare different hyperparameter tuning methods.
    
    Args:
        biomarker (str): Biomarker to test
        n_samples (int): Number of samples to generate
        
    Returns:
        dict: Results for different tuning methods
    """
    print(f"\n{'='*80}")
    print(f"Comparing hyperparameter tuning methods for {biomarker.upper()} prediction")
    print(f"{'='*80}")
    
    # Generate dataset
    X, y = generate_synthetic_dataset(n_samples, biomarker)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Define configurations
    configs = {
        'no_tuning': {
            'hyperparameter_tuning_method': None,
            'verbose': 1,
            'random_state': 42
        },
        'randomized_search': {
            'hyperparameter_tuning_method': 'randomizedsearch',
            'n_tuning_iter': 30,
            'cv_folds': 3,
            'verbose': 1,
            'random_state': 42
        },
        'grid_search': {
            'hyperparameter_tuning_method': 'gridsearch',
            'cv_folds': 3,
            'verbose': 1,
            'random_state': 42
        }
    }
    
    # Train and evaluate with each configuration
    results = {}
    for name, config in configs.items():
        print(f"\nTesting configuration: {name}")
        predictor = BiomarkerPredictor(biomarker, config=config)
        predictor.fit(X_train, y_train)
        
        # Evaluate
        eval_results = predictor.evaluate(X_test, y_test)
        results[name] = {
            'r2': eval_results['r2'],
            'mae': eval_results['mae']
        }
        print(f"R² score: {eval_results['r2']:.4f}, MAE: {eval_results['mae']:.4f}")
    
    # Print summary
    print("\nTuning methods comparison summary:")
    for name, res in results.items():
        print(f"{name}: R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}")
    
    return results

if __name__ == "__main__":
    # Make sure output directories exist
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test all biomarkers with smaller sample size for faster execution
    results = test_all_biomarkers(n_samples=300)
    
    # Optional: Compare tuning methods
    # tuning_results = compare_tuning_methods(biomarker='glucose', n_samples=200)
