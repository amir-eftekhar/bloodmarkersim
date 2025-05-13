#!/usr/bin/env python3
"""
Test script for enhanced cholesterol prediction capabilities using the original BiomarkerPredictor.
Focused specifically on cholesterol and related lipid biomarkers.
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

# Import the existing BiomarkerPredictor
from src.ml.biomarker_prediction import BiomarkerPredictor
from test_model import generate_test_ppg, extract_features

def generate_lipid_dataset(n_samples=300, biomarker='total_cholesterol'):
    """
    Generate a synthetic dataset for training and evaluating cholesterol prediction.
    
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

def train_enhanced_lipid_model(X, y, biomarker_type, config=None):
    """
    Train and evaluate the enhanced biomarker predictor for lipid biomarkers.
    
    Args:
        X (DataFrame): Feature dataframe
        y (array): Target values
        biomarker_type (str): Type of lipid biomarker
        config (dict, optional): Configuration parameters
        
    Returns:
        tuple: (predictor, results) - trained model and evaluation results
    """
    print(f"\nTraining Enhanced {biomarker_type.upper()} predictor...")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Enhanced configuration for lipid prediction
    if config is None:
        config = {
            # Force 'lipids' as model type for enhanced lipid-specific features
            'model_type': 'advanced_lipid_ensemble',
            'hyperparameter_tuning_method': 'randomizedsearch',
            'n_tuning_iter': 50,  # More iterations for better hyperparameter optimization
            'cv_folds': 5,        # K-fold cross-validation (increased for better validation)
            'apply_polynomial_features': True,  # Add polynomial features for capturing non-linear relationships
            'poly_degree': 2,
            'feature_selection_method': 'rfe',  # Use RFE for better feature selection
            'n_features_to_select': [20, 30, 40],  # Try different feature counts
            'scaler_type': 'robust',  # Robust scaling for better handling of outliers
            'random_state': 42,
            'verbose': 1
        }
    
    # Initialize and train the enhanced model
    predictor = BiomarkerPredictor(biomarker_type, config=config)
    predictor.fit(X_train, y_train)
    
    # Evaluate on test set
    test_results = predictor.evaluate(X_test, y_test)
    print(f"Test set R² score: {test_results['r2']:.4f}")
    print(f"Test set MAE: {test_results['mae']:.4f}")
    print(f"Test set RMSE: {test_results['rmse']:.4f}")
    
    # Create results directory
    os.makedirs('data/results', exist_ok=True)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictor.predict(X_test), alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True Values (mg/dL)')
    plt.ylabel('Predicted Values (mg/dL)')
    plt.title(f'Enhanced {biomarker_type.capitalize()} Predictor\nR² = {test_results["r2"]:.4f}, MAE = {test_results["mae"]:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'data/results/{biomarker_type}_enhanced_predictions.png', dpi=300)
    plt.close()
    
    # Print and plot feature importance if available
    if hasattr(predictor, 'important_features_'):
        print("\nTop 15 important features:")
        for i, feature in enumerate(predictor.important_features_[:15]):
            print(f"{i+1}. {feature}")
        
        # Plot feature importance
        predictor.plot_feature_importance(top_n=15, 
                                         save_path=f'data/results/{biomarker_type}_feature_importance.png')
    
    return predictor, test_results

def test_all_lipid_biomarkers(n_samples=300):
    """
    Test all lipid-related biomarker predictions with enhanced models.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        dict: Test results for all lipid biomarkers
    """
    # Test different lipid biomarkers
    biomarkers = ['total_cholesterol', 'ldl', 'hdl', 'triglycerides']
    results = {}
    saved_models = {}
    
    for biomarker in biomarkers:
        print(f"\n{'='*80}")
        print(f"Testing ENHANCED {biomarker.upper()} prediction")
        print(f"{'='*80}")
        
        # Generate dataset
        X, y = generate_lipid_dataset(n_samples, biomarker)
        
        # Train and evaluate model
        predictor, eval_results = train_enhanced_lipid_model(X, y, biomarker)
        
        # Save model
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
    print("ENHANCED CHOLESTEROL PREDICTION SUMMARY")
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
    print("\nTesting Enhanced Cholesterol Prediction Capabilities")
    print("Focused on Lipid Profile Biomarkers\n")
    
    results, models = test_all_lipid_biomarkers(n_samples=300)
