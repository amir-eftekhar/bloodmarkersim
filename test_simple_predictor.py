#!/usr/bin/env python3
"""
Simple test script for the enhanced BiomarkerPredictor implementation
that doesn't rely on the feature extraction dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import our enhanced BiomarkerPredictor class
from src.ml.biomarker_prediction import BiomarkerPredictor

def generate_synthetic_data(n_samples=500, n_features=30, biomarker_type='spo2'):
    """
    Generate simple synthetic data for testing the BiomarkerPredictor.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        biomarker_type: Type of biomarker
        
    Returns:
        X, y: Features and target values
    """
    print(f"Generating {n_samples} samples with {n_features} features for {biomarker_type} prediction...")
    
    # Set appropriate target range based on biomarker type
    if biomarker_type == 'spo2':
        y_range = (0.75, 1.0)  # 75-100%
    elif biomarker_type == 'glucose':
        y_range = (60, 300)    # 60-300 mg/dL
    elif biomarker_type in ['lipids', 'total_cholesterol']:
        y_range = (120, 300)   # 120-300 mg/dL
    elif biomarker_type == 'hemoglobin':
        y_range = (8, 18)      # 8-18 g/dL
    else:
        y_range = (0, 100)     # Default range
    
    # Generate regression dataset
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=int(n_features * 0.8),  # 80% of features are informative
        noise=0.1,
        random_state=42
    )
    
    # Scale y to the appropriate biomarker range
    y_min, y_max = y_range
    y = (y - y.min()) / (y.max() - y.min()) * (y_max - y_min) + y_min
    
    # Generate meaningful feature names
    feature_names = []
    for i in range(n_features):
        wavelength = np.random.choice(['green', 'red', 'nir'])
        feature_type = np.random.choice(['mean', 'std', 'peak', 'valley', 'ratio', 'power'])
        feature_names.append(f'{wavelength}_{feature_type}_{i}')
    
    # Convert to DataFrame with named features
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y

def test_biomarker_predictor(biomarker_type='spo2', n_samples=500, n_features=30):
    """
    Test the enhanced BiomarkerPredictor implementation for a specific biomarker.
    
    Args:
        biomarker_type: Type of biomarker to predict
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        
    Returns:
        dict: Test results
    """
    print(f"\n{'='*80}")
    print(f"Testing Enhanced BiomarkerPredictor for {biomarker_type.upper()} prediction")
    print(f"{'='*80}")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples, n_features, biomarker_type)
    
    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Configure the predictor
    config = {
        'hyperparameter_tuning_method': 'randomizedsearch',
        'n_tuning_iter': 20,  # Reduced for faster testing
        'cv_folds': 3,        # Reduced for faster testing
        'verbose': 1,
        'random_state': 42
    }
    
    # Initialize and train the enhanced predictor
    print(f"\nTraining enhanced {biomarker_type} predictor...")
    predictor = BiomarkerPredictor(biomarker_type, config=config)
    predictor.fit(X_train, y_train)
    
    # Evaluate on test set
    print(f"\nEvaluating enhanced {biomarker_type} predictor...")
    test_results = predictor.evaluate(X_test, y_test)
    
    # Display results
    print(f"\nTest Results for {biomarker_type.upper()}:")
    print(f"R² score: {test_results['r2']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"RMSE: {test_results['rmse']:.4f}")
    
    # Display important features if available
    if hasattr(predictor, 'important_features_') and len(predictor.important_features_) > 0:
        print("\nTop 10 important features:")
        for i, feature in enumerate(predictor.important_features_[:10]):
            print(f"{i+1}. {feature}")
    
    # Save the model
    model_path = predictor.save()
    print(f"\nModel saved to: {model_path}")
    
    return {
        'predictor': predictor,
        'results': test_results,
        'model_path': model_path
    }

def test_all_biomarkers():
    """Test all supported biomarker types with the enhanced predictor."""
    results = {}
    
    # Create output directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test each biomarker type
    biomarker_types = ['spo2', 'glucose', 'lipids', 'hemoglobin']
    
    for biomarker in biomarker_types:
        # Use a smaller dataset for faster testing
        test_result = test_biomarker_predictor(biomarker, n_samples=300, n_features=20)
        results[biomarker] = test_result['results']
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    for biomarker, res in results.items():
        print(f"{biomarker.upper()}: R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}")
    
    return results

if __name__ == "__main__":
    test_all_biomarkers()
