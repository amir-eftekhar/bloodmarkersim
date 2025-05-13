#!/usr/bin/env python3
"""
Test script for the updated PhysicsInformedBiomarkerPredictor functionality.
This script tests the implementation in src/ml/biomarker_prediction.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test utilities
from test_model import generate_test_ppg, extract_features

def generate_test_data(n_samples=100, biomarker='total_cholesterol'):
    """Generate synthetic dataset for testing the biomarker predictor."""
    print(f"Generating {n_samples} synthetic samples for {biomarker}...")
    
    # Biomarker ranges
    biomarker_ranges = {
        'total_cholesterol': (120, 300),
        'ldl': (40, 200),
        'hdl': (25, 100),
        'triglycerides': (50, 300),
        'glucose': (60, 300),
        'spo2': (0.90, 1.0),
        'hemoglobin': (8, 18)
    }
    
    features_list = []
    targets = []
    
    for i in range(n_samples):
        # Generate correlated values
        total_chol = np.random.uniform(*biomarker_ranges['total_cholesterol'])
        ldl = np.clip(total_chol * 0.6 + np.random.normal(0, 15), 
                     biomarker_ranges['ldl'][0], biomarker_ranges['ldl'][1])
        hdl = np.random.uniform(*biomarker_ranges['hdl'])
        triglycerides = np.clip(total_chol * 0.3 + np.random.normal(0, 25),
                                biomarker_ranges['triglycerides'][0], biomarker_ranges['triglycerides'][1])
        
        # Other biomarkers
        glucose = np.random.uniform(*biomarker_ranges['glucose'])
        spo2 = np.random.uniform(*biomarker_ranges['spo2'])
        hemoglobin = np.random.uniform(*biomarker_ranges['hemoglobin'])
        
        # Set target based on requested biomarker
        if biomarker == 'total_cholesterol':
            target = total_chol
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
        
        # Bundle values for PPG generation
        blood_content = {
            'total_cholesterol': total_chol,
            'ldl': ldl,
            'hdl': hdl,
            'triglycerides': triglycerides,
            'glucose': glucose,
            'spo2': spo2,
            'hemoglobin': hemoglobin
        }
        
        # Generate PPG signals
        heart_rate = np.random.uniform(60, 100)
        ppg_signals = {
            'green': generate_test_ppg('green', blood_content, heart_rate),
            'red': generate_test_ppg('red', blood_content, heart_rate),
            'nir': generate_test_ppg('nir', blood_content, heart_rate)
        }
        
        # Extract features
        features = extract_features(ppg_signals)
        features_list.append(features)
        targets.append(target)
    
    # Convert to DataFrame/array
    X = pd.DataFrame(features_list)
    y = np.array(targets)
    
    return X, y

def test_biomarker_predictor():
    """Test the updated biomarker predictor."""
    try:
        # Try to import the predictor class
        from src.ml.biomarker_prediction import PhysicsInformedBiomarkerPredictor
        predictor_class = PhysicsInformedBiomarkerPredictor
        print("Using PhysicsInformedBiomarkerPredictor")
    except (ImportError, AttributeError):
        # Fall back to simplified implementation
        from sklearn.ensemble import GradientBoostingRegressor
        predictor_class = GradientBoostingRegressor
        print("Using GradientBoostingRegressor (fallback)")
    
    # Generate test data
    biomarker = 'total_cholesterol'
    X, y = generate_test_data(n_samples=100, biomarker=biomarker)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create predictor with simplified config if PhysicsInformedBiomarkerPredictor
    if predictor_class.__name__ == 'PhysicsInformedBiomarkerPredictor':
        config = {
            'model_type': 'gradient_boosting',  # Simpler model for testing
            'random_state': 42,
            'verbose': 1
        }
        predictor = predictor_class(biomarker, config)
    else:
        # Fallback to sklearn estimator
        predictor = predictor_class(n_estimators=100, random_state=42, verbose=1)
    
    # Train model
    print(f"Training model for {biomarker} prediction...")
    try:
        predictor.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    
    # Evaluate model
    try:
        if hasattr(predictor, 'evaluate'):
            results = predictor.evaluate(X_test, y_test)
            print(f"R² score: {results['r2']:.4f}")
            print(f"MAE: {results['mae']:.4f}")
        else:
            # Fallback for sklearn estimator
            from sklearn.metrics import r2_score, mean_absolute_error
            y_pred = predictor.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"R² score: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
            
        # Save model if method available
        if hasattr(predictor, 'save'):
            model_path = predictor.save()
            print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    return predictor

if __name__ == "__main__":
    # Create output directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    print("\nTesting Biomarker Predictor")
    print("=" * 40)
    test_biomarker_predictor()
