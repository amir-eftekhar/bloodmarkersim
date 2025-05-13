#!/usr/bin/env python3
"""
Standalone test script for evaluating the enhanced biomarker prediction capabilities.
This script creates a completely independent implementation to demonstrate the core concepts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

# Path for saving results
os.makedirs('data/results', exist_ok=True)

def generate_synthetic_biomarker_data(n_samples=500, biomarker_type='spo2'):
    """
    Generate synthetic data for biomarker prediction testing.
    
    Args:
        n_samples: Number of samples to generate
        biomarker_type: Type of biomarker
        
    Returns:
        X: Features dataframe
        y: Target values
    """
    print(f"Generating {n_samples} synthetic samples for {biomarker_type} prediction...")
    
    # Define target value ranges based on biomarker type
    ranges = {
        'spo2': (0.75, 1.0),           # 75-100%
        'glucose': (60, 300),          # 60-300 mg/dL
        'lipids': (120, 300),          # 120-300 mg/dL
        'hemoglobin': (8, 18)          # 8-18 g/dL
    }
    
    # Get appropriate range for target biomarker
    y_min, y_max = ranges.get(biomarker_type, (0, 100))
    
    # Generate regression dataset with controlled relationship to target
    X, y_raw = make_regression(
        n_samples=n_samples,
        n_features=30,
        n_informative=20,
        noise=0.2,
        random_state=42
    )
    
    # Scale y to appropriate biomarker range
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) * (y_max - y_min) + y_min
    
    # Define feature names that simulate PPG-derived features
    wavelengths = ['red', 'nir', 'green']
    feature_types = ['mean', 'std', 'peak', 'valley', 'ratio', 'power', 'ac_dc', 'frequency']
    
    # Create feature names
    feature_names = []
    for i in range(X.shape[1]):
        wavelength = wavelengths[i % len(wavelengths)]
        feature_type = feature_types[(i // len(wavelengths)) % len(feature_types)]
        feature_names.append(f"{wavelength}_{feature_type}_{i}")
    
    # Convert to dataframe
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y

class BiomarkerSpecificTransformer:
    """Transformer that creates biomarker-specific features."""
    
    def __init__(self, biomarker_type):
        self.biomarker_type = biomarker_type
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        
        # Add biomarker-specific derived features
        if self.biomarker_type == 'spo2':
            # Create R-ratio features for SpO2
            red_cols = [col for col in X.columns if col.startswith('red_')]
            nir_cols = [col for col in X.columns if col.startswith('nir_')]
            
            if red_cols and nir_cols:
                # Add r-ratio feature (key for SpO2)
                X_new['r_ratio'] = X[red_cols[0]] / X[nir_cols[0]] if len(red_cols) > 0 and len(nir_cols) > 0 else 0
                
                # Add empirical SpO2 approximation
                X_new['empirical_spo2'] = 110 - 25 * X_new['r_ratio'] 
                
        elif self.biomarker_type == 'glucose':
            # Create glucose-specific features
            nir_cols = [col for col in X.columns if col.startswith('nir_')]
            if nir_cols:
                # NIR features are particularly important for glucose
                X_new['nir_mean_sq'] = X[nir_cols[0]] ** 2 if len(nir_cols) > 0 else 0
                
                # Add derivative features if available
                if len(nir_cols) >= 2:
                    X_new['nir_ratio'] = X[nir_cols[0]] / X[nir_cols[1]] if X[nir_cols[1]].mean() != 0 else 0
        
        elif self.biomarker_type in ['lipids', 'total_cholesterol']:
            # Create lipid-specific features
            green_cols = [col for col in X.columns if col.startswith('green_')]
            red_cols = [col for col in X.columns if col.startswith('red_')]
            
            if green_cols and red_cols:
                # Green-to-red ratio is sensitive to lipid levels
                X_new['green_red_ratio'] = X[green_cols[0]] / X[red_cols[0]] if len(green_cols) > 0 and len(red_cols) > 0 else 0
                
        elif self.biomarker_type == 'hemoglobin':
            # Create hemoglobin-specific features
            green_cols = [col for col in X.columns if col.startswith('green_')]
            red_cols = [col for col in X.columns if col.startswith('red_')]
            
            if green_cols and red_cols:
                # Beer-Lambert inspired feature for hemoglobin
                X_new['optical_density_ratio'] = -np.log(X[green_cols[0]] / X[red_cols[0]]) if len(green_cols) > 0 and len(red_cols) > 0 else 0
        
        return X_new
    
    def get_feature_names_out(self, feature_names_in=None):
        """Return feature names for output features."""
        if feature_names_in is None:
            feature_names_in = []
            
        feature_names_out = list(feature_names_in)
        
        # Add names of new features based on biomarker type
        if self.biomarker_type == 'spo2':
            feature_names_out.extend(['r_ratio', 'empirical_spo2'])
        elif self.biomarker_type == 'glucose':
            feature_names_out.extend(['nir_mean_sq', 'nir_ratio'])
        elif self.biomarker_type in ['lipids', 'total_cholesterol']:
            feature_names_out.append('green_red_ratio')
        elif self.biomarker_type == 'hemoglobin':
            feature_names_out.append('optical_density_ratio')
            
        return np.array(feature_names_out)

def build_biomarker_pipeline(biomarker_type):
    """
    Build an optimized pipeline for the specified biomarker type.
    
    Args:
        biomarker_type: Type of biomarker to predict
        
    Returns:
        pipeline: Optimized scikit-learn pipeline
        param_grid: Parameter grid for hyperparameter optimization
    """
    steps = []
    
    # Add biomarker-specific transformer
    steps.append(('biomarker_features', BiomarkerSpecificTransformer(biomarker_type)))
    
    # Add scaling 
    steps.append(('scaler', StandardScaler()))
    
    # Add model based on biomarker type
    if biomarker_type == 'spo2':
        # SpO2 benefits from ensemble methods that can capture the non-linear R-ratio relationship
        base_estimators = [
            ('rf', RandomForestRegressor(random_state=42)),
            ('gbm', GradientBoostingRegressor(random_state=42))
        ]
        final_estimator = RandomForestRegressor(random_state=42)
        steps.append(('model', StackingRegressor(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=3
        )))
        
        # Parameter grid for SpO2
        param_grid = {
            'model__rf__n_estimators': [100, 200],
            'model__rf__max_depth': [10, 20, None],
            'model__gbm__n_estimators': [100, 200],
            'model__gbm__learning_rate': [0.05, 0.1],
            'model__final_estimator__n_estimators': [100, 200]
        }
        
    elif biomarker_type == 'glucose':
        # Glucose typically requires more complex models
        base_estimators = [
            ('rf', RandomForestRegressor(random_state=42, n_estimators=200)),
            ('gbm', GradientBoostingRegressor(random_state=42, n_estimators=200))
        ]
        final_estimator = GradientBoostingRegressor(random_state=42)
        steps.append(('model', StackingRegressor(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=3
        )))
        
        # Parameter grid for glucose
        param_grid = {
            'model__rf__max_depth': [20, 30, None],
            'model__gbm__learning_rate': [0.01, 0.05, 0.1],
            'model__final_estimator__n_estimators': [100, 200, 300],
            'model__final_estimator__learning_rate': [0.01, 0.05, 0.1]
        }
        
    elif biomarker_type in ['lipids', 'total_cholesterol']:
        # Lipids also benefit from ensemble methods
        steps.append(('model', GradientBoostingRegressor(random_state=42)))
        
        # Parameter grid for lipids
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [5, 10, 15],
            'model__min_samples_split': [2, 5, 10]
        }
        
    elif biomarker_type == 'hemoglobin':
        # Hemoglobin prediction
        steps.append(('model', RandomForestRegressor(random_state=42)))
        
        # Parameter grid for hemoglobin
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [10, 20, None],
            'model__min_samples_leaf': [1, 2, 4]
        }
        
    else:
        # Default model for other biomarkers
        steps.append(('model', RandomForestRegressor(random_state=42)))
        
        # Default parameter grid
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None]
        }
    
    return Pipeline(steps), param_grid

def train_and_evaluate_biomarker_model(biomarker_type, X, y):
    """
    Train and evaluate a biomarker prediction model.
    
    Args:
        biomarker_type: Type of biomarker to predict
        X: Feature dataframe 
        y: Target values
        
    Returns:
        results: Dictionary of evaluation results
    """
    print(f"\nTraining model for {biomarker_type} prediction...")
    
    # Split data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Build pipeline and param grid
    pipeline, param_grid = build_biomarker_pipeline(biomarker_type)
    
    # Perform hyperparameter tuning
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_grid,
        n_iter=20,  # Limited for faster execution
        scoring='r2',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Test R² score: {r2:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{biomarker_type.upper()} Prediction Results\nR² = {r2:.4f}, MAE = {mae:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f'data/results/{biomarker_type}_predictions.png', dpi=300)
    plt.close()
    
    return {
        'r2': r2,
        'mae': mae,
        'model': best_model,
        'best_params': search.best_params_
    }

def test_all_biomarkers(n_samples=300):
    """Test biomarker prediction for all biomarker types."""
    biomarker_types = ['spo2', 'glucose', 'lipids', 'hemoglobin']
    results = {}
    
    for biomarker in biomarker_types:
        print(f"\n{'='*80}")
        print(f"Testing {biomarker.upper()} prediction")
        print(f"{'='*80}")
        
        # Generate data
        X, y = generate_synthetic_biomarker_data(n_samples=n_samples, biomarker_type=biomarker)
        
        # Train and evaluate
        result = train_and_evaluate_biomarker_model(biomarker, X, y)
        results[biomarker] = result
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    for biomarker, res in results.items():
        print(f"{biomarker.upper()}: R² = {res['r2']:.4f}, MAE = {res['mae']:.4f}")
    
    return results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    test_all_biomarkers(n_samples=300)
