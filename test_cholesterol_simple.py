#!/usr/bin/env python3
"""
Simple test script for enhanced cholesterol prediction capabilities.
This script works with the existing dependencies without requiring TensorFlow.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import test utilities
from test_model import generate_test_ppg, extract_features

def generate_lipid_dataset(n_samples=200):
    """
    Generate a synthetic dataset for cholesterol prediction.
    """
    print(f"Generating {n_samples} synthetic samples for cholesterol prediction...")
    
    # Biomarker ranges
    biomarker_ranges = {
        'total_cholesterol': (120, 300),  # 120-300 mg/dL
        'ldl': (40, 200),                 # 40-200 mg/dL
        'hdl': (25, 100),                 # 25-100 mg/dL
        'triglycerides': (50, 300),       # 50-300 mg/dL
        'glucose': (60, 300),             # 60-300 mg/dL
        'spo2': (0.90, 1.0),              # 90-100%
    }
    
    # Initialize data
    features_list = []
    cholesterol_values = []
    
    # Generate samples
    for i in range(n_samples):
        if i % 20 == 0:
            print(f"Generating sample {i+1}/{n_samples}")
            
        # Generate correlated lipid values
        cholesterol = np.random.uniform(*biomarker_ranges['total_cholesterol'])
        hdl = np.random.uniform(*biomarker_ranges['hdl'])
        ldl = np.clip(cholesterol * 0.6 + np.random.normal(0, 15), 
                     biomarker_ranges['ldl'][0], biomarker_ranges['ldl'][1])
        triglycerides = np.clip(cholesterol * 0.3 + np.random.normal(0, 25),
                               biomarker_ranges['triglycerides'][0], biomarker_ranges['triglycerides'][1])
        
        # Other biomarkers
        glucose = np.random.uniform(*biomarker_ranges['glucose'])
        spo2 = np.random.uniform(*biomarker_ranges['spo2'])
        
        # Blood content for PPG generation
        blood_content = {
            'total_cholesterol': cholesterol,
            'ldl': ldl,
            'hdl': hdl,
            'triglycerides': triglycerides,
            'glucose': glucose,
            'spo2': spo2,
        }
        
        # Generate PPG signals with varying heart rate
        heart_rate = np.random.uniform(60, 100)
        ppg_signals = {
            'green': generate_test_ppg('green', blood_content, heart_rate),
            'red': generate_test_ppg('red', blood_content, heart_rate),
            'nir': generate_test_ppg('nir', blood_content, heart_rate)
        }
        
        # Extract features
        features = extract_features(ppg_signals)
        features_list.append(features)
        cholesterol_values.append(cholesterol)
    
    # Convert to DataFrame/array
    X = pd.DataFrame(features_list)
    y = np.array(cholesterol_values)
    
    return X, y

class CholesterolPredictor:
    """
    Simple cholesterol predictor using ensemble methods, optimized for lipid prediction.
    """
    def __init__(self, estimator_type='ensemble'):
        self.estimator_type = estimator_type
        self.feature_importances_ = None
        self.important_features_ = None
        self.feature_names_ = None
        self.is_fitted = False
        
        # Choose estimator based on type
        if estimator_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif estimator_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
        else:  # 'ensemble'
            # Create a VotingRegressor-like ensemble manually
            self.rf_model = RandomForestRegressor(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            self.gbm_model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
            self.model = None  # Will use both models for ensemble prediction
    
    def fit(self, X, y):
        """Train the model on training data."""
        print("Training cholesterol prediction model...")
        self.feature_names_ = list(X.columns)
        
        if self.estimator_type == 'ensemble':
            self.rf_model.fit(X, y)
            self.gbm_model.fit(X, y)
            
            # Get feature importance from both models and average them
            rf_importances = self.rf_model.feature_importances_
            gbm_importances = self.gbm_model.feature_importances_
            
            # Normalize and average
            rf_importances = rf_importances / np.sum(rf_importances)
            gbm_importances = gbm_importances / np.sum(gbm_importances)
            self.feature_importances_ = (rf_importances + gbm_importances) / 2
        else:
            self.model.fit(X, y)
            self.feature_importances_ = self.model.feature_importances_
        
        # Create sorted feature importance dataframe
        feature_importance_data = {
            'feature': self.feature_names_,
            'importance': self.feature_importances_
        }
        self.feature_importance_df_ = pd.DataFrame(feature_importance_data)
        self.feature_importance_df_ = self.feature_importance_df_.sort_values(
            'importance', ascending=False).reset_index(drop=True)
        self.important_features_ = self.feature_importance_df_['feature'].tolist()
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.estimator_type == 'ensemble':
            # Average predictions from both models
            rf_preds = self.rf_model.predict(X)
            gbm_preds = self.gbm_model.predict(X)
            return (rf_preds + gbm_preds) / 2
        else:
            return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'predictions': y_pred,
            'true_values': y
        }
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Plot the feature importance of the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
        
        # Get top features
        top_features = self.feature_importance_df_.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features for Cholesterol Prediction')
        plt.gca().invert_yaxis()  # Display top features at the top
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    
    def save(self, path=None):
        """Save the model to disk with metadata."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        if path is None:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.join('data', 'models'), exist_ok=True)
            
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join('data', 'models', f'cholesterol_model_{timestamp}.joblib')
        
        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model object
        joblib.dump(self, path)
        
        # Save metadata
        metadata_path = f"{os.path.splitext(path)[0]}_metadata.json"
        metadata = {
            'biomarker_type': 'total_cholesterol',
            'estimator_type': self.estimator_type,
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_names_),
            'top_10_features': self.important_features_[:10] if self.important_features_ else []
        }
        
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {path}")
        print(f"Metadata saved to {metadata_path}")
        
        return path

def main():
    """Run the cholesterol prediction test."""
    print("\nEnhanced Cholesterol Prediction Test")
    print("="*50)
    
    # Create output directories
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Generate dataset
    X, y = generate_lipid_dataset(n_samples=200)
    print(f"Dataset generated: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    model = CholesterolPredictor(estimator_type='ensemble')
    model.fit(X_train, y_train)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    print("\nTest Results for Cholesterol Prediction:")
    print(f"R² score: {results['r2']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f} mg/dL")
    print(f"Root Mean Squared Error: {results['rmse']:.4f} mg/dL")
    
    # Plot and save feature importance
    model.plot_feature_importance(
        top_n=15, 
        save_path='data/results/cholesterol_feature_importance.png')
    
    # Plot and save predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(results['true_values'], results['predictions'], alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True Cholesterol (mg/dL)')
    plt.ylabel('Predicted Cholesterol (mg/dL)')
    plt.title(f'Cholesterol Prediction\nR² = {results["r2"]:.4f}, MAE = {results["mae"]:.4f} mg/dL')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/results/cholesterol_predictions.png', dpi=300)
    
    # Save model
    model_path = model.save()
    
    return {
        'model': model,
        'results': results,
        'model_path': model_path
    }

if __name__ == "__main__":
    main()
