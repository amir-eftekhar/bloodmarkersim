#!/usr/bin/env python3
"""
Generate synthetic training datasets for the biomarker prediction models.
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import h5py
from tqdm import tqdm

# Add the parent directory to the path to import utility modules
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import GLUCOSE_BASELINE, OXYGEN_SATURATION, LIPIDS_BASELINE
from ml.feature_extraction import extract_features_from_segments


class SyntheticDataGenerator:
    """
    Class for generating synthetic training datasets for biomarker prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the synthetic data generator.
        
        Args:
            config (dict, optional): Configuration parameters.
        """
        self.config = config or self._default_config()
        self.data = None
    
    def _default_config(self):
        """Default configuration for data generation."""
        return {
            'n_samples': 1000,           # Number of samples to generate
            'glucose_range': (70, 200),  # Range of glucose values (mg/dL)
            'spo2_range': (0.88, 1.0),   # Range of SpO2 values (fraction)
            'lipids_ranges': {
                'total_cholesterol': (150, 300),  # mg/dL
                'hdl': (30, 80),                 # mg/dL
                'ldl': (70, 190),                # mg/dL
                'triglycerides': (50, 250)       # mg/dL
            },
            'inflammation_range': (0.1, 5.0),  # Arbitrary units
            'window_size': 4.0,           # Window size in seconds
            'overlap': 0.5,               # Overlap between windows
            'sampling_rate': 100,         # Hz
            'random_seed': 42             # Random seed
        }
    
    def create_biomarker_variations(self):
        """
        Create variations in biomarker values for the synthetic dataset.
        
        Returns:
            pandas.DataFrame: DataFrame of biomarker values.
        """
        np.random.seed(self.config['random_seed'])
        
        # Create DataFrame for biomarkers
        biomarkers = pd.DataFrame()
        
        # Generate random biomarker values
        n = self.config['n_samples']
        
        # Glucose (mg/dL)
        gl_min, gl_max = self.config['glucose_range']
        biomarkers['glucose'] = np.random.uniform(gl_min, gl_max, n)
        
        # SpO2 (fraction)
        spo2_min, spo2_max = self.config['spo2_range']
        biomarkers['spo2'] = np.random.uniform(spo2_min, spo2_max, n)
        
        # Lipids (mg/dL)
        for lipid_type, (min_val, max_val) in self.config['lipids_ranges'].items():
            biomarkers[lipid_type] = np.random.uniform(min_val, max_val, n)
        
        # Inflammation marker (arbitrary units)
        inf_min, inf_max = self.config['inflammation_range']
        biomarkers['inflammation'] = np.random.uniform(inf_min, inf_max, n)
        
        return biomarkers
    
    def generate_waveforms_for_biomarkers(self, biomarkers):
        """
        Generate PPG waveforms for the given biomarker values.
        
        Args:
            biomarkers (pandas.DataFrame): DataFrame of biomarker values.
            
        Returns:
            dict: Dictionary of generated waveforms and features.
        """
        # Import here to avoid circular imports
        from optical.waveforms import PPGSignalGenerator
        
        n_samples = len(biomarkers)
        duration = self.config['window_size']  # seconds
        sampling_rate = self.config['sampling_rate']  # Hz
        
        # Initialize result containers
        all_waveforms = {
            'green': [],
            'red': [],
            'nir': []
        }
        
        time_arrays = []
        
        print(f"Generating {n_samples} synthetic waveform samples...")
        
        # Create mock optical results for the waveform generator
        optical_results = {'detection_rate': 0.15}
        
        # Generate waveforms for each biomarker combination
        for i, row in tqdm(biomarkers.iterrows(), total=n_samples):
            # Create configuration for this sample
            waveform_config = {
                'sampling_rate': sampling_rate,
                'duration': duration,
                'heart_rate': 60 + 20 * np.random.randn(),  # Random heart rate with variation
                'heart_rate_variability': 3 * np.random.random(),  # Random HRV
                'respiratory_rate': 0.2 + 0.05 * np.random.randn(),  # Random respiratory rate
                'noise_level': 0.01 + 0.02 * np.random.random(),  # Random noise level
                'motion_artifacts': np.random.random() < 0.3,  # 30% chance of motion artifacts
                'baseline_drift': np.random.random() < 0.5,  # 50% chance of baseline drift
                'use_cfd_data': False,  # No CFD data for synthetic generation
                'oxygen_saturation': row['spo2'],  # Use the actual SpO2 value
                'glucose_level': row['glucose'],  # Use the actual glucose value
                'lipid_levels': {
                    'total_cholesterol': row['total_cholesterol'],
                    'hdl': row['hdl'],
                    'ldl': row['ldl'],
                    'triglycerides': row['triglycerides']
                },
                'inflammation_marker': row['inflammation'],
                'blood_volumes': {
                    'green': 0.03 + 0.04 * np.random.random(),
                    'red': 0.03 + 0.04 * np.random.random(), 
                    'nir': 0.03 + 0.04 * np.random.random()
                }
            }
            
            # Create waveform generator
            generator = PPGSignalGenerator(optical_results, config=waveform_config)
            
            # Generate waveforms for all wavelengths
            generator.generate_all_signals()
            
            # Store the generated waveforms
            for wavelength in all_waveforms.keys():
                all_waveforms[wavelength].append(generator.ppg_signals[wavelength])
            
            # Store time array
            time_arrays.append(generator.time)
        
        return {
            'waveforms': all_waveforms,
            'time_arrays': time_arrays,
            'biomarkers': biomarkers
        }
    
    def extract_features_from_waveforms(self, data):
        """
        Extract features from the generated waveforms.
        
        Args:
            data (dict): Dictionary containing waveforms and biomarkers.
            
        Returns:
            dict: Dictionary containing features and biomarkers.
        """
        n_samples = len(data['biomarkers'])
        all_features = []
        
        print("Extracting features from waveforms...")
        
        for i in tqdm(range(n_samples)):
            # Extract current waveforms
            current_waveforms = {
                wavelength: data['waveforms'][wavelength][i] 
                for wavelength in ['green', 'red', 'nir']
            }
            
            time = data['time_arrays'][i]
            
            # Extract features from the entire waveform (no segmentation)
            features = extract_features_from_segments(
                current_waveforms, 
                time, 
                window_size=self.config['window_size'],
                overlap=0.0  # No overlap for synthetic data
            )
            
            # If we got multiple segments, just take the first one
            if len(features) > 0:
                features = features.iloc[0].to_dict()
                all_features.append(features)
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(all_features)
        
        return {
            'features': features_df,
            'biomarkers': data['biomarkers']
        }
    
    def generate_dataset(self):
        """
        Generate a complete synthetic dataset for training biomarker prediction models.
        
        Returns:
            dict: Dictionary containing features and biomarkers.
        """
        # Create biomarker variations
        biomarkers = self.create_biomarker_variations()
        
        # Generate waveforms
        waveform_data = self.generate_waveforms_for_biomarkers(biomarkers)
        
        # Extract features
        self.data = self.extract_features_from_waveforms(waveform_data)
        
        print(f"Generated dataset with {len(self.data['features'])} samples and "
              f"{self.data['features'].shape[1]} features.")
        
        return self.data
    
    def save_dataset(self, output_dir):
        """
        Save the generated dataset to files.
        
        Args:
            output_dir (str): Directory to save the dataset.
        """
        if self.data is None:
            print("No dataset to save. Generate dataset first.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save features
        features_file = os.path.join(output_dir, 'features.csv')
        self.data['features'].to_csv(features_file, index=False)
        
        # Save biomarkers
        biomarkers_file = os.path.join(output_dir, 'biomarkers.csv')
        self.data['biomarkers'].to_csv(biomarkers_file, index=False)
        
        print(f"Dataset saved to {output_dir}")
    
    def save_to_h5(self, output_file):
        """
        Save the generated dataset to an HDF5 file.
        
        Args:
            output_file (str): Path to the output HDF5 file.
        """
        if self.data is None:
            print("No dataset to save. Generate dataset first.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with h5py.File(output_file, 'w') as f:
                # Create groups
                features_group = f.create_group('features')
                biomarkers_group = f.create_group('biomarkers')
                
                # Save features
                for column in self.data['features'].columns:
                    features_group.create_dataset(column, data=self.data['features'][column].values)
                
                # Save biomarkers
                for column in self.data['biomarkers'].columns:
                    biomarkers_group.create_dataset(column, data=self.data['biomarkers'][column].values)
                
                # Save metadata
                meta = f.create_group('metadata')
                for key, value in self.config.items():
                    if isinstance(value, (int, float, str, bool)):
                        meta.attrs[key] = value
            
            print(f"Dataset saved to {output_file}")
        except Exception as e:
            print(f"Error saving dataset to HDF5: {e}")


def generate_training_dataset(config=None, output_dir=None):
    """
    Generate a synthetic training dataset for biomarker prediction.
    
    Args:
        config (dict, optional): Configuration parameters.
        output_dir (str, optional): Directory to save the dataset.
        
    Returns:
        dict: Dictionary containing features and biomarkers.
    """
    # Create and run the data generator
    generator = SyntheticDataGenerator(config)
    data = generator.generate_dataset()
    
    # Save the dataset if output directory is provided
    if output_dir:
        generator.save_dataset(output_dir)
        
        # Also save to HDF5
        h5_file = os.path.join(output_dir, 'synthetic_dataset.h5')
        generator.save_to_h5(h5_file)
    
    return data


if __name__ == "__main__":
    # Generate a small synthetic dataset
    config = {
        'n_samples': 100,  # Smaller number for testing
        'window_size': 3.0,
        'sampling_rate': 100,
        'random_seed': 42
    }
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'synthetic')
    
    # Generate dataset
    data = generate_training_dataset(config, output_dir)
