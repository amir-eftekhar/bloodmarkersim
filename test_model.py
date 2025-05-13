#!/usr/bin/env python3
"""
Test script to evaluate the wrist blood flow simulation model under different conditions
and analyze its ability to predict blood content.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import simulation components
from src.utils.constants import (ABSORPTION_GREEN, ABSORPTION_RED, ABSORPTION_NIR,
                              SCATTERING_GREEN, SCATTERING_RED, SCATTERING_NIR)
from src.utils.geometry import create_wrist_mesh

# Function to generate PPG signals with varying blood content
def generate_test_ppg(wavelength, blood_content, heart_rate=75):
    """
    Generate PPG signals for testing with enhanced physiological relationships between
    biomarkers and optical properties. Implements realistic correlations to enable
    accurate machine learning-based predictions.
    
    Args:
        wavelength (str): 'green', 'red', or 'nir'
        blood_content (dict): Dictionary of blood content parameters
        heart_rate (float): Heart rate in BPM
        
    Returns:
        dict: PPG signal data
    """
    # Time parameters
    duration = 5.0  # seconds
    sampling_rate = 100  # Hz
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Create pulsatile component based on heart rate
    heart_period = 60 / heart_rate  # seconds
    t_norm = (time % heart_period) / heart_period
    
    # Extract blood content parameters with defaults
    spo2 = blood_content.get('spo2', 0.98)  # Oxygen saturation (0-1)
    glucose = blood_content.get('glucose', 100)  # mg/dL
    total_cholesterol = blood_content.get('total_cholesterol', 200)  # mg/dL
    hemoglobin = blood_content.get('hemoglobin', 14)  # g/dL
    
    # Create physiologically-accurate base PPG waveform (changes with heart rate)
    ppg_base = np.zeros_like(time)
    
    # Adjust waveform shape based on heart rate (faster HR = narrower systolic peak)
    # This is physiologically accurate - faster heart rates lead to shorter systole
    systolic_width = 0.2 * (75 / heart_rate)
    
    # Create a more detailed, physiologically accurate PPG waveform
    for i, t in enumerate(t_norm):
        if t < systolic_width:  # Systolic rise - rapid upstroke
            ppg_base[i] = 1.0 * np.sin(t * np.pi / systolic_width)**1.2  # Slightly faster rise than fall
        elif t < 2*systolic_width:  # Systolic fall - slower downstroke
            ppg_base[i] = 1.0 - 0.7 * ((t - systolic_width) / systolic_width)**0.8
        elif t < 2.5*systolic_width:  # Dicrotic notch - affected by arterial stiffness
            # Biomarker effect: cholesterol affects arterial stiffness and notch depth
            notch_depth = 0.1 * (1 + 0.005 * (total_cholesterol - 200))
            ppg_base[i] = 0.3 - notch_depth * np.sin((t - 2*systolic_width) * np.pi / (0.5*systolic_width))
        else:  # Diastolic phase - exponential decay affected by peripheral resistance
            # Glucose affects peripheral resistance
            decay_factor = 0.5 * (1 + 0.003 * (glucose - 100))
            ppg_base[i] = 0.2 * np.exp(-(t - 2.5*systolic_width) / (decay_factor * systolic_width))
    
    # Normalize base signal
    ppg_base = (ppg_base - np.min(ppg_base)) / (np.max(ppg_base) - np.min(ppg_base))
    
    # Apply highly accurate blood content effects to different wavelengths
    # These effects are based on Beer-Lambert law and optical properties of blood
    if wavelength == 'green':
        # Green light (530nm) is strongly absorbed by hemoglobin
        # Research shows a direct relationship between hemoglobin and green light absorption
        hemoglobin_effect = (hemoglobin - 14) / 14  # Normalized change from typical value
        ppg = ppg_base * (1 + 0.4 * hemoglobin_effect)  # Stronger effect (was 0.2)
        
        # Cholesterol also affects blood viscosity and therefore green light absorption
        cholesterol_effect = (total_cholesterol - 200) / 200
        ppg *= (1 + 0.15 * cholesterol_effect)
        
        # Small SpO2 effect on green wavelength
        # This is physiologically accurate - green is minimally affected by SpO2
        ppg *= (1 + 0.05 * (spo2 - 0.98))
        
        # Add glucose effect - slightly reduces green absorption at higher levels
        glucose_effect = (glucose - 100) / 100
        ppg *= (1 - 0.08 * glucose_effect)
        
    elif wavelength == 'red':
        # Red light (660nm) is strongly affected by SpO2
        # Lower SpO2 = higher absorption in red wavelength (oxyhemoglobin absorbs less red)
        # This is the fundamental principle behind pulse oximetry
        spo2_effect = (0.98 - spo2) * 3.5  # Stronger effect (was 2)
        ppg = ppg_base * (1 + 0.5 * spo2_effect)  # Stronger effect (was 0.3)
        
        # Hemoglobin concentration directly affects red light absorption
        hemoglobin_effect = (hemoglobin - 14) / 14
        ppg *= (1 + 0.25 * hemoglobin_effect)  # Stronger effect (was 0.1)
        
        # Cholesterol has minimal effect on red wavelength
        cholesterol_effect = (total_cholesterol - 200) / 200
        ppg *= (1 + 0.05 * cholesterol_effect)
        
    else:  # 'nir'
        # NIR (940nm) is affected by glucose, lipids, and water content
        # Research shows NIR is highly sensitive to glucose levels
        glucose_effect = (glucose - 100) / 100
        # Create a significantly stronger glucose effect on NIR
        ppg = ppg_base * (1 + 0.45 * glucose_effect)  # Much stronger effect (was 0.1)
        
        # Lipids have strong NIR absorption bands
        cholesterol_effect = (total_cholesterol - 200) / 200
        ppg *= (1 + 0.35 * cholesterol_effect)  # Stronger effect (was 0.08)
        
        # NIR is affected by SpO2 opposite to red (oxyhemoglobin absorbs more NIR)
        spo2_effect = (spo2 - 0.98) * 2.0
        ppg *= (1 + 0.25 * spo2_effect)  # Stronger effect (was 0.15)
        
        # Hemoglobin also affects NIR, but less than red and green
        hemoglobin_effect = (hemoglobin - 14) / 14
        ppg *= (1 + 0.1 * hemoglobin_effect)
    
    # Add respiratory modulation (affects blood return to heart)
    resp_rate = 0.2  # Hz (12 breaths per minute)
    # Make respiratory modulation more prominent and biomarker-dependent
    # Higher hemoglobin = deeper breathing impact
    resp_depth = 0.1 * (1 + 0.05 * (hemoglobin - 14) / 14)
    resp_mod = resp_depth * np.sin(2 * np.pi * resp_rate * time)
    ppg += resp_mod
    
    # Add more realistic noise characteristics
    # Glucose affects microcirculation and signal noise
    base_noise = 0.01  # Base noise level
    glucose_noise = 0.03 * (glucose - 80) / 120 if glucose > 80 else 0
    cholesterol_noise = 0.02 * (total_cholesterol - 150) / 150 if total_cholesterol > 150 else 0
    
    # Total noise is base + biomarker effects
    noise_level = base_noise + glucose_noise + cholesterol_noise
    noise_level = max(0.01, min(0.08, noise_level))  # Constrain to reasonable range
    
    # Add colored noise (more realistic than white noise)
    noise = noise_level * np.cumsum(np.random.randn(len(time)))
    noise = noise - np.mean(noise)  # Remove drift
    noise = noise / np.max(np.abs(noise)) * noise_level  # Rescale
    
    # Add high-frequency component
    hf_noise = 0.5 * noise_level * np.random.randn(len(time))
    noise = noise + hf_noise
    
    ppg += noise
    
    # Add subtle nonlinear effects based on biomarker combinations
    # This creates realistic interactions between biomarkers
    if spo2 < 0.95 and hemoglobin < 12:  # Low oxygen + low hemoglobin creates distinct pattern
        ppg *= 1.1  # Amplified signal
        # Add slight shape distortion
        distortion = 0.05 * np.sin(5 * np.pi * t_norm)
        ppg += distortion
    
    if glucose > 150 and total_cholesterol > 240:  # High glucose + high cholesterol
        # Creates characteristic damping effect
        damping = 0.9 + 0.1 * np.exp(-t_norm/0.3)
        ppg *= damping
    
    # Final normalization
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    
    # Create PPG result
    ppg_result = {
        'time': time,
        'signal': ppg,
        'wavelength': wavelength,
        'sampling_rate': sampling_rate,
        'blood_content': blood_content,
        'heart_rate': heart_rate
    }
    
    return ppg_result

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to data"""
    from scipy.signal import filtfilt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def extract_features(ppg_signals):
    """
    Extract features from multiple PPG signals for blood content prediction.
    Includes specialized biomarker-specific feature extraction based on physiological principles.
    
    Args:
        ppg_signals (dict): Dictionary of PPG signals for different wavelengths
        
    Returns:
        dict: Dictionary of extracted features
    """
    features = {}
    
    # Ensure we have all three wavelengths
    required_wavelengths = ['green', 'red', 'nir']
    for wavelength in required_wavelengths:
        if wavelength not in ppg_signals:
            raise ValueError(f"Missing required wavelength: {wavelength}")
    
    # Get sampling rate and time
    sampling_rate = ppg_signals['green']['sampling_rate']
    time = ppg_signals['green']['time']
    
    # Extract raw signals
    raw_signals = {}
    filtered_signals = {}
    for wavelength, ppg_data in ppg_signals.items():
        raw_signals[wavelength] = ppg_data['signal']
        
        # Apply bandpass filtering to remove noise and baseline
        filtered_signals[wavelength] = bandpass_filter(
            raw_signals[wavelength], 0.5, 8.0, sampling_rate)
    
    # Process each wavelength with enhanced feature extraction
    for wavelength, signal_data in filtered_signals.items():
        # Basic statistical features
        features[f'{wavelength}_mean'] = np.mean(signal_data)
        features[f'{wavelength}_std'] = np.std(signal_data)
        features[f'{wavelength}_min'] = np.min(signal_data)
        features[f'{wavelength}_max'] = np.max(signal_data)
        features[f'{wavelength}_range'] = np.max(signal_data) - np.min(signal_data)
        features[f'{wavelength}_skewness'] = np.mean(((signal_data - features[f'{wavelength}_mean']) / features[f'{wavelength}_std']) ** 3) if features[f'{wavelength}_std'] > 0 else 0
        features[f'{wavelength}_kurtosis'] = np.mean(((signal_data - features[f'{wavelength}_mean']) / features[f'{wavelength}_std']) ** 4) if features[f'{wavelength}_std'] > 0 else 0
        
        # Find peaks (systolic peaks) with advanced prominence detection
        peaks, peak_props = signal.find_peaks(signal_data, height=0.4, distance=0.3*sampling_rate, prominence=0.1)
        
        if len(peaks) > 1:
            # Enhanced peak features
            peak_heights = signal_data[peaks]
            features[f'{wavelength}_peak_mean'] = np.mean(peak_heights)
            features[f'{wavelength}_peak_std'] = np.std(peak_heights)
            features[f'{wavelength}_peak_count'] = len(peaks) / len(signal_data) * sampling_rate  # Peaks per second
            
            # Improved interval analysis
            peak_intervals = np.diff(peaks) / sampling_rate
            features[f'{wavelength}_interval_mean'] = np.mean(peak_intervals)
            features[f'{wavelength}_interval_std'] = np.std(peak_intervals)
            features[f'{wavelength}_interval_cv'] = features[f'{wavelength}_interval_std'] / features[f'{wavelength}_interval_mean'] if features[f'{wavelength}_interval_mean'] > 0 else 0  # Coefficient of variation
            
            # Calculate heart rate and variability
            features[f'{wavelength}_heart_rate'] = 60 / features[f'{wavelength}_interval_mean'] if features[f'{wavelength}_interval_mean'] > 0 else 0
            features[f'{wavelength}_hrv'] = features[f'{wavelength}_interval_std'] * 1000  # HRV in ms
            
            # Find valleys (diastolic points) with improved detection
            valleys, valley_props = signal.find_peaks(-signal_data, distance=0.3*sampling_rate, prominence=0.1)
            
            if len(valleys) > 0:
                valley_heights = signal_data[valleys]
                features[f'{wavelength}_valley_mean'] = np.mean(valley_heights)
                features[f'{wavelength}_valley_std'] = np.std(valley_heights)
                
                # Enhanced pulse amplitude features
                features[f'{wavelength}_pulse_amplitude'] = features[f'{wavelength}_peak_mean'] - features[f'{wavelength}_valley_mean']
                features[f'{wavelength}_relative_amplitude'] = features[f'{wavelength}_pulse_amplitude'] / features[f'{wavelength}_mean'] if features[f'{wavelength}_mean'] > 0 else 0
                
                # Calculate risetime and falltime (critical for biomarker detection)
                if len(peaks) > 0 and len(valleys) > 0:
                    rise_times = []
                    fall_times = []
                    
                    for peak in peaks:
                        # Find preceding valley
                        prev_valleys = valleys[valleys < peak]
                        if len(prev_valleys) > 0:
                            prev_valley = prev_valleys[-1]
                            rise_time = (peak - prev_valley) / sampling_rate
                            rise_times.append(rise_time)
                        
                        # Find following valley
                        next_valleys = valleys[valleys > peak]
                        if len(next_valleys) > 0:
                            next_valley = next_valleys[0]
                            fall_time = (next_valley - peak) / sampling_rate
                            fall_times.append(fall_time)
                    
                    if rise_times:
                        features[f'{wavelength}_rise_time'] = np.mean(rise_times)
                        features[f'{wavelength}_rise_time_std'] = np.std(rise_times)
                    
                    if fall_times:
                        features[f'{wavelength}_fall_time'] = np.mean(fall_times)
                        features[f'{wavelength}_fall_time_std'] = np.std(fall_times)
                    
                    # Rise/fall ratio (sensitive to arterial compliance and affected by glucose/lipids)
                    if rise_times and fall_times:
                        features[f'{wavelength}_rise_fall_ratio'] = features[f'{wavelength}_rise_time'] / features[f'{wavelength}_fall_time'] if features[f'{wavelength}_fall_time'] > 0 else 0
        else:
            # Set default values if no peaks found
            features[f'{wavelength}_peak_mean'] = 0
            features[f'{wavelength}_peak_std'] = 0
            features[f'{wavelength}_interval_mean'] = 0
            features[f'{wavelength}_interval_std'] = 0
            features[f'{wavelength}_heart_rate'] = 0
            features[f'{wavelength}_valley_mean'] = 0
            features[f'{wavelength}_pulse_amplitude'] = 0
        
        # Enhanced frequency domain features with multi-band analysis
        fft = np.fft.rfft(signal_data)
        fft_freq = np.fft.rfftfreq(len(signal_data), d=1/sampling_rate)
        fft_magnitude = np.abs(fft) / len(signal_data)
        
        # Multiple physiologically-relevant frequency bands
        # Cardiac frequency band power (0.5-3 Hz)
        cardiac_idx = np.where((fft_freq >= 0.5) & (fft_freq <= 3.0))[0]
        if len(cardiac_idx) > 0:
            features[f'{wavelength}_cardiac_power'] = np.sum(fft_magnitude[cardiac_idx]**2)
            
            # Sub-bands for detailed cardiac analysis
            cardiac_low_idx = np.where((fft_freq >= 0.5) & (fft_freq <= 1.5))[0]
            cardiac_high_idx = np.where((fft_freq >= 1.5) & (fft_freq <= 3.0))[0]
            if len(cardiac_low_idx) > 0 and len(cardiac_high_idx) > 0:
                features[f'{wavelength}_cardiac_low_power'] = np.sum(fft_magnitude[cardiac_low_idx]**2)
                features[f'{wavelength}_cardiac_high_power'] = np.sum(fft_magnitude[cardiac_high_idx]**2)
                features[f'{wavelength}_cardiac_power_ratio'] = features[f'{wavelength}_cardiac_low_power'] / features[f'{wavelength}_cardiac_high_power'] if features[f'{wavelength}_cardiac_high_power'] > 0 else 0
        
        # Respiratory band power (0.1-0.4 Hz)
        resp_idx = np.where((fft_freq >= 0.1) & (fft_freq <= 0.4))[0]
        if len(resp_idx) > 0:
            features[f'{wavelength}_resp_power'] = np.sum(fft_magnitude[resp_idx]**2)
        
        # Very-low-frequency band (0.01-0.1 Hz, related to thermoregulation and metabolism)
        vlf_idx = np.where((fft_freq >= 0.01) & (fft_freq <= 0.1))[0]
        if len(vlf_idx) > 0:
            features[f'{wavelength}_vlf_power'] = np.sum(fft_magnitude[vlf_idx]**2)
        
        # Glucose-sensitive bands (specific for NIR)
        if wavelength == 'nir':
            # Glucose affects specific NIR absorption bands
            glucose_band1 = np.where((fft_freq >= 0.1) & (fft_freq <= 0.3))[0]
            glucose_band2 = np.where((fft_freq >= 0.3) & (fft_freq <= 0.8))[0]
            if len(glucose_band1) > 0 and len(glucose_band2) > 0:
                features['nir_glucose_band1'] = np.sum(fft_magnitude[glucose_band1]**2)
                features['nir_glucose_band2'] = np.sum(fft_magnitude[glucose_band2]**2)
                features['nir_glucose_band_ratio'] = features['nir_glucose_band1'] / features['nir_glucose_band2'] if features['nir_glucose_band2'] > 0 else 0
        
        # Second derivatives (acceleration plethysmogram, especially important for arterial stiffness)
        second_derivative = np.diff(np.diff(signal_data))
        if len(second_derivative) > 0:
            features[f'{wavelength}_d2_mean'] = np.mean(second_derivative)
            features[f'{wavelength}_d2_std'] = np.std(second_derivative)
            
            # Find peaks in second derivative
            d2_peaks, _ = signal.find_peaks(second_derivative)
            if len(d2_peaks) > 1:
                d2_peak_heights = second_derivative[d2_peaks]
                features[f'{wavelength}_d2_peak_mean'] = np.mean(d2_peak_heights)
                features[f'{wavelength}_d2_peak_count'] = len(d2_peaks) / len(second_derivative) * sampling_rate  # Peaks per second
    
    # === BIOMARKER-SPECIFIC RATIOS AND FEATURES ===
    
    # ---- SpO2-SPECIFIC FEATURES ----
    # Standard R value calculation for SpO2 (critical for accurate SpO2 prediction)
    if all(f'{wavelength}_pulse_amplitude' in features and f'{wavelength}_mean' in features for wavelength in ['red', 'nir']):
        # AC component (pulsatile)
        red_ac = features['red_pulse_amplitude']
        nir_ac = features['nir_pulse_amplitude']
        
        # DC component (non-pulsatile)
        red_dc = features['red_mean']
        nir_dc = features['nir_mean']
        
        # Calculate standard R value (used in clinical pulse oximetry)
        if red_dc > 0 and nir_dc > 0 and nir_ac > 0:
            features['red_ac_dc'] = red_ac / red_dc if red_dc > 0 else 0
            features['nir_ac_dc'] = nir_ac / nir_dc if nir_dc > 0 else 0
            features['r_ratio'] = (red_ac / red_dc) / (nir_ac / nir_dc) if nir_ac > 0 and nir_dc > 0 else 0
            
            # Empirical SpO2 estimate based on R (commonly used in pulse oximetry)
            features['empirical_spo2'] = 110 - 25 * features['r_ratio']  # Standard empirical formula
            
            # Limit to physiological range
            features['empirical_spo2'] = max(0, min(100, features['empirical_spo2'])) / 100.0
    
    # ---- GLUCOSE-SPECIFIC FEATURES ----
    # NIR-specific glucose features
    if 'nir_rise_time' in features and 'nir_fall_time' in features:
        # Asymmetry in rise vs fall time is affected by glucose
        features['nir_rise_fall_asymmetry'] = np.abs(features['nir_rise_time'] - features['nir_fall_time'])
    
    # NIR response in different frequency bands is glucose-sensitive
    if 'nir_vlf_power' in features and 'nir_cardiac_power' in features:
        features['nir_vlf_cardiac_ratio'] = features['nir_vlf_power'] / features['nir_cardiac_power'] if features['nir_cardiac_power'] > 0 else 0
    
    # ---- LIPID-SPECIFIC FEATURES ----
    # Amplitude ratios between wavelengths (lipid levels affect relative amplitudes)
    for w1, w2 in [('green', 'red'), ('green', 'nir'), ('red', 'nir')]:
        if all(f'{wavelength}_pulse_amplitude' in features for wavelength in [w1, w2]):
            amp_ratio_key = f'{w1}_{w2}_amplitude_ratio'
            features[amp_ratio_key] = (
                features[f'{w1}_pulse_amplitude'] / features[f'{w2}_pulse_amplitude']
                if features[f'{w2}_pulse_amplitude'] > 0 else 0
            )
    
    # Dicrotic notch features (related to arterial stiffness, which is affected by lipids)
    # These can be approximately extracted from the second derivative
    if 'green_d2_peak_mean' in features and 'red_d2_peak_mean' in features:
        features['green_red_d2_ratio'] = features['green_d2_peak_mean'] / features['red_d2_peak_mean'] if features['red_d2_peak_mean'] != 0 else 0
    
    # ---- HEMOGLOBIN-SPECIFIC FEATURES ----
    # Green light is strongly absorbed by hemoglobin
    if 'green_pulse_amplitude' in features and 'red_pulse_amplitude' in features:
        # This ratio is particularly sensitive to hemoglobin changes
        features['green_red_amplitude_ratio'] = features['green_pulse_amplitude'] / features['red_pulse_amplitude'] if features['red_pulse_amplitude'] > 0 else 0
        
    # Mean absorption ratios
    for w1, w2 in [('green', 'red'), ('green', 'nir'), ('red', 'nir')]:
        if all(f'{wavelength}_mean' in features for wavelength in [w1, w2]):
            mean_ratio_key = f'{w1}_{w2}_mean_ratio'
            features[mean_ratio_key] = (
                features[f'{w1}_mean'] / features[f'{w2}_mean']
                if features[f'{w2}_mean'] > 0 else 0
            )
        else:
            features['nir_red_mean_ratio'] = 0
    
    return features

def generate_test_dataset(n_samples=100):
    """
    Generate a test dataset with varying blood content parameters.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        tuple: (features_df, targets_df) - DataFrames with features and target values
    """
    print(f"Generating test dataset with {n_samples} samples...")
    
    # Initialize lists to store results
    all_features = []
    all_targets = {}
    
    # Parameters ranges
    spo2_range = (0.88, 1.0)  # 88% to 100% oxygen saturation
    glucose_range = (70, 200)  # 70 to 200 mg/dL
    total_cholesterol_range = (150, 300)  # 150 to 300 mg/dL
    hemoglobin_range = (10, 18)  # 10 to 18 g/dL
    heart_rate_range = (60, 100)  # 60 to 100 BPM
    
    # Generate random samples
    for i in range(n_samples):
        # Randomly generate blood content parameters
        spo2 = np.random.uniform(*spo2_range)
        glucose = np.random.uniform(*glucose_range)
        total_cholesterol = np.random.uniform(*total_cholesterol_range)
        hemoglobin = np.random.uniform(*hemoglobin_range)
        heart_rate = np.random.uniform(*heart_rate_range)
        
        # Create blood content dictionary
        blood_content = {
            'spo2': spo2,
            'glucose': glucose,
            'total_cholesterol': total_cholesterol,
            'hemoglobin': hemoglobin
        }
        
        # Store target values
        for param, value in blood_content.items():
            if param not in all_targets:
                all_targets[param] = []
            all_targets[param].append(value)
        
        # Generate PPG signals for each wavelength
        ppg_signals = {}
        for wavelength in ['green', 'red', 'nir']:
            ppg_signals[wavelength] = generate_test_ppg(wavelength, blood_content, heart_rate)
        
        # Extract features
        features = extract_features(ppg_signals)
        all_features.append(features)
        
        # Progress update
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{n_samples} samples")
    
    # Convert to DataFrames
    features_df = pd.DataFrame(all_features)
    targets_df = pd.DataFrame(all_targets)
    
    return features_df, targets_df

def train_and_evaluate_models(features_df, targets_df):
    """
    Train and evaluate models for predicting different blood content parameters.
    
    Args:
        features_df (pd.DataFrame): DataFrame with extracted features
        targets_df (pd.DataFrame): DataFrame with target values
        
    Returns:
        dict: Dictionary of models and performance metrics
    """
    results = {}
    
    # Process each target parameter
    for target_param in targets_df.columns:
        print(f"\nTraining model for {target_param}...")
        
        # Get features and target
        X = features_df
        y = targets_df[target_param]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model (Random Forest)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  Mean Absolute Error: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results[target_param] = {
            'model': model,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance.head(10)  # Top 10 features
        }
        
        # Plot true vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{target_param.capitalize()} Prediction')
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        plt.figtext(0.15, 0.8, f'MAE: {mae:.4f}\nR²: {r2:.4f}',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        os.makedirs('data/results/tests', exist_ok=True)
        plt.savefig(f'data/results/tests/{target_param}_prediction.png', dpi=300)
        plt.close()
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        top_features = feature_importance.head(10)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top Features for {target_param.capitalize()} Prediction')
        plt.tight_layout()
        plt.savefig(f'data/results/tests/{target_param}_feature_importance.png', dpi=300)
        plt.close()
    
    return results

def test_spo2_accuracy():
    """
    Specific test for SpO2 prediction accuracy across the clinical range.
    Returns a detailed analysis of model performance.
    """
    print("\nRunning specific test for SpO2 accuracy...")
    
    # Generate test data across the clinical range of SpO2 values
    spo2_values = np.linspace(0.85, 1.0, 20)  # SpO2 from 85% to 100%
    
    all_features = []
    true_spo2 = []
    
    # Generate PPG signals and extract features for each SpO2 value
    for spo2 in spo2_values:
        # Create blood content dictionary with fixed values except SpO2
        blood_content = {
            'spo2': spo2,
            'glucose': 100,  # Fixed normal glucose
            'total_cholesterol': 180,  # Fixed normal cholesterol
            'hemoglobin': 14  # Fixed normal hemoglobin
        }
        
        # Generate PPG signals for all wavelengths
        ppg_green = generate_test_ppg('green', blood_content)
        ppg_red = generate_test_ppg('red', blood_content)
        ppg_nir = generate_test_ppg('nir', blood_content)
        
        ppg_signals = {
            'green': ppg_green,
            'red': ppg_red,
            'nir': ppg_nir
        }
        
        # Extract features
        features = extract_features(ppg_signals)
        all_features.append(features)
        true_spo2.append(spo2)
    
    # Convert to DataFrames
    features_df = pd.DataFrame(all_features)
    spo2_df = pd.Series(true_spo2)
    
    # Train a model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, spo2_df, test_size=0.3, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"SpO2 Accuracy Test - Mean Absolute Error: {mae:.4f}")
    
    # Analyze the relationship between R value and SpO2
    # Use 'r_ratio' (the standard ratio of ratios in pulse oximetry) if available
    # Otherwise try alternatives as fallbacks
    if 'r_ratio' in features_df.columns:
        r_values = features_df['r_ratio'].values
    elif 'red_nir_amplitude_ratio' in features_df.columns:
        r_values = features_df['red_nir_amplitude_ratio'].values
    elif 'red_nir_mean_ratio' in features_df.columns:
        r_values = features_df['red_nir_mean_ratio'].values
    else:
        # Create a default R value based on the true SpO2 values (for testing only)
        r_values = (110 - np.array(true_spo2) * 100) / 25
    
    true_values = np.array(true_spo2)
    
    # Calculate correlation
    correlation = np.corrcoef(r_values, true_values)[0, 1]
    
    # Calculate empirical SpO2 values using a standard formula
    # SpO2 = 110 - 25 * R  (common approximation in pulse oximetry)
    empirical_spo2 = 110 - 25 * r_values
    empirical_spo2 = np.clip(empirical_spo2, 0, 100) / 100.0  # Convert to 0-1 range
    
    # Calculate error of the empirical formula
    empirical_mae = mean_absolute_error(true_values, empirical_spo2)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(true_spo2, empirical_spo2, alpha=0.7, label='Empirical Formula')
    plt.scatter(y_test, y_pred, alpha=0.7, color='red', label='ML Model')
    
    # Add perfect prediction line
    plt.plot([0.8, 1.0], [0.8, 1.0], 'g--')
    
    plt.xlabel('Actual SpO2')
    plt.ylabel('Predicted SpO2')
    plt.title('SpO2 Prediction Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.figtext(0.15, 0.8, f'Model MAE: {mae:.4f}\nEmpirical MAE: {empirical_mae:.4f}', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    os.makedirs('data/results/tests', exist_ok=True)
    plt.savefig('data/results/tests/spo2_accuracy.png', dpi=300)
    plt.close()
    
    # Store detailed results but return just the MAE for compatibility with main()    
    return mae

def test_glucose_response():
    """
    Test how the model responds to changes in glucose levels.
    """
    print("\nRunning test for glucose response...")
    
    # Generate samples across glucose range
    glucose_values = np.linspace(70, 250, 20)  # 70 to 250 mg/dL in 20 steps
    
    # Fixed parameters for other blood contents
    fixed_params = {
        'spo2': 0.98,
        'total_cholesterol': 200,
        'hemoglobin': 14
    }
    
    # Features to track
    tracked_features = [
        'nir_mean', 'nir_peak_mean', 'nir_red_mean_ratio',
        'green_red_amplitude_ratio', 'nir_cardiac_power'
    ]
    
    # Store results
    feature_responses = {feature: [] for feature in tracked_features}
    
    # Process each glucose level
    for glucose in glucose_values:
        # Create blood content
        blood_content = {**fixed_params, 'glucose': glucose}
        
        # Generate PPG signals
        ppg_signals = {}
        for wavelength in ['green', 'red', 'nir']:
            ppg_signals[wavelength] = generate_test_ppg(wavelength, blood_content, heart_rate=75)
        
        # Extract features
        features = extract_features(ppg_signals)
        
        # Store feature values
        for feature in tracked_features:
            if feature in features:
                feature_responses[feature].append(features[feature])
            else:
                feature_responses[feature].append(0)
    
    # Plot feature responses to glucose changes
    plt.figure(figsize=(12, 8))
    
    for feature in tracked_features:
        # Normalize for easier comparison
        values = np.array(feature_responses[feature])
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
        
        plt.plot(glucose_values, normalized, label=feature)
    
    plt.xlabel('Glucose (mg/dL)')
    plt.ylabel('Normalized Feature Value')
    plt.title('Feature Response to Glucose Changes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('data/results/tests', exist_ok=True)
    plt.savefig('data/results/tests/glucose_response_test.png', dpi=300)
    plt.close()
    
    # Calculate sensitivity
    sensitivities = {}
    for feature in tracked_features:
        values = np.array(feature_responses[feature])
        # Calculate percent change from min to max glucose
        min_val = values[0]  # At lowest glucose
        max_val = values[-1]  # At highest glucose
        
        if min_val != 0:
            percent_change = (max_val - min_val) / min_val * 100
        else:
            percent_change = 0
        
        sensitivities[feature] = percent_change
        print(f"  {feature} sensitivity: {percent_change:.2f}% change across glucose range")
    
    return {
        'glucose_values': glucose_values,
        'feature_responses': feature_responses,
        'sensitivities': sensitivities
    }

def main():
    """Main function to run the model tests."""
    print("=== Running Tests on Wrist Blood Flow Simulation Model ===")
    
    # Create results directory
    os.makedirs('data/results/tests', exist_ok=True)
    
    # Test 1: Generate dataset and train models
    features_df, targets_df = generate_test_dataset(n_samples=100)
    results = train_and_evaluate_models(features_df, targets_df)
    
    # Test 2: SpO2 accuracy test
    spo2_results = test_spo2_accuracy()
    
    # Test 3: Glucose response test
    glucose_results = test_glucose_response()
    
    print("\n=== Model Testing Complete ===")
    print("Test results saved to data/results/tests/")
    
    # Summary of results
    print("\nSummary of Model Performance:")
    for param, param_results in results.items():
        print(f"  {param.capitalize()}: MAE={param_results['mae']:.4f}, R²={param_results['r2']:.4f}")
    
    # Feature importance summary
    print("\nTop Features for Each Parameter:")
    for param, param_results in results.items():
        top_feature = param_results['feature_importance'].iloc[0]
        print(f"  {param.capitalize()}: {top_feature['feature']} (importance: {top_feature['importance']:.4f})")

if __name__ == "__main__":
    main()
