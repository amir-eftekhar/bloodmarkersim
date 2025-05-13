#!/usr/bin/env python3
"""
Enhanced feature extraction from PPG signals for biomarker prediction.
Includes specialized biomarker-specific features, advanced signal processing,
and physiologically meaningful ratios for improved prediction accuracy.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, periodogram


def extract_time_domain_features(ppg_signal, sampling_rate):
    """
    Extract time domain features from a PPG signal.
    
    Args:
        ppg_signal (numpy.ndarray): PPG signal.
        sampling_rate (float): Sampling rate in Hz.
        
    Returns:
        dict: Dictionary of time domain features.
    """
    features = {}
    
    # Basic statistical features
    features['mean'] = np.mean(ppg_signal)
    features['std'] = np.std(ppg_signal)
    features['min'] = np.min(ppg_signal)
    features['max'] = np.max(ppg_signal)
    features['range'] = np.max(ppg_signal) - np.min(ppg_signal)
    features['median'] = np.median(ppg_signal)
    features['skewness'] = skew(ppg_signal)
    features['kurtosis'] = kurtosis(ppg_signal)
    
    # Percentiles
    features['p25'] = np.percentile(ppg_signal, 25)
    features['p75'] = np.percentile(ppg_signal, 75)
    features['iqr'] = features['p75'] - features['p25']
    
    # Find peaks
    peaks, properties = signal.find_peaks(ppg_signal, height=0.4, distance=0.5*sampling_rate)
    
    if len(peaks) > 1:
        # Calculate features from peaks
        peak_heights = properties['peak_heights']
        features['peak_mean'] = np.mean(peak_heights)
        features['peak_std'] = np.std(peak_heights)
        
        # Calculate peak intervals (related to heart rate)
        peak_intervals = np.diff(peaks) / sampling_rate  # in seconds
        features['interval_mean'] = np.mean(peak_intervals)
        features['interval_std'] = np.std(peak_intervals)
        
        # Calculate estimated heart rate
        features['heart_rate'] = 60 / features['interval_mean']  # beats per minute
    else:
        # No peaks found
        features['peak_mean'] = 0
        features['peak_std'] = 0
        features['interval_mean'] = 0
        features['interval_std'] = 0
        features['heart_rate'] = 0
    
    # First and second derivatives
    first_derivative = np.diff(ppg_signal)
    second_derivative = np.diff(first_derivative)
    
    features['first_derivative_mean'] = np.mean(first_derivative)
    features['first_derivative_std'] = np.std(first_derivative)
    features['first_derivative_max'] = np.max(first_derivative)
    features['first_derivative_min'] = np.min(first_derivative)
    
    features['second_derivative_mean'] = np.mean(second_derivative)
    features['second_derivative_std'] = np.std(second_derivative)
    features['second_derivative_max'] = np.max(second_derivative)
    features['second_derivative_min'] = np.min(second_derivative)
    
    # Area under the curve
    features['auc'] = np.trapz(ppg_signal) / len(ppg_signal)
    
    return features


def extract_frequency_domain_features(ppg_signal, sampling_rate):
    """
    Extract frequency domain features from a PPG signal.
    
    Args:
        ppg_signal (numpy.ndarray): PPG signal.
        sampling_rate (float): Sampling rate in Hz.
        
    Returns:
        dict: Dictionary of frequency domain features.
    """
    features = {}
    
    # Calculate FFT
    n = len(ppg_signal)
    fft = np.fft.rfft(ppg_signal)
    fft_magnitude = np.abs(fft)
    
    # Normalize FFT magnitude
    fft_magnitude = fft_magnitude / n
    
    # Calculate frequency bins
    freq_bins = np.fft.rfftfreq(n, d=1/sampling_rate)
    
    # Define frequency bands
    bands = {
        'very_low': (0.0, 0.04),    # Very low frequency
        'low': (0.04, 0.15),        # Low frequency
        'high': (0.15, 0.4),        # High frequency
        'very_high': (0.4, 1.0),    # Very high frequency
        'cardiac': (0.75, 2.0)      # Cardiac frequency (45-120 bpm)
    }
    
    # Calculate power in each frequency band
    for band_name, (low_freq, high_freq) in bands.items():
        band_indices = np.where((freq_bins >= low_freq) & (freq_bins <= high_freq))[0]
        if len(band_indices) > 0:
            band_power = np.sum(fft_magnitude[band_indices] ** 2)
            features[f'{band_name}_power'] = band_power
        else:
            features[f'{band_name}_power'] = 0
    
    # Calculate total power
    total_power = np.sum(fft_magnitude ** 2)
    features['total_power'] = total_power
    
    # Calculate relative power in each band
    for band_name in bands:
        if total_power > 0:
            features[f'{band_name}_relative_power'] = features[f'{band_name}_power'] / total_power
        else:
            features[f'{band_name}_relative_power'] = 0
    
    # Find dominant frequency
    if len(fft_magnitude) > 0:
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freq_bins[dominant_freq_idx]
        features['dominant_frequency_power'] = fft_magnitude[dominant_freq_idx] ** 2
    else:
        features['dominant_frequency'] = 0
        features['dominant_frequency_power'] = 0
    
    # Spectral entropy
    normalized_psd = fft_magnitude ** 2 / np.sum(fft_magnitude ** 2)
    normalized_psd = normalized_psd[normalized_psd > 0]  # Remove zeros to avoid log(0)
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd)) / np.log2(len(normalized_psd))
    features['spectral_entropy'] = spectral_entropy
    
    return features


def extract_wavelet_features(ppg_signal, wavelet='db4', level=5):
    """
    Extract wavelet transform features from a PPG signal.
    
    Args:
        ppg_signal (numpy.ndarray): PPG signal.
        wavelet (str): Wavelet type.
        level (int): Decomposition level.
        
    Returns:
        dict: Dictionary of wavelet features.
    """
    features = {}
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(ppg_signal, wavelet, level=level)
    
    # Extract features from each decomposition level
    for i, coef in enumerate(coeffs):
        if i == 0:
            name = 'approximation'
        else:
            name = f'detail_{i}'
        
        features[f'{name}_mean'] = np.mean(coef)
        features[f'{name}_std'] = np.std(coef)
        features[f'{name}_energy'] = np.sum(coef ** 2) / len(coef)
        features[f'{name}_entropy'] = np.sum(coef ** 2 * np.log(coef ** 2 + 1e-10))
    
    return features


def extract_multi_wavelength_features(ppg_signals, sampling_rate):
    """
    Extract features from multiple wavelength PPG signals, including specialized
    features for specific biomarkers based on physiological principles.
    
    Args:
        ppg_signals (dict): Dictionary of PPG signals for different wavelengths.
        sampling_rate (float): Sampling rate in Hz.
        
    Returns:
        dict: Dictionary of multi-wavelength features.
    """
    features = {}
    
    # First pass: extract basic features for each individual wavelength
    # These will be used later to calculate specialized ratios
    for wavelength, signal in ppg_signals.items():
        # Time domain features
        time_features = extract_time_domain_features(signal, sampling_rate)
        
        # Frequency domain features
        freq_features = extract_frequency_domain_features(signal, sampling_rate)
        
        # Wavelet features
        wavelet_features = extract_wavelet_features(signal)
        
        # Add wavelength prefix to feature names
        for feature_name, value in {**time_features, **freq_features, **wavelet_features}.items():
            features[f'{wavelength}_{feature_name}'] = value
    
    # Calculate specialized ratio features (crucial for accurate biomarker prediction)
    # ==== SpO2-SPECIFIC FEATURES ====
    if 'red' in ppg_signals and 'nir' in ppg_signals:
        red_signal = ppg_signals['red']
        nir_signal = ppg_signals['nir']
        
        # Enhanced peak detection with prominence parameter for better accuracy
        red_peaks, red_props = signal.find_peaks(red_signal, height=0.4, distance=0.5*sampling_rate, prominence=0.1)
        nir_peaks, nir_props = signal.find_peaks(nir_signal, height=0.4, distance=0.5*sampling_rate, prominence=0.1)
        
        red_valleys, _ = signal.find_peaks(-red_signal, distance=0.5*sampling_rate, prominence=0.1)
        nir_valleys, _ = signal.find_peaks(-nir_signal, distance=0.5*sampling_rate, prominence=0.1)
        
        # Standard SpO2 calculations - R ratio is critical for SpO2
        if len(red_peaks) > 0 and len(nir_peaks) > 0 and len(red_valleys) > 0 and len(nir_valleys) > 0:
            # AC components (pulsatile)
            red_ac = np.mean(red_signal[red_peaks]) - np.mean(red_signal[red_valleys])
            nir_ac = np.mean(nir_signal[nir_peaks]) - np.mean(nir_signal[nir_valleys])
            
            # DC components (non-pulsatile)
            red_dc = np.mean(red_signal)
            nir_dc = np.mean(nir_signal)
            
            if red_dc > 0 and nir_dc > 0 and nir_ac > 0:
                # R value used in clinical pulse oximetry
                red_ratio = red_ac / red_dc
                nir_ratio = nir_ac / nir_dc
                
                features['r_ratio'] = red_ratio / nir_ratio if nir_ratio > 0 else 0
                features['modified_r_ratio'] = np.log(red_ratio) / np.log(nir_ratio) if nir_ratio > 0 and red_ratio > 0 else 0
                features['perfusion_index'] = nir_ac / nir_dc * 100  # Perfusion index
                
                # Store individual ratios too
                features['red_ac_dc_ratio'] = red_ratio
                features['nir_ac_dc_ratio'] = nir_ratio
                
                # If we have second derivative (acceleration plethysmogram)
                red_d2 = np.diff(np.diff(red_signal))
                nir_d2 = np.diff(np.diff(nir_signal))
                if len(red_d2) > 0 and len(nir_d2) > 0:
                    # Calculate SDPT (Second Derivative Peak Times) used in some SpO2 algorithms
                    red_d2_peaks, _ = signal.find_peaks(red_d2)
                    nir_d2_peaks, _ = signal.find_peaks(nir_d2)
                    if len(red_d2_peaks) > 1 and len(nir_d2_peaks) > 1:
                        features['red_sdpt'] = np.mean(np.diff(red_d2_peaks))/sampling_rate
                        features['nir_sdpt'] = np.mean(np.diff(nir_d2_peaks))/sampling_rate
                        features['sdpt_ratio'] = features['red_sdpt']/features['nir_sdpt'] if features['nir_sdpt'] > 0 else 0
    
    # ==== GLUCOSE-SPECIFIC FEATURES ====
    if 'nir' in ppg_signals and 'red' in ppg_signals:
        # NIR absorption is affected by glucose levels
        nir_signal = ppg_signals['nir']
        
        # Calculate NIR signal slope features (relevant for glucose)
        nir_slopes = np.diff(nir_signal)
        if len(nir_slopes) > 0:
            features['nir_slope_mean'] = np.mean(nir_slopes)
            features['nir_slope_std'] = np.std(nir_slopes)
            features['nir_rising_slope_mean'] = np.mean(nir_slopes[nir_slopes > 0]) if any(nir_slopes > 0) else 0
            features['nir_falling_slope_mean'] = np.mean(nir_slopes[nir_slopes < 0]) if any(nir_slopes < 0) else 0
            
            # Calculate slope ratio (rising vs falling - glucose affects this asymmetry)
            if any(nir_slopes < 0) and np.mean(nir_slopes[nir_slopes < 0]) != 0:
                features['nir_slope_ratio'] = np.mean(nir_slopes[nir_slopes > 0]) / abs(np.mean(nir_slopes[nir_slopes < 0])) if any(nir_slopes > 0) else 0
            else:
                features['nir_slope_ratio'] = 0
        
        # Near-infrared spectral analysis (critical for glucose detection)
        nir_fft = np.fft.rfft(nir_signal)
        nir_fft_mag = np.abs(nir_fft) / len(nir_signal)
        freq = np.fft.rfftfreq(len(nir_signal), d=1/sampling_rate)
        
        # Specific frequency bands associated with glucose absorption in NIR
        glucose_band_low = np.where((freq >= 0.1) & (freq <= 0.3))[0]  # 0.1-0.3 Hz range
        glucose_band_mid = np.where((freq >= 0.3) & (freq <= 0.8))[0]  # 0.3-0.8 Hz range
        glucose_band_high = np.where((freq >= 0.8) & (freq <= 1.5))[0]  # 0.8-1.5 Hz range
        
        if len(glucose_band_low) > 0 and len(glucose_band_mid) > 0 and len(glucose_band_high) > 0:
            gb_low_power = np.sum(nir_fft_mag[glucose_band_low]**2)
            gb_mid_power = np.sum(nir_fft_mag[glucose_band_mid]**2)
            gb_high_power = np.sum(nir_fft_mag[glucose_band_high]**2)
            
            features['nir_glucose_band_low'] = gb_low_power
            features['nir_glucose_band_mid'] = gb_mid_power
            features['nir_glucose_band_high'] = gb_high_power
            features['nir_glucose_low_mid_ratio'] = gb_low_power / gb_mid_power if gb_mid_power > 0 else 0
            features['nir_glucose_high_mid_ratio'] = gb_high_power / gb_mid_power if gb_mid_power > 0 else 0
            
    # ==== LIPID-SPECIFIC FEATURES ====
    if 'green' in ppg_signals and 'nir' in ppg_signals and 'red' in ppg_signals:
        green_signal = ppg_signals['green']
        nir_signal = ppg_signals['nir']
        red_signal = ppg_signals['red']
        
        # Enhanced lipid detection with more sophisticated features
        # 1. Amplitude ratios (lipid levels affect the relative amplitudes)
        green_amp = np.max(green_signal) - np.min(green_signal)
        red_amp = np.max(red_signal) - np.min(red_signal)
        nir_amp = np.max(nir_signal) - np.min(nir_signal)
        
        features['green_red_amplitude_ratio'] = green_amp / red_amp if red_amp > 0 else 0
        features['nir_red_amplitude_ratio'] = nir_amp / red_amp if red_amp > 0 else 0
        features['green_nir_amplitude_ratio'] = green_amp / nir_amp if nir_amp > 0 else 0
        
        # 2. NEW: Non-linear lipid index based on both time and frequency domain
        # First, get mean absorption levels
        green_mean = np.mean(green_signal)
        red_mean = np.mean(red_signal)
        nir_mean = np.mean(nir_signal)
        
        # Beer-Lambert inspired lipid index
        # Different wavelengths have different absorption coefficients for lipids
        # Green:Red ratio is particularly sensitive to lipid presence
        features['lipid_absorption_index'] = np.log(green_mean/red_mean) if red_mean > 0 else 0
        features['lipid_absorption_index_norm'] = features['lipid_absorption_index'] / np.log(nir_mean/red_mean) if red_mean > 0 and nir_mean > 0 else 0
        
        # 3. NEW: Waveform morphology analysis for lipid detection
        # Find peaks for green signal (most affected by lipids)
        green_peaks, _ = signal.find_peaks(green_signal, height=0.4, distance=0.5*sampling_rate, prominence=0.1)
        if len(green_peaks) > 1:
            # Calculate peak to peak intervals and their variability
            green_peak_intervals = np.diff(green_peaks) / sampling_rate  # Convert to seconds
            features['green_peak_interval_mean'] = np.mean(green_peak_intervals) if len(green_peak_intervals) > 0 else 0
            features['green_peak_interval_std'] = np.std(green_peak_intervals) if len(green_peak_intervals) > 0 else 0
            
            # Calculate area under peaks (indicators of lipid content)
            if len(green_peaks) > 0 and len(green_signal) > 0:
                peak_width = int(0.2 * sampling_rate)  # ~200ms around peak
                green_areas = []
                for peak in green_peaks:
                    start = max(0, peak - peak_width//2)
                    end = min(len(green_signal), peak + peak_width//2)
                    if start < end:  # Ensure valid range
                        area = np.trapz(green_signal[start:end])
                        green_areas.append(area)
                
                if green_areas:
                    features['green_peak_area_mean'] = np.mean(green_areas)
                    features['green_peak_area_std'] = np.std(green_areas)
                    
                    # NEW: Area ratio between green and red (sensitive to lipid content)
                    red_areas = []
                    for peak in green_peaks:  # Use same peaks for alignment
                        start = max(0, peak - peak_width//2)
                        end = min(len(red_signal), peak + peak_width//2)
                        if start < end:  # Ensure valid range
                            area = np.trapz(red_signal[start:end])
                            red_areas.append(area)
                    
                    if red_areas:
                        features['green_red_area_ratio'] = np.mean(green_areas) / np.mean(red_areas) if np.mean(red_areas) > 0 else 0
        
        # 4. Advanced spectral analysis for lipid detection
        green_fft = np.fft.rfft(green_signal)
        green_fft_mag = np.abs(green_fft) / len(green_signal)
        red_fft = np.fft.rfft(red_signal)
        red_fft_mag = np.abs(red_fft) / len(red_signal)
        nir_fft = np.fft.rfft(nir_signal)
        nir_fft_mag = np.abs(nir_fft) / len(nir_signal)
        
        # NEW: Enhanced frequency bands for lipid detection
        # Research shows lipids affect specific frequency ranges
        lipid_band1 = np.where((freq >= 0.4) & (freq <= 0.8))[0]  # Adjusted for lipid detection
        lipid_band2 = np.where((freq >= 0.8) & (freq <= 1.5))[0]  # Adjusted for lipid detection
        lipid_band3 = np.where((freq >= 1.5) & (freq <= 3.0))[0]  # Higher frequency components
        
        if len(lipid_band1) > 0 and len(lipid_band2) > 0 and len(lipid_band3) > 0:
            # Power in different bands
            green_lb1 = np.sum(green_fft_mag[lipid_band1]**2)
            green_lb2 = np.sum(green_fft_mag[lipid_band2]**2)
            green_lb3 = np.sum(green_fft_mag[lipid_band3]**2)
            
            red_lb1 = np.sum(red_fft_mag[lipid_band1]**2)
            red_lb2 = np.sum(red_fft_mag[lipid_band2]**2)
            red_lb3 = np.sum(red_fft_mag[lipid_band3]**2)
            
            nir_lb1 = np.sum(nir_fft_mag[lipid_band1]**2)
            nir_lb2 = np.sum(nir_fft_mag[lipid_band2]**2)
            nir_lb3 = np.sum(nir_fft_mag[lipid_band3]**2)
            
            # Basic band ratios
            features['green_lipid_band_ratio'] = green_lb1 / green_lb2 if green_lb2 > 0 else 0
            features['red_lipid_band_ratio'] = red_lb1 / red_lb2 if red_lb2 > 0 else 0
            
            # NEW: Multi-band lipid indicators
            features['green_lipid_triband_ratio'] = green_lb1 / (green_lb2 + green_lb3) if (green_lb2 + green_lb3) > 0 else 0
            features['red_lipid_triband_ratio'] = red_lb1 / (red_lb2 + red_lb3) if (red_lb2 + red_lb3) > 0 else 0
            features['nir_lipid_triband_ratio'] = nir_lb1 / (nir_lb2 + nir_lb3) if (nir_lb2 + nir_lb3) > 0 else 0
            
            # NEW: Enhanced cross-wavelength lipid indicators
            features['green_red_lipid_ratio'] = features['green_lipid_band_ratio'] / features['red_lipid_band_ratio'] if features['red_lipid_band_ratio'] > 0 else 0
            
            # NEW: Combined tri-band cross-wavelength lipid index
            green_red_nir_index = (green_lb1 * red_lb2 * nir_lb3) / (green_lb3 * red_lb1 * nir_lb2) if (green_lb3 * red_lb1 * nir_lb2) > 0 else 0
            features['lipid_spectral_index'] = np.cbrt(green_red_nir_index) if green_red_nir_index > 0 else 0  # Cube root for scaling
    
    # ==== HEMOGLOBIN-SPECIFIC FEATURES ====
    if 'green' in ppg_signals and 'red' in ppg_signals:
        green_signal = ppg_signals['green']
        red_signal = ppg_signals['red']
        
        # Advanced hemoglobin detection based on optical absorption physics
        # Hemoglobin has distinctive absorption patterns in green and red wavelengths
        
        # 1. Enhanced peak detection with prominence parameters
        green_peaks, green_peak_props = signal.find_peaks(green_signal, height=0.4, distance=0.5*sampling_rate, prominence=0.1)
        red_peaks, red_peak_props = signal.find_peaks(red_signal, height=0.4, distance=0.5*sampling_rate, prominence=0.1)
        
        green_valleys, green_valley_props = signal.find_peaks(-green_signal, distance=0.5*sampling_rate, prominence=0.1)
        red_valleys, red_valley_props = signal.find_peaks(-red_signal, distance=0.5*sampling_rate, prominence=0.1)
        
        if len(green_peaks) > 0 and len(red_peaks) > 0 and len(green_valleys) > 0 and len(red_valleys) > 0:
            # 2. Improved AC components calculation
            # Use peak and valley values more precisely with robust statistics
            green_peak_values = green_signal[green_peaks]
            green_valley_values = green_signal[green_valleys]
            red_peak_values = red_signal[red_peaks]
            red_valley_values = red_signal[red_valleys]
            
            # Use median for robustness against outliers
            green_ac = np.median(green_peak_values) - np.median(green_valley_values)
            red_ac = np.median(red_peak_values) - np.median(red_valley_values)
            
            # DC components with filtering for stable baseline
            green_dc = np.median(green_signal)
            red_dc = np.median(red_signal)
            
            # 3. NEW: Advanced hemoglobin-sensitive feature set
            if green_dc > 0 and red_dc > 0:
                # Basic optical ratios
                features['green_ac_dc_ratio'] = green_ac / green_dc
                features['red_ac_dc_ratio'] = red_ac / red_dc
                
                # NEW: The classic hemoglobin index
                features['hemoglobin_index'] = (green_ac / green_dc) / (red_ac / red_dc) if (red_ac / red_dc) > 0 else 0
                
                # NEW: Non-linear hemoglobin estimation based on Beer-Lambert Law
                # Hemoglobin has ~10x higher absorption in green vs red
                # A*(log(green_dc) - k*log(red_dc)) where k is a wavelength-dependent constant
                features['hemoglobin_absorption_index'] = np.log(green_dc) - 0.1 * np.log(red_dc) if red_dc > 0 else 0
                
                # NEW: Normalized index (more robust across different baseline conditions)
                max_green = np.max(green_signal)
                min_green = np.min(green_signal)
                max_red = np.max(red_signal)
                min_red = np.min(red_signal)
                
                if (max_green > min_green) and (max_red > min_red):
                    norm_green = (green_signal - min_green) / (max_green - min_green)
                    norm_red = (red_signal - min_red) / (max_red - min_red)
                    
                    # Area under normalized curves (sensitive to hemoglobin)
                    green_area = np.trapz(norm_green)
                    red_area = np.trapz(norm_red)
                    features['hemoglobin_area_ratio'] = green_area / red_area if red_area > 0 else 0
                
                # NEW: Advanced hemoglobin metrics from second derivatives
                # Second derivatives reveal subtle changes in waveform morphology
                # that correlate strongly with hemoglobin levels
                green_d2 = np.diff(np.diff(green_signal))
                red_d2 = np.diff(np.diff(red_signal))
                
                if len(green_d2) > 0 and len(red_d2) > 0:
                    # Find inflection points - highly sensitive to blood properties
                    green_d2_peaks, _ = signal.find_peaks(green_d2)
                    red_d2_peaks, _ = signal.find_peaks(red_d2)
                    green_d2_valleys, _ = signal.find_peaks(-green_d2)
                    red_d2_valleys, _ = signal.find_peaks(-red_d2)
                    
                    # Count inflection points (related to oxygen carrying capacity)
                    features['green_inflection_count'] = len(green_d2_peaks) + len(green_d2_valleys)
                    features['red_inflection_count'] = len(red_d2_peaks) + len(red_d2_valleys)
                    features['inflection_ratio'] = features['green_inflection_count'] / features['red_inflection_count'] if features['red_inflection_count'] > 0 else 0
                    
                    # Energy in second derivatives
                    green_d2_energy = np.sum(green_d2 ** 2) / len(green_d2)
                    red_d2_energy = np.sum(red_d2 ** 2) / len(red_d2)
                    features['d2_energy_ratio'] = green_d2_energy / red_d2_energy if red_d2_energy > 0 else 0
        
        # 4. NEW: Phase analysis for hemoglobin estimation
        # Hemoglobin affects the phase relationship between green and red signals
        if len(green_signal) > 1 and len(red_signal) > 1:
            # Cross-correlation to measure phase relationship
            cross_corr = signal.correlate(green_signal, red_signal, mode='full')
            lags = signal.correlation_lags(len(green_signal), len(red_signal), mode='full')
            lag = lags[np.argmax(cross_corr)]
            
            # Normalized and scaled to be physiologically meaningful
            features['green_red_phase_lag_ms'] = lag * (1000 / sampling_rate)  # Convert to milliseconds
            
            # NEW: Frequency domain phase relationship
            green_fft = np.fft.rfft(green_signal)
            red_fft = np.fft.rfft(red_signal)
            
            # Calculate phase angles
            green_phase = np.angle(green_fft)
            red_phase = np.angle(red_fft)
            
            # Phase difference in cardiac frequency range
            cardiac_freq_idx = np.where((freq >= 0.5) & (freq <= 2.0))[0]  # 0.5-2.0 Hz (30-120 BPM)
            if len(cardiac_freq_idx) > 0:
                phase_diff = green_phase[cardiac_freq_idx] - red_phase[cardiac_freq_idx]
                # Normalize phase differences to be within -π to π
                phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                
                # Phase difference at peak cardiac frequency
                cardiac_power = np.abs(green_fft[cardiac_freq_idx]) * np.abs(red_fft[cardiac_freq_idx])
                if np.sum(cardiac_power) > 0:
                    weighted_phase = np.sum(phase_diff * cardiac_power) / np.sum(cardiac_power)
                    features['hemoglobin_phase_index'] = weighted_phase
    
    # Calculate general ratios between all wavelength pairs for completeness
    for w1, w2 in [('green', 'red'), ('green', 'nir'), ('red', 'nir')]:
        if w1 in ppg_signals and w2 in ppg_signals:
            # Mean ratios
            features[f'{w1}_{w2}_mean_ratio'] = (
                np.mean(ppg_signals[w1]) / np.mean(ppg_signals[w2])
                if np.mean(ppg_signals[w2]) > 0 else 0
            )
            
            # Energy ratios
            energy_w1 = np.sum(ppg_signals[w1] ** 2)
            energy_w2 = np.sum(ppg_signals[w2] ** 2)
            features[f'{w1}_{w2}_energy_ratio'] = (
                energy_w1 / energy_w2 if energy_w2 > 0 else 0
            )
            
            # Phase and correlation features
            features[f'{w1}_{w2}_correlation'] = np.corrcoef(ppg_signals[w1], ppg_signals[w2])[0, 1]
            
            # Calculate cross-power spectral density features
            f, Pxy = signal.csd(ppg_signals[w1], ppg_signals[w2], fs=sampling_rate, nperseg=min(256, len(ppg_signals[w1])))
            features[f'{w1}_{w2}_csd_max'] = np.max(np.abs(Pxy)) if len(Pxy) > 0 else 0
            
            # Calculate mutual information approximation using histogram method
            hist_2d, x_edges, y_edges = np.histogram2d(ppg_signals[w1], ppg_signals[w2], bins=20)
            p_xy = hist_2d / float(np.sum(hist_2d))
            p_xy = p_xy[p_xy > 0]  # Use only non-zero probabilities for log
            features[f'{w1}_{w2}_entropy'] = -np.sum(p_xy * np.log(p_xy))
    
    return features


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply bandpass filter to data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_physiological_ratios(ppg_signals):
    """Calculate physiologically relevant ratios between signals"""
    ratios = {}
    # Implement clinical ratio calculations - example for SpO2
    if 'red' in ppg_signals and 'nir' in ppg_signals:
        red_min = np.min(ppg_signals['red'])
        red_max = np.max(ppg_signals['red'])
        nir_min = np.min(ppg_signals['nir'])
        nir_max = np.max(ppg_signals['nir'])
        
        red_range = red_max - red_min
        nir_range = nir_max - nir_min
        
        if nir_range > 0:
            # Standard R value calculation for SpO2 estimation
            r_value = red_range / nir_range
            ratios['r_value'] = r_value
            # Empirical SpO2 estimation formula
            ratios['estimated_spo2'] = 110 - 25 * r_value
    return ratios

def extract_all_features(ppg_signals, time_data, sampling_rate):
    """
    Extract comprehensive features from PPG signals with biomarker-specific elements.
    
    Args:
        ppg_signals (dict): Dictionary of PPG signals for different wavelengths.
        time_data (numpy.ndarray): Time data for PPG signals.
        sampling_rate (float): Sampling rate in Hz.
        
    Returns:
        pandas.DataFrame: DataFrame of extracted features.
    """
    all_features = {}
    
    # Preprocess signals - apply filtering to clean the signals
    filtered_signals = {}
    for wavelength, signal in ppg_signals.items():
        # Apply bandpass filter to remove noise and baseline wander
        # 0.5-8Hz covers cardiac and respiratory components while removing noise
        filtered_signals[wavelength] = bandpass_filter(signal, 0.5, 8.0, sampling_rate)
    
    # Extract standard time domain features (with filtered signals)
    for wavelength, signal in filtered_signals.items():
        time_features = extract_time_domain_features(signal, sampling_rate)
        all_features.update({f'{wavelength}_{name}': value for name, value in time_features.items()})
    
    # Extract frequency domain features
    for wavelength, signal in filtered_signals.items():
        freq_features = extract_frequency_domain_features(signal, sampling_rate)
        all_features.update({f'{wavelength}_{name}': value for name, value in freq_features.items()})
    
    # Extract wavelet features
    for wavelength, signal in filtered_signals.items():
        wavelet_features = extract_wavelet_features(signal)
        all_features.update({f'{wavelength}_{name}': value for name, value in wavelet_features.items()})
    
    # Extract specialized multi-wavelength features (critical for biomarker prediction)
    multi_features = extract_multi_wavelength_features(filtered_signals, sampling_rate)
    all_features.update(multi_features)
    
    # Calculate physiological ratios
    physio_ratios = calculate_physiological_ratios(filtered_signals)
    all_features.update(physio_ratios)
    
    # Convert to DataFrame
    return pd.DataFrame([all_features])


def segment_signal(signal, time, window_size, overlap=0.5):
    """
    Segment a signal into windows for feature extraction.
    
    Args:
        signal (numpy.ndarray): Signal to segment.
        time (numpy.ndarray): Time data for the signal.
        window_size (float): Window size in seconds.
        overlap (float): Overlap between consecutive windows, as a fraction.
        
    Returns:
        tuple: (segments, segment_times) - List of signal segments and corresponding time segments.
    """
    sampling_rate = 1 / (time[1] - time[0])
    samples_per_window = int(window_size * sampling_rate)
    step_size = int(samples_per_window * (1 - overlap))
    
    segments = []
    segment_times = []
    
    for i in range(0, len(signal) - samples_per_window + 1, step_size):
        segment = signal[i:i+samples_per_window]
        segment_time = time[i:i+samples_per_window]
        
        segments.append(segment)
        segment_times.append(segment_time)
    
    return segments, segment_times


def extract_features_from_segments(ppg_signals, time_data, window_size=4.0, overlap=0.5):
    """
    Extract features from segmented PPG signals with enhanced preprocessing.
    
    Args:
        ppg_signals (dict): Dictionary of PPG signals for different wavelengths.
        time_data (numpy.ndarray): Time data for PPG signals.
        window_size (float): Window size in seconds.
        overlap (float): Overlap between consecutive windows, as a fraction.
        
    Returns:
        pandas.DataFrame: DataFrame of extracted features for each segment.
    """
    sampling_rate = 1 / (time_data[1] - time_data[0])
    all_features = []
    
    # Apply bandpass filtering to full signals before segmentation
    filtered_signals = {}
    for wavelength, signal in ppg_signals.items():
        filtered_signals[wavelength] = bandpass_filter(signal, 0.5, 8.0, sampling_rate)
    
    # Get segments for each wavelength
    segmented_signals = {}
    segment_times = None
    
    for wavelength, signal in filtered_signals.items():
        segments, times = segment_signal(signal, time_data, window_size, overlap)
        segmented_signals[wavelength] = segments
        
        if segment_times is None:
            segment_times = times
    
    # Extract features for each segment
    for i in range(len(segment_times)):
        segment_ppg = {wavelength: segments[i] for wavelength, segments in segmented_signals.items()}
        segment_time = segment_times[i]
        
        # Extract individual wavelength features
        segment_features = {}
        
        # Time domain features for each wavelength
        for wavelength, signal in segment_ppg.items():
            time_features = extract_time_domain_features(signal, sampling_rate)
            segment_features.update({f'{wavelength}_{name}': value for name, value in time_features.items()})
        
        # Frequency domain features for each wavelength
        for wavelength, signal in segment_ppg.items():
            freq_features = extract_frequency_domain_features(signal, sampling_rate)
            segment_features.update({f'{wavelength}_{name}': value for name, value in freq_features.items()})
        
        # Wavelet features for each wavelength
        for wavelength, signal in segment_ppg.items():
            wavelet_features = extract_wavelet_features(signal)
            segment_features.update({f'{wavelength}_{name}': value for name, value in wavelet_features.items()})
        
        # Extract specialized multi-wavelength features
        multi_features = extract_multi_wavelength_features(segment_ppg, sampling_rate)
        segment_features.update(multi_features)
        
        # Calculate physiological ratios
        physio_ratios = calculate_physiological_ratios(segment_ppg)
        segment_features.update(physio_ratios)
        
        # Add segment metadata
        segment_features['segment_start'] = segment_time[0]
        segment_features['segment_end'] = segment_time[-1]
        segment_features['segment_duration'] = segment_time[-1] - segment_time[0]
        
        all_features.append(segment_features)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    return df


if __name__ == "__main__":
    # Test feature extraction
    import matplotlib.pyplot as plt
    
    # Generate synthetic PPG signals
    time = np.arange(0, 10, 0.01)
    sampling_rate = 100  # Hz
    
    # Create synthetic signals for different wavelengths
    ppg_green = 0.5 * np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 0.2 * time) + 0.1 * np.random.randn(len(time))
    ppg_red = 0.4 * np.sin(2 * np.pi * 1.2 * time) + 0.3 * np.sin(2 * np.pi * 0.25 * time) + 0.1 * np.random.randn(len(time))
    ppg_nir = 0.6 * np.sin(2 * np.pi * 1.2 * time) + 0.1 * np.sin(2 * np.pi * 0.18 * time) + 0.1 * np.random.randn(len(time))
    
    ppg_signals = {
        'green': ppg_green,
        'red': ppg_red,
        'nir': ppg_nir
    }
    
    # Extract features
    features_df = extract_all_features(ppg_signals, time, sampling_rate)
    print(features_df.shape)
    print(features_df.columns[:10])  # Print first 10 columns
    
    # Extract features from segments
    segment_features = extract_features_from_segments(ppg_signals, time, window_size=2.0)
    print(f"Number of segments: {len(segment_features)}")
    print(f"Number of features per segment: {segment_features.shape[1]}")
    
    # Plot signals and segments
    plt.figure(figsize=(12, 8))
    
    for wavelength, signal in ppg_signals.items():
        plt.plot(time, signal, label=wavelength)
    
    # Mark segmentation boundaries
    for i, row in segment_features.iterrows():
        plt.axvline(x=row['segment_start'], color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('PPG Signals and Segments')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
