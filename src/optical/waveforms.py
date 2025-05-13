#!/usr/bin/env python3
"""
Generate PPG waveforms from optical simulation results with blood flow dynamics.
This module creates synthetic PPG signals by combining optical and CFD simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Add the parent directory to the path to import utility modules
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import HEART_RATE, CARDIAC_PERIOD


class PPGSignalGenerator:
    """
    Class to generate synthetic PPG signals from Monte Carlo and CFD simulations.
    """
    
    def __init__(self, optical_results, cfd_results=None, config=None):
        """
        Initialize the PPG signal generator.
        
        Args:
            optical_results (dict): Results from Monte Carlo optical simulation.
            cfd_results (dict, optional): Results from CFD blood flow simulation.
            config (dict, optional): Configuration parameters.
        """
        self.optical_results = optical_results
        self.cfd_results = cfd_results
        self.config = config or self._default_config()
        
        # PPG signal properties
        self.sampling_rate = self.config['sampling_rate']  # Hz
        self.duration = self.config['duration']  # seconds
        self.time = np.arange(0, self.duration, 1/self.sampling_rate)
        
        # Initialize signals
        self.ppg_signals = {}
        self.detected_intensity = {}
        self.noise_level = self.config['noise_level']
        
    def _default_config(self):
        """Default configuration for PPG signal generation."""
        return {
            'sampling_rate': 100,  # Hz
            'duration': 10,        # seconds
            'heart_rate': HEART_RATE,  # bpm
            'heart_rate_variability': 3,  # bpm variation
            'respiratory_rate': 0.2,  # Hz
            'noise_level': 0.02,   # Fraction of signal amplitude
            'motion_artifacts': True,
            'baseline_drift': True,
            'use_cfd_data': True,  # Whether to incorporate CFD data
            'oxygen_saturation': 0.98,  # 98% baseline
            'blood_volumes': {
                'green': 0.05,  # Fraction of tissue volume that is blood
                'red': 0.05,
                'nir': 0.05
            }
        }
    
    def _generate_base_waveform(self, wavelength):
        """
        Generate a basic PPG waveform for a specific wavelength.
        
        Args:
            wavelength (str): Wavelength band ('green', 'red', or 'nir').
            
        Returns:
            numpy.ndarray: Base PPG waveform.
        """
        # Vary the heart rate slightly over time
        if self.config['heart_rate_variability'] > 0:
            heart_rate = self.config['heart_rate'] + \
                       self.config['heart_rate_variability'] * \
                       np.sin(2 * np.pi * self.config['respiratory_rate'] * self.time)
        else:
            heart_rate = np.ones_like(self.time) * self.config['heart_rate']
        
        # Calculate instantaneous period
        instantaneous_period = 60 / heart_rate
        
        # Calculate phase
        phase = np.zeros_like(self.time)
        phase[0] = 0
        for i in range(1, len(self.time)):
            dt = self.time[i] - self.time[i-1]
            phase[i] = phase[i-1] + dt * (2 * np.pi / instantaneous_period[i-1])
            phase[i] = phase[i] % (2 * np.pi)
        
        # Generate base waveform
        # Different wavelengths have different morphologies
        if wavelength == 'green':
            # Green has strong systolic peak, weak dicrotic notch
            waveform = 0.8 * np.exp(-((phase - 0.3*np.pi) / 0.4)**2) + \
                     0.2 * np.exp(-((phase - 0.8*np.pi) / 0.3)**2)
        elif wavelength == 'red':
            # Red has moderate systolic peak, more pronounced dicrotic notch
            waveform = 0.7 * np.exp(-((phase - 0.3*np.pi) / 0.5)**2) + \
                     0.3 * np.exp(-((phase - 0.8*np.pi) / 0.4)**2)
        else:  # 'nir'
            # NIR has broader peaks
            waveform = 0.6 * np.exp(-((phase - 0.3*np.pi) / 0.6)**2) + \
                     0.4 * np.exp(-((phase - 0.8*np.pi) / 0.5)**2)
        
        # Normalize
        waveform = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform))
        
        return waveform
    
    def _add_blood_volume_variations(self, base_waveform, wavelength):
        """
        Add variations based on blood volume changes.
        
        Args:
            base_waveform (numpy.ndarray): Base PPG waveform.
            wavelength (str): Wavelength band.
            
        Returns:
            numpy.ndarray: Modified waveform with blood volume variations.
        """
        # Base blood volume
        blood_volume_base = self.config['blood_volumes'][wavelength]
        
        # Scale waveform by wavelength-specific factors
        if wavelength == 'green':
            # Green has highest absorption by blood
            sensitivity = 1.0
        elif wavelength == 'red':
            # Red has less absorption
            sensitivity = 0.6
        else:  # 'nir'
            # NIR has least absorption by blood
            sensitivity = 0.4
        
        # Calculate absorption based on blood volume variations
        # Beer-Lambert law: I = I0 * exp(-μ * d)
        # Where μ is the absorption coefficient and d is the optical path length
        # Blood volume variations affect the effective path length
        
        # Modified blood volume over time
        blood_volume = blood_volume_base * (1 + sensitivity * (base_waveform - 0.5))
        
        # Use modified waveform
        modified_waveform = 1 - blood_volume  # Inverted to match PPG convention
        
        # Normalize
        modified_waveform = (modified_waveform - np.min(modified_waveform)) / (np.max(modified_waveform) - np.min(modified_waveform))
        
        return modified_waveform
    
    def _incorporate_cfd_data(self, waveform, wavelength):
        """
        Incorporate CFD data into the waveform if available.
        
        Args:
            waveform (numpy.ndarray): Base PPG waveform.
            wavelength (str): Wavelength band.
            
        Returns:
            numpy.ndarray: Waveform modified by CFD data.
        """
        if not self.cfd_results or not self.config['use_cfd_data']:
            return waveform
        
        try:
            # Extract velocity data from CFD results
            cfd_time = self.cfd_results.get('time', [])
            velocity_profiles = self.cfd_results.get('velocity_profiles', [])
            
            if not cfd_time or len(cfd_time) < 2:
                print("Insufficient CFD data to incorporate into waveform.")
                return waveform
            
            # Get mean velocity over time (average across the vessel)
            mean_velocities = [np.mean(profile) for profile in velocity_profiles]
            
            # Interpolate CFD data to match PPG time points
            # Make interpolation cyclical by extending data
            n_cycles = int(np.ceil(self.duration / CARDIAC_PERIOD))
            extended_time = np.array([])
            extended_velocity = np.array([])
            
            for i in range(n_cycles):
                cycle_time = np.array(cfd_time) + i * CARDIAC_PERIOD
                extended_time = np.append(extended_time, cycle_time)
                extended_velocity = np.append(extended_velocity, mean_velocities)
            
            # Create interpolation function
            velocity_interp = interp1d(extended_time, extended_velocity, 
                                      bounds_error=False, fill_value="extrapolate")
            
            # Sample at PPG time points
            interpolated_velocity = velocity_interp(self.time)
            
            # Normalize
            normalized_velocity = (interpolated_velocity - np.min(interpolated_velocity)) / \
                                (np.max(interpolated_velocity) - np.min(interpolated_velocity))
            
            # Blend with base waveform
            if wavelength == 'green':
                blend_factor = 0.7  # Green is strongly affected by blood volume/velocity
            elif wavelength == 'red':
                blend_factor = 0.5  # Red is moderately affected
            else:  # 'nir'
                blend_factor = 0.3  # NIR is less affected
            
            # Combine base waveform with CFD data
            modified_waveform = (1 - blend_factor) * waveform + blend_factor * normalized_velocity
            
            # Re-normalize
            modified_waveform = (modified_waveform - np.min(modified_waveform)) / \
                             (np.max(modified_waveform) - np.min(modified_waveform))
            
            return modified_waveform
            
        except Exception as e:
            print(f"Error incorporating CFD data: {e}")
            return waveform
    
    def _add_baseline_drift(self, waveform):
        """
        Add baseline drift to the waveform.
        
        Args:
            waveform (numpy.ndarray): Base PPG waveform.
            
        Returns:
            numpy.ndarray: Waveform with baseline drift.
        """
        if not self.config['baseline_drift']:
            return waveform
        
        # Create slow oscillations for baseline drift
        respiration = 0.05 * np.sin(2 * np.pi * self.config['respiratory_rate'] * self.time)
        very_low_freq = 0.03 * np.sin(2 * np.pi * 0.05 * self.time)  # 0.05 Hz
        ultra_low_freq = 0.02 * np.sin(2 * np.pi * 0.01 * self.time)  # 0.01 Hz
        
        # Combine with waveform
        modified_waveform = waveform + respiration + very_low_freq + ultra_low_freq
        
        # Re-normalize
        modified_waveform = (modified_waveform - np.min(modified_waveform)) / \
                         (np.max(modified_waveform) - np.min(modified_waveform))
        
        return modified_waveform
    
    def _add_motion_artifacts(self, waveform):
        """
        Add motion artifacts to the waveform.
        
        Args:
            waveform (numpy.ndarray): Base PPG waveform.
            
        Returns:
            numpy.ndarray: Waveform with motion artifacts.
        """
        if not self.config['motion_artifacts']:
            return waveform
        
        # Generate random motion artifacts
        n_artifacts = np.random.poisson(2)  # Poisson distributed number of artifacts
        
        modified_waveform = waveform.copy()
        
        for _ in range(n_artifacts):
            # Random timing of artifact
            artifact_time = np.random.uniform(0.1, 0.9) * self.duration
            artifact_duration = np.random.uniform(0.2, 1.0)  # seconds
            
            # Generate artifact shape
            artifact_start = max(0, int((artifact_time - artifact_duration/2) * self.sampling_rate))
            artifact_end = min(len(self.time), int((artifact_time + artifact_duration/2) * self.sampling_rate))
            
            artifact_shape = np.zeros_like(waveform)
            t = np.linspace(-np.pi, np.pi, artifact_end - artifact_start)
            artifact_shape[artifact_start:artifact_end] = 0.2 * np.sin(t) * (1 + np.random.rand())
            
            # Add artifact to signal
            modified_waveform += artifact_shape
        
        # Re-normalize
        modified_waveform = (modified_waveform - np.min(modified_waveform)) / \
                         (np.max(modified_waveform) - np.min(modified_waveform))
        
        return modified_waveform
    
    def _add_noise(self, waveform):
        """
        Add realistic sensor noise to the waveform.
        
        Args:
            waveform (numpy.ndarray): Base PPG waveform.
            
        Returns:
            numpy.ndarray: Noisy waveform.
        """
        # White noise
        white_noise = np.random.normal(0, self.noise_level, len(waveform))
        
        # High-frequency noise (electronic)
        t = np.arange(len(waveform))
        hf_noise = 0.3 * self.noise_level * np.sin(2 * np.pi * 50 * t / self.sampling_rate)  # 50 Hz noise
        
        # Combine noise components
        total_noise = white_noise + hf_noise
        
        # Add to waveform
        noisy_waveform = waveform + total_noise
        
        return noisy_waveform
    
    def _apply_biomarker_effects(self, waveform, wavelength):
        """
        Modify the waveform based on biomarker levels.
        
        Args:
            waveform (numpy.ndarray): Base PPG waveform.
            wavelength (str): Wavelength band.
            
        Returns:
            numpy.ndarray: Waveform modified by biomarker effects.
        """
        modified_waveform = waveform.copy()
        
        # Effect of oxygen saturation
        if 'oxygen_saturation' in self.config:
            spo2 = self.config['oxygen_saturation']
            
            # O2 saturation affects the relative absorption at different wavelengths
            if wavelength == 'red':
                # Red is more affected by deoxygenated blood
                o2_effect = 0.1 * (1 - spo2)
                modified_waveform = modified_waveform * (1 - o2_effect)
            elif wavelength == 'nir':
                # NIR is less affected by deoxygenation
                o2_effect = 0.05 * (1 - spo2)
                modified_waveform = modified_waveform * (1 + o2_effect)
        
        # Effect of glucose level (primarily affects NIR)
        if 'glucose_level' in self.config:
            glucose = self.config['glucose_level']
            glucose_baseline = 100  # mg/dL
            
            if wavelength == 'nir':
                # Simplified model: higher glucose slightly increases NIR absorption
                glucose_effect = 0.01 * (glucose - glucose_baseline) / 100
                modified_waveform = modified_waveform * (1 + glucose_effect)
        
        # Effect of lipid levels (primarily affects NIR)
        if 'lipid_levels' in self.config:
            lipids = self.config['lipid_levels']['total_cholesterol']
            lipid_baseline = 200  # mg/dL
            
            if wavelength == 'nir':
                # Simplified model: higher lipids slightly increases NIR absorption
                lipid_effect = 0.01 * (lipids - lipid_baseline) / 100
                modified_waveform = modified_waveform * (1 + lipid_effect)
        
        # Re-normalize if necessary
        if np.max(modified_waveform) - np.min(modified_waveform) > 0:
            modified_waveform = (modified_waveform - np.min(modified_waveform)) / \
                             (np.max(modified_waveform) - np.min(modified_waveform))
        
        return modified_waveform
    
    def filter_signal(self, signal, lowcut=0.5, highcut=8.0):
        """
        Apply bandpass filtering to the signal.
        
        Args:
            signal (numpy.ndarray): Input signal.
            lowcut (float): Low cutoff frequency in Hz.
            highcut (float): High cutoff frequency in Hz.
            
        Returns:
            numpy.ndarray: Filtered signal.
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Create butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def generate_ppg_signal(self, wavelength):
        """
        Generate a complete PPG signal for a specific wavelength.
        
        Args:
            wavelength (str): Wavelength band ('green', 'red', or 'nir').
            
        Returns:
            numpy.ndarray: Generated PPG signal.
        """
        print(f"Generating PPG signal for {wavelength} wavelength...")
        
        # Generate base waveform
        waveform = self._generate_base_waveform(wavelength)
        
        # Add blood volume variations
        waveform = self._add_blood_volume_variations(waveform, wavelength)
        
        # Incorporate CFD data if available
        waveform = self._incorporate_cfd_data(waveform, wavelength)
        
        # Apply biomarker effects
        waveform = self._apply_biomarker_effects(waveform, wavelength)
        
        # Add baseline drift
        waveform = self._add_baseline_drift(waveform)
        
        # Add motion artifacts
        waveform = self._add_motion_artifacts(waveform)
        
        # Add noise
        waveform = self._add_noise(waveform)
        
        # Store raw signal
        self.ppg_signals[wavelength] = waveform
        
        # Calculate detected intensity based on optical simulation
        # This is a simplified model relating absorption to detected intensity
        detection_rate = self.optical_results.get('detection_rate', 0.1)
        
        # Scale intensity based on detection rate from optical simulation
        wavelength_factors = {'green': 0.3, 'red': 0.6, 'nir': 1.0}  # Relative detection efficiency
        base_intensity = detection_rate * wavelength_factors[wavelength]
        
        # Modulate intensity with the waveform
        self.detected_intensity[wavelength] = base_intensity * (1 - waveform)
        
        return waveform
    
    def generate_all_signals(self):
        """
        Generate PPG signals for all wavelengths.
        
        Returns:
            dict: Dictionary of generated PPG signals for each wavelength.
        """
        wavelengths = ['green', 'red', 'nir']
        
        for wavelength in wavelengths:
            self.generate_ppg_signal(wavelength)
        
        return self.ppg_signals
    
    def visualize_signals(self, save_dir=None):
        """
        Visualize the generated PPG signals.
        
        Args:
            save_dir (str, optional): Directory to save visualizations. If None, figures will be displayed.
        """
        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Plot all wavelengths together
        plt.figure(figsize=(12, 8))
        
        if 'green' in self.ppg_signals:
            plt.plot(self.time, self.ppg_signals['green'], 'g-', label='Green', alpha=0.7)
        if 'red' in self.ppg_signals:
            plt.plot(self.time, self.ppg_signals['red'], 'r-', label='Red', alpha=0.7)
        if 'nir' in self.ppg_signals:
            plt.plot(self.time, self.ppg_signals['nir'], 'k-', label='NIR', alpha=0.7)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.title('PPG Signals at Different Wavelengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'ppg_signals.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # Plot each wavelength with detected intensity
        for wavelength in self.ppg_signals:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(self.time, self.ppg_signals[wavelength], label='PPG Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Normalized Amplitude')
            plt.title(f'{wavelength.capitalize()} PPG Signal')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(self.time, self.detected_intensity[wavelength], label='Detected Intensity')
            plt.xlabel('Time (s)')
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'{wavelength.capitalize()} Detected Light Intensity')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'ppg_signal_{wavelength}.png'), dpi=300)
                plt.close()
            else:
                plt.show()
    
    def export_signals(self, output_file):
        """
        Export generated signals to a file.
        
        Args:
            output_file (str): Path to the output file.
        """
        try:
            import h5py
            
            with h5py.File(output_file, 'w') as f:
                # Create groups
                waveform_group = f.create_group('ppg_waveforms')
                
                # Store time data
                waveform_group.create_dataset('time', data=self.time)
                
                # Store signals
                for wavelength, signal in self.ppg_signals.items():
                    waveform_group.create_dataset(f'signal_{wavelength}', data=signal)
                
                # Store detected intensities
                for wavelength, intensity in self.detected_intensity.items():
                    waveform_group.create_dataset(f'intensity_{wavelength}', data=intensity)
                
                # Store simulation parameters
                params = waveform_group.create_group('parameters')
                for key, value in self.config.items():
                    if isinstance(value, (int, float, str, bool)):
                        params.attrs[key] = value
            
            print(f"PPG signals exported to {output_file}")
        except ImportError:
            print("h5py not installed. Cannot export signals.")


def generate_waveforms(optical_results, cfd_results=None, config=None, output_dir=None):
    """
    Generate PPG waveforms using optical and CFD simulation results.
    
    Args:
        optical_results (dict): Results from optical simulation.
        cfd_results (dict, optional): Results from CFD simulation.
        config (dict, optional): Configuration parameters.
        output_dir (str, optional): Directory to save output files.
        
    Returns:
        dict: Dictionary containing the generated waveforms.
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and run the waveform generator
    generator = PPGSignalGenerator(optical_results, cfd_results, config)
    signals = generator.generate_all_signals()
    
    # Visualize the waveforms
    if output_dir:
        generator.visualize_signals(os.path.join(output_dir, 'figures'))
        
        # Export waveforms
        output_file = os.path.join(output_dir, 'ppg_waveforms.h5')
        generator.export_signals(output_file)
    
    return signals


if __name__ == "__main__":
    # Test with mock optical and CFD results
    optical_results = {
        'detection_rate': 0.15,
        'wavelength': 'red'
    }
    
    cfd_results = {
        'time': np.linspace(0, 0.8, 20),  # One cardiac cycle
        'velocity_profiles': [np.random.rand(10) for _ in range(20)]  # Random velocity profiles
    }
    
    config = {
        'sampling_rate': 100,  # Hz
        'duration': 5,         # seconds
        'heart_rate': 75,      # bpm
        'heart_rate_variability': 3,  # bpm variation
        'respiratory_rate': 0.2,  # Hz
        'noise_level': 0.02,   # Fraction of signal amplitude
        'motion_artifacts': True,
        'baseline_drift': True,
        'use_cfd_data': True,
        'oxygen_saturation': 0.98,
        'glucose_level': 120,  # mg/dL
        'lipid_levels': {
            'total_cholesterol': 220,  # mg/dL
            'hdl': 45,          # mg/dL
            'ldl': 150,         # mg/dL
            'triglycerides': 180 # mg/dL
        },
        'blood_volumes': {
            'green': 0.05,
            'red': 0.05,
            'nir': 0.05
        }
    }
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'results', 'waveforms')
    
    # Generate waveforms
    signals = generate_waveforms(optical_results, cfd_results, config, output_dir)
