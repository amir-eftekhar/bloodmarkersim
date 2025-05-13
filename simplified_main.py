#!/usr/bin/env python3
"""
Simplified main script for running the wrist blood flow simulation without fenics dependency.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import required modules
from src.utils.geometry import create_wrist_mesh, visualize_wrist_model
from src.utils.constants import (BLOOD_DENSITY, BLOOD_VISCOSITY, WOMERSLEY_NUMBER,
                               ABSORPTION_GREEN, ABSORPTION_RED, ABSORPTION_NIR,
                               SCATTERING_GREEN, SCATTERING_RED, SCATTERING_NIR)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Wrist Blood Flow Simulation")
    
    parser.add_argument("--output_dir", type=str, default="data/results",
                       help="Output directory for simulation results")
    
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Enable visualization of results")
    
    return parser.parse_args()

def setup_simple_cfd_model():
    """Create a simplified CFD model for blood flow in radial artery."""
    # Parameters
    radius = 0.15  # cm (1.5 mm)
    length = 5.0   # cm
    density = BLOOD_DENSITY
    viscosity = BLOOD_VISCOSITY
    
    # Create mesh
    n_r = 20
    n_x = 50
    r = np.linspace(0, radius, n_r)
    x = np.linspace(0, length, n_x)
    R, X = np.meshgrid(r, x, indexing='ij')
    
    # Calculate velocity profile (Poiseuille flow)
    pressure_gradient = 1000  # dyne/cm³
    velocity_profile = pressure_gradient / (4 * viscosity) * (radius**2 - R**2)
    
    # Calculate wall shear stress
    wall_shear_stress = -viscosity * (velocity_profile[-1, :] - velocity_profile[-2, :]) / (r[-1] - r[-2])
    
    # Create result dictionary
    results = {
        'r': r,
        'x': x,
        'velocity_profile': velocity_profile,
        'wall_shear_stress': wall_shear_stress,
        'pressure_gradient': pressure_gradient,
        'radius': radius,
        'length': length,
        'density': density,
        'viscosity': viscosity
    }
    
    return results

def visualize_cfd_results(results, output_dir):
    """Visualize the simplified CFD results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    r = results['r']
    x = results['x']
    velocity_profile = results['velocity_profile']
    wall_shear_stress = results['wall_shear_stress']
    
    # Plot velocity profile
    plt.figure(figsize=(10, 6))
    for idx, x_pos in enumerate([0, len(x)//4, len(x)//2, 3*len(x)//4, -1]):
        plt.plot(r, velocity_profile[:, x_pos], label=f'x = {x[x_pos]:.2f} cm')
    
    plt.xlabel('Radial position (cm)')
    plt.ylabel('Axial velocity (cm/s)')
    plt.title('Velocity Profiles in Radial Artery')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'velocity_profiles.png'), dpi=300)
    
    # Plot wall shear stress
    plt.figure(figsize=(10, 6))
    plt.plot(x, wall_shear_stress)
    plt.xlabel('Axial position (cm)')
    plt.ylabel('Wall Shear Stress (dyne/cm²)')
    plt.title('Wall Shear Stress along Radial Artery')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'wall_shear_stress.png'), dpi=300)
    
    plt.close('all')

def create_optical_model(wrist_model, wavelength='red'):
    """Create a simplified optical model for the wrist."""
    # Select appropriate optical properties based on wavelength
    if wavelength == 'green':
        absorption = ABSORPTION_GREEN
        scattering = SCATTERING_GREEN
    elif wavelength == 'red':
        absorption = ABSORPTION_RED
        scattering = SCATTERING_RED
    else:  # 'nir'
        absorption = ABSORPTION_NIR
        scattering = SCATTERING_NIR
    
    # Extract tissue types from wrist model
    tissue_types = wrist_model['tissue_types']
    
    # Create simplified absorption and scattering maps
    absorption_map = np.zeros_like(tissue_types, dtype=float)
    scattering_map = np.zeros_like(tissue_types, dtype=float)
    
    # Map tissue types to optical properties
    # 0=outside, 1=epidermis, 2=dermis, 3=fat, 4=muscle, 5=blood, 6=bone
    absorption_map[tissue_types == 0] = 0.0  # outside (air)
    absorption_map[tissue_types == 1] = absorption['epidermis']
    absorption_map[tissue_types == 2] = absorption['dermis']
    absorption_map[tissue_types == 3] = absorption['fat']
    absorption_map[tissue_types == 4] = absorption['muscle']
    absorption_map[tissue_types == 5] = 0.75 * absorption['oxyhemoglobin'] + 0.25 * absorption['deoxyhemoglobin']
    absorption_map[tissue_types == 6] = 0.1  # bone (approximate)
    
    scattering_map[tissue_types == 0] = 0.0  # outside (air)
    scattering_map[tissue_types == 1] = scattering['epidermis']
    scattering_map[tissue_types == 2] = scattering['dermis']
    scattering_map[tissue_types == 3] = scattering['fat']
    scattering_map[tissue_types == 4] = scattering['muscle']
    scattering_map[tissue_types == 5] = scattering['blood']
    scattering_map[tissue_types == 6] = 30.0  # bone (approximate)
    
    # Create optical model
    optical_model = {
        'absorption_map': absorption_map,
        'scattering_map': scattering_map,
        'wavelength': wavelength,
        'wrist_model': wrist_model
    }
    
    return optical_model

def visualize_optical_model(optical_model, output_dir):
    """Visualize the optical model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    absorption_map = optical_model['absorption_map']
    scattering_map = optical_model['scattering_map']
    wavelength = optical_model['wavelength']
    
    # Get cross-sections
    mid_x = absorption_map.shape[0] // 2
    mid_y = absorption_map.shape[1] // 2
    mid_z = absorption_map.shape[2] // 2
    
    # Plot absorption cross-section
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f'Absorption Coefficient ({wavelength})')
    plt.imshow(absorption_map[mid_x, :, :].T, origin='lower', cmap='hot')
    plt.colorbar(label='Absorption Coefficient (cm⁻¹)')
    plt.xlabel('Y Position')
    plt.ylabel('Z Position')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Scattering Coefficient ({wavelength})')
    plt.imshow(scattering_map[mid_x, :, :].T, origin='lower', cmap='Blues')
    plt.colorbar(label='Scattering Coefficient (cm⁻¹)')
    plt.xlabel('Y Position')
    plt.ylabel('Z Position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'optical_properties_{wavelength}.png'), dpi=300)
    
    # Plot another view
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f'Absorption Coefficient ({wavelength})')
    plt.imshow(absorption_map[:, mid_y, :].T, origin='lower', cmap='hot')
    plt.colorbar(label='Absorption Coefficient (cm⁻¹)')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    
    plt.subplot(1, 2, 2)
    plt.title(f'Scattering Coefficient ({wavelength})')
    plt.imshow(scattering_map[:, mid_y, :].T, origin='lower', cmap='Blues')
    plt.colorbar(label='Scattering Coefficient (cm⁻¹)')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'optical_properties_xz_{wavelength}.png'), dpi=300)
    
    plt.close('all')

def create_simplified_ppg(cfd_results, optical_model):
    """Create a simplified PPG signal based on CFD and optical models."""
    # Time parameters
    duration = 5.0  # seconds
    sampling_rate = 100  # Hz
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Create pulsatile component (heartbeat)
    heart_rate = 75  # bpm
    heart_period = 60 / heart_rate  # seconds
    
    # Normalize time to cardiac cycle
    t_norm = (time % heart_period) / heart_period
    
    # Create basic pulse wave
    ppg_base = np.zeros_like(time)
    for i, t in enumerate(t_norm):
        if t < 0.2:  # Systolic rise
            ppg_base[i] = 1.0 * np.sin(t * np.pi / 0.2)
        elif t < 0.4:  # Systolic fall
            ppg_base[i] = 1.0 - 0.7 * ((t - 0.2) / 0.2)
        elif t < 0.5:  # Dicrotic notch
            ppg_base[i] = 0.3 - 0.1 * np.sin((t - 0.4) * np.pi / 0.1)
        else:  # Diastolic phase
            ppg_base[i] = 0.2 * np.exp(-(t - 0.5) / 0.2)
    
    # Scale and normalize
    ppg_base = (ppg_base - np.min(ppg_base)) / (np.max(ppg_base) - np.min(ppg_base))
    
    # Apply wavelength-specific effects
    wavelength = optical_model['wavelength']
    
    if wavelength == 'green':
        # Green has high absorption by blood
        ppg_green = 0.8 * ppg_base + 0.1 * np.sin(2 * np.pi * 0.2 * time)  # Add respiratory component
        ppg = ppg_green
    elif wavelength == 'red':
        # Red has moderate absorption
        ppg_red = 0.7 * ppg_base + 0.2 * np.sin(2 * np.pi * 0.2 * time) + 0.1 * np.random.rand(len(time))
        ppg = ppg_red
    else:  # 'nir'
        # NIR has lower absorption by blood
        ppg_nir = 0.6 * ppg_base + 0.15 * np.sin(2 * np.pi * 0.2 * time) + 0.05 * np.random.rand(len(time))
        ppg = ppg_nir
    
    # Add noise
    noise = 0.05 * np.random.randn(len(time))
    ppg += noise
    
    # Create PPG result
    ppg_result = {
        'time': time,
        'signal': ppg,
        'wavelength': wavelength,
        'sampling_rate': sampling_rate,
        'heart_rate': heart_rate
    }
    
    return ppg_result

def visualize_ppg(ppg_result, output_dir):
    """Visualize the PPG signal."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    time = ppg_result['time']
    signal = ppg_result['signal']
    wavelength = ppg_result['wavelength']
    
    # Plot PPG signal
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.title(f'Simulated PPG Signal ({wavelength})')
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f'ppg_signal_{wavelength}.png'), dpi=300)
    
    # Plot frequency content
    plt.figure(figsize=(12, 6))
    
    # Calculate FFT
    N = len(signal)
    sampling_rate = ppg_result['sampling_rate']
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(N, d=1/sampling_rate)
    fft_magnitude = np.abs(fft) / N
    
    plt.plot(freq, fft_magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Content of PPG Signal ({wavelength})')
    plt.xlim(0, 10)  # Focus on frequencies up to 10 Hz
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, f'ppg_spectrum_{wavelength}.png'), dpi=300)
    
    plt.close('all')

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Starting Wrist Blood Flow Simulation ===")
    
    # Step 1: Create simplified CFD model
    print("\nCreating CFD model...")
    cfd_results = setup_simple_cfd_model()
    
    # Visualize CFD results
    if args.visualize:
        cfd_output_dir = os.path.join(args.output_dir, 'cfd')
        visualize_cfd_results(cfd_results, cfd_output_dir)
    
    # Step 2: Create wrist anatomical model
    print("\nCreating wrist anatomical model...")
    wrist_model = create_wrist_mesh()
    
    # Visualize wrist model
    if args.visualize:
        anatomy_output_dir = os.path.join(args.output_dir, 'anatomy')
        os.makedirs(anatomy_output_dir, exist_ok=True)
        visualize_wrist_model(wrist_model, os.path.join(anatomy_output_dir, 'wrist_model.png'))
    
    # Step 3: Create optical models for different wavelengths
    wavelengths = ['green', 'red', 'nir']
    ppg_signals = {}
    
    for wavelength in wavelengths:
        print(f"\nCreating optical model for {wavelength} wavelength...")
        optical_model = create_optical_model(wrist_model, wavelength)
        
        # Visualize optical model
        if args.visualize:
            optical_output_dir = os.path.join(args.output_dir, 'optical', wavelength)
            visualize_optical_model(optical_model, optical_output_dir)
        
        # Step 4: Create PPG signals
        print(f"Generating PPG signal for {wavelength} wavelength...")
        ppg_result = create_simplified_ppg(cfd_results, optical_model)
        ppg_signals[wavelength] = ppg_result
        
        # Visualize PPG signals
        if args.visualize:
            ppg_output_dir = os.path.join(args.output_dir, 'ppg')
            visualize_ppg(ppg_result, ppg_output_dir)
    
    # Step 5: Create combined visualization of all wavelengths
    if args.visualize:
        plt.figure(figsize=(12, 6))
        
        for wavelength, ppg_result in ppg_signals.items():
            if wavelength == 'green':
                color = 'green'
            elif wavelength == 'red':
                color = 'red'
            else:  # 'nir'
                color = 'black'
            
            plt.plot(ppg_result['time'], ppg_result['signal'], color=color, label=wavelength, alpha=0.7)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.title('PPG Signals at Different Wavelengths')
        plt.legend()
        plt.grid(True)
        
        ppg_output_dir = os.path.join(args.output_dir, 'ppg')
        os.makedirs(ppg_output_dir, exist_ok=True)
        plt.savefig(os.path.join(ppg_output_dir, 'ppg_combined.png'), dpi=300)
        
        plt.close('all')
    
    print("\n=== Simulation Complete ===")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
